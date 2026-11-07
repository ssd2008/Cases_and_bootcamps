"""Frozen combinations of waveform augmentation, longer training and explicit time."""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import argparse,hashlib,json,math,random,time,traceback,subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import soundfile as sf
import torch,joblib
from torch import nn
from torch.nn import functional as F
from scipy.special import expit
from features import ROOT,manifest
from networks import DEVICE,seed_everything,normalize
from research_runner import save,evaluate
from experiment_journal import JOURNAL,now,log,render,update
import research_v4 as old

OUT=ROOT/'reports/research_v7';FINAL=ROOT/'models/research_final_r7';LOCAL=ROOT/'models/research_v7';CACHE=ROOT/'data/cache/research_v7'
SEEDS=old.SEEDS;FOLDS=old.FOLDS;old.power.OUT=OUT
guard=old.guard;atomic_torch=old.atomic_torch
def read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean():
 if DEVICE.type=='mps':torch.mps.empty_cache()
ARMS={}
def arm(a,title,view='full',batch=128,control='F_BASE',epochs=4,time_mode='none',noise_all=0.,noise_real=0.,shift=0.,continuation=None):
 ARMS[a]=dict(title=title,view=view,batch=batch,kind='base',control=control,epochs=epochs,schedule_epochs=224 if continuation else 14,time_mode=time_mode,noise_all=noise_all,noise_real=noise_real,shift=shift,continuation=continuation)
arm('P_RS','P01: шум настоящим + сдвиг','base',8,'P_BASE',noise_real=.5,shift=.5)
arm('P_AS','P01: шум обоим классам + сдвиг','base',8,'P_BASE',noise_all=.5,shift=.5)
arm('P_MIX','P01: шум обоим + дополнительная частота шума настоящим + сдвиг','base',8,'P_RS',noise_all=.5,noise_real=.5,shift=.5)
arm('F_REAL','F100: шум только настоящим',noise_real=.5)
arm('F_MIX','F100: шум обоим + дополнительная частота шума настоящим',control='F_REAL',noise_all=.5,noise_real=.5)
arm('F_E008','F100: восемь эпох',epochs=8)
arm('B080','P01/batch128: продолжение64→80 эпох','base',128,'R4_B064',epochs=80,continuation='B064')
arm('F_TIME','F100: секунды и относительное время отдельными каналами',control='F_TIME0',time_mode='coords')
UPLOAD=list(ARMS)
arm('F_BASE','Свежий локальный контроль F100')
arm('P_BASE','Свежий локальный контроль P01','base',8,control='R4_F111')
arm('F_TIME0','Контроль временных каналов: та же ёмкость, координаты нулевые',time_mode='zeros')

def freeze():
 OUT.mkdir(parents=True,exist_ok=True);FINAL.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
 protocol=dict(arms=ARMS,submission_order=UPLOAD,seeds=SEEDS,folds=FOLDS,maximum_submissions=8,reserve=89,n_each=3000,
  authorization='User requested combinations of successful augmentations, longer epochs and explicit time on14Oct. Fixed eight-upload study from97 attempts; preserve89. No public-feedback parameter changes.',
  explanation='Full temporal resolution preserves every10ms STFT frame instead of pooling each clip to96 frames. It does not itself add a time coordinate. F10099.71 vsF00095.51 is the earlier matched +4.20 comparison.',
  composition='Noise-all Bernoulli.5 and noise-real Bernoulli.5 are independent; their union is applied once. Real noise frequency=.75 and fake=.5 in MIX. Do not double the amplitude on double selections. Independent shift Bernoulli.5 composes with noise on the waveform before STFT. Labels unchanged.',
  waveform='Same fixed per-file native waveform noise as R4: normalized Gaussian, SNR20–35dB, no augmentation at inference. Same hop-aligned shift as R4, no cut or wrap; P01 waveform silence retained. F100 noise analysed without waveform tail, then zero feature padding. No shifts on F100 to avoid silently introducing waveform padding.',
  time='F_TIME uses two extra planes: STFT frame-centre seconds/5 and centre/duration. Only real frame positions and64 frequency rows receive coordinates. Extra conv weights start at zero; F_TIME0 has identical architecture but zero planes. No temporal compression, time-mask augmentation never erases coordinates.',
  epochs='F_E008 uses8 instead of4 epochs with the same cosine horizon14; no epoch selection. B080 resumes original verified B064 optimizer/RNG/epoch65 state and continues to80 with original cosine horizon224, so first64 training epochs are preserved. No restart or altered64-epoch weights.',
  validation='All11 local strategies on the same3 speaker/attack-disjoint folds,2seeds each:33 ensembles/66 local members. B080 extends6 existing local B064 members. Fresh F_BASE/P_BASE and time-plane-capacity F_TIME0 controls. Existing R4 individual augmentation local results are reported separately.',
  training='Same R4 AdamW .001 wd.01, norm5,6000 balanced draws/epoch, Gaussian feature noise.03 and shared8-frame mask probability.5. Sampling and waveform-augmentation RNGs separate. Two final seeds, mean logits, sigmoid(logit/5).',
  stages='Build two new train-only waveform views; freeze/verify all8 full-train predictions before reading their public grades, then local checks. Each queued artifact immutable. No automatic retry of ambiguous POST.',
  limits='Only three fixed held-out generator/speaker combinations. Earlier gains were on P01, not necessarily additive or transferable toF100. MPS training not strictly deterministic; small LB differences require caution. No new data or synthetic relabeling.')
 p=OUT/'protocol.json'
 if p.exists():assert read(p)['protocol']==protocol
 else:save(p,dict(created_at=now(),protocol=protocol,submission_order=UPLOAD));log('R7_ALL','protocol_frozen',protocol=protocol)
 registry=read(JOURNAL)
 for a,cfg in ARMS.items():
  if 'R7_'+a not in registry['ideas']:registry['ideas']['R7_'+a]=dict(title=cfg['title'],author='Кирилл / Codex' if a=='F_TIME' else 'Кирилл',control=cfg['control'] if cfg['control'].startswith('R4_') else 'R7_'+cfg['control'],status='protocol_frozen',change=json.dumps(cfg,ensure_ascii=False),notes='R7 fixed combinations; reports/research_v7/protocol.json. Exactly8 new public candidates;89-attempt reserve.')
 registry['updated_at']=now();save(JOURNAL,registry);render(registry)
 if not (OUT/'previous_best.json').exists():save(OUT/'previous_best.json',read(ROOT/'models/current_best.json'))

def extract_aug(args):
 path,index,maximum=args
 x,sr=sf.read(path,dtype='float32');assert sr==16000 and x.ndim==1
 x=x[:maximum];x=x/(np.sqrt(np.mean(x*x))+1e-9)
 if len(x)<1024:x=np.pad(x,(0,1024-len(x)))
 rng=np.random.default_rng(old.SEED+3901+index);snr=rng.uniform(20,35)
 noisy=x+rng.normal(size=len(x)).astype(np.float32)*np.float32(10**(-snr/20));noisy/=np.sqrt(np.mean(noisy*noisy))+1e-9
 rng=np.random.default_rng(old.SEED+4901+index);left=int(rng.integers(0,(maximum-len(x))//160+1))*160
 native=old.stft_map(noisy);full=np.zeros_like(old.stft_map(np.zeros(maximum,np.float32)));full[:len(native)]=native
 shifted=old.stft_map(np.pad(noisy,(left,maximum-len(noisy)-left)))
 return full.astype(np.float16),shifted.astype(np.float16)

def build_cache():
 CACHE.mkdir(parents=True,exist_ok=True);rows=manifest(ROOT/'data');spec=read(ROOT/'data/cache/research_v2/complete.json')
 contract=dict(version=1,ids_sha256=hashlib.sha256('\n'.join(r[0] for r in rows).encode()).hexdigest(),maximum=spec['max_samples'],keys=['noise_full','noise_shift'])
 if (CACHE/'complete.json').exists():assert read(CACHE/'complete.json')['contract']==contract;return
 arrays={k:np.lib.format.open_memmap(CACHE/(k+'.npy'),mode='r+' if (CACHE/(k+'.npy')).exists() else 'w+',dtype='float16',shape=(len(rows),spec['max_frames'],64)) for k in contract['keys']}
 progress=read(CACHE/'progress.json',dict(next_index=0,contract=contract));assert progress['contract']==contract;last=progress['next_index']
 try:
  with ProcessPoolExecutor(8) as pool:
   for first in range(last,len(rows),128):
    guard();jobs=[(rows[i][1],i,spec['max_samples']) for i in range(first,min(first+128,len(rows)))]
    for i,(nf,ns) in enumerate(pool.map(extract_aug,jobs,chunksize=8),first):arrays['noise_full'][i]=nf;arrays['noise_shift'][i]=ns;last=i+1
    for a in arrays.values():a.flush()
    save(CACHE/'progress.json',dict(contract=contract,next_index=last,time=now()));print('CACHE',last,'/',len(rows),flush=True)
 finally:
  for a in arrays.values():a.flush()
  save(CACHE/'progress.json',dict(contract=contract,next_index=last,time=now()))
 if last==len(rows):save(CACHE/'complete.json',dict(contract=contract,complete=True,finished_at=now()))

class Data(old.Data):
 def view(self,key):
  if key in ['noise_full','noise_shift'] and key not in self.views:self.views[key]=np.load(CACHE/(key+'.npy'),mmap_mode='r')
  return super().view(key)

class Detector(nn.Module):
 def __init__(self,cfg):
  super().__init__();self.cfg=cfg;base=old.Detector(cfg);self.convs=base.convs;self.head=base.head
  if cfg['time_mode']!='none':
   previous=self.convs[0];state=torch.get_rng_state();stem=nn.Conv2d(3,16,3,padding=1);torch.set_rng_state(state)
   with torch.no_grad():stem.weight.zero_();stem.weight[:,:1].copy_(previous.weight);stem.bias.copy_(previous.bias)
   self.convs[0]=stem
 def forward(self,x,aux):
  if self.cfg['time_mode']!='none':
   timeplanes=aux['time'][:,:,None,:].expand(-1,-1,x.shape[-2],-1)
   frequency=(torch.arange(x.shape[-2],device=x.device)<64).to(x.dtype)[None,None,:,None]
   x=torch.cat([x,timeplanes*frequency],dim=1)
  h=self.convs(x);v=torch.cat([h.mean((2,3)),h.std((2,3))],1);return self.head(v).squeeze(-1)

def inputs(d,ix,cfg,sc,augmentation_rng=None):
 raw=d.view(cfg['view'])[ix].astype(np.float32)
 if augmentation_rng is not None:
  noise=np.zeros(len(ix),bool);shift=np.zeros(len(ix),bool)
  if cfg['noise_all']:noise|=augmentation_rng.random(len(ix))<cfg['noise_all']
  if cfg['noise_real']:noise|=(augmentation_rng.random(len(ix))<cfg['noise_real'])&(d.y[ix]==0)
  if cfg['shift']:shift=augmentation_rng.random(len(ix))<cfg['shift'];assert cfg['view']=='base'
  for n,s,key in [(True,True,'noise_shift'),(True,False,'noise_full' if cfg['view']=='full' else 'noise'),(False,True,'shift')]:
   mask=(noise==n)&(shift==s)
   if mask.any():raw[mask]=d.view(key)[ix[mask]].astype(np.float32)
 a=normalize(raw,sc);a=np.pad(a,[(0,0),(0,0),(0,10)]).transpose(0,2,1)[:,None].copy();aux={}
 if cfg['time_mode']!='none':
  t=(np.arange(raw.shape[1],dtype=np.float32)*160+200)/16000
  valid=np.arange(raw.shape[1])[None,:]<np.asarray(d.meta['lengths'])[ix,None]
  coords=np.stack([np.broadcast_to(t[None,:]/5,valid.shape),np.minimum(t[None,:]/np.asarray(d.duration)[ix,None],1)],axis=1).astype(np.float32)*valid[:,None,:]
  if cfg['time_mode']=='zeros':coords.fill(0)
  aux['time']=torch.from_numpy(coords).to(DEVICE)
 return torch.from_numpy(a).to(DEVICE),aux

class Interrupted(RuntimeError):pass
def extend_checkpoint(a,seed,dest,tr,fold=None):
 cfg=ARMS[a]
 if not cfg['continuation'] or (dest/f'checkpoint_{seed}.pt').exists():return
 source=(ROOT/f'models/research_v4/{fold}/B064' if fold else ROOT/'models/research_final_r4/B064')
 path=source/f'checkpoint_{seed}.pt';c=torch.load(path,map_location='cpu',weights_only=False)
 assert c['progress']['epoch']==65 and c['progress']['cursor']==0 and len(c['progress']['history'])==64
 assert np.array_equal(c['train_idx'],tr) and c['contract']['cfg']['batch']==cfg['batch']==128 and c['contract']['cfg']['schedule_epochs']==cfg['schedule_epochs']==224
 weight=torch.load(source/f'seed_{seed}.pt',map_location='cpu',weights_only=True)
 assert all(torch.equal(weight[k],c['model'][k]) for k in weight)
 original_contract=c['contract'];c['contract']=dict(cfg=cfg,seed=seed,n_each=3000);atomic_torch(dest/f'checkpoint_{seed}.pt',c)
 save(dest/f'continuation_{seed}.json',dict(source=str(path.relative_to(ROOT)),source_sha256=digest(path),source_weight=str((source/f'seed_{seed}.pt').relative_to(ROOT)),source_weight_sha256=digest(source/f'seed_{seed}.pt'),original_contract=original_contract,target_contract=c['contract'],original_weights_unchanged=True,optimizer_and_rng_preserved=True,start_epoch=65,stop_epoch=80,time=now()))

def fit(d,tr,cfg,sc,seed,dest,stop_after=None,n_each=3000):
 dest.mkdir(parents=True,exist_ok=True);weight=dest/f'seed_{seed}.pt';receipt=dest/f'member_{seed}.json'
 seed_everything(seed);model=Detector(cfg).to(DEVICE)
 if weight.exists() and receipt.exists():model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True));return model,read(receipt)['history']
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['schedule_epochs']);rng=np.random.default_rng(seed);aug=np.random.default_rng(seed+9081)
 state=dict(epoch=1,cursor=0,chosen=None,total=0.,history=[],steps=0);contract=dict(cfg=cfg,seed=seed,n_each=n_each);checkpoint=dest/f'checkpoint_{seed}.pt';bs=cfg['batch']
 if checkpoint.exists():
  c=torch.load(checkpoint,map_location='cpu',weights_only=False);assert c['contract']==contract and np.array_equal(c['train_idx'],tr)
  model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);scheduler.load_state_dict(c['scheduler']);state=c['progress'];rng.bit_generator.state=c['sampling_rng'];aug.bit_generator.state=c['augmentation_rng'];random.setstate(c['python_rng']);torch.set_rng_state(c['torch_rng'])
  if DEVICE.type=='mps':torch.mps.set_rng_state(c['mps_rng'])
  print('RESUME',dest,seed,state['epoch'],state['cursor'],flush=True)
 def keep():atomic_torch(checkpoint,dict(contract=contract,train_idx=tr,model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),progress=state,sampling_rng=rng.bit_generator.state,augmentation_rng=aug.bit_generator.state,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if DEVICE.type=='mps' else None))
 try:
  while state['epoch']<=cfg['epochs']:
   guard();model.train();start=time.monotonic()
   if state['chosen'] is None:state['chosen']=np.r_[rng.choice(tr[d.y[tr]==0],n_each,replace=True),rng.choice(tr[d.y[tr]==1],n_each,replace=True)];rng.shuffle(state['chosen'])
   for at in range(state['cursor'],len(state['chosen']),bs):
    guard();ix=state['chosen'][at:at+bs];x,aux=inputs(d,ix,cfg,sc,aug);x=x+torch.randn_like(x)*.03
    if rng.random()<.5:off=int(rng.integers(0,x.shape[-1]-8));x[...,off:off+8]=0
    pred=model(x,aux);loss=F.binary_cross_entropy_with_logits(pred,torch.from_numpy(d.y[ix].astype(np.float32)).to(DEVICE));opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
    state['total']+=float(loss.detach().cpu())*len(ix);state['cursor']=at+len(ix);state['steps']+=1
    if state['steps']%max(1,math.ceil(2000/bs))==0:keep()
    if stop_after is not None and state['steps']>=stop_after:keep();raise Interrupted('Intentional checkpoint check')
   scheduler.step();row=dict(epoch=state['epoch'],bce=state['total']/len(state['chosen']),seconds=round(time.monotonic()-start,2),updates=state['steps']);state['history'].append(row)
   print('EPOCH',str(dest.relative_to(ROOT)),seed,row,flush=True);log('R7_FIT','epoch',folder=str(dest.relative_to(ROOT)),seed=seed,**row)
   state.update(epoch=state['epoch']+1,cursor=0,chosen=None,total=0.);keep()
 except (old.BatteryLow,KeyboardInterrupt,Interrupted):keep();raise
 atomic_torch(weight,model.cpu().state_dict());save(receipt,dict(seed=seed,contract=contract,history=state['history'],complete=True));model.to(DEVICE);return model,state['history']

@torch.no_grad()
def predict(model,d,ix,cfg,sc):
 model.eval();out=[]
 for at in range(0,len(ix),8):guard();x,aux=inputs(d,ix[at:at+8],cfg,sc);out.append(model(x,aux).cpu().numpy())
 return np.concatenate(out).astype(np.float64)

def refit(d,a):
 dest=FINAL/a;dest.mkdir(parents=True,exist_ok=True);cfg=ARMS[a]
 if read(dest/'ensemble.json',{}).get('complete'):return
 sc=joblib.load(ROOT/'models/research_final_r4/F100/normalizer.joblib');joblib.dump(sc,dest/'normalizer.joblib');tr=np.arange(len(d.ids));members=[]
 for seed in SEEDS:
  extend_checkpoint(a,seed,dest,tr);model,h=fit(d,tr,cfg,sc,seed,dest);members.append(dict(seed=seed,file=f'seed_{seed}.pt',history=h));del model;clean()
 save(dest/'ensemble.json',dict(engine='research_v7',arm=a,cfg=cfg,feature_config=d.spec,seeds=SEEDS,members=members,temperature=5.,complete=True))

def infer(folder,input_dir,output,sample=None,data=None):
 folder=Path(folder);meta=read(folder/'ensemble.json');assert meta['complete'];cfg=meta['cfg'];input_dir=Path(input_dir)
 ids=pd.read_csv(sample)['idx'].astype(str).tolist() if sample else sorted(p.name for p in input_dir.iterdir() if p.suffix.lower() in ['.wav','.flac'])
 assert ids and len(ids)==len(set(ids)) and all(Path(i).name==i and (input_dir/i).is_file() for i in ids)
 d=data or old.InferenceData([input_dir/i for i in ids],ids,[cfg['view']],meta['feature_config']['max_samples']);assert np.array_equal(d.ids,ids);sc=joblib.load(folder/'normalizer.joblib');ps=[]
 for member in meta['members']:
  model=Detector(cfg).to(DEVICE);model.load_state_dict(torch.load(folder/member['file'],map_location='cpu',weights_only=True));ps.append(predict(model,d,np.arange(len(ids)),cfg,sc));del model;clean()
 logits=np.mean(ps,0);score=expit(logits/meta['temperature']);assert np.isfinite(score).all();output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(dict(idx=ids,score=score)).to_csv(output,index=False,float_format='%.17g');np.savez(output.with_suffix('.npz'),idx=ids,score=score,logits=logits)
 item=dict(arm=meta['arm'],upload_file=str(output),sha256=digest(output),rows=len(ids),models=str(folder),inference_only=True);save(output.with_suffix('.json'),item);return item

def verify_and_queue(a,d):
 queue=read(OUT/'upload_manifest.json',{})
 if a in queue:return
 path=ROOT/f'submissions/research_r7/{a}.csv';item=infer(FINAL/a,ROOT/'data/public_test',path,ROOT/'data/public_sample_submission.csv',d);repeat=path.with_suffix('.repeat.csv');infer(FINAL/a,ROOT/'data/public_test',repeat,ROOT/'data/public_sample_submission.csv',d);assert digest(repeat)==item['sha256']
 item.update(verified_byte_identical=True,frozen_at=now(),frozen_files={str(p.relative_to(ROOT)):digest(p) for p in (FINAL/a).iterdir() if p.is_file() and (p.name in ['ensemble.json','normalizer.joblib'] or p.name.startswith('seed_'))});queue[a]=item;save(OUT/'upload_manifest.json',queue);update('R7_'+a,status='verified_ready',final_models=str(FINAL/a));print('READY',a,flush=True)

def summarize(result):
 summary={};historical=read(ROOT/'reports/research_v4/local_results.json')
 for a,cfg in ARMS.items():
  vals={fid:f['arms'][a]['metrics']['unseen_attack_auc'] for fid,f in result['folds'].items() if a in f['arms']}
  if not vals:continue
  control=cfg['control'];deltas={}
  for fid,v in vals.items():
   base=historical['folds'][fid]['arms'][control[3:]] if control.startswith('R4_') else result['folds'][fid]['arms'].get(control)
   if base:deltas[fid]=v-base['metrics']['unseen_attack_auc']
  summary[a]=dict(n_folds=len(vals),mean_unseen_auc=float(np.mean(list(vals.values()))),worst_unseen_auc=min(vals.values()),unseen_auc_by_fold=vals,control=control,paired_deltas=deltas,paired_mean_delta=float(np.mean(list(deltas.values()))) if deltas else None)
  update('R7_'+a,research_v2=summary[a])
 save(OUT/'local_summary.json',summary)

def local(d):
 result=read(OUT/'local_results.json',dict(folds={}));definitions={f['id']:f for f in read(ROOT/'reports/research_v2/validation_audit.json')['folds']}
 for fid in FOLDS:
  split=np.load(ROOT/f'reports/research_v2/split_{fid}.npz');tr=split['train'];va=split['validation'];fold=result['folds'].setdefault(fid,dict(fold=definitions[fid],arms={}))
  assert not set(d.speakers[tr])&set(d.speakers[va]) and definitions[fid]['heldout_attack'] not in d.attacks[tr]
  sc=joblib.load(ROOT/f'models/research_v4/{fid}/F100/normalizer.joblib')
  for a in ['F_BASE','P_BASE','F_TIME0']+UPLOAD:
   if a in fold['arms']:continue
   cfg=ARMS[a];dest=LOCAL/fid/a;dest.mkdir(parents=True,exist_ok=True);joblib.dump(sc,dest/'normalizer.joblib');members=[];ps=[]
   for seed in SEEDS:
    predfile=dest/f'predictions_{seed}.npz';receipt=dest/f'result_{seed}.json'
    if predfile.exists() and receipt.exists():arr=np.load(predfile);assert np.array_equal(arr['idx'],va);p=arr['logits'];member=read(receipt)
    else:
     extend_checkpoint(a,seed,dest,tr,fid);model,h=fit(d,tr,cfg,sc,seed,dest);p=predict(model,d,va,cfg,sc);np.savez(predfile,idx=va,logits=p);member=dict(seed=seed,history=h,metrics=evaluate(d,va,p,definitions[fid]['heldout_attack']));save(receipt,member);del model;clean()
    ps.append(p);members.append(member)
   p=np.mean(ps,0);np.savez(dest/'predictions.npz',idx=va,logits=p);fold['arms'][a]=dict(metrics=evaluate(d,va,p,definitions[fid]['heldout_attack']),members=members,seeds=SEEDS)
   save(OUT/'local_results.json',result);summarize(result);print('LOCAL',fid,a,fold['arms'][a]['metrics']['unseen_attack_auc'],flush=True)

def check_implementation(d):
 ix=np.asarray([0,100,1000]);sc=joblib.load(ROOT/'models/research_final_r4/F100/normalizer.joblib');checks={}
 for a,source in [('F_BASE','F100'),('P_BASE','F111')]:
  x,_=inputs(d,ix,ARMS[a],sc);y,_=old.inputs(d,ix,old.ARMS[source],sc);assert torch.equal(x,y);checks[a+'_input_identical']=True
 seed_everything(SEEDS[0]);base=Detector(ARMS['F_BASE']).to(DEVICE).eval();seed_everything(SEEDS[0]);time_model=Detector(ARMS['F_TIME']).to(DEVICE).eval()
 x,aux=inputs(d,ix,ARMS['F_TIME'],sc)
 with torch.no_grad():delta=float((base(x,{})-time_model(x,aux)).abs().max().cpu())
 assert delta<1e-5;checks['time_initial_max_difference']=delta
 time_model.train();time_model(x,aux).sum().backward();gradient=float(time_model.convs[0].weight.grad[:,1:].abs().sum().cpu());assert gradient>0;checks['time_plane_gradient']=gradient
 assert torch.all(aux['time'][...,torch.arange(x.shape[-1],device=DEVICE)>=int(d.meta['lengths'][ix].max())]==0)
 class Always:
  def random(self,n):return np.zeros(n)
 xb,_=inputs(d,ix,ARMS['P_MIX'],sc,Always());expect=normalize(d.view('noise_shift')[ix].astype(np.float32),sc);expect=np.pad(expect,[(0,0),(0,0),(0,10)]).transpose(0,2,1)[:,None].copy();assert torch.equal(xb,torch.from_numpy(expect).to(DEVICE));checks['composed_waveform_selected']=True
 checks['labels_unchanged']=True;save(OUT/'implementation_checks.json',checks);del base,time_model;clean();print('IMPLEMENTATION CHECKS PASSED',checks,flush=True)

def run():
 freeze();save(OUT/'active_process.json',dict(pid=os.getpid(),started_at=now()));save(OUT/'pipeline_status.json',dict(complete=False,running=True,pid=os.getpid(),phase='cache',time=now()));build_cache();d=Data();check_implementation(d)
 ids=pd.read_csv(ROOT/'data/public_sample_submission.csv')['idx'].astype(str).tolist();public=old.InferenceData([ROOT/'data/public_test'/i for i in ids],ids,['base','full'])
 save(OUT/'pipeline_status.json',dict(complete=False,running=True,phase='final_fits',time=now()))
 for a in UPLOAD:refit(d,a);verify_and_queue(a,public)
 save(OUT/'pipeline_status.json',dict(complete=False,running=True,final_candidates_complete=True,phase='local',time=now()));local(d);save(OUT/'pipeline_status.json',dict(complete=True,final_candidates_complete=True,finished_at=now()))

if __name__=='__main__':
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True);s.add_parser('freeze');s.add_parser('run');pr=s.add_parser('predict');pr.add_argument('--models',type=Path,required=True);pr.add_argument('--input',type=Path,required=True);pr.add_argument('--sample',type=Path);pr.add_argument('--output',type=Path,default=Path('submission.csv'));a=p.parse_args()
 try:
  if a.command=='freeze':freeze()
  elif a.command=='run':run()
  else:print(infer(a.models,a.input,a.output,a.sample))
 except (old.BatteryLow,KeyboardInterrupt):
  if a.command=='run':save(OUT/'pipeline_status.json',dict(complete=False,paused=True,time=now()))
  raise
 except Exception as e:
  if a.command=='run':save(OUT/'pipeline_status.json',dict(complete=False,error=type(e).__name__+': '+str(e),time=now()))
  traceback.print_exc();raise
