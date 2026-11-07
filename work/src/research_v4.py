"""Native-audio audit and predeclared factorial, localization and augmentation trials.

Training uses supplied train only. No labels are changed by augmentation. Saved
weights, statistics and immutable predictions have explicit provenance.
"""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import argparse,hashlib,json,math,random,shutil,time,zipfile
import subprocess,tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from scipy.special import expit
import joblib,torch
from torch import nn
from torch.nn import functional as F
from features import ROOT,time_pool,manifest
from networks import DEVICE,seed_everything,fit_scaler,normalize
from research_runner import Data as OldData,ResearchCNN,CONFIGS,SEED,save,evaluate
from experiment_journal import JOURNAL,now,log,render,update
import research_v3 as power

OUT=ROOT/'reports/research_v4';LOCAL=ROOT/'models/research_v4';FINAL=ROOT/'models/research_final_r4';CACHE=ROOT/'data/cache/research_v4'
power.OUT=OUT
guard=power.guard;atomic_torch=power.atomic_torch;BatteryLow=power.BatteryLow
SEEDS=[SEED,SEED+1]
FOLDS=['G3_A03','G5_A05','G1_A06']
ARMS={}
for detail,pad,small in [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]:
 a=f'F{detail}{pad}{small}'
 view=('base' if detail else 'padpool') if pad else ('full' if detail else 'pooled')
 ARMS[a]=dict(title=f'Факторы: полная карта={detail}, звуковой паддинг={pad}, малый батч={small}',view=view,batch=8 if small else 128,kind='base',control='F000' if a!='F000' else 'V2_P00',author='Кирилл')
def arm(a,title,control='F111',author='Кирилл',**kw):ARMS[a]=dict(title=title,view='base',batch=8,kind='base',control=control,author=author,**kw)
arm('B064','P01: настоящий batch=128, 64 эпохи',control='F110',batch_override=128,epochs=64,schedule_epochs=224)
arm('N001','Центрирование по исходным кадрам без длинного хвоста',view_override='realcenter')
arm('S010','Полная карта: окно STFT 10 мс',view_override='s10')
arm('S050','Полная карта: окно STFT 50 мс',view_override='s50')
arm('A001','Шум исходной волны для обоих классов',augmentation='noise')
arm('A002','Шум исходной волны только для настоящих записей',augmentation='noise_real')
arm('A003','Сдвиг целой записи внутри паддинга, оба класса',augmentation='shift')
arm('L000','Посекундная голова: средний ответ',kind_override='local_mean')
arm('L001','Посекундная голова: soft maximum',control='L000',kind_override='local_lse')
arm('C000','Проверяющая CNN: три равномерных секундных участка',kind_override='crop_uniform')
arm('C001','Проверяющая CNN: три подозрительных секундных участка',control='C000',kind_override='crop_selected')
arm('Q128','128 частотных полос вместо 64',author='Codex',view_override='f128')
arm('H000','Контроль широкой головы: повтор глобальных статистик',author='Codex',kind_override='frequency_control')
arm('H001','Сохранить положение частот в финальной голове',control='H000',author='Codex',kind_override='frequency_head')
for c in ARMS.values():
 for k in ['view','kind','batch']:
  if k+'_override' in c:c[k]=c.pop(k+'_override')
 c.setdefault('epochs',4);c.setdefault('schedule_epochs',14)
for original in ['F110','F111']:
 ARMS[original+'S1']={**ARMS[original],'title':ARMS[original]['title']+'; один seed для сравнения B064','derived_from':original,'final_seeds':[SEED]}
ARMS['B064']=ARMS.pop('B064');ARMS['B064'].update(final_seeds=SEEDS,control='F110')

PROTOCOL=dict(arms=ARMS,local_seeds=SEEDS,final_seeds=SEEDS,folds=FOLDS,n_each=3000,
 optimizer='AdamW .001 wd=.01; gradient norm 5',map_augmentation='Gaussian .03 and shared eight-frame mask with probability .5 per batch',
 factor_order='F bits: full time grid, waveform padding, batch8. F000 reproduces P00; F111 reproduces P01. No-waveform-padding full maps use zero feature padding for batching, exactly as original full.npy. Eight-cell factorial, not only three unpaired runs.',
 large_batch='True batch128. 64 epochs give 3008 updates vs 3000 in batch8/4 epochs. Cosine epoch horizon224 retains relative LR progress. Exposure is 16 times larger. No gradient accumulation substitution.',
 centering='N001 changes only per-file centering: mean across the unpadded-STFT frame range and all real frequency bands. Full waveform-padded tail retained, historical per-band scaler retained. Boundary-overlap frames beyond the original STFT range excluded from mean.',
 spectral='S010: window160 hop160 FFT512 bands64; S050: window800 hop160 FFT1024 bands64; Q128: window400 hop160 FFT512 bands128. Per-band statistics from unpadded 96-frame maps of corresponding representation, training records only.',
 native='All provided train/public files audited: mono 16kHz. Native reads assert this; no resampling or stereo-to-mono conversion. No duplicate no-op submissions.',
 augmentation='One deterministic alternate waveform per training file, original/alternate chosen independently with p=.5. Noise SNR uniform20–35dB on original-duration signal, labels retained. Shift is hop-aligned left padding with compensating right padding, no wrap/cut. No augmentation during inference. Sampling RNG separate from augmentation RNG.',
 localization='Shared CNN produces temporal logits; average into fixed second bins, excluding artificial tail by duration. L000 mean bins; L001 logmeanexp(2*z)/2. Utterance labels only; scores are weakly supervised, not validated per-second probabilities.',
 verification='C000/C001 identical CNNs applied to three 100-STFT-frame crops, mean crop logits. Selected starts from frozen L001 of same train fold/seed; in-sample selection on training records, independent held-out selection. No OOF inner selector fit; limitation explicit. Last crop clamped to original STFT span; short recordings padded. Crops are on existing STFT grid, not waveform re-STFT.',
 frequency_head='H001 retains nine frequency positions, time mean/std ->1728->64->1. H000 same head/parameter count with nine copies of global mean/std; contrasts frequency position at matched head capacity.',
 validation='Three previously fixed hard speaker/attack-disjoint folds, two local seeds and two final seeds, including B064. Easier A01/A02/A04 saturated in R3 and omitted. No validation checkpoint selection. Per-seed metrics and mean-logit ensemble metrics retained; paired seed deltas reported.',
 stages='Full train fits first; two seeds for all trained candidates, B064 last. F110S1/F111S1 are extra derived single-seed controls without new training. Each immutable standalone release verified and queued for upload immediately. Three fixed local folds with two seeds follow. Grades read after all final models are ready or compute stops. Independent predeclared local fits may then still run; they never receive scores or change parameters.',
 feedback='Each candidate weights, statistics and predictions frozen before its upload. Submission overlaps independent predefined training. No refit/tuning using feedback. Mains power amendment restores two B064 final seeds and expands local validation to two seeds on three folds before B064 training, any R4 uploads or scores; other candidate settings and completed weights unchanged.',
 battery='caffeinate -i, screen may sleep; checkpoints every ~2000 examples and each epoch. Stop/save at<=1% discharging. Resume optimizer/RNG/sample cursor. Physical power loss cannot be prevented.')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def freeze():
 OUT.mkdir(parents=True,exist_ok=True);p=OUT/'protocol.json'
 if p.exists():assert json.loads(p.read_text())['protocol']==PROTOCOL
 else:save(p,dict(created_at=now(),protocol=PROTOCOL));log('R4_ALL','protocol_frozen',protocol=PROTOCOL)
 r=json.loads(JOURNAL.read_text())
 for a,c in ARMS.items():
  key='R4_'+a
  if key not in r['ideas']:r['ideas'][key]=dict(title=c['title'],author=c['author'],control=c['control'] if c['control'].startswith('V2_') else 'R4_'+c['control'],status='protocol_frozen',change=json.dumps(c,ensure_ascii=False),notes='R4: fixed factorial and native audio audit; three hard folds, two local seeds, two final seeds. See reports/research_v4/protocol.json.')
 r['updated_at']=now();JOURNAL.write_text(json.dumps(r,ensure_ascii=False,indent=2));render(r)

def stft_map(x,window=400,bands=64):
 fft=512 if window<=512 else 1024
 _,_,z=signal.stft(x,fs=16000,window='hann',nperseg=window,noverlap=window-160,nfft=fft,boundary=None,padded=True)
 p=np.abs(z)**2+1e-10;p=p[1:fft//2+1].reshape(bands,fft//(2*bands),-1).mean(1)
 lp=np.log(p+1e-10);lp-=lp.mean();return lp.T

def extract_record(args):
 path,index,max_samples,keys=args
 x,sr=sf.read(path,dtype='float32');assert sr==16000 and x.ndim==1,'Native mono 16kHz required; no lossy conversion'
 x=x[:max_samples];original=len(x);assert original>0
 x=x/(np.sqrt(np.mean(x*x))+1e-9)
 if len(x)<1024:x=np.pad(x,(0,1024-len(x)))
 unpad=stft_map(x);padded=np.pad(x,(0,max_samples-len(x)));base=stft_map(padded)
 out={'duration':original/16000,'length':len(unpad)}
 if 'base' in keys:out['base']=base.astype(np.float16)
 if 'full' in keys:
  a=np.zeros_like(base);a[:len(unpad)]=unpad;out['full']=a.astype(np.float16)
 if 'pooled' in keys:out['pooled']=time_pool(unpad.T).T.astype(np.float16)
 if 'padpool' in keys:out['padpool']=time_pool(base.T).T.astype(np.float16)
 if 'realcenter' in keys:out['realcenter']=(base-base[:len(unpad)].mean()).astype(np.float16)
 for key,window,bands in [('s10',160,64),('s50',800,64),('f128',400,128)]:
  if key in keys or key+'_pool' in keys:
   if key in keys:out[key]=stft_map(padded,window,bands).astype(np.float16)
   if key+'_pool' in keys:out[key+'_pool']=time_pool(stft_map(x,window,bands).T).T.astype(np.float16)
 if 'noise' in keys:
  rng=np.random.default_rng(SEED+3901+index);snr=rng.uniform(20,35)
  a=x+rng.normal(size=len(x)).astype(np.float32)*np.float32(10**(-snr/20));a/=np.sqrt(np.mean(a*a))+1e-9
  out['noise']=stft_map(np.pad(a,(0,max_samples-len(a)))).astype(np.float16)
 if 'shift' in keys:
  rng=np.random.default_rng(SEED+4901+index);left=int(rng.integers(0,(max_samples-len(x))//160+1))*160
  out['shift']=stft_map(np.pad(x,(left,max_samples-len(x)-left))).astype(np.float16)
 return out

NEW_KEYS=['padpool','realcenter','s10','s10_pool','s50','s50_pool','f128','f128_pool','noise','shift']
def build_cache():
 CACHE.mkdir(parents=True,exist_ok=True);rows=manifest(ROOT/'data');spec=json.loads((ROOT/'data/cache/research_v2/complete.json').read_text());maximum=spec['max_samples']
 contract=dict(ids_sha256=hashlib.sha256('\n'.join(r[0] for r in rows).encode()).hexdigest(),keys=NEW_KEYS,max_samples=maximum,version=1)
 done=CACHE/'complete.json';progress=CACHE/'progress.json'
 if done.exists():assert json.loads(done.read_text())['contract']==contract;return
 shapes={k:(96,128 if k.startswith('f128') else 64) if k.endswith('_pool') or k=='padpool' else stft_map(np.zeros(maximum,np.float32),160 if k=='s10' else 800 if k=='s50' else 400,128 if k=='f128' else 64).shape for k in NEW_KEYS}
 arrays={k:np.lib.format.open_memmap(CACHE/(k+'.npy'),mode='r+' if (CACHE/(k+'.npy')).exists() else 'w+',dtype='float16',shape=(len(rows),*shapes[k])) for k in NEW_KEYS}
 start=0
 if progress.exists():
  prior=json.loads(progress.read_text());assert prior['contract']==contract;start=prior['next_index']
 last=start
 try:
  with ProcessPoolExecutor(8) as pool:
   for first in range(start,len(rows),128):
    args=[(rows[i][1],i,maximum,NEW_KEYS) for i in range(first,min(first+128,len(rows)))]
    for i,v in enumerate(pool.map(extract_record,args,chunksize=8),first):
     guard()
     for key in NEW_KEYS:arrays[key][i]=v[key]
     last=i+1
    if last%128==0 or last==len(rows):
     for a in arrays.values():a.flush()
     save(progress,dict(contract=contract,next_index=last,time=now()));print('CACHE',last,'/',len(rows),flush=True)
 finally:
  for a in arrays.values():a.flush()
  save(progress,dict(contract=contract,next_index=last,time=now()))
 if last==len(rows):save(done,dict(contract=contract,shapes=shapes,complete=True,finished_at=now()))

class Data(OldData):
 def __init__(self):
  super().__init__();self.views={'base':self.padded,'full':self.full,'pooled':self.maps[:,0]};self.stats={}
 def view(self,key):
  if key not in self.views:self.views[key]=np.load(CACHE/(key+'.npy'),mmap_mode='r')
  return self.views[key]
 def scaler(self,tr,cfg):
  key=cfg['view'];key=key if key in ['s10','s50','f128'] else 'base'
  h=hashlib.sha256(np.asarray(tr,dtype=np.int64).tobytes()).hexdigest();cachekey=(h,key)
  if cachekey not in self.stats:
   if key=='base':self.stats[cachekey]=fit_scaler(self.maps[tr,0].astype(np.float32))
   else:self.stats[cachekey]=fit_scaler(self.view(key+'_pool')[tr].astype(np.float32))
  return self.stats[cachekey]

class InferenceData:
 def __init__(self,paths,ids,keys,max_samples=211007):
  self.ids=np.asarray(ids);self.spec={'max_samples':max_samples};self.views={};self.y=None
  with ProcessPoolExecutor(8) as pool:rows=list(pool.map(extract_record,[(str(p),i,max_samples,keys) for i,p in enumerate(paths)],chunksize=8))
  for key in keys:self.views[key]=np.stack([r[key] for r in rows])
  self.duration=np.asarray([r['duration'] for r in rows]);self.meta={'lengths':np.asarray([r['length'] for r in rows])}
 def view(self,key):return self.views[key]

class Detector(nn.Module):
 def __init__(self,cfg):
  super().__init__();self.cfg=cfg;base=ResearchCNN(CONFIGS['P01']);self.convs=base.convs;self.head=base.head
  if cfg['kind'].startswith('frequency_'):self.head=nn.Sequential(nn.Linear(1728,64),nn.GELU(),nn.Dropout(.3),nn.Linear(64,1))
 def forward(self,x,aux,heat=False):
  crops=x.ndim==5
  if crops:b,k=x.shape[:2];x=x.reshape(b*k,*x.shape[2:])
  h=self.convs(x);kind=self.cfg['kind']
  if kind.startswith('local_'):
   v=torch.cat([h.mean(2),h.std(2)],1).transpose(1,2);z=self.head(v).squeeze(-1)
   times=torch.arange(z.shape[1],device=z.device)*.08+.0475;bins=times.floor().long();seconds=int((z.shape[1]-1)*.08+.0475)+1
   valid=times[None,:]<aux['duration'][:,None]
   assignment=F.one_hot(bins,num_classes=seconds).to(z.dtype)
   sums=(z*valid)@assignment
   counts=valid.float()@assignment;mask=counts>0;zs=sums/counts.clamp_min(1)
   p=(zs*mask).sum(1)/mask.sum(1).clamp_min(1) if kind=='local_mean' else torch.logsumexp((2*zs).masked_fill(~mask,-1e9),1)/2-torch.log(mask.sum(1).clamp_min(1))/2
   return (p,zs,mask) if heat else p
  if kind=='frequency_head':v=torch.cat([h.mean(3),h.std(3)],1).flatten(1)
  else:
   v=torch.cat([h.mean((2,3)),h.std((2,3))],1)
   if kind=='frequency_control':v=v[:,:,None].expand(-1,-1,9).flatten(1)
  z=self.head(v).squeeze(-1)
  return z.reshape(b,k).mean(1) if crops else z

def inputs(d,ix,cfg,sc,starts=None,augmentation_rng=None):
 raw=d.view(cfg['view'])[ix].astype(np.float32)
 if augmentation_rng is not None and cfg.get('augmentation'):
  kind=cfg['augmentation'];choose=augmentation_rng.random(len(ix))<.5
  if kind=='noise_real':choose &= d.y[ix]==0
  if choose.any():raw[choose]=d.view('shift' if kind=='shift' else 'noise')[ix[choose]].astype(np.float32)
 a=normalize(raw,sc);a=np.pad(a,[(0,0),(0,0),(0,10)]).transpose(0,2,1)[:,None].copy()
 if cfg['kind'].startswith('crop_'):
  assert starts is not None
  a=np.stack([np.stack([v[...,int(s):int(s)+100] for s in starts[int(idx)]]) for v,idx in zip(a,ix)])
 return torch.from_numpy(a).to(DEVICE),{'duration':torch.from_numpy(d.duration[ix].astype(np.float32)).to(DEVICE)}

def choose_starts(seconds,mask,lengths):
 out=[]
 for scores,valid,n in zip(seconds,mask,lengths):
  maximum=max(0,int(n)-100);chosen=[]
  for sec in np.argsort(-scores,kind='stable'):
   if not valid[sec]:continue
   start=min(int(sec)*100,maximum)
   if all(abs(start-old)>=50 for old in chosen):chosen.append(start)
   if len(chosen)==3:break
  while len(chosen)<3:chosen.append(chosen[-1] if chosen else 0)
  out.append(chosen)
 return np.asarray(out,np.int32)

@torch.no_grad()
def predict(model,d,ix,cfg,sc,starts=None,heat=False):
 model.eval();out=[];seconds=[];masks=[]
 for at in range(0,len(ix),8):
  guard();x,aux=inputs(d,ix[at:at+8],cfg,sc,starts)
  if heat:p,z,m=model(x,aux,True);seconds.append(z.cpu().numpy());masks.append(m.cpu().numpy())
  else:p=model(x,aux)
  out.append(p.cpu().numpy())
 return np.concatenate(out).astype(np.float64),np.concatenate(seconds) if seconds else None,np.concatenate(masks) if masks else None

def crop_starts(d,cfg,sc,seed,selector_folder=None):
 if cfg['kind']=='crop_uniform':return np.stack([np.rint(np.linspace(0,max(0,int(n)-100),3)).astype(np.int32) for n in d.meta['lengths']])
 if cfg['kind']!='crop_selected':return None
 assert selector_folder is not None
 source=Path(selector_folder)/f'seed_{seed}.pt';model=Detector(ARMS['L001']).to(DEVICE);model.load_state_dict(torch.load(source,map_location='cpu',weights_only=True))
 _,seconds,mask=predict(model,d,np.arange(len(d.ids)),ARMS['L001'],sc,heat=True);del model
 return choose_starts(seconds,mask,d.meta['lengths'])

class Interrupted(RuntimeError):pass
def fit(d,tr,cfg,sc,seed,dest,starts=None,n_each=3000,stop_after=None):
 guard();seed_everything(seed);model=Detector(cfg).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['schedule_epochs']);rng=np.random.default_rng(seed);aug=np.random.default_rng(seed+9081)
 bs=cfg['batch'];neg=tr[d.y[tr]==0];pos=tr[d.y[tr]==1];ckpt=dest/f'checkpoint_{seed}.pt'
 state=dict(epoch=1,cursor=0,chosen=None,total=0.,history=[],steps=0);contract=dict(cfg=cfg,seed=seed,n_each=n_each)
 if ckpt.exists():
  c=torch.load(ckpt,map_location='cpu',weights_only=False);assert c['contract']==contract and np.array_equal(c['train_idx'],tr)
  model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);sched.load_state_dict(c['scheduler']);state=c['progress'];rng.bit_generator.state=c['sampling_rng'];aug.bit_generator.state=c['augmentation_rng'];random.setstate(c['python_rng']);torch.set_rng_state(c['torch_rng'])
  if DEVICE.type=='mps':torch.mps.set_rng_state(c['mps_rng'])
  print('RESUME',dest.name,seed,state['epoch'],state['cursor'],flush=True)
 def checkpoint():atomic_torch(ckpt,dict(contract=contract,train_idx=tr,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),progress=state,sampling_rng=rng.bit_generator.state,augmentation_rng=aug.bit_generator.state,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if DEVICE.type=='mps' else None))
 try:
  while state['epoch']<=cfg['epochs']:
   model.train();started=time.monotonic()
   if state['chosen'] is None:
    chosen=np.r_[rng.choice(neg,n_each,replace=True),rng.choice(pos,n_each,replace=True)];rng.shuffle(chosen);state['chosen']=chosen
   for at in range(state['cursor'],len(state['chosen']),bs):
    guard();ix=state['chosen'][at:at+bs];x,aux=inputs(d,ix,cfg,sc,starts,aug);x=x+torch.randn_like(x)*.03
    if rng.random()<.5:
     off=int(rng.integers(0,x.shape[-1]-8));x[...,off:off+8]=0
    p=model(x,aux);target=torch.from_numpy(d.y[ix].astype(np.float32)).to(DEVICE);loss=F.binary_cross_entropy_with_logits(p,target)
    opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
    state['total']+=float(loss.detach().cpu())*len(ix);state['cursor']=at+len(ix);state['steps']+=1
    if state['steps']%max(1,math.ceil(2000/bs))==0:checkpoint()
    if stop_after is not None and state['steps']>=stop_after:checkpoint();raise Interrupted('Intentional resume check')
   sched.step();row=dict(epoch=state['epoch'],bce=state['total']/len(state['chosen']),seconds=round(time.monotonic()-started,2),updates=state['steps'],battery=power.battery())
   state['history'].append(row);print('EPOCH',dest.name,seed,json.dumps(row),flush=True)
   state.update(epoch=state['epoch']+1,cursor=0,chosen=None,total=0.);checkpoint()
 except BatteryLow:checkpoint();raise
 return model,state['history']

def summarize():
 r=json.loads((OUT/'local_results.json').read_text());summary={}
 for a,cfg in ARMS.items():
  vals={fid:f['arms'][a]['metrics']['unseen_attack_auc'] for fid,f in r['folds'].items() if a in f['arms']}
  if not vals:continue
  deltas={fid:v-r['folds'][fid]['arms'][cfg['control']]['metrics']['unseen_attack_auc'] for fid,v in vals.items() if cfg['control'] in r['folds'][fid]['arms']}
  seed_deltas={};seed_auc={}
  for fid,f in r['folds'].items():
   if a not in f['arms']:continue
   seed_auc[fid]={str(m['seed']):m['metrics']['unseen_attack_auc'] for m in f['arms'][a]['members']}
   if cfg['control'] in f['arms']:
    other={str(m['seed']):m['metrics']['unseen_attack_auc'] for m in f['arms'][cfg['control']]['members']}
    seed_deltas[fid]={s:v-other[s] for s,v in seed_auc[fid].items() if s in other}
  summary[a]=dict(n_folds=len(vals),mean_unseen_auc=float(np.mean(list(vals.values()))),worst_unseen_auc=min(vals.values()),unseen_auc_by_fold=vals,paired_mean_delta=float(np.mean(list(deltas.values()))) if deltas else None,paired_deltas=deltas,control=cfg['control'],unseen_auc_by_seed=seed_auc,paired_seed_deltas=seed_deltas)
  graded=json.loads(JOURNAL.read_text())['ideas']['R4_'+a].get('public_score') is not None
  update('R4_'+a,research_v2=summary[a],**({} if graded else dict(status='completed_local' if len(vals)==len(FOLDS) else 'partial_local')))
 save(OUT/'local_summary.json',summary);return summary

def local(d):
 p=OUT/'local_results.json';result=json.loads(p.read_text()) if p.exists() else dict(folds={},created_at=now())
 definitions={f['id']:f for f in json.loads((ROOT/'reports/research_v2/validation_audit.json').read_text())['folds']}
 for a,cfg in ARMS.items():
  if cfg.get('derived_from'):continue
  for fid in FOLDS:
   split=np.load(ROOT/f'reports/research_v2/split_{fid}.npz');tr=split['train'];va=split['validation'];fold=definitions[fid]
   assert not set(d.speakers[tr])&set(d.speakers[va]) and fold['heldout_attack'] not in d.attacks[tr]
   entry=result['folds'].setdefault(fid,dict(fold=fold,arms={}))
   if a in entry['arms']:continue
   guard();dest=LOCAL/fid/a;dest.mkdir(parents=True,exist_ok=True);sc=d.scaler(tr,cfg);joblib.dump(sc,dest/'normalizer.joblib')
   members=[];predictions=[]
   for seed in SEEDS:
    receipt=dest/f'local_seed_{seed}.json';predfile=dest/f'predictions_seed_{seed}.npz'
    if receipt.exists():
     member=json.loads(receipt.read_text());arr=np.load(predfile);assert np.array_equal(arr['idx'],va);pred=arr['logits']
    else:
     selector=LOCAL/fid/'L001';starts=crop_starts(d,cfg,sc,seed,selector)
     print('START LOCAL',fid,a,seed,flush=True);log('R4_'+a,'local_training_started',fold=fid,seed=seed)
     model,h=fit(d,tr,cfg,sc,seed,dest,starts);pred,z,m=predict(model,d,va,cfg,sc,starts,heat=cfg['kind'].startswith('local_'))
     atomic_torch(dest/f'seed_{seed}.pt',model.cpu().state_dict());np.savez(predfile,idx=va,logits=pred,seconds=z if z is not None else [],mask=m if m is not None else [])
     member=dict(seed=seed,metrics=evaluate(d,va,pred,fold['heldout_attack']),history=h,parameters=sum(p.numel() for p in model.parameters()))
     save(receipt,member);del model
     if DEVICE.type=='mps':torch.mps.empty_cache()
    members.append(member);predictions.append(pred)
   pred=np.mean(predictions,0);np.savez(dest/'predictions.npz',idx=va,logits=pred)
   entry['arms'][a]=dict(metrics=evaluate(d,va,pred,fold['heldout_attack']),members=members,seeds=SEEDS,parameters=members[0]['parameters'],seed_auc_std=float(np.std([m['metrics']['unseen_attack_auc'] for m in members])))
   save(p,result);summarize();print('RESULT LOCAL',fid,a,entry['arms'][a]['metrics']['unseen_attack_auc'],flush=True);log('R4_'+a,'fold_complete',fold=fid,metrics=entry['arms'][a]['metrics'])
 summarize()

def refit(d,a,out=FINAL):
 cfg=ARMS[a];dest=out/a;dest.mkdir(parents=True,exist_ok=True);path=dest/'ensemble.json'
 if cfg.get('derived_from'):
  source=refit(d,cfg['derived_from'],out);meta=json.loads((source/'ensemble.json').read_text());meta.update(arm=a,cfg=cfg,seeds=[SEED],members=[m for m in meta['members'] if m['seed']==SEED],derived_from=cfg['derived_from'],complete=True)
  for name in ['normalizer.joblib',meta['members'][0]['file']]:shutil.copy2(source/name,dest/name)
  save(path,meta);return dest
 if path.exists():
  meta=json.loads(path.read_text());assert meta['cfg']==cfg
  if meta['complete']:return dest
 else:
  sc=d.scaler(np.arange(len(d.ids)),cfg);joblib.dump(sc,dest/'normalizer.joblib');meta=dict(engine='research_v4',arm=a,cfg=cfg,seeds=cfg.get('final_seeds',SEEDS),members=[],temperature=5.,feature_config=d.spec,complete=False,created_at=now());save(path,meta)
 sc=joblib.load(dest/'normalizer.joblib');tr=np.arange(len(d.ids))
 if cfg['kind']=='crop_selected':refit(d,'L001',out)
 for seed in meta['seeds']:
  if any(m['seed']==seed for m in meta['members']):continue
  starts=crop_starts(d,cfg,sc,seed,out/'L001');print('START REFIT',a,seed,flush=True)
  model,h=fit(d,tr,cfg,sc,seed,dest,starts);name=f'seed_{seed}.pt';atomic_torch(dest/name,model.cpu().state_dict());del model
  member=dict(seed=seed,file=name,history=h)
  if cfg['kind']=='crop_selected':
   selector=f'selector_{seed}.pt';shutil.copy2(out/'L001'/name,dest/selector);member['selector_file']=selector
  meta['members'].append(member);save(path,meta)
  if DEVICE.type=='mps':torch.mps.empty_cache()
 meta['complete']=True;save(path,meta);log('R4_'+a,'full_train_refit_complete',models=str(dest));return dest

def infer(folder,input_dir,output,sample=None,data=None):
 folder=Path(folder);meta=json.loads((folder/'ensemble.json').read_text());assert meta['complete'];cfg=meta['cfg']
 ids=pd.read_csv(sample)['idx'].astype(str).tolist() if sample else sorted(p.name for p in input_dir.iterdir() if p.suffix.lower() in ['.wav','.flac'])
 assert ids and len(ids)==len(set(ids)) and all(Path(v).name==v and (input_dir/v).is_file() for v in ids)
 keys={cfg['view']}
 if cfg['kind']=='crop_selected':keys.add('base')
 d=data or InferenceData([input_dir/v for v in ids],ids,sorted(keys));assert np.array_equal(ids,d.ids);sc=joblib.load(folder/'normalizer.joblib');ps=[];heat=[];selected=[]
 for member in meta['members']:
  if cfg['kind']=='crop_selected':
   selector=Detector(ARMS['L001']).to(DEVICE);selector.load_state_dict(torch.load(folder/member['selector_file'],map_location='cpu',weights_only=True));_,z,m=predict(selector,d,np.arange(len(ids)),ARMS['L001'],sc,heat=True);del selector;starts=choose_starts(z,m,d.meta['lengths']);selected.append(starts)
  else:starts=crop_starts(d,cfg,sc,member['seed'])
  model=Detector(cfg).to(DEVICE);model.load_state_dict(torch.load(folder/member['file'],map_location='cpu',weights_only=True));p,z,m=predict(model,d,np.arange(len(ids)),cfg,sc,starts,heat=cfg['kind'].startswith('local_'));ps.append(p);del model
  if z is not None:heat.append(z)
 logits=np.mean(ps,0);scores=expit(logits/meta['temperature']);assert np.isfinite(scores).all()
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(dict(idx=ids,score=scores)).to_csv(output,index=False,float_format='%.17g');np.savez(output.with_suffix('.npz'),idx=ids,score=scores,logits=logits,seconds=np.asarray(heat),selected_starts=np.asarray(selected))
 receipt=dict(arm=meta['arm'],file=str(output),upload_file=str(output),rows=len(ids),sha256=digest(output),inference_only=True,models=str(folder));save(output.with_suffix('.json'),receipt);return receipt

def package(a):
 from importlib.metadata import version
 dest=ROOT/f'releases/research_r4_{a}';dest.mkdir(parents=True,exist_ok=True)
 for name in ['research_v4.py','research_v3.py','features.py','networks.py','research_runner.py','research_final.py','research_features.py','experiment_journal.py']:shutil.copy2(ROOT/name,dest/name)
 models=dest/'models/final';models.mkdir(parents=True,exist_ok=True)
 for p in (FINAL/a).iterdir():
  if p.name in ['ensemble.json','normalizer.joblib'] or p.name.startswith(('seed_','selector_')):shutil.copy2(p,models/p.name)
 (dest/'predict.py').write_text("import argparse\nfrom pathlib import Path\nfrom research_v4 import infer\nif __name__=='__main__':\n p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--sample',type=Path);p.add_argument('--output',type=Path,default=Path('submission.csv'));a=p.parse_args();print(infer(Path(__file__).resolve().parent/'models/final',a.input,a.output,a.sample))\n")
 (dest/'train.py').write_text(f"import argparse\nfrom pathlib import Path\nfrom features import ROOT,make_cache\nfrom research_features import build,prepare_headers\nfrom research_v4 import build_cache,Data,refit\nif __name__=='__main__':\n p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--out',type=Path,default=ROOT/'models/retrained');a=p.parse_args();assert (a.train/'targets.txt').is_file() and (a.train/'flac').is_dir();(ROOT/'data').mkdir(exist_ok=True);link=ROOT/'data/train'\n if link.exists():assert link.resolve()==a.train.resolve()\n else:link.symlink_to(a.train.resolve(),target_is_directory=True)\n if not (ROOT/'data/cache/train_features.npz').exists():make_cache(ROOT/'data',ROOT/'data/cache',8)\n prepare_headers();build();build_cache();print(refit(Data(),'{a}',a.out))\n")
 (dest/'requirements.txt').write_text('\n'.join(p+'=='+version(p) for p in ['numpy','scipy','soundfile','torch','scikit-learn','pandas','joblib','tqdm'])+'\n')
 (dest/'README.md').write_text(f"# RuCode C R4 {a}\n\n{ARMS[a]['title']}\n\nFrozen final seeds: {ARMS[a].get('final_seeds',SEEDS)}. Supplied native mono 16kHz audio only, no inference adaptation. Cap 211007 samples. Parameters in models/final/ensemble.json.\n\nPredict: `python predict.py --input /path/audio --sample /path/sample.csv --output submission.csv`. Also saves NPZ; localizer scores are weakly supervised, not ground-truth per-second probabilities.\n\nTrain: `python train.py --train /path/train --out /path/fresh_output`. All feature caches together need about 45GB. Crop-selected model additionally trains its L001 selector on train only. Atomic resume checkpoints; supplied weights preserved.\n")
 archive=ROOT/f'RuCode2026_C_research_r4_{a}.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(dest.rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(dest))
 return dict(archive=str(archive),release=str(dest),sha256=digest(archive))

def verify_release(a,receipt,release):
 path=OUT/'release_reproducibility.json';checks=json.loads(path.read_text()) if path.exists() else {}
 if a in checks and checks[a]['archive_sha256']==release['sha256'] and checks[a]['sha256']==receipt['sha256']:return
 with tempfile.TemporaryDirectory(prefix='rucode_r4_verify_') as temp:
  temp=Path(temp)
  with zipfile.ZipFile(release['archive']) as z:z.extractall(temp)
  output=temp/'submission.csv';p=subprocess.run([str(ROOT/'.venv/bin/python'),str(temp/'predict.py'),'--input',str(ROOT/'data/public_test'),'--sample',str(ROOT/'data/public_sample_submission.csv'),'--output',str(output)],capture_output=True,text=True)
  if p.returncode:raise RuntimeError(a+' standalone verification failed: '+p.stderr[-2000:])
  assert digest(output)==receipt['sha256'],a
 checks[a]=dict(byte_identical=True,sha256=receipt['sha256'],archive_sha256=release['sha256'],verified_at=now());save(path,checks);print('VERIFIED',a,flush=True)

def pipeline():
 freeze();build_cache();d=Data()
 uploads=json.loads((OUT/'upload_manifest.json').read_text()) if (OUT/'upload_manifest.json').exists() else {}
 releases=json.loads((OUT/'release_archives.json').read_text()) if (OUT/'release_archives.json').exists() else {}
 data=None
 for a in ARMS:
  guard();refit(d,a)
  if a in uploads:continue
  if data is None:
   ids=pd.read_csv(ROOT/'data/public_sample_submission.csv')['idx'].astype(str).tolist();keys=sorted({c['view'] for c in ARMS.values()});data=InferenceData([ROOT/'data/public_test'/i for i in ids],ids,keys)
  receipt=infer(FINAL/a,ROOT/'data/public_test',ROOT/f'submissions/research_r4/{a}/submission_seed_{SEED}.csv',ROOT/'data/public_sample_submission.csv',data);release=package(a);verify_release(a,receipt,release)
  frozenpath=OUT/'refits_frozen.json';frozen=json.loads(frozenpath.read_text()) if frozenpath.exists() else dict(created_at=now(),files={})
  frozen['files'].update({str(p.relative_to(ROOT)):digest(p) for p in (FINAL/a).iterdir() if p.is_file() and (p.name in ['ensemble.json','normalizer.joblib'] or p.name.startswith(('seed_','selector_')))})
  frozen['complete']=len(uploads)+1==len(ARMS);save(frozenpath,frozen)
  uploads[a]=receipt;releases[a]=release;save(OUT/'release_archives.json',releases);save(OUT/'final_candidates.json',dict(complete=len(uploads)==len(ARMS),arms=list(ARMS),submissions=uploads,created_at=now()));save(OUT/'upload_manifest.json',uploads)
 local(d)
 summary=summarize();order=sorted(summary,key=lambda a:(-summary[a]['worst_unseen_auc'],-summary[a]['mean_unseen_auc'],a));save(OUT/'primary_selection.json',dict(train_only=True,selected=order[0],ranking=order,selected_at=now(),rule='worst unseen AUC, mean unseen AUC, ID'))
 save(OUT/'pipeline_status.json',dict(complete=True,finished_at=now(),battery=power.battery(True)));print('ALL R4 CANDIDATES READY',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','cache','run']);a=p.parse_args()
 if a.command=='run':
  status=OUT/'pipeline_status.json'
  if status.exists():
   history=OUT/'status_history';history.mkdir(exist_ok=True);shutil.copy2(status,history/f'before_resume_{os.getpid()}.json')
  save(OUT/'active_process.json',dict(pid=os.getpid(),started_at=now(),protocol_sha256=digest(OUT/'protocol.json')))
  save(status,dict(complete=False,running=True,pid=os.getpid(),started_at=now()))
 try:
  if a.command=='freeze':freeze()
  elif a.command=='cache':build_cache()
  else:pipeline()
 except BatteryLow as e:save(OUT/'pipeline_status.json',dict(complete=False,stopped_low_battery=True,time=now(),message=str(e),battery=power.battery(True)));print(str(e),flush=True)
