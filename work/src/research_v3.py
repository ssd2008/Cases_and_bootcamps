"""Five predeclared train-only hypotheses, resumable on battery power."""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import argparse, hashlib, json, random, re, shutil, subprocess, time, zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import expit
from features import ROOT
from networks import DEVICE, seed_everything, normalize
from research_runner import Data, ResearchCNN, CONFIGS, SEED, OUT as OLD_OUT, save, evaluate
from research_final import InferenceData
from experiment_journal import JOURNAL, now, log, render, update

OUT=ROOT/'reports/research_v3'
LOCAL=ROOT/'models/research_v3'
FINAL=ROOT/'models/research_final_r3'
CORR_LAGS=[1,2,4,8,12,16,24,32]
ARMS={
 'M00':dict(title='Контроль маски: нормировка до паддинга, обычная CNN',kind='mask',masked=False,duration=False,control='P01'),
 'M01':dict(title='Маска добавленного паддинга в нормировках и pooling',kind='mask',masked=True,duration=False,control='M00'),
 'M02':dict(title='Маска паддинга и отдельная длительность',kind='mask',masked=True,duration=True,control='M01'),
 'B00':dict(title='Контроль специалистов: перемешанные частотные полосы',kind='bands',ordered=False,control='P01'),
 'B01':dict(title='Специалисты четырёх физических частотных диапазонов',kind='bands',ordered=True,control='B00'),
 'F00':dict(title='Контроль отдельной ветки частотных корреляций',kind='correlation',correlation=False,control='P01'),
 'F01':dict(title='Корреляции удалённых частот в отдельной ветке',kind='correlation',correlation=True,control='F00'),
 'S00':dict(title='Контроль головы диктора с отсоединённым encoder',kind='speaker',adversarial=False,control='P01'),
 'S01':dict(title='Обращение градиента распознавания диктора',kind='speaker',adversarial=True,control='S00'),
 'D01':dict(title='Повышение веса трудных способов генерации',kind='dro',control='P01'),
}
PROTOCOL={
 'arms':ARMS,'local_seeds':[SEED],'final_seeds':[SEED,SEED+1],
 'epochs':4,'n_each':3000,'batch':8,'optimizer':'AdamW lr=.001 wd=.01',
 'schedule_epochs':14,'augmentations':'Gaussian .03; random eight-frame block masking, identical per matched pair',
 'fold_order':['G5_A05','G1_A06','G3_A03','G1_A01','G2_A02','G4_A04'],
 'stages':'three difficult folds for every arm, all two-seed full-train refits, remaining three folds, frozen public inference; no new public feedback until all training finishes',
 'mask':'Only synthetic padding is masked. Genuine pauses retained. Input centered on real clip before padding; M00 matches this preprocessing but keeps ordinary BN/pooling. M01 masks BN, convolution activations, average pooling, final moments; M02 adds train-standardized log duration to same 193-input head (zero scalar in M00/M01).',
 'bands':'Four independent grouped-convolution local experts, 16 contiguous bands each, plus full-context expert. B00 uses one fixed permutation of 64 frequency bands; B01 physical order. Both have identical parameter count. This contrasts physical band locality/order, not capacity alone.',
 'correlation':{'lags':CORR_LAGS,'definition':'per-frame Pearson correlation over overlapping frequency segments of the raw padded log spectrum; constant spectra map to zero','branch':'8→16→32→32 temporal CNN, mean/std aggregation, zero inputs in F00'},
 'speaker':'Training speaker labels only. Auxiliary 192→64→n-speakers classifier, gradient reversal .1 for S01, detached encoder for S00. No dropout in speaker head. Main and auxiliary gradient norms clipped separately to 5.',
 'dro':'Class-balanced draws as P01; positive attack-group weights updated once per epoch with eta=4 using .5*mean attack BCE+.5*mean bonafide BCE. Multiplicative weights clipped to [.5/K,2/K] then renormalized (max/min ratio <=4). Held-out attack never participates.',
 'controls':'Historical P01 has identical base architecture/sampling/optimizer for D01. Other hypotheses have new matched controls. Historical cross-run comparisons remain subject to MPS noise.',
 'battery':'Prevent idle sleep on battery with caffeinate -i; screen allowed to sleep. Atomic checkpoints every 250 batches and every epoch, including optimizer/RNG/sample cursor. Stop after saving at <=1% if still discharging; cannot prevent physical exhaustion or lid-close sleep.',
 'limits':'One local seed and six paired speaker/attack folds. Two final seeds. No guarantee of public/private transfer; no changes after public scores.'}

class BatteryLow(RuntimeError):pass
_last_battery_time=0.
_last_battery=None
def battery(force=False):
 global _last_battery_time,_last_battery
 if force or time.monotonic()-_last_battery_time>30:
  output=subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout if shutil.which('pmset') else ''
  m=re.search(r'(\d+)%',output)
  _last_battery={'time':now(),'percent':int(m[1]) if m else None,'discharging':"Battery Power" in output and 'discharging' in output}
  _last_battery_time=time.monotonic()
  OUT.mkdir(parents=True,exist_ok=True)
  with (OUT/'battery.jsonl').open('a') as f:f.write(json.dumps(_last_battery)+'\n')
 return _last_battery

def guard():
 b=battery()
 if b and b['discharging'] and b['percent'] is not None and b['percent']<=1:
  raise BatteryLow('Checkpoint preserved; battery at or below one percent')

def atomic_torch(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:
  torch.save(value,f);f.flush();os.fsync(f.fileno())
 tmp.replace(path)

def correlations(raw):
 # raw: utterance,time,frequency. Pearson over actual overlapping bands.
 x=np.asarray(raw,dtype=np.float32);out=[]
 for lag in CORR_LAGS:
  a=x[...,:-lag];b=x[...,lag:]
  a=a-a.mean(-1,keepdims=True);b=b-b.mean(-1,keepdims=True)
  den=np.sqrt(np.sum(a*a,-1)*np.sum(b*b,-1))
  out.append(np.divide(np.sum(a*b,-1),np.maximum(den,1e-6)))
 return np.stack(out,axis=1).astype(np.float32)

def correlation_cache(d):
 path=ROOT/'data/cache/research_v3_correlations.npy';done=path.with_suffix('.json')
 if done.exists():
  meta=json.loads(done.read_text());assert meta['lags']==CORR_LAGS and meta['n']==len(d.ids)
  return np.load(path,mmap_mode='r')
 partial=path.with_suffix('.partial.npy')
 dest=np.lib.format.open_memmap(partial,mode='w+',dtype='float16',shape=(len(d.ids),len(CORR_LAGS),d.spec['max_frames']))
 for i in range(0,len(d.ids),128):
  guard();dest[i:i+128]=correlations(d.padded[i:i+128])
 dest.flush();del dest;partial.replace(path)
 save(done,{'lags':CORR_LAGS,'n':len(d.ids),'source':'train-only provided padded log spectra','created_at':now()})
 return np.load(path,mmap_mode='r')

class MaskedBN(nn.BatchNorm2d):
 def forward(self,x,mask):
  if self.training:
   n=(mask.sum()*x.shape[-2]).clamp_min(1.)
   mean=(x*mask).sum((0,2,3))/n
   var=((x-mean[None,:,None,None]).square()*mask).sum((0,2,3))/n
   with torch.no_grad():
    self.num_batches_tracked.add_(1)
    self.running_mean.lerp_(mean.detach(),self.momentum)
    self.running_var.lerp_((var*n/(n-1).clamp_min(1.)).detach(),self.momentum)
  else:mean=self.running_mean;var=self.running_var
  y=(x-mean[None,:,None,None])*torch.rsqrt(var[None,:,None,None]+self.eps)
  return y*self.weight[None,:,None,None]+self.bias[None,:,None,None]

class Reverse(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,strength):ctx.strength=strength;return x.view_as(x)
 @staticmethod
 def backward(ctx,g):return -ctx.strength*g,None

def tower(in_channels,widths,groups=1):
 layers=[];old=in_channels
 for i,w in enumerate(widths):
  layers.extend([nn.Conv2d(old,w,3,padding=1,groups=groups),nn.BatchNorm2d(w),nn.GELU()])
  if i<3:layers.append(nn.AvgPool2d(2))
  old=w
 return nn.Sequential(*layers)

def moments(h,mask=None):
 if mask is None:return torch.cat([h.mean((2,3)),h.std((2,3))],1)
 n=(mask.sum((2,3))*h.shape[-2]).clamp_min(1.)
 mean=(h*mask).sum((2,3))/n
 var=((h-mean[:,:,None,None]).square()*mask).sum((2,3))/(n-1).clamp_min(1.)
 return torch.cat([mean,torch.sqrt(var.clamp_min(1e-12))],1)

class Detector(nn.Module):
 def __init__(self,cfg,n_speakers=20):
  super().__init__();self.cfg=cfg
  base=ResearchCNN(CONFIGS['P01']);self.convs=base.convs;self.head=base.head
  kind=cfg['kind']
  if kind=='mask':
   if cfg['masked']:
    for i,layer in enumerate(self.convs):
     if isinstance(layer,nn.BatchNorm2d):self.convs[i]=MaskedBN(layer.num_features)
   self.head=nn.Sequential(nn.Linear(193,64),nn.GELU(),nn.Dropout(.3),nn.Linear(64,1))
  elif kind=='bands':
   self.convs=tower(4,[24,48,96,144],groups=4)
   self.context=tower(1,[6,12,24,36])
   self.head=nn.Sequential(nn.Linear(360,64),nn.GELU(),nn.Dropout(.3),nn.Linear(64,1))
   order=np.arange(64) if cfg['ordered'] else np.random.default_rng(SEED+113).permutation(64)
   self.register_buffer('order',torch.from_numpy(order))
  elif kind=='correlation':
   self.aux=nn.Sequential(nn.Conv1d(8,16,5,padding=2),nn.GELU(),nn.AvgPool1d(2),nn.Conv1d(16,32,5,padding=2),nn.GELU(),nn.AvgPool1d(2),nn.Conv1d(32,32,3,padding=1),nn.GELU())
   self.head=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Dropout(.3),nn.Linear(64,1))
  elif kind=='speaker':
   state=torch.get_rng_state()
   self.domain=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,n_speakers))
   torch.set_rng_state(state)
 def encode(self,x,aux):
  kind=self.cfg['kind']
  if kind=='bands':
   raw=x[:,:,:64,:];local=raw[:,0,self.order,:].reshape(len(x),4,16,x.shape[-1])
   return torch.cat([moments(self.convs(local)),moments(self.context(x))],1)
  if kind=='mask' and self.cfg['masked']:
   h=x;mask=aux['mask']
   for layer in self.convs:
    if isinstance(layer,MaskedBN):h=layer(h,mask)
    elif isinstance(layer,nn.AvgPool2d):
     h=layer(h*mask);mask=F.avg_pool2d(mask,(1,2));h=h/mask.clamp_min(1e-6)
    else:h=layer(h)
    h=h*mask.gt(0)
   v=moments(h,mask)
  else:v=moments(self.convs(x))
  if kind=='mask':v=torch.cat([v,aux['duration'] if self.cfg['duration'] else torch.zeros_like(aux['duration'])],1)
  if kind=='correlation':
   a=self.aux(aux['correlation']);v=torch.cat([v,a.mean(-1),a.std(-1)],1)
  return v
 def forward(self,x,aux,domain=False):
  v=self.encode(x,aux);p=self.head(v).squeeze(-1)
  if domain:
   z=Reverse.apply(v,.1) if self.cfg['adversarial'] else v.detach()
   return p,self.domain(z)
  return p

def extra_scaler(d,tr):
 z=np.log1p(d.duration[tr]);return {'duration_mean':float(z.mean()),'duration_std':float(max(z.std(),1e-4))}

def inputs(d,ix,cfg,sc,extra,corr=None):
 aux={}
 if cfg['kind']=='mask':
  raw=d.full[ix].astype(np.float32);lengths=d.meta['lengths'][ix]
  mask=np.arange(raw.shape[1])[None,:]<lengths[:,None]
  x=normalize(raw,sc['maps'][0]);x*=mask[:,:,None]
  x=np.pad(x,[(0,0),(0,0),(0,10)]).transpose(0,2,1)[:,None].copy()
  x=torch.from_numpy(x).to(DEVICE)
  aux['mask']=torch.from_numpy(mask[:,None,None].astype(np.float32)).to(DEVICE)
  z=(np.log1p(d.duration[ix])-extra['duration_mean'])/extra['duration_std']
  aux['duration']=torch.from_numpy(z.astype(np.float32)[:,None]).to(DEVICE)
 else:x,_=d.input(ix,CONFIGS['P01'],sc)
 if cfg['kind']=='correlation':
  c=np.asarray(corr[ix],dtype=np.float32) if cfg['correlation'] else np.zeros((len(ix),8,x.shape[-1]),np.float32)
  aux['correlation']=torch.from_numpy(c).to(DEVICE)
 return x,aux

@torch.no_grad()
def predict(model,d,ix,cfg,sc,extra,corr=None):
 model.eval();out=[]
 for at in range(0,len(ix),8):
  guard();x,aux=inputs(d,ix[at:at+8],cfg,sc,extra,corr);out.append(model(x,aux).cpu().numpy())
 return np.concatenate(out).astype(np.float64)

def fit(d,tr,cfg,sc,extra,seed,dest,corr=None,epochs=4,n_each=3000):
 guard();seed_everything(seed)
 speakers=sorted(set(d.speakers[tr]));spk={v:i for i,v in enumerate(speakers)}
 attack_names=sorted(set(d.attacks[tr])-{'-'});attack={v:i for i,v in enumerate(attack_names)}
 model=Detector(cfg,len(speakers)).to(DEVICE)
 opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,14);rng=np.random.default_rng(seed)
 neg=tr[d.y[tr]==0];pos=tr[d.y[tr]==1]
 state={'epoch':1,'cursor':0,'chosen':None,'total':0.,'speaker_total':0.,'speaker_correct':0,'history':[],
        'group_q':np.full(len(attack),1/len(attack)),'group_loss':np.zeros(len(attack)),'group_count':np.zeros(len(attack)),
        'negative_loss':0.,'negative_count':0,'epoch_seconds':0.}
 ckpt=dest/f'checkpoint_{seed}.pt'
 if ckpt.exists():
  c=torch.load(ckpt,map_location='cpu',weights_only=False)
  assert c['cfg']==cfg and c['seed']==seed and np.array_equal(c['train_idx'],tr)
  assert c['training_contract']=={'epochs':epochs,'n_each':n_each}
  model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);sched.load_state_dict(c['scheduler']);state=c['progress']
  rng.bit_generator.state=c['numpy_rng'];random.setstate(c['python_rng']);torch.set_rng_state(c['torch_rng'])
  if DEVICE.type=='mps':torch.mps.set_rng_state(c['mps_rng'])
  print('RESUME',dest,seed,state['epoch'],state['cursor'],flush=True)
 def checkpoint():
  atomic_torch(ckpt,{'cfg':cfg,'seed':seed,'train_idx':tr,'training_contract':{'epochs':epochs,'n_each':n_each},'model':model.state_dict(),'optimizer':opt.state_dict(),
   'scheduler':sched.state_dict(),'progress':state,'numpy_rng':rng.bit_generator.state,
   'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
   'mps_rng':torch.mps.get_rng_state() if DEVICE.type=='mps' else None})
 try:
  while state['epoch']<=epochs:
   ep=state['epoch'];model.train();started=time.monotonic()
   if state['chosen'] is None:
    chosen=np.r_[rng.choice(neg,n_each,replace=True),rng.choice(pos,n_each,replace=True)];rng.shuffle(chosen);state['chosen']=chosen
   chosen=state['chosen']
   for at in range(state['cursor'],len(chosen),8):
    guard();ix=chosen[at:at+8];x,aux=inputs(d,ix,cfg,sc,extra,corr)
    x=x+torch.randn_like(x)*.03
    if cfg['kind']=='mask':x=x*aux['mask']
    if rng.random()<.5:
     off=int(rng.integers(0,x.shape[-1]-8));x[...,off:off+8]=0
     if 'correlation' in aux:aux['correlation'][...,off:off+8]=0
    target=torch.from_numpy(d.y[ix].astype(np.float32)).to(DEVICE)
    if cfg['kind']=='speaker':p,domain=model(x,aux,True)
    else:p=model(x,aux)
    bce=F.binary_cross_entropy_with_logits(p,target,reduction='none')
    if cfg['kind']=='dro':
     group=np.asarray([attack.get(v,-1) for v in d.attacks[ix]])
     weights=np.ones(len(ix),np.float32);positive=group>=0
     weights[positive]=state['group_q'][group[positive]]*len(attack)
     loss=(bce*torch.from_numpy(weights).to(DEVICE)).mean()
     values=bce.detach().cpu().numpy()
     for g in range(len(attack)):
      state['group_loss'][g]+=values[group==g].sum();state['group_count'][g]+=(group==g).sum()
     state['negative_loss']+=float(values[~positive].sum());state['negative_count']+=int((~positive).sum())
    else:loss=bce.mean()
    if cfg['kind']=='speaker':
     speaker_y=torch.tensor([spk[v] for v in d.speakers[ix]],device=DEVICE)
     domain_loss=F.cross_entropy(domain,speaker_y);loss=loss+domain_loss
     state['speaker_total']+=float(domain_loss.detach().cpu())*len(ix)
     state['speaker_correct']+=int((domain.argmax(1)==speaker_y).sum().cpu())
    opt.zero_grad(set_to_none=True);loss.backward()
    main=[v for name,v in model.named_parameters() if not name.startswith('domain.')]
    nn.utils.clip_grad_norm_(main,5)
    if cfg['kind']=='speaker':nn.utils.clip_grad_norm_(model.domain.parameters(),5)
    opt.step();state['total']+=float(bce.detach().mean().cpu())*len(ix);state['cursor']=at+len(ix)
    if state['cursor']%2000==0:checkpoint()
   sched.step();seconds=state['epoch_seconds']+time.monotonic()-started
   row={'epoch':ep,'bce':state['total']/len(chosen),'seconds':round(seconds,2),'battery':battery()}
   if cfg['kind']=='speaker':row.update(speaker_ce=state['speaker_total']/len(chosen),speaker_train_accuracy=state['speaker_correct']/len(chosen))
   if cfg['kind']=='dro':
    risk=.5*state['group_loss']/np.maximum(state['group_count'],1)+.5*state['negative_loss']/max(state['negative_count'],1)
    logits=np.log(state['group_q'])+4*risk;logits-=logits.max();q=np.exp(logits);q/=q.sum()
    q=np.clip(q,.5/len(q),2/len(q));q/=q.sum();state['group_q']=q
    row.update(group_risk=dict(zip(attack_names,risk.tolist())),next_group_weights=dict(zip(attack_names,(q*len(q)).tolist())))
   state['history'].append(row);print('EPOCH',dest.name,seed,json.dumps(row),flush=True)
   state.update(epoch=ep+1,cursor=0,chosen=None,total=0.,speaker_total=0.,speaker_correct=0,group_loss=np.zeros(len(attack)),group_count=np.zeros(len(attack)),negative_loss=0.,negative_count=0,epoch_seconds=0.)
   checkpoint()
 except BatteryLow:
  state['epoch_seconds']+=time.monotonic()-started;checkpoint();raise
 return model,state['history'],speakers

def freeze():
 OUT.mkdir(parents=True,exist_ok=True);path=OUT/'protocol.json'
 if path.exists():assert json.loads(path.read_text())['protocol']==PROTOCOL
 else:save(path,{'created_at':now(),'protocol':PROTOCOL,'initial_battery':battery(True)});log('R3_ALL','protocol_frozen',protocol=PROTOCOL)
 r=json.loads(JOURNAL.read_text())
 for arm,cfg in ARMS.items():
  key='R3_'+arm
  if key not in r['ideas']:r['ideas'][key]={'title':cfg['title'],'author':'Codex','control':'V2_P01' if cfg['control']=='P01' else 'R3_'+cfg['control'],'status':'protocol_frozen','change':json.dumps(cfg,ensure_ascii=False),'notes':PROTOCOL['limits']}
 for key in ['R3_PADMASK','R3_BANDS','R3_FCORR','R3_SPEAKER','R3_DRO']:
  if r['ideas'][key].get('status')!='completed_local_and_public':r['ideas'][key]['status']='implementation_and_testing'
 r['updated_at']=now();JOURNAL.write_text(json.dumps(r,ensure_ascii=False,indent=2));render(r)

def local(d,corr,fold_ids):
 result_path=OUT/'local_results.json';r=json.loads(result_path.read_text()) if result_path.exists() else {'folds':{},'created_at':now()}
 folds={f['id']:f for f in json.loads((OLD_OUT/'validation_audit.json').read_text())['folds']}
 old=json.loads((OLD_OUT/'results.json').read_text())
 for fid in fold_ids:
  fold=folds[fid];split=np.load(OLD_OUT/f'split_{fid}.npz');tr=split['train'];va=split['validation']
  assert not set(d.speakers[tr])&set(d.speakers[va]) and fold['heldout_attack'] not in d.attacks[tr]
  sc=joblib.load(ROOT/'models/research_v2'/fid/'scalers.joblib');extra=extra_scaler(d,tr)
  folder=LOCAL/fid;folder.mkdir(parents=True,exist_ok=True);joblib.dump(sc,folder/'scalers.joblib');save(folder/'extra_scaler.json',extra)
  entry=r['folds'].setdefault(fid,{'fold':fold,'arms':{},'historical_P01':old['folds'][fid]['arms']['P01']['metrics']})
  for arm,cfg in ARMS.items():
   if arm in entry['arms']:continue
   guard();dest=folder/arm;print('START LOCAL',fid,arm,flush=True);log('R3_'+arm,'training_started',fold=fid)
   model,h,speakers=fit(d,tr,cfg,sc,extra,SEED,dest,corr)
   p=predict(model,d,va,cfg,sc,extra,corr)
   atomic_torch(dest/'seed.pt',model.cpu().state_dict());np.savez(dest/'predictions.npz',idx=va,logits=p)
   row={'metrics':evaluate(d,va,p,fold['heldout_attack']),'history':h,'parameters':sum(v.numel() for v in model.parameters()),'train_speakers':speakers,'seed':SEED}
   entry['arms'][arm]=row;save(result_path,r);log('R3_'+arm,'fold_complete',fold=fid,metrics=row['metrics'])
   print('RESULT LOCAL',fid,arm,row['metrics']['unseen_attack_auc'],flush=True)
   del model
   if DEVICE.type=='mps':torch.mps.empty_cache()
 summarize(r)

def summarize(r=None):
 r=r or json.loads((OUT/'local_results.json').read_text());summary={};registry=json.loads(JOURNAL.read_text())
 for arm,cfg in ARMS.items():
  values={fid:f['arms'][arm]['metrics']['unseen_attack_auc'] for fid,f in r['folds'].items() if arm in f['arms']}
  if not values:continue
  deltas={}
  for fid,value in values.items():
   f=r['folds'][fid];control=cfg['control']
   if control=='P01':deltas[fid]=value-f['historical_P01']['unseen_attack_auc']
   elif control in f['arms']:deltas[fid]=value-f['arms'][control]['metrics']['unseen_attack_auc']
  summary[arm]={'n_folds':len(values),'mean_unseen_auc':float(np.mean(list(values.values()))),'worst_unseen_auc':min(values.values()),'unseen_auc_by_fold':values,'control':cfg['control'],'paired_deltas':deltas,'paired_mean_delta':float(np.mean(list(deltas.values()))) if deltas else None}
  status='completed_public_and_six_fold_pilot' if len(values)==6 and registry['ideas']['R3_'+arm].get('public_score') is not None else 'completed_six_fold_local' if len(values)==6 else 'partial_local'
  update('R3_'+arm,status=status,research_v2=summary[arm])
 save(OUT/'local_summary.json',summary);return summary

def refit(d,corr,arm,out=FINAL,cfg=None):
 cfg=cfg if cfg is not None else ARMS[arm];dest=out/arm;dest.mkdir(parents=True,exist_ok=True);path=dest/'ensemble.json'
 if path.exists():
  meta=json.loads(path.read_text());assert meta['cfg']==cfg
  if meta['complete']:return dest
 else:
  tr=np.arange(len(d.ids));sc=joblib.load(ROOT/'models/research_final_v2/train_scalers.joblib') if (ROOT/'models/research_final_v2/train_scalers.joblib').exists() else d.scalers(tr)
  joblib.dump(sc,dest/'scalers.joblib');save(dest/'extra_scaler.json',extra_scaler(d,tr))
  meta={'engine':'research_v3','arm':arm,'cfg':cfg,'feature_config':d.spec,'seeds':[SEED,SEED+1],'epochs':4,'n_each':3000,'temperature':5.,'members':[],'complete':False,'created_at':now(),'inference_length_policy':'Cap each incoming audio to training max 211007 samples; no test-derived parameters'}
  save(path,meta)
 sc=joblib.load(dest/'scalers.joblib');extra=json.loads((dest/'extra_scaler.json').read_text());tr=np.arange(len(d.ids))
 for seed in meta['seeds']:
  if any(m['seed']==seed for m in meta['members']):continue
  guard();print('START REFIT',arm,seed,flush=True)
  model,h,speakers=fit(d,tr,cfg,sc,extra,seed,dest,corr);file=f'seed_{seed}.pt';atomic_torch(dest/file,model.cpu().state_dict())
  meta['members'].append({'seed':seed,'file':file,'history':h,'n_speakers':len(speakers)});save(path,meta);del model
  if DEVICE.type=='mps':torch.mps.empty_cache()
 meta['complete']=True;save(path,meta);log('R3_'+arm,'full_train_refit_complete',models=str(dest));return dest

def infer(folder,input_dir,output,sample=None,data=None):
 meta=json.loads((folder/'ensemble.json').read_text());assert meta['complete'];cfg=meta['cfg']
 ids=pd.read_csv(sample)['idx'].astype(str).tolist() if sample else sorted(p.name for p in input_dir.iterdir() if p.suffix.lower() in ['.wav','.flac'])
 assert ids and len(ids)==len(set(ids)) and all(Path(n).name==n and (input_dir/n).is_file() for n in ids)
 d=data or InferenceData([input_dir/n for n in ids],ids,meta['feature_config'])
 assert np.array_equal(ids,d.ids)
 corr=correlations(d.padded).astype(np.float16) if cfg['kind']=='correlation' else None
 sc=joblib.load(folder/'scalers.joblib');extra=json.loads((folder/'extra_scaler.json').read_text());ps=[]
 for m in meta['members']:
  model=Detector(cfg,m['n_speakers']).to(DEVICE);model.load_state_dict(torch.load(folder/m['file'],map_location='cpu',weights_only=True))
  ps.append(predict(model,d,np.arange(len(ids)),cfg,sc,extra,corr));del model
 logits=np.mean(ps,0);p=expit(logits/meta['temperature']);assert np.isfinite(p).all()
 output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame({'idx':ids,'score':p}).to_csv(output,index=False,float_format='%.17g');np.savez(output.with_suffix('.npz'),idx=ids,score=p,logits=logits)
 receipt={'arm':meta['arm'],'file':str(output),'rows':len(ids),'unique_scores':len(np.unique(p)),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'models':str(folder),'inference_only':True}
 save(output.with_suffix('.json'),receipt);return receipt

def package(arm):
 from importlib.metadata import version
 dest=ROOT/f'releases/research_r3_{arm}';dest.mkdir(parents=True,exist_ok=True)
 for name in ['research_v3.py','features.py','networks.py','research_runner.py','research_final.py','research_features.py','experiment_journal.py']:shutil.copy2(ROOT/name,dest/name)
 models=dest/'models/final';models.mkdir(parents=True,exist_ok=True);meta=json.loads((FINAL/arm/'ensemble.json').read_text())
 for name in ['ensemble.json','scalers.joblib','extra_scaler.json']+[m['file'] for m in meta['members']]:shutil.copy2(FINAL/arm/name,models/name)
 (dest/'predict.py').write_text('''import argparse
from pathlib import Path
from research_v3 import infer
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--sample',type=Path);p.add_argument('--output',type=Path,default=Path('submission.csv'));a=p.parse_args();print(infer(Path(__file__).resolve().parent/'models/final',a.input,a.output,a.sample))
''')
 (dest/'train.py').write_text(f'''import argparse
from pathlib import Path
from features import ROOT,make_cache
from research_features import prepare_headers,build
from research_v3 import Data,correlation_cache,refit
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--out',type=Path,default=ROOT/'models/retrained');a=p.parse_args()
 assert (a.train/'targets.txt').is_file() and (a.train/'flac').is_dir()
 (ROOT/'data').mkdir(exist_ok=True);link=ROOT/'data/train'
 if link.exists():assert link.resolve()==a.train.resolve()
 else:link.symlink_to(a.train.resolve(),target_is_directory=True)
 if not (ROOT/'data/cache/train_features.npz').exists():make_cache(ROOT/'data',ROOT/'data/cache',8)
 prepare_headers();build();d=Data();corr=correlation_cache(d) if {ARMS[arm]['kind']=='correlation'!r} else None
 print(refit(d,corr,'{arm}',out=a.out,cfg={meta['cfg']!r}))
''')
 (dest/'requirements.txt').write_text('\n'.join(p+'=='+version(p) for p in ['numpy','scipy','soundfile','torch','scikit-learn','pandas','joblib','tqdm'])+'\n')
 (dest/'README.md').write_text(f"# RuCode C R3 {arm}\n\n{ARMS[arm]['title']}\n\nFrozen two-seed ensemble, four epochs of 6000 balanced draws, trained only on supplied train. No inference fitting or adaptation. Audio capped to supplied train maximum 13.1879375 s.\n\nPredict: `python predict.py --input /path/audio --sample /path/sample.csv --output submission_seed_{SEED}.csv`\n\nTrain: `python train.py --train /path/train --out /path/fresh_output`. Checkpoints resume interrupted runs. Provided weights remain intact. Data/cache generation requires approximately 13 GB.\n")
 path=ROOT/f'RuCode2026_C_research_r3_{arm}.zip'
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(dest.rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':z.write(p,p.relative_to(dest))
 with zipfile.ZipFile(path) as z:assert z.testzip() is None
 return {'archive':str(path),'release':str(dest),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def pipeline():
 freeze();d=Data();corr=correlation_cache(d)
 local(d,corr,PROTOCOL['fold_order'][:3])
 for arm in ARMS:refit(d,corr,arm)
 local(d,corr,PROTOCOL['fold_order'][3:])
 frozen={'created_at':now(),'complete':True,'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for arm in ARMS for p in (FINAL/arm).iterdir() if p.name in ['ensemble.json','scalers.joblib','extra_scaler.json'] or p.name.startswith('seed_')}}
 save(OUT/'refits_frozen.json',frozen)
 data=InferenceData.load(ROOT/'data/cache/research_v2_public_inference.npz',d.spec)
 uploads={};releases={}
 for arm in ARMS:
  guard();receipt=infer(FINAL/arm,ROOT/'data/public_test',ROOT/f'submissions/research_r3/{arm}/submission_seed_{SEED}.csv',ROOT/'data/public_sample_submission.csv',data)
  uploads[arm]={**receipt,'upload_file':receipt['file']};releases[arm]=package(arm)
 save(OUT/'final_candidates.json',{'complete':True,'arms':list(ARMS),'submissions':uploads,'created_at':now()});save(OUT/'upload_manifest.json',uploads);save(OUT/'release_archives.json',releases)
 save(OUT/'pipeline_status.json',{'complete':True,'finished_at':now(),'battery':battery(True)})
 print('ALL R3 CANDIDATES READY',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 sub.add_parser('run');a=p.parse_args()
 try:pipeline()
 except BatteryLow as e:
  save(OUT/'pipeline_status.json',{'complete':False,'stopped_low_battery':True,'time':now(),'message':str(e),'battery':battery(True)})
  print(str(e),flush=True)
