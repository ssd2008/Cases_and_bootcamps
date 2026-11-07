"""Paired new hypotheses on acoustic-group-disjoint, unseen-synthesis folds.
All choices are fixed before any evaluation; no public records are read here.
"""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import argparse,json,time,math,hashlib
from pathlib import Path
import numpy as np
import torch,joblib
from torch import nn
from torch.nn import functional as F
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from features import ROOT,time_pool
from networks import MapCNN,DEVICE,seed_everything,fit_scaler,normalize
from experiment_journal import now,log
from research_features import CACHE

def metrics(y,p,attacks):
 out={'auc':float(roc_auc_score(y,p))}
 for attack in sorted(set(attacks)-{'-'}):
  selected=(attacks==attack)|(y==0);out[attack]=float(roc_auc_score(y[selected],p[selected]))
 out['robust_auc']=min(out[k] for k in out if k.startswith('A'))
 return out

OUT=ROOT/'reports/research_v2';MODELS=ROOT/'models/research_v2'
SEED=639320752
CONFIGS={
 'C00':{'title':'Исходный zero74','source':'old'},
 'P00':{'title':'Вся запись до 13.19 с, сжатая в 96 интервалов','source':'pooled'},
 'P01':{'title':'Паддинг нулевого звука до максимума train','source':'padding','control':'P00'},
 'P02':{'title':'Обрезка до минимума: центр','source':'crop_center','control':'P00'},
 'P03':{'title':'Обрезка до минимума: максимальная энергия','source':'crop_energy','control':'P02'},
 'P04':{'title':'Обрезка до минимума: максимальный spectral flux','source':'crop_flux','control':'P02'},
 'W00':{'title':'Шесть равномерных физических окон, средний логит','source':'windows_uniform','control':'P02'},
 'W01':{'title':'Шесть окон по энергии','source':'windows_energy','control':'W00'},
 'W02':{'title':'Шесть окон по spectral flux','source':'windows_flux','control':'W00'},
 'W03':{'title':'Отдельный обучаемый селектор шести окон','source':'windows_uniform','selector':True,'control':'W00'},
 'S00':{'title':'Контроль трёх плоскостей: спектр и два нуля','source':'planes','planes':[0,None,None],'control':'P00'},
 'S01':{'title':'Один спектр: окно 10 мс','source':'single','map':1,'control':'P00'},
 'S02':{'title':'Один спектр: окно 50 мс','source':'single','map':2,'control':'P00'},
 'S03':{'title':'Контроль: две одинаковые карты 25 мс','source':'planes','planes':[0,0,None],'control':'S00'},
 'S04':{'title':'Карты 25 и 10 мс','source':'planes','planes':[0,1,None],'control':'S03'},
 'S05':{'title':'Карты 25 и 50 мс','source':'planes','planes':[0,2,None],'control':'S03'},
 'D01':{'title':'Разность по времени до сжатия','source':'planes','planes':[0,3,None],'control':'S00'},
 'D02':{'title':'Разность по соседним частотам','source':'planes','planes':[0,4,None],'control':'S00'},
 'F00':{'title':'Контроль скалярной ветки с нулевыми входами','source':'pooled','scalar':'zeros','control':'P00'},
 'F01':{'title':'Максимум, среднее и медиана частотных перепадов','source':'pooled','scalar':'frequency','control':'F00'},
 'H01':{'title':'Фаза соседних кадров: cos и sin','source':'planes','planes':[0,5,6],'control':'S00'},
 'H02':{'title':'Фаза с подавлением тихих частот','source':'planes','planes':[0,7,8],'control':'H01'},
 'T01':{'title':'PE секунд на чисто спектральной модели','source':'old','time_pe':True,'control':'C00'},
}
# Fixed independently of outcomes. These are pilot settings, not the earlier 12k-example epochs.
PROTOCOL={'created_at':now(),'epochs':4,'examples_per_class_per_epoch':3000,'schedule_epochs':14,
 'seed':SEED,'batch_equivalent_frames':128*96,'selection':'mean and worst held-out-attack ROC-AUC; paired fold deltas; no validation checkpoint selection',
 'window_length':'minimum provided training duration, 0.6524375 s; cached STFT crops contain 64 frames',
 'crop_note':'Windows are selected on the full STFT grid (10ms hop), not recomputed after cutting the waveform. Boundary-context difference from exact waveform cutting is acknowledged.',
 'padding':'Exact waveform zero padding to max train duration 13.1879375 s, including silence in STFT and per-file centering; inference clips longer than train max will require separate policy.',
 'selector':'Separate 128→32→1 MLP scores six fixed-grid candidate windows, softmax aggregation trained jointly via utterance labels; no continuous boundary regressor or oracle labels.',
 'augmentations':'Gaussian .03 and 8-frame time block masking; identical across paired arms, padded silence is not masked out of CNN.',
 'blends':'Predeclared equal-logit P00+S01+S02; W00 aggregation top 1/3 checked without retraining.',
 'limits':'One seed pilot. Six group/attack combinations are not a full 30-fold grid. Acoustic groups may also reflect text/channel. MPS is not strictly deterministic.'}

def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)

class ResearchCNN(MapCNN):
 def __init__(self,cfg):
  super().__init__(74);self.cfg=cfg
  if cfg['source'] in ['planes','padding_planes']:
   old=self.convs[0];state=torch.get_rng_state();stem=nn.Conv2d(3,16,3,padding=1);torch.set_rng_state(state)
   with torch.no_grad():stem.weight.zero_();stem.weight[:,:1].copy_(old.weight);stem.bias.copy_(old.bias)
   self.convs[0]=stem
  if cfg.get('scalar'):
   self.head=nn.Sequential(nn.Linear(195,64),nn.GELU(),nn.Dropout(.3),nn.Linear(64,1))
  if cfg.get('selector'):
   self.selector=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))
   nn.init.zeros_(self.selector[-1].weight);nn.init.zeros_(self.selector[-1].bias)
 def forward(self,x,scalars=None,return_windows=False):
  if x.ndim==5:
   b,k,c,h,t=x.shape;flat=x.reshape(b*k,c,h,t)
  else:flat=x;b=k=None
  h=self.convs(flat);v=torch.cat([h.mean((2,3)),h.std((2,3))],1)
  if self.cfg.get('scalar'):v=torch.cat([v,scalars],1)
  logits=self.head(v).squeeze(-1)
  if b is not None:
   logits=logits.reshape(b,k)
   if self.cfg.get('selector'):
    raw=flat[:,:,:64,:].squeeze(1)
    desc=torch.cat([raw.mean(2),raw.std(2)],1).reshape(b,k,128)
    weights=torch.softmax(self.selector(desc).squeeze(-1),1)
    pred=(weights*logits).sum(1)
   else:weights=torch.full_like(logits,1/k);pred=logits.mean(1)
   return (pred,logits,weights) if return_windows else pred
  return logits

class Data:
 def __init__(self):
  self.meta=np.load(CACHE/'metadata.npz');self.spec=json.loads((CACHE/'complete.json').read_text())
  self.maps=np.load(CACHE/'maps.npy',mmap_mode='r');self.full=np.load(CACHE/'full.npy',mmap_mode='r');self.padded=np.load(CACHE/'padded.npy',mmap_mode='r')
  self.energy=np.load(CACHE/'energy.npy',mmap_mode='r');self.flux=np.load(CACHE/'flux.npy',mmap_mode='r')
  old=np.load(ROOT/'data/cache/train_features.npz');self.old=old['seq'][:,:,:64];self.y=old['y'];self.attacks=old['attacks'];self.speakers=old['speakers'];self.ids=old['ids']
  self.duration=np.load(ROOT/'data/cache/train_headers.npz')['frames']/16000
  assert np.array_equal(self.ids,self.meta['ids'])
 def scalers(self,tr):
  # Scalers for each representation are fitted on training records only.
  out={'maps':[]}
  for j in range(self.maps.shape[1]):out['maps'].append(fit_scaler(self.maps[tr,j].astype(np.float32)))
  out['old']=fit_scaler(self.old[tr]);f=self.meta['frequency_stats'][tr]
  out['scalar']={'mean':f.mean(0),'std':np.maximum(f.std(0),1e-4)}
  return out
 def input(self,ix,cfg,sc):
  src=cfg['source'];aux=None
  if src=='old':a=normalize(self.old[ix],sc['old'])[:,None,:,:]
  elif src in ['pooled','single']:a=normalize(self.maps[ix,cfg.get('map',0)].astype(np.float32),sc['maps'][cfg.get('map',0)])[:,None,:,:]
  elif src=='planes':
   a=np.stack([normalize(self.maps[ix,j].astype(np.float32),sc['maps'][j]) if j is not None else np.zeros((len(ix),96,64),np.float32) for j in cfg['planes']],axis=1)
  elif src=='padding':a=normalize(self.padded[ix].astype(np.float32),sc['maps'][0])[:,None,:,:]
  elif src=='padding_planes':
   # Derive signed adjacent-band differences before scaling, clipping or adding
   # the ten legacy empty rows. The cached spectrum is float16 in both arms.
   raw=self.padded[ix].astype(np.float32)
   delta=np.diff(raw,axis=-1,prepend=raw[...,:1])
   values={'spectrum':normalize(raw,sc['maps'][0]),
           'delta_frequency':normalize(delta,sc['maps'][4])}
   a=np.stack([values[j] if j is not None else np.zeros_like(raw) for j in cfg['planes']],axis=1)
  else:
   win=self.spec['crop_frames'];multi=src.startswith('windows');k=6 if multi else 1;clips=[]
   for idx in ix:
    n=int(self.meta['lengths'][idx]);maximum=max(0,n-win)
    if src in ['crop_center','windows_uniform']:
     starts=np.rint(np.linspace(0,maximum,k)).astype(int) if multi else np.asarray([maximum//2])
    else:starts=self.meta['starts'][idx,0 if 'energy' in src else 1,:k]
    clips.append(np.stack([self.full[idx,int(start):int(start)+win].astype(np.float32) for start in starts]))
   a=normalize(np.stack(clips),sc['maps'][0])
   a=a[:,:,None,:,:] if multi else a[:,0,None,:,:]
  if cfg.get('time_pe'):
   t=(np.arange(a.shape[-2],dtype=np.float32)+.5)/a.shape[-2]*np.minimum(self.duration[ix,None],10)*10
   freq=np.exp(np.arange(0,64,2,dtype=np.float32)*(-np.log(10000)/64));pe=np.zeros((len(ix),a.shape[-2],64),np.float32)
   pe[:,:,0::2]=np.sin(t[:,:,None]*freq);pe[:,:,1::2]=np.cos(t[:,:,None]*freq);a[:,0]+=.1*pe
  # Audio dimension ordering: ... channels,time,freq -> ... channels,freq,time.
  a=np.pad(a,[(0,0)]*(a.ndim-1)+[(0,10)]).swapaxes(-1,-2).copy()
  if cfg.get('scalar'):
   aux=normalize(self.meta['frequency_stats'][ix],sc['scalar']) if cfg['scalar']=='frequency' else np.zeros((len(ix),3),np.float32)
  return torch.from_numpy(a).to(DEVICE),torch.from_numpy(aux).to(DEVICE) if aux is not None else None

def batch_size(cfg,spec):
 if 'batch_override' in cfg:return cfg['batch_override']
 length=spec['max_frames'] if cfg['source'] in ['padding','padding_planes'] else spec['crop_frames']*6 if cfg['source'].startswith('windows') else 96
 # Full-padding batches are aligned to 8; other arms retain adequate BatchNorm sample count.
 return max(8,min(128,8*(int(128*96/length)//8)))

@torch.no_grad()
def predict(model,d,ix,cfg,sc,with_windows=False):
 model.eval();out=[];windows=[];weights=[];bs=batch_size(cfg,d.spec)
 for start in range(0,len(ix),bs):
  x,s=d.input(ix[start:start+bs],cfg,sc)
  if with_windows and cfg['source'].startswith('windows'):
   p,w,a=model(x,s,True);windows.append(w.cpu().numpy());weights.append(a.cpu().numpy())
  else:p=model(x,s)
  out.append(p.cpu().numpy())
 return np.concatenate(out).astype(np.float64),np.concatenate(windows) if windows else None,np.concatenate(weights) if weights else None

def fit(d,tr,cfg,sc,seed,epochs=4,n_each=3000):
 seed_everything(seed);model=ResearchCNN(cfg).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,14);rng=np.random.default_rng(seed)
 neg=tr[d.y[tr]==0];pos=tr[d.y[tr]==1];bs=batch_size(cfg,d.spec);history=[]
 for ep in range(1,epochs+1):
  model.train();start=time.time();chosen=np.r_[rng.choice(neg,n_each,replace=True),rng.choice(pos,n_each,replace=True)];rng.shuffle(chosen);total=0.
  for at in range(0,len(chosen),bs):
   ix=chosen[at:at+bs];x,s=d.input(ix,cfg,sc);x=x+torch.randn_like(x)*.03
   if rng.random()<.5:
    off=int(rng.integers(0,x.shape[-1]-8));x[...,off:off+8]=0
   pred=model(x,s);target=torch.from_numpy(d.y[ix].astype(np.float32)).to(DEVICE);loss=F.binary_cross_entropy_with_logits(pred,target)
   opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach().cpu())*len(ix)
  sched.step();row={'epoch':ep,'loss':total/len(chosen),'seconds':round(time.time()-start,2)};history.append(row);print(row,flush=True)
 return model,history

def evaluate(d,va,p,attack):
 m=metrics(d.y[va],p,d.attacks[va]);m['unseen_attack_auc']=m[attack]
 m['speaker_auc']={str(s):float(roc_auc_score(d.y[va][d.speakers[va]==s],p[d.speakers[va]==s])) for s in np.unique(d.speakers[va])}
 m['mean_speaker_auc']=float(np.mean(list(m['speaker_auc'].values())));m['worst_speaker_auc']=min(m['speaker_auc'].values())
 return m

def run(arms,fold_ids):
 if not (OUT/'protocol.json').exists():
  save(OUT/'protocol.json',{'protocol':PROTOCOL,'configs':CONFIGS});log('V02','research_protocol_frozen',protocol=PROTOCOL,configs=CONFIGS)
 else:
  prior=json.loads((OUT/'protocol.json').read_text());assert prior['configs']==CONFIGS,'Do not change a frozen arm definition'
 d=Data();audit=json.loads((OUT/'validation_audit.json').read_text());result_path=OUT/'results.json'
 r=json.loads(result_path.read_text()) if result_path.exists() else {'created_at':now(),'folds':{},'complete':False}
 for fold in audit['folds']:
  fid=fold['id']
  if fold_ids and fid not in fold_ids:continue
  split=np.load(OUT/f'split_{fid}.npz');tr=split['train'];va=split['validation'];sc=d.scalers(tr)
  folder=MODELS/fid;folder.mkdir(parents=True,exist_ok=True);joblib.dump(sc,folder/'scalers.joblib');entry=r['folds'].setdefault(fid,{'fold':fold,'arms':{}})
  for arm in arms:
   if arm in entry['arms']:continue
   cfg=CONFIGS[arm];print('START',fid,arm,cfg['title'],flush=True);log('V2_'+arm,'research_training_started',fold=fid,seed=SEED)
   model,h=fit(d,tr,cfg,sc,SEED);p,w,a=predict(model,d,va,cfg,sc,True)
   dest=folder/arm;dest.mkdir(exist_ok=True);torch.save(model.cpu().state_dict(),dest/'seed.pt')
   np.savez(dest/'predictions.npz',idx=va,logits=p,window_logits=w if w is not None else np.empty(0),window_weights=a if a is not None else np.empty(0))
   row={'metrics':evaluate(d,va,p,fold['heldout_attack']),'history':h,'seed':SEED,'parameters':sum(v.numel() for v in model.parameters())}
   if w is not None:
    top=np.sort(w,axis=1)[:,-2:].mean(1);row['top_third_metrics']=evaluate(d,va,top,fold['heldout_attack']);row['mean_selector_entropy']=float(np.mean(-np.sum(a*np.log(a+1e-9),axis=1)))
   entry['arms'][arm]=row;r['updated_at']=now();save(result_path,r);log('V2_'+arm,'research_fold_complete',fold=fid,metrics=row['metrics'])
   print('RESULT',fid,arm,json.dumps(row['metrics']),flush=True);del model;torch.mps.empty_cache() if DEVICE.type=='mps' else None
  # Frozen equal blend of separately trained resolutions; evaluated only when all are present.
  if all(a in entry['arms'] for a in ['P00','S01','S02']):
   p=np.mean([np.load(folder/a/'predictions.npz')['logits'] for a in ['P00','S01','S02']],0)
   entry['arms']['E01']={'metrics':evaluate(d,va,p,fold['heldout_attack']),'components':['P00','S01','S02'],'weights':[1/3]*3};save(result_path,r)
 summarize(r)
 print('REQUESTED RUN COMPLETE',flush=True)

def summarize(r=None):
 r=r or json.loads((OUT/'results.json').read_text());arms=sorted(set(a for f in r['folds'].values() for a in f['arms']))
 summary={}
 for arm in arms:
  items=[(fid,f['arms'][arm]['metrics']) for fid,f in r['folds'].items() if arm in f['arms']]
  vals=[m['unseen_attack_auc'] for _,m in items];control=CONFIGS.get(arm,{}).get('control')
  delta=[m['unseen_attack_auc']-r['folds'][fid]['arms'][control]['metrics']['unseen_attack_auc'] for fid,m in items if control in r['folds'][fid]['arms']] if control else []
  summary[arm]={'title':CONFIGS.get(arm,{}).get('title','Равный ансамбль трёх спектральных разрешений'),'n_folds':len(vals),'mean_unseen_auc':float(np.mean(vals)),'worst_unseen_auc':float(min(vals)),'unseen_auc_by_fold':dict((fid,m['unseen_attack_auc']) for fid,m in items),'control':control,'paired_mean_delta':float(np.mean(delta)) if delta else None,'paired_min_delta':float(min(delta)) if delta else None,'paired_wins':sum(v>0 for v in delta)}
 save(OUT/'summary.json',summary)
 text='# Новые эксперименты: локальная проверка\n\nФиксированная эпоха 4, один seed, 6000 сбалансированных примеров на эпоху. Все цифры — локальный ROC-AUC, не баллы ЛБ. Полный протокол: `protocol.json`.\n\n| ID | Идея | Folds | Средний AUC неизвестной атаки | Худший | Средняя Δ к контролю |\n|---|---|---:|---:|---:|---:|\n'
 for a,v in summary.items():text+=f'| {a} | {v["title"]} | {v["n_folds"]} | {v["mean_unseen_auc"]:.5f} | {v["worst_unseen_auc"]:.5f} | '+(f'{v["paired_mean_delta"]:+.5f}' if v['paired_mean_delta'] is not None else '—')+' |\n'
 (OUT/'RESULTS.md').write_text(text)
 return summary

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arms',nargs='+',default=list(CONFIGS));ap.add_argument('--folds',nargs='*');a=ap.parse_args();run(a.arms,a.folds)
