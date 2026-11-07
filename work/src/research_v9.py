"""R9: frozen P_MIX waveform ablations and exact checkpoint continuations."""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, copy, hashlib, json, math, random, signal, subprocess, time, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch, joblib
from scipy import signal as dsp
from features import ROOT, manifest
from networks import DEVICE, normalize
from research_runner import save, evaluate
from experiment_journal import JOURNAL, now, log, render, update
import research_v7 as r

OUT=ROOT/'reports/research_v9'; FINAL=ROOT/'models/research_final_r9'; LOCAL=ROOT/'models/research_v9'; CACHE=ROOT/'data/cache/research_v9'
SEEDS=r.SEEDS; FOLDS=r.FOLDS; SOURCE=ROOT/'models/research_final_r7/P_MIX'
BASE=copy.deepcopy(r.ARMS['P_MIX']); ORIGINAL_INPUTS=r.inputs
ARMS={}
def arm(name,title,extra=None,epochs=4,source=None,control='R7_P_MIX'):
 ARMS[name]={**BASE,'title':title,'epochs':epochs,'extra_aug':extra,'source_arm':source,'continuation':None,'control':control}
arm('E008','P_MIX: продолжение 4→8 эпох',epochs=8,source='R7_P_MIX')
arm('E010','P_MIX: продолжение 4→10 эпох',epochs=10,source='E008',control='R9_E008')
arm('E012','P_MIX: продолжение 4→12 эпох',epochs=12,source='E010',control='R9_E010')
arm('GAIN','P_MIX: уровень ±6 дБ после RMS-нормировки',extra='gain')
arm('TEMPO','P_MIX: темп 0.9–1.1 с сохранением тона',extra='tempo')
arm('SPEED','P_MIX: скорость 0.9–1.1 с изменением тона',extra='speed')
arm('FLAT','P_MIX: замена коротких участков их средним',extra='flat')
arm('LOCAL','P_MIX: локальный временной сдвиг',extra='local')
arm('GT','P_MIX: уровень + темп',extra='gain_tempo',control='R9_TEMPO')
arm('FL','P_MIX: среднее участков + локальный сдвиг',extra='flat_local',control='R9_FLAT')
arm('ALL','P_MIX: все четыре новых аугментации',extra='all')
arm('ALL008','P_MIX: все аугментации, продолжение 4→8 эпох',extra='all',epochs=8,source='ALL',control='R9_ALL')
arm('ENVELOPE','P_MIX: плавное изменение громкости внутри записи',extra='envelope')
arm('PITCH','P_MIX: небольшой сдвиг тона короткого фрагмента',extra='pitch')
arm('ENV_TEMPO','P_MIX: плавная громкость + темп',extra='envelope_tempo',control='R9_TEMPO')
UPLOAD=list(ARMS); EXTRAS=['gain','tempo','speed','flat','local','gain_tempo','flat_local','all','envelope','pitch','envelope_tempo']
# Existing waveform decisions and all sampling/torch RNGs remain unchanged.
def derived_rng(rng):
 payload=json.dumps(rng.bit_generator.state,sort_keys=True).encode()+b'R9_EXTRA_AUG'
 return np.random.default_rng(int.from_bytes(hashlib.blake2b(payload,digest_size=8).digest(),'little'))

def active_start(x,width,rng):
 width=min(width,len(x)); attempts=rng.integers(0,len(x)-width+1,size=8)
 energies=np.array([np.mean(x[s:s+width]**2) for s in attempts])
 good=attempts[energies>=max(float(np.mean(x*x))*.1,1e-10)]
 return int(rng.choice(good)) if len(good) else int(attempts[np.argmax(energies)])

def short_flat(x,rng):
 y=x.copy(); regions=[]
 for _ in range(int(rng.integers(1,4))):
  width=min(int(rng.integers(80,321)),max(1,len(y)//100))
  s=active_start(x,width,rng); original=y[s:s+width].copy(); target=float(original.mean())
  blend=np.ones(width,np.float32); ramp=min(16,width//4)
  if ramp:blend[:ramp]=np.linspace(0,1,ramp);blend[-ramp:]=np.linspace(1,0,ramp)
  y[s:s+width]=original*(1-blend)+target*blend;regions.append([s,width])
 return y,regions

def local_move(x,rng):
 y=x.copy();width=min(int(rng.integers(640,1921)),max(32,len(y)//10));s=active_start(x,width,rng)
 delta=int(rng.integers(32,min(161,max(33,width//4))))*int(rng.choice([-1,1]))
 part=y[s:s+width].copy();moved=np.zeros_like(part)
 if delta>0:moved[delta:]=part[:-delta]
 else:moved[:delta]=part[-delta:]
 blend=np.ones(width,np.float32);ramp=min(32,width//4)
 blend[:ramp]=np.linspace(0,1,ramp);blend[-ramp:]=np.linspace(1,0,ramp)
 y[s:s+width]=part*(1-blend)+moved*blend
 return y,dict(start=s,width=width,delta=delta)

def transformed(x,index,maximum,key):
 # Same transform parameters for clean/noisy alternatives; one cached view/file.
 y=x.copy();details={};prefix=SEEDS[0]+19015+index
 if key in ['tempo','speed','gain_tempo','all','envelope_tempo']:
  rate_rng=np.random.default_rng(prefix+1);requested=float(rate_rng.uniform(.9,1.1))
  rate=max(requested,len(y)/maximum);details.update(requested_rate=requested,effective_rate=rate,slowdown_limited=rate!=requested)
  if key=='speed':y=dsp.resample_poly(y,10000,int(round(rate*10000))).astype(np.float32)
  else:
   import librosa
   y=librosa.effects.time_stretch(y,rate=rate,n_fft=512,hop_length=128,win_length=400).astype(np.float32)
  y=y[:maximum];details['output_samples']=len(y)
 if key in ['envelope','envelope_tempo']:
  points=np.random.default_rng(prefix+5).uniform(-6,6,size=5);db=np.interp(np.arange(len(y)),np.linspace(0,len(y)-1,5),points)
  y*=np.power(10,db/20).astype(np.float32);details['envelope_db']=points.tolist()
 if key=='pitch':
  import librosa
  rng=np.random.default_rng(prefix+6);width=min(int(rng.integers(3200,9601)),len(y));start=active_start(y,width,rng);steps=float(rng.uniform(.5,1.))*int(rng.choice([-1,1]))
  original=y[start:start+width].copy();changed=librosa.effects.pitch_shift(original,sr=16000,n_steps=steps,n_fft=512,hop_length=128,win_length=400)
  blend=np.ones(width,np.float32);fade=min(160,width//4);blend[:fade]=np.linspace(0,1,fade);blend[-fade:]=np.linspace(1,0,fade)
  y[start:start+width]=original*(1-blend)+changed*blend;details['local_pitch']=dict(start=start,width=width,semitones=steps)
 if key in ['flat','flat_local','all']:y,details['flat_regions']=short_flat(y,np.random.default_rng(prefix+2))
 if key in ['local','flat_local','all']:y,details['local_shift']=local_move(y,np.random.default_rng(prefix+3))
 y/=np.sqrt(np.mean(y*y))+1e-9
 if key in ['gain','gain_tempo','all']:
  db=float(np.random.default_rng(prefix+4).uniform(-6,6));y*=np.float32(10**(db/20));details['gain_db']=db
 return y,details

def extract_extra(args):
 path,index,maximum,key=args
 x,sr=sf.read(path,dtype='float32');assert sr==16000 and x.ndim==1
 x=x[:maximum];x/=np.sqrt(np.mean(x*x))+1e-9
 if len(x)<1024:x=np.pad(x,(0,1024-len(x)))
 rng=np.random.default_rng(SEEDS[0]+3901+index);snr=rng.uniform(20,35)
 noisy=x+rng.normal(size=len(x)).astype(np.float32)*np.float32(10**(-snr/20));noisy/=np.sqrt(np.mean(noisy*noisy))+1e-9
 views={};details=None
 for n,source in [(0,x),(1,noisy)]:
  y,details=transformed(source,index,maximum,key)
  left=int(np.random.default_rng(SEEDS[0]+4901+index).integers(0,(maximum-len(y))//160+1))*160
  for shifted,offset in [(0,0),(1,left)]:
   views[f'{key}_{n}{shifted}']=r.old.stft_map(np.pad(y,(offset,maximum-len(y)-offset))).astype(np.float16)
 return views,details

def build_extra(key):
 CACHE.mkdir(parents=True,exist_ok=True);rows=manifest(ROOT/'data');spec=r.read(ROOT/'data/cache/research_v2/complete.json')
 maximum=spec['max_samples'];keys=[f'{key}_{n}{s}' for n in [0,1] for s in [0,1]]
 contract=dict(version=1,key=key,rows=len(rows),ids_sha256=hashlib.sha256('\n'.join(v[0] for v in rows).encode()).hexdigest(),maximum=maximum,keys=keys)
 done=CACHE/f'{key}_complete.json';progress=CACHE/f'{key}_progress.json'
 if done.exists():assert r.read(done)['contract']==contract;return
 arrays={k:np.lib.format.open_memmap(CACHE/f'{k}.npy',mode='r+' if (CACHE/f'{k}.npy').exists() else 'w+',dtype='float16',shape=(len(rows),spec['max_frames'],64)) for k in keys}
 state=r.read(progress,dict(next_index=0,contract=contract,limited_slowdowns=0));assert state['contract']==contract
 first=state['next_index'];details_file=CACHE/f'{key}_parameters.jsonl'
 # Chunks commit only after all workers return; unfinished chunks are recomputed.
 with ProcessPoolExecutor(6) as pool:
  for start in range(first,len(rows),128):
   r.guard();jobs=[(rows[i][1],i,maximum,key) for i in range(start,min(start+128,len(rows)))];batch=list(pool.map(extract_extra,jobs,chunksize=8))
   with details_file.open('a') as f:
    for i,(views,details) in enumerate(batch,start):
     for k in keys:arrays[k][i]=views[k]
     f.write(json.dumps(dict(index=i,**details))+'\n');state['limited_slowdowns']+=int(details.get('slowdown_limited',False))
   for a in arrays.values():a.flush()
   state.update(next_index=start+len(batch),time=now());save(progress,state)
   print('CACHE',key,state['next_index'],'/',len(rows),flush=True)
 save(done,dict(contract=contract,complete=True,finished_at=now(),limited_slowdowns=state['limited_slowdowns']))

class Data(r.Data):
 def view(self,key):
  if key.rsplit('_',1)[0] in EXTRAS and key not in self.views:self.views[key]=np.load(CACHE/f'{key}.npy',mmap_mode='r')
  return super().view(key)

def inputs(d,ix,cfg,sc,augmentation_rng=None):
 if augmentation_rng is None or cfg.get('extra_aug') is None:return ORIGINAL_INPUTS(d,ix,cfg,sc,augmentation_rng)
 # Reproduce original noise/shift decisions exactly, then choose an extra view
 # from a separate stream derived without advancing augmentation_rng.
 extra=derived_rng(augmentation_rng).random(len(ix))<.5
 noise=np.zeros(len(ix),bool);shift=np.zeros(len(ix),bool)
 if cfg['noise_all']:noise|=augmentation_rng.random(len(ix))<cfg['noise_all']
 if cfg['noise_real']:noise|=(augmentation_rng.random(len(ix))<cfg['noise_real'])&(d.y[ix]==0)
 if cfg['shift']:shift=augmentation_rng.random(len(ix))<cfg['shift']
 raw=d.view('base')[ix].astype(np.float32)
 for n in [0,1]:
  for s in [0,1]:
   mask=(noise==bool(n))&(shift==bool(s))
   if not mask.any():continue
   oldkey={(0,0):'base',(0,1):'shift',(1,0):'noise',(1,1):'noise_shift'}[(n,s)]
   oldmask=mask&~extra;newmask=mask&extra
   if oldmask.any():raw[oldmask]=d.view(oldkey)[ix[oldmask]].astype(np.float32)
   if newmask.any():raw[newmask]=d.view(f'{cfg["extra_aug"]}_{n}{s}')[ix[newmask]].astype(np.float32)
 a=normalize(raw,sc);a=np.pad(a,[(0,0),(0,0),(0,10)]).transpose(0,2,1)[:,None].copy()
 return torch.from_numpy(a).to(DEVICE),{}

r.inputs=inputs;r.OUT=OUT;r.old.power.OUT=OUT
r.log=lambda idea,event,**kw:log('R9_FIT' if idea=='R7_FIT' else idea,event,**kw)

def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 protocol=dict(arms=ARMS,submission_order=UPLOAD,seeds=SEEDS,folds=FOLDS,max_new_submissions=15,user_cap=15,initial_remaining_snapshot=88,reserve=73,
  authorization='15Oct user explicitly requested five hypotheses on P_MIX99.90, autonomous overnight work and at most15 public posts. Fifteen candidates predeclared; no optional expansion from public grades.',
  baseline='Exact P_MIX architecture, padded waveform/STFT, noise_real=.5/noise_all=.5/shift=.5, batch8,6000 balanced draws/epoch, two seeds, original frozen per-band scaler, mean logits and fixed temperature5. Inference identical to R7 P_MIX.',
  gain='Audit physical gain before RMS as near-no-op. GAIN multiplies normalized training waveform by10**(db/20), db~U[-6,6], after RMS and before STFT; global log centering removes most gain except fixed log floor/silence. No clipping. This is explicitly a level/floor robustness ablation, not a new amplitude input channel.',
  tempo='One fixed rate U[.9,1.1] per training record. TEMPO preserves pitch using librosa phase vocoder; SPEED uses rational resampling and also changes pitch. If slowdown would exceed original fixed train maximum, rate is capped at length/maximum; no content crop except at most rounding samples. Native sample rate stays16k. Same rate for clean/noisy views.',
  flat='1–3 short5–20ms spans capped at1% of clip each, sampled preferentially from active sound. Replace by local arithmetic mean with1ms edge fades. Constant amplitude is not a musical tone. Labels preserved for both classes.',
  envelope='Five random gain knots U[-6,6]dB linearly interpolated across the actual clip, before final RMS. Unlike global gain this time-varying gain survives normalization. ENV_TEMPO adds pitch-preserving tempo.',
  pitch='One active200–600ms span, pitch shift ±0.5–1semitone with preserved duration using librosa and10ms crossfades, then final RMS. This is the frequency interpretation of local shift, separate from temporal LOCAL.',
  local_shift='One40–120ms region capped at10% of clip, internal shift±2–10ms, zero-fill vacated samples and2ms fades to original at region boundaries. Total length unchanged. This locally discards a few samples, explicitly unlike the lossless global P_MIX shift.',
  composition='Every single/new combination selected independently withp=.5 from a separate derived stream; existing waveform noise/shift decisions are unchanged. Combos jointly apply their stated transforms in order tempo, flat, local, RMS, gain. Existing noise precedes new transforms; global shift follows. Cached deterministic alternate per file matches previous augmentation methodology.',
  epochs='E008/E010/E012 continue exact original4-epoch P_MIX optimizer/RNG state to8,10,12, chaining checkpoints without changing sources. Original cosine horizon14 preserved, no training past horizon. ALL008 continues the new ALL4 checkpoint to8. No checkpoint/epoch selection by public metrics.',
  local='All15 variants on same3 speaker/attack-disjoint splits and2seeds:45 ensembles/90 members. Epoch variants continue original local P_MIX checkpoints. Historical exact P_MIX is paired baseline; report per-seed and ensemble deltas, three-fold worst/mean, correlation descriptive only.',
  grades='All15 public artifacts frozen and repeat-inference verified before reading any new grade. Confirmed/ambiguous posts never repeated. Live quota checked before each post, new budget established from actual counter, maximum15.',
  power='Own temporary caffeinate only, display may sleep. Checkpoint every2000 examples/epoch and signal pause, guard discharging<=1%. No global settings changes.',
  limits='Small repeated public benchmark and only3 local generator groups; phase vocoder/flat/shift can themselves introduce synthetic artifacts. Gain mostly removed by invariant preprocessing. No guarantee of private benefit.')
 p=OUT/'protocol.json'
 if p.exists():assert r.read(p)['protocol']==protocol
 else:save(p,dict(created_at=now(),protocol=protocol,submission_order=UPLOAD));log('R9_ALL','protocol_frozen',protocol=protocol)
 registry=r.read(JOURNAL)
 for a,cfg in ARMS.items():
  registry['ideas'].setdefault('R9_'+a,dict(title=cfg['title'],author='Кирилл',control=cfg['control'],status='protocol_frozen',change=json.dumps(cfg,ensure_ascii=False),notes='Fixed overnight study15Oct, P_MIX99.90 base; fifteen of at most15 authorised posts.'))
 registry['updated_at']=now();save(JOURNAL,registry);render(registry)
 if not (OUT/'previous_best.json').exists():save(OUT/'previous_best.json',r.read(ROOT/'models/current_best.json'))

def extend(a,seed,dest,tr,fid=None):
 cfg=ARMS[a];source=cfg['source_arm']
 if not source or (dest/f'checkpoint_{seed}.pt').exists() or (dest/f'seed_{seed}.pt').exists():return
 src=(ROOT/f'models/research_v7/{fid}/P_MIX' if fid else SOURCE) if source=='R7_P_MIX' else ((LOCAL/fid/source) if fid else FINAL/source)
 p=src/f'checkpoint_{seed}.pt';c=torch.load(p,map_location='cpu',weights_only=False)
 start=4 if source in ['R7_P_MIX','ALL'] else ARMS[source]['epochs']
 assert c['progress']['epoch']==start+1 and c['progress']['cursor']==0 and len(c['progress']['history'])==start
 assert np.array_equal(c['train_idx'],tr) and c['contract']['cfg']['schedule_epochs']==14
 weights=torch.load(src/f'seed_{seed}.pt',map_location='cpu',weights_only=True)
 assert all(torch.equal(weights[k],c['model'][k]) for k in weights)
 before=dict(checkpoint=str(p.relative_to(ROOT)),checkpoint_sha256=r.digest(p),weight=str((src/f'seed_{seed}.pt').relative_to(ROOT)),weight_sha256=r.digest(src/f'seed_{seed}.pt'),original_contract=c['contract'])
 c['contract']=dict(cfg=cfg,seed=seed,n_each=3000);r.atomic_torch(dest/f'checkpoint_{seed}.pt',c)
 save(dest/f'continuation_{seed}.json',dict(**before,start_epoch=start+1,end_epoch=cfg['epochs'],optimizer_rng_preserved=True,time=now()))

def refit(d,a):
 dest=FINAL/a;dest.mkdir(parents=True,exist_ok=True);cfg=ARMS[a]
 if r.read(dest/'ensemble.json',{}).get('complete'):return
 sc=joblib.load(SOURCE/'normalizer.joblib');joblib.dump(sc,dest/'normalizer.joblib');tr=np.arange(len(d.ids));members=[]
 for seed in SEEDS:
  extend(a,seed,dest,tr);model,h=r.fit(d,tr,cfg,sc,seed,dest);members.append(dict(seed=seed,file=f'seed_{seed}.pt',history=h));del model;r.clean()
 save(dest/'ensemble.json',dict(engine='research_v7',arm=a,cfg=cfg,feature_config=d.spec,seeds=SEEDS,members=members,temperature=5.,complete=True,study='research_v9'))

def queue(a,public):
 q=r.read(OUT/'upload_manifest.json',{})
 if a in q:return
 path=ROOT/f'submissions/research_r9/{a}.csv';item=r.infer(FINAL/a,ROOT/'data/public_test',path,ROOT/'data/public_sample_submission.csv',public)
 repeat=path.with_suffix('.repeat.csv');r.infer(FINAL/a,ROOT/'data/public_test',repeat,ROOT/'data/public_sample_submission.csv',public);assert r.digest(repeat)==item['sha256']
 item.update(verified_byte_identical=True,frozen_at=now(),frozen_files={str(p.relative_to(ROOT)):r.digest(p) for p in (FINAL/a).iterdir() if p.name in ['ensemble.json','normalizer.joblib'] or p.name.startswith('seed_')})
 q[a]=item;save(OUT/'upload_manifest.json',q);update('R9_'+a,status='verified_ready',final_models=str(FINAL/a));print('READY',a,flush=True)

def summarize(result):
 base=r.read(ROOT/'reports/research_v7/local_results.json');summary={}
 for a,cfg in ARMS.items():
  values={fid:f['arms'][a]['metrics']['unseen_attack_auc'] for fid,f in result['folds'].items() if a in f['arms']}
  if not values:continue
  deltas={fid:value-base['folds'][fid]['arms']['P_MIX']['metrics']['unseen_attack_auc'] for fid,value in values.items()}
  summary[a]=dict(n_folds=len(values),mean_unseen_auc=float(np.mean(list(values.values()))),worst_unseen_auc=min(values.values()),unseen_auc_by_fold=values,paired_deltas=deltas,paired_mean_delta=float(np.mean(list(deltas.values()))),control='R7_P_MIX')
  update('R9_'+a,research_v2=summary[a])
 save(OUT/'local_summary.json',summary)

def local(d):
 result=r.read(OUT/'local_results.json',dict(folds={}));definitions={f['id']:f for f in r.read(ROOT/'reports/research_v2/validation_audit.json')['folds']}
 for fid in FOLDS:
  split=np.load(ROOT/f'reports/research_v2/split_{fid}.npz');tr=split['train'];va=split['validation'];fold=result['folds'].setdefault(fid,dict(fold=definitions[fid],arms={}))
  assert not set(d.speakers[tr])&set(d.speakers[va]) and definitions[fid]['heldout_attack'] not in d.attacks[tr]
  sc=joblib.load(ROOT/f'models/research_v7/{fid}/P_MIX/normalizer.joblib')
  for a,cfg in ARMS.items():
   if a in fold['arms']:continue
   dest=LOCAL/fid/a;dest.mkdir(parents=True,exist_ok=True);joblib.dump(sc,dest/'normalizer.joblib');members=[];ps=[]
   for seed in SEEDS:
    pf=dest/f'predictions_{seed}.npz';receipt=dest/f'result_{seed}.json'
    if pf.exists() and receipt.exists():arr=np.load(pf);assert np.array_equal(arr['idx'],va);p=arr['logits'];member=r.read(receipt)
    else:
     extend(a,seed,dest,tr,fid);model,h=r.fit(d,tr,cfg,sc,seed,dest);p=r.predict(model,d,va,cfg,sc);np.savez(pf,idx=va,logits=p);member=dict(seed=seed,history=h,metrics=evaluate(d,va,p,definitions[fid]['heldout_attack']));save(receipt,member);del model;r.clean()
    ps.append(p);members.append(member)
   p=np.mean(ps,0);np.savez(dest/'predictions.npz',idx=va,logits=p);fold['arms'][a]=dict(metrics=evaluate(d,va,p,definitions[fid]['heldout_attack']),members=members,seeds=SEEDS)
   save(OUT/'local_results.json',result);summarize(result);print('LOCAL',fid,a,fold['arms'][a]['metrics']['unseen_attack_auc'],flush=True)

def checks():
 rows=manifest(ROOT/'data');maximum=r.read(ROOT/'data/cache/research_v2/complete.json')['max_samples'];audit=[]
 for i in [0,100,1000,10000,20000]:
  x,sr=sf.read(rows[i][1],dtype='float32');u=x/(np.sqrt(np.mean(x*x))+1e-9);scaled=x*2;v=scaled/(np.sqrt(np.mean(scaled*scaled))+1e-9)
  a=r.old.stft_map(np.pad(u,(0,maximum-len(u))));b=r.old.stft_map(np.pad(v,(0,maximum-len(v))));post=r.old.stft_map(np.pad(u*2,(0,maximum-len(u))))
  audit.append(dict(index=i,gain_before_RMS_wave_max=float(np.max(np.abs(u-v))),gain_before_RMS_map_max=float(np.max(np.abs(a-b))),gain_after_RMS_map_mean=float(np.mean(np.abs(a-post)))))
 toy=np.sin(2*np.pi*220*np.arange(16000)/16000).astype(np.float32)
 flat,regions=short_flat(toy,np.random.default_rng(8));move,detail=local_move(toy,np.random.default_rng(8))
 assert len(flat)==len(move)==len(toy) and np.isfinite(flat).all() and np.isfinite(move).all()
 rng=np.random.default_rng(18);before=copy.deepcopy(rng.bit_generator.state);derived_rng(rng).random(100);assert before==rng.bit_generator.state
 save(OUT/'implementation_checks.json',dict(gain_audit=audit,flat_regions=regions,local_shift=detail,unchanged_lengths=True,new_rng_does_not_advance_old=True,time=now()))
 print('INITIAL CHECKS',audit,flush=True)

def run():
 freeze();checks();save(OUT/'active_process.json',dict(pid=os.getpid(),started_at=now()))
 def pause(signum,frame):raise KeyboardInterrupt('Requested checkpoint pause')
 signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGINT,pause)
 save(OUT/'pipeline_status.json',dict(complete=False,running=True,pid=os.getpid(),phase='epoch_continuations',time=now()))
 d=Data();ids=pd.read_csv(ROOT/'data/public_sample_submission.csv')['idx'].astype(str).tolist();public=r.old.InferenceData([ROOT/'data/public_test'/i for i in ids],ids,['base'])
 for a in ['E008','E010','E012']:refit(d,a);queue(a,public)
 for a,cfg in ARMS.items():
  if a in ['E008','E010','E012']:continue
  save(OUT/'pipeline_status.json',dict(complete=False,running=True,pid=os.getpid(),phase='augmentation_cache_and_fits',arm=a,time=now()))
  build_extra(cfg['extra_aug']);refit(d,a);queue(a,public)
 save(OUT/'pipeline_status.json',dict(complete=False,running=True,pid=os.getpid(),final_candidates_complete=True,phase='local',time=now()))
 local(d);save(OUT/'pipeline_status.json',dict(complete=True,final_candidates_complete=True,running=False,finished_at=now()))
 print('R9 TRAINING COMPLETE',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','check','run']);a=parser.parse_args()
 try:
  if a.command=='freeze':freeze()
  elif a.command=='check':OUT.mkdir(parents=True,exist_ok=True);checks()
  else:run()
 except (r.old.BatteryLow,KeyboardInterrupt):
  save(OUT/'pipeline_status.json',dict(complete=False,running=False,paused=True,time=now()));raise
 except Exception as e:
  save(OUT/'pipeline_status.json',dict(complete=False,running=False,error=type(e).__name__+': '+str(e),time=now()));traceback.print_exc();raise
