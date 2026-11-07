"""R10: four frozen GAIN/FLAT/LOCAL combinations; no contest networking."""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, copy, fcntl, hashlib, json, signal, subprocess, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch, joblib
import research_v9 as previous
from features import ROOT, manifest
from experiment_journal import JOURNAL, now, log, render, update
from research_runner import save, evaluate

r=previous.r
OUT=ROOT/'reports/research_v10'
FINAL=ROOT/'models/research_final_r10'
LOCAL=ROOT/'models/research_v10'
CACHE=ROOT/'data/cache/research_v10'
OLD_SEEDS=list(previous.SEEDS)
NEW_SEEDS=[639320754,639320755]
FOLDS=list(previous.FOLDS)
COMPONENT={'gain_flat':'flat','gain_local':'local','gain_flat_local':'flat_local'}
ARMS={}
for a,title,key,control in [
 ('GF','GAIN + FLAT','gain_flat','R9_GAIN'),
 ('GL','GAIN + LOCAL','gain_local','R9_GAIN'),
 ('FL_REPEAT','FLAT + LOCAL: независимые seed','flat_local','R9_FL'),
 ('GFL','GAIN + FLAT + LOCAL','gain_flat_local','R9_FL')]:
 ARMS[a]={**copy.deepcopy(previous.BASE),'title':title,'epochs':4,'extra_aug':key,'source_arm':None,'continuation':None,'control':control}
UPLOAD=list(ARMS)
def seeds(a):return NEW_SEEDS if a=='FL_REPEAT' else OLD_SEEDS

class Data(previous.Data):
 def view(self,key):
  if key.rsplit('_',1)[0] in COMPONENT and key not in self.views:
   self.views[key]=np.load(CACHE/f'{key}.npy',mmap_mode='r')
  return super().view(key)

# Training and inference keep the exact R9 input logic; only cached waveform
# combinations and FL_REPEAT's explicitly declared model seeds differ.
r.inputs=previous.inputs
r.OUT=OUT
r.old.power.OUT=OUT
r.log=lambda idea,event,**kw:log('R10_FIT',event,**kw)

def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 protocol=dict(arms=ARMS,submission_order=UPLOAD,seeds={a:seeds(a) for a in UPLOAD},folds=FOLDS,
  max_new_submissions=4,initial_remaining=73,reserve=69,
  authorization='Human on15Oct requests exactly four LB attempts combining GAIN, FLAT and LOCAL. Existing FL99.74 repeated with two new model seeds rather than resubmitting identical predictions.',
  baseline='Exact R9 P_MIX recipe: batch8,4 epochs,6000 balanced draws/epoch,AdamW.001/wd.01,cosine horizon14,original scaler,noise and global shifts retained;two members,mean logits,sigmoid(logit/5).',
  composition='R9 deterministic FLAT/LOCAL views reused in exact order, final RMS then GAIN with the same per-file +/-6dB values as R9. Combination chosen jointly withp=.5 using unchanged separate R9 extra RNG. No tempo or other new augmentation. Native audio/padding/architecture unchanged.',
  controls='GF/GL/GFL use original seeds639320752/753, so corresponding R9 single and FL comparisons are paired. FL_REPEAT uses754/755 with exactly old flat_local cache; compare to original FL as between-seed replication, not a paired ablation.',
  validation='All four on three previous fixed speaker/attack-disjoint folds,2members each:12 local ensembles/24 members. No split/parameter changes from public scores. Report local vs old singles/FL; public gain does not establish private or seed stability.',
  freeze='All four full models and repeat-verified CSVs frozen before reading any new public grade; ordinary authenticated browser forms; no confirmed/ambiguous POST replay; total<=4/live reserve69.',
  power='Own temporary caffeinate only, atomic optimizer/RNG/cursor checkpoints; signal saves,<=1%discharging saves and pauses.')
 p=OUT/'protocol.json'
 if p.exists():assert r.read(p)['protocol']==protocol
 else:save(p,dict(created_at=now(),protocol=protocol));log('R10_ALL','protocol_frozen',protocol=protocol)
 registry=r.read(JOURNAL)
 for a,cfg in ARMS.items():
  registry['ideas'].setdefault('R10_'+a,dict(title=cfg['title'],author='Кирилл',control=cfg['control'],status='protocol_frozen',change=json.dumps(cfg,ensure_ascii=False),notes='Four fixed public attempts; FL_REPEAT is a new-seed replication, all other combinations share original model seeds.'))
 registry['updated_at']=now();save(JOURNAL,registry);render(registry)
 if not (OUT/'previous_best.json').exists():save(OUT/'previous_best.json',r.read(ROOT/'models/current_best.json'))

def extract_combo(args):
 path,index,maximum,key=args
 # Recompute from waves because uniform log gain is not exact under the fixed
 # spectral floor; adding an offset to a cached map would be incorrect.
 import soundfile as sf
 x,sr=sf.read(path,dtype='float32');assert sr==16000 and x.ndim==1
 x=x[:maximum];x/=np.sqrt(np.mean(x*x))+1e-9
 if len(x)<1024:x=np.pad(x,(0,1024-len(x)))
 rng=np.random.default_rng(OLD_SEEDS[0]+3901+index);snr=rng.uniform(20,35)
 noisy=x+rng.normal(size=len(x)).astype(np.float32)*np.float32(10**(-snr/20));noisy/=np.sqrt(np.mean(noisy*noisy))+1e-9
 db=float(np.random.default_rng(OLD_SEEDS[0]+19015+index+4).uniform(-6,6));result={}
 for n,source in [(0,x),(1,noisy)]:
  y,details=previous.transformed(source,index,maximum,COMPONENT[key]);y*=np.float32(10**(db/20))
  left=int(np.random.default_rng(OLD_SEEDS[0]+4901+index).integers(0,(maximum-len(y))//160+1))*160
  for shifted,offset in [(0,0),(1,left)]:result[f'{key}_{n}{shifted}']=r.old.stft_map(np.pad(y,(offset,maximum-len(y)-offset))).astype(np.float16)
 return result,dict(**details,gain_db=db)

def build_cache(key):
 CACHE.mkdir(parents=True,exist_ok=True);rows=manifest(ROOT/'data');spec=r.read(ROOT/'data/cache/research_v2/complete.json');maximum=spec['max_samples']
 keys=[f'{key}_{n}{s}' for n in [0,1] for s in [0,1]]
 contract=dict(version=1,key=key,rows=len(rows),ids_sha256=hashlib.sha256('\n'.join(v[0] for v in rows).encode()).hexdigest(),maximum=maximum,keys=keys)
 done=CACHE/f'{key}_complete.json';progress=CACHE/f'{key}_progress.json'
 if done.exists():assert r.read(done)['contract']==contract;return
 arrays={k:np.lib.format.open_memmap(CACHE/f'{k}.npy',mode='r+' if (CACHE/f'{k}.npy').exists() else 'w+',dtype='float16',shape=(len(rows),spec['max_frames'],64)) for k in keys}
 state=r.read(progress,dict(next_index=0,contract=contract));assert state['contract']==contract
 with ProcessPoolExecutor(6) as pool:
  for start in range(state['next_index'],len(rows),128):
   r.guard();batch=list(pool.map(extract_combo,[(rows[i][1],i,maximum,key) for i in range(start,min(start+128,len(rows)))],chunksize=8))
   for i,(views,details) in enumerate(batch,start):
    for k in keys:arrays[k][i]=views[k]
   for v in arrays.values():v.flush()
   state.update(next_index=start+len(batch),time=now());save(progress,state);print('CACHE',key,state['next_index'],'/',len(rows),flush=True)
 save(done,dict(contract=contract,complete=True,time=now()))

def refit(d,a):
 dest=FINAL/a;dest.mkdir(parents=True,exist_ok=True);cfg=ARMS[a]
 if r.read(dest/'ensemble.json',{}).get('complete'):return
 sc=joblib.load(previous.SOURCE/'normalizer.joblib');joblib.dump(sc,dest/'normalizer.joblib');members=[]
 for seed in seeds(a):
  model,h=r.fit(d,np.arange(len(d.ids)),cfg,sc,seed,dest);members.append(dict(seed=seed,file=f'seed_{seed}.pt',history=h));del model;r.clean()
 save(dest/'ensemble.json',dict(engine='research_v7',study='research_v10',arm=a,cfg=cfg,feature_config=d.spec,seeds=seeds(a),members=members,temperature=5.,complete=True))

def queue(a,public):
 q=r.read(OUT/'upload_manifest.json',{})
 if a in q:return
 path=ROOT/f'submissions/research_r10/{a}.csv';item=r.infer(FINAL/a,ROOT/'data/public_test',path,ROOT/'data/public_sample_submission.csv',public)
 repeat=path.with_suffix('.repeat.csv');r.infer(FINAL/a,ROOT/'data/public_test',repeat,ROOT/'data/public_sample_submission.csv',public);assert r.digest(repeat)==item['sha256']
 item.update(verified_byte_identical=True,frozen_at=now(),frozen_files={str(p.relative_to(ROOT)):r.digest(p) for p in (FINAL/a).iterdir() if p.name in ['ensemble.json','normalizer.joblib'] or p.name.startswith('seed_')})
 q[a]=item;save(OUT/'upload_manifest.json',q);update('R10_'+a,status='verified_ready',final_models=str(FINAL/a));print('READY',a,flush=True)

def summarize(result):
 old=r.read(ROOT/'reports/research_v9/local_results.json');summary={}
 for a in UPLOAD:
  values={fid:f['arms'][a]['metrics']['unseen_attack_auc'] for fid,f in result['folds'].items() if a in f['arms']}
  if not values:continue
  control=ARMS[a]['control'][3:];deltas={fid:v-old['folds'][fid]['arms'][control]['metrics']['unseen_attack_auc'] for fid,v in values.items()}
  summary[a]=dict(n_folds=len(values),mean_unseen_auc=float(np.mean(list(values.values()))),worst_unseen_auc=min(values.values()),unseen_auc_by_fold=values,control=ARMS[a]['control'],paired_deltas=deltas,paired_mean_delta=float(np.mean(list(deltas.values()))),paired_seeds=a!='FL_REPEAT')
  update('R10_'+a,research_v2=summary[a])
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
   for seed in seeds(a):
    pf=dest/f'predictions_{seed}.npz';receipt=dest/f'result_{seed}.json'
    if pf.exists() and receipt.exists():p=np.load(pf)['logits'];member=r.read(receipt)
    else:
     model,h=r.fit(d,tr,cfg,sc,seed,dest);p=r.predict(model,d,va,cfg,sc);np.savez(pf,idx=va,logits=p);member=dict(seed=seed,history=h,metrics=evaluate(d,va,p,definitions[fid]['heldout_attack']));save(receipt,member);del model;r.clean()
    ps.append(p);members.append(member)
   p=np.mean(ps,0);np.savez(dest/'predictions.npz',idx=va,logits=p);fold['arms'][a]=dict(metrics=evaluate(d,va,p,definitions[fid]['heldout_attack']),members=members,seeds=seeds(a))
   save(OUT/'local_results.json',result);summarize(result);print('LOCAL',fid,a,fold['arms'][a]['metrics']['unseen_attack_auc'],flush=True)

def run():
 freeze();owner=(OUT/'compute.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(OUT/'active_process.json',dict(pid=os.getpid(),started_at=now()))
 guard=subprocess.Popen(['/usr/bin/caffeinate','-is','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 save(OUT/'power_guard.json',dict(pid=guard.pid,owner_pid=os.getpid(),temporary=True,time=now()))
 def pause(signum,frame):raise KeyboardInterrupt('Requested checkpoint pause')
 signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGINT,pause)
 try:
  d=Data();ids=pd.read_csv(ROOT/'data/public_sample_submission.csv')['idx'].astype(str).tolist();public=r.old.InferenceData([ROOT/'data/public_test'/i for i in ids],ids,['base'])
  for a,cfg in ARMS.items():
   save(OUT/'pipeline_status.json',dict(complete=False,running=True,pid=os.getpid(),phase='final_cache_and_fit',arm=a,time=now()))
   if cfg['extra_aug'] in COMPONENT:build_cache(cfg['extra_aug'])
   refit(d,a);queue(a,public)
  save(OUT/'pipeline_status.json',dict(complete=False,running=True,pid=os.getpid(),final_candidates_complete=True,phase='local',time=now()))
  local(d);save(OUT/'pipeline_status.json',dict(complete=True,running=False,final_candidates_complete=True,finished_at=now()));print('R10 TRAINING COMPLETE',flush=True)
 except (KeyboardInterrupt,r.old.BatteryLow) as e:
  save(OUT/'pipeline_status.json',dict(complete=False,running=False,paused=True,reason=str(e),time=now()));raise
 except Exception as e:
  save(OUT/'pipeline_status.json',dict(complete=False,running=False,error=type(e).__name__+': '+str(e),time=now()));raise
 finally:
  guard.terminate();save(OUT/'power_release.json',dict(pid=guard.pid,released=True,time=now()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run']);args=p.parse_args();freeze() if args.command=='freeze' else run()
