"""Refit frozen research arms, then inference only from saved train-fitted scalers."""
import argparse,json,hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch,joblib
from scipy.special import expit
from research_runner import Data,ResearchCNN,CONFIGS,SEED,OUT,MODELS,save,fit,predict
from research_features import extract_one,CACHE
from features import ROOT,extract,make_cache
from networks import DEVICE
from experiment_journal import now,log

FINAL=ROOT/'models/research_final_v2'

def train_one(arm,seeds=(SEED,SEED+1),data=None,cfg=None):
 cfg=cfg if cfg is not None else CONFIGS[arm]
 folder=FINAL/arm;meta_path=folder/'ensemble.json'
 if meta_path.exists():
  meta=json.loads(meta_path.read_text())
  if meta.get('complete'):
   assert meta['cfg']==cfg and meta['seeds']==list(seeds),'Saved refit differs from requested configuration'
   return folder
  raise RuntimeError('Inspect incomplete refit before reusing its path')
 d=data or Data();tr=np.arange(len(d.y));shared=FINAL/'train_scalers.joblib'
 FINAL.mkdir(parents=True,exist_ok=True)
 if shared.exists():sc=joblib.load(shared)
 else:
  sc=d.scalers(tr);joblib.dump(sc,shared)
 folder.mkdir(parents=True,exist_ok=True);joblib.dump(sc,folder/'scalers.joblib')
 meta={'arm':arm,'cfg':cfg,'feature_config':d.spec,'seeds':list(seeds),'members':[],'epochs':4,'n_each':3000,'temperature':5.,'aggregation':'mean_logits','complete':False,'created_at':now(),'selection':'frozen from train-only validation, no public fitting','inference_length_policy':'clip each incoming recording to provided training max 13.1879375 seconds; no test-derived parameters'}
 save(meta_path,meta)
 for seed in seeds:
  model,h=fit(d,tr,cfg,sc,seed);name=f'seed_{seed}.pt';torch.save(model.cpu().state_dict(),folder/name);meta['members'].append({'seed':seed,'file':name,'history':h});save(meta_path,meta);del model
  if DEVICE.type=='mps':torch.mps.empty_cache()
 meta['complete']=True;save(meta_path,meta);log('V2_'+arm,'full_train_refit_complete',models=str(folder));return folder

class InferenceData(Data):
 def __init__(self,paths,ids,spec,need_old=False):
  self.spec=spec;self.ids=np.asarray(ids)
  args=[(str(p),spec['max_samples'],spec['min_samples']) for p in paths]
  with ProcessPoolExecutor(8) as pool:rows=list(pool.map(extract_one,args,chunksize=16))
  self.maps=np.stack([r[1] for r in rows]);self.padded=np.stack([r[6] for r in rows]);self.full=np.zeros_like(self.padded)
  for i,row in enumerate(rows):self.full[i,:len(row[0])]=row[0]
  self.meta={'lengths':np.asarray([len(r[0]) for r in rows]),'starts':np.stack([r[4] for r in rows]),'frequency_stats':np.stack([r[5] for r in rows])}
  self.duration=np.asarray([r[7]/16000 for r in rows])
  if need_old:
   with ProcessPoolExecutor(8) as pool:self.old=np.stack([r[1][:,:64] for r in pool.map(extract,paths,chunksize=16)])

 def dump(self,path):
  savepath=Path(path);np.savez(savepath,maps=self.maps,padded=self.padded,full=self.full,lengths=self.meta['lengths'],starts=self.meta['starts'],frequency_stats=self.meta['frequency_stats'],duration=self.duration,ids=self.ids,old=self.old if hasattr(self,'old') else np.empty(0))

 @classmethod
 def load(cls,path,spec):
  x=np.load(path);d=cls.__new__(cls);d.spec=spec;d.ids=x['ids'];d.maps=x['maps'];d.full=x['full'];d.padded=x['padded'];d.duration=x['duration'];d.meta={k:x[k] for k in ['lengths','starts','frequency_stats']};d.old=x['old'];return d

def infer(folder,input_dir,output,sample=None,data=None):
 meta=json.loads((folder/'ensemble.json').read_text());assert meta['complete'];cfg=meta['cfg']
 ids=pd.read_csv(sample)['idx'].astype(str).tolist() if sample else sorted(p.name for p in input_dir.iterdir() if p.suffix.lower() in ['.wav','.flac'])
 assert ids and len(ids)==len(set(ids)) and all(Path(n).name==n and (input_dir/n).is_file() for n in ids)
 if data is None:data=InferenceData([input_dir/n for n in ids],ids,meta['feature_config'],cfg['source']=='old')
 assert np.array_equal(ids,data.ids)
 sc=joblib.load(folder/'scalers.joblib');ps=[]
 for member in meta['members']:
  model=ResearchCNN(cfg).to(DEVICE);model.load_state_dict(torch.load(folder/member['file'],map_location='cpu',weights_only=True));p,_,_=predict(model,data,np.arange(len(ids)),cfg,sc);ps.append(p);del model
 p=expit(np.mean(ps,0)/meta['temperature']);assert np.isfinite(p).all();output.parent.mkdir(parents=True,exist_ok=True)
 pd.DataFrame({'idx':ids,'score':p}).to_csv(output,index=False,float_format='%.17g')
 np.savez(output.with_suffix('.npz'),idx=ids,score=p,logits=np.mean(ps,0))
 receipt={'arm':meta['arm'],'file':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'rows':len(ids),'unique_scores':len(np.unique(p)),'models':str(folder),'inference_only':True}
 save(output.with_suffix('.json'),receipt);log('V2_'+meta['arm'],'inference_complete',**receipt);return receipt

if __name__=='__main__':
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
 tr=sub.add_parser('train');tr.add_argument('--arm',choices=list(CONFIGS),required=True)
 pr=sub.add_parser('predict');pr.add_argument('--models',type=Path,required=True);pr.add_argument('--input',type=Path,required=True);pr.add_argument('--output',type=Path,required=True);pr.add_argument('--sample',type=Path)
 a=ap.parse_args()
 if a.command=='train':train_one(a.arm)
 else:print(infer(a.models,a.input,a.output,a.sample))
