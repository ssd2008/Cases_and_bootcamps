"""Portable frozen inference for individual models and research score mixtures."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch,joblib
from scipy.special import expit
from features import ROOT
import networks
import research_runner as runner
import research_v4 as r4
import research_v7 as r7
from research_final import InferenceData


def resolve(path):
    p=Path(path);return p if p.is_absolute() else ROOT/p


def device(name):
    if name=='auto':name='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    d=torch.device(name)
    for module in [networks,runner,r4,r7]:module.DEVICE=d
    return d


def predict(input_dir,sample,recipe_path,output,batch_size=32,device_name='auto'):
    recipe=json.loads(resolve(recipe_path).read_text());components=recipe['components']
    if sample:
        with Path(sample).open() as f:ids=[r['idx'] for r in csv.DictReader(f)]
    else:ids=sorted([p.name for p in Path(input_dir).iterdir() if p.suffix.lower() in ['.wav','.flac']],key=lambda x:(0,int(Path(x).stem)) if Path(x).stem.isdigit() else (1,x))
    assert ids and len(ids)==len(set(ids)) and all(Path(i).name==i and (Path(input_dir)/i).is_file() for i in ids)
    assert abs(sum(c['weight'] for c in components)-1)<1e-12 and all(c['weight']>0 for c in components)
    selected = device(device_name)
    default_spec = json.loads((ROOT/'config/research_v2/complete.json').read_text())
    metadata = {
        component['model_dir']: json.loads(
            (resolve(component['model_dir'])/'ensemble.json').read_text()
        )
        for component in components
    }
    spec = next(iter(metadata.values())).get('feature_config', default_spec)
    if any(meta.get('feature_config', default_spec) != spec for meta in metadata.values()):
        raise ValueError('All ensemble components must use the same feature configuration.')
    d=InferenceData([Path(input_dir)/i for i in ids],ids,spec,False)
    shared=r4.InferenceData.__new__(r4.InferenceData);shared.ids=np.asarray(ids);shared.spec={'max_samples':spec['max_samples']}
    shared.views={'base':d.padded,'full':d.full};shared.duration=d.duration;shared.meta={'lengths':d.meta['lengths']};shared.y=None
    ix=np.arange(len(ids));base_scores={};parts={};source_hashes={};raw={}
    for component in components:
        key=component['model_dir'];folder=resolve(key)
        if key not in base_scores:
            meta=metadata[key];assert meta.get('complete')
            engine=meta.get('engine');model_type=r7.Detector if engine=='research_v7' else r4.Detector if engine=='research_v4' else runner.ResearchCNN
            scaler_name='normalizer.joblib' if engine in ['research_v7','research_v4'] else 'scalers.joblib'
            scaler=joblib.load(folder/scaler_name);members=[]
            for file in [folder/'ensemble.json',folder/scaler_name,*[folder/m['file'] for m in meta['members']]]:
                source_hashes[str(file.relative_to(ROOT)) if file.is_relative_to(ROOT) else str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
            for member in meta['members']:
                model=model_type(meta['cfg']).to(selected);model.load_state_dict(torch.load(folder/member['file'],map_location='cpu',weights_only=True));model.eval()
                if engine in ['research_v7','research_v4']:
                    output_parts=[]
                    with torch.inference_mode():
                        for at in range(0,len(ids),batch_size):
                            x,aux=(r7.inputs if engine=='research_v7' else r4.inputs)(shared,ix[at:at+batch_size],meta['cfg'],scaler)
                            output_parts.append(model(x,aux).cpu().numpy())
                    p=np.concatenate(output_parts).astype(np.float64)
                else:
                    top=component.get('window_aggregation')=='top2'
                    p,windows,_=runner.predict(model,d,ix,meta['cfg'],scaler,with_windows=top)
                    if top:p=np.sort(windows,axis=1)[:,-2:].mean(axis=1)
                members.append(p);del model
                if selected.type=='mps':torch.mps.empty_cache()
            # Keep original legacy dtype through its mean/sigmoid.
            logits=np.mean(members,axis=0);base_scores[key]=expit(logits/meta.get('temperature',5.))
            raw[key]=np.asarray(members)
        score=base_scores[key]
        if component.get('transform')=='clip_f100':score=np.clip((score-.1)/.8,0,1)
        elif component.get('transform','identity')!='identity':raise ValueError('Unknown score transform')
        parts[component['id']]=score
    # Recipes preserve mean-of-components arithmetic for historical nine/pair.
    if recipe.get('hierarchy'):
        h=recipe['hierarchy'];nine=np.mean([parts[k] for k in h['nine']],axis=0);pair=np.mean([parts[k] for k in h['pair']],axis=0)
        score=h['nine_weight']*nine+(1-h['nine_weight'])*pair
    elif recipe.get('equal_mean'):score=np.mean([parts[c['id']] for c in components],axis=0)
    else:score=sum(c['weight']*parts[c['id']] for c in components)
    assert np.isfinite(score).all() and ((score>=0)&(score<=1)).all()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['idx','score']);w.writerows(zip(ids,[format(x,'.17g') for x in score]))
    np.savez_compressed(output.with_suffix('.npz'),idx=np.asarray(ids),score=score,**{k.replace('/','__'):v for k,v in parts.items()})
    receipt=dict(recipe=recipe['name'],research_only=recipe.get('research_only',False),rows=len(ids),device=str(selected),batch_size=batch_size,
        output=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),model_sources=source_hashes,inference_only=True,training_runs=0,network_requests=0)
    output.with_suffix('.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False,indent=2));return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--sample',type=Path)
    p.add_argument('--recipe',default='recipes/official_best.json');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    a=p.parse_args();assert a.batch_size>0;predict(a.input,a.sample,a.recipe,a.output,a.batch_size,a.device)
