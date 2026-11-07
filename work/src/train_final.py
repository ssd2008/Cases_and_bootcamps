"""Train one selected final recipe on provided train; no evaluation or uploads."""
import argparse,json,signal,copy
from pathlib import Path
import numpy as np
import joblib
from features import ROOT,make_cache
from research_features import prepare_headers,build
import research_v4 as r4
import research_v7 as r7
import research_v9 as r9
import research_v10 as r10
from predict_ensemble import device

TEMPLATES={'FL_REPEAT':'research_final_r10/FL_REPEAT','GL':'research_final_r10/GL','GFL':'research_final_r10/GFL',
 'GAIN':'research_final_r9/GAIN','FLAT':'research_final_r9/FLAT','LOCAL':'research_final_r9/LOCAL','PITCH':'research_final_r9/PITCH',
 'P_MIX':'research_final_r7/P_MIX','F_E008':'research_final_r7/F_E008','F100':'research_final_r4/F100','A002':'research_final_r4/A002'}


def main(a):
    train=a.train.resolve();out=a.out.resolve()
    assert (train/'targets.txt').is_file() and (train/'flac').is_dir()
    if not out.is_relative_to(ROOT/'models/retrained'):raise ValueError('--out must be under models/retrained; bundled models stay immutable')
    (ROOT/'data').mkdir(exist_ok=True);link=ROOT/'data/train'
    if link.exists():assert link.resolve()==train
    else:link.symlink_to(train,target_is_directory=True)
    template=json.loads((ROOT/'models'/TEMPLATES[a.recipe]/'ensemble.json').read_text());cfg=copy.deepcopy(template['cfg']);seeds=template['seeds']
    draws_per_class = getattr(a, 'draws_per_class', 3000)
    if (out/'ensemble.json').exists():
        existing=json.loads((out/'ensemble.json').read_text());assert existing['cfg']==cfg and existing['seeds']==seeds
        if existing.get('draws_per_class', 3000) != draws_per_class:
            raise ValueError('Use a different --out when changing --draws-per-class.')
        if existing.get('complete'):print('Already complete:',out);return
    selected=device(a.device);r9.DEVICE=selected
    if not (ROOT/'data/cache/train_features.npz').exists():make_cache(ROOT/'data',ROOT/'data/cache',a.workers)
    # A copied complete.json alone must not pretend that large feature arrays exist.
    v2=ROOT/'data/cache/research_v2'
    if (v2/'complete.json').exists() and not (v2/'full.npy').exists():
        raise RuntimeError('data/cache/research_v2 has metadata without feature arrays. Keep shipped metadata in config/, not data/cache/.')
    prepare_headers();build();r4.build_cache();r7.build_cache()
    if cfg.get('extra_aug'):
        if cfg['extra_aug'] in r10.COMPONENT:r10.build_cache(cfg['extra_aug'])
        else:r9.build_extra(cfg['extra_aug'])
    d=r10.Data();tr=np.arange(len(d.ids));sc=d.scaler(tr,cfg)
    out.mkdir(parents=True,exist_ok=True);joblib.dump(sc,out/'normalizer.joblib')
    engine=r7 if template['engine']=='research_v7' else r4
    meta=dict(engine=template['engine'],arm=a.recipe,cfg=cfg,feature_config=d.spec,seeds=seeds,members=[],temperature=template['temperature'],complete=False,
              draws_per_class=draws_per_class, origin='Fresh training on provided train only; not an exact optimizer continuation of bundled weights')
    def stop(*_):raise KeyboardInterrupt('Checkpoint stop requested')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    (out/'ensemble.json').write_text(json.dumps(meta,indent=2)+'\n')
    for seed in seeds:
        model,history=engine.fit(d,tr,cfg,sc,seed,out,n_each=draws_per_class)
        meta['members'].append(dict(seed=seed,file=f'seed_{seed}.pt',history=history));del model;engine.clean()
        (out/'ensemble.json').write_text(json.dumps(meta,indent=2)+'\n')
    meta['complete']=True;(out/'ensemble.json').write_text(json.dumps(meta,indent=2)+'\n');print('Complete:',out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train',type=Path,required=True);p.add_argument('--recipe',choices=list(TEMPLATES),default='FL_REPEAT')
    p.add_argument('--out',type=Path,default=ROOT/'models/retrained/FL_REPEAT');p.add_argument('--workers',type=int,default=8)
    p.add_argument('--device',choices=['auto','mps','cpu','cuda'],default='auto');main(p.parse_args())
