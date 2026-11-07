"""Train-only feature cache for physical windows, spectral differences and phase."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import soundfile as sf
from scipy import signal
from tqdm import tqdm
from features import ROOT,time_pool,manifest

CACHE=ROOT/'data/cache/research_v2'
MAP_NAMES=['spectrum25','spectrum10','spectrum50','delta_time','delta_frequency','phase_cos','phase_sin','phase_weighted_cos','phase_weighted_sin']

def prepare_headers():
    dest=ROOT/'data/cache/train_headers.npz'
    if dest.exists():return dest
    rows=manifest(ROOT/'data')
    def header(row):
        info=sf.info(row[1]);return info.frames,info.samplerate
    with ThreadPoolExecutor(8) as pool:values=np.asarray(list(pool.map(header,rows)))
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez(dest,frames=values[:,0],sample_rates=values[:,1],ids=np.asarray([r[0] for r in rows]))
    return dest

def stft(x,window_ms=25):
    window=int(window_ms*16);nfft=512 if window<=512 else 1024
    _,_,z=signal.stft(x,fs=16000,window='hann',nperseg=window,noverlap=window-160,nfft=nfft,boundary=None,padded=True)
    p=np.abs(z)**2+1e-10
    # Same 64 uniformly spaced frequency bands, even when the FFT changes.
    band=p[1:nfft//2+1].reshape(64,nfft//128,-1).mean(1)
    logp=np.log(band+1e-10);logp-=logp.mean()
    return z,p,logp

def choose_starts(scores,window,k=6):
    n=len(scores);width=min(window,n);cs=np.r_[0,np.cumsum(scores,dtype=np.float64)]
    objective=cs[width:]-cs[:-width]
    order=np.argsort(-objective,kind='stable');chosen=[]
    for start in order:
        if all(abs(int(start)-old)>=max(1,width//2) for old in chosen):chosen.append(int(start))
        if len(chosen)==k:break
    while len(chosen)<k:chosen.append(chosen[-1] if chosen else 0)
    return np.asarray(chosen,dtype=np.int32)

def extract_one(args):
    path,max_samples,min_samples=args
    x,sr=sf.read(path,dtype='float32')
    if x.ndim>1:x=x.mean(axis=1)
    if sr!=16000:
        g=np.gcd(sr,16000);x=signal.resample_poly(x,16000//g,sr//g).astype(np.float32)
    # Training max is frozen before inference; longer incoming clips are capped per file.
    x=x[:max_samples]
    original=len(x);x=x/(np.sqrt(np.mean(x*x))+1e-9)
    if len(x)<1024:x=np.pad(x,(0,1024-len(x)))
    z,p,lp=stft(x);n=lp.shape[1]
    _,_,short=stft(x,10);_,_,long=stft(np.pad(x,(0,max(0,800-len(x)))),50)
    dt=np.diff(lp,axis=1,prepend=lp[:,:1]);df=np.diff(lp,axis=0,prepend=lp[:1])
    fft=512;bins=np.arange(1,257)[:,None];unit=z[1:257,1:]*z[1:257,:-1].conj()
    unit=unit/np.maximum(np.abs(unit),1e-12)*np.exp(-2j*np.pi*bins*160/fft)
    unit=np.concatenate([unit[:,:1],unit],axis=1)
    pc=unit.real.reshape(64,4,-1).mean(1);ps=unit.imag.reshape(64,4,-1).mean(1)
    weights=p[1:257]/np.maximum(p[1:257].max(axis=0,keepdims=True),1e-10)
    pwc=(unit.real*weights).reshape(64,4,-1).mean(1);pws=(unit.imag*weights).reshape(64,4,-1).mean(1)
    maps=np.stack([time_pool(a).T for a in [lp,short,long,dt,df,pc,ps,pwc,pws]]).astype(np.float16)
    energy=p[1:257].sum(0).astype(np.float32)
    flux=np.mean(np.abs(dt),axis=0).astype(np.float32)
    crop_frames=max(8,1+int(np.ceil((min_samples-400)/160)))
    starts=np.stack([choose_starts(energy,crop_frames),choose_starts(flux,crop_frames)])
    absdf=np.abs(df[1:]);stats=np.asarray([absdf.max(),absdf.mean(),np.median(absdf)],dtype=np.float32)
    # Actual zero-waveform padding changes all-file centering; retain the exact padded spectrum.
    padded=np.pad(x,(0,max_samples-len(x)));_,_,padlp=stft(padded)
    return lp.T.astype(np.float16),maps,energy,flux,starts,stats,padlp.T.astype(np.float16),original

def build():
    CACHE.mkdir(parents=True,exist_ok=True)
    if (CACHE/'complete.json').exists():print('cache already complete');return
    rows=manifest(ROOT/'data');h=np.load(prepare_headers());assert np.array_equal(h['ids'],[r[0] for r in rows])
    max_samples=int(h['frames'].max());min_samples=int(h['frames'].min());n=len(rows)
    max_frames=1+int(np.ceil((max_samples-400)/160));crop_frames=1+int(np.ceil((min_samples-400)/160))
    shape=(n,max_frames,64)
    full=np.lib.format.open_memmap(CACHE/'full.npy',mode='w+',dtype='float16',shape=shape)
    padded=np.lib.format.open_memmap(CACHE/'padded.npy',mode='w+',dtype='float16',shape=shape)
    maps=np.lib.format.open_memmap(CACHE/'maps.npy',mode='w+',dtype='float16',shape=(n,len(MAP_NAMES),96,64))
    energy=np.lib.format.open_memmap(CACHE/'energy.npy',mode='w+',dtype='float32',shape=(n,max_frames))
    flux=np.lib.format.open_memmap(CACHE/'flux.npy',mode='w+',dtype='float32',shape=(n,max_frames))
    lengths=np.zeros(n,np.int32);starts=np.zeros((n,2,6),np.int32);stats=np.zeros((n,3),np.float32)
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs=((r[1],max_samples,min_samples) for r in rows)
        for i,(lp,mm,ee,ff,ss,st,pp,orig) in enumerate(tqdm(pool.map(extract_one,jobs,chunksize=16),total=n,desc='research train features')):
            nn=len(lp);lengths[i]=nn;full[i,:nn]=lp;padded[i]=pp;maps[i]=mm;energy[i,:nn]=ee;flux[i,:nn]=ff;starts[i]=ss;stats[i]=st
    for a in [full,padded,maps,energy,flux]:a.flush()
    np.savez(CACHE/'metadata.npz',ids=np.asarray([r[0] for r in rows]),lengths=lengths,starts=starts,frequency_stats=stats)
    config={'version':2,'map_names':MAP_NAMES,'max_samples':max_samples,'min_samples':min_samples,'max_frames':max_frames,'crop_frames':crop_frames,'hop_samples':160,'sample_rate':16000,'k_windows':6,'pause_relative_db':-30,'train_only':True}
    (CACHE/'complete.json').write_text(json.dumps(config,indent=2));print(config,flush=True)
if __name__=='__main__':build()
