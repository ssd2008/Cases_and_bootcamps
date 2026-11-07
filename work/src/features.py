"""Features inspired by adjacent-sample reconstruction and signed ratios.

All operations are label-free and per-file. No test-set normalization is fitted.
"""
import argparse
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import dct
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
FEATURE_VERSION = 1

def stats(v):
    v = np.asarray(v, dtype=np.float32)
    return np.array([v.mean(), v.std(), *np.quantile(v, [.01,.1,.25,.5,.75,.9,.99]),
                     np.mean(np.abs(v)), np.sqrt(np.mean(v*v))], dtype=np.float32)

def time_pool(v, n=96):
    """Average variable length frame sequences into a fixed number of tokens."""
    # Interpolation of cumulative sums avoids expensive Python loops.
    v = np.atleast_2d(v)
    t = np.linspace(0, v.shape[1], n+1)
    cs = np.pad(np.cumsum(v, axis=1, dtype=np.float64), ((0,0),(1,0)))
    i = np.minimum(t.astype(int), v.shape[1]-1)
    a = t-i
    out = cs[:,i]*(1-a) + cs[:,i+1]*a
    return np.diff(out, axis=1) / np.diff(t)

def extract(path):
    x,sr = sf.read(str(path), dtype='float32')
    return extract_samples(x, sr)

def extract_samples(x, sr):
    """Same feature transform for an already loaded, possibly augmented waveform."""
    if x.ndim > 1: x = x.mean(axis=1)
    if sr != 16000:
        g = np.gcd(sr,16000)
        x=signal.resample_poly(x,16000//g,sr//g).astype(np.float32)
    # Keep edges: silence itself can carry synthesis artifacts. Cap only unusually long clips.
    if len(x)>160000: x=x[:160000]
    if len(x)<1024: x=np.pad(x,(0,1024-len(x)))
    rms=np.sqrt(np.mean(x*x))+1e-9
    x=x/rms
    out=[]
    out.extend(stats(np.clip(x,-20,20)))
    frame_features=[]
    for lag in [1,2,4,8,16]:
        a,b,c=x[:-2*lag],x[lag:-lag],x[2*lag:]
        err=b-(a+c)/2
        ratio=(b-c)/(np.abs(a-c)+1e-9)
        ratio=np.clip(ratio,-100,100)
        signedlog=np.sign(ratio)*np.log1p(np.abs(ratio))
        delta=np.diff(signedlog)
        # Exact user scalar first within each lag, plus amplitude-normalized alternatives.
        out.extend([np.mean(np.abs(err)), np.mean(err*err)])
        for v in [err,np.abs(err),signedlog,delta,b-c]: out.extend(stats(v))
        # Fixed frame maps retain local fluctuations; pooled global statistics cannot.
        if lag in [1,4]:
            for v in [err, np.abs(err), signedlog, np.abs(delta), np.abs(b-c)]:
                frame_features.append(time_pool(v)[0])
    # LFCC and log-linear spectra retain high-frequency synthesis evidence.
    _,_,z=signal.stft(x,fs=16000,window='hann',nperseg=400,noverlap=240,nfft=512,boundary=None)
    p=np.abs(z)**2 + 1e-10
    bands=p[1:257].reshape(64,4,-1).mean(axis=1)
    logp=np.log(bands+1e-10)
    logp-=logp.mean()  # Per-file gain invariance; not normalization across files.
    ceps=dct(logp,type=2,axis=0,norm='ortho')[:40]
    for mat in [logp,ceps,np.diff(ceps,axis=1)]:
        out.extend(mat.mean(axis=1));out.extend(mat.std(axis=1))
        out.extend(np.quantile(mat,.1,axis=1));out.extend(np.quantile(mat,.9,axis=1))
    phase=np.angle(z)
    phase_acc=np.angle(np.exp(1j*np.diff(phase,n=2,axis=1)))
    for mat in [phase_acc]:
        b=mat[1:257].reshape(64,4,-1)
        out.extend(np.cos(b).mean(axis=(1,2)));out.extend(np.sin(b).mean(axis=(1,2)))
    seq=np.concatenate([time_pool(logp),np.asarray(frame_features)],axis=0).T
    out=np.nan_to_num(np.array(out,dtype=np.float32),nan=0,posinf=100,neginf=-100)
    return out,seq.astype(np.float32)

def manifest(data):
    rows=[]
    for line in (data/'train'/'targets.txt').read_text().splitlines():
        speaker,idx,_,attack,label=line.split()
        rows.append((idx,str(data/'train'/'flac'/f'{idx}.flac'),int(label=='spoof'),speaker,attack))
    return rows

def make_cache(data, dest, workers=10):
    rows=manifest(data)
    paths=[r[1] for r in rows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        feats=list(tqdm(pool.map(extract,paths,chunksize=32),total=len(paths),desc='train features'))
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'train_features.npz',x=np.stack([f[0] for f in feats]),
                        seq=np.stack([f[1] for f in feats]),y=np.array([r[2] for r in rows]),
                        ids=np.array([r[0] for r in rows]),speakers=np.array([r[3] for r in rows]),
                        attacks=np.array([r[4] for r in rows]),version=FEATURE_VERSION)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=ROOT/'data')
    ap.add_argument('--out',type=Path,default=ROOT/'data'/'cache');ap.add_argument('--workers',type=int,default=10)
    args=ap.parse_args();make_cache(args.data,args.out,args.workers)
