import copy
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
torch.set_num_threads(8)

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)

def fit_scaler(x):
    return {'mean':x.mean(axis=tuple(range(x.ndim-1)),dtype=np.float64).astype(np.float32),
            'std':np.maximum(x.std(axis=tuple(range(x.ndim-1)),dtype=np.float64),1e-4).astype(np.float32)}

def normalize(x,scaler):
    return np.clip((x-scaler['mean'])/scaler['std'],-12,12).astype(np.float32)

class MLP(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.3),
            nn.Linear(256,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.25),nn.Linear(128,1))
    def forward(self,x):return self.net(x).squeeze(-1)

class MapCNN(nn.Module):
    """2-D feature image: rows are spectral/ratio channels, columns time."""
    def __init__(self,dim=74):
        super().__init__()
        self.convs=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.GELU(),nn.AvgPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.GELU(),nn.AvgPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU(),nn.AvgPool2d(2),
            nn.Conv2d(64,96,3,padding=1),nn.BatchNorm2d(96),nn.GELU())
        self.head=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Dropout(.3),nn.Linear(64,1))
    def forward(self,x):
        h=self.convs(x.transpose(1,2).unsqueeze(1))
        return self.head(torch.cat([h.mean((2,3)),h.std((2,3))],dim=1)).squeeze(-1)

class MaskedRestorer(nn.Module):
    def __init__(self,dim=74):
        super().__init__()
        self.input=nn.Linear(dim,64)
        self.mask_token=nn.Parameter(torch.zeros(64))
        p=torch.arange(96).float()[:,None]
        freq=torch.exp(torch.arange(0,64,2).float()*(-np.log(10000)/64))
        pe=torch.zeros(96,64);pe[:,0::2]=torch.sin(p*freq);pe[:,1::2]=torch.cos(p*freq)
        self.register_buffer('pe',pe)
        self.attention=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,192,.1,
            batch_first=True,norm_first=True,activation='gelu'),2,enable_nested_tensor=False)
        self.output=nn.Linear(64,dim)
    def forward(self,x,mask):
        h=self.input(x)
        h=torch.where(mask[:,:,None],self.mask_token[None,None,:],h)
        return self.output(self.attention(h+self.pe[None,:x.shape[1],:]))

@torch.no_grad()
def predict_logits(model,x,batch=256):
    model.eval();out=[]
    for i in range(0,len(x),batch):
        out.append(model(torch.from_numpy(x[i:i+batch]).to(DEVICE)).cpu().numpy())
    return np.concatenate(out)

def train_classifier(kind,x,y,train_idx,val_idx,seed,epochs=20,schedule_epochs=None):
    seed_everything(seed)
    model=(MapCNN(x.shape[-1]) if kind=='cnn' else MLP(x.shape[-1])).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,schedule_epochs or epochs)
    rng=np.random.default_rng(seed);best=-1;best_epoch=epochs;best_state=None
    history=[];batch=128 if kind=='cnn' else 256
    neg=train_idx[y[train_idx]==0];pos=train_idx[y[train_idx]==1]
    # A balanced epoch prevents the 9:1 class imbalance from dominating BCE.
    n_each=min(6000,max(len(neg),len(pos)))
    for ep in range(1,epochs+1):
        t=time.time();model.train()
        chosen=np.concatenate([rng.choice(neg,n_each,replace=True),rng.choice(pos,n_each,replace=True)])
        rng.shuffle(chosen);loss_sum=0
        for i in range(0,len(chosen),batch):
            ix=chosen[i:i+batch];xb=torch.from_numpy(x[ix]).to(DEVICE)
            if kind=='cnn':
                xb=xb+torch.randn_like(xb)*.03
                if rng.random()<.5:
                    start=int(rng.integers(0,xb.shape[1]-8));xb[:,start:start+8]=0
            logits=model(xb);yb=torch.from_numpy(y[ix].astype(np.float32)).to(DEVICE)
            loss=F.binary_cross_entropy_with_logits(logits,yb)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            loss_sum+=float(loss.detach().cpu())*len(ix)
        sched.step()
        auc=None
        if val_idx is not None:
            logits=predict_logits(model,x[val_idx]);auc=roc_auc_score(y[val_idx],logits)
            if auc>best:
                best=auc;best_epoch=ep;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        row={'epoch':ep,'loss':loss_sum/len(chosen),'auc':auc,'seconds':round(time.time()-t,2)}
        history.append(row);print(kind,seed,row,flush=True)
    if best_state is not None:model.load_state_dict(best_state)
    return model,best_epoch,history

def train_restorer(seq,y,train_idx,val_idx,seed,epochs=24):
    seed_everything(seed);model=MaskedRestorer(seq.shape[-1]).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    tr=train_idx[y[train_idx]==0];rng=np.random.default_rng(seed)
    history=[]
    for ep in range(1,epochs+1):
        model.train();rng.shuffle(tr);total=0;t=time.time()
        for i in range(0,len(tr),64):
            xb=torch.from_numpy(seq[tr[i:i+64]]).to(DEVICE)
            mask=torch.rand(xb.shape[:2],device=DEVICE)<.35
            pred=model(xb,mask)
            mse=((pred-xb)**2).mean(-1)
            cos=1-F.cosine_similarity(pred,xb,dim=-1)
            loss=((mse+.2*cos)*mask).sum()/mask.sum()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
            total+=float(loss.detach().cpu())*len(xb)
        history.append({'epoch':ep,'loss':total/len(tr),'seconds':round(time.time()-t,2)})
        print('restorer',seed,history[-1],flush=True)
    return model,history

@torch.no_grad()
def restoration_features(model,seq,batch=128):
    model.eval();out=[]
    # Four interleaved masks: each token is predicted without seeing its own value.
    for i in range(0,len(seq),batch):
        xb=torch.from_numpy(seq[i:i+batch]).to(DEVICE);pred=torch.zeros_like(xb)
        for k in range(4):
            m=(torch.arange(xb.shape[1],device=DEVICE)%4==k)[None,:].expand(len(xb),-1)
            estimate=model(xb,m);pred=torch.where(m[:,:,None],estimate,pred)
        diff=pred-xb;cos=1-F.cosine_similarity(pred,xb,dim=-1)
        f=torch.cat([diff.abs().mean(1),diff.square().mean(1),
                     cos.mean(1,keepdim=True),cos.std(1,keepdim=True)],dim=1)
        out.append(f.cpu().numpy())
    return np.concatenate(out)
