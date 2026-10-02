"""Two end-to-end neural networks, implemented in NumPy.
AI assistance: ChatGPT generated the initial implementation; see START_HERE.txt.
Run: python train.py --epochs 40
"""
import argparse, copy, csv, json, platform, time
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class MLP:
    def __init__(self, seed=42):
        r=np.random.default_rng(seed)
        self.p={'w1':r.normal(0,np.sqrt(2/64),(64,64)), 'b1':np.zeros(64),
                'w2':r.normal(0,np.sqrt(2/64),(64,32)), 'b2':np.zeros(32),
                'w3':r.normal(0,np.sqrt(2/32),(32,10)), 'b3':np.zeros(10)}
    def forward(self,x):
        p=self.p; self.x=x.reshape(-1,64)
        self.h1=np.maximum(0,self.x@p['w1']+p['b1'])
        self.h2=np.maximum(0,self.h1@p['w2']+p['b2'])
        return self.h2@p['w3']+p['b3']
    def backward(self,d):
        p=self.p; g={'w3':self.h2.T@d,'b3':d.sum(0)}
        d=(d@p['w3'].T)*(self.h2>0)
        g.update(w2=self.h1.T@d,b2=d.sum(0))
        d=(d@p['w2'].T)*(self.h1>0)
        g.update(w1=self.x.T@d,b1=d.sum(0)); return g

class CNN:
    """Valid 3x3 convolution (8 filters), ReLU, flatten, dense32, dense10.
    Every convolution filter is trained by backpropagation, not fixed.
    """
    def __init__(self,seed=42):
        r=np.random.default_rng(seed)
        self.p={'k':r.normal(0,np.sqrt(2/9),(9,8)), 'bc':np.zeros(8),
                'w1':r.normal(0,np.sqrt(2/288),(288,32)), 'b1':np.zeros(32),
                'w2':r.normal(0,np.sqrt(2/32),(32,10)), 'b2':np.zeros(10)}
    def forward(self,x):
        p=self.p; n=len(x)
        self.patches=np.lib.stride_tricks.sliding_window_view(x,(3,3),axis=(1,2)).reshape(n,36,9)
        self.conv=np.maximum(0,self.patches@p['k']+p['bc'])
        self.flat=self.conv.reshape(n,288)
        self.h=np.maximum(0,self.flat@p['w1']+p['b1'])
        return self.h@p['w2']+p['b2']
    def backward(self,d):
        p=self.p; g={'w2':self.h.T@d,'b2':d.sum(0)}
        d=(d@p['w2'].T)*(self.h>0)
        g.update(w1=self.flat.T@d,b1=d.sum(0))
        d=(d@p['w1'].T).reshape(-1,36,8)*(self.conv>0)
        g.update(k=np.einsum('npi,npf->if',self.patches,d),bc=d.sum((0,1)))
        return g

def softmax(z):
    z=z-z.max(1,keepdims=True); p=np.exp(z); return p/p.sum(1,keepdims=True)

def loss_grad(z,y):
    p=softmax(z); loss=-np.log(p[np.arange(len(y)),y]+1e-12).mean()
    p[np.arange(len(y)),y]-=1
    return float(loss),p/len(y)

def prepare(seed=42):
    d=load_digits(); x=d.images.astype(np.float64)/16.0; y=d.target
    ids=np.arange(len(y))
    tr,rest=train_test_split(ids,test_size=.30,stratify=y,random_state=seed)
    va,te=train_test_split(rest,test_size=.50,stratify=y[rest],random_state=seed)
    assert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    return x,y,tr,va,te

def train(epochs=40,out='results',seed=42):
    out=Path(out); out.mkdir(exist_ok=True,parents=True)
    x,y,tr,va,te=prepare(seed)
    np.savez(out/'split_indices.npz',train=tr,validation=va,test=te)
    rows=[]; histories={}
    for name,cls in [('MLP',MLP),('CNN',CNN)]:
        model=cls(seed); rng=np.random.default_rng(seed)
        m={k:np.zeros_like(v) for k,v in model.p.items()}; v=copy.deepcopy(m)
        step=0; best=-1; best_epoch=0; hist=[]; start=time.perf_counter()
        for epoch in range(1,epochs+1):
            order=rng.permutation(tr); total=0
            for i in range(0,len(order),64):
                batch=order[i:i+64]
                loss,d=loss_grad(model.forward(x[batch]),y[batch]); g=model.backward(d)
                step+=1; total+=loss*len(batch)
                for k in model.p:
                    m[k]=.9*m[k]+.1*g[k]; v[k]=.999*v[k]+.001*g[k]**2
                    model.p[k]-=.001*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
            val=float(accuracy_score(y[va],model.forward(x[va]).argmax(1)))
            hist.append({'epoch':epoch,'train_loss':total/len(tr),'val_accuracy':val})
            if val>best:
                best=val; best_epoch=epoch; checkpoint=copy.deepcopy(model.p)
            if epoch%10==0: print(name,epoch,'validation',round(val,4),flush=True)
        seconds=time.perf_counter()-start; model.p=checkpoint
        pred=model.forward(x[te]).argmax(1)
        pr,re,f1,_=precision_recall_fscore_support(y[te],pred,average='macro',zero_division=0)
        row=dict(model=name,parameters=sum(a.size for a in model.p.values()),best_epoch=best_epoch,
                 validation_accuracy=best,test_accuracy=float(accuracy_score(y[te],pred)),
                 macro_precision=float(pr),macro_recall=float(re),macro_f1=float(f1),training_seconds=seconds)
        rows.append(row); histories[name]=hist
        np.savez(out/f'{name}_weights.npz',**model.p)
        np.savetxt(out/f'{name}_confusion.csv',confusion_matrix(y[te],pred),fmt='%d',delimiter=',')
        with (out/f'{name}_predictions.csv').open('w') as f:
            w=csv.writer(f); w.writerow(['dataset_index','true_digit','predicted_digit'])
            w.writerows(zip(te,y[te],pred))
        print(row,flush=True)
    payload={'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
             'seed':seed,'epochs':epochs,'batch_size':64,'learning_rate':.001,
             'split_sizes':{'train':len(tr),'validation':len(va),'test':len(te)},
             'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,
             'platform':platform.platform(),'results':rows}
    (out/'metrics.json').write_text(json.dumps(payload,indent=2))
    (out/'history.json').write_text(json.dumps(histories,indent=2))
    with (out/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    return payload

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--epochs',type=int,default=40)
    a.add_argument('--out',default='results'); a.add_argument('--seed',type=int,default=42)
    args=a.parse_args(); train(args.epochs,args.out,args.seed)
