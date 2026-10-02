"""Check analytical backpropagation against finite differences before training."""
import numpy as np
from train import MLP,CNN,loss_grad
for cls in [MLP,CNN]:
    model=cls(); rng=np.random.default_rng(23)
    x=rng.random((4,8,8)); y=np.array([1,3,5,7])
    _,d=loss_grad(model.forward(x),y); grads=model.backward(d)
    maximum=0
    for k,p in model.p.items():
        for _ in range(5):
            ix=tuple(rng.integers(n) for n in p.shape); old=p[ix]; eps=1e-5
            p[ix]=old+eps; hi=loss_grad(model.forward(x),y)[0]
            p[ix]=old-eps; lo=loss_grad(model.forward(x),y)[0]; p[ix]=old
            err=abs((hi-lo)/(2*eps)-grads[k][ix]); maximum=max(maximum,err)
    assert maximum<1e-5,(cls.__name__,maximum)
    print(cls.__name__,'gradient check PASS; max absolute error',maximum)
