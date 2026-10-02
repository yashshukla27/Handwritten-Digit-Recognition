"""Show predictions on held-out examples using the saved CNN."""
import numpy as np
import matplotlib.pyplot as plt
from train import CNN,prepare
from pathlib import Path
root=Path(__file__).resolve().parent
x,y,tr,va,te=prepare()
m=CNN(); m.p=dict(np.load(root/'results/CNN_weights.npz'))
ids=te[:12]; pred=m.forward(x[ids]).argmax(1)
fig,axs=plt.subplots(3,4,figsize=(8,6))
for ax,i,p in zip(axs.flat,ids,pred):
    ax.imshow(x[i],cmap='gray',vmin=0,vmax=1)
    ax.set_title(f'Actual {y[i]} | Predicted {p}'); ax.axis('off')
plt.tight_layout(); plt.savefig(root/'results/demo.png',dpi=160)
print('Saved results/demo.png'); plt.show()
