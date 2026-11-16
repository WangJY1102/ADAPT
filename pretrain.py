"""Optional source-only supervised pretraining; not a reconstruction of the lost historical run."""
import argparse
import csv
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from data import read_manifest, TRANSFORM
from model import Network


class Images(torch.utils.data.Dataset):
    def __init__(self, rows, labels):
        self.rows, self.labels = rows, labels

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        r=self.rows[index]
        with Image.open(r['_path']) as im: x=TRANSFORM(im.convert('RGB'))
        if random.random()<.5: x=x.flip(-1)
        return x,self.labels[r['label']]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--data-root')
    p.add_argument('--output',required=True,type=Path);p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seed',type=int,default=222);a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(4);torch.manual_seed(a.seed);random.seed(a.seed)
    rows=read_manifest(a.manifest,a.data_root)
    labels={c:i for i,c in enumerate(sorted({r['label'] for r in rows}))}
    dataset=Images(rows,labels)
    loader=torch.utils.data.DataLoader(dataset,batch_size=a.batch_size,shuffle=True,drop_last=True)
    if len(loader)==0:raise ValueError('Not enough source images for one full batch')
    model=Network(len(labels)).to(a.device).train();optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    for epoch in range(a.epochs):
        total=0.
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x.to(a.device)),y.to(a.device))
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
            loss.backward();optimizer.step();total+=float(loss.detach())
        print(dict(epoch=epoch+1,loss=total/len(loader)),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),classes=labels,seed=a.seed,epochs=a.epochs,
        description='New supervised source-only pretraining; not historical FER checkpoint'),a.output)


if __name__=='__main__':main()
