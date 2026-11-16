"""Source training and nested PCEP evaluation for the minimal ADAPT release."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from data import read_manifest, pools, tensors, source_task
from model import build, task_state, call, adapt


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_json(p, value):
    Path(p).write_text(json.dumps(value, indent=2), encoding='utf-8')


def event(p, value):
    with Path(p).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value)+'\n')


def runtime(device):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device(device)


def train(a):
    device = runtime(a.device)
    pool = pools(read_manifest(a.manifest, a.data_root))
    if len(pool) < 5 or min(map(len, pool.values())) < 20:
        raise ValueError('Source training requires >=5 classes and >=20 images per class')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'last.pt').exists() or (out/'train.jsonl').exists():
        raise FileExistsError('Use a new output directory; refusing to overwrite an existing run')
    model, basis, report = build(a.seed, device, a.pretrained, a.group.endswith('1'))
    opt = torch.optim.Adam([p for p in list(model.parameters())+list(basis.parameters()) if p.requires_grad], lr=1e-4)
    save_json(out/'config.json', dict(group=a.group, seed=a.seed, iterations=a.iterations,
        meta_batch=4, support=5, query=15, inner_steps=5, encoder_lr=.001, head_lr=.1,
        outer_lr=1e-4, scale=10, image_size=128, first_order=True,
        pretrained_sha256=digest(a.pretrained), manifest_sha256=digest(a.manifest),
        load_report=report, torch=torch.__version__))
    for step in range(1, a.iterations+1):
        n = random.Random(a.seed*137+step).choice([3,4,5]) if a.group.startswith('D') else 5
        opt.zero_grad(set_to_none=True)
        total, accuracy = 0., 0.
        for j in range(4):
            classes, s, q, rng = source_task(pool, a.seed, step, j, n)
            sx, sy, qx, qy = tensors(s,q,classes,device,rng)
            pp, bb = task_state(model,basis,n)
            pp = adapt(model,pp,bb,sx,sy)
            for first in range(0,len(qy),5):
                last = min(first+5,len(qy))
                logits = call(model,pp,bb,qx[first:last])
                loss = F.cross_entropy(logits,qy[first:last])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite query loss')
                weight = (last-first)/len(qy)/4
                (weight*loss).backward(retain_graph=last<len(qy))
                total += float(loss.detach())*weight
                accuracy += int((logits.detach().argmax(-1)==qy[first:last]).sum())/len(qy)/4
            del pp, bb, sx, sy, qx, qy, logits, loss
        parameters = list(model.parameters())+list(basis.parameters())
        if any(not torch.isfinite(p.grad).all() for p in parameters if p.grad is not None):
            raise FloatingPointError('Non-finite meta gradient')
        bg = 0. if basis.basis_raw.grad is None else float(basis.basis_raw.grad.norm())
        opt.step()
        row = dict(step=step,n=n,loss=total,accuracy=accuracy,basis_gradient_norm=bg)
        event(out/'train.jsonl',row)
        if step==1 or step%10==0:
            print(row,flush=True)
        if step%50==0 or step==a.iterations:
            torch.save(dict(model=model.state_dict(),basis=basis.state_dict(),optimizer=opt.state_dict(),
                step=step,seed=a.seed,group=a.group,torch_rng=torch.get_rng_state()),out/'last.pt')


def make_bank(a):
    rows = read_manifest(a.manifest, a.data_root)
    groups_s, groups_q = set(a.support_groups or []), set(a.query_groups or [])
    if bool(groups_s) != bool(groups_q) or groups_s & groups_q:
        raise ValueError('Provide both disjoint support/query group lists, or neither')
    source = pools([r for r in rows if r['group'] in groups_s] if groups_s else rows)
    query = pools([r for r in rows if r['group'] in groups_q] if groups_q else rows)
    if set(source) != set(query) or len(source)<5:
        raise ValueError('Both pools must contain the same >=5 classes')
    for c in source:
        if len(source[c]) < (5 if groups_s else 20) or len(query[c])<15:
            raise ValueError(f'Insufficient support/query pool for {c}')
    tasks = []
    for family in range(a.families):
        rng = random.Random(a.bank_seed+family)
        order = rng.sample(sorted(source),5)
        s,q = {},{}
        for c in order:
            if groups_s:
                s[c],q[c] = rng.sample(source[c],5),rng.sample(query[c],15)
            else:
                selected = rng.sample(source[c],20)
                s[c],q[c] = selected[:5],selected[5:]
        for n in [3,4,5]:
            for k in [1,5]:
                tasks.append(dict(family=family,n=n,k=k,q=15,classes=order[:n],
                    support=[r['path'] for c in order[:n] for r in s[c][:k]],
                    query=[r['path'] for c in order[:n] for r in q[c]]))
    out = Path(a.output)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    save_json(out,dict(manifest_sha256=digest(a.manifest),bank_seed=a.bank_seed,
        families=a.families,support_groups=sorted(groups_s),query_groups=sorted(groups_q),tasks=tasks))
    print(f'Saved {len(tasks)} tasks; reuse this file across methods/seeds.')


def evaluate(a):
    device = runtime(a.device)
    rows = read_manifest(a.manifest,a.data_root)
    lookup = {r['path']:r for r in rows}
    bank = json.loads(Path(a.bank).read_text(encoding='utf-8'))
    if bank['manifest_sha256'] != digest(a.manifest):
        raise ValueError('Task bank does not match this manifest')
    ck = torch.load(a.checkpoint,map_location=device,weights_only=True)
    model,basis,_ = build(ck['seed'],device,learn_basis=ck['group'].endswith('1'))
    model.load_state_dict(ck['model']); basis.load_state_dict(ck['basis'])
    out = Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'evaluation.jsonl').exists(): raise FileExistsError(out/'evaluation.jsonl')
    scored = []
    for task in bank['tasks']:
        s,q = [lookup[x] for x in task['support']],[lookup[x] for x in task['query']]
        if set(task['support']) & set(task['query']): raise ValueError('Image overlap')
        if bank['support_groups'] and {r['group'] for r in s}&{r['group'] for r in q}:
            raise ValueError('Video/group overlap')
        sx,sy,qx,qy = tensors(s,q,task['classes'],device)
        pp,bb = task_state(model,basis,task['n'])
        pp = adapt(model,pp,bb,sx,sy)
        with torch.no_grad():
            pred = torch.cat([call(model,pp,bb,x) for x in qx.split(5)]).argmax(-1)
            right,anchor = pred==qy,qy<3
            row = dict(family=task['family'],n=task['n'],k=task['k'],
                overall=float(right.float().mean()),anchor=float(right[anchor].float().mean()),
                added=float(right[~anchor].float().mean()) if (~anchor).any() else None)
        scored.append(row);event(out/'evaluation.jsonl',row)
        del pp,bb,sx,sy,qx,qy
        print(f"{len(scored)}/{len(bank['tasks'])}",flush=True)
    reference = {(r['family'],r['k']):r['anchor'] for r in scored if r['n']==3}
    one = {(r['family'],r['n']):r['overall'] for r in scored if r['k']==1}
    summary=[]
    for n in [3,4,5]:
        for k in [1,5]:
            part=[r for r in scored if (r['n'],r['k'])==(n,k)]
            summary.append(dict(n=n,k=k,accuracy=statistics.mean(r['overall'] for r in part),
                anchor=statistics.mean(r['anchor'] for r in part),
                added=statistics.mean(r['added'] for r in part) if n>3 else None,
                R_dyn=statistics.mean(r['anchor']-reference[r['family'],k] for r in part) if n>3 else None,
                G_shot=statistics.mean(r['overall']-one[r['family'],n] for r in part) if k>1 else None))
    save_json(out/'summary.json',dict(seed=ck['seed'],group=ck['group'],step=ck['step'],
        checkpoint_sha256=digest(a.checkpoint),bank_sha256=digest(a.bank),configurations=summary,
        units='fractions; multiply accuracy by100 for percent and differences by100 for pp'))


def summarize(a):
    records=[json.loads(Path(p).read_text(encoding='utf-8')) for p in a.inputs]
    if len({r['seed'] for r in records})!=len(records): raise ValueError('Duplicate seeds')
    if len({(r['group'],r['bank_sha256']) for r in records})!=1:
        raise ValueError('Use the same method and task bank for all seeds')
    result=[]
    for index in range(6):
        entries=[r['configurations'][index] for r in records]
        row={k:entries[0][k] for k in ['n','k']}
        for metric in ['accuracy','anchor','added','R_dyn','G_shot']:
            values=[e[metric]*100 for e in entries if e[metric] is not None]
            row[metric]=dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else None) if values else None
        result.append(row)
    save_json(a.output,dict(seeds=[r['seed'] for r in records],configurations=result,
        units='accuracy percent; R_dyn and G_shot percentage points'))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ['train','make-bank','evaluate']:
        s=sub.add_parser(name);s.add_argument('--manifest',required=True);s.add_argument('--data-root')
        s.add_argument('--output',required=True)
        if name!='make-bank':s.add_argument('--device',default='cuda:0')
        if name=='train':
            s.add_argument('--pretrained',required=True);s.add_argument('--seed',type=int,default=222)
            s.add_argument('--group',choices=['C0','C1','D0','D1'],default='D1')
            s.add_argument('--iterations',type=int,default=1000)
        elif name=='make-bank':
            s.add_argument('--support-groups',nargs='+');s.add_argument('--query-groups',nargs='+')
            s.add_argument('--families',type=int,default=30);s.add_argument('--bank-seed',type=int,default=20261028)
        else:
            s.add_argument('--checkpoint',required=True);s.add_argument('--bank',required=True)
    s=sub.add_parser('summarize');s.add_argument('--inputs',nargs='+',required=True);s.add_argument('--output',required=True)
    a=p.parse_args()
    if getattr(a,'iterations',1)<1 or getattr(a,'families',1)<1: p.error('Counts must be positive')
    {'train':train,'make-bank':make_bank,'evaluate':evaluate,'summarize':summarize}[a.command](a)


if __name__=='__main__':main()
