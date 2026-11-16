"""User-supplied CSV manifests; no dataset or private manifest is shipped."""
import csv
import random
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)), transforms.ToTensor(),
    transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def read_manifest(filename, root=None):
    filename = Path(filename)
    root = Path(root) if root else filename.parent
    with filename.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or not {'path', 'label'} <= rows[0].keys():
        raise ValueError('CSV must contain path,label columns and at least one row')
    seen = set()
    for row in rows:
        if not row['label'].strip():
            raise ValueError('Empty label')
        p = Path(row['path'])
        p = p if p.is_absolute() else root / p
        p = p.resolve()
        if not p.is_file() or str(p).casefold() in seen:
            raise ValueError(f'Missing or duplicate image: {p}')
        seen.add(str(p).casefold())
        row['_path'] = str(p)
        row['group'] = row.get('group', '').strip()
    return rows


def pools(rows):
    return {c: [r for r in rows if r['label'] == c] for c in sorted({r['label'] for r in rows})}


def tensors(support, query, classes, device, rng=None):
    labels = {c: i for i, c in enumerate(classes)}
    def batch(rows):
        xs = []
        for row in rows:
            with Image.open(row['_path']) as im:
                x = TRANSFORM(im.convert('RGB'))
            if rng is not None and rng.random() < .5:
                x = x.flip(-1)
            xs.append(x)
        return torch.stack(xs).to(device), torch.tensor([labels[r['label']] for r in rows], device=device)
    sx, sy = batch(support)
    qx, qy = batch(query)
    return sx, sy, qx, qy


def source_task(pool, seed, step, task, n, k=5, q=15):
    rng = random.Random(seed * 10000019 + step * 101 + task)
    classes = rng.sample(sorted(pool), len(pool))[:n]
    if len(classes) != n:
        raise ValueError('Not enough source classes')
    support, query = [], []
    for c in classes:
        chosen = rng.sample(pool[c], k+q)
        support.extend(chosen[:k]); query.extend(chosen[k:])
    return classes, support, query, rng
