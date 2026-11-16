"""Create manifests locally from ImageFolder or original JAFFE TIFF images."""
import argparse
import csv
import re
from pathlib import Path
from data import EXTENSIONS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--format', choices=['imagefolder', 'jaffe'], required=True)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    a.root = a.root.resolve()
    rows = []
    mapping = dict(AN='anger', DI='disgust', FE='fear', HA='happiness', NE='neutral', SA='sadness', SU='surprise')
    for f in sorted(a.root.rglob('*')):
        if not f.is_file() or f.suffix.lower() not in EXTENSIONS:
            continue
        rel = f.relative_to(a.root)
        if a.format == 'jaffe':
            match = re.match(r'^([A-Za-z]+)\.([A-Z]{2})\d+', f.name)
            if not match or match[2] not in mapping:
                raise ValueError(f'Unrecognized JAFFE filename: {f.name}')
            label, group = mapping[match[2]], match[1]
        else:
            if len(rel.parts) < 2:
                raise ValueError('ImageFolder requires root/class/image.ext; point to FER train only')
            label, group = rel.parts[0], ''
        rows.append(dict(path=rel.as_posix(), label=label, group=group))
    if not rows:
        raise ValueError('No images found')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    with a.output.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['path','label','group']); w.writeheader(); w.writerows(rows)
    print(f'Wrote {len(rows)} records. Use --data-root {a.root} with this manifest.')


if __name__ == '__main__':
    main()
