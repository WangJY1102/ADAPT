# Third-party notices

`nets/yolo.py` and `nets/darknet.py` are taken from the recovered experimental
backbone, including prior project modifications. They retain the Megvii copyright
header and YOLOX-derived components (Apache-2.0). The SS2D / cross-scan components
are based on VMamba (MIT). Full license texts are in `licenses/`:

- YOLOX: https://github.com/Megvii-BaseDetection/YOLOX
- VMamba: https://github.com/MzeroMiko/VMamba

`nets/selective_scan_torch.py` is the first-order PyTorch compatibility backend
used by the recovered experiments, avoiding a mandatory compiled scan extension.
The three backbone source files are copied unchanged from the frozen experiment
snapshot; SHA-256 values are in `backbone_provenance.json`. The release does not
claim authorship of YOLOX/Mamba architecture components.

The original MAML backup MIT notice (Jackie Loong, 2019) is retained in
`licenses/MAML-backup-MIT.txt`. 

