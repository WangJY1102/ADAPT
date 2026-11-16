# ADAPT

Minimal source release for cross-domain few-shot facial-state recognition:
Mamba encoder, shared-direction cosine initialization (SB-CTA), dynamic
N-way first-order meta-training, and nested PCEP evaluation.


## Installation

Use a new environment in this repository; do not reuse another person's environment.
Tested environment: Python 3.10, PyTorch 2.5.1+cu118, torchvision 0.20.1+cu118.
For an NVIDIA CUDA 11.8 installation:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python smoke_test.py
python smoke_test.py --full --device cuda:0
```


## Data and privacy

- **FER-2013**: acquire the dataset independently from its authorized distributor
  (original challenge: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
  Convert the **training split only** to `FER/train/class_name/image.jpg`.
- **JAFFE**: obtain the original TIFF collection independently from
  https://zenodo.org/records/3451524 and follow its usage conditions.
- **FER-Students**: images and image-level annotations are not distributed here
  because of privacy restrictions. For academic access inquiries, email
  **wangjiayu4353@stu.sau.edu.cn**. Provide institutional affiliation, a description
  of the proposed research, and supporting documentation confirming its academic
  purpose. Requests are reviewed individually; any access granted is subject to
  applicable permissions, privacy requirements, and restrictions on redistribution.
  Do not submit private data or supporting documents through public GitHub issues.
  See [Data availability and access](DATA_AVAILABILITY.md) for the full policy.

Create manifests locally (paths are relative to `--data-root`):

```bash
python prepare_data.py --format imagefolder --root /path/to/FER/train --output manifests/fer_train.csv
python prepare_data.py --format jaffe --root /path/to/jaffe --output manifests/jaffe.csv
```


## Meta-training

```bash
python run.py train --manifest manifests/fer_train.csv --data-root /path/to/FER/train --pretrained checkpoints/fer_source.pt --group D1 --seed 222 --output runs/D1_222 --device cuda:0
```

Repeat with seeds 333 and 444 and different output directories. Independent runs
can use `cuda:0` and `cuda:1`; this CLI does not implement distributed training.
The default is 1,000 meta-iterations, four tasks per iteration, 5 support / 15 query
images per class, five support updates, encoder lr=.001, classifier lr=.1, outer lr=.0001, fixed cosine scale10, no bias, RGB128 with ImageNet normalization.
BN running statistics are frozen, encoder affine parameters remain trainable.
Source images receive horizontal flips with probability .5; target images do not.

| Group | Source task size | Outer-learned shared direction |
|---|---|---|
| C0 | Fixed5-way | No |
| C1 | Fixed5-way | Yes |
| D0 | Uniform3/4/5-way | No |
| D1 (ADAPT) | Uniform3/4/5-way | Yes |

One N is drawn per meta-batch and shared by its four tasks. The detached support
gradients implement the first-order approximation while preserving the explicit
basis-to-task-head path. Task heads are tensors, not newly wrapped Parameters.
Training records losses, basis gradient norms, settings, input hashes and periodic
checkpoints. Existing run outputs are not overwritten; automatic resume is not
part of this minimal CLI. Target results never select a checkpoint.

## PCEP evaluation

JAFFE uses30 nested families, six conditions (3/4/5-way x1/5-shot),15 queries/class:

```bash
python run.py make-bank --manifest manifests/jaffe.csv --data-root /path/to/jaffe --output manifests/jaffe_bank.json
python run.py evaluate --manifest manifests/jaffe.csv --data-root /path/to/jaffe --bank manifests/jaffe_bank.json --checkpoint runs/D1_222/last.pt --output results/jaffe_222 --device cuda:0
```

FER-Students partition A uses support V01 V02 and query V03 V04 V05 V06:

```bash
python run.py make-bank --manifest manifests/students.csv --data-root /path/to/students --support-groups V01 V02 --query-groups V03 V04 V05 V06 --bank-seed 20261028 --output manifests/students_A.json
```

Partition B uses support V03 V04 and query V01 V02 V05 V06; use bank seed20361028.
Evaluate each partition separately. Reuse the saved bank across all seeds/methods.
Each family orders classes and support examples at the maximum5-way5-shot size;
smaller conditions select prefixes and retain shared queries. Each task starts
from the same trained initialization, with no parameter carry-over. Scoring uses
all N class candidates even when evaluating only anchor or added queries.

Summary outputs contain overall/anchor/added accuracy, R_dyn relative to paired
3-way anchors, and G_shot relative to paired1-shot accuracy. A_5,5 is the reference
cross-domain criterion and A_5,1 is sparse-annotation accuracy. Single-seed files
average families; summarize **three different training seeds** as follows:

```bash
python run.py summarize --inputs results/jaffe_222/summary.json results/jaffe_333/summary.json results/jaffe_444/summary.json --output results/jaffe_3seeds.json
```


## Files and scope

`model.py`/`basis.py`: method; `nets/`: encoder and scan backend; `data.py` and
`prepare_data.py`: local image loading/manifests; `run.py`: train/PCEP/summarize;
`pretrain.py`: optional source initialization; `smoke_test.py`: synthetic checks.
`THIRD_PARTY_NOTICES.md` and `licenses/` for upstream attribution.
