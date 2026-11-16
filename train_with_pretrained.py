"""Run source-domain meta-training and cross-domain adaptation from this backup."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import random
import warnings
import numpy as np
import torch
from RAFDB import RAF_train, RAF_test
from meta_yolo import Meta

PROJECT = Path(__file__).resolve().parent


def collate_4_diff_tasks(batch, dataset):
    n_way = random.choice([n for n in (3, 4, 5) if n <= dataset.cls_num])
    dataset.set_n_way(n_way)
    tasks = [dataset[i] for i in range(len(batch))]
    return (*[torch.stack([task[i] for task in tasks]) for i in range(4)],
            [n_way] * len(tasks))


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def evaluate(model, dataset, device):
    results, total_correct, total_count = [], defaultdict(int), defaultdict(int)
    for index in range(len(dataset)):
        sx, sy, qx, qy, mapping = dataset[index]
        trajectory, correct, count = model.finetunning(
            sx.to(device), sy.to(device), qx.to(device), qy.to(device), mapping)
        results.append(trajectory)
        for name, number in count.items():
            total_count[name] += number
            total_correct[name] += correct[name]
    values = np.stack(results)
    return {
        'episodes': len(dataset), 'n_way': dataset.n_way,
        'k_shot': dataset.k_shot, 'k_query': dataset.k_query,
        'accuracy_by_step': values.mean(0).tolist(),
        'episode_std_by_step': values.std(0, ddof=1).tolist() if len(values) > 1 else None,
        'per_class': {name: {'correct': total_correct[name], 'total': count,
                            'accuracy': total_correct[name] / count}
                      for name, count in sorted(total_count.items())},
        'macro_accuracy': float(np.mean([total_correct[name] / count
                                        for name, count in total_count.items()])),
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epoch', type=int, default=30000, help='Legacy name: total meta iterations')
    p.add_argument('--steps', type=int, default=None, help='Override number of meta iterations')
    p.add_argument('--n_way', type=int, default=5, help='Evaluation way; fixed training way with --fixed-way')
    p.add_argument('--fixed-way', action='store_true')
    p.add_argument('--n-min', type=int, default=3)
    p.add_argument('--n-max', type=int, default=5)
    p.add_argument('--k_spt', type=int, default=5)
    p.add_argument('--k_qry', type=int, default=15)
    p.add_argument('--imgsz', type=int, default=128)
    p.add_argument('--imgc', type=int, default=3, choices=[3])
    p.add_argument('--task_num', type=int, default=4)
    p.add_argument('--meta_lr', type=float, default=0.0005)
    p.add_argument('--update_lr', type=float, default=0.005)
    p.add_argument('--update_step', type=int, default=5)
    p.add_argument('--update_step_test', type=int, default=10)
    p.add_argument('--pretrained_path', default=None)
    p.add_argument('--freeze_backbone', action='store_true', default=True)
    p.add_argument('--no_freeze_backbone', dest='freeze_backbone', action='store_false')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--seed', type=int, default=222)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--train-root', default='FER-2013')
    p.add_argument('--train-split', default='train')
    p.add_argument('--eval-root', default='JAFFE_JPG')
    p.add_argument('--eval-split', default='test')
    p.add_argument('--eval-episodes', type=int, default=300)
    p.add_argument('--eval-every', type=int, default=180)
    p.add_argument('--log-every', type=int, default=30)
    p.add_argument('--save-every', type=int, default=180)
    p.add_argument('--run-dir', default=None)
    p.add_argument('--resume', default=None)
    return p


def main(args=None):
    args = parser().parse_args() if args is None else args
    steps = args.steps if args.steps is not None else args.epoch
    if min(steps, args.task_num, args.eval_episodes, args.log_every,
           args.eval_every, args.save_every, args.cpu_threads) < 1:
        raise ValueError('Step counts, episode counts and thread counts must be positive')
    if args.imgsz < 32 or args.imgsz % 32:
        raise ValueError('This YOLO/Mamba backbone requires image size divisible by 32')
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cuda_available = torch.cuda.is_available()
    if args.device == 'cuda' and not cuda_available:
        raise RuntimeError('CUDA is unavailable; choose --device cpu or auto')
    device = torch.device('cuda' if args.device == 'auto' and cuda_available else
                          ('cpu' if args.device == 'auto' else args.device))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    print(f'Python/PyTorch runtime: torch={torch.__version__}; device={device}', flush=True)
    if args.pretrained_path:
        args.pretrained_path = str(resolve_path(args.pretrained_path))
    if args.freeze_backbone and args.pretrained_path is None and args.resume is None:
        warnings.warn('Backbone is frozen without pretrained weights. This is only a runtime check; '
                      'use --pretrained_path or --no_freeze_backbone for training.')
    train = RAF_train(resolve_path(args.train_root), args.train_split, steps * args.task_num,
                      args.n_way, args.k_spt, args.k_qry, args.imgsz)
    test = RAF_test(resolve_path(args.eval_root), args.eval_split, args.eval_episodes,
                    args.n_way, args.k_spt, args.k_qry, args.imgsz)
    if not args.fixed_way and not 1 <= args.n_min <= args.n_max <= train.cls_num:
        raise ValueError('Invalid dynamic way range for the source dataset')
    model = Meta(args).to(device)
    run_dir = resolve_path(args.run_dir or ('runs/' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    events = run_dir / 'metrics.jsonl'
    start = 0
    if args.resume:
        checkpoint = torch.load(resolve_path(args.resume), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.meta_optim.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['step']
        if 'rng_state' in checkpoint:
            rng = checkpoint['rng_state']
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'].cpu())
            if device.type == 'cuda' and rng.get('cuda') is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])
    print(f'Train classes: {train.cls_names}; target classes: {test.cls_names}', flush=True)
    print(f'Run directory: {run_dir}', flush=True)

    def record(value):
        with events.open('a', encoding='utf-8') as output:
            output.write(json.dumps(value) + '\n')

    def save(step):
        torch.save({'model': model.state_dict(), 'optimizer': model.meta_optim.state_dict(),
                    'step': step, 'config': vars(args),
                    'rng_state': {'python': random.getstate(), 'numpy': np.random.get_state(),
                                  'torch': torch.get_rng_state(),
                                  'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}},
                   run_dir / 'last.pt')

    for step in range(start, steps):
        n_way = args.n_way if args.fixed_way else random.randint(args.n_min, args.n_max)
        train.set_n_way(n_way)
        tasks = [train[i] for i in range(args.task_num)]
        tensors = [torch.stack([task[i] for task in tasks]).to(device) for i in range(4)]
        accuracy, loss = model(*tensors, n_way)
        record({'type': 'train', 'step': step + 1, 'n_way': n_way, 'loss': loss.item(),
                'accuracy_by_step': accuracy.tolist(), 'grad_norm': model.last_grad_norm,
                'basis_grad_norm': model.last_basis_grad_norm})
        if step == start or (step + 1) % args.log_every == 0 or step + 1 == steps:
            print(f'Step {step+1}/{steps}: {n_way}-way; loss={loss.item():.6f}; '
                  f'accuracy={accuracy.tolist()}; grad_norm={model.last_grad_norm:.6g}', flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == steps:
            result = evaluate(model, test, device)
            record({'type': 'evaluation', 'step': step + 1, **result})
            print('Evaluation: ' + json.dumps(result), flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == steps:
            save(step + 1)
    print(f'Completed; checkpoint={run_dir / "last.pt"}', flush=True)


if __name__ == '__main__':
    main()
