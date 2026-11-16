"""YOLO/Mamba encoder and first-order task-local adaptation."""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from nets.yolo import YOLOPAFPN, FeatureFusion
from basis import SharedCosineBasis, CosineClassifier


class Network(nn.Module):
    def __init__(self, classes=5):
        super().__init__()
        self.backbone = YOLOPAFPN(0.33, 0.50, depthwise=False)
        self.fusion_module = FeatureFusion()
        self.classifier = nn.Linear(224, classes)
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        return self.classifier(self.fusion_module(*self.backbone(x)))


def load_pretrained(model, filename):
    """Accept compatible source-supervised weights, never silently randomize."""
    ck = torch.load(filename, map_location='cpu', weights_only=True)
    state = ck.get('model', ck.get('state_dict', ck))
    own = model.state_dict()
    matched = {}
    for key, value in state.items():
        key = key.removeprefix('module.').removeprefix('net.')
        if key in own and hasattr(value, 'shape') and own[key].shape == value.shape:
            matched[key] = value
    missing_encoder = [k for k in own if not k.startswith('classifier.') and k not in matched]
    if missing_encoder:
        raise ValueError(f'Incomplete/incompatible encoder checkpoint: {missing_encoder[:8]}')
    model.load_state_dict(matched, strict=False)
    return {'matched_tensors': len(matched), 'ignored_tensors': len(state)-len(matched)}


def build(seed, device, pretrained=None, learn_basis=True):
    torch.manual_seed(seed)
    model = Network().to(device)
    report = load_pretrained(model, pretrained) if pretrained else None
    model.classifier = CosineClassifier(224, 5, 10).to(device)
    basis = SharedCosineBasis(224, 10, 20261024).to(device)
    basis.basis_raw.requires_grad_(learn_basis)
    model.eval()  # Freeze BN running statistics; encoder affine parameters remain trainable.
    return model, basis, report


def task_state(model, basis, n):
    params = OrderedDict(model.named_parameters())
    head = basis.initial_weight(n)
    if not head.requires_grad:
        head = head.detach().requires_grad_()
    params['classifier.weight'] = head  # Do not wrap in nn.Parameter or detach learned B.
    buffers = OrderedDict((k, v.clone()) for k, v in model.named_buffers())
    return params, buffers


def call(model, params, buffers, images):
    return torch.func.functional_call(model, (params, buffers), (images,))


def adapt(model, params, buffers, sx, sy, steps=5, encoder_lr=.001, head_lr=.1):
    for _ in range(steps):
        loss = F.cross_entropy(call(model, params, buffers, sx), sy)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite support loss')
        names = [k for k, p in params.items() if p.requires_grad]
        grads = torch.autograd.grad(loss, [params[k] for k in names], allow_unused=True)
        updated = OrderedDict(params)
        for name, grad in zip(names, grads):
            if grad is not None:
                lr = head_lr if name == 'classifier.weight' else encoder_lr
                updated[name] = params[name] - lr * grad.detach()
        params = updated
    return params
