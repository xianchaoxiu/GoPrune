import os
import math
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


# =========================
# Set
# =========================
class Args:
    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 5
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4

    method: str = "pam"

    pam_beta: float = 1.5e-3
    pam_rho1: float = 1.5e-3
    pam_rho2: float = 1.5e-3
    pam_lambda: float = 5e-3
    pam_p: float = 2 / 3

    prune_ratio: float = 0.3
    sensitive_keep: float = 0.3

def build_dataloaders(args: Args) -> Tuple[DataLoader, DataLoader]:
    normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
    test_tf = T.Compose([T.ToTensor(), normalize])
    train_set = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, test_loader


# =========================
# Model
# =========================
def get_model() -> nn.Module:
    # ResNet18 on CIFAR-10
    model = torchvision.models.resnet18(num_classes=10)
    return model

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total, total_loss / total


# =========================
# PAM
# =========================
def _pam_collect_convs(model: nn.Module) -> List[Tuple[str, nn.Conv2d]]:
    return [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]


def _prox_l2p_norm(a: torch.Tensor, lam: float, p: float) -> torch.Tensor:
    eps = 1e-12
    a = torch.clamp(a, min=eps)

    if abs(p - 0.0) < 1e-6:
        thresh = math.sqrt(2 * lam)
        return torch.where(a > thresh, a, torch.zeros_like(a))

    elif abs(p - 0.5) < 1e-6:
        t = (3 * lam / 2) ** (2 / 3)
        mask = a > t
        res = torch.zeros_like(a)
        if mask.any():
            aa = a[mask]
            phi = torch.acos(torch.clamp(lam / 4 * (aa / 3).pow(-1.5), -1 + 1e-7, 1 - 1e-7))
            res[mask] = (2 / 3) * aa * (1 + torch.cos(2 * math.pi / 3 - 2 * phi / 3))
        return res

    elif abs(p - 2 / 3) < 1e-6:
        c = (2 * lam * (1 - p)) ** (1 / (2 - p))
        return torch.relu(a - c * (a ** (p - 1)))

    else:
        x = a.clone()
        for _ in range(3):
            x = torch.relu(a - lam * p * x.pow(p - 1))
        return x


def _pam_l2p_update_U(W, U, args):
    beta, rho2, lam, p = args.pam_beta, args.pam_rho2, args.pam_lambda, args.pam_p
    N = (beta * W + rho2 * U) / (beta + rho2)
    U_new = torch.zeros_like(U)
    Cout = W.size(0)
    for j in range(Cout):
        Nj = N[j].reshape(-1)
        normNj = Nj.norm(p=2)
        if normNj < 1e-12:
            U_new[j].zero_()
        else:
            coeff = _prox_l2p_norm(normNj, lam / (beta + rho2), p) / normNj
            U_new[j] = coeff * N[j]
    return U_new

# =========================
# GoPrune-PAM
# =========================
def compress_pam(train_loader, val_loader, model, device, args):
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    convs = _pam_collect_convs(model)
    U_list = [conv.weight.detach().clone() for _, conv in convs]
    W_prev = [conv.weight.detach().clone() for _, conv in convs]

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss, total_ce, correct, total = 0., 0., 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            ce_loss = F.cross_entropy(logits, y)
            reg = 0.0
            for (name, conv), U, Wk in zip(convs, U_list, W_prev):
                W = conv.weight
                reg += 0.5 * args.pam_beta * torch.norm(W - U, p=2) ** 2
                reg += 0.5 * args.pam_rho1 * torch.norm(W - Wk, p=2) ** 2
            loss = ce_loss + reg
            loss.backward()
            opt.step()

            with torch.no_grad():
                total_loss += loss.item() * x.size(0)
                total_ce += ce_loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

        with torch.no_grad():
            for i, (name, conv) in enumerate(convs):
                W_now = conv.weight.detach().clone()
                U_list[i] = _pam_l2p_update_U(W_now, U_list[i], args)
                W_prev[i] = W_now.clone()

        acc = correct / total
        val_acc, val_loss = evaluate(model, val_loader, device)
        print(f"[GoPrune-PAM][Epoch {epoch+1}/{args.epochs}] "
              f"TrainLoss={total_loss/total:.4f} (CE={total_ce/total:.4f}) "
              f"TrainAcc={acc*100:.2f}%  ValAcc={val_acc*100:.2f}%  "
              f"Time={time.time()-t0:.1f}s")

# =========================
# Prune
# =========================
def channel_prune_L1_global_in_main(model: nn.Module, ratio: float, sensitive_keep: float = 0.3):
    def iter_named_convs(m):
        for n, mod in m.named_modules():
            if isinstance(mod, nn.Conv2d):
                yield n, mod

    def channel_score_L1_out(conv: nn.Conv2d) -> torch.Tensor:
        with torch.no_grad():
            w = conv.weight.detach().abs()
            return w.sum(dim=(1, 2, 3))

    def is_sensitive(name: str) -> bool:
        low = name.lower()
        return any(k in low for k in ["conv1", "layer4", "downsample"])

    per_layer_scores, all_scores = {}, []
    for name, conv in iter_named_convs(model):
        s = channel_score_L1_out(conv)
        per_layer_scores[name] = s
        all_scores.append(s)
    if not all_scores:
        return
    all_scores = torch.cat(all_scores)
    total_channels = all_scores.numel()
    prune_num = int(total_channels * ratio)
    prune_num = max(0, min(prune_num, total_channels - 1))
    if prune_num == 0:
        return
    keep_num = total_channels - prune_num
    threshold = torch.topk(all_scores, k=keep_num, largest=True).values.min()

    layer_masks = {}
    for name, conv in iter_named_convs(model):
        s = per_layer_scores[name]
        keep = (s > threshold).to(torch.uint8)
        if is_sensitive(name):
            min_keep = max(1, int(math.ceil(keep.numel() * sensitive_keep)))
            if keep.sum().item() < min_keep:
                topk_idx = torch.topk(s, k=min_keep, largest=True).indices
                keep = torch.zeros_like(keep)
                keep[topk_idx] = 1
        if keep.sum().item() == 0:
            keep[s.argmax().item()] = 1
        layer_masks[name] = keep

    for name, conv in iter_named_convs(model):
        keep = layer_masks[name]
        mask_w = keep[:, None, None, None].float().expand_as(conv.weight)
        prune.CustomFromMask.apply(conv, name="weight", mask=mask_w)
        prune.remove(conv, "weight")

    print(f"[Prune] 全局阈值={float(threshold):.6f}，剪掉比例={ratio*100:.1f}%")

def main():
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = build_dataloaders(args)
    model = get_model().to(device)

    if args.method.lower() == "pam":
        print(">>> 使用 GoPrune-PAM 压缩 (ResNet18 on CIFAR-10, 5 epochs)")
        compress_pam(train_loader, test_loader, model, device, args)
    else:
        print(">>> 常规训练")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for epoch in range(args.epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                F.cross_entropy(logits, y).backward()
                opt.step()
            val_acc, _ = evaluate(model, test_loader, device)
            print(f"[TrainOnly][Epoch {epoch+1}] ValAcc={val_acc*100:.2f}%")

    if args.prune_ratio > 0:
        print("\n>>> 进行L1通道剪枝 (PAM后)")
        channel_prune_L1_global_in_main(model, ratio=args.prune_ratio, sensitive_keep=args.sensitive_keep)
        val_acc, val_loss = evaluate(model, test_loader, device)
        print(f"[After Prune] ValAcc={val_acc*100:.2f}%  ValLoss={val_loss:.4f}")

    print("全部流程完成。")


if __name__ == "__main__":
    main()
