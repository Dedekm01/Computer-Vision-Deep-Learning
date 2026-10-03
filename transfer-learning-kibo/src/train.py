"""Transfer learning 3 model (MobileNetV3-Large, ResNet-18, ResNet-50) pada dataset crop.

Contoh:
    python src/train.py --data data/cls --out results/raw --seed 1 --epochs 8
    python src/train.py --models ResNet-18 --seed 1          # hanya satu model
Strategi (partial fine-tuning): bobot ImageNet; semua lapisan dibekukan KECUALI stage terakhir
backbone + classifier baru. BatchNorm pada lapisan beku tetap mode eval.
"""
import argparse, copy, glob, json, os, time
import numpy as np, torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from pytorchcv.model_provider import get_model
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

MODELS = {"MobileNetV3-Large": "mobilenetv3_large_w1", "ResNet-18": "resnet18", "ResNet-50": "resnet50"}
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/cls")
ap.add_argument("--out", default="results/raw")
ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--res", type=int, default=144, help="resize awal (px)")
ap.add_argument("--crop", type=int, default=128, help="ukuran input jaringan (px)")
ap.add_argument("--lr-backbone", type=float, default=3e-4)
ap.add_argument("--lr-head", type=float, default=2e-3)
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
classes = sorted(os.listdir(f"{a.data}/train")); C = len(classes)


def load(split):
    X, y = [], []
    for i, c in enumerate(classes):
        for f in sorted(glob.glob(f"{a.data}/{split}/{c}/*.png")):
            X.append(Image.open(f).convert("RGB").resize((a.res, a.res), Image.BILINEAR)); y.append(i)
    return X, y


norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tr_t = T.Compose([T.RandomCrop(a.crop), T.RandomRotation(10), T.ColorJitter(.3, .3, .3, .05), T.ToTensor(), norm])
ev_t = T.Compose([T.CenterCrop(a.crop), T.ToTensor(), norm])


class DS(Dataset):
    def __init__(s, X, y, t): s.X, s.y, s.t = X, y, t
    def __len__(s): return len(s.X)
    def __getitem__(s, i): return s.t(s.X[i]), s.y[i]


Xtr, ytr = load("train"); Xva, yva = load("valid"); Xte, yte = load("test")
print("train/valid/test:", len(Xtr), len(Xva), len(Xte), flush=True)
dl_tr = DataLoader(DS(Xtr, ytr, tr_t), a.batch_size, shuffle=True, num_workers=1, persistent_workers=True)
dl_va = DataLoader(DS(Xva, yva, ev_t), 64); dl_te = DataLoader(DS(Xte, yte, ev_t), 64)


def build(name):
    m = get_model(name, pretrained=True)
    m.features.final_pool = nn.AdaptiveAvgPool2d(1)  # pooling asli fixed 7x7 (untuk input 224)
    if name.startswith("mobilenet"):
        m.output.conv2 = nn.Conv2d(1280, C, 1); head = list(m.output.conv2.parameters())
        unfrozen = [m.features.stage5, m.features.final_block, m.output.conv1]
    else:
        m.output = nn.Linear(m.output.in_features, C); head = list(m.output.parameters())
        unfrozen = [m.features.stage4]
    for p in m.parameters(): p.requires_grad = False
    bb = []
    for mod in unfrozen:
        for p in mod.parameters(): p.requires_grad = True; bb.append(p)
    for p in head: p.requires_grad = True
    return m, bb, head


@torch.no_grad()
def evaluate(m, dl):
    m.eval(); L = 0; P = []; Y = []
    for x, y in dl:
        o = m(x).flatten(1); L += nn.functional.cross_entropy(o, y, reduction="sum").item()
        P += o.argmax(1).tolist(); Y += y.tolist()
    P, Y = np.array(P), np.array(Y)
    return L / len(Y), (P == Y).mean(), P, Y


@torch.no_grad()
def latency(m, n=60):
    """Median latensi inferensi: CPU, 1 thread, batch 1, input crop x crop."""
    m.eval(); x = torch.randn(1, 3, a.crop, a.crop); torch.set_num_threads(1)
    for _ in range(10): m(x)
    t = []
    for _ in range(n):
        s = time.perf_counter(); m(x); t.append((time.perf_counter() - s) * 1000)
    torch.set_num_threads(a.threads)
    return float(np.median(t))


for label in a.models:
    m, bb, head = build(MODELS[label])
    ntr = sum(p.numel() for p in m.parameters() if p.requires_grad); ntot = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr_backbone}, {"params": head, "lr": a.lr_head}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_backbone, a.lr_head],
                                              total_steps=a.epochs * len(dl_tr), pct_start=0.2)
    hist = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[]); best = (-1, None, 0); t0 = time.time()
    for ep in range(a.epochs):
        m.train()
        for mod in m.modules():  # BN pada lapisan beku tetap eval
            if isinstance(mod, nn.BatchNorm2d) and not any(p.requires_grad for p in mod.parameters()): mod.eval()
        L = A = N = 0
        for x, y in dl_tr:
            o = m(x).flatten(1); loss = nn.functional.cross_entropy(o, y, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            L += loss.item() * len(y); A += (o.argmax(1) == y).sum().item(); N += len(y)
        vl, va, _, _ = evaluate(m, dl_va)
        for k, v in zip(hist, [L / N, A / N, vl, va]): hist[k].append(float(v))
        if va > best[0]: best = (va, copy.deepcopy(m.state_dict()), ep + 1)  # checkpoint terbaik by val acc
        print(f"{label} ep{ep+1}/{a.epochs} tl={L/N:.3f} ta={A/N:.3f} vl={vl:.3f} va={va:.3f} t={time.time()-t0:.0f}s", flush=True)
    ttrain = time.time() - t0
    m.load_state_dict(best[1]); tl, ta, P, Y = evaluate(m, dl_te)
    r = dict(label=label, seed=a.seed, hist=hist, best_epoch=best[2], best_val_acc=float(best[0]), test_loss=float(tl),
             test_acc=float(ta), test_f1=float(f1_score(Y, P, average="macro")),
             cm=confusion_matrix(Y, P, labels=range(C)).tolist(), params_total_M=ntot / 1e6,
             params_trained_M=ntr / 1e6, latency_ms=latency(m), train_time_s=ttrain, classes=classes)
    print(label, "TEST acc", ta, "f1", r["test_f1"], "lat", r["latency_ms"], flush=True)
    json.dump(r, open(f"{a.out}/res_{label}_s{a.seed}.json", "w"))
