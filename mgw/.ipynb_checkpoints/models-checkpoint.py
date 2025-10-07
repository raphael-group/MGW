import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class PhiModel(nn.Module):
    # Simple MLP: E (dim_e) -> Feature space (dim_f)
    def __init__(self, dim_e:int, dim_f:int, widths=(64,128,128,64)):
        super().__init__()
        layers = []
        in_dim = dim_e
        for w in widths:
            layers += [nn.Linear(in_dim, w), nn.Softplus()]
            in_dim = w
        layers += [nn.Linear(in_dim, dim_f)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_phi(model:PhiModel, x, y, lr=1e-3, niter=2000, print_every=100, device="cpu"):
    model = model.to(device)
    x = x.to(device); y = y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    for t in range(1, niter+1):
        pred = model(x)
        loss = mse(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t % print_every == 0:
            print(f"[train_phi] step={t} loss={loss.item():.6f}")
    return model

class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, m=64, sigma=2.0):
        super().__init__()
        B = torch.randn(in_dim, m) * sigma
        self.register_buffer('B', B)
    def forward(self, x):
        # x: (N, in_dim)
        xb = 2*np.pi * x @ self.B  # (N, m)
        return torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)  # (N, 2m)

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
    def forward(self, x):
        h = F.silu(self.fc1(x))
        h = self.fc2(h)
        return F.silu(x + h)

class PhiModelFFN(nn.Module):
    def __init__(self, dim_e, dim_f, width=256, depth=4, fourier_m=64, sigma=2.0, dropout=0.0):
        super().__init__()
        self.ff = FourierFeatures(dim_e, m=fourier_m, sigma=sigma)
        inp = 2*fourier_m
        self.inp = nn.Linear(inp, width)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.out = nn.Linear(width, dim_f)
    def forward(self, x):
        x = self.ff(x)
        x = F.silu(self.inp(x))
        for b in self.blocks:
            x = b(x)
            x = self.drop(x)
        return self.out(x)

def train_phi_pro(model, x_np, y_np, *, device='cuda' if torch.cuda.is_available() else 'cpu',
                  lr=3e-4, weight_decay=1e-4, batch_size=1024, steps=10000,
                  print_every=500, jac_reg=1e-4, ema_decay=0.995, val_split=0.1):
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    
    # split
    N = x.shape[0]
    idx = torch.randperm(N)
    n_val = max(1, int(val_split*N))
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    xtr, ytr = x[tr_idx], y[tr_idx]
    xva, yva = x[va_idx], y[va_idx]
    
    dl = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    ema_params = [p.clone().detach() for p in model.parameters()]
    huber = nn.HuberLoss(delta=1.0)
    
    def apply_ema():
        with torch.no_grad():
            for p, ep in zip(model.parameters(), ema_params):
                ep.mul_(ema_decay).add_(p.detach(), alpha=1-ema_decay)

    def swap_to_ema():
        # copy EMA weights into model (for eval)
        backup = [p.clone() for p in model.parameters()]
        with torch.no_grad():
            for p, ep in zip(model.parameters(), ema_params):
                p.copy_(ep)
        return backup
    
    best_val = float('inf'); best_state = None
    it = 0
    while it < steps:
        for xb, yb in dl:
            model.train()
            pred = model(xb)
            loss = huber(pred, yb)

            # Jacobian regularizer (small): ||J||_F^2 averaged on a tiny subsample
            if jac_reg > 0:
                xb_small = xb[::max(1, xb.shape[0]//64)].detach().requires_grad_(True)
                yh = model(xb_small)  # (k, dim_f)
                # sum over outputs to backprop once
                jac_pen = 0.0
                for d in range(yh.shape[1]):
                    g = torch.autograd.grad(yh[:, d].sum(), xb_small, create_graph=False, retain_graph=True)[0]
                    jac_pen = jac_pen + (g**2).sum(dim=1).mean()
                loss = loss + jac_reg * jac_pen

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); apply_ema()
            it += 1

            if it % print_every == 0:
                # eval with EMA weights
                backup = swap_to_ema()
                model.eval()
                with torch.no_grad():
                    val = huber(model(xva), yva).item()
                # restore current (pre-EMA) weights
                with torch.no_grad():
                    for p, b in zip(model.parameters(), backup):
                        p.copy_(b)
                print(f"[{it}/{steps}] train={loss.item():.5f}  val={val:.5f}  lr={sched.get_last_lr()[0]:.2e}")
                if val < best_val:
                    best_val = val
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if it >= steps:
                break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
    # set to EMA for inference
    with torch.no_grad():
        for p, ep in zip(model.parameters(), ema_params):
            p.copy_(ep)
    model.eval()
    return model
