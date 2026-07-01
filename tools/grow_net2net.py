"""Net2Net function-preserving growth: ChessCNN 20x256 (SE on blocks 10-19) → 20x320 (SE on ALL 20).
Net2WiderNet (Chen et al. 2015): replicate channels + divide the consuming layer's weights by the
replication count, so the wider net computes the SAME function. New SE blocks (0-9) are identity-init
(final-Linear weight=0, bias=+15 → sigmoid≈1 → per-channel scale 1). Verifies old(x)≈new(x) before
writing. Usage: python3 tools/grow_net2net.py <in.pth> <out.pth>
"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class Mish(nn.Module):
    def forward(self, x): return x * torch.tanh(F.softplus(x))
class SEBlock(nn.Module):
    def __init__(s, c, r=4, bias=False):
        super().__init__(); s.squeeze=nn.AdaptiveAvgPool2d(1)
        s.excitation=nn.Sequential(nn.Linear(c,c//r,bias=False),Mish(),nn.Linear(c//r,c,bias=bias),nn.Sigmoid())
    def forward(s,x):
        b,c,_,_=x.size(); y=s.squeeze(x).view(b,c); y=s.excitation(y).view(b,c,1,1); return x*y.expand_as(x)
class ResidualBlock(nn.Module):
    def __init__(s,c,use_se=False,se_bias=False):
        super().__init__()
        s.conv1=nn.Conv2d(c,c,3,padding=1,bias=False); s.bn1=nn.BatchNorm2d(c)
        s.conv2=nn.Conv2d(c,c,3,padding=1,bias=False); s.bn2=nn.BatchNorm2d(c)
        s.act=Mish(); s.se=SEBlock(c,bias=se_bias) if use_se else None
    def forward(s,x):
        r=x; o=s.act(s.bn1(s.conv1(x))); o=s.bn2(s.conv2(o))
        if s.se: o=s.se(o)
        o+=r; return s.act(o)
class ChessCNN(nn.Module):
    def __init__(s, filters=256, blocks=20, se_start=10, se_bias=False):
        super().__init__()
        s.input_conv=nn.Sequential(nn.Conv2d(120,filters,3,padding=1,bias=False),nn.BatchNorm2d(filters),Mish())
        s.res_blocks=nn.ModuleList([ResidualBlock(filters,use_se=(i>=se_start),se_bias=se_bias) for i in range(blocks)])
        s.policy_head=nn.Sequential(nn.Conv2d(filters,32,1,bias=False),nn.BatchNorm2d(32),Mish(),nn.Flatten(),nn.Linear(32*64,4672))
        s.value_head=nn.Sequential(nn.Conv2d(filters,1,1,bias=False),nn.BatchNorm2d(1),Mish(),nn.Flatten(),nn.Linear(64,256),Mish(),nn.Linear(256,3))
        s.material_head=nn.Sequential(nn.Conv2d(filters,1,1,bias=False),nn.BatchNorm2d(1),Mish(),nn.Flatten(),nn.Linear(64,1))
        s.plies_head=nn.Sequential(nn.Conv2d(filters,1,1,bias=False),nn.BatchNorm2d(1),Mish(),nn.Flatten(),nn.Linear(64,1))
        s.reply_head=nn.Sequential(nn.Conv2d(filters,32,1,bias=False),nn.BatchNorm2d(32),Mish(),nn.Flatten(),nn.Linear(32*64,4672))
    def forward(s,x,with_aux=False):
        x=s.input_conv(x)
        for b in s.res_blocks: x=b(x)
        p,v=s.policy_head(x),s.value_head(x)
        if not with_aux: return p,v
        return p,v,torch.tanh(s.material_head(x)).squeeze(-1),torch.sigmoid(s.plies_head(x)).squeeze(-1),s.reply_head(x)

NF_O, NF_N, SEH_O, SEH_N = 256, 320, 64, 80
rng = np.random.default_rng(0)
g   = list(range(NF_O)) + [int(rng.integers(0, NF_O)) for _ in range(NF_N-NF_O)]      # 320→256
rep = np.array([g.count(i) for i in range(NF_O)], dtype=np.float32)
gse = list(range(SEH_O)) + [int(rng.integers(0, SEH_O)) for _ in range(SEH_N-SEH_O)]  # 80→64
repse = np.array([gse.count(i) for i in range(SEH_O)], dtype=np.float32)
G, RIN = torch.tensor(g), torch.tensor([rep[g[j]] for j in range(NF_N)]).view(1,-1,1,1)  # per-new-in divisor
GSE, RINSE = torch.tensor(gse), torch.tensor([repse[gse[j]] for j in range(SEH_N)])

def conv_io(w):  return w[G][:, G] / RIN                                   # widen out(copy)+in(÷rep)
def conv_o(w):   return w[G]                                               # widen out only (input fixed)
def conv_in(w):  return w[:, G] / RIN                                      # widen in only (÷rep)  [O,I,1,1]
def bn(d):       return {k: v[G] for k, v in d.items()}                    # copy BN by g
def se_l1(w):    return w[GSE][:, G] / RIN.view(1,-1)                       # Linear (256→64)→(320→80)
def se_l2(w):    return w[G][:, GSE] / RINSE.view(1,-1)                     # Linear (64→256)→(80→320)

def grow(inp, outp):
    raw = torch.load(inp, map_location="cpu", weights_only=False)
    sd  = raw.get("model_state_dict", raw.get("state_dict", raw))
    sd  = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    old = ChessCNN(256,20,10,False)
    miss, unexp = old.load_state_dict(sd, strict=False)   # pre-aux ckpts lack aux heads → keep random
    if miss:   print(f"  (note: {len(miss)} missing keys — e.g. {miss[0]} — kept at init; unused for anchor)")
    if unexp:  print(f"  (note: {len(unexp)} unexpected keys ignored — e.g. {unexp[0]})")
    old.eval()
    new = ChessCNN(320,20,0,True);   new.eval()
    o, n = old.state_dict(), new.state_dict()

    out = {}
    out["input_conv.0.weight"] = conv_o(o["input_conv.0.weight"])
    for k in ("weight","bias","running_mean","running_var"):
        out[f"input_conv.1.{k}"] = o[f"input_conv.1.{k}"][G]
    out["input_conv.1.num_batches_tracked"] = o["input_conv.1.num_batches_tracked"]
    for i in range(20):
        p=f"res_blocks.{i}."
        out[p+"conv1.weight"]=conv_io(o[p+"conv1.weight"])
        out[p+"conv2.weight"]=conv_io(o[p+"conv2.weight"])
        for b_ in ("bn1","bn2"):
            for k in ("weight","bias","running_mean","running_var"): out[p+f"{b_}.{k}"]=o[p+f"{b_}.{k}"][G]
            out[p+f"{b_}.num_batches_tracked"]=o[p+f"{b_}.num_batches_tracked"]
        if i>=10:   # existing SE: widen; new bias=0
            out[p+"se.excitation.0.weight"]=se_l1(o[p+"se.excitation.0.weight"])
            out[p+"se.excitation.2.weight"]=se_l2(o[p+"se.excitation.2.weight"])
            out[p+"se.excitation.2.bias"]=torch.zeros(NF_N)
        else:        # NEW SE: identity (weight 0, final bias +15 → sigmoid≈1)
            out[p+"se.excitation.0.weight"]=torch.zeros(SEH_N, NF_N)
            out[p+"se.excitation.2.weight"]=torch.zeros(NF_N, SEH_N)
            out[p+"se.excitation.2.bias"]=torch.full((NF_N,), 20.0)   # sigmoid(20)≈1 → exact identity
    # heads: only the first conv's INPUT widens; everything else identical
    for h in ("policy_head","reply_head"):
        out[f"{h}.0.weight"]=conv_in(o[f"{h}.0.weight"])
        for k in ("weight","bias","running_mean","running_var"): out[f"{h}.1.{k}"]=o[f"{h}.1.{k}"]
        out[f"{h}.1.num_batches_tracked"]=o[f"{h}.1.num_batches_tracked"]
        out[f"{h}.4.weight"]=o[f"{h}.4.weight"]; out[f"{h}.4.bias"]=o[f"{h}.4.bias"]
    for h in ("value_head","material_head","plies_head"):
        out[f"{h}.0.weight"]=conv_in(o[f"{h}.0.weight"])
        for k in ("weight","bias","running_mean","running_var"): out[f"{h}.1.{k}"]=o[f"{h}.1.{k}"]
        out[f"{h}.1.num_batches_tracked"]=o[f"{h}.1.num_batches_tracked"]
        for idx in (4,6):
            if f"{h}.{idx}.weight" in o: out[f"{h}.{idx}.weight"]=o[f"{h}.{idx}.weight"]; out[f"{h}.{idx}.bias"]=o[f"{h}.{idx}.bias"]

    missing = set(n) - set(out); extra = set(out) - set(n)
    assert not missing and not extra, f"key mismatch missing={list(missing)[:3]} extra={list(extra)[:3]}"
    new.load_state_dict(out, strict=True)

    # ── verify function preservation on random inputs ──
    torch.manual_seed(1)
    x = torch.randn(8,120,8,8)
    with torch.no_grad():
        po,vo = old(x); pn,vn = new(x)
    dp = (po-pn).abs().max().item(); dv = (vo-vn).abs().max().item()
    print(f"{inp.split('/')[-1]} → {outp.split('/')[-1]} | max|Δpolicy|={dp:.2e} max|Δvalue|={dv:.2e}")
    assert dp < 1e-3 and dv < 1e-3, "FUNCTION NOT PRESERVED — widening is wrong"

    save = {"model_state_dict": new.state_dict()}   # drop optimizer/scheduler (arch changed → fresh)
    torch.save(save, outp)
    print(f"  ✓ preserved, wrote 20x320 all-SE checkpoint → {outp}")

if __name__ == "__main__":
    grow(sys.argv[1], sys.argv[2])
