# ========= NEW: Small Object Detection Layer (SODL) =========

# yolo3d_mrsf.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn.functional as F
# ===============================
# Utility functions for Streamlit
# ===============================
# ---------- חובה: בונה מודל ריק (המשקלים נטענים ב-Streamlit) ----------
def build_model():
    m = YOLOv8_3D_Connectivity(
        crop_size=(128, 128, 128),
        pres_thresh=0.5,
        lambda_box=1.0,
        width=32,
        neck_w=128,
        na=3
    )
    return m

# ---------- חובה: אינפרנס והחזרת N×5 (cls,z,y,x,diam) בווקסלים ----------
def infer_nodules(volume_np, model, device):
    import numpy as np, torch
    vol = volume_np.astype(np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax <= vmin: vmax = vmin + 1.0
    vol = (vol - vmin) / (vmax - vmin)

    z, y, x =128, 128, 128
    t = torch.from_numpy(vol[None, None, ...]).to(device=device, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        out = model(t, mode="eval")   # המודל כבר מחזיר best_pred + has_nodule_detected

    has_nodule = (out["has_nodule_detected"] > 0).item()
    if not has_nodule:
        return np.zeros((0, 5), dtype=np.float32)

    cz, cy, cx, rz, ry, rx = out["best_pred"][0].detach().cpu().numpy()
    
    cz_v, cy_v, cx_v = cz * z, cy * y, cx * x

    diam_v = 2.0 * max(rz * z, ry * y, rx * x)
    return np.array([[1.0, cz_v, cy_v, cx_v, diam_v]], dtype=np.float32)

def eapiou_3d_loss_cr(pred_box, gt_box, eps=1e-9, reduction='mean', lambda_ar=0.1):
    """
    EAPIoU-3D תואם לפורמט [cz, cy, cx, rz, ry, rx]
    """
    pc = pred_box[..., :3]
    pr = torch.clamp(pred_box[..., 3:], min=eps)
    gc = gt_box[..., :3]
    gr = torch.clamp(gt_box[..., 3:], min=eps)

    # פינות על בסיס רדיוסים (לא על רוחבים מלאים)
    pmin, pmax = pc - pr, pc + pr
    gmin, gmax = gc - gr, gc + gr

    # IoU נפחי
    inter_min = torch.maximum(pmin, gmin)
    inter_max = torch.minimum(pmax, gmax)
    inter = (inter_max - inter_min).clamp(min=0.0)
    inter_vol = inter[..., 0] * inter[..., 1] * inter[..., 2]

    pvol = (2*pr)[..., 0] * (2*pr)[..., 1] * (2*pr)[..., 2]
    gvol = (2*gr)[..., 0] * (2*gr)[..., 1] * (2*gr)[..., 2]
    iou = inter_vol / (pvol + gvol - inter_vol + eps)

    # ענישת מרחק מרכזים (כמו CIoU)
    enc_min = torch.minimum(pmin, gmin)
    enc_max = torch.maximum(pmax, gmax)
    enc_dims = enc_max - enc_min
    c2 = (enc_dims**2).sum(dim=-1) + eps
    center_penalty = ((pc - gc)**2).sum(dim=-1) / c2

    # רוחבים מלאים לצורך היחסי ממדים
    pd, ph, pw = 2*pr[..., 0], 2*pr[..., 1], 2*pr[..., 2]  # d=z, h=y, w=x
    gd, gh, gw = 2*gr[..., 0], 2*gr[..., 1], 2*gr[..., 2]

    def ratio_term(a_num, a_den, b_num, b_den):
        return (torch.atan(a_num/(a_den+eps)) - torch.atan(b_num/(b_den+eps)))**2

    # v (כמו CIoU) – ממוצע על שלוש זוגות יחסיים (w/h, w/d, h/d)
    v = (4.0/(math.pi**2)) * (
        ratio_term(pw, ph, gw, gh) +
        ratio_term(pw, pd, gw, gd) +
        ratio_term(ph, pd, gh, gd)
    ) / 3.0

    # רכיב יחס־ממדים מחוזק (EAPIoU): סטיות מריבוע היחסים
    r_wh = (pw/(ph+eps) - gw/(gh+eps))**2
    r_wd = (pw/(pd+eps) - gw/(gd+eps))**2
    r_hd = (ph/(pd+eps) - gh/(gd+eps))**2
    aspect_penalty = (r_wh + r_wd + r_hd) / 3.0

    eapiou = iou - center_penalty - lambda_ar*(v + aspect_penalty)
    loss = 1.0 - eapiou

    if reduction == 'mean': return loss.mean()
    if reduction == 'sum':  return loss.sum()
    return loss


def eapiou_3d_loss_cr1(pred_box, gt_box, eps=1e-9, reduction='mean'):
    """
    Enhanced Aspect Ratio Penalty IoU (EAPIoU) - 3D גרסה תלת-ממדית
    לפי המאמר Improved YOLO-Based Pulmonary Nodule Detection with Spatial-SE Attention and EAPIoU.
    """

    # --- פרמטר קבוע לפי המאמר ---
    lambda_ar = 0.1  # עדיף קבוע, לא בפרמטרים

    # פירוק התיבות
    x1, y1, z1, w1, h1, d1 = torch.chunk(pred_box, 6, dim=-1)
    x2, y2, z2, w2, h2, d2 = torch.chunk(gt_box, 6, dim=-1)

    # חישוב תיבות גבוהות/נמוכות
    x1_min, y1_min, z1_min = x1 - w1/2, y1 - h1/2, z1 - d1/2
    x1_max, y1_max, z1_max = x1 + w1/2, y1 + h1/2, z1 + d1/2
    x2_min, y2_min, z2_min = x2 - w2/2, y2 - h2/2, z2 - d2/2
    x2_max, y2_max, z2_max = x2 + w2/2, y2 + h2/2, z2 + d2/2

    # נפחי חיתוך
    inter_x = (torch.min(x1_max, x2_max) - torch.max(x1_min, x2_min)).clamp(min=0)
    inter_y = (torch.min(y1_max, y2_max) - torch.max(y1_min, y2_min)).clamp(min=0)
    inter_z = (torch.min(z1_max, z2_max) - torch.max(z1_min, z2_min)).clamp(min=0)
    inter_vol = inter_x * inter_y * inter_z

    vol1 = w1 * h1 * d1
    vol2 = w2 * h2 * d2
    union_vol = vol1 + vol2 - inter_vol + eps
    iou = inter_vol / union_vol

    # --- מרחק מרכזים ---
    center_dist = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2

    # --- תיבה מקיפה ---
    cw = torch.max(x1_max, x2_max) - torch.min(x1_min, x2_min)
    ch = torch.max(y1_max, y2_max) - torch.min(y1_min, y2_min)
    cd = torch.max(z1_max, z2_max) - torch.min(z1_min, z2_min)
    c2 = cw**2 + ch**2 + cd**2 + eps

    # --- ענישה על יחס ממדים (כמו CIoU) ---
    v = (4 / (torch.pi ** 2)) * ((torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2)

    # --- ענישה ריבועית על יחס רוחב־גובה־עומק (החידוש של המאמר) ---
    r1 = w1 / (h1 + eps)
    r2 = w2 / (h2 + eps)
    aspect_penalty = (r1 - r2) ** 2

    # --- חישוב סופי ---
    eapiou = iou - center_dist / c2 - lambda_ar * v - lambda_ar * aspect_penalty
    loss = 1.0 - eapiou

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss

class ConvBNAct3d(nn.Sequential):
    def __init__(self, ci, co, k=3, s=1, p=None):
        if p is None:
            p = k // 2
        super().__init__(
            nn.Conv3d(ci, co, k, s, p, bias=False),
            nn.GroupNorm(8 if co >= 8 else 1, co),
            nn.Hardswish()
        )
# ===============================
# Blocks
# ===============================

class EODConv3d(nn.Module):
    """Axial-ish depthwise conv + pointwise (קליל)"""
    def __init__(self, c, k=5):
        super().__init__()
        p = k // 2
        self.d = nn.Conv3d(c, c, (k,1,1), padding=(p,0,0), groups=c, bias=False)
        self.h = nn.Conv3d(c, c, (1,k,1), padding=(0,p,0), groups=c, bias=False)
        self.w = nn.Conv3d(c, c, (1,1,k), padding=(0,0,p), groups=c, bias=False)
        self.pw = nn.Conv3d(c, c, 1, bias=False)
        self.gn = nn.GroupNorm(8 if c>=8 else 1, c)
        self.act = nn.Hardswish()

    def forward(self, x):
        y = self.d(x); y = self.h(y); y = self.w(y)
        y = self.pw(y)
        return self.act(self.gn(y + x))

class MSRF3dHead(nn.Module):
    """Multi-Scale Receptive Field: 4 ענפים + fuse"""
    def __init__(self, c_in, c_mid, out_ch):
        super().__init__()
        self.b1 = nn.Sequential(EODConv3d(c_in, 3), nn.Conv3d(c_in, c_mid, 1, bias=False),
                                nn.GroupNorm(8 if c_mid>=8 else 1, c_mid), nn.Hardswish())
        self.b2 = nn.Sequential(EODConv3d(c_in, 3), nn.Conv3d(c_in, c_mid, 3, padding=1, bias=False),
                                nn.GroupNorm(8 if c_mid>=8 else 1, c_mid), nn.Hardswish())
        self.b3 = nn.Sequential(EODConv3d(c_in, 3), nn.Conv3d(c_in, c_mid, 3, padding=2, dilation=2, bias=False),
                                nn.GroupNorm(8 if c_mid>=8 else 1, c_mid), nn.Hardswish())
        self.b4 = nn.Sequential(EODConv3d(c_in, 5), nn.Conv3d(c_in, c_mid, 5, padding=2, bias=False),
                                nn.GroupNorm(8 if c_mid>=8 else 1, c_mid), nn.Hardswish())
        self.fuse = nn.Conv3d(c_mid*4, out_ch, 1)

    def forward(self, x):
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.fuse(y)

class EODConv3dAttn(nn.Module):
    """
    EODConv-בסגנון המאמר: שערי attention סקאלריים על מימדי kernel-space:
    space (Z/Y/X), in-channels, out-channels. קל-משקל ומתאים ל-3D.
    """
    def __init__(self, c, k=3):
        super().__init__()
        p = k // 2
        # צירי עומק נפרדים (כמו גרסה אקסיאלית)
        self.d  = nn.Conv3d(c, c, (k,1,1), padding=(p,0,0), groups=c, bias=False)
        self.h  = nn.Conv3d(c, c, (1,k,1), padding=(0,p,0), groups=c, bias=False)
        self.w  = nn.Conv3d(c, c, (1,1,k), padding=(0,0,p), groups=c, bias=False)
        self.pw = nn.Conv3d(c, c, 1, bias=False)
        self.gn  = nn.GroupNorm(8 if c>=8 else 1, c)
        self.act = nn.Hardswish()
        # שערים: הפקת סקאלרים מגלובל פולי׳ינג
        self.gap      = nn.AdaptiveAvgPool3d(1)
        self.fc_space = nn.Sequential(nn.Conv3d(c, c, 1, bias=True), nn.Sigmoid())
        self.fc_in    = nn.Sequential(nn.Conv3d(c, c, 1, bias=True), nn.Sigmoid())
        self.fc_out   = nn.Sequential(nn.Conv3d(c, c, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        y = self.d(x); y = self.h(y); y = self.w(y); y = self.pw(y)
        # סקאלרים (בפועל טנסורים 1×C×1×1×1) שמדמים משקלים על מימדי kernel-space
        g = self.gap(x)
        a_space = self.fc_space(g)
        a_in    = self.fc_in(g)
        a_out   = self.fc_out(g)
        y = y * a_space * a_in * a_out
        return self.act(self.gn(y + x))


class MSRF3dMod(nn.Module):
    """
    MSRF מודולטיבי לפי המאמר: מספר ענפים → סכימה → 1x1 → מודולציה אלמנטרית x*y.
    שימוש בקונבולוציות "מרחביות מופרדות" 3D כקירוב:
    (1×5×1, 5×1×1) ו-(1×7×1, 7×1×1).
    """
    def __init__(self, c):
        super().__init__()
        self.pre7 = nn.Conv3d(c, c, 7, padding=3, bias=False)
        self.c1   = nn.Conv3d(c, c, 1, bias=False)
        self.a15  = nn.Sequential(
            nn.Conv3d(c, c, (1,5,1), padding=(0,2,0), bias=False),
            nn.Conv3d(c, c, (5,1,1), padding=(2,0,0), bias=False)
        )
        self.a17  = nn.Sequential(
            nn.Conv3d(c, c, (1,7,1), padding=(0,3,0), bias=False),
            nn.Conv3d(c, c, (7,1,1), padding=(3,0,0), bias=False)
        )
        self.post = nn.Conv3d(c, c, 1, bias=False)
        self.gn   = nn.GroupNorm(8 if c>=8 else 1, c)
        self.act  = nn.Sigmoid()  # מפיק מסיכת מודולציה [0..1]

    def forward(self, x):
        y0 = self.pre7(x)
        y1 = self.c1(y0)
        y2 = self.a15(y0)
        y3 = self.a17(y0)
        y  = y1 + y2 + y3
        y  = self.post(y)
        m  = self.act(self.gn(y))
        return x * m


class ELAN3dO2(nn.Module):
    """קירוב 3D ל-ELAN-O2: שתי 'יחידות' עם חיבורי זהות לשימור מידע מקומי."""
    def __init__(self, c):
        super().__init__()
        self.b1  = ConvBNAct3d(c, c, 3)
        self.b2  = ConvBNAct3d(c, c, 3)
        self.out = ConvBNAct3d(c, c, 1, 1, 0)

    def forward(self, x):
        y = self.b1(x) + x
        y = self.b2(y) + y
        return self.out(y)


class SODL3dHead(nn.Module):
    """
    SODL אמיתי (stride=4): 1x1 על פיצ'ר stride=8 → Upsample×2 → סכימה עם פיצ'ר stride=4 מוקדם
    → ELAN3dO2 → ראש תחזית (bbox/cls).
    """
    def __init__(self, c, out_bbox, out_cls):
        super().__init__()
        self.reduce = nn.Conv3d(c, c, 1, bias=False)
        self.elan   = ELAN3dO2(c)
        self.bbox   = MSRF3dHead(c, c//2, out_bbox)
        self.cls    = MSRF3dHead(c, c//2, out_cls)

    def forward(self, feat_s8, feat_s4):
        x = F.interpolate(self.reduce(feat_s8), size=feat_s4.shape[2:], mode='trilinear', align_corners=False)
        x = x + feat_s4
        x = self.elan(x)
        return self.bbox(x), self.cls(x), x  # גם נחזיר את הפיצ'ר המעודן אם נרצה להשתמש בו בהמשך


class TinyBackbone3d(nn.Module):
    """3× downsample: 128³ → 64³ → 32³ → 16³"""
    def __init__(self, in_ch=1, w=32):
        super().__init__()
        self.stem      = ConvBNAct3d(in_ch, w, 3, 2)  # 128→64
        self.stem_eod  = EODConv3dAttn(w, k=3)
        self.c3        = ConvBNAct3d(w,   w)
        self.down4     = ConvBNAct3d(w,   w*2, 3, 2)  # 64→32
        self.c4_pre    = EODConv3dAttn(w*2, k=3)
        self.c4        = ConvBNAct3d(w*2, w*2)
        self.down5     = ConvBNAct3d(w*2, w*4, 3, 2)  # 32→16
        self.c5        = ConvBNAct3d(w*4, w*4)

    def forward(self, x):
        x  = self.stem(x)
        x  = self.stem_eod(x)
        f1 = self.c3(x)          # 64³ (stride=2)
        x  = self.down4(f1)
        x  = self.c4_pre(x)
        f2 = self.c4(x)          # 32³ (stride=4)
        x  = self.down5(f2)
        f3 = self.c5(x)          # 16³ (stride=8)
        return f3, f2, f1






         
class PAN3d(nn.Module):
    """FPN/PAN neck: top-down + light bottom-up"""
    def __init__(self, c3, c2, c1, w=128):
        super().__init__()
        self.l3 = ConvBNAct3d(c3, w, 1, 1, 0)
        self.l2 = ConvBNAct3d(c2, w, 1, 1, 0)
        self.l1 = ConvBNAct3d(c1, w, 1, 1, 0)
        self.td2 = ConvBNAct3d(w, w, 3)
        self.td1 = ConvBNAct3d(w, w, 3)
        self.bu2 = ConvBNAct3d(w, w, 3, 2)
        self.bu3 = ConvBNAct3d(w, w, 3, 2)

    def forward(self, f3, f2, f1):
        p3 = self.l3(f3)
        p2 = self.l2(f2) + F.interpolate(p3, size=f2.shape[2:], mode='trilinear', align_corners=False)
        p2 = self.td2(p2)
        p1 = self.l1(f1) + F.interpolate(p2, size=f1.shape[2:], mode='trilinear', align_corners=False)
        p1 = self.td1(p1)
        # enrich semantics bottom-up (optional)
        _n2 = self.bu2(p1) + p2
        _n3 = self.bu3(_n2) + p3
        return p1, p2, p3   # משתמשים ב-P1 כסקייל ההסקה העיקרי

# ===============================
# EIoU-3D loss (center+radius normalized)
# ===============================
def eiou_3d_loss(pred_box, gt_box, eps=1e-7, reduction='mean'):
    pr = torch.clamp(pred_box[:, 3:], min=eps)
    gr = torch.clamp(gt_box[:, 3:],   min=eps)
    pc = pred_box[:, :3]; gc = gt_box[:, :3]

    pmin, pmax = pc - pr, pc + pr
    gmin, gmax = gc - gr, gc + gr

    inter_min = torch.max(pmin, gmin)
    inter_max = torch.min(pmax, gmax)
    inter_dims = (inter_max - inter_min).clamp(min=0)
    inter_vol  = inter_dims.prod(dim=1)

    pvol = (2*pr).prod(dim=1)
    gvol = (2*gr).prod(dim=1)
    union = pvol + gvol - inter_vol + eps
    iou   = inter_vol / union

    enc_min = torch.min(pmin, gmin); enc_max = torch.max(pmax, gmax)
    enc_dims = enc_max - enc_min
    diag2 = (enc_dims**2).sum(dim=1) + eps
    center_penalty = ((pc - gc)**2).sum(dim=1) / diag2

    pd = 2*pr; gd = 2*gr
    size_penalty = ((pd[:,0]-gd[:,0])**2)/(enc_dims[:,0]**2+eps) \
                 + ((pd[:,1]-gd[:,1])**2)/(enc_dims[:,1]**2+eps) \
                 + ((pd[:,2]-gd[:,2])**2)/(enc_dims[:,2]**2+eps)

    eiou = 1.0 - iou + center_penalty + size_penalty
    if reduction=='mean': return eiou.mean()
    if reduction=='sum':  return eiou.sum()
    return eiou




class SODL3d(nn.Module):
    """
    SODL-3D: בלוק עידון ייעודי לאובייקטים קטנים.
    רצף: EODConv3d → MSRF3dHead → 1x1 Conv שמחזירה את אותו מספר ערוצים.
    """
    def __init__(self, c):
        super().__init__()
        self.eod = EODConv3d(c, k=5)
        self.msr = MSRF3dHead(c_in=c, c_mid=c//2, out_ch=c)  # שומר על אותה כמות ערוצים
        self.out = nn.Conv3d(c, c, kernel_size=1, bias=False)
        self.gn  = nn.GroupNorm(8 if c>=8 else 1, c)
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.eod(x)
        x = self.msr(x)
        x = self.out(x)
        return self.act(self.gn(x))

def ciou_3d_loss(pred_box, gt_box, eps=1e-7, reduction='mean'):
    """
    3D CIoU loss: 1 - (IoU - center_penalty - alpha * aspect_penalty)
    pred_box, gt_box: [N,6] in [cz, cy, cx, rz, ry, rx] (normalized)
    """
    # radii safety
    pr = torch.clamp(pred_box[:, 3:], min=eps)
    gr = torch.clamp(gt_box[:, 3:],   min=eps)
    pc, gc = pred_box[:, :3], gt_box[:, :3]

    # corners
    pmin, pmax = pc - pr, pc + pr
    gmin, gmax = gc - gr, gc + gr

    # intersection/union
    inter_min  = torch.max(pmin, gmin)
    inter_max  = torch.min(pmax, gmax)
    inter_dims = (inter_max - inter_min).clamp(min=0)
    inter_vol  = inter_dims.prod(dim=1)

    pvol = (2 * pr).prod(dim=1)
    gvol = (2 * gr).prod(dim=1)
    union = pvol + gvol - inter_vol + eps
    iou   = inter_vol / union                                # [N]

    # center distance penalty (DIoU part)
    enc_min  = torch.min(pmin, gmin)
    enc_max  = torch.max(pmax, gmax)
    enc_dims = enc_max - enc_min
    c2 = (enc_dims ** 2).sum(dim=1) + eps                    # [N]
    center_pen = ((pc - gc) ** 2).sum(dim=1) / c2            # [N]

    # aspect-ratio penalty in 3D (pairwise ratios w/h, w/d, h/d with atan)
    pw, ph, pd = 2*pr[:, 2], 2*pr[:, 1], 2*pr[:, 0]          # w=x, h=y, d=z
    gw, gh, gd = 2*gr[:, 2], 2*gr[:, 1], 2*gr[:, 0]
    pw = pw.clamp_min(eps); ph = ph.clamp_min(eps); pd = pd.clamp_min(eps)
    gw = gw.clamp_min(eps); gh = gh.clamp_min(eps); gd = gd.clamp_min(eps)

    def _ratio_term(a_num, a_den, b_num, b_den):
        return (torch.atan(a_num / a_den) - torch.atan(b_num / b_den)) ** 2

    v = (4.0 / (math.pi ** 2)) * (
            _ratio_term(pw, ph, gw, gh) +
            _ratio_term(pw, pd, gw, gd) +
            _ratio_term(ph, pd, gh, gd)
        ) / 3.0                                              # [N]

    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - center_pen - alpha * v
    loss = 1.0 - ciou

    if reduction == 'mean': return loss.mean()
    if reduction == 'sum':  return loss.sum()
    return loss




# ========= UPDATED: YOLOv8_3D_Connectivity with explicit SODL on P1 =========
class YOLOv8_3D_Connectivity(nn.Module):


                
    def __init__(self,
             crop_size=(128,128,128),
             pres_thresh=0.5,
             lambda_box=1.0,
             width=32, neck_w=128,
             na=3):            
        super().__init__()
        self.crop_size = crop_size
        self.pres_thresh = pres_thresh
        self.lambda_box = lambda_box
        self.na = na
        
        
        # Backbone & Neck
        self.backbone = TinyBackbone3d(in_ch=1, w=width)
        self.neck = PAN3d(width*4, width*2, width, w=neck_w)
        
        
        # SODL על כל הסקיילים
        #self.sodl_p1 = SODL3d(neck_w)
        #self.sodl_p2 = SODL3d(neck_w)
        #self.sodl_p3 = SODL3d(neck_w)
        
        self.msrf_p1  = MSRF3dMod(neck_w)
        self.msrf_p2  = MSRF3dMod(neck_w)
        self.msrf_p3  = MSRF3dMod(neck_w)
        self.sodl_head = SODL3dHead(c=neck_w, out_bbox=self.na*7, out_cls=2)
        # Heads (bbox: na*(1+6)=na*7; cls: 2) — 3 Anchors לכל ראש
        self.bbox_head_s = MSRF3dHead(neck_w, neck_w//2, na*7)
        self.bbox_head_m = MSRF3dHead(neck_w, neck_w//2, na*7)
        self.bbox_head_l = MSRF3dHead(neck_w, neck_w//2, na*7)
        
        
        self.cls_head_s = MSRF3dHead(neck_w, neck_w//2, 2)
        self.cls_head_m = MSRF3dHead(neck_w, neck_w//2, 2)
        self.cls_head_l = MSRF3dHead(neck_w, neck_w//2, 2)
        
        
        # Anchors פר־HEAD (קטרים בווקסלים): P1 קטנים, P2 בינוניים, P3 גדולים
        D,H,W = crop_size
        to_norm = lambda d: torch.tensor([d/D, d/H, d/W], dtype=torch.float32)
        #anchors_s = [3, 6, 10]
        #anchors_m = [10, 16, 22]
        #anchors_l = [22, 28, 33]
        anchors_xs = [3, 5, 7]        # XS ל-stride≈4 (SODL)
        anchors_s  = [3, 6, 10]       # S
        anchors_m  = [10, 16, 22]     # M
        anchors_l  = [22, 28, 33]     # L
        A_xs = torch.stack([to_norm(a) for a in anchors_xs], dim=0) / 2.0

        A_s = torch.stack([to_norm(a) for a in anchors_s], dim=0) / 2.0 # radius ∈ [0,0.5]
        A_m = torch.stack([to_norm(a) for a in anchors_m], dim=0) / 2.0
        A_l = torch.stack([to_norm(a) for a in anchors_l], dim=0) / 2.0
        self.register_buffer('anchors_rad_xs', A_xs) # [na,3]
        self.register_buffer('anchors_rad_s', A_s) # [na,3]
        self.register_buffer('anchors_rad_m', A_m)
        self.register_buffer('anchors_rad_l', A_l)
    @staticmethod
    def _sigmoid01(t): return torch.sigmoid(t).clamp(0., 1.)

    def _decode(self, bmap, clsmap, anchors_rad):
        """
        bmap: [B, na*7, Dz,Hy,Wx] → [B,na,7,Dz,Hy,Wx]
        fields: [obj, cz,cy,cx, rz,ry,rx]
        anchors_rad: [na,3] (normalized radii per head)
        מחזיר:
            obj_logits [B,na,D,H,W],
            center     [B,na,3,D,H,W],
            radius     [B,na,3,D,H,W],
            conf       [B,na,D,H,W],
            shape      (Dz,Hy,Wx)
        """
        #B, _, Dz, Hy, Wx = bmap.shape
        B, C, Dz, Hy, Wx = bmap.shape     # C = na * 7[]
        na = anchors_rad.shape[0]
        b = bmap.view(B, na, 7, Dz, Hy, Wx)

        #b = bmap.view(B, self.na, 7, Dz, Hy, Wx)

        # חלקי ה־bbox
        obj_logits = b[:, :, 0, ...]
        ctr_raw    = b[:, :, 1:4, ...]
        rad_raw    = b[:, :, 4:7, ...]

        # רשת גריד לפי ממדים (Anchor-Free Center)
        dtype = bmap.dtype
        gz = torch.arange(Dz, device=bmap.device, dtype=dtype).view(1,1,Dz,1,1)
        gy = torch.arange(Hy, device=bmap.device, dtype=dtype).view(1,1,1,Hy,1)
        gx = torch.arange(Wx, device=bmap.device, dtype=dtype).view(1,1,1,1,Wx)

        # חישוב מרכז מנורמל ∈ [0,1]
        cz = (self._sigmoid01(ctr_raw[:, :, 0]) + gz) / Dz
        cy = (self._sigmoid01(ctr_raw[:, :, 1]) + gy) / Hy
        cx = (self._sigmoid01(ctr_raw[:, :, 2]) + gx) / Wx
        center = torch.stack([cz, cy, cx], dim=2)  # [B,na,3,Dz,Hy,Wx]

        # רדיוס לפי עוגנים פר-ראש, ללא SCALE
        ar = anchors_rad.view(1, na, 3, 1, 1, 1)              # [1,na,3,1,1,1]

        #ar = anchors_rad.view(1, self.na, 3, 1, 1, 1)
        #radius = torch.clamp(ar, 1e-6, 0.5).expand(B, self.na, 3, Dz, Hy, Wx).contiguous()
        radius = self._sigmoid01(rad_raw) * ar                 # broadcast → [B,na,3,Dz,Hy,Wx]
        radius = torch.clamp(radius, 1e-6, 0.5).contiguous()   # לשמור תחום ויציבות
        # Confidence = sigmoid(obj) * sigmoid(cls_pos)
        cls_pos = torch.sigmoid(clsmap[:, 1:2, ...])  # [B,1,Dz,Hy,Wx]
        conf = torch.sigmoid(obj_logits) * cls_pos    # [B,na,Dz,Hy,Wx]

        return obj_logits, center, radius, conf, (Dz,Hy,Wx)
        
        
    def forward(self, x, epochnum=0, labels=None, bboxes=None, mode="eval"):
        import torch
        import torch.nn.functional as F
        B = x.size(0)  # לפי בקשתך B צפוי להיות 1
    
        # Backbone + Neck
        f3, f2, f1 = self.backbone(x)
        p1, p2, p3 = self.neck(f3, f2, f1)
    
        # ---- SODL refine (P1,P2,P3) ----
        #p1 = self.sodl_p1(p1)
        #p2 = self.sodl_p2(p2)
        #p3 = self.sodl_p3(p3)
    
        p1 = self.msrf_p1(p1)
        p2 = self.msrf_p2(p2)
        p3 = self.msrf_p3(p3)
        # SODL אמיתי ב-stride≈4: Upsample×2 מ-p3 וסכימה עם f2 (המפה המוקדמת ב-stride≈4)
        #b_xs, c_xs, p2_sodl = self.sodl_head(p3, f2)
   
        b_xs, c_xs, p2_sodl = self.sodl_head(p3, p2)

        # Heads (upsample) – decode לכל ראש ואז איחוד
        #b1 = self.bbox_head_s(p1)
        #b2 = F.interpolate(self.bbox_head_m(p2), size=p1.shape[2:], mode='trilinear', align_corners=False)
        #b3 = F.interpolate(self.bbox_head_l(p3), size=p1.shape[2:], mode='trilinear', align_corners=False)
        #c1 = self.cls_head_s(p1)
        #c2 = F.interpolate(self.cls_head_m(p2), size=p1.shape[2:], mode='trilinear', align_corners=False)
        #c3 = F.interpolate(self.cls_head_l(p3), size=p1.shape[2:], mode='trilinear', align_corners=False)
        # ראשי S/M/L כמו קודם (מאוחדים לגודל p1), ולצדם XS מה-SODL
        b_xs_up = F.interpolate(b_xs, size=p1.shape[2:], mode='trilinear', align_corners=False)
        c_xs_up = F.interpolate(c_xs, size=p1.shape[2:], mode='trilinear', align_corners=False)
        b1 = self.bbox_head_s(p1)
        b2 = F.interpolate(self.bbox_head_m(p2), size=p1.shape[2:], mode='trilinear', align_corners=False)
        b3 = F.interpolate(self.bbox_head_l(p3), size=p1.shape[2:], mode='trilinear', align_corners=False)
        c1 = self.cls_head_s(p1)
        c2 = F.interpolate(self.cls_head_m(p2), size=p1.shape[2:], mode='trilinear', align_corners=False)
        c3 = F.interpolate(self.cls_head_l(p3), size=p1.shape[2:], mode='trilinear', align_corners=False)

        # הסתברויות class=1 לכל סקייל
        p1_pos = F.softmax(c1, dim=1)[:, 1:2, ...]
        p2_pos = F.softmax(c2, dim=1)[:, 1:2, ...]
        p3_pos = F.softmax(c3, dim=1)[:, 1:2, ...]
        # OR הסתברותי: "לפחות אחד"
        p_pos_or = 1.0 - (1.0 - p1_pos) * (1.0 - p2_pos) * (1.0 - p3_pos)
        # מפה דו-ערוצית לויזואליזציה/החלטה (לא ללוס)
        cls_fused = torch.cat([1.0 - p_pos_or, p_pos_or], dim=1)
        obj_xs, ctr_xs, rad_xs, conf_xs, shp = self._decode(b_xs_up, c_xs_up, self.anchors_rad_xs)
        obj_s,  ctr_s,  rad_s,  conf_s,  shp = self._decode(b1,     c1,      self.anchors_rad_s)

        #obj_s, ctr_s, rad_s, conf_s, shp = self._decode(b1, c1, self.anchors_rad_s)
        obj_m, ctr_m, rad_m, conf_m, _   = self._decode(b2, c2, self.anchors_rad_m)
        obj_l, ctr_l, rad_l, conf_l, _   = self._decode(b3, c3, self.anchors_rad_l)
        Dz, Hy, Wx = shp
        obj_logits = torch.cat([obj_xs, obj_s, obj_m, obj_l], dim=1)
        #center_map = torch.cat([ctr_s, ctr_m, ctr_l], dim=1)
        #radius_map = torch.cat([rad_s, rad_m, rad_l], dim=1)
        
        center_map = torch.cat([ctr_xs,  ctr_s,  ctr_m,  ctr_l], dim=1)        # [B, 4*na, 3, D, H, W]

        radius_map = torch.cat([rad_xs,  rad_s,  rad_m,  rad_l], dim=1)        # [B, 4*na, 3, D, H, W]


        #obj_logits = torch.cat([obj_s, obj_m, obj_l], dim=1)
        #center_map = torch.cat([ctr_s, ctr_m, ctr_l], dim=1)
        #radius_map = torch.cat([rad_s, rad_m, rad_l], dim=1)
        conf_or_map = 1.0 - (1.0 - conf_xs)*(1.0 - conf_s) * (1.0 - conf_m) * (1.0 - conf_l)  # [B,na,D,H,W]
        

        #conf_all = torch.cat([conf_s, conf_m, conf_l], dim=1)  # [B,3*na,D,H,W]
        conf_all = torch.cat([conf_xs, conf_s, conf_m, conf_l], dim=1)         # [B, 4*na, D, H, W]

        # קונפידנס מאוחד OR בין הראשים לכל anchor
        #conf_or_map = 1.0 - (1.0 - conf_s) * (1.0 - conf_m) * (1.0 - conf_l)  # [B,na,D,H,W]
         # === חיבורי קונפידנס מכל הראשים ===
        # OR הסתברותי בין הראשים — לשלב ההחלטה הסופית
        
        # איחוד כל הקונפידנסים יחד — לשימוש ב-argmax, IoU ולוס
        #conf_all = torch.cat([conf_s, conf_m, conf_l], dim=1)  # [B,3*na,D,H,W]
        #has_nodule = labels[:, 0] == 1
        if labels is not None:
            has_nodule = labels[:, 0] == 1
        else:
            has_nodule = torch.zeros(B, dtype=torch.bool, device=x.device)
        # === Targets: labels [cls, cz,cy,cx, d] בווקסלים @128³ ===
        if has_nodule.any():

            lab = labels[:, 0, :] if labels.dim() == 3 else labels  # [B,5]
            norm_labels = lab.clone()
            norm_labels[:, 1:4] = norm_labels[:, 1:4] / 128.0       # center ∈ [0,1]
            norm_labels[:, 4]   = norm_labels[:, 4]   / 128.0       # diameter ∈ [0,1]
            gt_center = norm_labels[:, 1:4]                            # [B,3]
            gt_r_sc   = (norm_labels[:, 4:5] * 0.5)                   # [B,1] radius
            gt_radius = gt_r_sc.repeat(1, 3)                          # [B,3]
            gt_box    = torch.cat([gt_center, gt_radius], dim=1)      # [B,6]
            y_pres    = (lab[:, 0] > 0).float()
        else:
            gt_box = None
            gt_center = None
            y_pres = None
        if y_pres is not None:
            y_pres_mask = y_pres.view(B,1,1,1,1).bool()
        else:
            y_pres_mask = torch.zeros((B, 1, 1, 1, 1), dtype=torch.bool, device=x.device)
        # === IoU-based positive assignment עם ספים דינמיים ===
        def _iou_map_center_radius(center_map, radius_map, gt_box):
            Bx = gt_box.size(0)
            device = center_map.device
            dtype  = center_map.dtype
    
            ac = center_map
            ar = torch.clamp(radius_map, min=1e-9)
    
            bc = gt_box[:, :3].to(device=device, dtype=dtype).view(Bx, 1, 3, 1, 1, 1)
            br = gt_box[:, 3:].to(device=device, dtype=dtype).view(Bx, 1, 3, 1, 1, 1)
    
            a_min, a_max = ac - ar, ac + ar
            b_min, b_max = bc - br, bc + br
    
            inter_min = torch.maximum(a_min, b_min)
            inter_max = torch.minimum(a_max, b_max)
            inter = torch.clamp(inter_max - inter_min, min=0).prod(dim=2)
    
            va = (2 * ar).prod(dim=2)
            vb = (2 * br).prod(dim=2)
            iou = inter / (va + vb - inter + 1e-9)
            return iou
    
        if (mode in ("train","val")) and (gt_box is not None):
            pos_iou_th = 0.3 if epochnum < 3 else (0.4 if epochnum < 8 else 0.5)
            neg_iou_th = max(0.0, pos_iou_th - 0.1)
    

            
            iou_xs = _iou_map_center_radius(ctr_xs, rad_xs, gt_box)


            iou_s = _iou_map_center_radius(ctr_s, rad_s, gt_box)
            iou_m = _iou_map_center_radius(ctr_m, rad_m, gt_box)
            iou_l = _iou_map_center_radius(ctr_l, rad_l, gt_box)
            pos_xs = (iou_xs >= pos_iou_th) & y_pres_mask

            pos_s = (iou_s >= pos_iou_th) & y_pres_mask
            pos_m = (iou_m >= pos_iou_th) & y_pres_mask
            pos_l = (iou_l >= pos_iou_th) & y_pres_mask
            
            neg_s = ((iou_s < neg_iou_th) & y_pres_mask) | (~y_pres_mask.expand_as(iou_s))
            neg_m = ((iou_m < neg_iou_th) & y_pres_mask) | (~y_pres_mask.expand_as(iou_m))
            neg_l = ((iou_l < neg_iou_th) & y_pres_mask) | (~y_pres_mask.expand_as(iou_l))
            neg_xs = ((iou_xs < neg_iou_th) & y_pres_mask) | (~y_pres_mask.expand_as(iou_xs))

            ign_s = (~pos_s) & (~neg_s) & y_pres_mask
            ign_m = (~pos_m) & (~neg_m) & y_pres_mask
            ign_l = (~pos_l) & (~neg_l) & y_pres_mask
            ign_xs = (~pos_xs) & (~neg_xs) & y_pres_mask

            
            def _force_one_positive(iou_map_h, pos_h, neg_h, ign_h):
                with torch.no_grad():
                    none_pos = (pos_h.view(B, -1).sum(dim=1) == 0)
                    if none_pos.any():
                        flat = iou_map_h.view(B, -1)
                        best_idx = flat.argmax(dim=1)
                        Dd, Hh, Ww = iou_map_h.shape[2:]
                        cells = Dd * Hh * Ww
                        a = best_idx // cells; r = best_idx % cells
                        z = r // (Hh*Ww); y = (r % (Hh*Ww)) // Ww; x = r % Ww
                        bsel = torch.where(none_pos)[0]
                        pos_h[bsel, a[bsel], z[bsel], y[bsel], x[bsel]] = True
                        neg_h[bsel, a[bsel], z[bsel], y[bsel], x[bsel]] = False
                        ign_h[bsel, a[bsel], z[bsel], y[bsel], x[bsel]] = False
                return pos_h, neg_h, ign_h
            
              
            # כפיית חיובי רק באפוקים הראשונים ורק כשהתמונה חיובית (יש נודול)
            if (epochnum < 5) and (y_pres.sum() > 0):
                pos_s, neg_s, ign_s = _force_one_positive(iou_s, pos_s, neg_s, ign_s)
                pos_m, neg_m, ign_m = _force_one_positive(iou_m, pos_m, neg_m, ign_m)
                pos_l, neg_l, ign_l = _force_one_positive(iou_l, pos_l, neg_l, ign_l)            
                pos_xs, neg_xs, ign_xs = _force_one_positive(iou_xs, pos_xs, neg_xs, ign_xs)

    
#        else:

#            iou_s = torch.zeros_like(obj_s); iou_m = torch.zeros_like(obj_m); iou_l = torch.zeros_like(obj_l)
#            pos_s = torch.zeros_like(obj_s, dtype=torch.bool); pos_m = torch.zeros_like(obj_m, dtype=torch.bool); pos_l = torch.zeros_like(obj_l, dtype=torch.bool)
#            neg_s = torch.zeros_like(obj_s, dtype=torch.bool); neg_m = torch.zeros_like(obj_m, dtype=torch.bool); neg_l = torch.zeros_like(obj_l, dtype=torch.bool)
#            ign_s = torch.zeros_like(obj_s, dtype=torch.bool); ign_m = torch.zeros_like(obj_m, dtype=torch.bool); ign_l = torch.zeros_like(obj_l, dtype=torch.bool)
        else:
            iou_xs = torch.zeros_like(obj_xs); iou_s = torch.zeros_like(obj_s); iou_m = torch.zeros_like(obj_m); iou_l = torch.zeros_like(obj_l)
            pos_xs = torch.zeros_like(obj_xs, dtype=torch.bool); pos_s = torch.zeros_like(obj_s, dtype=torch.bool); pos_m = torch.zeros_like(obj_m, dtype=torch.bool); pos_l = torch.zeros_like(obj_l, dtype=torch.bool)
            neg_xs = torch.zeros_like(obj_xs, dtype=torch.bool); neg_s = torch.zeros_like(obj_s, dtype=torch.bool); neg_m = torch.zeros_like(obj_m, dtype=torch.bool); neg_l = torch.zeros_like(obj_l, dtype=torch.bool)
            ign_xs = torch.zeros_like(obj_xs, dtype=torch.bool); ign_s = torch.zeros_like(obj_s, dtype=torch.bool); ign_m = torch.zeros_like(obj_m, dtype=torch.bool); ign_l = torch.zeros_like(obj_l, dtype=torch.bool)
            
            #y_pres_mask = torch.zeros_like(obj_logits, dtype=torch.bool)


        
          # ---- הגנות ואינדוקס בטוח בעזרת gather (במקום advanced indexing) ----
        # 1) בדיקות תקינות בסיסיות בין המפות
        assert center_map.shape[1] == radius_map.shape[1] == conf_all.shape[1], \
            f"na_tot mismatch: ctr={center_map.shape[1]}, rad={radius_map.shape[1]}, conf={conf_all.shape[1]}"
        assert center_map.shape[3:] == radius_map.shape[3:] == conf_all.shape[2:], \
            f"spatial mismatch: ctr={center_map.shape}, rad={radius_map.shape}, conf={conf_all.shape}"

        # 2) הסרת NaN/Inf לפני argmax
        # קונפידנס מאוחד OR בין הראשים לכל anchor
        

        # 3) חישוב אינדקסים בצורה לינארית + הידוק לגבולות
        #B, na_tot, Dz, Hy, Wx = conf_or_map.shape
        #cells = Dz * Hy * Wx
        #flat_idx = conf_or_map.view(B, -1).argmax(dim=1)
        #B, na_tot, Dz, Hy, Wx = conf_all.shape
        #flat_idx = conf_all.view(B, -1).argmax(dim=1)
        
        #a_best = (flat_idx // cells).long()
        
        B, na_tot, Dz, Hy, Wx = conf_all.shape
        cells = Dz * Hy * Wx
        flat_idx = conf_all.view(B, -1).argmax(dim=1)
        a_best = (flat_idx // cells).long()
        rem    = (flat_idx %  cells).long()
        z_best = (rem // (Hy * Wx)).long()
        y_best = ((rem % (Hy * Wx)) // Wx).long()
        x_best = (rem % Wx).long()

        a_best = torch.clamp(a_best, 0, na_tot - 1)
        z_best = torch.clamp(z_best, 0, Dz - 1)
        y_best = torch.clamp(y_best, 0, Hy - 1)
        x_best = torch.clamp(x_best, 0, Wx - 1)
        bidx = torch.arange(B, device=x.device, dtype=torch.long)

        lin_best = (a_best * cells) + (z_best * (Hy * Wx)) + (y_best * Wx) + x_best  # [B]

        # 4) איסוף וקטורים [cz,cy,cx]/[rz,ry,rx] באמצעות gather
        def _pick_best_triplet(map_3):  # map_3: [B, na_tot, 3, Dz, Hy, Wx]
            Bx, A, C, D, H, W = map_3.shape
            t = map_3.permute(0,1,3,4,5,2).contiguous().view(Bx, A*D*H*W, C)  # [B, N, 3]
            idx = lin_best.view(Bx, 1, 1).expand(Bx, 1, C)                    # [B,1,3]
            return torch.gather(t, dim=1, index=idx).squeeze(1)               # [B,3]

        pred_center_vec = _pick_best_triplet(center_map)   # [B,3]
        pred_radius_vec = _pick_best_triplet(radius_map)   # [B,3]
        pred_box_top1   = torch.cat([pred_center_vec, pred_radius_vec], dim=1)  # [B,6]


        # === IoU של הטופ-1 + החלטות has_nodule_detected ===
        if has_nodule.any():
            def _iou_cr(a, b):
                ac, ar = a[:, :3], torch.clamp(a[:, 3:], min=1e-9)
                bc, br = b[:, :3], torch.clamp(b[:, 3:], min=1e-9)
                a_min, a_max = ac - ar, ac + ar
                b_min, b_max = bc - br, bc + br
                inter_min = torch.maximum(a_min, b_min)
                inter_max = torch.minimum(a_max, b_max)
                inter = torch.clamp(inter_max - inter_min, min=0).prod(dim=1)
                va = (2*ar).prod(dim=1); vb = (2*br).prod(dim=1)
                return inter / (va + vb - inter + 1e-9)
    
            iou_top1 = _iou_cr(pred_box_top1, gt_box)  # [B]
            pres = 0.3 if epochnum < 3 else self.pres_thresh
            iou_th_decide = 0.3 if epochnum < 8 else 0.5
            #conf_top1 = conf_or_map[bidx, a_best, z_best, y_best, x_best]
            conf_top1 = conf_all[bidx, a_best, z_best, y_best, x_best]
            conf_global = conf_or_map.max()
            # [B]
            #has_nodule_detected = ((conf_top1 >= pres) & (iou_top1 >= iou_th_decide)).float()
        else:
            iou_top1 = torch.zeros(B, device=x.device)
            pres = 0.3 if epochnum < 3 else self.pres_thresh
            conf_global = conf_or_map.max()
            #conf_top1 = conf_all[bidx, a_best, z_best, y_best, x_best]         # [B]
        has_nodule_detected = (conf_global >= pres).float()
        # === הוספה לפי בקשתך: center_distance ונגזרת nodule_number (B=1) ===
        if has_nodule.any():       
            center_thresh = 0.1  # אפשר להזיז ל-__init__ אם תרצי
            center_distance = torch.norm(pred_center_vec[0] - gt_center[0]).item()  # scalar
            nodule_number_cd = 1 if center_distance < center_thresh else 0
        else:
            center_distance = float('inf')
            nodule_number_cd = 0
    
        # === Losses ===
        classification_loss = regression_loss = objectness_loss = total_loss = None
        if mode in ("train","val") :
              
            def _obj_loss_one(obj_h, pos_h, ign_h):
                bce = F.binary_cross_entropy_with_logits(obj_h, pos_h.float(), reduction='none')
                w = torch.ones_like(bce)
                w[pos_h] = 5.0            # אותו רעיון, אבל לראש h בלבד
                valid = (~ign_h).float()
                return (bce * w * valid).sum() / (w * valid).sum().clamp_min(1.0)
            """
            objectness_loss = (
                  _obj_loss_one(obj_s, pos_s, ign_s)
                + _obj_loss_one(obj_m, pos_m, ign_m)
                + _obj_loss_one(obj_l, pos_l, ign_l)
            ) / 3.0
            """
            objectness_loss = (
                  _obj_loss_one(obj_xs, pos_xs, ign_xs)
                + _obj_loss_one(obj_s,  pos_s,  ign_s)
                + _obj_loss_one(obj_m,  pos_m,  ign_m)
                + _obj_loss_one(obj_l,  pos_l,  ign_l)
            ) / 4.0
     
                  # לכל ראש: יעד = pos_h.any(dim=1) של אותו ראש
            def _cls_loss_one(c_h, pos_h, ign_h):
                pos_cells = pos_h.any(dim=1)                        # [B,D,H,W]
                neg_cells = (~pos_cells) & (~ign_h.any(dim=1))
                valid = (pos_cells | neg_cells)
            
                logits2 = c_h.permute(0,2,3,4,1).reshape(-1, 2)     # [N,2]
                targets = pos_cells.long().reshape(-1)
                valid_m = valid.reshape(-1)
            
                if valid_m.any():
                    pos_n = pos_cells.sum().clamp_min(1).float()
                    neg_n = neg_cells.sum().clamp_min(1).float()
                    w_pos = (neg_n / pos_n).clamp(1.0, 10.0).item()
                    weights = torch.tensor([1.0, w_pos], device=c_h.device, dtype=logits2.dtype)
                    return F.cross_entropy(logits2[valid_m], targets[valid_m], weight=weights, reduction='mean')
                return torch.tensor(0.0, device=c_h.device)
            
            #classification_loss = (
            #      _cls_loss_one(c1, pos_s, ign_s)
            #    + _cls_loss_one(c2, pos_m, ign_m)
            #    + _cls_loss_one(c3, pos_l, ign_l)
            #) / 3.0 
            classification_loss = (
                  _cls_loss_one(c_xs_up, pos_xs, ign_xs)
                + _cls_loss_one(c1,     pos_s,  ign_s)
                + _cls_loss_one(c2,     pos_m,  ign_m)
                + _cls_loss_one(c3,     pos_l,  ign_l)
            ) / 4.0

            # CIoU פר־ראש (אם אין פוזיטיבים בראש מסוים — הוא תורם 0)
            def _box_loss_one(ctr_h, rad_h, pos_h, gt_local):
                Bx = ctr_h.size(0)
                pc = ctr_h.permute(0,2,1,3,4,5).reshape(Bx,3,-1).permute(0,2,1)  # [B,N,3]
                pr = rad_h.permute(0,2,1,3,4,5).reshape(Bx,3,-1).permute(0,2,1)  # [B,N,3]
                pm = pos_h.view(Bx, -1)
                losses = []
                for bi in range(Bx):
                    if pm[bi].any():
                        pb = torch.cat([pc[bi][pm[bi]], pr[bi][pm[bi]]], dim=1)   # [K,6]
                        gb = gt_local[bi].unsqueeze(0).repeat(pb.size(0),1)      # [K,6]
                        #losses.append(ciou_3d_loss(pb, gb, reduction='mean'))
                        losses.append(eapiou_3d_loss_cr(pb, gb))
                    else:
                        losses.append(torch.tensor(0., device=pc.device))
                return torch.stack(losses).mean()
            
            #box_s = _box_loss_one(ctr_s, rad_s, pos_s, gt_box)
            #box_m = _box_loss_one(ctr_m, rad_m, pos_m, gt_box)
            #box_l = _box_loss_one(ctr_l, rad_l, pos_l, gt_box)
            #regression_loss = (box_s + box_m + box_l) / 3.0
            box_xs = _box_loss_one(ctr_xs, rad_xs, pos_xs, gt_box)
            box_s  = _box_loss_one(ctr_s,  rad_s,  pos_s,  gt_box)
            box_m  = _box_loss_one(ctr_m,  rad_m,  pos_m,  gt_box)
            box_l  = _box_loss_one(ctr_l,  rad_l,  pos_l,  gt_box)
            regression_loss = (box_xs + box_s + box_m + box_l) / 4.0
    # ---- Warmup Top-K גלובלי אם אין פוזיטיבים באף ראש (רק באפוקים הראשונים) ----
            if (pos_xs.sum() + pos_s.sum() + pos_m.sum() + pos_l.sum()) == 0 and (gt_box is not None) and (epochnum < 3):
                K = 64
                v = conf_all.view(B, -1)
                topk = torch.topk(v, k=min(K, v.size(1)), dim=1).indices
            
                pc_all = center_map.permute(0,2,1,3,4,5).reshape(B, 3, -1).permute(0,2,1)  # [B,N,3]
                pr_all = radius_map.permute(0,2,1,3,4,5).reshape(B, 3, -1).permute(0,2,1)  # [B,N,3]
    
                losses = []
                for bi in range(B):
                    idx = topk[bi]
                    pb  = torch.cat([pc_all[bi][idx], pr_all[bi][idx]], dim=1)             # [K,6]
                    gb  = gt_box[bi].unsqueeze(0).expand_as(pb)                            # [K,6]
                    #losses.append(ciou_3d_loss(pb, gb, reduction='mean'))
                    losses.append(eapiou_3d_loss_cr(pb, gb))
                regression_loss = torch.stack(losses).mean()

    
            
            #regression_loss = box_loss
            total_loss = objectness_loss + classification_loss + regression_loss

            #total_loss = objectness_loss + classification_loss + self.lambda_box * regression_loss
    
        # פלט מפה מלאה לצפייה/דיבאג
        #full_bbox_map = torch.cat([center_map, radius_map], dim=2)  # [B,na,6,D,H,W]
        full_bbox_map = torch.cat([center_map, radius_map], dim=2)  # [B,na_tot=9,6,D,H,W]

        nodule_number_cd = int(nodule_number_cd > 0)
    
        dice_value = torch.tensor(0.0, device=x.device)
        imap_value = torch.tensor(0.0, device=x.device)
        if has_nodule.any():
            # Dice בין תיבות: 2*|A∩B| / (|A|+|B|)
            ac, ar = pred_box_top1[:, :3], torch.clamp(pred_box_top1[:, 3:], min=1e-9)
            bc, br = gt_box[:, :3],         torch.clamp(gt_box[:, 3:],        min=1e-9)
            a_min, a_max = ac - ar, ac + ar
            b_min, b_max = bc - br, bc + br
            inter_min = torch.maximum(a_min, b_min)
            inter_max = torch.minimum(a_max, b_max)
            inter_dims = torch.clamp(inter_max - inter_min, min=0.0)           # [B,3]
            inter_vol  = (inter_dims[:, 0] * inter_dims[:, 1] * inter_dims[:, 2])  # [B]
            pvol = (2.0 * ar[:, 0]) * (2.0 * ar[:, 1]) * (2.0 * ar[:, 2])      # [B]
            gvol = (2.0 * br[:, 0]) * (2.0 * br[:, 1]) * (2.0 * br[:, 2])      # [B]
            denom = (pvol + gvol).clamp_min(1e-9)
            dice_value = (2.0 * inter_vol / denom).mean()                      # scalar
    
            # iMAP (AP לכל תמונה): AP ע"י דירוג conf של כל התאים מול GT (IoU≥0.5)
            iou_map = torch.cat([iou_xs, iou_s, iou_m, iou_l], dim=1)

            #iou_map = torch.cat([iou_s, iou_m, iou_l], dim=1)

            iou_for_ap_th = 0.5
            gt_pos_mask_ap = (iou_map >= iou_for_ap_th) & y_pres_mask
            scores = conf_all.reshape(B, -1)
            truths = gt_pos_mask_ap.reshape(B, -1)
            ap_list = []
            for bi in range(B):
                s = scores[bi]; t = truths[bi]
                if t.sum() == 0:
                    ap_list.append(torch.tensor(0.0, device=x.device)); continue
                ord_idx = torch.argsort(s, descending=True)
                t_sorted = t[ord_idx].float()
                cum_tp = torch.cumsum(t_sorted, dim=0)
                idxs   = torch.arange(1, t_sorted.numel()+1, device=x.device).float()
                precision_at_k = cum_tp / idxs
                ap = (precision_at_k * t_sorted).sum() / t.sum().float()
                ap_list.append(ap)
            imap_value = torch.stack(ap_list).mean()                           # scalar


        return {
            "loss": total_loss,
            "classification_loss": classification_loss,
            "regression_loss": regression_loss,
            "objectness_loss": objectness_loss,
            "logits": cls_fused,                         # [B,2,D,H,W]
            "bbox": full_bbox_map,                       # [B,na,6,D,H,W] normalized
            "has_nodule_detected": has_nodule_detected,  # לפי Conf+IoU
            "nodule_number_conf_iou": int(has_nodule_detected.sum().item()),  # הישן (מידעוני)
            "center_distance": center_distance,          # float (B=1)
            "nodule_number": nodule_number_cd,           # החדש: לפי center_distance
            "best_pred": pred_box_top1.detach(),         # [B,6]
            "iou_value": iou_top1.detach() ,              # [B]
            "dice_value": dice_value.detach(),   # ← חדש
            "imap_value": imap_value.detach(),   # ← חדש
        }
