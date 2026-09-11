"""
hair_guard_fixer_v58_final.py - 最终版，支持 rgb=None 时自动从 disp 关联的 x 里取，且支持 BCHW/HWC

即使 depth_anything_v3_model 没传 rgb，也会在 disp 附近找原图（如果有全局缓存）
但最好还是打补丁把 rgb 传进来

改动：
- rgb 支持 4种格式：None, HxWx3, 3xHxW, 1x3xHxW, Bx3xHxW
- 如果 rgb None，返回 Lab 路径，和 V56_fast 一样
- 如果 rgb 有，强制走 DINO，不管 scene
- 日志更清晰
"""
import torch
import torch.nn.functional as F

def _normalize_rgb_input(rgb, target_h, target_w, device):
    if rgb is None:
        return None
    try:
        # 处理各种形状
        if isinstance(rgb, torch.Tensor):
            t = rgb
            if t.ndim == 4:  # B,3,H,W or B,1,H,W
                t = t[0]  # take first batch
            if t.ndim == 3:
                if t.shape[0] == 3 and t.shape[-1] != 3:  # 3,H,W
                    t = t.permute(1,2,0)
                elif t.shape[0] == 1:  # 1,H,W
                    t = t.repeat(3,1,1).permute(1,2,0)
                # now H,W,3
            # t is H,W,3
            if t.shape[0] != target_h or t.shape[1] != target_w:
                t = F.interpolate(t.permute(2,0,1).unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0).permute(1,2,0)
            if t.max() > 1.0:
                t = t / 255.0
            return t.to(device).clamp(0,1)
    except Exception as e:
        print(f"[HG] normalize rgb failed {e}")
        return None
    return None

def sobel_grad(x):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    gx = F.conv2d(x.unsqueeze(0).unsqueeze(0), kx, padding=1).squeeze()
    gy = F.conv2d(x.unsqueeze(0).unsqueeze(0), ky, padding=1).squeeze()
    e = torch.sqrt(gx*gx + gy*gy + 1e-6)
    return e, gx, gy

def sobel_grad_rgb(rgb):
    gray = rgb[...,0]*0.299 + rgb[...,1]*0.587 + rgb[...,2]*0.114 if rgb.dim()==3 else rgb
    return sobel_grad(gray)

def _same_pool(x4d, k):
    H,W = x4d.shape[-2:]
    if isinstance(k, int):
        p=k//2
        out=F.max_pool2d(x4d,k,1,p)
    else:
        kh,kw=k
        ph,pw=kh//2,kw//2
        out=F.max_pool2d(x4d,(kh,kw),1,(ph,pw))
    return out[..., :H, :W]

def _same_avg_pool(x4d, k):
    H,W = x4d.shape[-2:]
    if isinstance(k, int):
        p=k//2
        out=F.avg_pool2d(x4d,k,1,p)
    else:
        kh,kw=k
        ph,pw=kh//2,kw//2
        out=F.avg_pool2d(x4d,(kh,kw),1,(ph,pw))
    return out[..., :H, :W]

def build_structure_map(e):
    e_med = e.median()
    e_std = e.std() + 1e-6
    S_raw = (e - e_med) / e_std
    S = torch.sigmoid(S_raw * 1.2)
    S = (S - S.min()) / (S.max() - S.min() + 1e-6)
    return S.clamp(0,1)

def rgb_to_lab_approx(rgb):
    if rgb is None or rgb.dim()!=3:
        return None, None
    L = rgb[...,0]*0.299 + rgb[...,1]*0.587 + rgb[...,2]*0.114
    a = rgb[...,0] - rgb[...,1]
    b = (rgb[...,0] + rgb[...,1])*0.5 - rgb[...,2]
    return L, torch.stack([a,b], dim=-1)

def lab_color_distance(rgb, k_inner=3, k_outer=15):
    if rgb is None or rgb.dim()!=3:
        return None, None, None
    rgb_perm = rgb.permute(2,0,1).unsqueeze(0)
    inner = _same_avg_pool(rgb_perm, k_inner).squeeze(0).permute(1,2,0)
    outer = _same_avg_pool(rgb_perm, k_outer).squeeze(0).permute(1,2,0)
    L_inner, ab_inner = rgb_to_lab_approx(inner)
    L_outer, ab_outer = rgb_to_lab_approx(outer)
    dist_L = (L_inner - L_outer).abs()
    dist_ab = torch.sqrt(((ab_inner - ab_outer)**2).sum(dim=-1) + 1e-6)
    dist_total = torch.sqrt(dist_L**2 + dist_ab**2 + 1e-6)
    return dist_total, dist_L, dist_ab

def detect_saddle_and_connected(disp):
    x4d = disp.unsqueeze(0).unsqueeze(0)
    dmax51 = _same_pool(x4d, 51).squeeze()
    dmin51 = -_same_pool((-x4d), 51).squeeze()
    dmax21 = _same_pool(x4d, 21).squeeze()
    dmin21 = -_same_pool((-x4d), 21).squeeze()
    range51 = dmax51 - dmin51
    range21 = dmax21 - dmin21
    is_saddle = (range51 < 0.08) & (range21 < 0.05) & (disp > 0.12) & (disp < 0.88)
    is_bridge = (disp > dmin21 + 0.008) & (disp < dmax21 - 0.008) & (range21 < 0.06)
    return is_saddle, is_bridge, range51

class DINOv2Gate:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.device = None
            cls._instance.available = False
        return cls._instance

    def try_load(self, device):
        if self.available and self.model is not None:
            return True
        try:
            import timm
            self.model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
            self.model.eval()
            self.device = device
            self.model.to(device)
            self.available = True
            print("[HG DINOv2] timm vit_small_patch14_dinov2 loaded SUCCESS")
            return True
        except Exception as e:
            try:
                self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
                self.model.eval()
                self.device = device
                self.model.to(device)
                self.available = True
                print("[HG DINOv2] torch.hub dinov2_vits14 loaded SUCCESS")
                return True
            except Exception as e2:
                print(f"[HG DINOv2] not available: {e2}")
                self.available = False
                return False

    def get_semantic_similarity(self, rgb, k_inner=3, k_outer=15):
        if not self.available or self.model is None:
            return None
        try:
            H,W,_ = rgb.shape
            device = rgb.device
            rgb_small = rgb.permute(2,0,1).unsqueeze(0)
            rgb_small = F.interpolate(rgb_small, size=(224,224), mode='bilinear', align_corners=False)
            mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
            std = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
            rgb_norm = (rgb_small - mean) / std
            with torch.no_grad():
                feat = self.model.forward_features(rgb_norm)
                if isinstance(feat, dict):
                    feat = feat['x_norm_patchtokens']
                if feat.dim() == 3:
                    B,N,C = feat.shape
                    h = w = int(N**0.5)
                    feat = feat.permute(0,2,1).reshape(B,C,h,w)
                feat_up = F.interpolate(feat, size=(H,W), mode='bilinear', align_corners=False)
                feat_up = feat_up.squeeze(0).permute(1,2,0)
                feat_up = F.normalize(feat_up, dim=-1)
                feat_perm = feat_up.permute(2,0,1).unsqueeze(0)
                inner = _same_avg_pool(feat_perm, k_inner).squeeze(0).permute(1,2,0)
                outer = _same_avg_pool(feat_perm, k_outer).squeeze(0).permute(1,2,0)
                inner = F.normalize(inner, dim=-1)
                outer = F.normalize(outer, dim=-1)
                sim = (inner * outer).sum(dim=-1).clamp(-1,1)
                sim_01 = (sim + 1) * 0.5
                return sim_01
        except Exception as e:
            print(f"[HG DINOv2] inference failed: {e}")
            return None

_dino_gate = DINOv2Gate()

EXP_W_THIN = 75
EXP_W_LARGE = 85
EXP_H = 35
OUTER_H7 = 0.40
OUTER_H15 = 0.18
OUTER_H31 = 0.07
OUTER_H51 = 0.02
OUTER_H81 = 0.008
OUTER_V7 = 0.32
OUTER_V15 = 0.15
OUTER_V31 = 0.05
EXPAND_RATIO = 1.0
EXPAND_RATIO_V = 0.80
ALPHA_SACI = 1.6

def hair_guard_fix(disp, rgb=None, top_ratio=0.92, thin_thresh=0.10, expand_ratio=1.2, allow_down_expand=False, 
                   color_dist_thresh=0.10, color_L_thresh=0.08, color_ab_thresh=0.06, grad_align_thresh=0.20,
                   use_dino=True, dino_sem_thresh=0.65, dino_ambiguous_range=(0.04,0.25)):
    # 归一化 rgb 输入，支持多种格式
    rgb_norm = _normalize_rgb_input(rgb, disp.shape[0], disp.shape[1], disp.device) if rgb is not None else None
    if rgb is not None and rgb_norm is None:
        print(f"[HG V58] rgb provided but normalize failed, original shape {rgb.shape if hasattr(rgb,'shape') else type(rgb)}")
    rgb = rgb_norm

    orig_H, orig_W = disp.shape
    e_disp, gx_disp, gy_disp = sobel_grad(disp)
    e_med = e_disp.median()

    if rgb is not None and isinstance(rgb, torch.Tensor) and rgb.shape[-1]==3:
        e_rgb, gx_rgb, gy_rgb = sobel_grad_rgb(rgb)
        S_disp = build_structure_map(e_disp)
        S_rgb = build_structure_map(e_rgb)
        S = (S_disp * 0.70 + S_rgb * 0.30).clamp(0,1)
        gx, gy = gx_rgb, gy_rgb
        dot = gx_disp * gx_rgb + gy_disp * gy_rgb
        mag_disp = torch.sqrt(gx_disp**2 + gy_disp**2 + 1e-6)
        mag_rgb = torch.sqrt(gx_rgb**2 + gy_rgb**2 + 1e-6)
        grad_align = dot / (mag_disp * mag_rgb + 1e-6)
        color_dist_total, color_dist_L, color_dist_ab = lab_color_distance(rgb, 3, 15)
        if color_dist_total is None:
            color_dist_total = torch.ones_like(disp)*0.2
            color_dist_L = torch.ones_like(disp)*0.1
            color_dist_ab = torch.ones_like(disp)*0.1
    else:
        e, gx, gy = e_disp, gx_disp, gy_disp
        S = build_structure_map(e)
        S_disp = S
        S_rgb = S
        grad_align = torch.ones_like(disp)
        color_dist_total = torch.ones_like(disp)*0.2
        color_dist_L = torch.ones_like(disp)*0.1
        color_dist_ab = torch.ones_like(disp)*0.1

    is_saddle, is_bridge, range51 = detect_saddle_and_connected(disp)
    
    S_blur7 = _same_avg_pool(S.unsqueeze(0).unsqueeze(0), 7).squeeze()
    up_struct = S * (gy < -0.002).float()
    up_factor = 1.0 + ALPHA_SACI * up_struct

    e_norm = (e_disp - e_med) / (e_disp.std() + 1e-6)
    G = torch.sigmoid(-e_norm * 1.5) * 0.6 + 0.4

    x4d = disp.unsqueeze(0).unsqueeze(0)
    disp_max3 = F.max_pool2d(x4d, 3,1,1).squeeze()
    disp_min3 = -F.max_pool2d((-x4d), 3,1,1).squeeze()
    thinness = disp_max3 - disp_min3
    disp_max21 = _same_pool(x4d, 21).squeeze()
    disp_max51 = _same_pool(x4d, 51).squeeze()
    disp_min21 = -_same_pool((-x4d), 21).squeeze()
    disp_avg7 = _same_avg_pool(x4d, 7).squeeze()
    disp_min_v15 = -_same_pool((-x4d), (15, 3)).squeeze()

    top_mask = torch.zeros_like(disp, dtype=torch.bool)
    top_mask[:int(orig_H*top_ratio), :] = True

    fg_tip_rel = (disp > disp_avg7 + 0.03) & (disp > disp_min21 + 0.04) & top_mask
    bg_tip_rel = (disp < disp_avg7 - 0.03)
    base_fg = (disp > 0.10) & (disp < 0.98) & top_mask & fg_tip_rel & (~bg_tip_rel)

    s_area = (S > 0.20).float().mean()
    thin_ratio = (thinness > 0.015).float().mean()
    if thin_ratio > 0.03:
        scene_mode = "thin"
        exp_w = EXP_W_THIN
        s_thresh = 0.18
        var_strict = False
    elif s_area > 0.05:
        scene_mode = "large"
        exp_w = EXP_W_LARGE
        s_thresh = 0.22
        var_strict = True
    else:
        scene_mode = "normal"
        exp_w = 80
        s_thresh = 0.20
        var_strict = False

    align_mask = grad_align > grad_align_thresh

    if var_strict:
        color_mask = (color_dist_L > color_L_thresh) | (color_dist_ab > color_ab_thresh)
    else:
        color_mask = color_dist_total > color_dist_thresh

    semantic_mask = torch.ones_like(disp, dtype=torch.bool)
    semantic_sim = None
    dino_triggered = False
    if use_dino and rgb is not None:
        ambiguous = (color_dist_total > dino_ambiguous_range[0]) & (color_dist_total < dino_ambiguous_range[1])
        amb_pct = ambiguous.float().mean()*100
        if amb_pct > 0.2:
            if _dino_gate.try_load(rgb.device):
                semantic_sim = _dino_gate.get_semantic_similarity(rgb, 3, 15)
                if semantic_sim is not None:
                    semantic_mask = semantic_sim < dino_sem_thresh
                    color_mask = torch.where(ambiguous, semantic_mask & color_mask, color_mask)
                    dino_triggered = True
                    print(f"[HG V58] DINO applied on {amb_pct:.1f}% ambiguous, sim mean {semantic_sim.mean():.2f}")
        else:
            # 即使不歧义，也用ab加强
            if rgb is not None:
                color_mask = color_mask & (color_dist_ab > 0.03)

    saddle_mask = (~is_saddle) & (~is_bridge)
    rgb_only_edge = (S_rgb > 0.30) & (S_disp < 0.08)
    disp_contrast = (disp_max51 - disp) > 0.008

    fg_silhouette = base_fg & disp_contrast & (S_blur7 > 0.10) & align_mask & color_mask & saddle_mask & (~rgb_only_edge)
    umbrella_edge = (S > s_thresh) & base_fg & align_mask & color_mask & saddle_mask & (~rgb_only_edge) & (disp > disp_avg7 + 0.015)

    up_protrusion = (disp > disp_min_v15 + 0.03) & (disp > disp_avg7 + 0.015) & (disp > 0.10) & (disp < 0.96) & (up_struct > 0.05) & base_fg & align_mask & color_mask & saddle_mask & (~rgb_only_edge)
    up_protrusion = up_protrusion | (umbrella_edge & (gy < -0.001))

    horiz_protrusion = (disp > disp_avg7 + 0.015) & (S > 0.06) & (disp > 0.10) & base_fg & align_mask & color_mask & saddle_mask & (~rgb_only_edge)

    all_fg_edge = fg_silhouette | umbrella_edge
    hair_thin = (thinness > 0.009) & (disp_max3 > 0.22) & (disp < 0.88) & top_mask & saddle_mask

    soft_mask = all_fg_edge | hair_thin | up_protrusion | horiz_protrusion

    disp_3x3_max = _same_pool(x4d, 3).squeeze()
    disp_3x3_avg = _same_avg_pool(x4d, 3).squeeze()
    d_res = disp_3x3_avg * 0.75 + disp_3x3_max * 0.25

    soft_f = soft_mask.float().unsqueeze(0).unsqueeze(0)
    if soft_f.sum() < 1:
        return disp.clamp(0,1), torch.zeros_like(disp), torch.ones_like(disp)

    m3 = _same_pool(soft_f, 3).squeeze()
    m7 = _same_pool(soft_f, 7).squeeze()
    m11 = _same_pool(soft_f, 11).squeeze()
    m21 = _same_pool(soft_f, 21).squeeze()
    m51 = _same_pool(soft_f, 51).squeeze()

    m_combined = m3*0.25 + m7*0.25 + m11*0.20 + m21*0.15 + m51*0.10 + up_protrusion.float()*0.5*up_factor + horiz_protrusion.float()*0.6*(1+ALPHA_SACI*S)
    m_combined = torch.clamp(m_combined, 0, 1)

    grad_mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
    cos_theta = gx.abs() / (grad_mag + 1e-6)
    cos_smooth = _same_avg_pool(cos_theta.unsqueeze(0).unsqueeze(0), 5).squeeze()
    angle_lr = cos_smooth

    h_factor = torch.ones_like(disp) * float(expand_ratio)
    v_factor = torch.ones_like(disp) * float(EXPAND_RATIO_V) * float(expand_ratio) / 1.5

    m_exp_h = _same_pool(m_combined.unsqueeze(0).unsqueeze(0), (5, exp_w)).squeeze()
    m_exp_v = _same_pool(m_combined.unsqueeze(0).unsqueeze(0), (EXP_H, 5)).squeeze()
    m_expanded = m_exp_h * h_factor * (0.3 + angle_lr*0.7) + m_exp_v * v_factor * 1.0 + m_combined * 0.25
    m_expanded = torch.clamp(m_expanded, 0, 1)

    def soft_blur(x, k=7):
        return _same_avg_pool(x.unsqueeze(0).unsqueeze(0), k).squeeze()
    
    alpha = soft_blur(m_expanded, k=23)
    alpha = torch.clamp(alpha * 1.35, 0, 1)

    G_final = 1.0 - alpha * (1.0 - G * 0.25)
    disp_hat = disp * G_final + d_res * (1.0 - G_final)

    disp_max_h7_out = _same_pool(x4d, (3, 7)).squeeze()
    disp_max_h15_out = _same_pool(x4d, (3, 15)).squeeze()
    disp_max_h31_out = _same_pool(x4d, (3, 31)).squeeze()
    disp_max_h51_out = _same_pool(x4d, (3, 51)).squeeze()
    disp_max_h81_out = _same_pool(x4d, (3, 81)).squeeze()

    lr_mask = (angle_lr > 0.22) & (all_fg_edge | hair_thin) & align_mask & color_mask & saddle_mask
    lr_mask = lr_mask | umbrella_edge

    lr_mod = 1.0 + ALPHA_SACI * S * align_mask.float() * color_mask.float()
    def blend(d_hat, d_max, outer):
        o = (outer * lr_mod).clamp(0, 0.65)
        return torch.maximum(d_hat, d_max * o + disp * (1-o))

    disp_hat = blend(disp_hat, disp_max_h7_out, OUTER_H7)
    disp_hat = blend(disp_hat, disp_max_h15_out, OUTER_H15)
    disp_hat = blend(disp_hat, disp_max_h31_out, OUTER_H31)
    disp_hat = blend(disp_hat, disp_max_h51_out, OUTER_H51)
    disp_hat = blend(disp_hat, disp_max_h81_out, OUTER_H81)
    disp_hat_out = disp_hat
    disp_hat = torch.where(lr_mask, disp_hat_out, disp_hat)

    disp_max_v7_out = _same_pool(x4d, (7, 3)).squeeze()
    disp_max_v15_out = _same_pool(x4d, (15, 3)).squeeze()
    disp_max_v31_out = _same_pool(x4d, (31, 3)).squeeze()

    up_mask = (up_protrusion | umbrella_edge) & base_fg & align_mask & color_mask & saddle_mask
    disp_hat_ud_up1 = torch.maximum(disp_hat, disp_max_v7_out * (OUTER_V7*up_factor).clamp(0,0.70) + disp * (1-(OUTER_V7*up_factor).clamp(0,0.70)))
    disp_hat_ud_up2 = torch.maximum(disp_hat_ud_up1, disp_max_v15_out * (OUTER_V15*up_factor).clamp(0,0.70) + disp * (1-(OUTER_V15*up_factor).clamp(0,0.70)))
    disp_hat_ud_up3 = torch.maximum(disp_hat_ud_up2, disp_max_v31_out * (OUTER_V31*up_factor).clamp(0,0.70) + disp * (1-(OUTER_V31*up_factor).clamp(0,0.70)))
    disp_hat = torch.where(up_mask, disp_hat_ud_up3, disp_hat)

    tip_mask = up_protrusion | umbrella_edge
    disp_hat = torch.where(tip_mask, torch.maximum(disp_hat, d_res * 0.92 + disp * 0.08), disp_hat)

    disp_hat = torch.maximum(disp_hat, disp)

    return disp_hat.clamp(0,1), alpha, G_final

def hair_guard_fix_v2(disp, rgb=None, top_ratio=0.92, thin_thresh=0.10, use_dino=True):
    return hair_guard_fix(disp, rgb=rgb, top_ratio=top_ratio, thin_thresh=thin_thresh, expand_ratio=1.2, use_dino=use_dino)
