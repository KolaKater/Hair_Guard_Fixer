"""
hair_guard_fixer.py - V46 左右背景模糊不扩修复 + 上下拆分下扩弱版，V44基础上纵向0.70->0.85/0.55->0.70 +31px +gy阈值放宽，之前还是偏大：不需要尖角，突出来就算
- 你说的对，背景模糊时不需要尖角判断，只要前景突出来就行
- 树叶/突起不一定尖锐，只要比周围亮+gy大就选
"""
import torch
import torch.nn.functional as F

def sobel_grad(x):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    gx = F.conv2d(x.unsqueeze(0).unsqueeze(0), kx, padding=1).squeeze()
    gy = F.conv2d(x.unsqueeze(0).unsqueeze(0), ky, padding=1).squeeze()
    e = torch.sqrt(gx*gx + gy*gy + 1e-6)
    return e, gx, gy

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

EXP_W = 121
EXP_H = 51
OUTER_H7 = 0.80         # 0.75->0.80 左右更硬，解决男人红衣边缘不扩
OUTER_H15 = 0.60        # 0.55->0.60
OUTER_H31 = 0.35        # 0.30->0.35
OUTER_H51 = 0.18        # 0.15->0.18
OUTER_V7 = 0.85  # 上保持硬
OUTER_V15 = 0.70
OUTER_V31 = 0.40
OUTER_V7_DOWN = 0.55  # 下扩弱，防止女人手把后面裙子带变形
OUTER_V15_DOWN = 0.35
OUTER_V31_DOWN = 0.18
EXPAND_RATIO = 1.3      # 1.5->1.3
EXPAND_RATIO_V = 1.1

def hair_guard_fix(disp, top_ratio=0.90, thin_thresh=0.10, expand_ratio=1.5):
    orig_H, orig_W = disp.shape
    e, gx, gy = sobel_grad(disp)
    e_med = e.median()
    e_norm = (e - e_med) / (e.std() + 1e-6)
    G = torch.sigmoid(-e_norm * 1.5) * 0.6 + 0.4

    x4d = disp.unsqueeze(0).unsqueeze(0)
    disp_max3 = F.max_pool2d(x4d, 3,1,1).squeeze()
    disp_min3 = -F.max_pool2d((-x4d), 3,1,1).squeeze()
    thinness = disp_max3 - disp_min3
    disp_max21 = _same_pool(x4d, 21).squeeze()
    disp_min21 = -_same_pool((-x4d), 21).squeeze()
    disp_avg7 = _same_avg_pool(x4d, 7).squeeze()
    disp_avg15 = _same_avg_pool(x4d, 15).squeeze()
    disp_min_v15 = -_same_pool((-x4d), (15, 3)).squeeze()
    disp_min_v7 = -_same_pool((-x4d), (7, 3)).squeeze()
    disp_max_v15 = _same_pool(x4d, (15, 3)).squeeze()
    disp_max_v7 = _same_pool(x4d, (7, 3)).squeeze()

    e_blur7 = _same_avg_pool(e.unsqueeze(0).unsqueeze(0), 7).squeeze()
    e_blur15 = _same_avg_pool(e.unsqueeze(0).unsqueeze(0), 15).squeeze()
    e_blur31 = _same_avg_pool(e.unsqueeze(0).unsqueeze(0), 31).squeeze()
    e_blur_lr = _same_avg_pool(e.unsqueeze(0).unsqueeze(0), (3, 7)).squeeze()  # 左右专用3x7
    e_blur_lr_5 = _same_avg_pool(e.unsqueeze(0).unsqueeze(0), (3, 5)).squeeze() # 左右3x5

    # 背景模糊判断：放宽，e小就是模糊，阈值1.30->1.60更多算模糊
    bg_is_blurry = e_blur15 < e_med * 1.60
    bg_is_blurry_strong = e_blur15 < e_med * 1.20
    # 左右专用模糊判断：放宽 1.40->1.70, 1.80->1.50
    bg_is_blurry_lr = e_blur_lr < e_med * 1.70  # 之前1.40，现在1.70更多算模糊
    bg_is_clear_lr = e_blur_lr_5 > e_med * 1.50  # 之前1.80，现在1.50更松，清晰条件放宽

    top_mask = torch.zeros_like(disp, dtype=torch.bool)
    top_mask[:int(orig_H*top_ratio), :] = True

    eff_thresh = thin_thresh * 0.22

    missed_visible = (disp < 0.58) & (disp_max3 > 0.48) & (thinness > eff_thresh) & top_mask
    flicker_bright = (disp > 0.40) & (thinness > eff_thresh*0.5) & top_mask

    # 前景相对：突出来比周围亮，不一定尖
    fg_tip_rel = (disp > disp_avg7 + 0.03) & (disp > disp_min21 + 0.04) & top_mask  # 0.04/0.06->0.03/0.04更松
    bg_tip_rel = (disp < disp_avg7 - 0.03)

    # 基础前景边：左右模糊条件放宽，e阈值0.05->0.03更松
    fg_silhouette = (disp > 0.18) & (disp < 0.98) & ((disp_max21 - disp) > 0.010) & (e_blur7 > e_med * 0.03) & top_mask & fg_tip_rel & (~bg_tip_rel)
    # 左右专用前景边：背景模糊时e更松
    fg_silhouette_lr = (disp > 0.18) & (disp < 0.98) & ((disp_max21 - disp) > 0.008) & (e_blur_lr > e_med * 0.02) & top_mask & fg_tip_rel & (~bg_tip_rel) & bg_is_blurry_lr
    fg_silhouette = fg_silhouette | fg_silhouette_lr

    # ===== 模糊背景通用上下突起，不需要尖角 =====
    # 只要前景突出来+gy大，就选，不要求尖锐
    # 向上突起：当前比下面邻域的最小高，且上面背景模糊
    up_protrusion = (disp > disp_min_v15 + 0.06) & (disp > disp_avg7 + 0.03) & (disp > 0.20) & (disp < 0.96) & (gy.abs() > gx.abs() * 0.35) & top_mask & fg_tip_rel & (~bg_tip_rel)
    down_protrusion = (disp > disp_min_v15 + 0.06) & (disp > disp_avg7 + 0.03) & (disp > 0.20) & (disp < 0.96) & (gy.abs() > gx.abs() * 0.35) & top_mask & fg_tip_rel & (~bg_tip_rel)
    # 模糊背景中，只要突起，e可以很低
    up_protrusion_blurry = up_protrusion & (bg_is_blurry | (e_blur7 < e_med * 0.80))
    down_protrusion_blurry = down_protrusion & (bg_is_blurry | (e_blur7 < e_med * 0.80))
    # 水平边突起通用
    horiz_protrusion = (disp > disp_avg7 + 0.035) & (disp > disp_min_v7 + 0.05) & (gy.abs() > gx.abs() * 0.30) & (disp > 0.20) & top_mask & fg_tip_rel & (~bg_tip_rel)

    # 叶子尖还是保留，但阈值更低
    corner_tip = (disp < 0.80) & (disp > disp_avg7 + 0.04) & top_mask & fg_tip_rel

    all_fg_edge = fg_silhouette | missed_visible | flicker_bright
    hair_thin = (thinness > 0.018) & (disp_max3 > 0.36) & (disp < 0.88) & top_mask

    # 核心：上下突起不需要尖角，突出来就算
    soft_mask = all_fg_edge | hair_thin | up_protrusion | down_protrusion | horiz_protrusion | up_protrusion_blurry | down_protrusion_blurry | corner_tip

    disp_3x3 = _same_pool(x4d, 3).squeeze()
    d_res = disp_3x3

    soft_f = soft_mask.float().unsqueeze(0).unsqueeze(0)
    if soft_f.sum() < 1:
        return disp.clamp(0,1), torch.zeros_like(disp), torch.ones_like(disp)

    m3 = _same_pool(soft_f, 3).squeeze()
    m7 = _same_pool(soft_f, 7).squeeze()
    m11 = _same_pool(soft_f, 11).squeeze()
    m21 = _same_pool(soft_f, 21).squeeze()
    m_combined = m3 * 0.30 + m7 * 0.30 + m11 * 0.20 + m21 * 0.10 + up_protrusion.float()*0.5 + down_protrusion.float()*0.5 + horiz_protrusion.float()*0.6
    m_combined = torch.clamp(m_combined, 0, 1)

    grad_mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
    cos_theta = gx.abs() / (grad_mag + 1e-6)
    cos_smooth = _same_avg_pool(cos_theta.unsqueeze(0).unsqueeze(0), 5).squeeze()
    angle_lr = cos_smooth

    exp_w = EXP_W
    exp_h = EXP_H
    h_factor = torch.ones_like(disp) * float(expand_ratio)
    v_factor = torch.ones_like(disp) * float(EXPAND_RATIO_V) * float(expand_ratio) / 1.5

    m_exp_h = _same_pool(m_combined.unsqueeze(0).unsqueeze(0), (5, exp_w)).squeeze()
    m_exp_v = _same_pool(m_combined.unsqueeze(0).unsqueeze(0), (exp_h, 5)).squeeze()
    # 左右向2边扩权重加大，0.5+angle*0.5 -> 0.3+angle*0.7，垂直边扩更远
    m_expanded = m_exp_h * h_factor * (0.3 + angle_lr*0.7) + m_exp_v * v_factor * 1.0 + m_combined * 0.25
    m_expanded = torch.clamp(m_expanded, 0, 1)

    def soft_blur(x, k=7):
        return _same_avg_pool(x.unsqueeze(0).unsqueeze(0), k).squeeze()
    
    alpha = soft_blur(m_expanded, k=15)
    alpha = torch.clamp(alpha * 3.5, 0, 1)  # 4.2->3.5再回调

    G_final = 1.0 - alpha * (1.0 - G * 0.25)
    disp_hat = disp * G_final + d_res * (1.0 - G_final)

    # 左右：V46 修复背景模糊不扩 - 加入hair_thin + blurry判断
    disp_max_h7_out = _same_pool(x4d, (3, 7)).squeeze()
    disp_max_h15_out = _same_pool(x4d, (3, 15)).squeeze()
    disp_max_h31_out = _same_pool(x4d, (3, 31)).squeeze()
    disp_max_h51_out = _same_pool(x4d, (3, 51)).squeeze()
    # 之前只 all_fg_edge，红衣男人这种大块前景的 thinness>0.018 但不是 all_fg_edge，漏了
    lr_mask = (angle_lr > 0.28) & (all_fg_edge | hair_thin | fg_silhouette_lr)  # 0.35->0.28 + hair_thin
    # 背景模糊时额外再加一层左右扩，即使angle不够
    lr_mask_blurry = bg_is_blurry_lr & (disp > 0.18) & (disp < 0.98) & top_mask & fg_tip_rel
    lr_mask = lr_mask | lr_mask_blurry
    disp_hat_out1 = torch.maximum(disp_hat, disp_max_h7_out * OUTER_H7 + disp * (1-OUTER_H7))
    disp_hat_out2 = torch.maximum(disp_hat_out1, disp_max_h15_out * OUTER_H15 + disp * (1-OUTER_H15))
    disp_hat_out3 = torch.maximum(disp_hat_out2, disp_max_h31_out * OUTER_H31 + disp * (1-OUTER_H31))
    disp_hat_out4 = torch.maximum(disp_hat_out3, disp_max_h51_out * OUTER_H51 + disp * (1-OUTER_H51))
    disp_hat = torch.where(lr_mask, disp_hat_out4, disp_hat)

    # 上下：V46 拆分上下，解决女人右手向下把后面裙子带变形
    disp_max_v7_out = _same_pool(x4d, (7, 3)).squeeze()
    disp_max_v15_out = _same_pool(x4d, (15, 3)).squeeze()
    disp_max_v31_out = _same_pool(x4d, (31, 3)).squeeze()
    # 上：保持硬 0.85/0.70/0.40
    up_mask = up_protrusion & (disp > 0.18) & (disp < 0.98) & top_mask & fg_tip_rel & (~bg_tip_rel)
    # 上要下面是背景才扩，gy<0向上
    up_mask = up_mask & (gy < -0.01)  # 向上边缘
    disp_hat_ud_up1 = torch.maximum(disp_hat, disp_max_v7_out * OUTER_V7 + disp * (1-OUTER_V7))
    disp_hat_ud_up2 = torch.maximum(disp_hat_ud_up1, disp_max_v15_out * OUTER_V15 + disp * (1-OUTER_V15))
    disp_hat_ud_up3 = torch.maximum(disp_hat_ud_up2, disp_max_v31_out * OUTER_V31 + disp * (1-OUTER_V31))
    disp_hat = torch.where(up_mask, disp_hat_ud_up3, disp_hat)

    # 下：弱很多 0.55/0.35/0.18，且要求下面是真背景（disp_min远小于当前），否则是另一个人的裙子不扩
    # down_protrusion 已经要求 disp > disp_min_v15+0.06，即下面比当前暗很多才是真背景
    down_mask = down_protrusion & (disp > 0.18) & (disp < 0.98) & top_mask & fg_tip_rel & (~bg_tip_rel) & (gy > 0.01)
    disp_hat_ud_down1 = torch.maximum(disp_hat, disp_max_v7_out * OUTER_V7_DOWN + disp * (1-OUTER_V7_DOWN))
    disp_hat_ud_down2 = torch.maximum(disp_hat_ud_down1, disp_max_v15_out * OUTER_V15_DOWN + disp * (1-OUTER_V15_DOWN))
    disp_hat_ud_down3 = torch.maximum(disp_hat_ud_down2, disp_max_v31_out * OUTER_V31_DOWN + disp * (1-OUTER_V31_DOWN))
    disp_hat = torch.where(down_mask, disp_hat_ud_down3, disp_hat)

    # 突起单独硬扩：上下突起直接用max，不再软混合
    tip_mask = up_protrusion | down_protrusion
    # 纵向突起硬扩：直接用3x3 max硬填充，替代之前的alpha软混
    disp_hat = torch.where(tip_mask, torch.maximum(disp_hat, d_res * 0.92 + disp * 0.08), disp_hat)
    # 横向突起保持软
    tip_mask_h = horiz_protrusion
    disp_hat = torch.where(tip_mask_h, disp * (1-alpha*0.92) + d_res * (alpha*0.92), disp_hat)

    disp_hat = torch.maximum(disp_hat, disp)

    return disp_hat.clamp(0,1), alpha, G_final

def hair_guard_fix_v2(disp, rgb=None, top_ratio=0.90, thin_thresh=0.10):
    return hair_guard_fix(disp, top_ratio=top_ratio, thin_thresh=thin_thresh, expand_ratio=1.5)
