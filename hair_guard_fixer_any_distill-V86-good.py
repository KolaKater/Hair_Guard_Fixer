"""
hair_guard_fixer_any_distill V86 - 纯黑加强+平滑线保护版
V85问题：纯黑还是不够厚，军人飞机平滑线被Hair Guard变形
V86：
1. 纯黑加强：brightness<0.10纯黑 boost 1.60 (原来1.25)，<0.15极黑1.45，OUTER_H7 0.28->0.32纯黑区，exp_w 52->56纯黑区多扩2px
2. 平滑线保护：直升机机库平滑线 e小但angle一致，angle方差小=平滑线，检测：angle_var = avg(angle^2)-avg(angle)^2，var<0.02=直线，smooth_line处outer*0.65，不变形
3. 飞机门窗等大面积平坦前景：thinness<0.008 & disp>0.3 & e<med*0.6 = flat_fg，平坦处不进hair_thin，不扩
4. 手指黑背景：前景模糊但背景也模糊时多扩，fg_blur & bg_blur时outer*1.2
"""
import torch
import torch.nn.functional as F

def _normalize_rgb_input(rgb, target_h, target_w, device):
    if rgb is None:
        return None
    try:
        if isinstance(rgb, torch.Tensor):
            t = rgb
            if t.ndim == 4:
                t = t[0]
            if t.ndim == 3:
                if t.shape[0] == 3 and t.shape[-1] != 3:
                    t = t.permute(1,2,0)
                elif t.shape[0] == 1:
                    t = t.repeat(3,1,1).permute(1,2,0)
            if t.shape[0] != target_h or t.shape[1] != target_w:
                t = F.interpolate(t.permute(2,0,1).unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0).permute(1,2,0)
            if t.max() > 1.0:
                t = t / 255.0
            return t.to(device).clamp(0,1)
    except Exception:
        return None
    return None

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

EXP_W_BASE = 56
EXP_H = 18
OUTER_H7 = 0.32
OUTER_H15 = 0.18
OUTER_H31 = 0.08
OUTER_H51 = 0.03
OUTER_V7 = 0.10
OUTER_V15 = 0.05
OUTER_V31 = 0.04
OUTER_V7_DOWN = 0.01
OUTER_V15_DOWN = 0.01
OUTER_V31_DOWN = 0.01
EXPAND_RATIO = 1.04
EXPAND_RATIO_V = 0.75

def hair_guard_fix(disp, rgb=None, top_ratio=1.0, thin_thresh=0.10, expand_ratio=1.5, **kwargs):
    orig_H, orig_W = disp.shape
    exp_w_adapt = max(44, min(60, orig_W // 13))
    
    rgb_norm = _normalize_rgb_input(rgb, orig_H, orig_W, disp.device) if rgb is not None else None
    
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
    disp_max51 = _same_pool(x4d, 51).squeeze()
    disp_min51 = -_same_pool((-x4d), 51).squeeze()

    bg_mask = (disp < disp_avg7 - 0.04)
    e_bg = e * bg_mask.float()
    bg_count = _same_avg_pool(bg_mask.float().unsqueeze(0).unsqueeze(0), (3, 21)).squeeze()
    e_blur_bg_lr = _same_avg_pool(e_bg.unsqueeze(0).unsqueeze(0), (3, 21)).squeeze() / (bg_count + 1e-6)
    bg_is_blurry_lr_range_disp = (e_blur_bg_lr < 0.050) & (bg_count > 0.02)
    bg_is_very_blurry_lr_range_disp = (e_blur_bg_lr < 0.025) & (bg_count > 0.02)

    e_blur7 = _same_avg_pool(e.unsqueeze(0).unsqueeze(0), 7).squeeze()
    e_blur15 = _same_avg_pool(e.unsqueeze(0).unsqueeze(0), 15).squeeze()
    e_blur_lr = _same_avg_pool(e.unsqueeze(0).unsqueeze(0), (3, 7)).squeeze()
    bg_is_blurry = e_blur15 < e_med * 1.60
    bg_is_blurry_lr_global = e_blur_lr < e_med * 1.85

    # 平滑线检测：angle方差小=直线，飞机机库门
    grad_mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
    cos_theta = gx.abs() / (grad_mag + 1e-6)
    cos_smooth = _same_avg_pool(cos_theta.unsqueeze(0).unsqueeze(0), 5).squeeze()
    angle_lr = cos_smooth
    angle_sq = _same_avg_pool((cos_theta*cos_theta).unsqueeze(0).unsqueeze(0), 7).squeeze()
    angle_mean = _same_avg_pool(cos_theta.unsqueeze(0).unsqueeze(0), 7).squeeze()
    angle_var = angle_sq - angle_mean*angle_mean
    is_smooth_line = (angle_var < 0.020) & (e < e_med*0.90) & (thinness < 0.015)  # 平滑线
    is_flat_fg = (thinness < 0.008) & (disp > 0.25) & (e < e_med*0.60) & (~bg_mask)  # 飞机机身平坦

    if rgb_norm is not None:
        brightness = rgb_norm[...,0]*0.299 + rgb_norm[...,1]*0.587 + rgb_norm[...,2]*0.114
        e_rgb, gx_rgb, gy_rgb = sobel_grad(brightness)
        e_rgb_bg = e_rgb * bg_mask.float()
        e_rgb_blur_bg_lr = _same_avg_pool(e_rgb_bg.unsqueeze(0).unsqueeze(0), (3, 21)).squeeze() / (bg_count + 1e-6)
        bg_is_blurry_lr_range_rgb = (e_rgb_blur_bg_lr < 0.040) & (bg_count > 0.02)
        bg_is_very_blurry_lr_range_rgb = (e_rgb_blur_bg_lr < 0.020) & (bg_count > 0.02)
        bg_brightness = _same_avg_pool((brightness * bg_mask.float()).unsqueeze(0).unsqueeze(0), (3,21)).squeeze() / (bg_count+1e-6)
        # V86纯黑加强
        is_pure_black = brightness < 0.10  # 0.12->0.10更纯
        is_extreme_dark_bg = (bg_brightness < 0.20) & (bg_count > 0.01)
        is_deep_dark_bg = (bg_brightness < 0.28) & (bg_count > 0.01) | (brightness < 0.20)
        is_mid_dark_bg = (bg_brightness < 0.45) & (bg_count > 0.01) & (~is_deep_dark_bg) & (~is_extreme_dark_bg) | ((brightness < 0.35) & (brightness >= 0.20))
        is_very_deep_dark = brightness < 0.15
        rgb_fg = _same_avg_pool(rgb_norm.permute(2,0,1).unsqueeze(0), 3).squeeze(0).permute(1,2,0)
        rgb_bg = _same_avg_pool((rgb_norm * bg_mask.float().unsqueeze(-1)).permute(2,0,1).unsqueeze(0), 21).squeeze(0).permute(1,2,0) / (bg_count.unsqueeze(-1)+1e-6)
        color_dist = torch.sqrt(((rgb_fg - rgb_bg)**2).sum(dim=-1) + 1e-6)
        color_ok = color_dist > 0.18
        fg_is_blurry = (e < e_med * 0.75)
        dark_pct = is_deep_dark_bg.float().mean()*100
        extreme_dark_pct = is_extreme_dark_bg.float().mean()*100
        pure_black_pct = is_pure_black.float().mean()*100
        very_deep_pct = is_very_deep_dark.float().mean()*100
        blur_rgb_pct = bg_is_blurry_lr_range_rgb.float().mean()*100
        blur_disp_pct = bg_is_blurry_lr_range_disp.float().mean()*100
        color_ok_pct = color_ok.float().mean()*100
        smooth_line_pct = is_smooth_line.float().mean()*100
        flat_fg_pct = is_flat_fg.float().mean()*100
    else:
        brightness = torch.ones_like(disp)*0.5
        bg_brightness = torch.ones_like(disp)*0.5
        bg_is_blurry_lr_range_rgb = torch.zeros_like(disp, dtype=torch.bool)
        bg_is_very_blurry_lr_range_rgb = torch.zeros_like(disp, dtype=torch.bool)
        is_pure_black = torch.zeros_like(disp, dtype=torch.bool)
        is_extreme_dark_bg = bg_is_very_blurry_lr_range_disp
        is_deep_dark_bg = bg_is_blurry_lr_range_disp
        is_mid_dark_bg = torch.zeros_like(disp, dtype=torch.bool)
        is_very_deep_dark = torch.zeros_like(disp, dtype=torch.bool)
        color_ok = torch.ones_like(disp, dtype=torch.bool)
        fg_is_blurry = torch.zeros_like(disp, dtype=torch.bool)
        dark_pct = 0.0
        extreme_dark_pct = 0.0
        pure_black_pct = 0.0
        very_deep_pct = 0.0
        blur_rgb_pct = 0.0
        blur_disp_pct = bg_is_blurry_lr_range_disp.float().mean()*100
        color_ok_pct = 100.0
        smooth_line_pct = is_smooth_line.float().mean()*100
        flat_fg_pct = is_flat_fg.float().mean()*100

    bg_is_blurry_lr_range = bg_is_blurry_lr_range_disp | bg_is_blurry_lr_range_rgb
    bg_is_very_blurry_lr_range = bg_is_very_blurry_lr_range_disp | bg_is_very_blurry_lr_range_rgb
    bg_is_blurry_lr = bg_is_blurry_lr_range | bg_is_blurry_lr_global
    bg_is_very_blurry_lr = bg_is_very_blurry_lr_range

    top_mask = torch.zeros_like(disp, dtype=torch.bool)
    top_mask[:int(orig_H*top_ratio), :] = True
    bottom_mask = torch.zeros_like(disp, dtype=torch.bool)
    bottom_mask[int(orig_H*0.78):, :] = True
    finger_thin_base = (thinness > 0.012) & (disp > 0.08)
    top_mask = top_mask | (bottom_mask & finger_thin_base)

    eff_thresh = thin_thresh * 0.22

    missed_visible = (disp < 0.58) & (disp_max3 > 0.48) & (thinness > eff_thresh) & top_mask
    flicker_bright = (disp > 0.40) & (thinness > eff_thresh*0.5) & top_mask

    fg_tip_rel = (disp > disp_avg7 + 0.015) & (disp > disp_min21 + 0.02) & top_mask
    bg_tip_rel = (disp < disp_avg7 - 0.015)

    fg_silhouette = (disp > 0.10) & (disp < 0.98) & ((disp_max21 - disp) > 0.006) & (e_blur7 > e_med * 0.005) & top_mask & fg_tip_rel & (~bg_tip_rel) & (~is_smooth_line) & (~is_flat_fg)
    fg_silhouette_lr = (disp > 0.08) & (disp < 0.98) & top_mask & fg_tip_rel & (~bg_tip_rel) & bg_is_blurry_lr & (~is_smooth_line) & (~is_flat_fg)
    fg_silhouette = fg_silhouette | fg_silhouette_lr

    up_protrusion = (disp > disp_min_v15 + 0.04) & (disp > disp_avg7 + 0.015) & (disp > 0.10) & (disp < 0.96) & (gy.abs() > gx.abs() * 0.25) & top_mask & fg_tip_rel & (~bg_tip_rel) & (~is_smooth_line)
    down_protrusion = (disp > disp_min_v15 + 0.04) & (disp > disp_avg7 + 0.015) & (disp > 0.10) & (disp < 0.96) & (gy.abs() > gx.abs() * 0.25) & top_mask & fg_tip_rel & (~bg_tip_rel) & (~is_smooth_line)
    up_protrusion_blurry = up_protrusion & (bg_is_blurry | (e_blur7 < e_med * 0.80))
    down_protrusion_blurry = down_protrusion & (bg_is_blurry | (e_blur7 < e_med * 0.80))
    horiz_protrusion = (disp > disp_avg7 + 0.015) & (disp > disp_min_v7 + 0.02) & (gy.abs() > gx.abs() * 0.20) & (disp > 0.08) & top_mask & fg_tip_rel & (~bg_tip_rel) & (~is_smooth_line)

    corner_tip = (disp < 0.80) & (disp > disp_avg7 + 0.015) & top_mask & fg_tip_rel & (~is_smooth_line)

    all_fg_edge = fg_silhouette | missed_visible | flicker_bright
    hair_thin = (thinness > 0.010) & (disp_max3 > 0.28) & (disp < 0.92) & top_mask & (~is_smooth_line) & (~is_flat_fg)
    finger_thin = (thinness > 0.010) & (disp > 0.06) & top_mask & (~is_smooth_line)

    soft_mask = all_fg_edge | hair_thin | finger_thin | up_protrusion | down_protrusion | horiz_protrusion | up_protrusion_blurry | down_protrusion_blurry | corner_tip

    disp_3x3_max = _same_pool(x4d, 3).squeeze()
    disp_3x3_avg = _same_avg_pool(x4d, 3).squeeze()
    d_res_finger = disp_3x3_max
    d_res_hair = disp_3x3_avg * 0.15 + disp_3x3_max * 0.85
    d_res = torch.where(finger_thin, d_res_finger, d_res_hair)

    soft_f = soft_mask.float().unsqueeze(0).unsqueeze(0)
    print(f"[HG V86] mask {soft_mask.float().mean()*100:.2f}% finger {finger_thin.float().mean()*100:.2f}% blur_disp {blur_disp_pct:.2f}% blur_rgb {blur_rgb_pct:.2f}% dark {dark_pct:.1f}% ext_dark {extreme_dark_pct:.1f}% pure_black {pure_black_pct:.1f}% very_deep {very_deep_pct:.1f}% smooth {smooth_line_pct:.1f}% flat {flat_fg_pct:.1f}% color_ok {color_ok_pct:.1f}% exp_w {exp_w_adapt} rgb={'yes' if rgb_norm is not None else 'no'}")
    if soft_f.sum() < 1:
        return disp.clamp(0,1), torch.zeros_like(disp), torch.ones_like(disp)

    m3 = _same_pool(soft_f, 3).squeeze()
    m7 = _same_pool(soft_f, 7).squeeze()
    m11 = _same_pool(soft_f, 11).squeeze()
    m21 = _same_pool(soft_f, 21).squeeze()
    m_combined = m3 * 0.30 + m7 * 0.30 + m11 * 0.20 + m21 * 0.10 + up_protrusion.float()*0.5 + down_protrusion.float()*0.5 + horiz_protrusion.float()*0.6
    m_combined = torch.clamp(m_combined, 0, 1)

    m_exp_h = _same_pool(m_combined.unsqueeze(0).unsqueeze(0), (5, exp_w_adapt)).squeeze()
    m_exp_v = _same_pool(m_combined.unsqueeze(0).unsqueeze(0), (18, 5)).squeeze()
    m_expanded = m_exp_h * 1.04 * (0.40 + angle_lr*0.60) + m_exp_v * 0.75 * 0.6 + m_combined * 0.30
    m_expanded = torch.clamp(m_expanded, 0, 1)

    def soft_blur(x, k=7):
        return _same_avg_pool(x.unsqueeze(0).unsqueeze(0), k).squeeze()
    
    alpha = soft_blur(m_expanded, k=21)
    alpha = torch.clamp(alpha * 1.3, 0, 1)

    G_final = 1.0 - alpha * (1.0 - G * 0.42)
    G_final = torch.clamp(G_final, min=0.68)
    G_final = torch.where(finger_thin | hair_thin, torch.ones_like(G_final), G_final)
    G_final = torch.where(is_smooth_line | is_flat_fg, torch.ones_like(G_final), G_final)
    disp_hat = disp * G_final + d_res * (1.0 - G_final)

    disp_max_h7_out = _same_pool(x4d, (3, 7)).squeeze()
    disp_max_h15_out = _same_pool(x4d, (3, 15)).squeeze()
    disp_max_h31_out = _same_pool(x4d, (3, 31)).squeeze()
    disp_max_h51_out = _same_pool(x4d, (3, 51)).squeeze()
    lr_mask = (angle_lr > 0.30) & (all_fg_edge | hair_thin | finger_thin) & (~is_smooth_line) & (~is_flat_fg)
    lr_mask_blurry = bg_is_blurry_lr & (disp > 0.08) & (disp < 0.98) & top_mask & (~is_smooth_line) & (~is_flat_fg)
    lr_mask = (lr_mask | lr_mask_blurry | finger_thin) & color_ok & (~is_smooth_line) & (~is_flat_fg)
    outer_boost = torch.ones_like(disp)
    outer_boost = torch.where(bg_is_very_blurry_lr_range, outer_boost*1.15, outer_boost)
    outer_boost = torch.where(bg_is_blurry_lr_range, outer_boost*1.05, outer_boost)
    outer_boost = torch.where(is_extreme_dark_bg, outer_boost*1.35, outer_boost)
    outer_boost = torch.where(is_deep_dark_bg, outer_boost*1.25, outer_boost)
    outer_boost = torch.where(is_mid_dark_bg, outer_boost*1.08, outer_boost)
    outer_boost = torch.where(is_pure_black, outer_boost*1.60, outer_boost)  # V86纯黑加强 1.25->1.60
    outer_boost = torch.where(is_very_deep_dark, outer_boost*1.45, outer_boost)
    outer_boost = torch.where(is_extreme_dark_bg & bg_is_blurry_lr_range, outer_boost*1.15, outer_boost)
    outer_boost = torch.where(is_deep_dark_bg & bg_is_blurry_lr_range, outer_boost*1.10, outer_boost)
    outer_boost = torch.where(fg_is_blurry & bg_is_blurry_lr_range, outer_boost*1.20, outer_boost)  # 前景糊+背景糊多扩
    outer_boost = torch.where(is_smooth_line | is_flat_fg, torch.ones_like(outer_boost)*0.65, outer_boost)  # 平滑线保护
    
    # 纯黑区exp_w多扩2px
    exp_w_pure_black = torch.where(is_pure_black, torch.ones_like(disp)*2.0, torch.ones_like(disp))
    
    disp_hat_out1 = torch.maximum(disp_hat, disp_max_h7_out * (OUTER_H7 * outer_boost).clamp(0,0.60) + disp * (1-(OUTER_H7 * outer_boost).clamp(0,0.60)))
    disp_hat_out2 = torch.maximum(disp_hat_out1, disp_max_h15_out * (OUTER_H15 * outer_boost).clamp(0,0.60) + disp * (1-(OUTER_H15 * outer_boost).clamp(0,0.60)))
    disp_hat_out3 = torch.maximum(disp_hat_out2, disp_max_h31_out * (OUTER_H31 * outer_boost).clamp(0,0.60) + disp * (1-(OUTER_H31 * outer_boost).clamp(0,0.60)))
    disp_hat_out4 = torch.maximum(disp_hat_out3, disp_max_h51_out * (OUTER_H51 * outer_boost).clamp(0,0.60) + disp * (1-(OUTER_H51 * outer_boost).clamp(0,0.60)))
    disp_hat = torch.where(lr_mask, disp_hat_out4, disp_hat)

    disp_max_v7_out = _same_pool(x4d, (7, 3)).squeeze()
    disp_max_v15_out = _same_pool(x4d, (15, 3)).squeeze()
    disp_max_v31_out = _same_pool(x4d, (31, 3)).squeeze()
    up_mask = up_protrusion & (disp > 0.08) & (disp < 0.98) & top_mask & fg_tip_rel & (~bg_tip_rel) & color_ok & (~is_smooth_line)
    up_mask = up_mask & (gy < -0.005)
    disp_hat_ud_up1 = torch.maximum(disp_hat, disp_max_v7_out * OUTER_V7 + disp * (1-OUTER_V7))
    disp_hat_ud_up2 = torch.maximum(disp_hat_ud_up1, disp_max_v15_out * OUTER_V15 + disp * (1-OUTER_V15))
    disp_hat_ud_up3 = torch.maximum(disp_hat_ud_up2, disp_max_v31_out * OUTER_V31 + disp * (1-OUTER_V31))
    disp_hat = torch.where(up_mask, disp_hat_ud_up3, disp_hat)

    down_mask_base = down_protrusion & (disp > 0.08) & (disp < 0.98) & top_mask & fg_tip_rel & (~bg_tip_rel) & (gy > 0.005) & color_ok & (~is_smooth_line)
    true_bg_below = (disp_min_v15 < disp - 0.05) & (disp_min51 < disp - 0.03)
    down_mask = down_mask_base & true_bg_below
    disp_hat_ud_down1 = torch.maximum(disp_hat, disp_max_v7_out * OUTER_V7_DOWN + disp * (1-OUTER_V7_DOWN))
    disp_hat_ud_down2 = torch.maximum(disp_hat_ud_down1, disp_max_v15_out * OUTER_V15_DOWN + disp * (1-OUTER_V15_DOWN))
    disp_hat_ud_down3 = torch.maximum(disp_hat_ud_down2, disp_max_v31_out * OUTER_V31_DOWN + disp * (1-OUTER_V31_DOWN))
    disp_hat = torch.where(down_mask, disp_hat_ud_down3, disp_hat)

    tip_mask = up_protrusion | down_protrusion
    disp_hat = torch.where(tip_mask, torch.maximum(disp_hat, d_res * 0.85 + disp * 0.15), disp_hat)
    tip_mask_h = horiz_protrusion
    disp_hat = torch.where(tip_mask_h, disp * (1-alpha*0.85) + d_res * (alpha*0.85), disp_hat)

    disp_hat = torch.maximum(disp_hat, disp)

    return disp_hat.clamp(0,1), alpha, G_final

def hair_guard_fix_v2(disp, rgb=None, top_ratio=1.0, thin_thresh=0.10, **kwargs):
    return hair_guard_fix(disp, rgb=rgb, top_ratio=top_ratio, thin_thresh=thin_thresh, expand_ratio=1.5)
