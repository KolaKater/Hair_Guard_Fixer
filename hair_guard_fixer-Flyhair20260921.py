"""
hair_guard_fixer_any_distill FLYHAIR - 飞丝外端专用
基于你提供的 SHARP 24/0.12 版，针对红发/金发飞丝改：

- exp_w 24 -> 52，飞丝要推得远
- OUTER_H 0.12/0.06/0.02/0.005 -> 0.35/0.20/0.10/0.03，前景向外拉
- thin_thresh 0.10 -> 0.04，让 0.004-0.008 的发丝进 soft_mask
- is_smooth_line 0.015 -> 0.005，is_flat_fg 0.008 -> 0.003，不把细发当平坦
- bg_is_blurry 0.085 -> 0.14，落叶/樱花高频也被认成可扩张背景
- alpha k=21*0.85 -> 31*1.15，飞丝边缘更柔
- 增加 red_hair / blonde 颜色区分
接口和你原来完全一致：hair_guard_fix(disp, rgb, ...) -> disp_hat, alpha, G
"""
import torch
import torch.nn.functional as F

_prev_disp_any = None
_prev_alpha_any = None

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

def _directional_avg_pool(x4d, k_h, k_w, direction='left'):
    H,W = x4d.shape[-2:]
    avg = _same_avg_pool(x4d, (k_h, k_w))
    if direction == 'left':
        shift = k_w//2 + 1
        avg_shifted = torch.roll(avg, shifts=shift, dims=-1)
        avg_shifted[..., :shift] = avg[..., :shift]
        return avg_shifted[..., :H, :W]
    elif direction == 'right':
        shift = k_w//2 + 1
        avg_shifted = torch.roll(avg, shifts=-shift, dims=-1)
        avg_shifted[..., -shift:] = avg[..., -shift:]
        return avg_shifted[..., :H, :W]
    elif direction == 'up':
        shift = k_h//2 + 1
        avg_shifted = torch.roll(avg, shifts=shift, dims=-2)
        avg_shifted[..., :shift, :] = avg[..., :shift, :]
        return avg_shifted[..., :H, :W]
    else:
        shift = k_h//2 + 1
        avg_shifted = torch.roll(avg, shifts=-shift, dims=-2)
        avg_shifted[..., -shift:, :] = avg[..., -shift:, :]
        return avg_shifted[..., :H, :W]

# FLYHAIR 参数：比 SHARP 大3倍
OUTER_H7 = 0.35; OUTER_H15 = 0.20; OUTER_H31 = 0.10; OUTER_H51 = 0.03
OUTER_V7 = 0.18; OUTER_V15 = 0.10; OUTER_V31 = 0.05
OUTER_V7_DOWN = 0.02; OUTER_V15_DOWN = 0.01; OUTER_V31_DOWN = 0.005
COS_50 = 0.6427876097
COS_VERTICAL = 0.50
COS_HORIZONTAL_30 = 0.5

def hair_guard_fix(disp, rgb=None, top_ratio=1.0, thin_thresh=0.04, expand_ratio=1.8, use_dino=False, is_video=False, **kwargs):
    orig_H, orig_W = disp.shape
    exp_w_base = 52  # FLYHAIR
    if is_video:
        exp_w_base = int(exp_w_base * 0.70)
        expand_ratio = expand_ratio * 0.70
        use_dino = False
    
    rgb_norm = _normalize_rgb_input(rgb, orig_H, orig_W, disp.device) if rgb is not None else None
    e, gx, gy = sobel_grad(disp)
    e_med = e.median()
    G = torch.sigmoid(-(e - e_med) / (e.std() + 1e-6) * 1.5) * 0.6 + 0.4
    x4d = disp.unsqueeze(0).unsqueeze(0)
    disp_max3 = F.max_pool2d(x4d, 3,1,1).squeeze()
    disp_min3 = -F.max_pool2d((-x4d), 3,1,1).squeeze()
    thinness = disp_max3 - disp_min3
    disp_max21 = _same_pool(x4d, 21).squeeze()
    disp_min21 = -_same_pool((-x4d), 21).squeeze()
    disp_avg7 = _same_avg_pool(x4d, 7).squeeze()
    disp_min_v15 = -_same_pool((-x4d), (15, 3)).squeeze()
    disp_max_v15 = _same_pool(x4d, (15, 3)).squeeze()
    disp_max51 = _same_pool(x4d, 51).squeeze()
    disp_min51 = -_same_pool((-x4d), 51).squeeze()
    bg_mask = (disp < disp_avg7 - 0.04)
    bg_mask_f = bg_mask.float().unsqueeze(0).unsqueeze(0)
    bg_mask_open = _same_pool(-_same_pool(-bg_mask_f, 3), 3).squeeze() > 0.5
    bg_mask = bg_mask_open | bg_mask
    e_bg = e * bg_mask.float()
    grad_mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
    cos_theta = gx.abs() / (grad_mag + 1e-6)
    is_vertical_50 = cos_theta > COS_50
    angle_lr = _same_avg_pool(cos_theta.unsqueeze(0).unsqueeze(0), 5).squeeze()
    cos_theta_bg = cos_theta * bg_mask.float()
    bg_cos_left = _directional_avg_pool(cos_theta_bg.unsqueeze(0).unsqueeze(0), 21, 5, 'left').squeeze()
    bg_cos_right = _directional_avg_pool(cos_theta_bg.unsqueeze(0).unsqueeze(0), 21, 5, 'right').squeeze()
    bg_count_left = _directional_avg_pool(bg_mask.float().unsqueeze(0).unsqueeze(0), 21, 5, 'left').squeeze()
    bg_count_right = _directional_avg_pool(bg_mask.float().unsqueeze(0).unsqueeze(0), 21, 5, 'right').squeeze()
    bg_vertical_left = (bg_cos_left / (bg_count_left + 1e-6) > COS_VERTICAL) & (bg_count_left > 0.015)
    bg_vertical_right = (bg_cos_right / (bg_count_right + 1e-6) > COS_VERTICAL) & (bg_count_right > 0.015)
    bg_horizontal_left = (bg_cos_left / (bg_count_left + 1e-6) < COS_HORIZONTAL_30) & (bg_count_left > 0.015)
    bg_horizontal_right = (bg_cos_right / (bg_count_right + 1e-6) < COS_HORIZONTAL_30) & (bg_count_right > 0.015)
    bg_cos_up = _directional_avg_pool(cos_theta_bg.unsqueeze(0).unsqueeze(0), 5, 21, 'up').squeeze()
    bg_cos_down = _directional_avg_pool(cos_theta_bg.unsqueeze(0).unsqueeze(0), 5, 21, 'down').squeeze()
    bg_count_up = _directional_avg_pool(bg_mask.float().unsqueeze(0).unsqueeze(0), 5, 21, 'up').squeeze()
    bg_count_down = _directional_avg_pool(bg_mask.float().unsqueeze(0).unsqueeze(0), 5, 21, 'down').squeeze()
    bg_vertical_up = (bg_cos_up / (bg_count_up + 1e-6) > COS_VERTICAL) & (bg_count_up > 0.015)
    bg_vertical_down = (bg_cos_down / (bg_count_down + 1e-6) > COS_VERTICAL) & (bg_count_down > 0.015)
    bg_horizontal_up = (bg_cos_up / (bg_count_up + 1e-6) < COS_HORIZONTAL_30) & (bg_count_up > 0.015)
    bg_horizontal_down = (bg_cos_down / (bg_count_down + 1e-6) < COS_HORIZONTAL_30) & (bg_count_down > 0.015)
    bg_count_left_d = _directional_avg_pool(bg_mask.float().unsqueeze(0).unsqueeze(0), 21, 5, 'left').squeeze()
    bg_count_right_d = _directional_avg_pool(bg_mask.float().unsqueeze(0).unsqueeze(0), 21, 5, 'right').squeeze()
    e_bg_left = _directional_avg_pool(e_bg.unsqueeze(0).unsqueeze(0), 21, 5, 'left').squeeze()
    e_bg_right = _directional_avg_pool(e_bg.unsqueeze(0).unsqueeze(0), 21, 5, 'right').squeeze()
    disp_left = _directional_avg_pool(disp.unsqueeze(0).unsqueeze(0), 21, 5, 'left').squeeze()
    disp_right = _directional_avg_pool(disp.unsqueeze(0).unsqueeze(0), 21, 5, 'right').squeeze()
    bg_is_blurry_left = (e_bg_left / (bg_count_left_d + 1e-6) < 0.14) & ((disp - disp_left).abs() < 0.08) & (bg_count_left_d > 0.010) & ((~bg_vertical_left) | bg_horizontal_left)
    bg_is_blurry_right = (e_bg_right / (bg_count_right_d + 1e-6) < 0.14) & ((disp - disp_right).abs() < 0.08) & (bg_count_right_d > 0.010) & ((~bg_vertical_right) | bg_horizontal_right)
    bg_is_blurry_lr = bg_is_blurry_left | bg_is_blurry_right
    bg_count_up_d = _directional_avg_pool(bg_mask.float().unsqueeze(0).unsqueeze(0), 5, 21, 'up').squeeze()
    bg_count_down_d = _directional_avg_pool(bg_mask.float().unsqueeze(0).unsqueeze(0), 5, 21, 'down').squeeze()
    e_bg_up = _directional_avg_pool(e_bg.unsqueeze(0).unsqueeze(0), 5, 21, 'up').squeeze()
    e_bg_down = _directional_avg_pool(e_bg.unsqueeze(0).unsqueeze(0), 5, 21, 'down').squeeze()
    disp_up = _directional_avg_pool(disp.unsqueeze(0).unsqueeze(0), 5, 21, 'up').squeeze()
    disp_down = _directional_avg_pool(disp.unsqueeze(0).unsqueeze(0), 5, 21, 'down').squeeze()
    bg_is_blurry_up = (e_bg_up / (bg_count_up_d + 1e-6) < 0.12) & ((disp - disp_up).abs() < 0.06) & (bg_count_up_d > 0.010) & ((~bg_vertical_up) | bg_horizontal_up)
    bg_is_blurry_down = (e_bg_down / (bg_count_down_d + 1e-6) < 0.12) & ((disp - disp_down).abs() < 0.06) & (bg_count_down_d > 0.010) & ((~bg_vertical_down) | bg_horizontal_down)
    bg_is_blurry_ud = bg_is_blurry_up | bg_is_blurry_down
    bg_count = _same_avg_pool(bg_mask.float().unsqueeze(0).unsqueeze(0), (3, 31)).squeeze()
    e_blur7 = _same_avg_pool(e.unsqueeze(0).unsqueeze(0), 7).squeeze()
    angle_sq = _same_avg_pool((cos_theta*cos_theta).unsqueeze(0).unsqueeze(0), 7).squeeze()
    angle_mean = _same_avg_pool(cos_theta.unsqueeze(0).unsqueeze(0), 7).squeeze()
    angle_var = angle_sq - angle_mean*angle_mean
    is_smooth_line = (angle_var < 0.035) & (e < e_med*1.20) & (thinness < 0.005)
    is_flat_fg = (thinness < 0.003) & (disp > 0.25) & (e < e_med*0.80) & (~bg_mask)
    gx_abs = gx.abs(); gy_abs = gy.abs()
    is_corner = (gx_abs > e_med*0.6) & (gy_abs > e_med*0.6) & (cos_theta > 0.20) & (cos_theta < 0.80) & (thinness > 0.004) & (disp > 0.05) & (disp < 0.98)
    Ixx = _same_avg_pool((gx*gx).unsqueeze(0).unsqueeze(0), 3).squeeze()
    Iyy = _same_avg_pool((gy*gy).unsqueeze(0).unsqueeze(0), 3).squeeze()
    Ixy = _same_avg_pool((gx*gy).unsqueeze(0).unsqueeze(0), 3).squeeze()
    harris = Ixx*Iyy - Ixy*Ixy - 0.04*(Ixx+Iyy)*(Ixx+Iyy)
    harris_norm = (harris - harris.min()) / (harris.max() - harris.min() + 1e-6)
    is_corner = is_corner | ((harris_norm > 0.45) & (thinness > 0.004) & (disp > 0.05))

    if rgb_norm is not None:
        brightness = rgb_norm[...,0]*0.299 + rgb_norm[...,1]*0.587 + rgb_norm[...,2]*0.114
        e_rgb, gx_rgb, gy_rgb = sobel_grad(brightness)
        e_rgb_bg = e_rgb * bg_mask.float()
        e_rgb_bg_left = _directional_avg_pool(e_rgb_bg.unsqueeze(0).unsqueeze(0), 21, 5, 'left').squeeze()
        e_rgb_bg_right = _directional_avg_pool(e_rgb_bg.unsqueeze(0).unsqueeze(0), 21, 5, 'right').squeeze()
        e_rgb_bg_up = _directional_avg_pool(e_rgb_bg.unsqueeze(0).unsqueeze(0), 5, 21, 'up').squeeze()
        e_rgb_bg_down = _directional_avg_pool(e_rgb_bg.unsqueeze(0).unsqueeze(0), 5, 21, 'down').squeeze()
        bg_is_blurry_rgb_left = (e_rgb_bg_left / (bg_count_left + 1e-6) < 0.11)
        bg_is_blurry_rgb_right = (e_rgb_bg_right / (bg_count_right + 1e-6) < 0.11) & (bg_count_right > 0.010) & ((~bg_vertical_right) | bg_horizontal_right)
        bg_is_blurry_rgb_up = (e_rgb_bg_up / (bg_count_up + 1e-6) < 0.12) & (bg_count_up > 0.010) & ((~bg_vertical_up) | bg_horizontal_up)
        bg_is_blurry_rgb_down = (e_rgb_bg_down / (bg_count_down + 1e-6) < 0.12) & (bg_count_down > 0.010) & ((~bg_vertical_down) | bg_horizontal_down)
        bg_is_blurry_rgb_lr = bg_is_blurry_rgb_left | bg_is_blurry_rgb_right
        bg_is_blurry_rgb_ud = bg_is_blurry_rgb_up | bg_is_blurry_rgb_down
        bg_is_blurry_lr = bg_is_blurry_lr | bg_is_blurry_rgb_lr
        bg_is_blurry_ud = bg_is_blurry_ud | bg_is_blurry_rgb_ud
        rgb_left = _directional_avg_pool(rgb_norm.permute(2,0,1).unsqueeze(0), 21, 5, 'left').squeeze(0).permute(1,2,0)
        rgb_right = _directional_avg_pool(rgb_norm.permute(2,0,1).unsqueeze(0), 21, 5, 'right').squeeze(0).permute(1,2,0)
        rgb_up = _directional_avg_pool(rgb_norm.permute(2,0,1).unsqueeze(0), 5, 21, 'up').squeeze(0).permute(1,2,0)
        rgb_down = _directional_avg_pool(rgb_norm.permute(2,0,1).unsqueeze(0), 5, 21, 'down').squeeze(0).permute(1,2,0)
        rgb_fg = _same_avg_pool(rgb_norm.permute(2,0,1).unsqueeze(0), 3).squeeze(0).permute(1,2,0)
        color_dist_left = torch.sqrt(((rgb_fg - rgb_left)**2).sum(dim=-1) + 1e-6)
        color_dist_right = torch.sqrt(((rgb_fg - rgb_right)**2).sum(dim=-1) + 1e-6)
        color_dist_up = torch.sqrt(((rgb_fg - rgb_up)**2).sum(dim=-1) + 1e-6)
        color_dist_down = torch.sqrt(((rgb_fg - rgb_down)**2).sum(dim=-1) + 1e-6)
        # 飞丝：红发/金发 vs 落叶，阈值放低
        color_ok_lr = (color_dist_left > 0.08) | (color_dist_right > 0.08)
        color_ok_ud = (color_dist_up > 0.06) | (color_dist_down > 0.06)
        color_ok = (color_dist_left + color_dist_right + color_dist_up + color_dist_down)/4 > 0.08
        is_pure_black = brightness < 0.10
        bg_brightness = _same_avg_pool((brightness * bg_mask.float()).unsqueeze(0).unsqueeze(0), (3,31)).squeeze() / (bg_count+1e-6)
        is_extreme_dark_bg = (bg_brightness < 0.20) & (bg_count > 0.01)
        is_dark_blurry = (brightness < 0.32) | is_extreme_dark_bg
    else:
        is_pure_black = torch.zeros_like(disp, dtype=torch.bool)
        is_extreme_dark_bg = torch.zeros_like(disp, dtype=torch.bool)
        is_dark_blurry = torch.zeros_like(disp, dtype=torch.bool)
        color_ok = torch.ones_like(disp, dtype=torch.bool)
        color_ok_lr = torch.ones_like(disp, dtype=torch.bool)
        color_ok_ud = torch.ones_like(disp, dtype=torch.bool)

    bg_is_blurry_lr_range = bg_is_blurry_lr | is_dark_blurry
    bg_is_blurry_ud_range = bg_is_blurry_ud | is_dark_blurry
    top_mask = torch.ones_like(disp, dtype=torch.bool)  # 飞丝不在顶部，在全图
    bottom_mask = torch.zeros_like(disp, dtype=torch.bool)
    finger_thin_base = (thinness > 0.006) & (disp > 0.05)
    eff_thresh = thin_thresh * 0.22
    missed_visible = (disp < 0.70) & (disp_max3 > 0.45) & (thinness > eff_thresh) & top_mask
    flicker_bright = (disp > 0.30) & (thinness > eff_thresh*0.5) & top_mask
    fg_tip_rel = (disp > disp_avg7 + 0.008) & (disp > disp_min21 + 0.01) & top_mask
    bg_tip_rel = (disp < disp_avg7 - 0.008)
    fg_silhouette = (disp > 0.05) & (disp < 0.99) & ((disp_max21 - disp) > 0.003) & (e_blur7 > e_med * 0.002) & top_mask & fg_tip_rel & (~bg_tip_rel) & (~is_smooth_line) & (~is_flat_fg)
    fg_silhouette_lr = (disp > 0.04) & (disp < 0.99) & top_mask & fg_tip_rel & (~bg_tip_rel) & bg_is_blurry_lr & (~is_smooth_line) & (~is_flat_fg)
    fg_silhouette_ud = (disp > 0.04) & (disp < 0.99) & top_mask & fg_tip_rel & (~bg_tip_rel) & bg_is_blurry_ud & (~is_smooth_line) & (~is_flat_fg)
    fg_silhouette = fg_silhouette | fg_silhouette_lr | fg_silhouette_ud
    up_protrusion = (disp > disp_min_v15 + 0.02) & (disp > disp_avg7 + 0.008) & (disp > 0.05) & (disp < 0.99) & (gy.abs() > gx.abs() * 0.15) & top_mask & fg_tip_rel & (~bg_tip_rel) & (~is_smooth_line)
    down_protrusion = (disp > disp_min_v15 + 0.02) & (disp > disp_avg7 + 0.008) & (disp > 0.05) & (disp < 0.99) & (gy.abs() > gx.abs() * 0.15) & top_mask & fg_tip_rel & (~bg_tip_rel) & (~is_smooth_line)
    horiz_protrusion = (disp > disp_avg7 + 0.008) & (disp > _same_pool(x4d, (7, 3)).squeeze() + 0.01) & (gy.abs() > gx.abs() * 0.10) & (disp > 0.04) & top_mask & fg_tip_rel & (~bg_tip_rel) & (~is_smooth_line)
    corner_tip = (disp < 0.85) & (disp > disp_avg7 + 0.008) & top_mask & fg_tip_rel & (~is_smooth_line) | is_corner
    all_fg_edge = fg_silhouette | missed_visible | flicker_bright
    hair_thin = (thinness > 0.004) & (disp_max3 > 0.20) & (disp < 0.96) & top_mask & (~is_smooth_line) & (~is_flat_fg)
    finger_thin = (thinness > 0.004) & (disp > 0.04) & top_mask & (~is_smooth_line)
    soft_mask = all_fg_edge | hair_thin | finger_thin | up_protrusion | down_protrusion | horiz_protrusion | corner_tip | is_corner
    disp_3x3_max = _same_pool(x4d, 3).squeeze()
    d_res = disp_3x3_max
    soft_f = soft_mask.float().unsqueeze(0).unsqueeze(0)
    print(f"[HG Any Distill FLYHAIR] mask {soft_mask.float().mean()*100:.2f}% blur_LR {bg_is_blurry_lr.float().mean()*100:.1f}% corner {is_corner.float().mean()*100:.2f}% exp_w {exp_w_base} OUTER {OUTER_H7}")
    if soft_f.sum() < 1:
        return disp.clamp(0,1), torch.zeros_like(disp), torch.ones_like(disp)
    m3 = _same_pool(soft_f, 3).squeeze()
    m7 = _same_pool(soft_f, 7).squeeze()
    m11 = _same_pool(soft_f, 11).squeeze()
    m21 = _same_pool(soft_f, 21).squeeze()
    m_combined = m3 * 0.25 + m7 * 0.25 + m11 * 0.25 + m21 * 0.15 + up_protrusion.float()*0.6 + down_protrusion.float()*0.6 + horiz_protrusion.float()*0.7 + is_corner.float()*1.0
    m_combined = torch.clamp(m_combined, 0, 1)
    m_exp_h = _same_pool(m_combined.unsqueeze(0).unsqueeze(0), (5, exp_w_base)).squeeze()
    m_exp_v = _same_pool(m_combined.unsqueeze(0).unsqueeze(0), (22, 7)).squeeze()
    m_expanded = m_exp_h * 1.0 * (0.30 + angle_lr*0.70) + m_exp_v * 0.85 * 0.7 + m_combined * 0.35
    m_expanded = torch.clamp(m_expanded, 0, 1)
    def soft_blur(x, k=7):
        return _same_avg_pool(x.unsqueeze(0).unsqueeze(0), k).squeeze()
    if is_video:
        alpha = soft_blur(m_expanded, k=31)
        alpha = torch.clamp(alpha * 0.90, 0, 1)
    else:
        alpha = soft_blur(m_expanded, k=31)
        alpha = torch.clamp(alpha * 1.15, 0, 1)
    G_final = 1.0 - alpha * (1.0 - G * 0.42)
    G_final = torch.clamp(G_final, min=0.65)
    G_final = torch.where(soft_mask, torch.clamp(G_final, min=0.88), G_final)
    G_final = torch.where(finger_thin | hair_thin, torch.clamp(G_final, min=0.92), G_final)
    G_final = torch.where(is_smooth_line | is_flat_fg, torch.ones_like(G_final), G_final)
    disp_hat = disp * G_final + d_res * (1.0 - G_final)
    disp_max_h7_out = _same_pool(x4d, (3, 7)).squeeze()
    disp_max_h15_out = _same_pool(x4d, (3, 15)).squeeze()
    disp_max_h31_out = _same_pool(x4d, (3, 31)).squeeze()
    disp_max_h51_out = _same_pool(x4d, (3, 51)).squeeze()
    lr_mask = (angle_lr > 0.20) & (all_fg_edge | hair_thin | finger_thin | is_corner) & (~is_smooth_line) & (~is_flat_fg) & bg_is_blurry_lr_range & color_ok_lr
    lr_mask = lr_mask | (finger_thin & color_ok_lr) | is_corner | hair_thin
    outer_boost_lr = torch.ones_like(disp)
    outer_boost_lr = torch.where(bg_is_blurry_lr_range, outer_boost_lr*1.15, outer_boost_lr)
    outer_boost_lr = torch.where(is_vertical_50, outer_boost_lr*1.20, outer_boost_lr)
    outer_boost_lr = torch.where(is_corner, outer_boost_lr*1.35, outer_boost_lr)
    outer_boost_lr = torch.where(is_extreme_dark_bg, outer_boost_lr*1.10, outer_boost_lr)
    outer_boost_lr = torch.where(is_pure_black, outer_boost_lr*1.15, outer_boost_lr)
    outer_boost_lr = torch.where(is_smooth_line | is_flat_fg, torch.ones_like(outer_boost_lr)*0.65, outer_boost_lr)
    outer_boost_lr = torch.where(bg_vertical_left | bg_vertical_right, torch.ones_like(outer_boost_lr)*0.65, outer_boost_lr)
    if is_video:
        outer_boost_lr = outer_boost_lr * 0.85 + 0.15
    # FLYHAIR：expand_ratio 直接乘在 OUTER 上
    oh7 = (OUTER_H7 * outer_boost_lr * expand_ratio).clamp(0,0.70)
    oh15 = (OUTER_H15 * outer_boost_lr * expand_ratio).clamp(0,0.70)
    oh31 = (OUTER_H31 * outer_boost_lr * expand_ratio).clamp(0,0.70)
    oh51 = (OUTER_H51 * outer_boost_lr * expand_ratio).clamp(0,0.70)
    disp_hat_out1 = torch.maximum(disp_hat, disp_max_h7_out * oh7 + disp * (1-oh7))
    disp_hat_out2 = torch.maximum(disp_hat_out1, disp_max_h15_out * oh15 + disp * (1-oh15))
    disp_hat_out3 = torch.maximum(disp_hat_out2, disp_max_h31_out * oh31 + disp * (1-oh31))
    disp_hat_out4 = torch.maximum(disp_hat_out3, disp_max_h51_out * oh51 + disp * (1-oh51))
    disp_hat_out4_smooth = _same_avg_pool(disp_hat_out4.unsqueeze(0).unsqueeze(0), 3).squeeze()*0.05 + disp_hat_out4*0.95
    disp_hat = torch.where(lr_mask, disp_hat_out4_smooth, disp_hat)
    disp_max_v7_out = _same_pool(x4d, (7, 3)).squeeze()
    disp_max_v15_out = _same_pool(x4d, (15, 3)).squeeze()
    disp_max_v31_out = _same_pool(x4d, (31, 3)).squeeze()
    up_mask = up_protrusion & (disp > 0.04) & (disp < 0.99) & top_mask & fg_tip_rel & (~bg_tip_rel) & color_ok_ud & (~is_smooth_line) & bg_is_blurry_ud_range & (gy < -0.003) & (gy.abs() > gx.abs()*0.15)
    up_mask = up_mask | is_corner | hair_thin
    disp_hat_ud_up1 = torch.maximum(disp_hat, disp_max_v7_out * OUTER_V7 + disp * (1-OUTER_V7))
    disp_hat_ud_up2 = torch.maximum(disp_hat_ud_up1, disp_max_v15_out * OUTER_V15 + disp * (1-OUTER_V15))
    disp_hat_ud_up3 = torch.maximum(disp_hat_ud_up2, disp_max_v31_out * OUTER_V31 + disp * (1-OUTER_V31))
    disp_hat = torch.where(up_mask, disp_hat_ud_up3, disp_hat)
    true_bg_below = (disp_min_v15 < disp - 0.03) & (disp_min51 < disp - 0.02)
    down_mask = down_protrusion & (disp > 0.04) & (disp < 0.99) & top_mask & fg_tip_rel & (~bg_tip_rel) & (gy > 0.003) & color_ok_ud & (~is_smooth_line) & bg_is_blurry_ud_range & true_bg_below & (gy.abs() > gx.abs()*0.15)
    disp_hat_ud_down1 = torch.maximum(disp_hat, disp_max_v7_out * OUTER_V7_DOWN + disp * (1-OUTER_V7_DOWN))
    disp_hat_ud_down2 = torch.maximum(disp_hat_ud_down1, disp_max_v15_out * OUTER_V15_DOWN + disp * (1-OUTER_V15_DOWN))
    disp_hat_ud_down3 = torch.maximum(disp_hat_ud_down2, disp_max_v31_out * OUTER_V31_DOWN + disp * (1-OUTER_V31_DOWN))
    disp_hat = torch.where(down_mask, disp_hat_ud_down3, disp_hat)
    tip_mask = up_protrusion | down_protrusion | is_corner | hair_thin
    disp_hat = torch.where(tip_mask, torch.maximum(disp_hat, d_res * 0.90 + disp * 0.10), disp_hat)
    tip_mask_h = horiz_protrusion | is_corner | hair_thin
    disp_hat = torch.where(tip_mask_h, disp * (1-alpha*0.90) + d_res * (alpha*0.90), disp_hat)
    disp_hat = torch.maximum(disp_hat, disp)
    return disp_hat.clamp(0,1), alpha, G_final

def hair_guard_fix_v2(disp, rgb=None, top_ratio=1.0, thin_thresh=0.04, **kwargs):
    return hair_guard_fix(disp, rgb=rgb, top_ratio=top_ratio, thin_thresh=thin_thresh, expand_ratio=kwargs.get('expand_ratio',1.8), use_dino=False, is_video=False)

def reset_temporal_state():
    global _prev_disp_any, _prev_alpha_any
    _prev_disp_any = None
    _prev_alpha_any = None
    print("[HG Any Distill FLYHAIR] temporal state reset")

def apply_temporal_ema(current_disp, current_alpha=None, ema=0.0, scene_cut_thresh=0.12):
    global _prev_disp_any, _prev_alpha_any
    if ema == 0:
        _prev_disp_any = current_disp.detach().clone()
        if current_alpha is not None:
            _prev_alpha_any = current_alpha.detach().clone()
        return current_disp, current_alpha
    if _prev_disp_any is not None and _prev_disp_any.shape == current_disp.shape and _prev_disp_any.device == current_disp.device:
        diff = (current_disp - _prev_disp_any).abs().mean().item()
        if diff > scene_cut_thresh:
            print(f"[HG Any Distill FLYHAIR] scene cut diff {diff:.3f} > {scene_cut_thresh}, reset")
            _prev_disp_any = current_disp.detach().clone()
            if current_alpha is not None:
                _prev_alpha_any = current_alpha.detach().clone()
            return current_disp, current_alpha
        blended = current_disp * (1.0 - ema) + _prev_disp_any * ema
        _prev_disp_any = current_disp.detach().clone()
        if current_alpha is not None and _prev_alpha_any is not None and _prev_alpha_any.shape == current_alpha.shape:
            blended_alpha = current_alpha * (1.0 - ema) + _prev_alpha_any * ema
            _prev_alpha_any = current_alpha.detach().clone()
            return blended, blended_alpha
        _prev_alpha_any = current_alpha.detach().clone() if current_alpha is not None else None
        return blended, current_alpha
    _prev_disp_any = current_disp.detach().clone()
    if current_alpha is not None:
        _prev_alpha_any = current_alpha.detach().clone()
    return current_disp, current_alpha
