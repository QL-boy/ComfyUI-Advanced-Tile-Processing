import torch
import torch.nn.functional as F
import numpy as np

def match_histograms(source, reference):
    """
    Reinhard color transfer: Match color distribution of source to reference.
    Input/Output: [H, W, C] Tensor, Range [0,1]
    """
    if source.dtype != torch.float32: source = source.float()
    if reference.dtype != torch.float32: reference = reference.float()

    s_mean = torch.mean(source, dim=(0, 1), keepdim=True)
    s_std = torch.std(source, dim=(0, 1), keepdim=True) + 1e-6
    r_mean = torch.mean(reference, dim=(0, 1), keepdim=True)
    r_std = torch.std(reference, dim=(0, 1), keepdim=True) + 1e-6
    
    s_matched = (source - s_mean) / s_std * r_std + r_mean
    return torch.clamp(s_matched, 0.0, 1.0)

def compute_optimal_seamline(overlap_left, overlap_right, direction='horizontal', search_radius=0, seam_mode='optimal'):
    """
    Compute optimal seamline using Graph Cut / Dynamic Programming.
    """
    device = overlap_left.device
    
    # [H, W]
    diff = torch.abs(overlap_left - overlap_right).mean(dim=-1)
    
    energy_map = diff.cpu().numpy()
    H, W = energy_map.shape
    
    if seam_mode == 'middle':
        seam_mask = torch.zeros((H, W), dtype=torch.float32, device=device)
        if direction == 'horizontal':
            seam_mask[:, :W // 2] = 1.0
        else:
            seam_mask[:H // 2, :] = 1.0
        return seam_mask

    # -- MODE: OPTIMAL --
    energy_range = np.max(energy_map) - np.min(energy_map)
    penalty_scale = 0.01 * (energy_range + 1e-6)
    
    if direction == 'horizontal':
        x_indices = np.arange(W)
        dist_map = np.abs(x_indices - W // 2)
        energy_map += (dist_map[None, :] * penalty_scale)
    else:
        y_indices = np.arange(H)
        dist_map = np.abs(y_indices - H // 2)
        energy_map += (dist_map[:, None] * penalty_scale)

    if search_radius > 0:
        large_val = 1e6
        if direction == 'horizontal': 
            center = W // 2
            start = max(0, center - search_radius)
            end = min(W, center + search_radius)
            energy_map[:, :start] += large_val
            energy_map[:, end:] += large_val
        else:
            center = H // 2
            start = max(0, center - search_radius)
            end = min(H, center + search_radius)
            energy_map[:start, :] += large_val
            energy_map[end:, :] += large_val

    if direction == 'horizontal':
        dp = energy_map.copy()
        backtrack = np.zeros_like(dp, dtype=np.int32)
        
        for i in range(1, H):
            for j in range(W):
                j_start = max(0, j-1)
                j_end = min(W, j+2)
                prev_slice = dp[i-1, j_start:j_end]
                min_idx_local = np.argmin(prev_slice)
                dp[i, j] += prev_slice[min_idx_local]
                backtrack[i, j] = j_start + min_idx_local
        
        mask_np = np.zeros((H, W), dtype=np.float32)
        curr_j = np.argmin(dp[-1, :])
        mask_np[-1, :curr_j] = 1.0
        for i in range(H-1, -1, -1):
            mask_np[i, :curr_j] = 1.0
            if i > 0: curr_j = backtrack[i, curr_j]
            
    else: # vertical
        dp = energy_map.copy()
        backtrack = np.zeros_like(dp, dtype=np.int32)
        
        for j in range(1, W):
            for i in range(H):
                i_start = max(0, i-1)
                i_end = min(H, i+2)
                prev_slice = dp[i_start:i_end, j-1]
                min_idx_local = np.argmin(prev_slice)
                dp[i, j] += prev_slice[min_idx_local]
                backtrack[i, j] = i_start + min_idx_local
                
        mask_np = np.zeros((H, W), dtype=np.float32)
        curr_i = np.argmin(dp[:, -1])
        mask_np[:curr_i, -1] = 1.0
        for j in range(W-1, -1, -1):
            mask_np[:curr_i, j] = 1.0
            if j > 0: curr_i = backtrack[curr_i, j]

    return torch.from_numpy(mask_np).to(device)

def multi_band_blend(img_a, img_b, mask, levels=5):
    """
    Laplacian Pyramid Blending with shape safety.
    """
    if img_a.ndim == 3: img_a = img_a.unsqueeze(0)
    if img_b.ndim == 3: img_b = img_b.unsqueeze(0)
    if mask.ndim == 3: mask = mask.unsqueeze(0)
    
    gauss_a = [img_a]
    gauss_b = [img_b]
    gauss_m = [mask]
    
    min_dim = min(img_a.shape[2], img_a.shape[3])
    max_levels = int(np.log2(min_dim)) - 1
    levels = min(levels, max_levels)
    
    # 1. Build Gaussian Pyramid
    for i in range(levels):
        gauss_a.append(F.avg_pool2d(gauss_a[-1], 2))
        gauss_b.append(F.avg_pool2d(gauss_b[-1], 2))
        gauss_m.append(F.avg_pool2d(gauss_m[-1], 2))
    
    # 2. Build Laplacian Pyramid
    lap_a = []
    lap_b = []
    
    for i in range(levels):
        # Shape of the CURRENT level
        h, w = gauss_a[i].shape[2], gauss_a[i].shape[3]
        # Upsample the NEXT level to match CURRENT
        up_a = F.interpolate(gauss_a[i+1], size=(h, w), mode='bilinear', align_corners=False)
        up_b = F.interpolate(gauss_b[i+1], size=(h, w), mode='bilinear', align_corners=False)
        
        # Now shapes match exactly
        lap_a.append(gauss_a[i] - up_a)
        lap_b.append(gauss_b[i] - up_b)
    
    lap_a.append(gauss_a[-1])
    lap_b.append(gauss_b[-1])
    
    # 3. Blend
    blend_lap = []
    for i in range(levels + 1):
        m = gauss_m[i]
        blend_lap.append(lap_a[i] * (1.0 - m) + lap_b[i] * m)
    
    # 4. Reconstruct
    recons = blend_lap[-1]
    for i in range(levels - 1, -1, -1):
        h, w = blend_lap[i].shape[2], blend_lap[i].shape[3]
        recons = F.interpolate(recons, size=(h, w), mode='bilinear', align_corners=False)
        recons = recons + blend_lap[i]
        
    return torch.clamp(recons, 0.0, 1.0)
