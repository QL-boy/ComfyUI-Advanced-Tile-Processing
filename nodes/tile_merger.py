import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
from ..utils.blending import (
    compute_optimal_seamline,
    match_histograms,
    multi_band_blend
)
import comfy.utils

class CustomTileMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_config": ("TILE_CONFIG",),
                "blend_mode": (["linear", "gaussian", "multi_band", "none"], {"default": "gaussian"}),
                "blend_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seam_mode": (["optimal", "middle"], {"default": "optimal"}),
                "histogram_matching": ("BOOLEAN", {"default": False}),
                "seam_search_radius": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
            "optional": {
                "processed_tiles_image": ("IMAGE",),
                "processed_tiles_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("merged_image", "merged_latent")
    FUNCTION = "merge_tiles"
    CATEGORY = "image/postprocessing"
    INPUT_IS_LIST = True

    def merge_tiles(self, tile_config, blend_mode, blend_strength, seam_mode, 
                    histogram_matching, seam_search_radius,
                    processed_tiles_image=None, processed_tiles_latent=None):
        
        config = tile_config[0]
        mode = blend_mode[0]
        strength = blend_strength[0]
        seam_type = seam_mode[0]
        do_hist_match = histogram_matching[0]
        search_radius = seam_search_radius[0]
        
        splits = config["splits"]
        batch_size = config["batch_size"]
        H, W = config["padded_height"], config["padded_width"]
        orig_H, orig_W = config["original_height"], config["original_width"]
        overlap = config["overlap"]
        
        # Optimization: Build a Grid Map for fast neighbor lookup
        # key: (row, col) -> value: index in splits list
        grid_map = {}
        for idx, s in enumerate(splits):
            grid_map[(s["row"], s["col"])] = idx

        # ================= IMAGE MERGING =================
        merged_img = None
        
        # processed_tiles_image is a LIST of [B, H, W, C] tensors (one per tile position)
        if processed_tiles_image is not None and len(processed_tiles_image) > 0:
            # Check consistency
            if len(processed_tiles_image) != len(splits):
                 print(f"Warning: Tile count mismatch. Config: {len(splits)}, Input: {len(processed_tiles_image)}")
            
            # Use first tile to determine device and channels
            ref_tile = processed_tiles_image[0]
            device = ref_tile.device
            C = ref_tile.shape[-1]
            
            # Check actual batch size from input (in case it changed during processing)
            actual_batch_size = ref_tile.shape[0]

            # Initialize Canvas
            canvas = torch.zeros((actual_batch_size, H, W, C), device=device)
            pbar = comfy.utils.ProgressBar(actual_batch_size * len(splits))
            
            for b in range(actual_batch_size):
                # 1. Prepare Tile Clones for this Batch
                # We need clones because Histogram Matching modifies them in-place
                current_batch_tiles = []
                for i in range(len(splits)):
                    if i < len(processed_tiles_image):
                        # Extract the specific image for batch b
                        # Input List [TileIdx] -> Tensor [Batch, H, W, C]
                        # We want [1, H, W, C] for easier processing logic
                        t = processed_tiles_image[i][b:b+1].clone()
                        current_batch_tiles.append(t)
                    else:
                        current_batch_tiles.append(torch.zeros_like(ref_tile[0:1]))

                # Initialize Masks (All 1.0)
                batch_masks = [torch.ones((s['h'], s['w'], 1), device=device) for s in splits]
                
                # 2. Process Tiles
                for i, split in enumerate(splits):
                    y, x, h, w = split['y'], split['x'], split['h'], split['w']
                    r, c = split['row'], split['col']
                    
                    tile_i = current_batch_tiles[i] # [1, h, w, c]
                    
                    # --- A. Histogram Matching (Look at Left Neighbor) ---
                    if do_hist_match and c > 0:
                        left_idx = grid_map.get((r, c - 1))
                        if left_idx is not None:
                            prev_tile = current_batch_tiles[left_idx]
                            
                            # Overlap: prev_tile Right vs tile_i Left
                            ref = prev_tile[0, :, -overlap:, :]
                            src = tile_i[0, :, :overlap, :]
                            
                            s_mean = torch.mean(src, dim=(0, 1), keepdim=True)
                            s_std = torch.std(src, dim=(0, 1), keepdim=True) + 1e-6
                            r_mean = torch.mean(ref, dim=(0, 1), keepdim=True)
                            r_std = torch.std(ref, dim=(0, 1), keepdim=True) + 1e-6
                            
                            # Apply to full tile
                            tile_i[0] = (tile_i[0] - s_mean) / s_std * r_std + r_mean
                            tile_i.clamp_(0.0, 1.0) # In-place clamp
                            current_batch_tiles[i] = tile_i # Update reference

                    # --- B. Seam Calculation (Cut Future Neighbors) ---
                    
                    # 1. Right Neighbor (Forward Cut)
                    right_idx = grid_map.get((r, c + 1))
                    if right_idx is not None:
                        tile_j = current_batch_tiles[right_idx]
                        
                        ov_left = tile_i[0, :, -overlap:, :] # Left Tile (i) Right part
                        ov_right = tile_j[0, :, :overlap, :] # Right Tile (j) Left part
                        
                        seam = compute_optimal_seamline(ov_left, ov_right, 'horizontal', search_radius, seam_type)
                        seam = seam.to(device).unsqueeze(-1) # [h, overlap, 1]
                        
                        # Tile J (Right) should be hidden where Seam is 1 (Left wins)
                        # So Mask J = 0 where Seam = 1
                        batch_masks[right_idx][:, :overlap, :] *= (1.0 - seam)

                    # 2. Bottom Neighbor (Forward Cut)
                    bottom_idx = grid_map.get((r + 1, c))
                    if bottom_idx is not None:
                        tile_k = current_batch_tiles[bottom_idx]
                        
                        ov_top = tile_i[0, -overlap:, :, :]
                        ov_bottom = tile_k[0, :overlap, :, :]
                        
                        seam = compute_optimal_seamline(ov_top, ov_bottom, 'vertical', search_radius, seam_type)
                        seam = seam.to(device).unsqueeze(-1)
                        
                        batch_masks[bottom_idx][:overlap, :, :] *= (1.0 - seam)

                # 3. Feathering
                if strength > 0 and mode != "none":
                    sigma = strength * (overlap / 4.0)
                    sigma = max(0.5, sigma)
                    for m_idx in range(len(batch_masks)):
                        m_tens = batch_masks[m_idx]
                        if m_tens.min() < 0.99: # Only blur if there are edges
                            m_np = m_tens.squeeze(-1).cpu().numpy()
                            # Nearest mode keeps borders valid
                            m_blur = scipy.ndimage.gaussian_filter(m_np, sigma=sigma, mode='nearest')
                            batch_masks[m_idx] = torch.from_numpy(m_blur).unsqueeze(-1).to(device)

                # 4. Blend to Canvas
                for i, split in enumerate(splits):
                    y, x, h, w = split['y'], split['x'], split['h'], split['w']
                    tile = current_batch_tiles[i]
                    mask = batch_masks[i]
                    
                    canvas_crop = canvas[b:b+1, y:y+h, x:x+w, :]
                    
                    if mode == "multi_band" and strength > 0:
                        t_nchw = tile.permute(0, 3, 1, 2)
                        c_nchw = canvas_crop.permute(0, 3, 1, 2)
                        m_nchw = mask.permute(2, 0, 1).unsqueeze(0)
                        
                        blended = multi_band_blend(c_nchw, t_nchw, m_nchw, levels=5)
                        canvas[b:b+1, y:y+h, x:x+w, :] = blended.permute(0, 2, 3, 1)
                    else:
                        # Standard blending
                        canvas[b:b+1, y:y+h, x:x+w, :] = canvas_crop * (1.0 - mask) + tile * mask
                    
                    pbar.update(1)

            merged_img = canvas[:, :orig_H, :orig_W, :]

        # ================= LATENT MERGING =================
        merged_lat = None
        if processed_tiles_latent is not None and len(processed_tiles_latent) > 0:
            # Latent List: [{"samples": Tensor[B, 4, H, W]}, ...]
            ref_lat = processed_tiles_latent[0]['samples']
            device = ref_lat.device
            actual_batch_size = ref_lat.shape[0]
            lc = ref_lat.shape[1]
            
            scale = 8
            L_H, L_W = H // scale, W // scale
            L_orig_H, L_orig_W = orig_H // scale, orig_W // scale
            
            canvas_l = torch.zeros((actual_batch_size, lc, L_H, L_W), device=device)
            
            for b in range(actual_batch_size):
                for i, split in enumerate(splits):
                    if i >= len(processed_tiles_latent): break
                    
                    # Get Latent Tile for Batch b
                    # processed_tiles_latent[i] is {"samples": [B, C, H, W]}
                    tile = processed_tiles_latent[i]['samples'][b:b+1]
                    
                    y, x = split['y'], split['x']
                    ly, lx = y // scale, x // scale
                    lh, lw = tile.shape[2], tile.shape[3]
                    lo = overlap // scale
                    
                    # Latent Blending: Hard Cut at Center of Overlap
                    crop_t, crop_b = 0, lh
                    crop_l, crop_r = 0, lw
                    paste_y, paste_x = ly, lx
                    
                    # Logic: If neighbor exists, crop half overlap
                    # We rely on grid indices from split dict
                    r, c = split['row'], split['col']
                    
                    if c > 0: # Left neighbor exists
                        crop_l += lo // 2
                        paste_x += lo // 2
                    if r > 0: # Top neighbor exists
                        crop_t += lo // 2
                        paste_y += lo // 2
                        
                    # Determine right/bottom overlap presence based on grid size
                    # n_cols is stored in config["cols"] if needed, or check neighbor
                    if grid_map.get((r, c + 1)) is not None:
                        crop_r -= lo // 2
                    if grid_map.get((r + 1, c)) is not None:
                        crop_b -= lo // 2
                        
                    if crop_r > crop_l and crop_b > crop_t:
                        snippet = tile[:, :, crop_t:crop_b, crop_l:crop_r]
                        ph, pw = snippet.shape[2], snippet.shape[3]
                        
                        # Boundary safe paste
                        y1 = max(0, paste_y)
                        x1 = max(0, paste_x)
                        y2 = min(L_H, paste_y + ph)
                        x2 = min(L_W, paste_x + pw)
                        
                        # Corresponding snippet coords
                        sy1 = y1 - paste_y
                        sx1 = x1 - paste_x
                        sy2 = sy1 + (y2 - y1)
                        sx2 = sx1 + (x2 - x1)
                        
                        if y2 > y1 and x2 > x1:
                            canvas_l[b:b+1, :, y1:y2, x1:x2] = snippet[:, :, sy1:sy2, sx1:sx2]

            merged_samples = canvas_l[:, :, :L_orig_H, :L_orig_W]
            merged_lat = {"samples": merged_samples}
            
        return (merged_img, merged_lat)
