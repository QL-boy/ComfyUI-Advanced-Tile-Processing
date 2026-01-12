import torch
import comfy.utils
from ..utils.blending import generate_weight_mask, laplacian_pyramid_blending

class CustomTileMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_config": ("TILE_CONFIG",),
            },
            "optional": {
                "processed_tiles_image": ("IMAGE",),
                "processed_tiles_latent": ("LATENT",),
                "blend_mode": (["gaussian", "linear", "cosine", "laplacian", "none"], {"default": "gaussian"}),
                "feather_percent": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 50.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("merged_image", "merged_latent")
    FUNCTION = "merge_tiles"
    CATEGORY = "image/processing/tile"
    
    INPUT_IS_LIST = True

    def merge_tiles(self, tile_config, processed_tiles_image=None, processed_tiles_latent=None, blend_mode=None, feather_percent=None):
        # Unpack params (since INPUT_IS_LIST=True, even single inputs are lists)
        tile_config = tile_config[0]
        blend_mode = blend_mode[0] if blend_mode else "gaussian"
        feather = feather_percent[0] / 100.0 if feather_percent else 0.25
        
        merged_image = None
        merged_latent = None
        
        splits = tile_config["splits"]
        batch_size = tile_config["batch_size"]
        
        # Restore padded dimensions for canvas
        H, W = tile_config["padded_height"], tile_config["padded_width"]
        orig_H, orig_W = tile_config["original_height"], tile_config["original_width"]

        # ----------------------------------------------------------------------
        # Helper: Flatten Input Lists
        # ----------------------------------------------------------------------
        def flatten_input(input_data, is_latent=False):
            flat = []
            if input_data is None: return flat
            
            for item in input_data:
                if item is None: continue
                if is_latent:
                    # item is {'samples': Tensor}
                    samples = item['samples']
                    for b in range(samples.shape[0]):
                        flat.append(samples[b:b+1]) # Keep [1, C, H, W]
                else:
                    # item is Tensor [B, H, W, C]
                    for b in range(item.shape[0]):
                        flat.append(item[b:b+1]) # Keep [1, H, W, C]
            return flat

        # ----------------------------------------------------------------------
        # Logic: Image Merging
        # ----------------------------------------------------------------------
        tiles_img = flatten_input(processed_tiles_image, is_latent=False)
        
        if len(tiles_img) > 0:
            # Check consistency
            expected = len(splits) * batch_size
            if len(tiles_img) != expected:
                print(f"[Warning] TileMerger: Expected {expected} tiles, got {len(tiles_img)}.")

            if blend_mode == "laplacian":
                # --- Laplacian Pyramid Blending ---
                # This mode accumulates whole pyramids, so it is VRAM heavy.
                # We process one Batch item at a time to save memory.
                
                final_canvases = []
                device = tiles_img[0].device
                
                pbar = comfy.utils.ProgressBar(batch_size)
                
                for b in range(batch_size):
                    # Gather tiles for this batch
                    batch_tiles_list = []
                    batch_splits_info = []
                    
                    start_idx = b * len(splits)
                    for i, (y, x, h, w) in enumerate(splits):
                        t_idx = start_idx + i
                        if t_idx < len(tiles_img):
                            batch_tiles_list.append(tiles_img[t_idx])
                            batch_splits_info.append((y, x, h, w))
                    
                    if not batch_tiles_list: continue

                    # Execute Pyramid Blending
                    # shape: (H, W, C) -> (C, H, W) for calc, then back
                    # The util function expects tensors in [C, H, W] or [1, C, H, W]
                    
                    # 1. Permute tiles to [1, C, H, W]
                    permuted_tiles = [t.permute(0, 3, 1, 2) for t in batch_tiles_list]
                    
                    merged_tensor = laplacian_pyramid_blending(
                        permuted_tiles, 
                        batch_splits_info, 
                        (H, W), 
                        feather=feather, 
                        device=device
                    )
                    
                    # 2. Permute back to [1, H, W, C]
                    final_canvases.append(merged_tensor.permute(0, 2, 3, 1))
                    pbar.update(1)

                if final_canvases:
                    merged_image = torch.cat(final_canvases, dim=0)

            else:
                # --- Standard Weighted Average (Gaussian/Linear/Cosine) ---
                C = tiles_img[0].shape[-1]
                device = tiles_img[0].device
                
                canvas = torch.zeros((batch_size, H, W, C), device=device)
                weight_map = torch.zeros((batch_size, H, W, C), device=device)
                
                # Precompute base mask (assuming uniform tile size)
                th, tw = tiles_img[0].shape[1], tiles_img[0].shape[2]
                base_mask = generate_weight_mask((th, tw), blend_mode, feather, device)
                base_mask = base_mask.unsqueeze(0).unsqueeze(-1) # [1, H, W, 1]

                pbar = comfy.utils.ProgressBar(len(tiles_img))
                
                for b in range(batch_size):
                    for i, (y, x, h, w) in enumerate(splits):
                        idx = b * len(splits) + i
                        if idx >= len(tiles_img): break
                        
                        tile = tiles_img[idx]
                        curr_h, curr_w = tile.shape[1], tile.shape[2]
                        
                        # Handle Dynamic Resize (Upscale)
                        scale_h = curr_h / tile_config["tile_height"]
                        scale_w = curr_w / tile_config["tile_width"]
                        
                        # Adjust target coords
                        tgt_y = int(y * scale_h)
                        tgt_x = int(x * scale_w)
                        tgt_h, tgt_w = curr_h, curr_w
                        
                        # If canvas size changed due to upscale
                        if canvas.shape[1] < int(H * scale_h):
                            new_H = int(H * scale_h)
                            new_W = int(W * scale_w)
                            # Re-init canvas
                            canvas = torch.zeros((batch_size, new_H, new_W, C), device=device)
                            weight_map = torch.zeros((batch_size, new_H, new_W, C), device=device)
                        
                        if curr_h != th or curr_w != tw:
                            curr_mask = generate_weight_mask((curr_h, curr_w), blend_mode, feather, device)
                            curr_mask = curr_mask.unsqueeze(0).unsqueeze(-1)
                        else:
                            curr_mask = base_mask
                            
                        # Accumulate
                        canvas[b:b+1, tgt_y:tgt_y+tgt_h, tgt_x:tgt_x+tgt_w, :] += tile * curr_mask
                        weight_map[b:b+1, tgt_y:tgt_y+tgt_h, tgt_x:tgt_x+tgt_w, :] += curr_mask
                        
                        pbar.update(1)

                merged_image = canvas / (weight_map + 1e-6)

            # Crop Padding if exists
            if merged_image is not None:
                if orig_H != H or orig_W != W:
                    merged_image = merged_image[:, :orig_H, :orig_W, :]

        # ----------------------------------------------------------------------
        # Logic: Latent Merging
        # ----------------------------------------------------------------------
        # (Latent merging rarely benefits from Laplacian blending due to non-spatial nature of VAE channels,
        # so we default to Standard Weighted for stability, or mirror simple blending)
        
        tiles_lat = flatten_input(processed_tiles_latent, is_latent=True)
        if len(tiles_lat) > 0:
            # Latent typically 1/8 scale
            scale = 8
            L_H, L_W = H // scale, W // scale
            L_orig_H, L_orig_W = orig_H // scale, orig_W // scale
            
            C = tiles_lat[0].shape[1]
            device = tiles_lat[0].device
            
            canvas_l = torch.zeros((batch_size, C, L_H, L_W), device=device)
            weight_l = torch.zeros((batch_size, 1, L_H, L_W), device=device)
            
            th, tw = tiles_lat[0].shape[2], tiles_lat[0].shape[3]
            # Use same blend mode but simpler mask gen
            mask_l = generate_weight_mask((C, th, tw), blend_mode, feather, device)
            
            for b in range(batch_size):
                for i, (y, x, h, w) in enumerate(splits):
                    idx = b * len(splits) + i
                    if idx >= len(tiles_lat): break
                    
                    tile = tiles_lat[idx]
                    ly, lx = y // scale, x // scale
                    lh, lw = tile.shape[2], tile.shape[3]
                    
                    # Safety bounds
                    ly = min(ly, L_H)
                    lx = min(lx, L_W)
                    lh = min(lh, L_H - ly)
                    lw = min(lw, L_W - lx)
                    
                    # Accumulate
                    canvas_l[b:b+1, :, ly:ly+lh, lx:lx+lw] += tile * mask_l
                    weight_l[b:b+1, :, ly:ly+lh, lx:lx+lw] += mask_l

            merged_samples = canvas_l / (weight_l + 1e-6)
            
            # Crop Latent Padding
            if L_orig_H != L_H or L_orig_W != L_W:
                merged_samples = merged_samples[:, :, :L_orig_H, :L_orig_W]
                
            merged_latent = {"samples": merged_samples}

        return (merged_image, merged_latent)
