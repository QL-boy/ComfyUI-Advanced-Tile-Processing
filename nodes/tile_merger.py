import torch
import comfy.utils
from ..utils.blending import generate_weight_mask

class CustomTileMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_config": ("TILE_CONFIG",),
                "blend_mode": (["gaussian", "linear", "cosine", "none"], {"default": "gaussian"}),
                "feather_percent": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 50.0, "step": 1.0}),
            },
            "optional": {
                "processed_tiles_image": ("IMAGE",),
                "processed_tiles_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("merged_image", "merged_latent")
    FUNCTION = "merge_tiles"
    CATEGORY = "image/processing/tile"
    
    # 关键修复：添加 INPUT_IS_LIST = True
    # 这告诉 ComfyUI 不要自动遍历列表，而是将整个列表一次性传给 merge_tiles 函数
    INPUT_IS_LIST = True

    def merge_tiles(self, tile_config, blend_mode, feather_percent, processed_tiles_image=None, processed_tiles_latent=None):
        # 由于开启了 INPUT_IS_LIST = True，所有输入参数都会变成列表。
        # 我们需要解包非列表类型的参数（即那些我们只期望接收一个值的参数）。
        
        # 解包配置和参数
        tile_config = tile_config[0]
        blend_mode = blend_mode[0]
        feather = feather_percent[0] / 100.0
        
        merged_image = None
        merged_latent = None
        
        splits = tile_config["splits"]
        batch_size = tile_config["batch_size"]

        # --- 处理图像合并 ---
        # processed_tiles_image 是一个列表，可能包含：
        # 情况 A (List输入): [Tensor(1,H,W,C), Tensor(1,H,W,C), ...]
        # 情况 B (Batch输入): [Tensor(N,H,W,C)]
        # 我们将其统一展平为 [Tensor(1,H,W,C), ...] 的列表
        
        if processed_tiles_image is not None:
            tiles = []
            for t in processed_tiles_image:
                # t 是 Tensor [B, H, W, C]
                # 即使输入是 List，列表中的每个元素也是 Tensor
                # 如果输入是 Batch，列表只有一个元素，但该 Tensor 的 B > 1
                for b in range(t.shape[0]):
                    tiles.append(t[b:b+1])

            # 验证数量
            expected_tiles = len(splits) * batch_size
            if len(tiles) != expected_tiles:
                print(f"Warning: Expected {expected_tiles} tiles, got {len(tiles)}. Merge might be incomplete or misaligned.")

            # 初始化画布 [B, H, W, C]
            H, W = tile_config["original_height"], tile_config["original_width"]
            # 确保 tiles 非空
            if len(tiles) > 0:
                C = tiles[0].shape[-1]
                device = tiles[0].device
                
                canvas = torch.zeros((batch_size, H, W, C), device=device)
                weight_map = torch.zeros((batch_size, H, W, C), device=device)
                
                # 预计算 Mask
                tile_h, tile_w = tiles[0].shape[1], tiles[0].shape[2]
                base_mask = generate_weight_mask((tile_h, tile_w), blend_mode, feather, device)
                base_mask = base_mask.unsqueeze(0).unsqueeze(-1) # [1, H, W, 1]

                pbar = comfy.utils.ProgressBar(len(tiles))
                
                for b in range(batch_size):
                    for i, (y, x, h, w) in enumerate(splits):
                        tile_idx = b * len(splits) + i
                        if tile_idx >= len(tiles): break
                        
                        tile = tiles[tile_idx] # [1, Th, Tw, C]
                        
                        # 动态 Resize 支持 (处理 Tiled Upscale 的情况)
                        curr_h, curr_w = tile.shape[1], tile.shape[2]
                        
                        if curr_h != h or curr_w != w:
                             scale_h = curr_h / tile_config["tile_height"]
                             scale_w = curr_w / tile_config["tile_width"]
                             
                             target_H = int(H * scale_h)
                             target_W = int(W * scale_w)
                             
                             # 如果画布尺寸需要调整（仅第一次检测到时）
                             if canvas.shape[1] != target_H:
                                 canvas = torch.zeros((batch_size, target_H, target_W, C), device=device)
                                 weight_map = torch.zeros((batch_size, target_H, target_W, C), device=device)
                             
                             tgt_y = int(y * scale_h)
                             tgt_x = int(x * scale_w)
                             tgt_h, tgt_w = curr_h, curr_w
                             
                             curr_mask = generate_weight_mask((tgt_h, tgt_w), blend_mode, feather, device)
                             curr_mask = curr_mask.unsqueeze(0).unsqueeze(-1)
                        else:
                             tgt_y, tgt_x, tgt_h, tgt_w = y, x, h, w
                             curr_mask = base_mask

                        # 累加
                        canvas[b:b+1, tgt_y:tgt_y+tgt_h, tgt_x:tgt_x+tgt_w, :] += tile * curr_mask
                        weight_map[b:b+1, tgt_y:tgt_y+tgt_h, tgt_x:tgt_x+tgt_w, :] += curr_mask
                        
                        pbar.update(1)

                merged_image = canvas / (weight_map + 1e-6)

        # --- 处理 Latent 合并 ---
        if processed_tiles_latent is not None:
             lat_tiles = []
             # 同样展平 Latent 列表
             for t_dict in processed_tiles_latent:
                 # ComfyUI 的 LATENT 类型是字典 {'samples': Tensor}
                 if 'samples' in t_dict:
                     full_samples = t_dict['samples'] # [B, C, H, W]
                     for b in range(full_samples.shape[0]):
                         lat_tiles.append(full_samples[b:b+1])
             
             if len(lat_tiles) > 0:
                 is_image_config = not tile_config.get("is_latent_config", False)
                 scale = 8 if is_image_config else 1
                 
                 orig_H = tile_config["original_height"] // scale
                 orig_W = tile_config["original_width"] // scale
                 C = lat_tiles[0].shape[1]
                 device = lat_tiles[0].device
                 
                 canvas_l = torch.zeros((batch_size, C, orig_H, orig_W), device=device)
                 weight_map_l = torch.zeros((batch_size, 1, orig_H, orig_W), device=device)
                 
                 th, tw = lat_tiles[0].shape[2], lat_tiles[0].shape[3]
                 mask_l = generate_weight_mask((C, th, tw), blend_mode, feather, device)
                 
                 for b in range(batch_size):
                    for i, (y, x, h, w) in enumerate(splits):
                        tile_idx = b * len(splits) + i
                        if tile_idx >= len(lat_tiles): break
                        
                        tile = lat_tiles[tile_idx]
                        
                        ly, lx = y // scale, x // scale
                        lh, lw = tile.shape[2], tile.shape[3]
                        
                        canvas_l[b:b+1, :, ly:ly+lh, lx:lx+lw] += tile * mask_l
                        weight_map_l[b:b+1, :, ly:ly+lh, lx:lx+lw] += mask_l
                
                 merged_samples = canvas_l / (weight_map_l + 1e-6)
                 merged_latent = {"samples": merged_samples}

        return (merged_image, merged_latent)