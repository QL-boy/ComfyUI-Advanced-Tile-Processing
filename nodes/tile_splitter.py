import torch
import torch.nn.functional as F
import math

class CustomTileSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "rows": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
                "columns": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
                "normalize": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("TILE_CONFIG", "IMAGE", "IMAGE", "LATENT", "LATENT")
    RETURN_NAMES = ("tile_config", "tiles_image_batch", "tiles_image_list", "tiles_latent_batch", "tiles_latent_list")
    OUTPUT_IS_LIST = (False, False, True, False, True)
    
    FUNCTION = "split_tiles"
    CATEGORY = "image/postprocessing"

    def split_tiles(self, tile_size=512, overlap=64, rows=0, columns=0, normalize=True, image=None, latent=None):
        if image is None and latent is None:
            return ({}, None, [], None, [])

        # 1. Base Dimensions
        base_h, base_w = 0, 0
        batch_size = 1
        is_image_input = image is not None
        
        if is_image_input:
            batch_size, base_h, base_w, _ = image.shape
        else:
            batch_size, _, l_h, l_w = latent['samples'].shape
            base_h, base_w = l_h * 8, l_w * 8

        # 2. Padding Logic
        align_step = 64 if normalize else 8
        target_h = max(base_h, tile_size)
        target_w = max(base_w, tile_size)
        
        if normalize:
            target_h = math.ceil(target_h / align_step) * align_step
            target_w = math.ceil(target_w / align_step) * align_step
            
        pad_h = target_h - base_h
        pad_w = target_w - base_w
        padded_h, padded_w = target_h, target_w

        # 3. Grid Calculation
        if overlap < 8: overlap = 8 

        if rows > 0 and columns > 0:
            n_rows = rows
            n_cols = columns
            tile_h = math.ceil((padded_h + (rows - 1) * overlap) / rows)
            tile_w = math.ceil((padded_w + (columns - 1) * overlap) / columns)
            tile_h = math.ceil(tile_h / 8) * 8
            tile_w = math.ceil(tile_w / 8) * 8
        else:
            tile_h, tile_w = tile_size, tile_size
            stride_h = tile_h - overlap
            stride_w = tile_w - overlap
            n_rows = math.ceil((padded_h - overlap) / stride_h) if stride_h > 0 else 1
            n_cols = math.ceil((padded_w - overlap) / stride_w) if stride_w > 0 else 1

        # 4. Generate Splits
        splits = []
        y_coords = []
        if n_rows == 1: y_coords = [0]
        else:
            max_y = padded_h - tile_h
            for r in range(n_rows):
                if n_rows > 1: y = int(round(r * (max_y / (n_rows - 1))))
                else: y = 0
                y = (y // 8) * 8 
                y_coords.append(y)
                
        x_coords = []
        if n_cols == 1: x_coords = [0]
        else:
            max_x = padded_w - tile_w
            for c in range(n_cols):
                if n_cols > 1: x = int(round(c * (max_x / (n_cols - 1))))
                else: x = 0
                x = (x // 8) * 8
                x_coords.append(x)

        for r_idx, y in enumerate(y_coords):
            for c_idx, x in enumerate(x_coords):
                splits.append({
                    "y": y, "x": x, 
                    "h": tile_h, "w": tile_w,
                    "row": r_idx, "col": c_idx
                })

        # 5. Data Extraction
        out_img_batch = None
        out_img_list = []
        out_lat_batch = None
        out_lat_list = []

        if is_image_input:
            img_padded = image.permute(0, 3, 1, 2)
            if pad_h > 0 or pad_w > 0:
                img_padded = F.pad(img_padded, (0, pad_w, 0, pad_h), mode='reflect')
            img_padded = img_padded.permute(0, 2, 3, 1)

            for s in splits:
                y, x, h, w = s["y"], s["x"], s["h"], s["w"]
                # Boundary check not strictly needed with reflect pad but good for safety
                h = min(h, padded_h - y)
                w = min(w, padded_w - x)
                tile = img_padded[:, y:y+h, x:x+w, :]
                out_img_list.append(tile)
            
            if len(out_img_list) > 0:
                try: out_img_batch = torch.cat(out_img_list, dim=0)
                except: out_img_batch = out_img_list[0] 

        if latent is not None:
            samples = latent['samples']
            l_pad_h = pad_h // 8
            l_pad_w = pad_w // 8
            if l_pad_h > 0 or l_pad_w > 0:
                samples = F.pad(samples, (0, l_pad_w, 0, l_pad_h), mode='reflect')
            
            lat_tiles = []
            for s in splits:
                y, x, h, w = s["y"], s["x"], s["h"], s["w"]
                ly, lx = y // 8, x // 8
                lh, lw = h // 8, w // 8
                tile = samples[:, :, ly:ly+lh, lx:lx+lw]
                lat_tiles.append(tile)
            
            out_lat_list = [{"samples": t} for t in lat_tiles]
            if len(lat_tiles) > 0:
                try: out_lat_batch = {"samples": torch.cat(lat_tiles, dim=0)}
                except: out_lat_batch = {"samples": lat_tiles[0]}

        tile_config = {
            "original_height": base_h,
            "original_width": base_w,
            "padded_height": padded_h,
            "padded_width": padded_w,
            "tile_height": tile_h,
            "tile_width": tile_w,
            "overlap": overlap,
            "splits": splits,
            "batch_size": batch_size,
            "rows": n_rows,
            "cols": n_cols
        }

        return (tile_config, out_img_batch, out_img_list, out_lat_batch, out_lat_list)
