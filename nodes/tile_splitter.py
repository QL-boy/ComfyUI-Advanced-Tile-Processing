import torch
import torch.nn.functional as F
from ..utils.blending import get_tiled_splits

class CustomTileSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "force_multiple_of_8": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),   # [B, H, W, C]
                "latent": ("LATENT",), # {'samples': [B, C, H, W]}
            }
        }

    # Strict UI Ordering: Config first, then Image Batch/List, then Latent Batch/List
    RETURN_TYPES = ("TILE_CONFIG", "IMAGE", "IMAGE", "LATENT", "LATENT")
    RETURN_NAMES = ("tile_config", "tiles_image_batch", "tiles_image_list", "tiles_latent_batch", "tiles_latent_list")
    
    # OUTPUT_IS_LIST: Config(False), ImgBatch(False), ImgList(True), LatBatch(False), LatList(True)
    OUTPUT_IS_LIST = (False, False, True, False, True)
    
    FUNCTION = "split_tiles"
    CATEGORY = "image/processing/tile"

    def split_tiles(self, tile_size=512, overlap=64, force_multiple_of_8=True, image=None, latent=None):
        tiles_image_batch = None
        tiles_image_list = []
        tiles_latent_batch = None
        tiles_latent_list = []
        
        full_h, full_w = 0, 0
        processing_image = False
        processing_latent = False
        batch_size = 1
        original_h, original_w = 0, 0
        
        # Smart Overlap: If 0, default to 1/8 tile size
        if overlap == 0:
            overlap = max(8, tile_size // 8)

        # --- Image Preprocessing ---
        if image is not None:
            processing_image = True
            batch_size, h, w, c = image.shape
            original_h, original_w = h, w
            
            if force_multiple_of_8 and (h % 8 != 0 or w % 8 != 0):
                pad_h = (8 - h % 8) % 8
                pad_w = (8 - w % 8) % 8
                # Pad format: (left, right, top, bottom) for last 2 dims if NCHW, 
                # but input image is NHWC. Torch pad works on last dims.
                # Use permute to NCHW for padding efficiency or pad manually.
                # Let's use simple padding on tensor
                image = F.pad(image, (0, 0, 0, pad_w, 0, pad_h), mode='reflect')
                full_h, full_w = image.shape[1], image.shape[2]
            else:
                full_h, full_w = h, w
        
        # --- Latent Preprocessing ---
        if latent is not None:
            processing_latent = True
            samples = latent['samples'] # [B, C, H, W]
            l_batch, l_c, l_h, l_w = samples.shape
            
            if not processing_image:
                # Estimate dimensions from latent
                scale_factor = 8 
                original_h, original_w = l_h * scale_factor, l_w * scale_factor
                full_h, full_w = original_h, original_w # Assume latent is already aligned or we track it
                
                # If forcing multiple of 8 on latent? Latents are usually 1/8th.
                # If image wasn't provided, we assume latent is the source of truth.
                # We do not pad latent here unless necessary, but usually VAE handles it.
                # Let's stick to using pixel dimensions for calculations.
                
                tile_size_act = tile_size // scale_factor
                overlap_act = overlap // scale_factor
            else:
                # If image exists, latent must match (approx).
                # If image was padded, latent should ideally match that padded structure,
                # but VAE encoding usually handles padding. 
                # We will just split latent based on the pixel-space split logic scaled down.
                pass

        if not processing_image and not processing_latent:
            raise ValueError("No input provided (Image or Latent required)")

        # Calculate Splits (on pixel space)
        splits = get_tiled_splits(full_h, full_w, tile_size, overlap)
        
        tile_config = {
            "original_height": original_h, # Unpadded size
            "original_width": original_w,
            "padded_height": full_h,       # Padded size
            "padded_width": full_w,
            "tile_height": tile_size,
            "tile_width": tile_size,
            "overlap": overlap,
            "splits": splits,
            "batch_size": batch_size,
            "force_multiple_of_8": force_multiple_of_8
        }

        # --- Slicing Image ---
        if processing_image:
            img_tiles = []
            for (y, x, h, w) in splits:
                tile = image[:, y:y+h, x:x+w, :]
                img_tiles.append(tile)
            
            tiles_image_list = img_tiles
            try:
                tiles_image_batch = torch.cat(img_tiles, dim=0)
            except:
                pass # Should not happen if tiles are same size

        # --- Slicing Latent ---
        if processing_latent:
            lat_tiles = []
            samples = latent['samples']
            scale = 8
            
            for (y, x, h, w) in splits:
                # Map pixel coords to latent coords
                ly, lx = y // scale, x // scale
                lh, lw = h // scale, w // scale
                
                # Safety check for latent boundaries
                # (When padding image, latent might not be perfectly aligned if not encoded from padded image)
                # We clamp to max latent dims
                max_lh, max_lw = samples.shape[2], samples.shape[3]
                ly = min(ly, max_lh)
                lx = min(lx, max_lw)
                lh = min(lh, max_lh - ly)
                lw = min(lw, max_lw - lx)

                tile = samples[:, :, ly:ly+lh, lx:lx+lw]
                lat_tiles.append(tile)
            
            tiles_latent_list = [{"samples": t} for t in lat_tiles]
            if len(lat_tiles) > 0:
                # Only if all tiles same size
                try:
                    tiles_latent_batch = {"samples": torch.cat(lat_tiles, dim=0)}
                except:
                    tiles_latent_batch = None

        return (tile_config, tiles_image_batch, tiles_image_list, tiles_latent_batch, tiles_latent_list)
