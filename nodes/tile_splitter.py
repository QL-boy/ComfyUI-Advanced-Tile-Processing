import torch
from ..utils.blending import get_tiled_splits

class CustomTileSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
            },
            "optional": {
                "image": ("IMAGE",),   # [B, H, W, C]
                "latent": ("LATENT",), # {'samples': [B, C, H, W]}
            }
        }

    # 修复：将 LIST 类型改为实际的 IMAGE/LATENT 类型，以便 UI 正确连接
    RETURN_TYPES = ("IMAGE", "IMAGE", "LATENT", "LATENT", "TILE_CONFIG")
    RETURN_NAMES = ("tiles_image_batch", "tiles_image_list", "tiles_latent_batch", "tiles_latent_list", "tile_config")
    
    # 新增：明确定义哪些输出是列表。
    # 对应位置：(False, True, False, True, False)
    # tiles_image_batch (False), tiles_image_list (True), tiles_latent_batch (False), tiles_latent_list (True), tile_config (False)
    OUTPUT_IS_LIST = (False, True, False, True, False)
    
    FUNCTION = "split_tiles"
    CATEGORY = "image/processing/tile"

    def split_tiles(self, tile_size=512, overlap=64, image=None, latent=None):
        tiles_image_batch = None
        tiles_image_list = []
        tiles_latent_batch = None
        tiles_latent_list = []
        
        full_h, full_w = 0, 0
        processing_image = False
        processing_latent = False
        batch_size = 1
        
        if image is not None:
            processing_image = True
            batch_size, full_h, full_w, channels = image.shape
        
        if latent is not None:
            processing_latent = True
            # latent samples: [B, C, H, W]
            l_batch, l_c, l_h, l_w = latent['samples'].shape
            if not processing_image:
                # 如果没有 Image 参照，根据 Latent 估算像素尺寸 (x8)
                scale_factor = 8 
                l_tile_size = tile_size // scale_factor
                l_overlap = overlap // scale_factor
                
                full_h, full_w = l_h, l_w
                tile_size_act = l_tile_size
                overlap_act = l_overlap
            else:
                tile_size_act = tile_size
                overlap_act = overlap
        else:
            if processing_image:
                tile_size_act = tile_size
                overlap_act = overlap
        
        if not processing_image and not processing_latent:
            raise ValueError("No input provided (Image or Latent required)")

        splits = get_tiled_splits(full_h, full_w, tile_size_act, overlap_act)
        
        tile_config = {
            "original_height": full_h,
            "original_width": full_w,
            "tile_height": tile_size_act,
            "tile_width": tile_size_act,
            "overlap": overlap_act,
            "splits": splits,
            "is_latent_config": (not processing_image and processing_latent),
            "batch_size": batch_size
        }

        # Image Processing
        if processing_image:
            img_tiles = []
            for (y, x, h, w) in splits:
                tile = image[:, y:y+h, x:x+w, :]
                img_tiles.append(tile)
            
            tiles_image_list = img_tiles
            try:
                tiles_image_batch = torch.cat(img_tiles, dim=0)
            except:
                pass

        # Latent Processing
        if processing_latent:
            lat_tiles = []
            samples = latent['samples']
            is_pixel_config = processing_image
            scale = 8 if is_pixel_config else 1
            
            for (y, x, h, w) in splits:
                ly, lx = y // scale, x // scale
                lh, lw = h // scale, w // scale
                tile = samples[:, :, ly:ly+lh, lx:lx+lw]
                lat_tiles.append(tile)
            
            # Latent 列表中的每个元素都需要是标准的 Latent 字典格式
            tiles_latent_list = [{"samples": t} for t in lat_tiles]
            if len(lat_tiles) > 0:
                tiles_latent_batch = {"samples": torch.cat(lat_tiles, dim=0)}

        return (tiles_image_batch, tiles_image_list, tiles_latent_batch, tiles_latent_list, tile_config)