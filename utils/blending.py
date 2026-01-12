import torch
import torch.nn.functional as F
import math

def generate_weight_mask(shape, blend_mode="gaussian", feather=0.25, device="cpu"):
    """
    Generates a weight mask with correct feathering logic.
    shape: (H, W) or (C, H, W)
    """
    if len(shape) == 3:
        h, w = shape[1], shape[2]
    else:
        h, w = shape[0], shape[1]
        
    # Grid: -1 to 1
    x = torch.linspace(-1, 1, w, device=device)
    y = torch.linspace(-1, 1, h, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # Feather calculation
    # We want a plateau of 1.0 in the center, decaying to 0.0 at edges.
    # The 'feather' parameter defines the % of the radius that is decaying.
    # Distance from center d: 0 -> 1
    # Effective feather region starts at 1 - feather
    
    # Use max dist on either axis (Linear box falloff) or Euclidean (Circular)
    # Usually box falloff is better for rectangular tiles
    
    dist_x = torch.abs(grid_x)
    dist_y = torch.abs(grid_y)
    
    # Combine distances based on mode
    if blend_mode == "gaussian":
        # Radial Euclidean distance for smooth circular falloff
        d = torch.sqrt(dist_x**2 + dist_y**2)
        # Sigma derived from feather. 
        # If feather=0.5, we want 1.0 at d=0.5 and near 0 at d=1.0
        # This is hard to map exactly to feather %.
        # Let's switch to a robust sigmoid-like falloff for Gaussian emulation or use standard Gaussian.
        sigma = 0.5 
        mask = torch.exp(-(d**2) / (2 * sigma**2))
        
    elif blend_mode == "cosine":
        # Cosine window
        mask_x = torch.cos(grid_x * math.pi / 2)
        mask_y = torch.cos(grid_y * math.pi / 2)
        mask = mask_x * mask_y
        
    else: # linear or default (box-like feather)
        # Calculates how far we are from the edge (0=center, 1=edge)
        # We want: 0 to (1-feather) -> Value 1
        #          (1-feather) to 1   -> Value 1 down to 0
        
        # Max distance on either axis
        d = torch.max(dist_x, dist_y)
        
        # Threshold
        start_decay = 1.0 - feather
        if start_decay >= 1.0: 
            mask = torch.ones_like(d)
        else:
            # Linear ramp: (d - start) / (1 - start) -> 0 to 1 in decay zone
            ramp = (d - start_decay) / (1.0 - start_decay + 1e-6)
            mask = 1.0 - torch.clamp(ramp, 0, 1)

    # Normalize mask range 0-1
    mask = torch.clamp(mask, 0, 1)

    if len(shape) == 3:
        return mask.unsqueeze(0)
    else:
        return mask

def get_tiled_splits(full_height, full_width, tile_size, overlap):
    """
    Calculates tile coordinates.
    Returns: list of (y, x, h, w)
    """
    stride = tile_size - overlap
    splits = []
    
    y_coords = []
    current_y = 0
    while current_y < full_height:
        y_coords.append(current_y)
        if current_y + tile_size >= full_height:
            break
        current_y += stride
    # Fix last
    if y_coords[-1] + tile_size > full_height:
        y_coords[-1] = max(0, full_height - tile_size)
        
    x_coords = []
    current_x = 0
    while current_x < full_width:
        x_coords.append(current_x)
        if current_x + tile_size >= full_width:
            break
        current_x += stride
    # Fix last
    if x_coords[-1] + tile_size > full_width:
        x_coords[-1] = max(0, full_width - tile_size)

    for y in y_coords:
        for x in x_coords:
            h = min(tile_size, full_height - y)
            w = min(tile_size, full_width - x)
            splits.append((y, x, h, w))
            
    return splits

# --- Laplacian Pyramid Blending Utilities ---

def gaussian_pyramid(img, levels):
    pyramid = [img]
    for i in range(levels - 1):
        # Blur and downsample
        # Kernel: [1, 4, 6, 4, 1] / 16 approximation
        # Using avg_pool for speed/memory or standard gaussian kernel
        # For seamless blending, simple bilinear downsample often suffices
        down = F.interpolate(pyramid[-1], scale_factor=0.5, mode='bilinear', align_corners=False)
        pyramid.append(down)
    return pyramid

def laplacian_pyramid(img, levels):
    gaussian_pyr = gaussian_pyramid(img, levels)
    laplacian_pyr = []
    for i in range(levels - 1):
        current = gaussian_pyr[i]
        next_up = F.interpolate(gaussian_pyr[i+1], size=current.shape[2:], mode='bilinear', align_corners=False)
        laplacian_pyr.append(current - next_up)
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr

def reconstruct_from_laplacian(lap_pyr):
    current = lap_pyr[-1]
    for i in range(len(lap_pyr) - 2, -1, -1):
        up = F.interpolate(current, size=lap_pyr[i].shape[2:], mode='bilinear', align_corners=False)
        current = lap_pyr[i] + up
    return current

def laplacian_pyramid_blending(tiles, splits, canvas_shape, feather=0.25, device="cpu"):
    """
    Performs Laplacian Pyramid Blending on a set of tiles.
    tiles: list of tensors [1, C, H, W]
    splits: list of (y, x, h, w)
    canvas_shape: (H, W)
    """
    H, W = canvas_shape
    C = tiles[0].shape[1]
    
    # Determine levels based on tile size
    min_dim = min(tiles[0].shape[2], tiles[0].shape[3])
    levels = int(math.log2(min_dim)) - 2 # Keep base ~4-8 pixels
    levels = max(1, min(levels, 5)) # Cap levels to save VRAM

    # Initialize Pyramids for the Canvas
    # We need to accumulate Weighted Laplacians and Weights for each level
    # Format: List of Tensors (one per level)
    canvas_L_pyr = [] # Numerator
    weight_G_pyr = [] # Denominator
    
    # Create empty canvas levels
    # Note: Dimensions change per level
    curr_h, curr_w = H, W
    for i in range(levels):
        canvas_L_pyr.append(torch.zeros((1, C, curr_h, curr_w), device=device))
        weight_G_pyr.append(torch.zeros((1, 1, curr_h, curr_w), device=device))
        curr_h = int(curr_h * 0.5)
        curr_w = int(curr_w * 0.5)

    # Process each tile
    for idx, tile in enumerate(tiles):
        y, x, h, w = splits[idx]
        
        # 1. Generate Mask for Tile
        # Use simple Linear feather for mask pyramid basis
        mask = generate_weight_mask((C, h, w), blend_mode="linear", feather=feather, device=device)
        mask = mask.unsqueeze(0) # [1, C, h, w] (broadcasts to C usually or 1)
        if mask.shape[1] != 1: mask = mask[:, 0:1, :, :] # Ensure [1, 1, h, w]

        # 2. Build Pyramids for Tile and Mask
        tile_L_pyr = laplacian_pyramid(tile, levels)
        mask_G_pyr = gaussian_pyramid(mask, levels) # Weight uses Gaussian pyr
        
        # 3. Accumulate into Canvas Pyramids
        for i in range(levels):
            t_L = tile_L_pyr[i]
            m_G = mask_G_pyr[i]
            
            # Dimensions at this level
            lvl_H, lvl_W = canvas_L_pyr[i].shape[2], canvas_L_pyr[i].shape[3]
            
            # Map coordinates to this level
            scale = 1.0 / (2**i)
            ly, lx = int(y * scale), int(x * scale)
            lh, lw = t_L.shape[2], t_L.shape[3]
            
            # Safety checks for rounding
            if ly + lh > lvl_H: lh = lvl_H - ly
            if lx + lw > lvl_W: lw = lvl_W - lx
            
            # Add to accumulators
            # L_final = Sum(L_i * W_i) / Sum(W_i)
            # Accumulate numerator
            canvas_L_pyr[i][:, :, ly:ly+lh, lx:lx+lw] += t_L[:, :, :lh, :lw] * m_G[:, :, :lh, :lw]
            # Accumulate denominator
            weight_G_pyr[i][:, :, ly:ly+lh, lx:lx+lw] += m_G[:, :, :lh, :lw]

    # Normalize each level
    normalized_L_pyr = []
    for i in range(levels):
        norm = canvas_L_pyr[i] / (weight_G_pyr[i] + 1e-6)
        normalized_L_pyr.append(norm)

    # Reconstruct
    return reconstruct_from_laplacian(normalized_L_pyr)
