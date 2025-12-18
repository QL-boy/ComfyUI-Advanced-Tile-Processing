import torch
import math

def generate_weight_mask(shape, blend_mode="gaussian", feather=0.25, device="cpu"):
    """
    生成用于分块融合的权重遮罩
    shape: (H, W) or (C, H, W)
    blend_mode: 'linear', 'gaussian', 'cosine', 'none'
    feather: 0.0 - 1.0 (羽化范围比例)
    """
    if len(shape) == 3:
        h, w = shape[1], shape[2]
    else:
        h, w = shape[0], shape[1]
        
    # 创建基础网格
    # linspace 生成从 0 到 1 的线性空间
    x = torch.linspace(-1, 1, w, device=device)
    y = torch.linspace(-1, 1, h, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # 调整羽化范围，feather越大，有效中心区域越小，边缘衰减越宽
    # 我们通过缩放网格坐标来控制边缘
    if feather <= 0:
        mask = torch.ones((h, w), device=device)
    else:
        # 映射使得边缘 feather% 的区域从 1 衰减到 0
        # 这里的逻辑是：在 1-feather 范围内是 1，超过这个范围开始衰减
        scale = 1.0 / (1.0 - feather + 1e-6) # 避免除零
        
        if blend_mode == "gaussian":
            # 高斯分布: exp(-(x^2 + y^2) / sigma)
            # 调整 sigma 使得在边缘处接近 0
            d = torch.sqrt(grid_x**2 + grid_y**2)
            sigma = 0.5  # 标准差
            mask = torch.exp(-(d**2) / (2 * sigma**2))
            # 归一化到 0-1 并截断边缘
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            
        elif blend_mode == "cosine":
            # 余弦窗口
            mask_x = torch.cos(grid_x * math.pi / 2)
            mask_y = torch.cos(grid_y * math.pi / 2)
            mask = mask_x * mask_y
            mask = torch.clamp(mask, 0, 1)
            
        elif blend_mode == "linear":
            # 线性金字塔
            mask_x = 1.0 - torch.abs(grid_x)
            mask_y = 1.0 - torch.abs(grid_y)
            mask = torch.min(mask_x, mask_y)
            mask = torch.clamp(mask, 0, 1)
            
        else: # none
            mask = torch.ones((h, w), device=device)

    # 扩展维度以匹配 Image (H,W,C) 或 Latent (C,H,W)
    # 假设输入 shape 决定了输出 mask 的维度
    if len(shape) == 3:
        # Latent: (C, H, W), mask 需要是 (1, H, W) 用于广播
        return mask.unsqueeze(0)
    else:
        # Image 通常在 Comfy 中处理时如果是单张 mask 是 (H, W)
        return mask

def get_tiled_splits(full_height, full_width, tile_size, overlap):
    """
    计算分块坐标。采用 "覆盖优先" 策略，最后一块可能会有较大的重叠，
    以避免产生极小的边缘切片。
    Returns: list of (y, x, h, w)
    """
    stride = tile_size - overlap
    splits = []
    
    # 简单的网格生成
    y_coords = []
    current_y = 0
    while current_y < full_height:
        y_coords.append(current_y)
        if current_y + tile_size >= full_height:
            break
        current_y += stride
    
    # 修正最后一个坐标，确保不越界且覆盖边缘
    if y_coords[-1] + tile_size > full_height:
        y_coords[-1] = max(0, full_height - tile_size)
        
    x_coords = []
    current_x = 0
    while current_x < full_width:
        x_coords.append(current_x)
        if current_x + tile_size >= full_width:
            break
        current_x += stride
        
    if x_coords[-1] + tile_size > full_width:
        x_coords[-1] = max(0, full_width - tile_size)

    for y in y_coords:
        for x in x_coords:
            # 实际切片大小（通常等于 tile_size，除非原图小于 tile_size）
            h = min(tile_size, full_height - y)
            w = min(tile_size, full_width - x)
            splits.append((y, x, h, w))
            
    return splits