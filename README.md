# ComfyUI-Advanced-Tile-Processing

**ComfyUI-Advanced-Tile-Processing** 是一套专为 ComfyUI  设计的高级自定义节点插件。它旨在解决超高分辨率图像生成和处理中的显存瓶颈（VRAM OOM），通过智能分块（Tiling）与基于加权累积的无缝合并（Merging）技术，让您能够在普通消费级显卡上稳定处理 4K、8K 甚至更高规格的图像。

## 🌟 核心特性

- **智能维度感知 (Smart Dimension Awareness)**：自动计算最优切片坐标。支持“边缘回退”策略，确保分块完全覆盖原始图像，无需引入黑边（Padding）从而避免模型产生伪影。
- **无缝加权融合 (Soft Blending)**：内置多种混合模式（Gaussian, Cosine, Linear）。通过多层重叠区域的加权平均，最大化消除物理拼接产生的“网格接缝”。
- **元数据驱动流 (Metadata-Driven Workflow)**：Splitter 生成唯一的 `TILE_CONFIG` 对象，包含原始尺寸、缩放因子及坐标。Merger 自动读取配置一键还原，用户无需手动对齐参数。
- **深度兼容性**：完美适配 ComfyUI 的列表执行机制。支持将分块作为 `BATCH` 提交以获得最高推理速度，或作为 `LIST` 提交以支持循环节点处理。显存不足的情况下推荐 `LIST` 替代 `BATCH` 。
- **高性能架构**：完全基于 PyTorch 张量运算，GPU 加速，保证处理处理大图时性能在线。
- **动态尺寸支持 (Tiled Upscale)**：支持动态尺寸调整（Tiled Upscale），在分块后进行放大处理。Merger 能够自动检测片段尺寸变化并动态调整画布大小。

## 🛠 安装说明

1. **环境要求**：ComfyUI 0.4.0+，Python 3.10+，PyTorch 2.0+。

2. 进入您的 ComfyUI 自定义节点目录：

```
cd ComfyUI/custom_nodes/
```

3. 克隆本仓库：
```
git clone [https://github.com/QL-boy/ComfyUI-Advanced-Tile-Processing.git](https://github.com/QL-boy/ComfyUI-Advanced-Tile-Processing.git)
```

4. 重启 ComfyUI。


## 🧩 节点详解

### 1. 🔧 Advanced Tile Splitter (分块器)
![](./nodes/tile_splitter.png)
将输入的大图或 Latent 空间分割成重叠的小块。

- **输入端口**:

    - `image`: (可选) 原始图像。
    - `latent`: (可选) 潜在空间数据。

- **核心参数**:
    
    - `tile_size`: 分块分辨率（如 512, 1024）。
    - `overlap`: 两个分块之间的重叠像素。建议设为 `tile_size` 的 10% 及以上以获得最佳融合效果。

- **输出端口**:

    - `tiles_image_batch`: 将所有分块合并为一个 Batch Tensor，适合高性能采样。
    - `tiles_image_list`: (List 模式) 分块图像列表，触发 ComfyUI 循环执行。
    - `tiles_latent_batch / list`: 对应 Latent 空间的输出。
    - `tile_config`: 核心配置文件（必需连接至 Merger），包含合并所需的所有元数据。

### 2. 🔧 Advanced Tile Merger (合并器)
![](./nodes/tile_merger.png)
根据分块配置将处理后的片段无缝还原。

- **输入端口**:
    
    - `tile_config`: 由 Splitter 输出的配置对象。
    - `processed_tiles_image`: (可选) 处理后的图像片段。
    - `processed_tiles_latent`: (可选) 处理后的 Latent 片段。

- **核心参数**:
    
    - `blend_mode`:
        - `gaussian` (默认): 中心权重高，边缘平滑衰减，融合效果最自然。
        - `cosine`: 经典的三角函数衰减，过渡区域较宽。
        - `linear`: 线性金字塔混合。
		- `none`: 硬边缘拼接（会有明显接缝）。
    - `feather_percent`: 羽化比例 (0-50%)，控制权重衰减的起始位置，值越大，边缘融合区域越宽。

- 内部机制:

    节点会自动解包 `LIST`  输入。如果您的上游是循环节点，请确保所有分块均已处理完成再输入 Merger。

## 📖 核心算法原理：加权融合（Soft Blending / 累积缓冲加权平均法）

为消除传统“剪切-粘贴”方式产生的接缝与亮度突变，本插件采用**累积缓冲加权平均法（Accumulation Buffer Weighted Averaging）**，实现多层像素在重叠区域的平滑融合。其核心流程如下：

1. **掩码生成（Weight Mask Generation）**  
   为每个图像分块（Tile）生成一个权重掩码矩阵 $M$。在 Gaussian 模式下，中心点权重为 1.0，并向边缘按指数规律衰减至接近 0，确保融合时中心区域贡献最大、边缘逐渐过渡。
2. **像素累加与权重累加**  
   - **像素累加**：在全局画布对应位置累加当前分块加权后的像素值：  
     $Canvas += Tile \times M$
   - **权重累加**：同步在权重分布图（Weight Map）中累加掩码值：  
     $WeightMap += M$
3. **归一化输出（Normalization）**  
   所有分块累加完成后，对每个像素进行归一化计算，得到最终图像：  
   $Pixel_{final} = \frac{Canvas}{WeightMap + \epsilon} = \frac{\sum (Tile_i \times M_i)}{\sum M_i + \epsilon}$
   其中 $\epsilon$ 是为避免除零而设置的极小常数。
### 关键优势
- **无缝过渡**：重叠区域由多个 Tile 按权重共同贡献，通过归一化保持亮度与对比度一致；
- **抗伪影**：权重平滑变化避免硬边缘，实现视觉上的自然渐变；
- **灵活性**：可通过调整权重衰减曲线适应不同分块重叠策略。

该方法本质上是一种**软混合（Soft Blending）**，通过在重叠区域进行多图层像素的加权平均，达成高质量、无接缝的大图合成效果。
## 🚀 示例工作流
### SDXL 经典分块采样高清修复复刻
![](<./example_workflows/SDXL 经典分块采样高清修复复刻.png>)

### Z-Image质量升级
![](./example_workflows/Z-Image质量升级.png)

### 

## ⚠️ 常见问题 (FAQ)

- **Q: 为什么合并后的图边缘还是有淡淡的印子？**

    - A: 尝试增加 `overlap` (建议不低于 64) 并确保 `blend_mode` 设为 `gaussian`。同时，检查重绘时的 `denoise` 是否过高，过高的降噪会导致分块内容产生剧烈变化。

- **Q: 支持 Latent 空间合并吗？**

    - A: 支持。但由于 VAE 编码器的特性，Latent 空间的重叠合并可能会在 Decode 后出现微小色差，通常推荐在 Image 空间执行最终合并。

- **Q: 为什么我连接了节点却输出了多张图？**

    - A: 请确保您使用的是 `Advanced Tile Merger` 并正确连接了 `tile_config`。如果输出依然是列表，请检查是否在 Merger 之后又连接了不支持 List 的旧版节点。

## 🤝 贡献与反馈

欢迎提交 Issue 或 Pull Request。如果您在使用中发现本项目可以改进的地方，请随时联系。

## 📜 许可说明

本项目基于 [Apache-2.0 license](https://github.com/QL-boy/comfyui-ps-plugin#Apache-2.0-1-ov-file) 许可证开源。
