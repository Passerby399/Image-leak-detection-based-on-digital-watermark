# 数字水印泄露检测系统技术报告

## 1. 算法概述
本系统基于**离散余弦变换(DCT)**实现了一种鲁棒的数字水印技术，用于图像版权保护和泄露检测。系统包含水印生成、嵌入、提取和泄露检测完整流程，并通过多种攻击测试验证水印的鲁棒性。

### 1.1 核心算法：DCT域水印嵌入
算法基于JPEG压缩标准中使用的DCT变换，在图像的频域中嵌入水印信息：

```math
\begin{aligned}
&\text{DCT变换:} \quad F(u,v) = c(u)c(v)\sum_{x=0}^{7}\sum_{y=0}^{7}f(x,y)\cos\frac{(2x+1)u\pi}{16}\cos\frac{(2y+1)v\pi}{16} \\
&\text{IDCT变换:} \quad f(x,y) = \sum_{u=0}^{7}\sum_{v=0}^{7}c(u)c(v)F(u,v)\cos\frac{(2x+1)u\pi}{16}\cos\frac{(2y+1)v\pi}{16} \\
&\text{其中} \quad c(k) = 
\begin{cases} 
\frac{1}{\sqrt{2}} & k=0 \\
1 & \text{otherwise}
\end{cases}
\end{aligned}
```

### 1.2 水印嵌入策略
- **频域选择**：在中频区域嵌入水印（低频对视觉影响大，高频对压缩敏感）
- **嵌入位置**：每个8×8块中的三个固定位置 `(2,2)`, `(4,2)`, `(2,4)`
- **嵌入公式**：`F'(u,v) = F(u,v) + α × W(i,j) × 100`
  - `α`: 水印强度因子（默认0.15）
  - `W(i,j)`: 水印像素值归一化到[0,1]

## 2. 系统架构
### 2.1 类结构：`DCTWatermarker`
```python
class DCTWatermarker:
    def __init__(self, strength=0.1, block_size=8, watermark_size=(64, 64)):
        # 参数初始化
    
    def generate_watermark(self, user_id, text=None):
        # 基于用户ID生成文本水印
    
    def embed_watermark(self, host_image, watermark_image):
        # 在宿主图像中嵌入水印
    
    def extract_watermark(self, watermarked_image, original_image=None):
        # 从图像中提取水印
    
    def detect_leak(self, original_watermark, extracted_watermark, threshold=0.7):
        # 检测水印泄露
    
    def test_robustness(self, watermarked_image, watermark_image, attacks):
        # 测试水印鲁棒性
```

### 2.2 处理流程
1. **图像预处理**：
   - 转换到YUV色彩空间，提取亮度通道(Y)
   - 分块处理（默认8×8块）
   
2. **水印生成**：
   ```mermaid
   graph TD
   A[用户ID] --> B[SHA256哈希]
   B --> C[取前16字符]
   C --> D[渲染为灰度图像]
   D --> E[水印图像]
   ```

3. **水印嵌入**：
   ```mermaid
   graph LR
   A[宿主图像] --> B[分块DCT]
   C[水印图像] --> D[降采样]
   B --> E[修改中频系数]
   D --> E
   E --> F[IDCT重构]
   F --> G[含水印图像]
   ```

4. **水印提取**：
   - 非盲提取：需要原始图像，计算DCT系数差值
   - 盲提取：直接读取DCT系数

## 3. 关键技术实现

### 3.1 水印生成算法
```python
def generate_watermark(self, user_id, text=None):
    # 生成哈希文本
    hash_obj = hashlib.sha256(user_id.encode())
    text = hash_obj.hexdigest()[:16]
    
    # 创建空白图像
    watermark = np.zeros((h, w), dtype=np.uint8)
    watermark[:] = 255  # 白色背景
    
    # 使用PIL添加文本
    pil_img = Image.fromarray(watermark)
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, fill=0)  # 黑色文本
    
    return np.array(pil_img)
```

### 3.2 水印嵌入核心逻辑
```python
for i in range(0, host_y.shape[0], self.block_size):
    for j in range(0, host_y.shape[1], self.block_size):
        # 获取当前块
        block = host_y[i:i+block_size, j:j+block_size]
        
        # DCT变换
        dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
        
        # 嵌入水印
        for pos in self.positions:
            dct_block[pos] += self.strength * wm_value * 100
        
        # IDCT重构
        idct_block = idct(idct(dct_block, axis=0, norm='ortho'), axis=1, norm='ortho')
```

### 3.3 泄露检测算法
```python
def detect_leak(self, original, extracted, threshold=0.7):
    # 二值化处理
    _, orig_bin = cv2.threshold(original, 128, 255, cv2.THRESH_BINARY)
    _, extr_bin = cv2.threshold(extracted, 128, 255, cv2.THRESH_BINARY)
    
    # 计算相似度
    match = np.sum(orig_bin == extr_bin)
    total = orig_bin.size
    similarity = match / total
    
    return similarity > threshold, similarity
```

## 4. 鲁棒性测试

### 4.1 攻击类型
系统实现了15种图像攻击方法：

| 攻击类型 | 函数名 | 参数 |
|---------|--------|------|
| 高斯噪声 | `apply_gaussian_noise` | sigma=0.05 |
| 椒盐噪声 | `apply_salt_pepper_noise` | amount=0.05 |
| 旋转 | `apply_rotation` | angle=15° |
| 裁剪 | `apply_cropping` | ratio=0.1 |
| 亮度调整 | `apply_brightness_adjustment` | factor=0.7 |
| 对比度调整 | `apply_contrast_adjustment` | factor=1.5 |
| 模糊 | `apply_blurring` | kernel_size=5 |
| JPEG压缩 | `apply_jpeg_compression` | quality=30 |
| 缩放 | `apply_scaling` | scale_factor=0.7 |
| 翻转 | `apply_flipping` | flip_code=1 |
| 平移 | `apply_translation` | tx=30, ty=20 |
| 对比度拉伸 | `apply_contrast_stretching` | - |
| 中值滤波 | `apply_median_filter` | kernel_size=3 |
| 锐化 | `apply_sharpening` | - |
| 直方图均衡化 | `apply_histogram_equalization` | - |

### 4.2 评估指标
1. **PSNR（峰值信噪比）**：
   ```math
   \text{PSNR} = 20 \cdot \log_{10}\left(\frac{\text{MAX}_I}{\sqrt{\text{MSE}}}\right)
   ```
   其中MSE为均方误差：
   ```math
   \text{MSE} = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}[I(i,j) - K(i,j)]^2
   ```

2. **相似度**：
   ```math
   \text{Similarity} = \frac{\text{Number of Matching Pixels}}{\text{Total Pixels}}
   ```

3. **泄露判定**：
   `Similarity > Threshold`（默认0.7）

## 5. 实验结果与可视化

系统生成包含16个面板的结果可视化图像：
1. 原始宿主图像
2. 生成的水印
3. 含水印的图像
4. 提取的水印
5-16. 各种攻击后的提取结果（包含PSNR、相似度和泄露状态）

## 6. 应用场景
1. **数字版权保护**：嵌入用户特定水印追踪图像来源
2. **泄露检测**：通过提取水印识别未授权分发
3. **内容认证**：验证图像完整性和来源真实性
4. **盗版追踪**：识别盗版内容的传播路径

## 7. 优势与局限
### 优势：
- 对常见图像处理操作具有鲁棒性
- 水印不可见性好
- 支持盲提取和非盲提取
- 水印与用户ID绑定，可追溯源头

### 局限：
- 对几何攻击（如大角度旋转）较敏感
- 水印容量有限
- 极端压缩（如quality<30）可能导致提取失败

## 8. 使用说明
1. 准备宿主图像（或使用系统生成的示例）
2. 初始化水印器：`watermarker = DCTWatermarker()`
3. 生成水印：`wm = watermarker.generate_watermark(user_id)`
4. 嵌入水印：`wm_image = watermarker.embed_watermark(host_img, wm)`
5. 提取水印：`extracted = watermarker.extract_watermark(wm_image)`
6. 测试鲁棒性：`results = watermarker.test_robustness(wm_image, wm, attacks)`
7. 可视化结果：`create_results_visualization(...)`

## 9. 结论
本系统实现了一种基于DCT变换的鲁棒数字水印方案，通过在中频系数嵌入用户特定的文本水印，有效平衡了不可见性和鲁棒性。实验表明，系统能抵抗多种常见图像处理操作，为数字内容版权保护提供了实用解决方案。
