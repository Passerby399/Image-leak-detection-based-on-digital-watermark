import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from scipy.fftpack import dct, idct
import os
import matplotlib
import hashlib
from PIL import Image, ImageDraw, ImageFont

# 使用非交互式后端
matplotlib.use('Agg')


class DCTWatermarker:
    def __init__(self, strength=0.1, block_size=8, watermark_size=(64, 64)):
        """
        初始化水印系统
        :param strength: 水印嵌入强度 (0.0 ~ 1.0)
        :param block_size: DCT分块大小
        :param watermark_size: 水印尺寸
        """
        self.strength = strength
        self.block_size = block_size
        self.watermark_size = watermark_size
        self.positions = [(self.block_size // 4, self.block_size // 4),
                          (self.block_size // 2, self.block_size // 4),
                          (self.block_size // 4, self.block_size // 2)]

    def _process_image(self, image):
        """预处理图像：转换为YUV并提取亮度通道"""
        if len(image.shape) == 3:
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:, :, 0].astype(np.float32)
            return y_channel, yuv
        return image.astype(np.float32), None

    def _reconstruct_image(self, y_channel, yuv):
        """从Y通道重建图像"""
        if yuv is not None:
            yuv[:, :, 0] = np.clip(y_channel, 0, 255)
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return y_channel.astype(np.uint8)

    def generate_watermark(self, user_id, text=None):
        """
        生成包含用户ID的可视化水印
        :param user_id: 用户唯一标识符
        :param text: 可选的自定义文本
        :return: 水印图像
        """
        # 创建空白图像
        w, h = self.watermark_size
        watermark = np.zeros((h, w), dtype=np.uint8)
        watermark[:] = 255  # 白色背景

        # 将图像转换为PIL格式以便添加文本
        pil_img = Image.fromarray(watermark)
        draw = ImageDraw.Draw(pil_img)

        # 使用用户ID生成哈希作为水印内容
        if text is None:
            hash_obj = hashlib.sha256(user_id.encode())
            text = hash_obj.hexdigest()[:16]  # 使用前16个字符

        # 添加文本到水印
        font = ImageFont.load_default()
        text_width, text_height = draw.textsize(text, font=font)
        position = ((w - text_width) // 2, (h - text_height) // 2)

        draw.text(position, text, font=font, fill=0)  # 黑色文本

        # 转换回numpy数组
        watermark = np.array(pil_img)
        return watermark

    def embed_watermark(self, host_image, watermark_image):
        """
        在宿主图像中嵌入水印
        :param host_image: 宿主图像 (numpy数组)
        :param watermark_image: 水印图像 (numpy数组)
        :return: 含水印的图像
        """
        # 预处理图像
        host_y, host_yuv = self._process_image(host_image)

        # 调整水印大小以匹配块结构
        watermark = cv2.resize(watermark_image,
                               (host_y.shape[1] // self.block_size,
                                host_y.shape[0] // self.block_size))
        watermark = watermark.astype(np.float32) / 255.0

        # 为每个块嵌入水印
        watermarked_y = host_y.copy()
        for i in range(0, host_y.shape[0], self.block_size):
            for j in range(0, host_y.shape[1], self.block_size):
                # 获取当前块
                block = watermarked_y[i:i + self.block_size, j:j + self.block_size]

                # 应用DCT
                dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

                # 嵌入水印到多个中频系数
                block_i, block_j = i // self.block_size, j // self.block_size
                if block_i < watermark.shape[0] and block_j < watermark.shape[1]:
                    wm_value = watermark[block_i, block_j]

                    # 在多个位置嵌入水印
                    for pos in self.positions:
                        dct_block[pos] += self.strength * wm_value * 100

                # 应用逆DCT
                idct_block = idct(idct(dct_block, axis=0, norm='ortho'), axis=1, norm='ortho')
                watermarked_y[i:i + self.block_size, j:j + self.block_size] = idct_block

        # 重建图像
        return self._reconstruct_image(watermarked_y, host_yuv)

    def extract_watermark(self, watermarked_image, original_image=None, watermark_shape=None):
        """
        从图像中提取水印
        :param watermarked_image: 含水印的图像
        :param original_image: 原始宿主图像 (可选，用于非盲提取)
        :param watermark_shape: 水印的原始形状
        :return: 提取的水印图像
        """
        # 预处理图像
        wm_y, _ = self._process_image(watermarked_image)

        if original_image is not None:
            orig_y, _ = self._process_image(original_image)
        else:
            orig_y = None

        # 确定水印尺寸
        if watermark_shape is None:
            watermark_shape = (wm_y.shape[0] // self.block_size,
                               wm_y.shape[1] // self.block_size)

        watermark = np.zeros(watermark_shape, dtype=np.float32)
        count = np.zeros(watermark_shape, dtype=np.float32)  # 用于平均多个位置

        # 从每个块提取水印
        for i in range(0, wm_y.shape[0], self.block_size):
            for j in range(0, wm_y.shape[1], self.block_size):
                # 获取当前块
                wm_block = wm_y[i:i + self.block_size, j:j + self.block_size]

                # 应用DCT
                dct_wm = dct(dct(wm_block, axis=0, norm='ortho'), axis=1, norm='ortho')

                # 提取水印
                block_i, block_j = i // self.block_size, j // self.block_size
                if block_i < watermark.shape[0] and block_j < watermark.shape[1]:
                    if original_image is not None:
                        # 非盲提取：使用原始图像作为参考
                        orig_block = orig_y[i:i + self.block_size, j:j + self.block_size]
                        dct_orig = dct(dct(orig_block, axis=0, norm='ortho'), axis=1, norm='ortho')

                        # 从多个位置提取并平均
                        for pos in self.positions:
                            diff = dct_wm[pos] - dct_orig[pos]
                            watermark[block_i, block_j] += diff
                            count[block_i, block_j] += 1
                    else:
                        # 盲提取：直接使用系数（效果较差）
                        for pos in self.positions:
                            watermark[block_i, block_j] += dct_wm[pos]
                            count[block_i, block_j] += 1

        # 计算平均值并归一化
        count[count == 0] = 1  # 避免除以零
        watermark /= count
        watermark /= (self.strength * 100)

        # 后处理提取的水印
        watermark = np.clip(watermark, 0, 1)
        watermark = (watermark * 255).astype(np.uint8)

        # 调整到原始水印大小
        if self.watermark_size:
            watermark = cv2.resize(watermark, self.watermark_size)

        return watermark

    def detect_leak(self, original_watermark, extracted_watermark, threshold=0.7):
        """
        检测图像泄露
        :param original_watermark: 原始水印
        :param extracted_watermark: 提取的水印
        :param threshold: 相似度阈值
        :return: 是否泄露, 相似度分数
        """
        # 确保尺寸一致
        if original_watermark.shape != extracted_watermark.shape:
            extracted_watermark = cv2.resize(extracted_watermark,
                                             (original_watermark.shape[1],
                                              original_watermark.shape[0]))

        # 二值化水印
        _, orig_bin = cv2.threshold(original_watermark, 128, 255, cv2.THRESH_BINARY)
        _, extr_bin = cv2.threshold(extracted_watermark, 128, 255, cv2.THRESH_BINARY)

        # 计算相似度
        match = np.sum(orig_bin == extr_bin)
        total = orig_bin.size
        similarity = match / total

        return similarity > threshold, similarity

    def test_robustness(self, watermarked_image, watermark_image, attacks):
        """
        测试水印的鲁棒性
        :param watermarked_image: 含水印的图像
        :param watermark_image: 原始水印图像
        :param attacks: 攻击列表，每个元素是(攻击名称, 攻击函数)
        :return: 包含攻击后提取水印的字典
        """
        results = {}

        for attack_name, attack_func in attacks:
            try:
                # 应用攻击
                attacked_img = attack_func(watermarked_image.copy())

                # 提取水印（使用非盲提取以获得更好结果）
                extracted_watermark = self.extract_watermark(
                    attacked_img,
                    watermark_shape=watermark_image.shape[:2]
                )

                # 检测泄露
                leaked, similarity = self.detect_leak(watermark_image, extracted_watermark)

                # 计算PSNR
                if watermark_image.shape != extracted_watermark.shape:
                    resized_wm = cv2.resize(watermark_image,
                                            (extracted_watermark.shape[1],
                                             extracted_watermark.shape[0]))
                    psnr = self.calculate_psnr(resized_wm, extracted_watermark)
                else:
                    psnr = self.calculate_psnr(watermark_image, extracted_watermark)

                results[attack_name] = {
                    'attacked_image': attacked_img,
                    'extracted_watermark': extracted_watermark,
                    'psnr': psnr,
                    'leaked': leaked,
                    'similarity': similarity
                }
            except Exception as e:
                print(f"攻击 '{attack_name}' 失败: {str(e)}")
                results[attack_name] = {
                    'attacked_image': watermarked_image.copy(),
                    'extracted_watermark': np.zeros_like(watermark_image),
                    'psnr': 0,
                    'leaked': False,
                    'similarity': 0.0
                }

        return results

    def calculate_psnr(self, img1, img2):
        """计算两幅图像的PSNR (峰值信噪比)"""
        if img1.size == 0 or img2.size == 0:
            return 0

        # 确保图像尺寸相同
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))


# 攻击函数定义
def apply_gaussian_noise(image, mean=0, sigma=0.05):
    """添加高斯噪声"""
    noisy = random_noise(image, mode='gaussian', mean=mean, var=sigma ** 2)
    return (noisy * 255).astype(np.uint8)


def apply_salt_pepper_noise(image, amount=0.05):
    """添加椒盐噪声"""
    noisy = random_noise(image, mode='s&p', amount=amount)
    return (noisy * 255).astype(np.uint8)


def apply_rotation(image, angle=15):
    """旋转图像"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def apply_cropping(image, ratio=0.1):
    """裁剪图像边缘"""
    h, w = image.shape[:2]
    crop_h = int(h * ratio)
    crop_w = int(w * ratio)
    cropped = image[crop_h:h - crop_h, crop_w:w - crop_w]
    # 调整回原始尺寸
    return cv2.resize(cropped, (w, h))


def apply_brightness_adjustment(image, factor=0.7):
    """调整亮度"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv = np.clip(hsv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_contrast_adjustment(image, factor=1.5):
    """调整对比度"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=factor, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_blurring(image, kernel_size=5):
    """应用高斯模糊"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_jpeg_compression(image, quality=50):
    """JPEG压缩"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, 1)


def apply_scaling(image, scale_factor=0.7):
    """缩放图像"""
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))


def apply_flipping(image, flip_code=1):
    """翻转图像 (0=垂直, 1=水平, -1=双向)"""
    return cv2.flip(image, flip_code)


def apply_translation(image, tx=20, ty=10):
    """平移图像"""
    h, w = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (w, h))


def apply_contrast_stretching(image):
    """对比度拉伸"""
    # 分别处理每个通道
    if len(image.shape) == 3:
        channels = cv2.split(image)
        stretched_channels = []
        for ch in channels:
            min_val = np.min(ch)
            max_val = np.max(ch)
            if max_val == min_val:
                stretched = ch
            else:
                stretched = (ch.astype(np.float32) - min_val) * (255.0 / (max_val - min_val))
            stretched_channels.append(np.clip(stretched, 0, 255).astype(np.uint8))
        return cv2.merge(stretched_channels)
    else:
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val == min_val:
            return image
        stretched = (image.astype(np.float32) - min_val) * (255.0 / (max_val - min_val))
        return np.clip(stretched, 0, 255).astype(np.uint8)


def apply_median_filter(image, kernel_size=3):
    """应用中值滤波"""
    return cv2.medianBlur(image, kernel_size)


def apply_sharpening(image):
    """图像锐化"""
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def apply_histogram_equalization(image):
    """直方图均衡化"""
    if len(image.shape) == 3:
        # 对于彩色图像，转换为YUV并对Y通道进行均衡化
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        return cv2.equalizeHist(image)


# 创建结果可视化图像
def create_results_visualization(host_img, watermark_img, watermarked_img,
                                 extracted_watermark, robustness_results):
    """创建结果可视化图像并保存"""
    fig = plt.figure(figsize=(20, 16))
    plt.suptitle('Digital Watermarking for Leak Detection', fontsize=16)

    # 显示原始图像和水印
    plt.subplot(4, 4, 1)
    plt.imshow(cv2.cvtColor(host_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Host Image')
    plt.axis('off')

    plt.subplot(4, 4, 2)
    plt.imshow(watermark_img, cmap='gray')
    plt.title('Watermark')
    plt.axis('off')

    plt.subplot(4, 4, 3)
    plt.imshow(cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB))
    plt.title('Watermarked Image')
    plt.axis('off')

    plt.subplot(4, 4, 4)
    plt.imshow(extracted_watermark, cmap='gray')
    plt.title('Extracted Watermark')
    plt.axis('off')

    # 显示鲁棒性测试结果
    for i, (attack_name, result) in enumerate(robustness_results.items()):
        plt.subplot(4, 4, i + 5)
        plt.imshow(result['extracted_watermark'], cmap='gray')
        leak_status = "LEAKED" if result['leaked'] else "NOT LEAKED"
        plt.title(
            f'{attack_name}\nPSNR: {result["psnr"]:.2f} dB\nSimilarity: {result["similarity"]:.2%}\n{leak_status}')
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('watermark_leak_detection_results.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("结果已保存为 'watermark_leak_detection_results.png'")


# 主函数：水印嵌入、提取和鲁棒性测试
def main():
    # 图像路径
    host_img_path = 'original_host.jpg'  # 替换为你的宿主图像路径

    # 加载宿主图像
    if not os.path.exists(host_img_path):
        # 如果没有宿主图像，创建一个示例图像
        host_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.putText(host_img, 'Sample Host Image', (50, 256),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(host_img_path, host_img)
        print(f"创建示例宿主图像: {host_img_path}")
    else:
        host_img = cv2.imread(host_img_path)
        if host_img is None:
            raise FileNotFoundError(f"宿主图像未找到或无法读取: {host_img_path}")

    # 创建水印器
    watermarker = DCTWatermarker(strength=0.15,
                                 block_size=8,
                                 watermark_size=(128, 128))

    # 生成水印（使用用户ID）
    user_id = "user12345@company.com"
    watermark_img = watermarker.generate_watermark(user_id)

    # 嵌入水印
    watermarked_img = watermarker.embed_watermark(host_img, watermark_img)

    # 提取水印（非盲提取）
    extracted_watermark = watermarker.extract_watermark(
        watermarked_img,
        original_image=host_img,
        watermark_shape=watermark_img.shape
    )

    # 定义攻击列表
    attacks = [
        ('Gaussian Noise', lambda img: apply_gaussian_noise(img, sigma=0.05)),
        ('Salt & Pepper', lambda img: apply_salt_pepper_noise(img, amount=0.05)),
        ('Rotation (15°)', lambda img: apply_rotation(img, angle=15)),
        ('Cropping (10%)', lambda img: apply_cropping(img, ratio=0.1)),
        ('Brightness Adj', lambda img: apply_brightness_adjustment(img, factor=0.7)),
        ('Contrast Adj', lambda img: apply_contrast_adjustment(img, factor=1.5)),
        ('Blurring', lambda img: apply_blurring(img, kernel_size=5)),
        ('JPEG Compress', lambda img: apply_jpeg_compression(img, quality=30)),
        ('Scaling (70%)', lambda img: apply_scaling(img, scale_factor=0.7)),
        ('Horizontal Flip', lambda img: apply_flipping(img, flip_code=1)),
        ('Translation', lambda img: apply_translation(img, tx=30, ty=20)),
        ('Contrast Stretch', apply_contrast_stretching),
        ('Median Filter', lambda img: apply_median_filter(img, kernel_size=3)),
        ('Sharpening', apply_sharpening),
        ('Histogram Eq', apply_histogram_equalization)
    ]

    # 测试鲁棒性和泄露检测
    robustness_results = watermarker.test_robustness(
        watermarked_img,
        watermark_img,
        attacks
    )

    # 创建结果可视化图像
    create_results_visualization(
        host_img, watermark_img, watermarked_img,
        extracted_watermark, robustness_results
    )

    # 保存所有图像到文件
    cv2.imwrite('original_host.jpg', host_img)
    cv2.imwrite('watermark_image.png', watermark_img)
    cv2.imwrite('watermarked_image.jpg', watermarked_img)
    cv2.imwrite('extracted_watermark.png', extracted_watermark)

    # 保存攻击后的图像
    for attack_name, result in robustness_results.items():
        # 清理文件名
        safe_name = attack_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', '')
        cv2.imwrite(f'attacked_{safe_name}.jpg', result['attacked_image'])
        cv2.imwrite(f'extracted_{safe_name}.png', result['extracted_watermark'])

    print("所有结果图像已保存到当前目录")
    print("泄露检测结果:")
    for attack_name, result in robustness_results.items():
        status = "检测到泄露" if result['leaked'] else "未检测到泄露"
        print(f"- {attack_name}: {status} (相似度: {result['similarity']:.2%}, PSNR: {result['psnr']:.2f} dB)")


if __name__ == "__main__":
    main()