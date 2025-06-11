import numpy as np
import rasterio
from PIL import Image
import os


def read_rgb_bands(tif_path, band_indices=(2, 1, 0)):
    """
    读取TIFF图像的RGB波段（默认顺序为B04红，B03绿，B02蓝）

    参数:
        tif_path (str): TIFF文件路径
        band_indices (tuple): RGB波段在数据中的索引（从0开始）

    返回:
        np.ndarray: 形状为(H, W, 3)的RGB图像数组（float类型）
    """
    with rasterio.open(tif_path) as src:
        bands = [src.read(i + 1).astype(np.float32) for i in band_indices]
        rgb_stack = np.stack(bands, axis=-1)
    return rgb_stack


def normalize_image(img):
    """
    将图像归一化到0-255的范围

    参数:
        img (np.ndarray): 输入图像，任意范围

    返回:
        np.ndarray: uint8类型的归一化图像
    """
    min_val = img.min()
    max_val = img.max()
    norm = (img - min_val) / (max_val - min_val + 1e-8)  # 防止除0
    return (norm * 255).astype(np.uint8)


def process_tif_to_rgb(tif_path):
    """
    读取并处理TIFF图像，返回归一化RGB图像

    参数:
        tif_path (str): TIFF图像路径

    返回:
        np.ndarray: 归一化后的RGB图像（uint8）
    """
    rgb_float = read_rgb_bands(tif_path)
    rgb_uint8 = normalize_image(rgb_float)
    return rgb_uint8


def save_image(image_array, output_path):
    """
    保存图像为PNG格式

    参数:
        image_array (np.ndarray): RGB图像数组
        output_path (str): 输出路径
    """
    Image.fromarray(image_array).save(output_path)


if __name__ == "__main__":
    input_tif = r"D:\Desktop\2019_1101_nofire_B2348_B12_10m_roi.tif"  # 你的TIFF路径
    output_png = r"D:\Desktop\output_image.png"  # 保存结果到桌面

    # 检查文件是否存在
    if not os.path.exists(input_tif):
        print("❌ 找不到TIFF文件，请检查路径是否正确")
    else:
        try:
            rgb_result = process_tif_to_rgb(input_tif)
            save_image(rgb_result, output_png)
            print(f"✅ 成功生成真彩色图像并保存到：{output_png}")
        except Exception as e:
            print(f"❌ 处理出错: {e}")
