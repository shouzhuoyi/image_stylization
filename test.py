import numpy as np
from PIL import Image, ImageEnhance
import os

def process_image(input_path, output_path):
    try:
        # 读取原图和彩色副本
        im_color = Image.open(input_path).convert('RGB')
        im_gray = im_color.convert('L')
        array = np.asarray(im_gray).astype(np.float64)

        # ===== 线稿提取 =====
        depth = 20
        grad_x, grad_y = np.gradient(array)
        grad_x = grad_x * depth / 100.
        grad_y = grad_y * depth / 100.
        dis = np.sqrt(grad_x**2 + grad_y**2 + 1.0)
        uni_x = grad_x/dis
        uni_y = grad_y/dis
        uni_z = 1.0/dis
        vec_el = np.pi / 2.2
        vec_az = np.pi / 4
        dx = np.cos(vec_el)*np.cos(vec_az)
        dy = np.cos(vec_el)*np.sin(vec_az)
        dz = np.sin(vec_el)
        out = 255*(uni_x*dx + uni_y*dy + uni_z*dz)
        out = out.clip(0, 255)
        img_shade = Image.fromarray(out.astype(np.uint8))

        # ===== 将线稿明度与原彩色图融合 =====
        hsv = im_color.convert('HSV')
        h, s, v = hsv.split()
        v_new = img_shade
        hsv_colored = Image.merge('HSV', (h, s, v_new))
        img_colored = hsv_colored.convert('RGB')

        # ===== 明度与饱和度调节 =====
        brightness_factor = 0.8
        saturation_factor = 0.65
        img_colored = ImageEnhance.Brightness(img_colored).enhance(brightness_factor)
        img_colored = ImageEnhance.Color(img_colored).enhance(saturation_factor)

        # ===== 添加单色高斯噪声（Monochromatic）=====
        img_np = np.asarray(img_colored).astype(np.float32)
        mean, sigma = 0, 12

        # ✅ 关键：生成灰度噪声
        gray_noise = np.random.normal(mean, sigma, img_np.shape[:2])  # shape=(H, W)
        gray_noise = np.repeat(gray_noise[:, :, np.newaxis], 3, axis=2)  # 复制到3个通道

        # 加噪
        noisy_img = img_np + gray_noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        img_noisy = Image.fromarray(noisy_img)

        img_noisy.save(output_path)
        print(f"Processed: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

if __name__ == '__main__':
    input_dir = "./明信片图片集"
    output_dir = "./output_images"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
    else:
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                process_image(input_path, output_path)
        print("Batch processing complete.")