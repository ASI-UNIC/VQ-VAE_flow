import numpy as np
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim


def interpolate_images(coarse_path, fine_path, output_path, interpolation_factor):
    img_coarse = np.array(Image.open(coarse_path).convert("RGB"))
    img_fine = np.array(Image.open(fine_path).convert("RGB"))
    img_coarse = img_coarse.astype(np.float32)
    img_fine = img_fine.astype(np.float32)

    interpolated_image_np = (
        1 - interpolation_factor
    ) * img_coarse + interpolation_factor * img_fine
    return interpolated_image_np.astype(np.uint8)


def calculate_metrics(interpolated_image_np, medium_path):
    img_medium = np.array(Image.open(medium_path).convert("RGB"))

    mse_value = np.mean((img_medium - interpolated_image_np) ** 2)
    psnr_value = 20 * np.log10(255 / np.sqrt(mse_value))
    ssim_value = ssim(
        img_medium,
        interpolated_image_np,
        channel_axis=2,
        data_range=img_medium.max() - interpolated_image_np.min(),
    )  # Ensure data_range is correctly set
    return psnr_value, ssim_value, mse_value


if __name__ == "__main__":
    timesteps = ["1104", "1128", "1152", "1176", "1200", "1224"]
    figure4_dir = "flow_interpolate_plot/Ioannis/figure4"
    interpolation_factor = 0.5  # Linear interpolation in the middle

    for timestep in timesteps:
        coarse_image_path = os.path.join(figure4_dir, f"{timestep}_coarse.png")
        fine_image_path = os.path.join(figure4_dir, f"{timestep}_fine.png")
        medium_image_path = os.path.join(figure4_dir, f"{timestep}_medium.png")
        interpolated_output_path = os.path.join(
            figure4_dir, f"{timestep}_interpolated.png"
        )

        interpolated_image_np = interpolate_images(
            coarse_image_path,
            fine_image_path,
            interpolated_output_path,
            interpolation_factor,
        )
        psnr, ssim_value, mse = calculate_metrics(
            interpolated_image_np, medium_image_path
        )

        print(f"Timestep: {timestep}")
        print(f"PSNR: {psnr:.4f}")
        print(f"SSIM: {ssim_value:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"Interpolated image saved to: {interpolated_output_path}")
        print("-" * 30)
