from unet_lightning import RVQUNetLightning
import torch
import numpy as np
from dataset import InterpolateIntermediateDataset
from glob import glob
import os
import torch.nn.functional as F
from plot import plot_visualization
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def main():
    max_epochs = 100
    test_size = 0.1

    config = {
        "resvq_embeddings": 256,  # 256,
        "num_embeddings": [256, 256],  # 256,
        "codebook_size": 64,  # 16,
    }

    num_embeddings = config["num_embeddings"]
    resvq_embeddings = config["resvq_embeddings"]
    codebook_size = config["codebook_size"]

    val_dataset = InterpolateIntermediateDataset(
        h5py_file="/Users/darylfung/Documents/Work/Nicosia/probe-dimitris/Flow project/data/flow/flow.h5",
        is_train=False,
        test_size=test_size,
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    checkpoint_files = glob("64_checkpoints_interpolate_intermediate_flow/*.ckpt")

    for checkpoint_file in checkpoint_files:
        interpolate_alpha = float(
            os.path.splitext(os.path.basename(checkpoint_file))[0].split("_")[-1]
        )

        try:
            model = RVQUNetLightning.load_from_checkpoint(
                checkpoint_file,
                n_channels=3,
                interpolate_alpha=interpolate_alpha,
                num_embeddings=num_embeddings,
                total_epochs=max_epochs,
                resvq_embeddings=resvq_embeddings,
                codebook_size=codebook_size,
                intermediate_training=True,
            )
        except RuntimeError:
            model = RVQUNetLightning.load_from_checkpoint(
                checkpoint_file,
                n_channels=3,
                interpolate_alpha=interpolate_alpha,
                num_embeddings=num_embeddings,
                total_epochs=max_epochs,
                resvq_embeddings=resvq_embeddings,
                codebook_size=64,
                intermediate_training=True,
            )
        # Visualize some reconstructions from the best model
        with torch.no_grad():
            model.eval()
            data, intermediate, target = next(iter(val_loader))
            data = data.to(model.device)
            intermediate = intermediate.to(model.device)
            target = target.to(model.device)

            # get mid point
            data_latent = model.get_latent(data)
            target_latent = model.get_latent(target)
            intermediate_latent = data_latent + (data_latent + target_latent) / 2

            data_image = model.generate_image(data_latent)
            intermediate_images = model.generate_image(intermediate_latent)
            target_image = model.generate_image(target_latent)

            data_images = (
                data_image.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255.0
            ).astype(np.uint8)
            data = (data.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255.0).astype(
                np.uint8
            )
            target = (
                target.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255.0
            ).astype(np.uint8)
            target_images = (
                target_image.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255.0
            ).astype(np.uint8)
            intermediate_images = (
                intermediate_images.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1)
                * 255.0
            ).astype(np.uint8)
            intermediate_ground_truth_images = (
                intermediate.permute(0, 2, 3, 1).clip(0, 1).detach().cpu().numpy()
                * 255.0
            ).astype(np.uint8)

            # Interpolate ground truth images
            data_np = data.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1)
            target_np = target.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1)
            intermediate_ground_truth_np = (
                intermediate.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1)
            )

            interpolated_ground_truth = (data_np + target_np) / 2.0
            interpolated_ground_truth_images = (interpolated_ground_truth * 255).astype(
                np.uint8
            )
            intermediate_ground_truth_images = (
                intermediate_ground_truth_np * 255
            ).astype(np.uint8)

            # Calculate metrics for interpolated ground truth
            interpolated_mse_loss = F.mse_loss(
                torch.Tensor(interpolated_ground_truth_images),
                torch.Tensor(intermediate_ground_truth_images),
            )
            interpolated_psnr = psnr(
                interpolated_ground_truth_images, intermediate_ground_truth_images
            )
            interpolated_ssim = ssim(
                interpolated_ground_truth_images,
                intermediate_ground_truth_images,
                channel_axis=3,
            )

            print("Interpolated Ground Truth Metrics:")
            print(f"  MSE: {interpolated_mse_loss:.2f}")
            print(f"  PSNR: {interpolated_psnr:.2f}")
            print(f"  SSIM: {interpolated_ssim:.2f}")
            print("--------------------------")

            # calculate loss
            print(f"Mean square error for {interpolate_alpha}:")
            data_mse_loss = F.mse_loss(torch.Tensor(data_images), torch.Tensor(data))
            data_psnr = psnr(data_images, data)
            data_ssim = ssim(data_images, data, channel_axis=3)
            target_mse_loss = F.mse_loss(
                torch.Tensor(target_images), torch.Tensor(target)
            )
            target_psnr = psnr(target_images, target)
            target_ssim = ssim(target_images, target, channel_axis=3)
            mse_loss = F.mse_loss(
                torch.Tensor(intermediate_images),
                torch.Tensor(intermediate_ground_truth_images),
            )
            intermediate_psnr = psnr(
                intermediate_images, intermediate_ground_truth_images
            )
            intermediate_ssim = ssim(
                intermediate_images, intermediate_ground_truth_images, channel_axis=3
            )
            mean_mse_loss = (data_mse_loss + target_mse_loss + mse_loss) / 3
            mean_psnr = (data_psnr + target_psnr + intermediate_psnr) / 3
            mean_ssim = (data_ssim + target_ssim + intermediate_ssim) / 3
            print(f"mean mse loss: {mean_mse_loss:.2f}")
            print(f"mean psnr: {mean_psnr:.2f}")
            print(f"mean ssim: {mean_ssim:.2f}")
            print("==========================")
            print(f"intermediate mse loss: {mse_loss:.2f}")
            print(f"intermediate psnr: {intermediate_psnr:.2f}")
            print(f"intermediate ssim: {intermediate_ssim:.2f}")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            # target_images = target_images.cpu().numpy()
            # target_images = target_images.transpose(0, 2, 3, 1).clip(0, 1) * 255.0
            # target_images = target_images.astype(np.uint8)

            plot_visualization(
                torch.Tensor(intermediate_ground_truth_images[:10]),
                torch.Tensor(intermediate_images[:10]),
                save_filename=f"interpolation_intermediate_reconstructions_{interpolate_alpha}.png",
                num_items=min(intermediate_ground_truth_images.shape[0], 10),
            )


if __name__ == "__main__":
    main()
