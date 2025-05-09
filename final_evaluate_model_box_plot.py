import matplotlib.pyplot as plt
from unet_lightning import RVQUNetLightning
import torch
import numpy as np
from dataset import InterpolateIntermediateDataset
from glob import glob
import os
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def main():
    max_epochs = 100
    test_size = 0.1

    config = {
        "resvq_embeddings": 256,  # 256,
        "num_embeddings": [256, 256],  # 256,
        "codebook_size": 16,  # 16,
    }

    num_embeddings = config["num_embeddings"]
    resvq_embeddings = config["resvq_embeddings"]
    codebook_size = config["codebook_size"]

    val_dataset = InterpolateIntermediateDataset(
        h5py_file="data/pretrain_interpolate/flow.h5",
        is_train=False,
        test_size=test_size,
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    checkpoint_files = glob("model_interpolate_intermediate_flow/*0.ckpt")

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
        total_data_loss = {"mse": [], "psnr": [], "ssim": []}
        total_intermediate_loss = {"mse": [], "psnr": [], "ssim": []}
        total_target_loss = {"mse": [], "psnr": [], "ssim": []}
        with torch.no_grad():
            model.eval()

            for batch in val_loader:
                data, intermediate, target = batch
                data = data.to(model.device)
                intermediate = intermediate.to(model.device)
                target = target.to(model.device)

                # get mid point
                data_latent = model.get_latent(data)
                target_latent = model.get_latent(target)
                intermediate_latent = model.get_latent(intermediate)

                data_image = model.generate_image(data_latent)
                intermediate_images = model.generate_image(intermediate_latent)
                target_image = model.generate_image(target_latent)

                data_images = (
                    data_image.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255.0
                ).astype(np.uint8)
                data = (
                    data.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255.0
                ).astype(np.uint8)
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

                # calculate loss
                print(f"Mean square error for {interpolate_alpha}:")
                data_mse_loss = F.mse_loss(
                    torch.Tensor(data_images), torch.Tensor(data)
                )
                data_psnr = psnr(data_images, data)
                data_ssim = ssim(data_images, data, channel_axis=3)
                target_mse_loss = F.mse_loss(
                    torch.Tensor(target_images), torch.Tensor(target)
                )
                target_psnr = psnr(target_images, target)
                target_ssim = ssim(target_images, target, channel_axis=3)
                intermediate_mse_loss = F.mse_loss(
                    torch.Tensor(intermediate_images),
                    torch.Tensor(intermediate_ground_truth_images),
                )
                intermediate_psnr = psnr(
                    intermediate_images, intermediate_ground_truth_images
                )
                intermediate_ssim = ssim(
                    intermediate_images,
                    intermediate_ground_truth_images,
                    channel_axis=3,
                )

                total_data_loss["mse"].append(data_mse_loss)
                total_data_loss["psnr"].append(data_psnr)
                total_data_loss["ssim"].append(data_ssim)

                total_intermediate_loss["mse"].append(intermediate_mse_loss)
                total_intermediate_loss["psnr"].append(intermediate_psnr)
                total_intermediate_loss["ssim"].append(intermediate_ssim)

                total_target_loss["mse"].append(target_mse_loss)
                total_target_loss["psnr"].append(target_psnr)
                total_target_loss["ssim"].append(target_ssim)

        # plot box plot and standard deviation
        metrics = ["mse", "psnr", "ssim"]
        for metric in metrics:
            current_data_to_plot = [
                total_data_loss[metric],
                total_intermediate_loss[metric],
                total_target_loss[metric],
            ]
            fig, ax = plt.subplots(figsize=(10, 6))
            bp = ax.boxplot(
                current_data_to_plot,
                patch_artist=True,
                notch=True,
                vert=True,
                widths=0.6,
                showfliers=True,
            )

            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
            whisker_color = "gray"
            cap_color = "gray"
            median_color = "black"
            flier_color = "red"
            flier_marker = "."

            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            for whisker in bp["whiskers"]:
                whisker.set(color=whisker_color, linewidth=1.5, linestyle="--")

            for cap in bp["caps"]:
                cap.set(color=cap_color, linewidth=2)

            for median in bp["medians"]:
                median.set(color=median_color, linewidth=2)

            for flier in bp["fliers"]:
                flier.set(marker=flier_marker, color=flier_color, alpha=0.5)

            ax.set_xlabel("Resolutions", fontsize=18, fontweight="bold")
            ax.set_ylabel(f"{metric}", fontsize=18, fontweight="bold")

            ax.tick_params(axis="y", labelsize=16)
            ax.set_xticklabels(
                ["Coarse Resolution", "Medium Resolution", "Fine Resolution"],
                fontsize=16,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()

            plt.savefig(f"{metric}_vqvae_boxplot.png")


if __name__ == "__main__":
    main()
