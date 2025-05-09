import imageio
from unet_lightning import RVQUNetLightning
import torch
import numpy as np
from pathlib import Path
from dataset import InterpolateIntermediateDataset
import einops


def main():
    save_dir = Path("flow_interpolate_plot")

    max_epochs = 100
    test_size = 0.1

    config = {
        "resvq_embeddings": 256,  # 256,
        "num_embeddings": [256, 256],  # 256,
        "codebook_size": 64,  # 16,
        "interpolate_alpha": 0,
    }

    num_embeddings = config["num_embeddings"]
    resvq_embeddings = config["resvq_embeddings"]
    codebook_size = config["codebook_size"]
    interpolate_alpha = config["interpolate_alpha"]

    val_dataset = InterpolateIntermediateDataset(
        h5py_file="/Users/darylfung/Documents/Work/Nicosia/probe-dimitris/Flow project/data/flow/flow.h5",
        is_train=False,
        test_size=test_size,
    )
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False)

    model = RVQUNetLightning.load_from_checkpoint(
        "64_checkpoints_interpolate_intermediate_flow/inter-rvq-unet_0.75.ckpt",
        n_channels=3,
        interpolate_alpha=interpolate_alpha,
        num_embeddings=num_embeddings,
        total_epochs=max_epochs,
        resvq_embeddings=resvq_embeddings,
        codebook_size=codebook_size,
        intermediate_training=True,
    )

    # Visualize some reconstructions from the best model
    with torch.no_grad():
        model.eval()

        steps = len(val_dataset) // 5
        batches = [val_dataset[i] for i in range(0, len(val_dataset), steps)]
        data = torch.stack([batch[0] for batch in batches])
        intermediate = torch.stack([batch[1] for batch in batches])
        target = torch.stack([batch[2] for batch in batches])
        data = data.to(model.device)
        intermediate = intermediate.to(model.device)
        target = target.to(model.device)

        data_latent = model.get_latent(data)
        intermediate_latent = model.get_latent(intermediate)
        target_latent = model.get_latent(target)
        orig_shape = data_latent.shape

        data_latent = data_latent.view(orig_shape[0], -1)
        intermediate_latent = intermediate_latent.view(orig_shape[0], -1)
        target_latent = target_latent.view(orig_shape[0], -1)

        for i in range(data_latent.shape[0]):
            interpolated_latent = torch.lerp(
                data_latent[i],
                target_latent[i],
                torch.linspace(0, 1, 11).view(-1, 1).to(model.device),
            )
            extrapolated_latent = target_latent[i] + 0.1 * (
                target_latent[i] - data_latent[i]
            )

            interpolated_latent = interpolated_latent.to(model.device)
            interpolated_latent = interpolated_latent.view(
                interpolated_latent.shape[0], *orig_shape[1:]
            )
            extrapolated_latent = extrapolated_latent.view(1, *orig_shape[1:])

            rec_data_images = model.generate_image(
                data_latent.view(data_latent.shape[0], *orig_shape[1:])
            )
            rec_intermediate_images = model.generate_image(
                intermediate_latent.view(intermediate_latent.shape[0], *orig_shape[1:])
            )
            rec_target_images = model.generate_image(
                target_latent.view(target_latent.shape[0], *orig_shape[1:])
            )

            # generate interpolated images
            interpolated_images = model.generate_image(interpolated_latent)
            interpolated_images = interpolated_images.cpu().numpy()
            interpolated_images = (
                interpolated_images.transpose(0, 2, 3, 1).clip(0, 1) * 255.0
            )
            interpolated_images = interpolated_images.astype(np.uint8)

            # generate extrapolated images
            extrapolated_images = model.generate_image(extrapolated_latent)
            extrapolated_images = extrapolated_images.cpu().numpy()
            extrapolated_images = (
                extrapolated_images.transpose(0, 2, 3, 1).clip(0, 1) * 255.0
            )
            extrapolated_images = extrapolated_images.astype(np.uint8)

            intermediate_image = (
                intermediate[i].cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255.0
            )
            intermediate_image = intermediate_image.astype(np.uint8)

            data_image = data[i].cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255.0
            data_image = data_image.astype(np.uint8)
            rec_data_image = (
                rec_data_images[i].cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255.0
            )
            rec_data_image = rec_data_image.astype(np.uint8)
            rec_intermediate_image = (
                rec_intermediate_images[i].cpu().numpy().transpose(1, 2, 0).clip(0, 1)
                * 255.0
            )
            rec_intermediate_image = rec_intermediate_image.astype(np.uint8)
            target_image = target[i].cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255.0
            target_image = target_image.astype(np.uint8)
            rec_target_image = (
                rec_target_images[i].cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255.0
            )
            rec_target_image = rec_target_image.astype(np.uint8)

            current_save_dir = save_dir / f"interpolation_{i}"
            current_save_dir.mkdir(parents=True, exist_ok=True)

            data_image = einops.repeat(
                data_image, "w h c -> k w h c", k=interpolated_images.shape[0]
            )
            intermediate_image = einops.repeat(
                intermediate_image,
                "w h c -> k w h c",
                k=interpolated_images.shape[0],
            )
            target_image = einops.repeat(
                target_image, "w h c -> k w h c", k=interpolated_images.shape[0]
            )

            main_images = np.concatenate(
                [data_image, intermediate_image, target_image], axis=2
            )
            pad_images = np.zeros(data_image.shape).astype(np.uint8)
            main_interpolated_images = np.concatenate(
                [pad_images, interpolated_images, pad_images], axis=2
            )
            final_images = np.concatenate(
                [main_images, main_interpolated_images], axis=1
            )

            imageio.imwrite(current_save_dir / "0_result.png", interpolated_images[0])
            imageio.imwrite(current_save_dir / "2_result.png", interpolated_images[2])
            imageio.imwrite(current_save_dir / "4_result.png", interpolated_images[4])
            imageio.imwrite(current_save_dir / "6_result.png", interpolated_images[6])
            imageio.imwrite(current_save_dir / "8_result.png", interpolated_images[8])
            imageio.imwrite(current_save_dir / "10_result.png", interpolated_images[10])

            imageio.mimsave(
                current_save_dir / "results.gif", final_images, duration=0.1
            )

            imageio.imwrite(
                current_save_dir / "data.png",
                data_image[0],
            )
            imageio.imwrite(
                current_save_dir / "intermediate.png",
                intermediate_image[0],
            )
            imageio.imwrite(
                current_save_dir / "target.png",
                target_image[0],
            )

            imageio.imwrite(
                current_save_dir / "rec_data.png",
                rec_data_image,
            )
            imageio.imwrite(
                current_save_dir / "rec_intermediate.png",
                rec_intermediate_image,
            )
            imageio.imwrite(
                current_save_dir / "rec_extrapolated.png",
                extrapolated_images[0],
            )
            imageio.imwrite(
                current_save_dir / "rec_midpoint_intermediate.png",
                interpolated_images[4],
            )
            imageio.imwrite(
                current_save_dir / "rec_target.png",
                rec_target_image,
            )


if __name__ == "__main__":
    main()
