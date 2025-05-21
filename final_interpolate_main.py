import numpy as np
from unet_lightning import RVQUNetLightning
import torch
import os
from dataset import InterpolateDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from plot import plot_visualization


def main():
    resvq_embeddings = 256
    n_channels = 3
    num_embeddings = [256, 256]
    codebook_size = 16
    max_epochs = 100

    train_dataset = InterpolateDataset(h5py_file="data/flow/flow.h5")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )

    model = RVQUNetLightning(
        3,
        bilinear=False,
        num_embeddings=num_embeddings,
        total_epochs=max_epochs,
        resvq_embeddings=resvq_embeddings,
        codebook_size=codebook_size,
    )

    model_dir = "checkpoints_interpolate_flow"
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=model_dir,
        filename="rvq-unet",
        save_top_k=1,
        mode="min",
    )

    early_stopping_callback = EarlyStopping("train_loss")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=True,
        log_every_n_steps=100,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader)
    torch.save(
        {
            "n_channels": n_channels,
            "num_embeddings": num_embeddings,
            "total_epochs": max_epochs,
            "resvq_embeddings": resvq_embeddings,
            "codebook_size": codebook_size,
        },
        os.path.join(model_dir, "config.json"),
    )

    decoder_model_filename = os.path.join(model_dir, "rvq-unet.ckpt")
    decoder_model_config = os.path.join(model_dir, "config.json")
    config = torch.load(decoder_model_config)
    n_channels = config["n_channels"]
    num_embeddings = config["num_embeddings"]
    max_epochs = config["total_epochs"]
    resvq_embeddings = config["resvq_embeddings"]
    codebook_size = config["codebook_size"]
    best_model = RVQUNetLightning.load_from_checkpoint(
        decoder_model_filename,
        n_channels=n_channels,
        num_embeddings=num_embeddings,
        total_epochs=max_epochs,
        resvq_embeddings=resvq_embeddings,
        codebook_size=codebook_size,
    )
    # Visualize some reconstructions from the best model
    with torch.no_grad():
        best_model.eval()
        data, intermediate, target = next(iter(train_loader))
        data = data.to(best_model.device)
        intermediate = intermediate.to(best_model.device)
        target = target.to(best_model.device)

        output_images, _ = best_model(data)
        intermediate_images, _ = best_model(intermediate)
        target_images, _ = best_model(target)

        output_images = output_images.cpu().numpy()
        output_images = output_images.transpose(0, 2, 3, 1).clip(0, 1) * 255.0
        output_images = output_images.astype(np.uint8)

        intermediate_images = intermediate_images.cpu().numpy()
        intermediate_images = (
            intermediate_images.transpose(0, 2, 3, 1).clip(0, 1) * 255.0
        )
        intermediate_images = intermediate_images.astype(np.uint8)

        target_images = target_images.cpu().numpy()
        target_images = target_images.transpose(0, 2, 3, 1).clip(0, 1) * 255.0
        target_images = target_images.astype(np.uint8)

        plot_visualization(
            torch.Tensor(output_images[:10]),
            torch.Tensor(intermediate_images[:10]),
            torch.Tensor(target_images[:10]),
            save_filename="interpolation_reconstructions.png",
            num_items=min(output_images.shape[0], 10),
        )


if __name__ == "__main__":
    main()
