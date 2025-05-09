from unet_lightning import RVQUNetLightning
import torch
import numpy as np
from dataset import InterpolateIntermediateDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
import os
from plot import plot_visualization


def main():
    max_epochs = 100
    test_size = 0.1

    config = {
        "resvq_embeddings": 256,  # 256,
        "num_embeddings": [256, 256],  # 256,
        "codebook_size": 64,  # 16,
        "interpolate_alpha": 1,
    }

    num_embeddings = config["num_embeddings"]
    resvq_embeddings = config["resvq_embeddings"]
    codebook_size = config["codebook_size"]
    interpolate_alpha = config["interpolate_alpha"]

    train_dataset = InterpolateIntermediateDataset(
        h5py_file="/Users/darylfung/Documents/Work/Nicosia/probe-dimitris/Flow project/data/flow/flow.h5",
        is_train=True,
        test_size=test_size,
    )
    val_dataset = InterpolateIntermediateDataset(
        h5py_file="/Users/darylfung/Documents/Work/Nicosia/probe-dimitris/Flow project/data/flow/flow.h5",
        is_train=False,
        test_size=test_size,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = RVQUNetLightning.load_from_checkpoint(
        "64_checkpoints_interpolate_intermediate_flow/inter-rvq-unet_0.ckpt",
        n_channels=3,
        interpolate_alpha=interpolate_alpha,
        num_embeddings=num_embeddings,
        total_epochs=max_epochs,
        resvq_embeddings=resvq_embeddings,
        codebook_size=codebook_size,
        intermediate_training=True,
    )

    model_dir = "model_interpolate_intermediate_flow"
    model_checkpoint_callback = ModelCheckpoint(
        model_dir, filename="inter-rvq-unet", save_top_k=1, mode="min"
    )
    early_stopping_callback = EarlyStopping("val_recon_loss", patience=5, mode="min")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=True,
        log_every_n_steps=100,
        callbacks=[model_checkpoint_callback, early_stopping_callback],
        logger=MLFlowLogger(
            experiment_name="flow_interpolation",
            tracking_uri="file:./mlruns",
        ),
    )
    trainer.fit(model, train_loader, val_loader)
    print(model.lowest_val_loss)

    torch.save(
        config,
        os.path.join(model_dir, "config.json"),
    )

    decoder_model_filename = os.path.join(model_dir, "inter-rvq-unet.ckpt")
    decoder_model_config = os.path.join(model_dir, "config.json")
    config = torch.load(decoder_model_config)
    num_embeddings = config["num_embeddings"]
    resvq_embeddings = config["resvq_embeddings"]
    codebook_size = config["codebook_size"]
    best_model = RVQUNetLightning.load_from_checkpoint(
        decoder_model_filename,
        n_channels=3,
        num_embeddings=num_embeddings,
        total_epochs=max_epochs,
        resvq_embeddings=resvq_embeddings,
        codebook_size=codebook_size,
    )
    # Visualize some reconstructions from the best model
    with torch.no_grad():
        best_model.eval()
        data, intermediate, target = next(iter(val_loader))
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
            save_filename="interpolation_intermediate_reconstructions.png",
            num_items=min(output_images.shape[0], 10),
        )


if __name__ == "__main__":
    main()
