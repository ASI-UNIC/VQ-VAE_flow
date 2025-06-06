from unet_lightning import RVQUNetLightning
import torch
from dataset import InterpolateIntermediateDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from ray import tune

max_epochs = 100


def train(config):
    num_embeddings = config["num_embeddings"]
    resvq_embeddings = config["resvq_embeddings"]
    codebook_size = config["codebook_size"]

    train_dataset = InterpolateIntermediateDataset(
        h5py_file="/Users/darylfung/Documents/Work/Nicosia/probe-dimitris/Flow project/data/flow/flow.h5",
        is_train=True,
    )
    val_dataset = InterpolateIntermediateDataset(
        h5py_file="/Users/darylfung/Documents/Work/Nicosia/probe-dimitris/Flow project/data/flow/flow.h5",
        is_train=False,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = RVQUNetLightning(
        n_channels=3,
        num_embeddings=num_embeddings,
        total_epochs=max_epochs,
        resvq_embeddings=resvq_embeddings,
        codebook_size=codebook_size,
        intermediate_training=True,
    )

    early_stopping_callback = EarlyStopping("val_recon_loss", patience=25, mode="min")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=True,
        log_every_n_steps=100,
        callbacks=[early_stopping_callback],
        logger=MLFlowLogger(
            experiment_name="flow_interpolation",
            tracking_uri="file:./mlruns",
        ),
    )
    trainer.fit(model, train_loader, val_loader)

    return {"val_loss": model.lowest_val_loss}


def main():
    config = {
        "resvq_embeddings": 128,  # 128,
        "num_embeddings": [128, 128],  # 256,
        "codebook_size": 128,  # 16,
    }
    model_dir = "checkpoints_interpolate_intermediate_flow"

    search_space = {  # ②
        "resvq_embeddings": tune.grid_search([32, 64, 128, 256]),
        "num_embeddings": tune.grid_search(
            [[32, 32], [64, 64], [128, 128], [256, 256]]
        ),
        "codebook_size": tune.grid_search([8, 16, 32, 64]),
    }

    tuner = tune.Tuner(train, param_space=search_space)  # ③
    results = tuner.fit()
    print(results.get_best_result(metric="val_loss", mode="min").config)

    # torch.save(
    #     config,
    #     os.path.join(model_dir, "config.json"),
    # )

    # decoder_model_filename = os.path.join(model_dir, "rvq-unet.ckpt")
    # decoder_model_config = os.path.join(model_dir, "config.json")
    # config = torch.load(decoder_model_config)
    # n_channels = config["n_channels"]
    # num_embeddings = config["num_embeddings"]
    # max_epochs = config["total_epochs"]
    # resvq_embeddings = config["resvq_embeddings"]
    # codebook_size = config["codebook_size"]
    # best_model = RVQUNetLightning.load_from_checkpoint(
    #     decoder_model_filename,
    #     n_channels=n_channels,
    #     num_embeddings=num_embeddings,
    #     total_epochs=max_epochs,
    #     resvq_embeddings=resvq_embeddings,
    #     codebook_size=codebook_size,
    # )
    # # Visualize some reconstructions from the best model
    # with torch.no_grad():
    #     best_model.eval()
    #     data, intermediate, target = next(iter(val_loader))
    #     data = data.to(best_model.device)
    #     intermediate = intermediate.to(best_model.device)
    #     target = target.to(best_model.device)

    #     output_images, _ = best_model(data)
    #     intermediate_images, _ = best_model(intermediate)
    #     target_images, _ = best_model(target)

    #     output_images = output_images.cpu().numpy()
    #     output_images = output_images.transpose(0, 2, 3, 1).clip(0, 1) * 255.0
    #     output_images = output_images.astype(np.uint8)

    #     intermediate_images = intermediate_images.cpu().numpy()
    #     intermediate_images = (
    #         intermediate_images.transpose(0, 2, 3, 1).clip(0, 1) * 255.0
    #     )
    #     intermediate_images = intermediate_images.astype(np.uint8)

    #     target_images = target_images.cpu().numpy()
    #     target_images = target_images.transpose(0, 2, 3, 1).clip(0, 1) * 255.0
    #     target_images = target_images.astype(np.uint8)

    #     plot_visualization(
    #         torch.Tensor(output_images[:10]),
    #         torch.Tensor(intermediate_images[:10]),
    #         torch.Tensor(target_images[:10]),
    #         save_filename="interpolation_intermediate_reconstructions.png",
    #         num_items=min(output_images.shape[0], 10),
    # )


if __name__ == "__main__":
    main()
