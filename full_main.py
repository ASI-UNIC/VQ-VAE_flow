from unet_lightning import RVQUNetLightning
from torch.utils.data import DataLoader
from dataset import AllTargetImageDataset
import torch
import click
import os
from plot import plot_visualization
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl

@click.command()
@click.option(
    '--image_h5_file',
    type=str,
    required=True
)
def main(image_h5_file: str):
    # CIFAR dataset preprocessing
    n_embeddings = 64
    resvq_embeddings=128
    n_channels = 3
    num_quantizers = 8
    codebook_size = 3
    max_epochs = 10


    image_dataset = AllTargetImageDataset(image_h5_file)
    train_loader = DataLoader(image_dataset, batch_size=8, shuffle=False, pin_memory=True)


    # Initialize the RVQUNet for CIFAR-10 image reconstruction
    model = RVQUNetLightning(n_channels=n_channels, num_embeddings=n_embeddings, total_epochs=max_epochs,
                                num_quantizers=num_quantizers, resvq_embeddings=resvq_embeddings, codebook_size=codebook_size)

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath="checkpoints_flow",
        filename="rvq-unet",
        save_top_k=1,
        mode="min",
    )

    early_stopping_callback = EarlyStopping('train_loss')
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=True,
        log_every_n_steps=100,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader)
    best_model_path = checkpoint_callback.best_model_path
    best_model_dirname = os.path.dirname(best_model_path)
    torch.save({'n_channels': n_channels, 'num_embeddings': n_embeddings, 
                'num_quantizers': num_quantizers, 'total_epochs': max_epochs, 
                'resvq_embeddings': resvq_embeddings, 'codebook_size': codebook_size}, os.path.join(best_model_dirname, 'config.json'))

    best_model_dirname = 'checkpoints_flow/'
    best_model_path = os.path.join(best_model_dirname, 'rvq-unet.ckpt')
    best_model_config = os.path.join(best_model_dirname, 'config.json')
    config = torch.load(best_model_config)
    n_channels = config['n_channels']
    n_embeddings = config['num_embeddings']
    num_quantizers = config['num_quantizers']
    max_epochs = config['total_epochs']
    resvq_embeddings = config['resvq_embeddings']
    codebook_size = config['codebook_size']
    best_model = RVQUNetLightning.load_from_checkpoint(best_model_path, n_channels=n_channels, num_embeddings=n_embeddings, total_epochs=max_epochs,
                                num_quantizers=num_quantizers, resvq_embeddings=resvq_embeddings, codebook_size=codebook_size)
    best_model.eval()
    # Visualize some reconstructions from the best model
    with torch.no_grad():
        data = next(iter(train_loader))
        data = data.to(best_model.device)

        output = best_model(data)
        # output = output.sample
        data = data.cpu()
        output = output.sample.cpu()

        # denormalize
        data = image_dataset.denormalize(data)
        output = image_dataset.denormalize(output)

        # change data back to 255
        data = data * 255.
        output = output * 255.

        plot_visualization(data[:10], output[:10], save_filename='pretrain_reconstructions.png', num_items=min(data.shape[0], 10))

if __name__ == "__main__":
    main()