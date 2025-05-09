import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch import optim
from vqvae import VQVAE
from ae.ae import AE


class RVQUNetLightning(pl.LightningModule):
    def __init__(
        self,
        n_channels,
        bilinear=True,
        interpolate_alpha=0.1,
        num_embeddings=(128, 128),
        resvq_embeddings=512,
        num_quantizers=8,
        codebook_size=256,
        total_epochs=10,
        warmup_epochs=5,
        initial_lr=8e-3,
        min_lr=1e-3,
        intermediate_training=False,
        model_type="vqvae",
    ):
        super(RVQUNetLightning, self).__init__()

        self.interpolate_alpha = interpolate_alpha
        self.intermediate_training = intermediate_training
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        # self.model = VQModel(in_channels=n_channels, num_vq_embeddings=codebook_size, vq_embed_dim=n_channels,
        #                     down_block_types=('DownEncoderBlock2D',),
        #                     block_out_channels=(num_embeddings,),
        #                     up_block_types=('UpDecoderBlock2D',))
        # self.model = UVit2DModel(n_channels, layers_per_block=1, num_vq_embeddings=resvq_embeddings, vq_embed_dim=codebook_size)

        model_factory = {"vqvae": VQVAE, "ae": AE}
        self.model = model_factory[model_type](
            in_channels=n_channels,
            embedding_dim=resvq_embeddings,
            num_embeddings=codebook_size,
            hidden_dims=num_embeddings,
        )
        # self.model = UNet(n_channels=n_channels, num_embeddings=num_embeddings, resvq_embeddings=resvq_embeddings,
        #   num_quantizers=num_quantizers, codebook_size=codebook_size)
        # self.model = UNet(n_channels=3, num_embeddings=num_embeddings, codebook_size=codebook_size, n_classes=3)

        self.criterion = nn.MSELoss()
        self.lowest_val_loss = float("inf")

    def get_latent(self, x):
        return self.model.get_latent(x)

    def generate_image(self, latent):
        return self.model.generate_image(latent)

    def forward(self, x):
        return self.model(x)

    def intermediate_step(self, batch, batch_idx, train_type="train"):
        data, intermediate, target = batch

        output, rvq_loss = self.model(data)
        data_rvq_loss = rvq_loss.mean()
        intermediate_output, rvq_loss = self.model(intermediate)
        intermediate_rvq_loss = rvq_loss.mean()
        target_output, rvq_loss = self.model(target)
        target_rvq_loss = rvq_loss.mean()

        # generate the mid point latent
        data_latent = self.model.get_latent(data)
        intermediate_latent = self.model.get_latent(intermediate)
        target_latent = self.model.get_latent(target)
        middle_latent = data_latent + (target_latent - data_latent) / 2
        intermediate_interpolate_output, intermediate_interpolate_rvq_loss = (
            self.model.generate_image(middle_latent, return_loss=True)
        )

        data_reconstruction_loss = self.criterion(output, data)
        intermediate_reconstruction_loss = self.criterion(
            intermediate_output, intermediate
        )
        latent_interpolation_loss = self.criterion(intermediate_latent, middle_latent)
        intermediate_interpolate_reconstruction_loss = self.criterion(
            intermediate_interpolate_output, intermediate
        )
        target_reconstruction_loss = self.criterion(target_output, target)

        _rvq_loss = torch.mean(
            data_rvq_loss
            + intermediate_rvq_loss
            + self.interpolate_alpha * intermediate_interpolate_rvq_loss
            + target_rvq_loss
        )
        _reconstruction_loss = torch.mean(
            data_reconstruction_loss
            + intermediate_reconstruction_loss
            + target_reconstruction_loss
            + self.interpolate_alpha * intermediate_interpolate_reconstruction_loss
        )

        loss = (
            _rvq_loss
            + _reconstruction_loss
            + self.interpolate_alpha * latent_interpolation_loss
        )

        self.log(f"{train_type}_loss", loss, prog_bar=False)
        self.log(f"{train_type}_recon_loss", _reconstruction_loss, prog_bar=True)
        self.logger.experiment.log_metric(
            run_id=self.logger.run_id,
            key=f"{train_type}_recon_loss",
            value=_reconstruction_loss,
            step=self.global_step,
        )
        return loss

    def training_step(self, batch, batch_idx):
        if self.intermediate_training:
            return self.training_intermediate_step(batch, batch_idx)

        data, intermediate, target = batch

        output, rvq_loss = self.model(data)
        data_rvq_loss = rvq_loss.mean()
        intermediate_output, rvq_loss = self.model(intermediate)
        intermediate_rvq_loss = rvq_loss.mean()
        target_output, rvq_loss = self.model(target)
        target_rvq_loss = rvq_loss.mean()

        data_reconstruction_loss = self.criterion(output, data)
        intermediate_reconstruction_loss = self.criterion(
            intermediate_output, intermediate
        )
        target_reconstruction_loss = self.criterion(target_output, target)

        train_rvq_loss = torch.mean(
            data_rvq_loss + intermediate_rvq_loss + target_rvq_loss
        )
        train_reconstruction_loss = torch.mean(
            data_reconstruction_loss
            + target_reconstruction_loss
            + intermediate_reconstruction_loss
        )

        loss = train_rvq_loss + train_reconstruction_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_loss", train_reconstruction_loss, prog_bar=True)
        return loss

    def training_intermediate_step(self, batch, batch_idx):
        return self.intermediate_step(batch, batch_idx, train_type="train")

    def validation_step(self, batch, batch_idx):
        if self.intermediate_training:
            return self.validation_intermediate_step(batch, batch_idx)

        data, intermediate, target = batch

        output, rvq_loss = self.model(data)
        data_rvq_loss = rvq_loss.mean()
        intermediate_output, rvq_loss = self.model(intermediate)
        intermediate_rvq_loss = rvq_loss.mean()
        target_output, rvq_loss = self.model(target)
        target_rvq_loss = rvq_loss.mean()

        # rvq_loss = output.commit_loss
        data_reconstruction_loss = self.criterion(output, data)
        intermediate_reconstruction_loss = self.criterion(
            intermediate_output, intermediate
        )
        target_reconstruction_loss = self.criterion(target_output, target)

        val_rvq_loss = torch.mean(
            data_rvq_loss + intermediate_rvq_loss + target_rvq_loss
        )
        val_reconstruction_loss = torch.mean(
            data_reconstruction_loss
            + target_reconstruction_loss
            + intermediate_reconstruction_loss
        )

        loss = val_rvq_loss + val_reconstruction_loss

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", val_reconstruction_loss, prog_bar=True)
        return loss

    def validation_intermediate_step(self, batch, batch_idx):
        return self.intermediate_step(batch, batch_idx, train_type="val")

    def configure_optimizers(self):
        # learning_rate = 10 / np.sqrt(sum(p.numel() for p in self.model.parameters()))
        # optimizer = Lion(self.parameters(), lr=1e-3, weight_decay=0)
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)

        # Create a warmup scheduler
        # warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=self.warmup_epochs)

        # Create a cosine annealing scheduler
        # cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.total_epochs - self.warmup_epochs, eta_min=self.min_lr)

        return optimizer  # [optimizer], [warmup_scheduler, cosine_scheduler]
