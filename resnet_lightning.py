import pytorch_lightning as pl
import torch.nn as nn
from torch import optim

from residual_net import ResidualMappingNetwork


class ResNetLightning(pl.LightningModule):
    def __init__(
        self,
        input_dim=128,
        output_dim=128,
    ):
        super(ResNetLightning, self).__init__()
        self.model = ResidualMappingNetwork(
            input_dim=output_dim, output_dim=output_dim
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        reconstruction_loss = self.criterion(output, target)
        self.log("train_loss", reconstruction_loss, prog_bar=True)
        return reconstruction_loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        reconstruction_loss = self.criterion(output, target)
        self.log("val_loss", reconstruction_loss, prog_bar=True)
        return reconstruction_loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.1)
