# -*- coding: utf-8 -*-
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F


class L1LossFlat(nn.SmoothL1Loss):
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return super().forward(input.view(-1), target.view(-1))


class EncodingBoxSubStage(nn.Module):
    """
    An encoding sub stage that performs convolution,
    activation, batch normalization, and addition.
    (No dropout).
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        dilation=1,
        stride=1,
        padding=1,
        activation=nn.SELU(),
    ):
        super(EncodingBoxSubStage, self).__init__()

        self.activation_function = activation
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding,
        )
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.activation_function(y)
        y = self.batch_norm(y)

        # Addition (Residual connection)
        if self.conv.stride == 1:
            return x + y

        return y


class DecodingBoxSubStage(EncodingBoxSubStage):
    """
    A decoding sub stage.
    Introduces dropout before every convolution.
    Can concatenate the input tensor to output of a previous sub stage.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        dilation=1,
        stride=1,
        padding=1,
        activation=nn.SELU(),
        dropout=0,  # CHANGED DROPOUT
        concatenate=False,
    ):
        super().__init__(
            in_channels * (2 if concatenate else 1),
            out_channels,
            kernel_size,
            dilation,
            stride,
            padding,
            activation,
        )
        self.concatenate = concatenate
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, concat_with=None):
        x = self.dropout(x)

        if self.concatenate:
            print(concat_with.shape , "c")
            print(x.shape , "x")
            x = torch.cat((concat_with, x), 1)
            print(x.shape)
        y = self.conv(x)
        y = self.activation_function(y)
        y = self.batch_norm(y)

        # Addition (Residual connection)
        if self.conv.stride == 1:
            return x + y
        print(y.shape , "y" )
        return y


class EncodingBox(nn.Module):
    """
    An encoding box that performs one encoding cycle.
    Uses four sub-stages with dilation rates
    1, 2, 3, 4. Outputs the results of the fourth
    and second sub stages.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        activation=nn.SELU(),
        rescale=True,
    ):
        super(EncodingBox, self).__init__()

        self.sub_stages = nn.ModuleList(
            [
                EncodingBoxSubStage(
                    in_channels=(in_channels if dilation == 1 else out_channels),
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=(2 if (dilation == 1 and rescale) else 1),
                    padding=dilation,
                    activation=activation,
                )
                for dilation in range(1, 5)
            ]
        )

    def forward(self, x):
        x = self.sub_stages[0](x)
        print(x.shape , "encoding ") 
        d2 = self.sub_stages[1](x)
        print(d2.shape)
        x = self.sub_stages[2](d2)
        x = self.sub_stages[3](x)

        return x, d2


class DecodingBox(nn.Module):
    """
    A helper module that performs one decoding cycle.
    Uses four sub stages with dilation rates
    4, 3, 2, 1. Concatenates `d4` and `d2` to the
    inputs of the first and third sub stage.
    """

    def __init__(
        self,
        in_channels=1,  # `The in_channels of the first sub stage`
        out_channels=1,  # The `out_channels` and `in_channels` of the middle sub stages.
        final_channels=None,  # `The out_channels` of the last sub stage
        kernel_size=3,
        stride=1,
        activation=nn.SELU(),
        final_activation=nn.SELU(),
        dropout=0.2,
    ):
        super(DecodingBox, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.sub_stages = nn.ModuleList(
            [
                DecodingBoxSubStage(
                    in_channels=(in_channels if (dilation == 4) else out_channels),
                    out_channels=(
                        out_channels
                        if (final_channels is None or dilation != 1)
                        else final_channels
                    ),
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=stride,
                    padding=dilation,
                    activation=(activation if (dilation != 1) else final_activation),
                    dropout=dropout,
                    concatenate={1: False, 2: True, 3: False, 4: True}[dilation],
                )
                for dilation in reversed(range(1, 5))
            ]
        )

    def forward(self, x, d4=None, d2=None):
        # Perform upsampling
        print(x.shape , "x")
        x = self.upsample(x)
        print(x.shape)
        print(d4.shape , "d")
        print(d2.shape)
        x = self.sub_stages[0](x, d4)
        print("d4")
        x = self.sub_stages[1](x)
        x = self.sub_stages[2](x, d2)
        x = self.sub_stages[3](x)
        print(x.shape)
        print()
        return x


class MResUNet(pl.LightningModule):
    """
    The modified Residual U-Net as specified in https://arxiv.org/pdf/2003.06135.pdf
    """

    def __init__(
        self,
        map_size=64,
        lr=0.001,
        input_channels=1,
        nb_enc_boxes=4,
        nb_channels_first_box=64,
        output_type="kappa_map",
        loss="mse",
        final_channels=1,
        final_relu=False,
        mass_plotter=None,
        trial=None,
        **kwargs,
    ):
        super(MResUNet, self).__init__()

        if trial is not None:
            self.lr = trial.suggest_loguniform("lr", 1e-5, 0.02)
            self.dropout = trial.suggest_uniform("dropout", 0, 0.5)
            self.nb_enc_boxes = trial.suggest_categorical("nb_enc_boxes", [3, 4])
            self.nb_channels_first_box = trial.suggest_categorical(
                "nb_channels_first_box", [32, 64]
            )
        else:
            self.lr = lr
            self.dropout = 0.2
            self.nb_enc_boxes = nb_enc_boxes
            self.nb_channels_first_box = nb_channels_first_box

        self.nb_channels_first_box = nb_channels_first_box
        self.output_type = output_type
        self.input_channels = input_channels
        self.final_relu = final_relu
        self.mass_plotter = mass_plotter
        self.return_y_in_pred = False
        self.map_size = map_size

        if loss == "msle":
            self.loss = lambda x, y: F.mse_loss(
                torch.log(x + 1e-14), torch.log(y + 1e-14)
            )
        else:
            # self.loss = nn.SmoothL1Loss()
            self.loss = F.mse_loss

        # `nb_enc_boxes` encoding boxes
        self.encoding = nn.ModuleList(
            [
                EncodingBox(
                    in_channels=input_channels,
                    out_channels=self.nb_channels_first_box,
                    kernel_size=3,
                    rescale=False,
                ),
                *[
                    EncodingBox(
                        in_channels=(2**i) * self.nb_channels_first_box,
                        out_channels=(2 ** (i + 1)) * self.nb_channels_first_box,
                        kernel_size=3,
                    )
                    for i in range(self.nb_enc_boxes - 1)
                ],
            ]
        )

        # (`nb_enc_boxes` - 1) decoding boxes
        self.decoding = nn.ModuleList(
            [
                *[
                    DecodingBox(
                        in_channels=(2 ** (i + 1)) * self.nb_channels_first_box,
                        out_channels=(2 ** (i + 1)) * self.nb_channels_first_box,
                        final_channels=(2 ** (i)) * self.nb_channels_first_box,
                        kernel_size=3,
                        dropout=self.dropout,
                    )
                    for i in reversed(range(self.nb_enc_boxes - 2))
                ],
                DecodingBox(
                    in_channels=self.nb_channels_first_box,
                    out_channels=self.nb_channels_first_box,
                    final_channels=final_channels,
                    kernel_size=3,
                    final_activation=nn.Identity(),
                    dropout=self.dropout,
                ),
            ]
        )

        # A convolution to divide the number of channels by 2 before the decoding stage
        self.reduce_channel = nn.Conv2d(
            in_channels=(2 ** (self.nb_enc_boxes - 1)) * self.nb_channels_first_box,
            out_channels=(2 ** (self.nb_enc_boxes - 2)) * self.nb_channels_first_box,
            kernel_size=3,
            padding=1,
        )

        if ["mass"] == self.output_type:
            self.final_layers = nn.Sequential(
                *[
                    nn.Flatten(),
                    nn.BatchNorm1d(
                        self.map_size * self.map_size,
                        eps=1e-05,
                        momentum=0.1,
                        affine=True,
                        track_running_stats=True,
                    ),
                    # nn.Dropout(p=0.1),
                    nn.Linear(self.map_size * self.map_size, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(
                        32,
                        eps=1e-05,
                        momentum=0.1,
                        affine=True,
                        track_running_stats=True,
                    ),
                    # nn.Dropout(p=0.1),
                    nn.Linear(32, 16, bias=True),
                    nn.BatchNorm1d(
                        16,
                        eps=1e-05,
                        momentum=0.1,
                        affine=True,
                        track_running_stats=True,
                    ),
                    nn.Linear(16, 1),
                ]
            )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MResUNet")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--nb_enc_boxes", type=int, default=4)
        parser.add_argument("--nb_channels_first_box", type=int, default=64)
        return parent_parser

    def forward(self, x):

        # Store outputs of sub-stages with dilation rates: 2, 4
        d4_list = []
        d2_list = []

        # Encoding
        for box in self.encoding:
            x, d = box(x)
            d4_list.append(torch.clone(x))
            d2_list.append(torch.clone(d))
            
        # Decoding
        x = self.reduce_channel(x)
        for i, box in enumerate(self.decoding):
            x = box(
                x,
                d4=d4_list[self.nb_enc_boxes - 2 - i],
                d2=d2_list[self.nb_enc_boxes - 2 - i],
            )

        if ["mass"] == self.output_type:
            mass = self.final_layers(x)
            print(mass.shape , "..")
            return mass.view((mass.shape[0], 1, 1, 1))

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        if "mass" in self.output_type:
            # Return values to plot graphs
            return {
                "loss": loss,
                "y": y.detach().cpu().numpy(),
                "y_hat": y_hat.detach().cpu().numpy(),
            }
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        val_loss = self.loss(y_hat, y)
        self.log("val_loss", val_loss)

        if "mass" in self.output_type:
            # Return values to plot graphs
            return {
                "val_loss": val_loss,
                "y": y.cpu().numpy(),
                "y_hat": y_hat.cpu().numpy(),
            }
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        """Plot graphs to writer at the end of each validation epoch"""

        if "mass" in self.output_type and self.mass_plotter is not None:
            self.mass_plotter.plot_all(outputs, self.current_epoch, step="validation")

    def training_epoch_end(self, outputs):
        """Plot graphs to writer at the end of each training epoch"""

        if "mass" in self.output_type and self.mass_plotter is not None:
            self.mass_plotter.plot_all(outputs, self.current_epoch, step="training")

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        if self.return_y_in_pred:
            return y, pred
        return pred
