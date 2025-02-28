import torch
import torch.nn as nn
import torch.nn.functional as F
from .UNet import UNetModel_newpreview
from .respace import SpacedDiffusion, space_timesteps
from . import gaussian_diffusion as gd

class MedSegDiffV1(nn.Module):
    def __init__(self, num_classes=2, num_channels=1, diffusion_steps=100):
        """
        MedSegDiff V1 for prostate segmentation

        :param num_classes: Number of output classes (prostate segmentation is 2 classes)
        :param num_channels: Number of input channels (MRI images are typically 1 channel)
        :param diffusion_steps: Number of steps in the Diffusion process
        """
        super(MedSegDiffV1, self).__init__()

        # Use UNet as Diffusion Backbone
        self.unet = UNetModel_newpreview(
            image_size=256, in_channels=num_channels,
            out_channels=num_classes, attention_resolutions=[16],
            dropout=0.1, channel_mult=(1, 1, 2, 2, 4, 4),
        )
        self.diffusion = create_gaussian_diffusion(steps=diffusion_steps)

    def forward(self, x, timesteps):
        """

        :param x: input MRI image (batch, 1, 256, 256)
        :param timesteps: Diffusion timestep (batch, )
        :return: predicted segementation (batch, num_classes, 256, 256)
        """
        # Unet process
        unet_output, _ = self.unet(x, timesteps)

        # Diffusion process
        diffusion_output = self.diffusion(unet_output, timesteps)

        return diffusion_output

    def sample(self, x, num_steps=100):
        """
        Reverse diffusion sampling to obtain final segmentation prediction.
        :param x: Initial noise input
        :param num_steps: Number of sampling steps
        :return: segmentation prediction
        """
        for t in reversed(range(num_steps)):
            x = self.diffusion.reverse_step(x, t)
        return x

# Introducing Gaussian Diffusion function from MedSegDiff script_util.py
def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    dpm_solver = False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        dpm_solver=dpm_solver,
        rescale_timesteps=rescale_timesteps,
    )


# Test the model
dummy_input = torch.randn(2, 1, 256, 256)  # input MRI images
timesteps = torch.randint(0, 100, (2,))  # Random timesteps

model = MedSegDiffV1(num_classes=2)
output = model(dummy_input, timesteps)

print("Model output shape:", output.shape)  # Expected (2, 2, 256, 256)