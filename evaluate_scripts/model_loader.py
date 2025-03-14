import torch
import argparse
from MedsegDiff_V1_Model.unet_copy import UNetModel_newpreview
from MedsegDiff_V1_Model.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
# Function copied from MedSegDiff-master-scripts-segmentation_train.py
def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="../dataset/brats2020/training",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint=None, #"/results/pretrainedmodel.pt"
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        multi_gpu = None, #"0,1,2"
        out_dir='./results/'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def custom_model_loader(path="../model/savedmodel105000.pt"):
    # Load model and do evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    args = create_argparser().parse_args()
    model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
    model = model.to(device)
    #model_path = "../model/savedmodel105000.pt"
    model.load_state_dict(torch.load(path))
    model.eval()  
    print("Load Model Success")

    return model

