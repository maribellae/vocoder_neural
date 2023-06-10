import sys
sys.path.append("..")

import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.loss import TransformerEvalLoss
import hparams
from text import *
from utils.utils import *
from utils.writer import get_writer
from utils.plot_image import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from another_architecture_lj_9 import model3

for parameter in model3.parameters():
    parameter.requires_grad = False  

model3.load_state_dict(torch.load("./weights/weights_27may_newidea_decoder_lr290000"))

model = model3.to(device)
    

def evaluate():
    train_loader, val_loader, collate_fn = prepare_dataloaders(hparams)
    model = model3.cuda()
    
    criterion = TransformerEvalLoss()

    full_mse , full_dur , full_pitch , full_energy , full_psnr , full_ssim = 0,0,0,0,0,0
    model.eval()
    print("Evaluation Start!!!")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            text_padded, text_lengths, mel_padded, mel_lengths, duration_padded, pitch_padded, energy_padded , embeddings= [
                reorder_batch(x, hparams.n_gpus).cuda() for x in batch
            ]

            mel_loss, duration_loss, pitch_loss, energy_loss , psnr_loss , ssim_loss  = model.evaluation(text_padded,
                                            mel_padded,
                                            duration_padded,
                                            text_lengths,
                                            mel_lengths,
                                            pitch_padded,
                                            energy_padded,
                                            embeddings,
                                            criterion)
            mel_loss, duration_loss, pitch_loss, energy_loss , psnr_loss , ssim_loss = [
                torch.sum(x) for x in [mel_loss, duration_loss, pitch_loss, energy_loss , psnr_loss , ssim_loss]
            ]    

            full_mse += mel_loss
            full_dur += duration_loss
            full_pitch += pitch_loss
            full_energy += energy_loss
            full_psnr += psnr_loss
            full_ssim += ssim_loss
    return ( full_mse , full_dur , full_pitch , full_energy , full_psnr , full_ssim)            
        
full_mse , full_dur , full_pitch , full_energy , full_psnr , full_ssim = evaluate()
print ( full_mse , full_dur , full_pitch , full_energy , full_psnr , full_ssim )
