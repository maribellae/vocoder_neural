import sys
sys.path.append("..")

import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.loss import TransformerLoss
import hparams
from text import *
from utils.utils import *
from utils.writer import get_writer
from utils.plot_image import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from another_architecture_lj_9 import model3

model = model3.to(device)
    

def training():
    train_loader, val_loader, collate_fn = prepare_dataloaders(hparams)
    model = model3.cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=384**-0.5 ,
                                 betas=(0.9, 0.98),
                                 eps=1e-09)
    criterion = TransformerLoss()
    #writer = get_writer(hparams.output_directory, hparams.log_directory)

    iteration, epoch, loss = 0, 0,0
    model.train()
    print("Training Start!!!")
    while iteration < (hparams.train_steps):
        for i, batch in enumerate(train_loader):
            text_padded, text_lengths, mel_padded, mel_lengths, duration_padded, pitch_padded, energy_padded , embeddings= [
                reorder_batch(x, hparams.n_gpus).cuda() for x in batch
            ]

            mel_loss, duration_loss, pitch_loss, energy_loss  = model(text_padded,
                                            mel_padded,
                                            duration_padded,
                                            text_lengths,
                                            mel_lengths,
                                            pitch_padded,
                                            energy_padded,
                                            embeddings,
                                            criterion)

            mel_loss, duration_loss, pitch_loss, energy_loss  = [
                torch.mean(x) for x in [mel_loss, duration_loss, pitch_loss, energy_loss]
            ]
            sub_loss = (mel_loss+duration_loss+pitch_loss+energy_loss)/hparams.accumulation
            sub_loss.backward()
 
            loss = loss + sub_loss.item()
            lr_scheduling(optimizer, iteration+1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
            optimizer.step()
            model.zero_grad()


            if (iteration%1000 == 0 ) :
               with open("/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/metrics/metrics_27may_newidea_decoder_lr2.txt", "a") as output:
                    output.write(str(loss/1000) + '\n')
               loss = 0
            
            if (iteration%10000 == 0 ) :
               torch.save(model.state_dict(), "/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/weights/weights_27may_newidea_decoder_lr2{}".format(iteration))

            iteration += 1

            if iteration==(hparams.train_steps):
                break


            
        
training()