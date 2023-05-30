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


for parameter in model3.parameters():
    parameter.requires_grad = False  


  
#                                                              CHANGE HERE

model3.load_state_dict(torch.load("/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/weights/weights_27may_newidea_60000"))



model = model3.to(device)
    

def checking():
    train_loader, val_loader, collate_fn = prepare_dataloaders(hparams)
    model = model3.cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=384**-0.5 ,
                                 betas=(0.9, 0.98),
                                 eps=1e-09)
    criterion = TransformerLoss()
    #writer = get_writer(hparams.output_directory, hparams.log_directory)

    iteration, epoch, loss = 0, 0,0
    model.eval()

    while iteration < 1:
        for i, batch in enumerate(train_loader):
            text_padded, text_lengths, mel_padded, mel_lengths, duration_padded, pitch_padded, energy_padded , embeddings= [
                reorder_batch(x, hparams.n_gpus).cuda() for x in batch
            ]
            with torch.no_grad(): 
                mel_out,durations_out, durations, pitch,energy  = model.outputs( text_padded, mel_padded,duration_padded,text_lengths, mel_lengths,embeddings )
            iteration += 1
            
#                                                              CHANGE HERE            
            torch.save(mel_out, "/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/mels/mels_from_model_27may_only4.pt")
            torch.save(mel_padded , "/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/mels/mels_from_ref_27may_only4.pt")
                    
            if iteration==(hparams.train_steps):
                break

      
checking()