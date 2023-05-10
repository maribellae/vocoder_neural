import sys
sys.path.append("..")

from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pylab
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa    
import glob
import logging
import hashlib
import sys
import speechbrain
import torch
import torchaudio
import sentencepiece
from types import SimpleNamespace
from torch.nn import SyncBatchNorm
from torch.nn import DataParallel as DP
from hyperpyyaml import load_hyperpyyaml
from speechbrain.pretrained.fetching import fetch
from speechbrain.dataio.preprocess import AudioNormalizer
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from speechbrain.utils.data_utils import split_path
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.batch import PaddedBatch, PaddedData
from speechbrain.utils.data_pipeline import DataPipeline
from speechbrain.utils.callchains import lengths_arg_exists
from speechbrain.utils.superpowers import import_from_path
import os
import matplotlib
import pylab
import librosa
import librosa.display
import numpy as np
from speechbrain.dataio.preprocess import AudioNormalizer

#import IPython.display as ipd

from librosa.feature.spectral import melspectrogram
from sklearn.preprocessing import MinMaxScaler
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import tqdm
import torch
import argparse
import numpy as np
import hparams as hp
from stft import TacotronSTFT
#from utils.utils import read_wav_np
from audio_processing import pitch
from text import phonemes_to_sequence
import pandas as pd


import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import unidecode
#import IPython.display as ipd
import pickle as pkl
import librosa
from text import *
import numpy as np
import torch
import hparams
from modules.model import Model
from text import phonemes_to_sequence
from g2p_en import G2p
from text.cleaners import punctuation_removers

import os
import glob
import tqdm
import torch
import argparse
import numpy as np
import hparams as hp
from stft import TacotronSTFT
from utils.utils import read_wav_np
from audio_processing import pitch

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np

import librosa
import shutil
from speechbrain.utils.data_utils import download_file
from speechbrain.pretrained import EncoderDecoderASR


import matplotlib

import pylab
import librosa
import librosa.display
import numpy as np

import IPython
import IPython.display as ipd

import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.model import Model
#from modules.loss import TransformerLoss
import hparams
from text import *
from utils.utils import *
#from utils.writer import get_writer
from utils.plot_image import *
import random
from utils.utils import get_mask_from_lengths


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('losses')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from another_architecture import model3

model3.to(device)


from preprocess_right1 import  input_lengths, mel_pad , datas


class TransformerLoss(nn.Module):
    def __init__(self):
        super(TransformerLoss, self).__init__()
        
    def forward(self, pred, target, lengths):
        mel_output, pitch_out, energy_out = pred
        mel, pitch, energy = target
        text_lengths, mel_lengths = lengths
       # print(mel_lengths)
        assert(mel_output.shape == mel.shape)                  #check for the shape
        assert(pitch_out.shape == pitch.shape)
        assert(energy_out.shape == energy.shape)
            
        #mel_mask = ~get_mask_from_lengths(mel_lengths)        # same for pitch and energy
        mel_mask = get_mask_from_lengths(torch.tensor([mel.shape[2]])) 
        mel_mask[:,:mel_lengths.item()] = True
        #print(mel_mask)
       # duration_mask = ~get_mask_from_lengths(text_lengths)
        mel_mask = mel_mask.to(device)
       # duration_mask = duration_mask.to(device)

        
        mel = mel.masked_select(mel_mask.unsqueeze(1))
        #print(mel)
        mel_output = mel_output.masked_select(mel_mask.unsqueeze(1))
        #print(mel_output)
        pitch = pitch.masked_select(mel_mask.unsqueeze(1))
        pitch_out = pitch_out.masked_select(mel_mask.unsqueeze(1))
        
        energy = energy.masked_select(mel_mask.unsqueeze(1))
        energy_out = energy_out.masked_select(mel_mask.unsqueeze(1))
  #      print("dddddddd")
  #      print( pitch,pitch_out)
  #      print(energy , energy_out)
  #      print(mel, mel_output)     
        mel_loss = nn.L1Loss()(mel_output, mel)
        pitch_loss = nn.MSELoss()(pitch_out, pitch)
        energy_loss = nn.MSELoss()(energy_out, energy)

        return mel_loss,pitch_loss,energy_loss

#optimizer = torch.optim.Adam(  list(model3.parameters())  , lr=3e-4)

#criterion = torch.nn.MSELoss().to(device)


class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model3, device, data_len ,data,  criterion, optimizer,iter_meter):
    full_loss=0
    losses = []
    r = list(range(data_len))
    random.shuffle(r)
    for i in r:
        word, mel,pitch, energy  = torch.tensor(data[i][0]).to(torch.float32),torch.tensor(data[i][1]).to(torch.float32) ,torch.tensor(data[i][2]).to(torch.float32), torch.tensor(data[i][3]).to(torch.float32)
      
      
        word, mel,pitch, energy = word[:,None].to(device),mel[None,:,:].to(device),pitch[None,:].to(device) , energy[None,:].to(device)
             
        optimizer.zero_grad()
        with torch.enable_grad():
             mel_out, pitch_out,energy_out,duration_out = model3(word, mel)

        mel_cut = mel


        if (mel_cut.shape[2]!=pitch.shape[1]) :
            print("!")
            delta = mel_cut.size(2) - pitch.size(1)
            pitch = pitch[None,:,:]
            pitch = torch.nn.functional.pad(pitch ,  (0, delta), "constant", 0)
            pitch = pitch.squeeze(0)
        if (mel_out.shape[2]!=pitch_out.shape[1]) :
            print("!!")          
            delta = mel_out.size(2) - pitch_out.size(1)
            pitch_out = pitch_out[None,:,:]
            pitch_out = torch.nn.functional.pad(pitch_out ,  (0, delta), "constant", 0)
            pitch_out = pitch_out.squeeze(0)
        if (mel_out.shape[2]!=energy_out.shape[1]) : 
            print("!!!")         
            delta = mel_out.size(2) - energy_out.size(1)
            energy_out = energy_out[None,:,:]
            energy_out = torch.nn.functional.pad(energy_out ,  (0, delta), "constant", 0)
            energy_out = energy_out.squeeze(0)
        if (mel_cut.shape[2]!=energy.shape[1]) :
            print("!!!!")           
            delta = mel_cut.size(2) - energy.size(1)
            energy = energy[None,:,:]
            energy = torch.nn.functional.pad(energy ,  (0, delta), "constant", 0)
            energy= energy.squeeze(0)

        if (mel_cut.size(2)- mel_out.size(2))<0 :
            delta = mel_out.size(2)-mel_cut.size(2) 
            mel_cut = torch.nn.functional.pad(mel_cut ,  (0, delta), "constant", 0)
            delta = pitch_out.size(1) - pitch.size(1)  
            
            pitch = pitch[None,:,:]
            pitch = torch.nn.functional.pad(pitch ,  (0, delta), "constant", 0)
            delta = energy_out.size(1) - energy.size(1)
            energy = energy[None,:,:]
            energy = torch.nn.functional.pad(energy,  (0, delta), "constant", 0)
            pitch = pitch.squeeze(0)
            energy = energy.squeeze(0)
        elif (mel_cut.size(2)- mel_out.size(2))>0 :
            
            
            delta = mel_cut.size(2) - mel_out.size(2)
           
            mel_out = torch.nn.functional.pad(mel_out ,  (0, delta), "constant", 0)
            delta = pitch.size(1) - pitch_out.size(1) 
            
            pitch_out= pitch_out[None,:,:]

            pitch_out = torch.nn.functional.pad(pitch_out ,  (0, delta), "constant", 0)

            delta = energy.size(1) - energy_out.size(1) 
            energy_out = energy_out[None,:,:]
            energy_out = torch.nn.functional.pad(energy_out,  (0, delta), "constant", 0)
            pitch_out = pitch_out.squeeze(0)
            energy_out = energy_out.squeeze(0)
        else:
            if (pitch.size(1)-pitch_out.size(1))<0 :
                delta = pitch_out.size(1) - pitch.size(1)  
         
                pitch = pitch[None,:,:]
            
                pitch = torch.nn.functional.pad(pitch ,  (0, delta), "constant", 0)
                delta = energy_out.size(1) - energy.size(1)
                energy = energy[None,:,:]

                energy = torch.nn.functional.pad(energy,  (0, delta), "constant", 0)
                pitch = pitch.squeeze(0)
                energy = energy.squeeze(0)
           




        mel_loss,pitch_loss,energy_loss = criterion((mel_out, pitch_out, energy_out),(mel_cut,  pitch, energy),(torch.tensor([word.shape[0]]), torch.tensor([len])))

     
        loss = (mel_loss+0.5*pitch_loss+0.5*energy_loss)/(hparams.accumulation)
    
 
        loss.backward()

        optimizer.step()
        iter_meter.step()
     
        full_loss +=loss.item() 
   
             
 
    return(full_loss)    


def check(model3, device, data_len ,data, mels,lengths,  iter_meter):
    for i in range(data_len):
        word, mel,pitch, energy ,leen = torch.tensor(data[i][0]).to(torch.float32), mels[i].to(torch.float32) ,torch.tensor(data[i][2]).to(torch.float32), torch.tensor(data[i][2]).to(torch.float32),lengths[i]
        len = torch.tensor([leen])  
        word, mel,pitch, energy,len = word[:,None].to(device),mel[None,:,:].to(device),pitch[None,:].to(device) , energy[None,:].to(device),len.to(device)
      #  print(word)
        

      #  print("ELEMENT " ,i )
        mel_out, pitch_out,energy_out,length = model3(word, mel,len)
        
        mel_cut = mel[:,:,:len]
        
  


        if (mel_cut.size(2)- mel_out.size(2))<0 :
        #    print("A")
            origin =mel_cut.size(2)
            delta = mel_out.size(2)-mel_cut.size(2) -1
         #   print(delta) 
            mel_cut = torch.nn.functional.pad(mel_cut ,  (1, delta), "constant", 0)
            delta = pitch_out.size(1) - pitch.size(1) - 1 
          #  print(delta) 
            pitch = pitch[None,:,:]
            energy = energy[None,:,:]
            
            pitch = torch.nn.functional.pad(pitch ,  (1, delta), "constant", 0)
            energy = torch.nn.functional.pad(energy,  (1, delta), "constant", 0)
            pitch = pitch.squeeze(0)
            energy = energy.squeeze(0)
        elif (mel_cut.size(2)- mel_out.size(2))>0 :
           # print("B")
            origin = mel_cut.size(2)
            delta = mel_cut.size(2) - mel_out.size(2)-1
            #print(delta)
            mel_out = torch.nn.functional.pad(mel_out ,  (1, delta), "constant", 0)
            delta = pitch.size(1) - pitch_out.size(1) - 1 
            #print(delta)
            pitch_out= pitch_out[None,:,:]
            energy_out = energy_out[None,:,:]
            pitch_out = torch.nn.functional.pad(pitch_out ,  (1, delta), "constant", 0)
            energy_out = torch.nn.functional.pad(energy_out,  (1, delta), "constant", 0)
            pitch_out = pitch_out.squeeze(0)
            energy_out = energy_out.squeeze(0)
        else:
            if (pitch.size(1)-pitch_out.size(1))<0 :
                delta = pitch_out.size(1) - pitch.size(1)  
          #  print(delta) 
                pitch = pitch[None,:,:]
                energy = energy[None,:,:]
            
                pitch = torch.nn.functional.pad(pitch ,  (0, delta), "constant", 0)
                energy = torch.nn.functional.pad(energy,  (0, delta), "constant", 0)
                pitch = pitch.squeeze(0)
                energy = energy.squeeze(0)

            torch.save(mel_output, "/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/mels/mels_from_model3_right_{}.pt".format(i))
            
            torch.save(mel_cut, "/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/mels/mels_from_ref_right_{}.pt".format(i))


        iter_meter.step()
        

iter_meter = IterMeter()

criterion = TransformerLoss().to(device)
#optimizer = torch.optim.Adam(model3.myfc2.parameters(),lr=3e-4,betas=(0.9, 0.98),eps=1e-09)
#criterion = torch.nn.MSELoss().to(device)
#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model3.parameters()) , lr=3e-4)

optimizer = torch.optim.Adam(list(model3.parameters()) , lr=3e-4 ,betas=(0.9, 0.98),eps=1e-09)


#print(model3)
model3.train()

epochs = 20

for epoch in range(1, epochs + 1):
      full_loss ,losses  = train(model3, device,datas.shape[0],datas, mel_pad, input_lengths, criterion, optimizer,  iter_meter)
      print("Epoch  ",  epoch , "MEAN LOSS IS  " , full_loss/datas.shape[0])
      writer.add_scalar('training loss', full_loss/datas.shape[0], epoch) 

      
      if ((epoch  == 1)or(epoch  == 11)or(epoch == 21)): 
         #  with open("/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/metrics/metrics_right_27april_withoutconvs_{}.txt".format(epoch), "w") as output:
          #      output.write(str(losses))
           torch.save(model3.state_dict(), "/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/weights/weights_right_27april_withoutconvs_{}".format(epoch-1))
      
