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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from second_united import model3

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
            

        mel_mask = get_mask_from_lengths(torch.tensor([mel.shape[2]])) 
        mel_mask[:,:mel_lengths.item()] = True

        mel_mask = mel_mask.to(device)


        
        mel = mel.masked_select(mel_mask.unsqueeze(1))

        mel_output = mel_output.masked_select(mel_mask.unsqueeze(1))
  
        pitch = pitch.masked_select(mel_mask.unsqueeze(1))
        pitch_out = pitch_out.masked_select(mel_mask.unsqueeze(1))
        
        energy = energy.masked_select(mel_mask.unsqueeze(1))
        energy_out = energy_out.masked_select(mel_mask.unsqueeze(1))
     
        mel_loss = nn.L1Loss()(mel_output, mel)
        pitch_loss = nn.MSELoss()(pitch_out, pitch)
        energy_loss = nn.MSELoss()(energy_out, energy)

        return mel_loss,pitch_loss,energy_loss


class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model3, device, data_len ,data, mels,lengths,  criterion, optimizer,iter_meter):
    full_loss=0
    losses = []
    r = list(range(data_len))
    random.shuffle(r)
    for i in r:
        word, mel,pitch, energy ,leen = torch.tensor(data[i][0]).to(torch.float32), mels[i].to(torch.float32) ,torch.tensor(data[i][2]).to(torch.float32), torch.tensor(data[i][3]).to(torch.float32),lengths[i]
        len = torch.tensor([leen]) 
 
        word, mel,pitch, energy,len = word[:,None].to(device),mel[None,:,:].to(device),pitch[None,:].to(device) , energy[None,:].to(device),len.to(device)
      
        mel_cut = mel[:,:,:len]
        optimizer.zero_grad()

        with torch.enable_grad():
             mel_out, pitch_out,energy_out,length = model3(word, mel,len)


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
           

        loss = criterion(mel_out,mel_cut)  +   criterion(pitch,pitch_out) +  criterion(energy,energy_out)   # УПРОСТИЛА ЛОСС с Transformer loss НА ПРОСТО MSE -> ОШИБКА НИКУДА НЕ ДЕЛАСЬ

        loss.retain_grad()
        model3.mylin3.weight.retain_grad() 
        loss.backward(retain_graph = True)

        print (model3.mylin3.weight.grad)            # NONE

        for name, p in model3.named_parameters():
             print(name, p.grad)                          #ВСЕГДА ПИШЕТ NONE ДЛЯ ВЕСОВ NET МОДЕЛИ , НО ДЛЯ ВЕСОВ FASTSPEECH ВСЕ ОК
      
       '''  myconv1.weight None
            myconv1.bias None
            myconv2.weight None
            myconv2.bias None
            myfc1.weight None                 #ВСЕГДА ПИШЕТ NONE ДЛЯ ВЕСОВ NET МОДЕЛИ , ДЛЯ ВЕСОВ FASTSPEECH ВСЕ ОК , ЕСЛИ ИХ ВКЛЮЧИТЬ
            myfc1.bias None
            myfc2.weight None
            myfc2.bias None
            myfc3.weight None
            myfc3.bias None
            mylin3.weight None
            mylin3.bias None
            '''
        optimizer.step()
        iter_meter.step()
        losses.append(loss.item())
        full_loss +=loss.item() 
         
    return(full_loss,losses)    




        
def count_parameters(model):
    for name, p in model.named_parameters():
        if p.requires_grad :
            print('IN TRAIN', name)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)        #ПИШЕТ ЧТО ВСЕ ВЕСА КОТОРЫЕ Я ЗАДАЛА REQUIRE GRAD ТАКИМИ И ЯВЛЯЮТСЯ


iter_meter = IterMeter()


criterion = torch.nn.MSELoss().to(device)

optimizer = torch.optim.Adam(list( model3.parameters()) , lr=3e-4)


model3.train()
count_parameters(model3)
epochs = 1

for epoch in range(1, epochs + 1):
      full_loss ,losses  = train(model3, device,1,datas, mel_pad, input_lengths, criterion, optimizer,  iter_meter)
      print("Epoch  ",  epoch , "MEAN LOSS IS  " , full_loss/1)

      
