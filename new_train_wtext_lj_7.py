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


#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('losses')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from another_architecture_lj_4 import model3

model3.to(device)


from preprocess_right_lj_4 import datas


class TransformerLoss(nn.Module):
    def __init__(self):
        super(TransformerLoss, self).__init__()
        
    def forward(self, pred, target, lengths):
        mel_output, pitch_out, energy_out,duration_out = pred
        mel, pitch, energy,dur = target
        text_lengths, mel_lengths = lengths
       # print(mel_lengths)
        assert(mel_output.shape == mel.shape)                  #check for the shape
        assert(pitch_out.shape == pitch.shape)
        assert(energy_out.shape == energy.shape)
        assert(duration_out.shape == dur.shape)
    
        mel_mask = ~get_mask_from_lengths(mel_lengths)        # same for pitch and energy
        #print(mel_mask)
        #mel_mask = get_mask_from_lengths(torch.tensor([mel.shape[2]])) 
        #mel_mask[:,:mel_lengths.item()] = True
        #print(mel_mask)
        duration_mask = ~get_mask_from_lengths(text_lengths)
        #print(duration_mask) 
        mel_mask = mel_mask.to(device)
        duration_mask = duration_mask.to(device)

        #print(mel)        
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
        dur = dur.masked_select(duration_mask)
        duration_out = duration_out.masked_select(duration_mask)
    
        mel_loss = nn.L1Loss()(mel_output, mel)
        pitch_loss = nn.MSELoss()(pitch_out, pitch)
        energy_loss = nn.MSELoss()(energy_out, energy)
        duration_loss = nn.MSELoss()(duration_out, dur)

        return mel_loss,pitch_loss,energy_loss,duration_loss

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
        word, mel,pitch, energy ,dur = torch.tensor(data[i][0]).to(torch.float32),torch.tensor(data[i][1]).to(torch.float32) ,torch.tensor(data[i][2]).to(torch.float32), torch.tensor(data[i][3]).to(torch.float32), torch.tensor(data[i][4]).to(torch.float32)

      
        s = mel.shape[1] - 1
        word, mel,pitch, energy, dur = word[:,None].to(device),mel[None,:,:s].to(device),pitch[None,:s].to(device) , energy[None,:s].to(device)  , dur[None,:].to(device)
        #print('DUR' , dur.shape , word.shape)     
        optimizer.zero_grad()
        with torch.enable_grad():
             mel_out, pitch_out,energy_out,duration_out, len  = model3(word, mel,dur)

        mel_cut = mel  
         

        mel_loss,pitch_loss,energy_loss, duration_loss = criterion((mel_out, pitch_out, energy_out,duration_out),(mel_cut,  pitch, energy,dur),(torch.tensor([word.shape[0]]), torch.tensor([len])))

     
        loss = (mel_loss+pitch_loss+energy_loss +duration_loss)/(hparams.accumulation)
    
 
        loss.backward()
        #print(model3.MyNet.myconv1.weight.grad)
        optimizer.step()
        iter_meter.step()
     
        full_loss +=loss.item() 
       # torch.save(mel_out, "/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/mels/mels_wtextonly.pt")
            
       # torch.save(mel_out, "/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/mels/mels_from_onlytext.pt")
 
             
 
    return(full_loss)    


def check(model3, device, data_len ,data, iter_meter):
    for i in range(data_len):
        #word, mel,pitch, energy  = torch.tensor(data[i][0]).to(torch.float32),torch.tensor(data[i][1]).to(torch.float32) ,torch.tensor(data[i][2]).to(torch.float32), torch.tensor(data[i][3]).to(torch.float32)
        word, mel,pitch, energy ,dur = torch.tensor(data[i][0]).to(torch.float32),torch.tensor(data[i][1]).to(torch.float32) ,torch.tensor(data[i][2]).to(torch.float32), torch.tensor(data[i][3]).to(torch.float32), torch.tensor(data[i][4]).to(torch.float32)


      
        s = mel.shape[1] - 1
        word, mel,pitch, energy, dur = word[:,None].to(device),mel[None,:,:s].to(device),pitch[None,:s].to(device) , energy[None,:s].to(device)  , dur[None,:].to(device)
             
        
        with torch.no_grad():
             mel_out, pitch_out,energy_out,duration_out, len  = model3(word, mel,dur)

        mel_cut = mel
        mel_loss,pitch_loss,energy_loss, duration_loss = criterion((mel_out, pitch_out, energy_out,duration_out),(mel_cut,  pitch, energy,dur),(torch.tensor([word.shape[0]]), torch.tensor([len])))

     
        loss = (mel_loss+pitch_loss+energy_loss +duration_loss)/(hparams.accumulation)
        print(loss) 
        torch.save(mel_out, "/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/mels/mels_from_22maytext_lj_{}.pt".format(i))
            
        #torch.save(mel_cut, "/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/mels/mels_from_ref_22maytext_lj_{}.pt".format(i))


        iter_meter.step()
        

iter_meter = IterMeter()

criterion = TransformerLoss().to(device)
#optimizer = torch.optim.Adam(model3.myfc2.parameters(),lr=3e-4,betas=(0.9, 0.98),eps=1e-09)
#criterion = torch.nn.MSELoss().to(device)
#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model3.parameters()) , lr=3e-4)

#optimizer = torch.optim.Adam(list(model3.parameters()) , lr=3e-4 ,betas=(0.9, 0.98),eps=1e-09)
optimizer = torch.optim.Adam(list(model3.parameters()) , lr=3e-4 ,betas=(0.9, 0.98),eps=1e-09)

check(model3, device, 5 ,datas, iter_meter)

#print(model3)



'''
model3.train()

epochs = 100

for epoch in range(1, epochs + 1):
      full_loss   = train(model3, device,np.shape(datas)[0],datas, criterion, optimizer,  iter_meter)
      print("Epoch  ",  epoch+50 , "MEAN LOSS IS  " , full_loss/np.shape(datas)[0])
      #writer.add_scalar('training loss', full_loss/datas.shape[0], epoch) 

      
      if ((epoch  == 25)or(epoch == 50)or(epoch == 75)or(epoch == 100)): 
          #  with open("/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/metrics/metrics_right_27april_withoutconvs_{}.txt".format(epoch), "w") as output:
          #      output.write(str(losses))
          torch.save(model3.state_dict(), "/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/weights/weights_19may_NEW_opt4_{}".format(epoch+50))

'''      