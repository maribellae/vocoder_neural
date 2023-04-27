#import sys
#sys.path.append("..")

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
import IPython.display as ipd
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
from utils.utils import read_wav_np
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
from modules.loss import TransformerLoss
import hparams
from text import *
from utils.utils import *
#from utils.writer import get_writer
from utils.plot_image import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_path = f"checkpoints_80000"
state_dict = {}
for k, v in torch.load(checkpoint_path , map_location=torch.device('cpu'))['state_dict'].items():
   state_dict[k[7:]]=v



def count_parameters(model):
    #for name, p in model.named_parameters():
    #    if p.requires_grad :
    #        print('dddd', name)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = lengths.new_tensor(torch.arange(0, max_len)) #torch.arange(0, max_len)       lengths.new_tensor(torch.arange(0, max_len)), giving some warning
    mask = (lengths.unsqueeze(1) <= ids).to(torch.bool)
    return mask  
    


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.myconv1 = nn.Conv2d(1, 3, 3)
        self.mypool = nn.MaxPool2d(2, 2)
        self.myconv2 = nn.Conv2d(3, 1, 3)
        self.myfc1 = nn.Linear(9792, 60)
        self.myfc2 = nn.Linear(60, 20)
         
        #self.myfc2 = nn.Linear(120, 84)
        self.myfc3 = nn.Linear(20,256)
 

class MyFastSpeech(Model):
    def __init__(self):
        super(MyFastSpeech,self).__init__(hparams)
        self.hp = hp
        self.load_state_dict(state_dict)
      #  print(self.Pitch.linear.weight)

        self.Embedding = nn.Embedding(1050, hp.symbols_embedding_dim)
       # self.Embedding = nn.Linear(60, 256)     
        

def energy_to_one_hot(e, is_inference = False, is_log_output = False, offset = 1):                                        #check for scale
    #e = de_norm_mean_std(e, hp.e_mean, hp.e_std)
    # For pytorch > = 1.6.0
    bins = torch.linspace(hparams.e_min, hparams.e_max, steps=255).to(torch.device("cuda" if hparams.ngpu > 0 else "cpu"))
    if is_inference and is_log_output:
        e = torch.clamp(torch.round(e.exp() - offset), min=0).long()
        
    e_quantize = bucketize(e.to(torch.device("cuda" if hparams.ngpu > 0 else "cpu")), bins)

    return F.one_hot(e_quantize.long(), 256).float()
    
    
def pitch_to_one_hot(f0, is_inference = False, is_log_output = False, offset = 1):
    # Required pytorch >= 1.6.0
    #f0 = de_norm_mean_std(f0, hp.f0_mean, hp.f0_std)
    bins = torch.exp(torch.linspace(np.log(hparams.p_min), np.log(hparams.p_max), 255)).to(torch.device("cuda" if hparams.ngpu > 0 else "cpu"))
    if is_inference and is_log_output:
        f0 = torch.clamp(torch.round(f0.exp() - offset), min=0).long()
        
    p_quantize = bucketize(f0.to(torch.device("cuda" if hparams.ngpu > 0 else "cpu")), bins)
    #p_quantize = p_quantize - 1  # -1 to convert 1 to 256 --> 0 to 255
    return F.one_hot(p_quantize.long(), 256).float()


def bucketize(tensor, bucket_boundaries):
    result = torch.zeros_like(tensor, dtype=torch.int64)
    for boundary in bucket_boundaries:
        result += (tensor > boundary).int()
    return result



class Combined_model( MyFastSpeech):
    def __init__(self):
        super().__init__()
      #  print(self.Pitch.linear.weight)
        self.MyNet = Net() 

       
    def forward(self, text, padded ,len):

        padded.requires_grad_()

        #text.requires_grad_()  

        padded = self.MyNet.mypool(F.relu(self.MyNet.myconv1(padded)))
       
        padded = self.MyNet.mypool(F.relu(self.MyNet.myconv2(padded)))
           
        padded = torch.flatten(padded, 1)
        
        padded = F.relu(self.MyNet.myfc1(padded))
           
        padded = F.relu(self.MyNet.myfc2(padded))
            
        padded = self.MyNet.myfc3(padded)
        
        #print(text.shape)              
        text_lengths = torch.tensor([text.shape[0]])     
        mel_lengths = torch.tensor([len])
        text =text.T
        #text=text/100
        #text_size= text.shape[1]
        #if text_size > 60 :
        #     text_size = 60
             

        #text_padded = torch.zeros(1,60)
        #text_padded[0,:text_size] = text.clone()
        #text_padded = text_padded.to(device)
        #print(device)
        #text_padded.requires_grad_()

        hidden_states = self.Embedding(text.long()).transpose(0,1)

        #hidden_states = self.Embedding(text_padded).transpose(0,1)
       # print(text_padded.shape)
       # hidden_states = self.Embedding(text).transpose(0,1)
       # hidden_states = hidden_states.float() 
       # print('fff',hidden_states.shape)
        
  


    #  padded_padded = torch.zeros(hidden_states.shape[0],1,256)

       # padded= padded.unsqueeze(0)
       # padded_padded = padded_padded.to(device)
       # padded_padded[:,:,:] = padded.clone()
     #   padded_padded.requires_grad_()
     #   hidden_states +=padded_padded 

      #  hidden_states.requires_grad_()
     
        hidden_states += self.alpha1*(self.pe[:text.size(1)].unsqueeze(1))
         
       # text_mask = text.new_zeros(1,text.size(1)).to(torch.bool)
        #text_mask =  get_mask_from_lengths(text_lengths)
        text_mask = text.new_zeros(1,text.size(1)).to(torch.bool)
    #    print('MASK',text_mask)                
        text_mask = text_mask.to(device)
       # print(hidden_states)    


        for layer in self.Encoder:
             hidden_states, _ = layer(hidden_states,
                                      src_key_padding_mask=text_mask)
           

        durations = self.Duration(hidden_states.permute(1,2,0))
              
        

        alpha=1.0      
        hidden_states_expanded = self.LR(hidden_states, durations, alpha, inference=True)


        if(hidden_states_expanded.size(0))>5000:
                 print("CUTTED")
                 hidden_states_expanded = hidden_states_expanded[:5000,:,:]

              
        pitch = self.Pitch(hidden_states_expanded.permute(1,2,0))

        energy = self.Energy(hidden_states_expanded.permute(1,2,0))
  
        pitch_one_hot = pitch_to_one_hot(pitch)
             

        energy_one_hot = energy_to_one_hot(energy)
              
          
        hidden_states_expanded = hidden_states_expanded + pitch_one_hot.transpose(1,0) + energy_one_hot.transpose(1,0)       #check for all device attributes
        
        hidden_states_expanded += self.alpha2*self.pe[:hidden_states_expanded.size(0)].unsqueeze(1)
              
       # hidden_states_expanded += self.alpha2*self.pe[:len].unsqueeze(1)

              
        mel_mask =  text.new_zeros(1, hidden_states_expanded.size(0)).to(torch.bool)
      #  mel_mask[0,:len] = False
       # mel_mask =  get_mask_from_lengths(mel_lengths)
     #   print('MASK',text_mask)                
        mel_mask = mel_mask.to(device)
    
        for layer in self.Decoder:
             hidden_states_expanded, _ = layer(hidden_states_expanded,
                                               src_key_padding_mask=mel_mask)
             
      
        mel_out = self.Projection(hidden_states_expanded.transpose(0,1)).transpose(1,2)
              
        
             
        return (mel_out,pitch,energy,durations,mel_out.shape[2])


model3 = Combined_model()
 
for parameter in model3.parameters():
    parameter.requires_grad = False    
#model3.load_state_dict(torch.load("/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/weights/weights_right_26_april_50"))
for parameter in model3.Embedding.parameters():
    parameter.requires_grad_()
for parameter in model3.Projection.parameters():
    parameter.requires_grad_()

#for parameter in model3.MyNet.myconv1.parameters():
#    parameter.requires_grad_() 
#for parameter in model3.MyNet.myconv2.parameters():
#    parameter.requires_grad_()
#for parameter in model3.MyNet.mypool.parameters():
#    parameter.requires_grad_()
#for parameter in model3.MyNet.myfc1.parameters():
#    parameter.requires_grad_()
#for parameter in model3.MyNet.myfc2.parameters():
#    parameter.requires_grad_()
#for parameter in model3.MyNet.myfc3.parameters():
#    parameter.requires_grad_()
for parameter in model3.parameters():
    parameter.requires_grad = False

print(f'The model has {count_parameters(model3):,} trainable parameters')
model3.load_state_dict(torch.load("/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/weights/weights_right_27april_withoutconvs_10"))

model3.to(device)
