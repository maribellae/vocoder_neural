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
    ids = lengths.new_tensor(torch.arange(0, max_len)).to(device) #torch.arange(0, max_len)       lengths.new_tensor(torch.arange(0, max_len)), giving some warning
    mask = (lengths.unsqueeze(1) <= ids).to(torch.bool).to(device)
    return mask  
    


      
class MyFastSpeech(Model):
    def __init__(self):
        super(MyFastSpeech,self).__init__(hparams)
        self.hp = hp
        self.load_state_dict(state_dict)
        self.mylin = nn.Linear(256,256)
 
        

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

       
    def outputs(self, text, padded,durations,text_lengths, mel_lengths,embeds):
        #mel_lengths = torch.tensor([padded.shape[2]])

         
        B, L, T = text.size(0), text.size(1), mel_lengths.max().item()
        #print("Batch",B,"\nTime Length",L,"\nMax Number of Frames",T)
 
              
        hidden_states = self.Embedding(text).transpose(0,1)


        hidden_states =  hidden_states +  self.alpha1*(self.pe[:L].unsqueeze(1))
        hidden_states = self.dropout(hidden_states) 
        text_mask = get_mask_from_lengths(text_lengths.to(device)).to(device)

        mel_mask = get_mask_from_lengths(mel_lengths.to(device)).to(device)


        for layer in self.Encoder:
             hidden_states, _ = layer(hidden_states,
                                      src_key_padding_mask=text_mask)

        embeds = self.mylin(embeds)
        hidden_states[:,:,:] = hidden_states[:,:,:] + embeds
        
        durations_out = self.Duration(hidden_states.permute(1,2,0))
              
        
        hidden_states_expanded = self.LR(hidden_states, durations)

              
        pitch = self.Pitch(hidden_states_expanded.permute(1,2,0))

        energy = self.Energy(hidden_states_expanded.permute(1,2,0))
  
        pitch_one_hot = pitch_to_one_hot(pitch)
             

        energy_one_hot = energy_to_one_hot(energy)
              
          
        hidden_states_expanded = hidden_states_expanded + pitch_one_hot.transpose(1,0) + energy_one_hot.transpose(1,0)       #check for all device attributes
        
        hidden_states_expanded += self.alpha2*self.pe[:T].unsqueeze(1)
        hidden_states_expanded = self.dropout(hidden_states_expanded)      
     
              
    
        for layer in self.Decoder:
             hidden_states_expanded, _ = layer(hidden_states_expanded,
                                               src_key_padding_mask=mel_mask)
             
      
        mel_out = self.Projection(hidden_states_expanded.transpose(0,1)).transpose(1,2)
        #print(mel_out.shape,"Output Mel Shape")       
        
             
        return (mel_out,durations_out, durations, pitch,energy)
    def forward(self, text,  melspec, durations, text_lengths, mel_lengths, pitch, energy,embeds , criterion):
                
	#print(text.device, melspec.device, durations.device, text_lengths.device, mel_lengths.device, pitch.device, energy.device)
	### Size ###
        text = text[:,:text_lengths.max().item()]                        #no need for this maybe [B, L]
        melspec = melspec[:,:,:mel_lengths.max().item()]                #no need for this maybe [B, 80, T]

        mel_out, duration_out, durations, pitch_out, energy_out = self.outputs(text.cuda(), melspec.cuda(), durations.cuda(), text_lengths.cuda(), mel_lengths.cuda(), embeds.cuda())
        mel_loss, duration_loss, pitch_loss, energy_loss = criterion((mel_out, duration_out, pitch_out, energy_out),
                                            (melspec, durations, pitch, energy),
                                            (text_lengths.to(device), mel_lengths.to(device)))
        
        return mel_loss, duration_loss, pitch_loss, energy_loss



    def LR(self, hidden_states, durations, alpha=1.0, inference=False):
        L, B, D = hidden_states.size()
        durations = torch.round(durations*alpha).to(torch.long)
        if inference:
            durations[durations<=0]=1
        T=int(torch.sum(durations, dim=-1).max().item()) 
        #print(T, "Number of mel frames")
        expanded = hidden_states.new_zeros(T, B, D)
        
        #print(hidden_states)
        for i, d in enumerate(durations):

            mel_len = torch.sum(d).item()
            #print(i,d,mel_len) 
            expanded[:mel_len, i] = torch.repeat_interleave(hidden_states[:,i],
                                                            d,
                                                            dim=0)
        #print(expanded, "Shape of expanded after LR")
        return (expanded)





model3 = Combined_model()



#model3.load_state_dict(torch.load("/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/weights/weights_11_may_15"))
 

for parameter in model3.parameters():
    parameter.requires_grad = False    
#model3.load_state_dict(torch.load("/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/weights/weights_right_26_april_50"))


#model3.load_state_dict(torch.load("/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/weights/weights_19may_NEW_50"))


for parameter in model3.Projection.parameters():
    parameter.requires_grad_()

for parameter in model3.Pitch.parameters():
    parameter.requires_grad_()
for parameter in model3.Duration.parameters():
    parameter.requires_grad_()
for parameter in model3.Energy.parameters():
    parameter.requires_grad_()
for parameter in model3.Decoder.parameters():
    parameter.requires_grad_()

for parameter in model3.mylin.parameters():
    parameter.requires_grad_()


#for parameter in model3.parameters():
#    parameter.requires_grad = False

print(f'The model has {count_parameters(model3):,} trainable parameters')
#model3.load_state_dict(torch.load("/home/common/dorohin.sv/makarova/FastSpeech2/vocoder_neural/weights/weights_19may_NEW_opt4_150"))

model3.to(device)