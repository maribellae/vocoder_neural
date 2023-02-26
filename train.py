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
import IPython.display as ipd
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
from utils.writer import get_writer
from utils.plot_image import *
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from second_model import model3,net
model3.to(device)
net.to(device)

from preprocess_to_second import text_padded, input_lengths, mel_pad, output_lengths, pitch_padded, energy_padded


optimizer = torch.optim.Adam(  list(model3.parameters()) + list(net.parameters())  , lr=3e-4)

criterion = torch.nn.MSELoss().to(device)


class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model3,net, device, data_len ,vector_w, mels,pitches, energies,  criterion, optimizer,iter_meter):
    full_loss=0
    # элементы у меня по убыванию там
    r = list(range(data_len))
    random.shuffle(r)
    for i in r:
        #audios , words, mels,durations = waveforms[0] ,vector_w, vector_m[0]  ,input_lengths
        word, mel,pitch, energy  = vector_w[i].to(torch.float32), mels[i].to(torch.float32) ,pitches[i].to(torch.float32), energies[i].to(torch.float32)  
       # print(audios , words, mels,durations)
        word, mel,pitch, energy = word.to(device),mel[None,:,:].to(device),pitch[None,:].to(device) , energy[None,:].to(device)

        optimizer.zero_grad()
      #  words = words[-1,-1,:]
        word = word[:,None]
      
        word = word.to(device)
       

        mel_info = net(mel)
       # mel_info.requires_grad_()

        mel_output, pitch_output,energy_output,length = model3(word, mel_info)
        #mel_output.requires_grad_()
      #  pitch_output.requires_grad_()
       # energy_output.requires_grad_()
       
        loss = criterion(mel_output,  mel[:,:,:length]) +  0.5*criterion(pitch_output,  pitch[:,:length] ) + 0.5*criterion(energy_output, energy[:,:length])
        loss.requires_grad = True
        # +  0.5*criterion(pitch_output,  pitch[:,:length] ) + 0.5*criterion(energy_output, energy[:,:length])
        
        loss.backward()
        #print(loss.grad)
      #  experiment.log_metric('loss', loss.item(), step=iter_meter.get())
       # experiment.log_metric('learning_rate', scheduler.get_lr(), step=iter_meter.get())

        optimizer.step()
         #   scheduler.step()
        iter_meter.step()
        #if i % 100 == 0 or i == data_len:
         #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i ,  loss.item()))
        print("I IS ",  i , "LOSS IS " , loss.item())
        full_loss +=loss.item() 
        
     #   out_audio = audio.cpu().float().numpy().astype(np.float32, order='C')
        
    return(mel_output,full_loss)    




epochs = 10
iter_meter = IterMeter()
for epoch in range(1, epochs + 1):
      #trainn(model3, device,len(train_loader.dataset), waveforms ,vector_w, vector_m, criterion, optimizer,  iter_meter)
      last_mel , full_loss  =train(model3, net, device,100,text_padded, mel_pad, pitch_padded,energy_padded, criterion, optimizer,  iter_meter)
