
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
#from utils.writer import get_writer
from utils.plot_image import *


from speechbrain.pretrained import EncoderDecoderASR
device = "cuda" if torch.cuda.is_available() else "cpu"

#FIRST MODEL 
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", 
                                           savedir="pretrained_models/speechbrain/asr-crdnn-rnnlm-librispeech",
                                           run_opts={"device":device},)

#DATASET LOADER
class LJSpeech(torch.utils.data.Dataset):

    def __init__(self, device=device):
        self.dataset = torchaudio.datasets.LJSPEECH(
            # root=os.path.expanduser("~/.cache"),
            root = './' ,
            #url='https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
            #  download=True,
            download = False,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        wav, sample_rate, _, _ = self.dataset[item]
        assert sample_rate == 22050
       

        return(wav.flatten())
        
        
        
dataset = LJSpeech()



#import speechbrain
#from speechbrain.utils.data_utils import download_file
#import shutil
#MINILIBRI_TEST_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
#download_file(MINILIBRI_TEST_URL, 'LJSpeech-1.1.tar.bz2')
#shutil.unpack_archive( 'LJSpeech-1.1.tar.bz2', '.')



loader = torch.utils.data.DataLoader(dataset, batch_size=1)        

datas = []


stft = TacotronSTFT(filter_length=hp.n_fft,
                     hop_length=hp.hop_length,
                     win_length=hp.win_length,
                        n_mel_channels=hp.n_mels,
                        sampling_rate=22050,
                        mel_fmin=hp.fmin,
                        mel_fmax=hp.fmax)
                        
rel_length = torch.tensor([1.0 ]).to(device)

N_SAMPLES= 480000
def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array
    
from tqdm import tqdm


g2p = G2p()
def prepare_input(text):
    text = punctuation_removers(text)
    phonemes = g2p(text)
    #print(phonemes)
    for i in range(0,len(phonemes)):
        if phonemes[i] == " ":
            phonemes[i]= "pau"

    sequence = phonemes_to_sequence(phonemes)
    return sequence

list_phonemes=[]
list_durations=[]

with open("../MFA_filelist/" + "alignment.txt", encoding='utf-8') as f:      #add all 13100 examples to filelist.txt 
     for lines in f:
         content = lines.split('|')
         id = content[4].split()[0].split('.')[0]
         if os.path.exists(os.path.join('./LJSpeech-1.1/wavs/', id + '.wav')):
             phoneme = content[3]
             duration = content[2]
             duration = duration.split()
             dur = np.array(duration, dtype = 'float32')   
             symbol_sequence = phonemes_to_sequence(phoneme,len(dur))  
             list_phonemes.append(symbol_sequence)
             list_durations.append(dur)

i=0
for audios in tqdm(loader):
    i+=1
  #  with torch.no_grad():
     
  #     res=asr_model.encode_batch(pad_or_trim(audios).to(device),rel_length).to(device)

  #     with torch.no_grad():
  #      predicted_tokens, _ = asr_model.mods.decoder(res, rel_length)
  #      predicted_words =[asr_model.tokenizer.decode_ids(token_seq)
  #              for token_seq in predicted_tokens
  #          ]
    
    audios = audios[0].cpu().numpy()

    if audios.dtype == np.int16:
        audios= audios / 32768.0
    elif audios.dtype == np.int32:
        audios= audios / 2147483648.0
    elif audios.dtype == np.uint8:
        audios = (audios - 128) / 128.0

    audios = audios.astype(np.float32)

    p = pitch(audios)  # [T, ] T = Number of frames 

    audios = torch.from_numpy(audios).unsqueeze(0)
    mel, mag = stft.mel_spectrogram(audios)         # mel [1, 80, T]  mag [1, num_mag, T]
    mel = mel.squeeze(0)                            # [num_mel, T]
    mag = mag.squeeze(0)                            # [num_mag, T]
    e = torch.norm(mag, dim=0)                      # [T, ]
    p = p[:mel.shape[1]]
    p = np.array(p, dtype='float32')
    #words = prepare_input(predicted_words[0])
#   print(predicted_words[0])
#   print(i)
    datas.append([list_phonemes[i-1],mel,p,e,list_durations[i-1]]) 

    if (i==5):
       break  
   