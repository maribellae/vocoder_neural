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




MINILIBRI_TEST_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
download_file(MINILIBRI_TEST_URL, 'test-clean.tar.gz')
shutil.unpack_archive( 'test-clean.tar.gz', '.')
from speechbrain.pretrained import EncoderDecoderASR



device = "cuda" if torch.cuda.is_available() else "cpu"

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-transformerlm-librispeech",  run_opts={"device":"cuda"},)

class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, split="test-clean", device=device):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        wav, sample_rate, _, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
       
       # if len(wav.shape) == 2:
        #    wav = wav[:, 0]
        #print(wav.shape)
      #  wav=pad_or_trim(wav.flatten()).to(device)
      #  print(wav.shape)

        


        return(wav.flatten())
        
        
        
dataset = LibriSpeech("test-clean")
loader = torch.utils.data.DataLoader(dataset, batch_size=1)        

datas = []



stft = TacotronSTFT(filter_length=hp.n_fft,
                     hop_length=hp.hop_length,
                     win_length=hp.win_length,
                        n_mel_channels=hp.n_mels,
                        sampling_rate=16000,
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
    
from tqdm.notebook import tqdm

i=0
for audios in tqdm(loader):
    i+=1
    with torch.no_grad():
      # wav=pad_or_trim(audios.flatten()).to(device)
       res=asr_model.encode_batch(pad_or_trim(audios).to(device),rel_length).to(device)
       predicted_tokens, _ = asr_model.mods.decoder(res, rel_length)
   # print(audios.shape)
    audios = audios[0].cpu().numpy()

    if audios.dtype == np.int16:
        audios= audios / 32768.0
    elif audios.dtype == np.int32:
        audios= audios / 2147483648.0
    elif audios.dtype == np.uint8:
        audios = (audios - 128) / 128.0

    audios = audios.astype(np.float32)

    p = pitch(audios)  # [T, ] T = Number of frames 
    print(p.shape)
    audios = torch.from_numpy(audios).unsqueeze(0)
    mel, mag = stft.mel_spectrogram(audios) # mel [1, 80, T]  mag [1, num_mag, T]
    mel = mel.squeeze(0) # [num_mel, T]
    mag = mag.squeeze(0) # [num_mag, T]
    e = torch.norm(mag, dim=0) # [T, ]
    p = p[:mel.shape[1]]
    p = np.array(p, dtype='float32')
    datas.append([predicted_tokens[0],mel,p,e])    
                        
                        
def PaddingData(batch):
        input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor([len(x[0]) for x in batch]),dim=0, descending=True)
        max_input_len = input_lengths[0]
        text_padded = torch.zeros(len(batch), max_input_len, dtype=torch.int64)
        for i in range(len(ids_sorted_decreasing)):
            text = torch.FloatTensor(batch[ids_sorted_decreasing[i]][0])
            text_padded[i, :text.shape[0]] = torch.tensor(text) # Tensor (batch size, max_input_len )
        
        
        #Mel 2-D padding
        num_mels = batch[0][1].shape[0]     #num of mel filters
        max_target_len = max([x[1].shape[1] for x in batch])   #max mel frames in the batch
        mel_padded = torch.zeros(len(batch), num_mels, max_target_len, dtype=torch.float32) #(batch size, num of filter banks, max_target_len)
        output_lengths = torch.LongTensor(len(batch))  
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]                       
            mel_padded[i, :, :mel.shape[1]] = torch.tensor(mel)             
            output_lengths[i] = mel.shape[1]    #retain the original number of frames in mel
        mel_pad = mel_padded.view(-1,num_mels, max_target_len)
        
        #alignment padding
   #     input_duration_lengths, _ = torch.sort(torch.LongTensor([len(x[2]) for x in batch]),dim=0, descending=True)
    #    max_duration_input_len = input_duration_lengths[0]   #max duration
     #   duration_padded = torch.zeros(len(batch), max_duration_input_len, dtype=torch.float32)
  #      for i in range(len(ids_sorted_decreasing)):
   #         duration = batch[ids_sorted_decreasing[i]][2]
    #        duration_padded[i, :duration.shape[0]] = torch.tensor(duration)  # Tensor (batch size, max duration)
            
        #pitch padding
        pitch_padded = torch.zeros(len(batch), max_target_len, dtype=torch.float32)
        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]][2]
            pitch_padded[i, :pitch.shape[0]] = torch.tensor(pitch)  # Tensor (batch size, max frames or max_target_len)
            
        #energy padding
        energy_padded = torch.zeros(len(batch), max_target_len, dtype=torch.float32)
        for i in range(len(ids_sorted_decreasing)):
            energy = batch[ids_sorted_decreasing[i]][3]
            energy_padded[i, :energy.shape[0]] = torch.tensor(energy) # Tensor (batch size, max frames or max_target_len) 
        
        assert(energy_padded.shape == pitch_padded.shape)
        
        return text_padded, input_lengths, mel_pad, output_lengths, pitch_padded, energy_padded                        
text_padded, input_lengths, mel_pad, output_lengths, pitch_padded, energy_padded  = PaddingData(datas)                        
