import sys
sys.path.append("..")
import librosa
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
from tts.tacotron2.speaker_embed import *
from tts.tacotron2.speaker_embed.encoder import inference as encoder
from pathlib import Path

def work(path):
    stft = TacotronSTFT(filter_length=hp.n_fft,
                        hop_length=hp.hop_length,
                        win_length=hp.win_length,
                        n_mel_channels=hp.n_mels,
                        sampling_rate=hp.sampling_rate,
                        mel_fmin=hp.fmin,
                        mel_fmax=hp.fmax)
    
    # wav_file loacation 
    wav_files = glob.glob(os.path.join(path, '**', '*.wav'), recursive=True)

    encoder_weights = Path('./tts/tacotron2/speaker_embed/encoder/pretrained.pt')
    encoder.load_model(encoder_weights)

    embedding_path = os.path.join(hp.data_path,'embeddings')
    os.makedirs(embedding_path, exist_ok=True)
    
    for wavpath in tqdm.tqdm(wav_files, desc='preprocess wav to embeds'):
        original_wav, sampling_rate = librosa.load(wavpath, sr=22050)
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate) 
        embed = encoder.embed_utterance(preprocessed_wav)
        id = os.path.basename(wavpath).split(".")[0]

        np.save('{}/{}.npy'.format(embedding_path,id), embed, allow_pickle=False)

         
work('./LJSpeech-1.1/wavs')