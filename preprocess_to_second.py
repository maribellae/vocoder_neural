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
