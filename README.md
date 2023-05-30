# vocoder_neural

1) `git clone https://github.com/carankt/FastSpeech2 `

   ` cd FastSpeech2 `
   
   ` git clone https://github.com/maribellae/vocoder_neural `
   
2) Download requirements.txt 

`pip install -r vocoder_neural/requirements.txt`


3)   Download [these](https://drive.google.com/file/d/12jW1KivfEjv4YBs6gAZVdVWJ-muZv6CQ/view) checkpoints and place them into vocoder_neural/ folder 

4)   ` cd FastSpeech2 `


5)  `python download_lj.py`


6)  `python audio_preprocess.py`


7)  `git clone https://github.com/tapsoft/tts.git `


8)   `python audio_preprocess_embeddings.py`


9)   `python train_git6.py`
