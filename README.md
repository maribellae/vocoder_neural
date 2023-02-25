# vocoder_neural

0) `git clone https://github.com/carankt/FastSpeech2

   cd FastSpeech2
   
   git clone https://github.com/maribellae/vocoder_neural`
   
1) Download requirements.txt 

`pip install -r vocoder_neural/requirements.txt`


2)  `python vocoder_neural/installs.py`

3)  `python vocoder_neural/preprocess_to_second.py`

4)   Download [these](https://drive.google.com/file/d/12jW1KivfEjv4YBs6gAZVdVWJ-muZv6CQ/view) checkpoints and place them in a FastSpeech2/ folder ( FastSpeech2/checkpoint_80000 )

5)  `python second_model.py`

6)  `python train.py`
