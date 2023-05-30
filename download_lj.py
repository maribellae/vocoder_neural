import speechbrain
from speechbrain.utils.data_utils import download_file
import shutil
MINILIBRI_TEST_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
download_file(MINILIBRI_TEST_URL, 'LJSpeech-1.1.tar.bz2')
shutil.unpack_archive( 'LJSpeech-1.1.tar.bz2', '.')
