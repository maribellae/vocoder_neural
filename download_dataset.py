import shutil
MINILIBRI_TEST_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
download_file(MINILIBRI_TEST_URL, 'test-clean.tar.gz')
shutil.unpack_archive( 'test-clean.tar.gz', '.')
