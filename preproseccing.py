import tqdm
import glob
import rwave
from conf import RATE, DATA_DIR, MFCC_DIR


def resampling(files):
    for wf in tqdm.tqdm(files):
        wav, fs = rwave.read_wave(wf)

        if fs != RATE:
            wav, fs = rwave.convert_fs(wav, fs, RATE)
            rwave.write_wave(wf, wav, RATE)


def exclude_silence(files):
    from pydub.silence import split_on_silence
    from pydub import AudioSegment

    for wf in tqdm.tqdm(files):
        wav = AudioSegment.from_file(wf, format='wav')
        chunks = split_on_silence(
            wav,
            min_silence_len=500,
            silence_thresh=-40,
            keep_silence=150
        )
        # FIXME: 分割後の音声の扱い


def set_dir(dir_name):
    import os
    import shutil

    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


print('リサンプリング...')
files = glob.glob('%s/*/*.wav' % DATA_DIR)
resampling(files)

print('無音区間を削除...')
files = glob.glob('%s/*/*.wav' % DATA_DIR)
exclude_silence(files)
