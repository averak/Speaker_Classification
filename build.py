# -*- coding: utf-8 -*-
'''
教師データをビルドするプログラム
　- config/audio.yamlに各ビルド設定を記載
　- それぞれ時間(sec)・サンプリングレート(fs)がバラバラの音源のsec・fsを統一
　- ビルド後の音源を保存
'''

import glob
import yaml
from tqdm import tqdm
import rwave


# 設定ファイルを読み込み
audio_config = yaml.load(open('config/audio.yaml'), Loader=yaml.SafeLoader)


## ========== 音源をビルド ==============================
print('Build Audio Data')
# 話者の音源一覧
wav_files = glob.glob('%s/*/*.wav' % audio_config['speaker_raw_path'])
# それぞれについて処理を行う
for file in tqdm(wav_files):
    # 変換後のファイルPATH
    out_filepath = file.replace(audio_config['speaker_raw_path'], audio_config['speaker_build_path'])
    # 元音源読み込み
    wav, fs = rwave.read_wave(file)
    # サンプリングレートを調整（8kHz）
    ds_wav, ds_fs = rwave.convert_fs(wav, fs, audio_config['wave_fs'])
    # 音源の秒数を調整
    sec_wav, sec_fs = rwave.convert_sec(wav, ds_fs, audio_config['wave_sec'])
    # ビルド後の音源を書き込み
    rwave.write_wave(out_filepath, sec_wav, sec_fs)


## ========== ノイズ音源をビルド ==============================
print('Build Noise Data')
# ノイズ音源一覧
wav_files = glob.glob('%s/*.wav' % audio_config['noise_raw_path'])
# それぞれについて処理を行う
for file in tqdm(wav_files):
    # 変換後のファイルPATH
    out_filepath = file.replace(audio_config['noise_raw_path'], audio_config['noise_build_path'])
    # 元音源読み込み
    wav, fs = rwave.read_wave(file)
    # サンプリンレートを調整グ（96kHz -> 8kHz）
    ds_wav, ds_fs = rwave.convert_fs(wav, fs, audio_config['wave_fs'])
    # 音源の秒数を調整（圧縮・拡張）
    sec_wav, sec_fs = rwave.convert_sec(wav, ds_fs, audio_config['wave_sec'])
    # ビルド後の音源を書き込み
    rwave.write_wave(out_filepath, sec_wav, sec_fs)

