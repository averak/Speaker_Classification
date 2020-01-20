# -*- coding: utf-8 -*-
import time, os
import glob
import yaml
from lib.record import Recording

# ===== 新規話者の登録 ===============

audio_config = yaml.load(open('config/audio.yaml'), Loader=yaml.SafeLoader)
dir = audio_config['noise_raw_path']

# ディレクトリ作成
os.makedirs(dir, exist_ok=True)

record = Recording()
os.system('clear')
print('*** ENTERを押して録音開始・終了 ***')

mode = 0  # 0：録音開始，1：録音終了
cnt = len(glob.glob('{0}/*.wav'.format(dir))) + 1

while True:
    key = input()

    if mode == 0:
        # 録音開始
        print("===== {0} START ===============".format(cnt))
        record.file = '{0}/{1}.wav'.format(dir, cnt)
        record.record_start.set()
        mode = 1

    else:
        # 録音終了
        print("===== END ===============")
        record.record_start.clear()
        mode = 0
        cnt += 1
