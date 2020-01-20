# -*- coding:utf-8 -*-
import os
import numpy as np
import yaml
import glob
import rwave
from tqdm import tqdm
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import Sequential

class CNN():
    def __init__(self, train=False, add_noise=False, model_path='./model/cnn.hdf5'):
        ## -----*----- コンストラクタ -----*----- ##
        self.audio_config = yaml.load(open('config/audio.yaml'), Loader=yaml.SafeLoader)
        self.classes      = yaml.load(open('config/class.yaml'), Loader=yaml.SafeLoader)
        self.hparams      = yaml.load(open('config/params.yaml'), Loader=yaml.SafeLoader)

        self.n_classes    = len(self.classes)
        self.batch_size = self.hparams['batch_size']
        self.epochs     = self.hparams['epochs']

        self.model_path = model_path

        # モデルのビルド
        self.__model = self.__build()

        if train:
            # 学習
            x, y = self.__features_extracter(add_noise)
            self.__train(x, y)
        else:
            # モデルの読み込み
            self.load_model()

    def __build(self):
        ## -----*----- NNを構築 -----*----- ##
        # モデルの定義
        model = Sequential([
            Conv2D(filters=11, kernel_size=(3, 3), input_shape=(12, 79, 1), strides=(1, 1),
                   padding='same', activation='relu'),
            Conv2D(filters=11, kernel_size=(3, 3), strides=(1, 1),
                   padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(self.hparams['dropout']),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                   padding='same', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                   padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(self.hparams['dropout']),
            Flatten(),
            Dense(units=512, activation='relu'),
            Dropout(self.hparams['dropout']),
            Dense(self.classes, activation='softmax')
        ])

        # モデルをコンパイル
        model.compile(
            optimizer=self.hparams['optimizer'],
            loss=self.hparams['loss'],
            metrics=["accuracy"]
        )

        return model

    def __train(self, x, y):
        ## -----*----- 学習 -----*-----##
        print("\n\nTrain")
        self.__model.fit(x, y,  epochs=self.hparams['epochs'], batch_size=self.hparams['batch_size'])

        # 最終の学習モデルを保存
        self.__model.save_weights(self.model_path)


    def __features_extracter(self, add_noise):
        ## -----*----- 特徴量を抽出 -----*----- ##
        x = [] # 入力
        y = [] # 出力ラベル

        # ノイズデータ
        if add_noise:
            noises = []
            noise_files = glob.glob('%s/*.wav' % self.audio_config['noise_build_path'])
            for file in noise_files:
                mfcc = rwave.to_mfcc(file, self.audio_config['wave_fs'])
                mfcc = rwave.nomalize(mfcc)
                mfcc = np.reshape(mfcc, (mfcc.hape[0], mfcc.shape[1], 1))
                noises.append(mfcc)

        index = 0
        for country in self.countries:
            print('Now : %s' % country)
            wav_files = glob.glob('%s/%s/*.wav' % (self.audio_config['speaker_build_path'], country))
            for file in tqdm(wav_files):
                # 音声をMFCCに変換
                mfcc = rwave.to_mfcc(file, self.audio_config['wave_fs'])
                mfcc = rwave.nomalize(mfcc)
                mfcc = np.reshape(mfcc, (mfcc.shape[0], mfcc.shape[1], 1))

                x.append(mfcc)
                y.append(index)

                # ノイズを加算
                if add_noise:
                    for noise in noises:
                        x.append(mfcc + noise)
                        y.append(index)

            index += 1

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.uint8)

        # ランダムに並べ替え
        perm = np.arange(len(x))
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]

        return x, y


    def load_model(self):
        ## -----*----- 学習済みモデルの読み込み -----*-----##
        # モデルが存在する場合，読み込む
        if os.path.exists(self.model_path):
            self.__model.load_weights(self.model_path)


    def predict(self, file):
        ## -----*----- 推論 -----*----- ##
        mfcc = rwave.to_mfcc(file, self.audio_config['wave_fs'])
        mfcc = rwave.nomalize(mfcc)
        mfcc = np.reshape(mfcc, (1, mfcc.shape[0], mfcc.shape[1], 1))
        score = self.__model.predict(mfcc, batch_size=None, verbose=0)
        pred = np.argmax(score)

        return self.classes[pred], score[0]
