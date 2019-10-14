# coding=utf-8

import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs
from keras.layers import *
from keras.models import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


class BertRunner(object):
    def __init__(self):
        self.maxlen = 32
        # todo move pre-trained model weihts to current repo path
        self.config_path = '/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
        self.checkpoint_path = '/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
        self.dict_path = '/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'
        self.restore_model_path = 'saved_models/bert_0801_1405.h5'
        self.token_dict = self._read_token_dict()
        self.tokenizer = OurTokenizer(self.token_dict)
        self.model = self._get_model()
        self._init_model()

    def _read_token_dict(self):
        token_dict = {}

        with codecs.open(self.dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)

        return token_dict

    def _seq_padding(self, X, padding=0):
        L = [len(x) for x in X]
        ML = max(L)
        padded_sent = np.array([
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])
        return padded_sent

    def _prepare_data(self, data_path):
        data = pd.read_csv(data_path)
        p = data['sentence1'].values[:]
        h = data['sentence2'].values[:]
        label = data['label'].values[:]
        X1, X2, Y = [], [], []
        for tmp_p, tmp_h, tmp_label in zip(p, h, label):
            x1, x2 = self.tokenizer.encode(first=tmp_p[:self.maxlen], second=tmp_h[:self.maxlen])
            X1.append(x1)
            X2.append(x2)
            Y.append([tmp_label])

        X1 = self._seq_padding(X1)
        X2 = self._seq_padding(X2)
        Y = self._seq_padding(Y)
        return X1, X2, Y

    def _get_model(self):
        # todo seq_len
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path, seq_len=None)

        for l in bert_model.layers:
            l.trainable = True

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)
        p = Dense(1, activation='sigmoid')(x)

        model = Model([x1_in, x2_in], p)
        return model

    def _init_model(self):
        sentence1 = ['好的']
        sentence2 = ['可以']
        self.model.load_weights(self.restore_model_path)
        X1, X2 = self._data_preprocesser(sentence1, sentence2)
        self.model.predict([X1, X2])
        print('Bert model loaded.')

    def _data_preprocesser(self, sentence1, sentence2, mode=None):
        X1, X2 = [], []
        if mode == 'predict':
            for tmp_sent1, tmp_sent2 in zip(sentence1, sentence2):
                x1, x2 = self.tokenizer.encode(first=tmp_sent1[-self.maxlen:], second=tmp_sent2[-self.maxlen:])
                X1.append(x1)
                X2.append(x2)
            X1 = self._seq_padding(X1)
            X2 = self._seq_padding(X2)
            return X1, X2

        for tmp_sent1, tmp_sent2 in zip(sentence1, sentence2):
            x1, x2 = self.tokenizer.encode(first=tmp_sent1[:self.maxlen], second=tmp_sent2[:self.maxlen])
            X1.append(x1)
            X2.append(x2)
        X1 = self._seq_padding(X1)
        X2 = self._seq_padding(X2)
        return X1, X2

    def predict(self, sentence1, sentence2):
        # self.model.load_weights(self.restore_model_path)
        X1, X2 = self._data_preprocesser(sentence1, sentence2, mode='predict')
        y_pred = self.model.predict([X1, X2], batch_size=1024)
        return y_pred


if __name__ == '__main__':
    bert_runner = BertRunner()
    print(bert_runner.predict(['哦好的'], ['不推荐给你']))