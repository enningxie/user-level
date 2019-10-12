# coding=utf-8
from keras.layers import Lambda, Add, Multiply, Concatenate, Dot, Permute, Input, Embedding, BatchNormalization, \
    Bidirectional, LSTM, Dropout, Dense, GlobalMaxPool1D, GlobalAvgPool1D, CuDNNLSTM
from keras.models import Model
from keras.activations import softmax
from keras.optimizers import Adam
from src.dataloader import DataLoader
from src.esim_config import ESIMConfig


class ESIM(object):
    def __init__(self, model_config):
        self.configer = model_config
        self.dataloader = DataLoader()
        self.model = self.get_model()
        self.model_path = 'saved_models/esim_LCQMC_32_LSTM_0715_1036.h5'
        self._init_model()

    def _init_model(self):
        self.model.load_weights(self.model_path)
        p_pred, h_pred = self.dataloader.char_index(['买过了'], ['买好了'], self.configer.maxlen, pad_mode='pre')
        self.model.predict([p_pred, h_pred]).item()
        print('ESIM model loaded.')

    def _unchanged_shape(self, input_shape):
        "Function for Lambda layer"
        return input_shape

    def _substract(self, input_1, input_2):
        "Substract element-wise"
        neg_input_2 = Lambda(lambda x: -x, output_shape=self._unchanged_shape)(input_2)
        out_ = Add()([input_1, neg_input_2])
        return out_

    def _submult(self, input_1, input_2):
        "Get multiplication and subtraction then concatenate results"
        mult = Multiply()([input_1, input_2])
        sub = self._substract(input_1, input_2)
        out_ = Concatenate()([sub, mult])
        return out_

    def _apply_multiple(self, input_, layers):
        "Apply layers to input then concatenate result"
        if not len(layers) > 1:
            raise ValueError('Layers list should contain more than 1 layer')
        else:
            agg_ = []
            for layer in layers:
                agg_.append(layer(input_))
            out_ = Concatenate()(agg_)
        return out_

    def _soft_attention_alignment(self, input_1, input_2):
        "Align text representation with neural soft attention"
        attention = Dot(axes=-1)([input_1, input_2])
        w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                         output_shape=self._unchanged_shape)(attention)
        w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                         output_shape=self._unchanged_shape)(attention))
        in1_aligned = Dot(axes=1)([w_att_1, input_1])
        in2_aligned = Dot(axes=1)([w_att_2, input_2])
        return in1_aligned, in2_aligned

    def get_model(self):
        q1 = Input(name='q1', shape=(self.configer.maxlen,))
        q2 = Input(name='q2', shape=(self.configer.maxlen,))

        embedding_op = Embedding(self.configer.max_features, self.configer.embedding_size,
                                 input_length=self.configer.maxlen)
        # embedding_q2 = Embedding(self.configer.max_features, self.configer.embedding_size,
        #                          input_length=self.configer.maxlen)
        bn = BatchNormalization()

        # embedding + batch normalization
        q1_embed = bn(embedding_op(q1))
        q2_embed = bn(embedding_op(q2))

        # todo 一个还是两个
        # bi-lstm
        encode = Bidirectional(LSTM(self.configer.lstm_dim, return_sequences=True))
        q1_encoded = encode(q1_embed)
        q2_encoded = encode(q2_embed)

        # Attention
        q1_aligned, q2_aligned = self._soft_attention_alignment(q1_encoded, q2_encoded)

        # Compose
        q1_combined = Concatenate()([q1_encoded, q2_aligned, self._submult(q1_encoded, q2_aligned)])
        q2_combined = Concatenate()([q2_encoded, q1_aligned, self._submult(q2_encoded, q1_aligned)])

        # todo
        compose = Bidirectional(LSTM(self.configer.lstm_dim, return_sequences=True))
        q1_compare = compose(q1_combined)
        q2_compare = compose(q2_combined)

        # Aggregate
        q1_rep = self._apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        q2_rep = self._apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        # Classifier
        merged = Concatenate()([q1_rep, q2_rep])

        dense = BatchNormalization()(merged)
        dense = Dense(512, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(self.configer.dropout_rate)(dense)
        dense = Dense(self.configer.dense_dim, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(self.configer.dropout_rate)(dense)
        out_ = Dense(1, activation='sigmoid')(dense)

        model = Model(inputs=[q1, q2], outputs=out_)
        model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def predict(self, sent1, sent2):
        p_pred, h_pred = self.dataloader.char_index(sent1, sent2, self.configer.maxlen)
        print(self.model.predict([p_pred, h_pred]).item())


if __name__ == '__main__':
    esim = ESIM(ESIMConfig())
    esim.predict('想问您有没有兴趣来看看啊？', '您要不要来看一下呀？')
