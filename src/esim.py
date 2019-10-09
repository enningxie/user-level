# coding=utf-8
import tensorflow as tf
from src.dataloader import DataLoader


def unchanged_shape(input_shape):
    """Function for Lambda layer"""
    return input_shape


# attention layer
class Attention(tf.keras.layers.Layer):
    """Align text representation with neural soft attention"""

    def __init__(self):
        super(Attention, self).__init__()

    def call(self, input_1, input_2):
        attention = tf.keras.layers.Dot(axes=-1)([input_1, input_2])
        w_att_1 = tf.keras.layers.Lambda(lambda x: tf.keras.activations.softmax(x, axis=1),
                                         output_shape=unchanged_shape)(attention)
        w_att_2 = tf.keras.layers.Permute((2, 1))(
            tf.keras.layers.Lambda(lambda x: tf.keras.activations.softmax(x, axis=2),
                                   output_shape=unchanged_shape)(attention))
        in1_aligned = tf.keras.layers.Dot(axes=1)([w_att_1, input_1])
        in2_aligned = tf.keras.layers.Dot(axes=1)([w_att_2, input_2])
        return in1_aligned, in2_aligned


class Submult(tf.keras.layers.Layer):
    """Get multiplication and subtraction then concatenate results"""

    def __init__(self):
        super(Submult, self).__init__()

    def call(self, input_1, input_2):
        mult = tf.keras.layers.Multiply()([input_1, input_2])
        sub = tf.keras.layers.Subtract()([input_1, input_2])
        return tf.keras.layers.Concatenate()([sub, mult])


class ApplyMultiple(tf.keras.layers.Layer):
    """Apply layers to input then concatenate result"""

    def __init__(self, layers):
        super(ApplyMultiple, self).__init__()
        self.layers = layers
        if not len(self.layers) > 1:
            raise ValueError('Layers list should contain more than 1 layer')

    def call(self, inputs):
        agg_ = []
        for layer in self.layers:
            agg_.append(layer(inputs))
        return tf.keras.layers.Concatenate()(agg_)


# esim model.
class ESIM(object):
    def __init__(self, config):
        # hyperparams
        self.max_len = config['max_len']
        self.max_features = config['max_features']
        self.embedding_size = config['embedding_size']
        self.lstm_hidden_size = config['lstm_hidden_size']
        self.dense_hidden_sizes = config['dense_hidden_sizes']
        self.dropout_rate = config['dropout_rate']
        # create model
        self.model = self.create_model()
        # train setup
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.train_data_path = config['train_data_path']
        self.dev_data_path = config['dev_data_path']
        self.test_data_path = config['test_data_path']

    def get_dataset(self, data_path, is_training=False, return_steps=False):
        tmp_dataloader = DataLoader(data_path, self.max_len)
        x_data, y_data = tmp_dataloader.load_char_data()
        tmp_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        if is_training:
            tmp_dataset = tmp_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        else:
            tmp_dataset = tmp_dataset.batch(self.batch_size)
        if return_steps:
            if x_data[0].shape[0] % self.batch_size == 0:
                tmp_steps = x_data[0].shape[0] // self.batch_size
            else:
                tmp_steps = x_data[0].shape[0] // self.batch_size + 1
            return tmp_dataset, tmp_steps
        else:
            return tmp_dataset

    def create_model(self):
        # input layer
        sentence_1 = tf.keras.layers.Input(shape=(self.max_len,))
        sentence_2 = tf.keras.layers.Input(shape=(self.max_len,))

        # embedding layer
        embedding_layer = tf.keras.layers.Embedding(input_dim=self.max_features, output_dim=self.embedding_size,
                                                    input_length=self.max_len)

        # batch normalization layer
        bn_layer = tf.keras.layers.BatchNormalization()

        # embedding + batch normalization, share
        sent1_embed = bn_layer(embedding_layer(sentence_1))
        sent2_embed = bn_layer(embedding_layer(sentence_2))

        # bi-lstm layer, encoder
        bilstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_hidden_size, return_sequences=True))

        sent1_encoded = bilstm_layer(sent1_embed)
        sent2_encoded = bilstm_layer(sent2_embed)

        # attention layer
        sent1_aligned, sent2_aligned = Attention()(sent1_encoded, sent2_encoded)

        # Compose op
        sent1_combined = tf.keras.layers.Concatenate()(
            [sent1_encoded, sent2_aligned, Submult()(sent1_encoded, sent2_aligned)])
        sent2_combined = tf.keras.layers.Concatenate()(
            [sent2_encoded, sent1_aligned, Submult()(sent2_encoded, sent1_aligned)])

        # bi-lstm layer
        bilstm_layer_ = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.lstm_hidden_size, return_sequences=True))
        sent1_composed = bilstm_layer_(sent1_combined)
        sent2_composed = bilstm_layer_(sent2_combined)

        # Aggregate
        sent1_aggregated = ApplyMultiple([tf.keras.layers.GlobalAvgPool1D(), tf.keras.layers.GlobalMaxPool1D()])(
            sent1_composed)
        sent2_aggregated = ApplyMultiple([tf.keras.layers.GlobalAvgPool1D(), tf.keras.layers.GlobalMaxPool1D()])(
            sent2_composed)

        # Classifier
        merged = tf.keras.layers.Concatenate()([sent1_aggregated, sent2_aggregated])
        x = tf.keras.layers.BatchNormalization()(merged)
        x = tf.keras.layers.Dense(self.dense_hidden_sizes[0], activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.Dense(self.dense_hidden_sizes[1], activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        output_ = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(x)

        model = tf.keras.Model(inputs=[sentence_1, sentence_2], outputs=output_)
        return model

    def train_(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )

        # prepare the training dataset.
        train_dataset, train_steps = self.get_dataset(self.train_data_path, is_training=True, return_steps=True)

        # Prepare the validation dataset.
        val_dataset, val_steps = self.get_dataset(self.dev_data_path, return_steps=True)

        history = self.model.fit(
            train_dataset.repeat(),
            epochs=self.epochs,
            validation_data=val_dataset.repeat(),
            steps_per_epoch=train_steps,
            validation_steps=val_steps
        )

    def train(self):
        # instantiate an optimizer to train the model.
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # instantiate a loss function.
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        # prepare the metrics.
        train_acc_metric = tf.keras.metrics.BinaryAccuracy()
        val_acc_metric = tf.keras.metrics.BinaryAccuracy()

        # prepare the training dataset.
        train_dataset = self.get_dataset(self.train_data_path, is_training=True)

        # Prepare the validation dataset.
        val_dataset = self.get_dataset(self.dev_data_path)

        # Iterate over epochs.
        for epoch in range(self.epochs):
            print('Start of epoch %d' % (epoch,))

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train)
                    loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # Update training metric.
                train_acc_metric(y_batch_train, logits)

                # Log every 200 batches.
                if step % 20 == 0:
                    print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                    print('Training acc (for one batch) at step %s: %s' % (step, float(train_acc_metric.result())))
                    print('Seen so far: %s samples' % ((step + 1) * self.batch_size))

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            print('Training acc over epoch: %s' % (float(train_acc),))
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in val_dataset:
                val_logits = self.model(x_batch_val)
                # Update val metrics
                val_acc_metric(y_batch_val, val_logits)
            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            print('Validation acc: %s' % (float(val_acc),))


if __name__ == '__main__':
    pass
