import tensorflow as tf
import tensorflow.keras.layers as layers

NUM_FFT = 512
NUM_FREQS = 257
# some tentative constants
NUM_MEL = 60
SAMPLE_RATE = 44100
F_MIN = 0
F_MAX = 12000

class LogMelgramLayer(tf.keras.layers.Layer):
    def __init__(self, num_fft, hop_length, **kwargs):
        super(LogMelgramLayer, self).__init__(**kwargs)
        self.num_fft = num_fft
        self.hop_length = hop_length

        assert num_fft // 2 + 1 == NUM_FREQS
        lin_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=NUM_MEL,
            num_spectrogram_bins=NUM_FREQS,
            sample_rate=SAMPLE_RATE,
            lower_edge_hertz=F_MIN,
            upper_edge_hertz=F_MAX,
        )

        self.lin_to_mel_matrix = lin_to_mel_matrix

    def build(self, input_shape):
        self.non_trainable_weights.append(self.lin_to_mel_matrix)
        super(LogMelgramLayer, self).build(input_shape)

    def call(self, input):
        """
        Args:
            input (tensor): Batch of mono waveform, shape: (None, N)
        Returns:
            log_melgrams (tensor): Batch of log mel-spectrograms, shape: (None, num_frame, mel_bins, channel=1)
        """

        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator
      
        # tf.signal.stft seems to be applied along the last axis
        stfts = tf.signal.stft(
            input, frame_length=self.num_fft, frame_step=self.hop_length
        )
        mag_stfts = tf.abs(stfts)

        melgrams = tf.tensordot(tf.square(mag_stfts), self.lin_to_mel_matrix, axes=[2, 0])
        log_melgrams = _tf_log10(melgrams + 10**-5)
        return tf.expand_dims(log_melgrams, 3)

    def get_config(self):
        config = {'num_fft': self.num_fft, 'hop_length': self.hop_length}
        base_config = super(LogMelgramLayer, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))

class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

def my_model(encoder, max_length):
    inputs = layers.Input(
        shape=(max_length,), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    log_melgram_layer = LogMelgramLayer(
        num_fft=512,
        hop_length=128,
    )

    log_melgrams = log_melgram_layer(inputs)

    # First conv block
    x = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(log_melgrams)

    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.BatchNormalization()(x)
    # Second conv block
    x = layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv3",
    )(x)
    
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    x = layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv4",
    )(x)
    
    x = layers.BatchNormalization()(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    if tf.__version__ >= '2.3':
        new_shape = (x.type_spec.shape[-3], x.type_spec.shape[-2]*x.type_spec.shape[-1])
    else:
        new_shape = (x.shape[-3], x.shape[-2]*x.shape[-1])
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128, activation="relu", name="dense2")(x)
    x = layers.Dropout(0.2)(x)
    
    # RNNs
    x = layers.LSTM(256, return_sequences=True, dropout=0.25)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(256, return_sequences=True, dropout=0.25)(x)
    x = layers.BatchNormalization()(x)
    # Output layer
    x = layers.Dense(
        len(encoder.classes_) + 1, activation="softmax", name="dense_final"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = tf.keras.models.Model(
        inputs=[inputs, labels], outputs=output, name="stt_model_v2"
    )
    # Optimizer
    opt = tf.keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    
    return model