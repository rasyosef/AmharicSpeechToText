import sys

from dataset_loader import load_audio_files, load_transcripts, load_spectrograms_with_transcripts, load_spectrograms_with_transcripts_in_batches
from resize_and_augment import resize_audios_mono, augment_audio, equalize_transcript_dimension
from FeatureExtraction import FeatureExtraction
from transcript_encoder import fit_label_encoder, encode_transcripts, decode_predicted
from models import model_1, model_2, model_3, model_4

import librosa   #for audio processing
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import warnings
import mlflow
import mlflow.keras
import os
import logging
len(os.listdir('../../ALFFA_PUBLIC/ASR/AMHARIC/data/train/wav/'))

sample_rate = 8000

# audio_files, maximum_length = load_audio_files('../../ALFFA_PUBLIC/ASR/AMHARIC/data/train/wav/', sample_rate, True)
# logging.info('loaded audio files')

# print("The longest audio is", maximum_length/sample_rate, 'seconds long')
# print("max length", maximum_length)

# demo_audio = list(audio_files.keys())[0]

transcripts = load_transcripts("../data/train/trsTrain.txt")
logging.info('loaded transcripts')

# audio_files = resize_audios_mono(audio_files, maximum_length)
# print("resized shape", audio_files[demo_audio].shape)

# audio_files = augment_audio(audio_files, sample_rate)
# print("augmented shape", audio_files[demo_audio].shape)

# feature_extractor = FeatureExtraction()
# mfcc_features = feature_extractor.extract_features(audio_files, sample_rate)

# feature_extractor.save_mfcc_spectrograms(mfcc_features, sample_rate, '../data/train/mfcc_spectros/')
# print('Saved mfcc spectros')
# feature_extractor.save_mel_spectrograms(audio_files, sample_rate, '../data/train/mel_spectros/')
# print('saved mel spectros')

audios = [s[:-4] for s in os.listdir('../data/train/mel_spectros/')][:1050]
mfcc_features = {}
for audio in audios:
    mfcc_features[audio] = 1

char_encoder = fit_label_encoder(transcripts)
transcripts_encoded = encode_transcripts(transcripts, char_encoder)
enc_aug_transcripts = equalize_transcript_dimension(mfcc_features, transcripts_encoded, 60)

print('model summary')
model = model_3(char_encoder)
print(model.summary())

import math

data_batch_size = 100
training_batch_size = 10
number_of_epochs = 150

number_of_batches = math.ceil(len(mfcc_features)/data_batch_size)

X_test, y_test = load_spectrograms_with_transcripts_in_batches(mfcc_features, 
                                                              enc_aug_transcripts, data_batch_size, number_of_batches-1,
                                                              '../data/train/mel_spectros/')

mlflow.keras.autolog()
for i in range(number_of_epochs):
    if i % 10 == 0:
        model.save('../models/mel_model_4_epoch_{}.h5'.format(i))
    for j in range(number_of_batches-1):
        print('Epoch {}: training batch {}'.format(i+1,j))
        X_train, y_train = load_spectrograms_with_transcripts_in_batches(mfcc_features, 
                                                              enc_aug_transcripts, data_batch_size, j,
                                                              '../data/train/mel_spectros/')
        history = model.fit([X_train, y_train], validation_data = [X_test[:10], y_test[:10]], batch_size = training_batch_size, epochs = 1)

# with mlflow.start_run() as run:
#     mlflow.log_metric("ctc_loss", history.history['loss'][-1])

predicted = model.predict([X_test,y_test])
predicted_trans = decode_predicted(predicted, char_encoder)
real_trans = [''.join(char_encoder.inverse_transform(y)) for y in y_test]
for i in range(5):
    print("pridicted:",predicted_trans[i])
    print("actual:",real_trans[i])