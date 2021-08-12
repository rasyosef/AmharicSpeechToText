import librosa   #for audio processing
import numpy as np
from matplotlib import image
import os

def load_audio_files(path : str, sampling_rate : int, to_mono : bool) -> (dict, int):

  """
  Load the audio files and produce a dictionary mapping the audio filenames 
  to numpy arrays of the audio sampled at the given sample rate.

  Inputs: 
  path - a path to the directory that contains the audio files
  sample_rate - the sampling rate for the audio files
  to_mono - a boolean value denoting whether to convert signal to mono

  Returns:
  audio_files - audios - a dictionary mapping the wav file names to the sampled audio array
  max_length - the maximum length of a sampled audio array in our dataset
  """

  audio_files = {}
  max_length = 0
  i = 0
  files = os.listdir(path)
  # import random
  # random.shuffle(files)
  for file in files:
    audio, rate = librosa.load(path+file, sr=sampling_rate, mono = to_mono)
    if len(audio)/rate <= 10:
      audio_files[file.split('.')[0]] = audio
      max_length = max(max_length,len(audio))
    i+=1
    if i%20 == 0:
      print('loaded',i,'audio files')
    if i == 12000:
      break
  return audio_files, max_length

def load_transcripts(filepath : str) -> dict:
  """
  Load the transcript file and produce a dictionary mapping the audio filenames 
  to the transcripts for those audio files.

  Inputs: 
  filepath - a path to the transcript file

  Returns:
  transcripts - a python dictionary mapping the wav file names to the transcripts
                of those audio files.
  """
  transcripts = {}
  with open (filepath, encoding="utf-8")as f:
    #print(f.readlines()[1])
    for line in f.readlines():
      
      text, filename = line.split("</s>")
      text, filename = text.strip()[3:], filename.strip()[1:-1]
      transcripts[filename] = text
    return transcripts


def load_spectrograms_with_transcripts(mfcc_features : dict, encoded_transcripts : dict, path : str):
  """
  Loads the spectrogram images as numpy arrays

  Inputs:
  mfcc_features - a python dictionary mapping the wav file names to the mfcc 
                  coefficients of the sampled audio files
  encoded_transcripts - a python dictionary mapping the wav file names to the 
                        encoded transcripts of those audio files.
  path - the path to the directory that contains the spectrogram images

  Returns:
  X_train - a numpy array containing the mfcc spectrograms of the sampled audio files
  y_train - a numpy array containing the encoded transcripts of the sampled audio files
            in the same order as they appear in X_train
  """
  X_train = []
  y_train = []
  for audio in mfcc_features:
    specgram = image.imread(path+f'{audio}.png')
    X_train.append(specgram)
    y_train.append(encoded_transcripts[audio])
  return np.array(X_train), np.array(y_train)

def load_spectrograms_with_transcripts_in_batches(mfcc_features : dict, encoded_transcripts : dict,
                                                 batch_size : int, batch_no : int, path : str):
  """
  Loads the spectrogram images as numpy arrays

  Inputs:
  mfcc_features - a python dictionary mapping the wav file names to the mfcc 
                  coefficients of the sampled audio files
  encoded_transcripts - a python dictionary mapping the wav file names to the 
                        encoded transcripts of those audio files.
  batch_size - the size of the batch when loading
  batch_no - the index of the batch
  path - the path to the directory that contains the spectrogram images

  Returns:
  X_train - a numpy array containing the mfcc spectrograms of the sampled audio files
  y_train - a numpy array containing the encoded transcripts of the sampled audio files
            in the same order as they appear in X_train
  """
  X_train = []
  y_train = []
  audio_names = list(mfcc_features.keys())
  i = batch_size*batch_no
  j = batch_size*(batch_no + 1)
  for audio in audio_names[i:j]:
    specgram = image.imread(path+f'{audio}.png')
    X_train.append(specgram)
    y_train.append(encoded_transcripts[audio])
  return np.array(X_train), np.array(y_train)
