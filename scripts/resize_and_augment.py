import numpy as np

def resize_audios_mono(audios : dict, max_length : int) -> dict:
  """
  Here we pad the sampled audio with zeros so tha all of the sampled audios 
  have equal length

  Inputs: 
  audios - a dictionary mapping the wav file names to the sampled audio array
  max_length - the maximum length of a sampled audio array in our dataset

  Returns:
  audios - a python dictionary mapping the wav file names to the padded
          audio samples
  """
  for name in audios:
    audios[name] = np.pad(audios[name], 
                          (0, max_length-len(audios[name])),
                          mode = 'constant')
  return audios


def augment_audio(audios : dict, sample_rate : int) -> dict:
  """
  Here we shift the wave by sample_rate/10 factor. This will move the wave to the 
  right by given factor along time axis. For achieving this I have used numpy’s 
  roll function to generate time shifting.

  Inputs: 
  audios - a dictionary mapping the wav file names to the sampled audio array
  sample_rate - the sample rate for the audio

  Returns:
  audios - a python dictionary mapping the wav file names to the augmented 
          audio samples
  """
  for name in audios:
    audios[name] = np.roll(audios[name], int(sample_rate/10))
  return audios

# def equalize_transcript_dimension(y, truncate_len):
#   """
#   Make all transcripts have equal number of characters by padding the the short
#   ones with spaces
#   """
#   max_len = max([len(trans) for trans in y])
#   print("maximum number of characters in a transcript:", max_len)
#   new_y = []
#   for trans in y:
#     new_y.append(np.pad(trans, 
#                           (0, max_len-len(trans)),
#                           mode = 'constant')[:truncate_len])
#   return np.array(new_y)

def equalize_transcript_dimension(mfccs, encoded_transcripts, truncate_len):
  """
  Make all transcripts have equal number of characters by padding the the short
  ones with spaces
  """
  max_len = max([len(encoded_transcripts[trans]) for trans in mfccs])
  print("maximum number of characters in a transcript:", max_len)
  new_trans = {}
  for trans in mfccs:
    new_trans[trans] = np.pad(encoded_transcripts[trans], 
                          (0, max_len-len(encoded_transcripts[trans])),
                          mode = 'constant')[:truncate_len]
  return new_trans