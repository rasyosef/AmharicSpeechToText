from sklearn.preprocessing import LabelEncoder
import numpy as np

def fit_label_encoder(transcripts : dict) -> LabelEncoder:
  """
  This function encodes the amharic characters in the given dictiionary of 
  transcripts into integers.

  Input:
  transcripts - a python dictionary mapping the wav file names to the transcripts
                of those audio files.
  Returns:
  encoder - an sklearn label encoder that has been fitted with all the characters 
  in the transcripts. 
  """
  characters = []
  for audio in transcripts:
    characters.extend(list(transcripts[audio]))
  encoder = LabelEncoder()
  encoder.fit_transform(characters)
  return encoder

def encode_transcripts(transcripts : dict, encoder : LabelEncoder) -> dict:
  """
  This function takes an sklearn label encoder that has already been fitted with
  the amharic characters from the transcripts, along with the original transcript
  and encodes the transcripts for each audio using the given label encoder.

  Input:
  transcripts - a python dictionary mapping the wav file names to the transcripts
                of those audio files.
  encoder - an sklearn label encoder that has been fitted with all the characters 
            in the transcripts.

  Returns:
  transcripts_encoded - a python dictionary mapping the wav file names to the encoded transcripts
                        of those audio files.
  """
  transcripts_encoded = {}
  for audio in transcripts:
    transcripts_encoded[audio] = encoder.transform(list(transcripts[audio]))
  return transcripts_encoded

def decode_predicted(pred,encoder):
  """
  remove the blank character from the predictions and decode the integers back to
  amharic characters.
  """
  dec = []
  for a in pred:
    l = [np.argmax(b) for b in a]
    newl = []
    for i in range(len(l)-1):
      if l[i]!=222 and l[i+1]!=l[i]:
        newl.append(l[i])
    if l[-1] != 222:
      newl.append(l[-1])
    dec.append(''.join(encoder.inverse_transform(newl)).strip())
  return dec