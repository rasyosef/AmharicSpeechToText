import librosa
import librosa.display
import matplotlib.pyplot as plt

class FeatureExtraction:
    def extract_features(self, audios : dict, sample_rate : int) -> dict:
        """
        The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of 
        features (usually about 10–20) which concisely describe the overall shape of a 
        spectral envelope. It models the characteristics of the human voice.
        We compute the Mel frequency cepstral coefficients for each audio file.

        Inputs: 
        audios - a dictionary mapping the wav file names to the sampled audio array
        sample_rate - the sample rate for the audio

        Returns:
        mfcc_features - a python dictionary mapping the wav file names to the mfcc 
                        coefficients of the sampled audio files
        """
        if type(audios) != dict or type(sample_rate) != int:
            raise TypeError("""argument audios must be of type dict and argument sample_rate
                            must be of type int""")

        mfcc_features = {}
        for audio in audios:
            mfcc_features[audio] = librosa.feature.mfcc(audios[audio], sr=sample_rate)
        return mfcc_features

    def save_mfcc_spectrograms(self, mfccs: dict, sample_rate: int, path: str) -> int:
        """
        The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of 
        features (usually about 10–20) which concisely describe the overall shape of a 
        spectral envelope. It models the characteristics of the human voice.

        A Spectrogram captures the nature of the audio as an image by decomposing 
        it into the set of frequencies that are included in it.

        We plot the MFCC spectrogram for each audio file, and save the plots as .png 
        image files to the given target directory.

        Inputs: 
        mfccs - a python dictionary mapping the wav file names to the mfcc 
                coefficients of the sampled audio files
        sample_rate - the sampling rate for the audio
        path - the file path to the target directory

        Returns:
        0 if the spectrograms were saved successfully, and 
        raises a FileNotFoundError if the given path doesn't exist
        """
        if type(mfccs) != dict or type(path) != str:
            raise TypeError("""argument mfccs must be of type dict and argument path
                            must be of type string (str)""")
        for audio in mfccs:
            fig, ax = plt.subplots()
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            librosa.display.specshow(mfccs[audio], sr=sample_rate, x_axis='time')
            try:
                plt.savefig(path+f'{audio}.png')
            except FileNotFoundError:
                raise FileNotFoundError(f'The directory {path} does not exist')
            plt.close()
        return 0

    def save_mel_spectrograms(self, audios: dict, sample_rate: int, path: str) -> int:
        """
        The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of 
        features (usually about 10–20) which concisely describe the overall shape of a 
        spectral envelope. It models the characteristics of the human voice.

        A Spectrogram captures the nature of the audio as an image by decomposing 
        it into the set of frequencies that are included in it.

        We plot the MFCC spectrogram for each audio file, and save the plots as .png 
        image files to the given target directory.

        Inputs: 
        mfccs - a python dictionary mapping the wav file names to the mfcc 
                coefficients of the sampled audio files
        sample_rate - the sampling rate for the audio
        path - the file path to the target directory

        Returns:
        0 if the spectrograms were saved successfully, and 
        raises a FileNotFoundError if the given path doesn't exist
        """
        if type(audios) != dict or type(path) != str:
            raise TypeError("""argument mfccs must be of type dict and argument path
                            must be of type string (str)""")
        for audio in audios:
            X = librosa.stft(audios[audio], n_fft = 512)
            Xdb = librosa.amplitude_to_db(abs(X))
            fig, ax = plt.subplots()
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')
            try:
                plt.savefig(path+f'{audio}.png')
            except FileNotFoundError:
                raise FileNotFoundError(f'The directory {path} does not exist')
            plt.close()
        return 0