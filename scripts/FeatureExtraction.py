import librosa
import librosa.display
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

    def save_mfcc_spectrograms(self, mfccs: dict, path: str) -> int:
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
            librosa.display.specshow(mfccs[audio], sr=44100, x_axis='time')
            try:
                plt.savefig(path+f'{audio}.png', dpi = 100)
            except FileNotFoundError:
                raise FileNotFoundError(f'The directory {path} does not exist')
            fig.clear()
        return 0