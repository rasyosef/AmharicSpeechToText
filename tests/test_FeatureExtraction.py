import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

from scripts.FeatureExtraction import FeatureExtraction

class TestExtractFeatures(unittest.TestCase):

    def test_input_value_extract_features(self):
        """
        Provide an assertion level for arg input
        """
        
        self.assertRaises(TypeError, FeatureExtraction.extract_features, True)

    def test_input_value_save_mfcc_spectrograms(self):
        """
        Provide an assertion level for arg input
        """
        
        self.assertRaises(TypeError, FeatureExtraction.save_mfcc_spectrograms, True)


if __name__ == '__main__':
    unittest.main()
