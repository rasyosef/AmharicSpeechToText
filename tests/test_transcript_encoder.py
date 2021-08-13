import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
from scripts.transcript_encoder import fit_label_encoder

class TestFitLabelEncoder(unittest.TestCase):
    def test_fit_label_encoder(self):
        """
        Test that it retunrs the average of a given list
        """
        data = {1:'abc', 2: 'bcd'}
        result = fit_label_encoder(data).transform(list('abc'))
        self.assertEqual(result, np.array([0,1,2]))

if __name__ == '__main__':
    unittest.main()
