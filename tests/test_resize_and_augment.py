import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
from scripts.resize_and_augment import resize_audios_mono

class TestResizeAndAugment(unittest.TestCase):
    def test_resize_audios_mono(self):
        """
        Test that it retunrs the average of a given list
        """
        data = {1:np.array([1,2]), 2: np.array([1])}
        result = resize_audios_mono(data, 4)
        self.assertEqual(result, {1:np.array([1,2,0,0]), 2: np.array([1,0,0,0])})

if __name__ == '__main__':
    unittest.main()