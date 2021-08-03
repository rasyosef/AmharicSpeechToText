import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

from scripts.script import find_average, count_occurence

class TestCases(unittest.TestCase):
    def test_find_average(self):
        """
        Test that it retunrs the average of a given list
        """
        data = [1, 2, 3]
        result = find_average(data)
        self.assertEqual(result, 2.0)

    def test_input_value(self):
        """
        Provide an assertion level for arg input
        """
        
        self.assertRaises(TypeError, find_average, True)

class TestCountOccurence(unittest.TestCase):
    def test_count_occurence(self):
        """
        Test that it returns the count of each unique values in the given list
        """
        data = [0,0,9,0,8,9,0,7]
        result = count_occurence(data)
        output = {0: 4, 9: 2, 8: 1, 7: 1}
        self.assertAlmostEqual(result, output)

    def test_input_value(self):
        """
        Provide an assertion level for arg input
        """
        self.assertRaises(TypeError, count_occurence, True)

if __name__ == '__main__':
    unittest.main()
