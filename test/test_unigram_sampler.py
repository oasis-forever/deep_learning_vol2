import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sys
sys.path.append("../lib/concerns")
from unigram_sampler import UnigramSampler

class TestUnigramSampler(unittest.TestCase):
    def setUp(self):
        corpus      = np.array([0, 1, 2, 3, 4, 1, 2, 3])
        power       = 0.75
        sample_size = 5
        self.unigram_sampler = UnigramSampler(corpus, power, sample_size)

    def test_get_negative_sample(self):
        target = None
        negative_sample = self.unigram_sampler.get_negative_sample(target)
        assert_array_equal(np.array([
            None
       ]), negative_sample)

if __name__ == "__main__":
    unittest.main()
