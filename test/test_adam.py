import unittest
import sys
sys.path.append("../lib/concerns")
from adam import Adam

class TestAdam(unittest.TestCase):
    def setUp(self):
        self.adam = Adam()

    def test_update(self):
        pass

if __name__ == "__main__":
    unittest.main()
