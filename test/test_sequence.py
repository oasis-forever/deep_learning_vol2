import unittest
import numpy as np
import sys
sys.path.append("../lib/concerns")
from sequence import Sequence

class TestSequence(unittest.TestCase):
    def setUp(self):
        self.sequence = Sequence()
        self.file_path = "../texts/addition.txt"

    def test_text_to_dict(self):
        questions, answers = self.sequence._text_to_dict(self.file_path)
        self.assertEqual(200000, len(questions))
        self.assertEqual(200000, len(answers))

    def test_update_vocab(self):
        text = "16+75  _91"
        self.sequence._update_vocab(text)
        self.assertEqual({
            "1": 0,
            "6": 1,
            "+": 2,
            "7": 3,
            "5": 4,
            " ": 5,
            "_": 6,
            "9": 7
        }, self.sequence.char_to_id)
        self.assertEqual({
            0: "1",
            1: "6",
            2: "+",
            3: "7",
            4: "5",
            5: " ",
            6: "_",
            7: "9"
        }, self.sequence.id_to_char)

    def test_create_numpy_array(self):
        questions, answers = self.sequence._text_to_dict(self.file_path)
        self.sequence._create_vocab_dict(questions, answers)
        x, t = self.sequence._create_numpy_array(questions, answers)
        self.assertEqual((50000, 7), x.shape)
        self.assertEqual((50000, 5), t.shape)

    def test_load_data(self):
        (x_train, t_train), (x_test, t_test) = self.sequence.load_data(self.file_path)
        self.assertEqual((135000, 7), x_train.shape)
        self.assertEqual((135000, 5), t_train.shape)
        self.assertEqual((15000, 7), x_test.shape)
        self.assertEqual((15000, 5), t_test.shape)

    def test_get_vocab(self):
        (x_train, t_train), (x_test, t_test) = self.sequence.load_data(self.file_path)
        vocab = self.sequence.get_vocab()
        char_to_id, id_to_char = vocab
        self.assertEqual({
            "1": 0,
            "6": 1,
            "+": 2,
            "7": 3,
            "5": 4,
            " ": 5,
            "_": 6,
            "9": 7,
            "2": 8,
            "0": 9,
            "3": 10,
            "8": 11,
            "4": 12
        }, char_to_id)
        self.assertEqual({
            0: "1",
            1: "6",
            2: "+",
            3: "7",
            4: "5",
            5: " ",
            6: "_",
            7: "9",
            8: "2",
            9: "0",
            10: "3",
            11: "8",
            12: "4"
        }, id_to_char)
        self.assertEqual("70+174 ", "".join([id_to_char[c] for c in x_train[0]]))
        self.assertEqual("_244 ", "".join([id_to_char[c] for c in t_train[0]]))

if __name__ == "__main__":
    unittest.main()
