"""
Created on June 17, 2019

@author: daniel
"""
import numpy as np
from unittest import TestCase
from Bio.Alphabet import  Gapped
from Bio.Alphabet.IUPAC import ExtendedIUPACProtein
from utils import build_mapping, convert_seq_to_numeric


class TestUtils(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.expected_gap_chars = {'-', '.', '*'}
        cls.alphabet = ExtendedIUPACProtein()
        cls.alphabet_str = cls.alphabet.letters
        cls.alphabet_list = [char for char in cls.alphabet_str]

    def test1a_build_mapping_from_str(self):
        size, gap_chars, mapping, reverse = build_mapping(alphabet=self.alphabet_str)
        self.assertEqual(size, len(self.alphabet_str))
        self.assertEqual(len(gap_chars), 3)
        self.assertEqual(gap_chars, self.expected_gap_chars)
        for char in gap_chars:
            self.assertEqual(mapping[char], size)
        self.assertEqual(len(mapping), size + len(gap_chars))
        self.assertEqual(len(reverse), size)
        for i in range(size):
            self.assertEqual(mapping[self.alphabet_str[i]], i)
            self.assertEqual(reverse[i], self.alphabet_str[i])

    def test1b_build_mapping_from_list(self):
        size, gap_chars, mapping, reverse = build_mapping(alphabet=self.alphabet_list)
        self.assertEqual(size, len(self.alphabet_list))
        self.assertEqual(len(gap_chars), 3)
        self.assertEqual(gap_chars, self.expected_gap_chars)
        for char in gap_chars:
            self.assertEqual(mapping[char], size)
        self.assertEqual(len(mapping), size + len(gap_chars))
        self.assertEqual(len(reverse), size)
        for i in range(size):
            self.assertEqual(mapping[self.alphabet_list[i]], i)
            self.assertEqual(reverse[i], self.alphabet_list[i])

    def test1c_build_mapping_from_alphabet(self):
        alphabet = ExtendedIUPACProtein()
        size, gap_chars, mapping, reverse = build_mapping(alphabet=alphabet)
        self.assertEqual(size, len(alphabet.letters))
        self.assertEqual(len(gap_chars), 3)
        self.assertEqual(gap_chars, self.expected_gap_chars)
        for char in gap_chars:
            self.assertEqual(mapping[char], size)
        self.assertEqual(len(mapping), size + len(gap_chars))
        self.assertEqual(len(reverse), size)
        for i in range(size):
            self.assertEqual(mapping[alphabet.letters[i]], i)
            self.assertEqual(reverse[i], alphabet.letters[i])

    def test1d_build_mapping_from_gapped_alphabet(self):
        alphabet = Gapped(ExtendedIUPACProtein())
        size, gap_chars, mapping, reverse = build_mapping(alphabet=alphabet)
        self.assertEqual(size, len(alphabet.letters))
        self.assertEqual(len(gap_chars), 2)
        self.assertEqual(gap_chars, {'.', '*'})
        for char in gap_chars:
            self.assertEqual(mapping[char], size)
        self.assertEqual(len(mapping), size + len(gap_chars))
        self.assertEqual(len(reverse), size)
        for i in range(size):
            self.assertEqual(mapping[alphabet.letters[i]], i)
            self.assertEqual(reverse[i], alphabet.letters[i])

    def test2_convert_seq_to_numeric(self):
        size, gap_chars, mapping, reverse = build_mapping(alphabet=self.alphabet_str)
        query_seq_7hvp = 'PQITLWQRPLVTIRIGGQLKEALLDTGADDTVLE--EMNL--PGKWK----PKMIGGIGGFIKVRQYDQIPVEI-GHKAIGTV---LVGPTP'\
                         'VNIIGRNLLTQIG-TLNF'
        expected_array = np.array([12, 13, 7, 16, 9, 18, 13, 14, 12, 9, 17, 16, 7, 14, 7, 5, 5, 13, 9, 8, 3, 0, 9, 9, 2,
                                   16, 5, 0, 2, 2, 16, 17, 9, 3, 26, 26, 3, 10, 11, 9, 26, 26, 12, 5, 8, 18, 8, 26, 26,
                                   26, 26, 12, 8, 10, 7, 5, 5, 7, 5, 5, 4, 7, 8, 17, 14, 13, 19, 2, 13, 7, 12, 17, 3, 7,
                                   26, 5, 6, 8, 0, 7, 5, 16, 17, 26, 26, 26, 9, 17, 5, 12, 16, 12, 17, 11, 7, 7, 5, 14,
                                   11, 9, 9, 16, 13, 7, 5, 26, 16, 9, 11, 4])
        numeric_seq_7hvp = convert_seq_to_numeric(seq=query_seq_7hvp, mapping=mapping)
        diff_in_conversion = expected_array - numeric_seq_7hvp
        self.assertEqual(numeric_seq_7hvp.shape, expected_array.shape)
        self.assertFalse(np.any(diff_in_conversion))
