"""
Created on June 17, 2019

@author: daniel
"""
import unittest
import numpy as np
from unittest import TestCase
from Bio.Alphabet import Gapped, ThreeLetterProtein
from Bio.Alphabet.IUPAC import IUPACProtein
from utils import build_mapping, convert_seq_to_numeric, compute_rank_and_coverage


class TestBuildMapping(TestCase):

    def test_string_alphabet(self):
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet)
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'-', '.', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '-': 20, '.': 20, '*': 20})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_string_alphabet_with_gap1(self):
        alphabet = 'ACDEFGHIKLMNPQRSTVWY-'
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet)
        self.assertEqual(size, 21)
        self.assertEqual(gap_characters, {'.', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '-': 20, '.': 21, '*': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-'])
        self.assertTrue(reverse_map_check.all())

    def test_string_alphabet_with_gap2(self):
        alphabet = 'ACDEFGHIKLMNPQRSTVWY.'
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet)
        self.assertEqual(size, 21)
        self.assertEqual(gap_characters, {'-', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '.': 20, '-': 21, '*': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '.'])
        self.assertTrue(reverse_map_check.all())

    def test_string_alphabet_with_gap3(self):
        alphabet = 'ACDEFGHIKLMNPQRSTVWY*'
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet)
        self.assertEqual(size, 21)
        self.assertEqual(gap_characters, {'-', '.'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '*': 20, '-': 21, '.': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '*'])
        self.assertTrue(reverse_map_check.all())

    def test_string_alphabet_with_skip_letter(self):
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet, skip_letters=['B'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'-', '.', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '-': 20, '.': 20, '*': 20, 'B': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_string_alphabet_with_skip_letters(self):
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet, skip_letters=['B', 'J'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'-', '.', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '-': 20, '.': 20, '*': 20, 'B': 21, 'J': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_string_alphabet_with_gap_skip_letter(self):
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet, skip_letters=['-'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'.', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '.': 20, '*': 20, '-': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_string_alphabet_with_gap_skip_letters(self):
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet, skip_letters=['.', '*'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'-'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '-': 20, '.': 21, '*': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_string_alphabet_skip_letter_overlap(self):
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet, skip_letters=['A'])
        self.assertEqual(size, 19)
        self.assertEqual(gap_characters, {'-', '.', '*'})
        self.assertEqual(mapping, {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'K': 7, 'L': 8,
                                   'M': 9, 'N': 10, 'P': 11, 'Q': 12, 'R': 13, 'S': 14, 'T': 15, 'V': 16, 'W': 17,
                                   'Y': 18, '-': 19, '.': 19, '*': 19, 'A': 20})
        reverse_map_check = reverse_mapping == np.array(['C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_list_alphabet(self):
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                    'Y']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet)
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'-', '.', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '-': 20, '.': 20, '*': 20})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_list_alphabet_with_gap1(self):
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                    'Y', '-']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet)
        self.assertEqual(size, 21)
        self.assertEqual(gap_characters, {'.', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '-': 20, '.': 21, '*': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-'])
        self.assertTrue(reverse_map_check.all())

    def test_list_alphabet_with_gap2(self):
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                    'Y', '.']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet)
        self.assertEqual(size, 21)
        self.assertEqual(gap_characters, {'-', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '.': 20, '-': 21, '*': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '.'])
        self.assertTrue(reverse_map_check.all())

    def test_list_alphabet_with_gap3(self):
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                    'Y', '*']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet)
        self.assertEqual(size, 21)
        self.assertEqual(gap_characters, {'-', '.'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '*': 20, '-': 21, '.': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '*'])
        self.assertTrue(reverse_map_check.all())

    def test_list_alphabet_with_skip_letter(self):
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                    'Y']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet, skip_letters=['B'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'-', '.', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '-': 20, '.': 20, '*': 20, 'B': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_list_alphabet_with_skip_letters(self):
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                    'Y']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet, skip_letters=['B', 'J'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'-', '.', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '-': 20, '.': 20, '*': 20, 'B': 21, 'J': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_list_alphabet_with_gap_skip_letter(self):
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                    'Y']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet, skip_letters=['-'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'.', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '.': 20, '*': 20, '-': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_list_alphabet_with_gap_skip_letters(self):
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                    'Y']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet, skip_letters=['.', '*'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'-'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '-': 20, '.': 21, '*': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_list_alphabet_skip_letter_overlap(self):
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                    'Y']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet, skip_letters=['A'])
        self.assertEqual(size, 19)
        self.assertEqual(gap_characters, {'-', '.', '*'})
        self.assertEqual(mapping, {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'K': 7, 'L': 8,
                                   'M': 9, 'N': 10, 'P': 11, 'Q': 12, 'R': 13, 'S': 14, 'T': 15, 'V': 16, 'W': 17,
                                   'Y': 18, '-': 19, '.': 19, '*': 19, 'A': 20})
        reverse_map_check = reverse_mapping == np.array(['C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_two_letter_list_alphabet(self):
        alphabet = ['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK', 'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS',
                    'TT', 'VV', 'WW', 'YY']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet)
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'--', '..', '**'})
        self.assertEqual(mapping, {'AA': 0, 'CC': 1, 'DD': 2, 'EE': 3, 'FF': 4, 'GG': 5, 'HH': 6, 'II': 7, 'KK': 8,
                                   'LL': 9, 'MM': 10, 'NN': 11, 'PP': 12, 'QQ': 13, 'RR': 14, 'SS': 15, 'TT': 16,
                                   'VV': 17, 'WW': 18, 'YY': 19, '--': 20, '..': 20, '**': 20})
        reverse_map_check = reverse_mapping == np.array(['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK',
                                                         'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS', 'TT', 'VV',
                                                         'WW', 'YY'])
        self.assertTrue(reverse_map_check.all())

    def test_two_letter_list_alphabet_with_gap1(self):
        alphabet = ['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK', 'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS',
                    'TT', 'VV', 'WW', 'YY', '--']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet)
        self.assertEqual(size, 21)
        self.assertEqual(gap_characters, {'..', '**'})
        self.assertEqual(mapping, {'AA': 0, 'CC': 1, 'DD': 2, 'EE': 3, 'FF': 4, 'GG': 5, 'HH': 6, 'II': 7, 'KK': 8,
                                   'LL': 9, 'MM': 10, 'NN': 11, 'PP': 12, 'QQ': 13, 'RR': 14, 'SS': 15, 'TT': 16,
                                   'VV': 17, 'WW': 18, 'YY': 19, '--': 20, '..': 21, '**': 21})
        reverse_map_check = reverse_mapping == np.array(['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK',
                                                         'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS', 'TT', 'VV',
                                                         'WW', 'YY', '--'])
        self.assertTrue(reverse_map_check.all())

    def test_two_letter_list_alphabet_with_gap2(self):
        alphabet = ['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK', 'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS',
                    'TT', 'VV', 'WW', 'YY', '..']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet)
        self.assertEqual(size, 21)
        self.assertEqual(gap_characters, {'--', '**'})
        self.assertEqual(mapping, {'AA': 0, 'CC': 1, 'DD': 2, 'EE': 3, 'FF': 4, 'GG': 5, 'HH': 6, 'II': 7, 'KK': 8,
                                   'LL': 9, 'MM': 10, 'NN': 11, 'PP': 12, 'QQ': 13, 'RR': 14, 'SS': 15, 'TT': 16,
                                   'VV': 17, 'WW': 18, 'YY': 19, '..': 20, '--': 21, '**': 21})
        reverse_map_check = reverse_mapping == np.array(['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK',
                                                         'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS', 'TT', 'VV',
                                                         'WW', 'YY', '..'])
        self.assertTrue(reverse_map_check.all())

    def test_two_letter_list_alphabet_with_gap3(self):
        alphabet = ['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK', 'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS',
                    'TT', 'VV', 'WW', 'YY', '**']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet)
        self.assertEqual(size, 21)
        self.assertEqual(gap_characters, {'--', '..'})
        self.assertEqual(mapping, {'AA': 0, 'CC': 1, 'DD': 2, 'EE': 3, 'FF': 4, 'GG': 5, 'HH': 6, 'II': 7, 'KK': 8,
                                   'LL': 9, 'MM': 10, 'NN': 11, 'PP': 12, 'QQ': 13, 'RR': 14, 'SS': 15, 'TT': 16,
                                   'VV': 17, 'WW': 18, 'YY': 19, '**': 20, '--': 21, '..': 21})
        reverse_map_check = reverse_mapping == np.array(['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK',
                                                         'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS', 'TT', 'VV',
                                                         'WW', 'YY', '**'])
        self.assertTrue(reverse_map_check.all())

    def test_two_letter_list_alphabet_with_skip_letter1(self):
        alphabet = ['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK', 'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS',
                    'TT', 'VV', 'WW', 'YY']
        with self.assertRaises(ValueError):
            build_mapping(alphabet=alphabet, skip_letters=['B'])

    def test_two_letter_list_alphabet_with_skip_letter2(self):
        alphabet = ['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK', 'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS',
                    'TT', 'VV', 'WW', 'YY']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet, skip_letters=['BB'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'--', '..', '**'})
        self.assertEqual(mapping, {'AA': 0, 'CC': 1, 'DD': 2, 'EE': 3, 'FF': 4, 'GG': 5, 'HH': 6, 'II': 7, 'KK': 8,
                                   'LL': 9, 'MM': 10, 'NN': 11, 'PP': 12, 'QQ': 13, 'RR': 14, 'SS': 15, 'TT': 16,
                                   'VV': 17, 'WW': 18, 'YY': 19, '**': 20, '--': 20, '..': 20, 'BB': 21})
        reverse_map_check = reverse_mapping == np.array(['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK',
                                                         'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS', 'TT', 'VV',
                                                         'WW', 'YY'])
        self.assertTrue(reverse_map_check.all())

    def test_two_letter_list_alphabet_with_skip_letters1(self):
        alphabet = ['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK', 'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS',
                    'TT', 'VV', 'WW', 'YY']
        with self.assertRaises(ValueError):
            build_mapping(alphabet=alphabet, skip_letters=['B', 'J'])

    def test_two_letter_list_alphabet_with_skip_letters2(self):
        alphabet = ['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK', 'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS',
                    'TT', 'VV', 'WW', 'YY']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet,
                                                                       skip_letters=['BB', 'JJ'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'--', '..', '**'})
        self.assertEqual(mapping, {'AA': 0, 'CC': 1, 'DD': 2, 'EE': 3, 'FF': 4, 'GG': 5, 'HH': 6, 'II': 7, 'KK': 8,
                                   'LL': 9, 'MM': 10, 'NN': 11, 'PP': 12, 'QQ': 13, 'RR': 14, 'SS': 15, 'TT': 16,
                                   'VV': 17, 'WW': 18, 'YY': 19, '--': 20, '..': 20, '**': 20, 'BB': 21, 'JJ': 21})
        reverse_map_check = reverse_mapping == np.array(['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK',
                                                         'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS', 'TT', 'VV',
                                                         'WW', 'YY'])
        self.assertTrue(reverse_map_check.all())

    def test_two_letter_list_alphabet_with_gap_skip_letter1(self):
        alphabet = ['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK', 'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS',
                    'TT', 'VV', 'WW', 'YY']
        with self.assertRaises(ValueError):
            build_mapping(alphabet=alphabet, skip_letters=['-'])

    def test_two_letter_list_alphabet_with_gap_skip_letter2(self):
        alphabet = ['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK', 'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS',
                    'TT', 'VV', 'WW', 'YY']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet, skip_letters=['--'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'..', '**'})
        self.assertEqual(mapping, {'AA': 0, 'CC': 1, 'DD': 2, 'EE': 3, 'FF': 4, 'GG': 5, 'HH': 6, 'II': 7, 'KK': 8,
                                   'LL': 9, 'MM': 10, 'NN': 11, 'PP': 12, 'QQ': 13, 'RR': 14, 'SS': 15, 'TT': 16,
                                   'VV': 17, 'WW': 18, 'YY': 19, '..': 20, '**': 20, '--': 21})
        reverse_map_check = reverse_mapping == np.array(['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK',
                                                         'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS', 'TT', 'VV',
                                                         'WW', 'YY'])
        self.assertTrue(reverse_map_check.all())

    def test_two_letter_list_alphabet_with_gap_skip_letters1(self):
        alphabet = ['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK', 'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS',
                    'TT', 'VV', 'WW', 'YY']
        with self.assertRaises(ValueError):
            build_mapping(alphabet=alphabet, skip_letters=['.', '*'])

    def test_two_letter_list_alphabet_with_gap_skip_letters2(self):
        alphabet = ['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK', 'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS',
                    'TT', 'VV', 'WW', 'YY']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet,
                                                                       skip_letters=['..', '**'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'--'})
        self.assertEqual(mapping, {'AA': 0, 'CC': 1, 'DD': 2, 'EE': 3, 'FF': 4, 'GG': 5, 'HH': 6, 'II': 7, 'KK': 8,
                                   'LL': 9, 'MM': 10, 'NN': 11, 'PP': 12, 'QQ': 13, 'RR': 14, 'SS': 15, 'TT': 16,
                                   'VV': 17, 'WW': 18, 'YY': 19, '--': 20, '..': 21, '**': 21})
        reverse_map_check = reverse_mapping == np.array(['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK',
                                                         'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS', 'TT', 'VV',
                                                         'WW', 'YY'])
        self.assertTrue(reverse_map_check.all())

    def test_two_letter_list_alphabet_skip_letter_overlap1(self):
        alphabet = ['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK', 'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS',
                    'TT', 'VV', 'WW', 'YY']
        with self.assertRaises(ValueError):
            build_mapping(alphabet=alphabet, skip_letters=['A'])

    def test_two_letter_list_alphabet_skip_letter_overlap2(self):
        alphabet = ['AA', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK', 'LL', 'MM', 'NN', 'PP', 'QQ', 'RR', 'SS',
                    'TT', 'VV', 'WW', 'YY']
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet, skip_letters=['AA'])
        self.assertEqual(size, 19)
        self.assertEqual(gap_characters, {'--', '..', '**'})
        self.assertEqual(mapping, {'CC': 0, 'DD': 1, 'EE': 2, 'FF': 3, 'GG': 4, 'HH': 5, 'II': 6, 'KK': 7, 'LL': 8,
                                   'MM': 9, 'NN': 10, 'PP': 11, 'QQ': 12, 'RR': 13, 'SS': 14, 'TT': 15, 'VV': 16,
                                   'WW': 17, 'YY': 18, '--': 19, '..': 19, '**': 19, 'AA': 20})
        reverse_map_check = reverse_mapping == np.array(['CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'KK', 'LL',
                                                         'MM', 'NN', 'PP', 'QQ', 'RR', 'SS', 'TT', 'VV', 'WW',
                                                         'YY'])
        self.assertTrue(reverse_map_check.all())

    def test_class_alphabet(self):
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=IUPACProtein())
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'-', '.', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '-': 20, '.': 20, '*': 20})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_class_alphabet_with_gap1(self):
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=Gapped(IUPACProtein(), gap_char='-'))
        self.assertEqual(size, 21)
        self.assertEqual(gap_characters, {'.', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '-': 20, '.': 21, '*': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-'])
        self.assertTrue(reverse_map_check.all())

    def test_class_alphabet_with_gap2(self):
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=Gapped(IUPACProtein(), gap_char='.'))
        self.assertEqual(size, 21)
        self.assertEqual(gap_characters, {'-', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '.': 20, '-': 21, '*': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '.'])
        self.assertTrue(reverse_map_check.all())

    def test_class_alphabet_with_gap3(self):
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=Gapped(IUPACProtein(), gap_char='*'))
        self.assertEqual(size, 21)
        self.assertEqual(gap_characters, {'-', '.'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '*': 20, '-': 21, '.': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '*'])
        self.assertTrue(reverse_map_check.all())

    def test_class_alphabet_with_skip_letter(self):
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=IUPACProtein(), skip_letters=['B'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'-', '.', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '-': 20, '.': 20, '*': 20, 'B': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_class_alphabet_with_skip_letters(self):
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=IUPACProtein(),
                                                                       skip_letters=['B', 'J'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'-', '.', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '-': 20, '.': 20, '*': 20, 'B': 21, 'J': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_class_alphabet_with_gap_skip_letter(self):
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=IUPACProtein(), skip_letters=['-'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'.', '*'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '.': 20, '*': 20, '-': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_class_alphabet_with_gap_skip_letters(self):
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=IUPACProtein(),
                                                                       skip_letters=['.', '*'])
        self.assertEqual(size, 20)
        self.assertEqual(gap_characters, {'-'})
        self.assertEqual(mapping, {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                                   'Y': 19, '-': 20, '.': 21, '*': 21})
        reverse_map_check = reverse_mapping == np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_class_alphabet_skip_letter_overlap(self):
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=IUPACProtein(), skip_letters=['A'])
        self.assertEqual(size, 19)
        self.assertEqual(gap_characters, {'-', '.', '*'})
        self.assertEqual(mapping, {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'K': 7, 'L': 8,
                                   'M': 9, 'N': 10, 'P': 11, 'Q': 12, 'R': 13, 'S': 14, 'T': 15, 'V': 16, 'W': 17,
                                   'Y': 18, '-': 19, '.': 19, '*': 19, 'A': 20})
        reverse_map_check = reverse_mapping == np.array(['C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                                                         'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        self.assertTrue(reverse_map_check.all())

    def test_three_letter_class_alphabet(self):
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=ThreeLetterProtein())
        self.assertEqual(size, 24)
        self.assertEqual(gap_characters, {'---', '...', '***'})
        self.assertEqual(mapping, {'Ala': 0, 'Asx': 1, 'Cys': 2, 'Asp': 3, 'Glu': 4, 'Phe': 5, 'Gly': 6, 'His': 7,
                                   'Ile': 8, 'Lys': 9, 'Leu': 10, 'Met': 11, 'Asn': 12, 'Pro': 13, 'Gln': 14,
                                   'Arg': 15, 'Ser': 16, 'Thr': 17, 'Sec': 18, 'Val': 19, 'Trp': 20, 'Xaa': 21,
                                   'Tyr': 22, 'Glx': 23, '---': 24, '...': 24, '***': 24})
        reverse_map_check = reverse_mapping == np.array(['Ala', 'Asx', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His',
                                                         'Ile', 'Lys', 'Leu', 'Met', 'Asn', 'Pro', 'Gln', 'Arg',
                                                         'Ser', 'Thr', 'Sec', 'Val', 'Trp', 'Xaa', 'Tyr', 'Glx'])
        self.assertTrue(reverse_map_check.all())

    def test_three_letter_class_alphabet_with_gap1(self):
        alphabet = ThreeLetterProtein()
        alphabet.letters.append('---')
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet)
        self.assertEqual(size, 25)
        self.assertEqual(gap_characters, {'...', '***'})
        self.assertEqual(mapping, {'Ala': 0, 'Asx': 1, 'Cys': 2, 'Asp': 3, 'Glu': 4, 'Phe': 5, 'Gly': 6, 'His': 7,
                                   'Ile': 8, 'Lys': 9, 'Leu': 10, 'Met': 11, 'Asn': 12, 'Pro': 13, 'Gln': 14,
                                   'Arg': 15, 'Ser': 16, 'Thr': 17, 'Sec': 18, 'Val': 19, 'Trp': 20, 'Xaa': 21,
                                   'Tyr': 22, 'Glx': 23, '---': 24, '...': 25, '***': 25})
        reverse_map_check = reverse_mapping == np.array(['Ala', 'Asx', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His',
                                                         'Ile', 'Lys', 'Leu', 'Met', 'Asn', 'Pro', 'Gln', 'Arg',
                                                         'Ser', 'Thr', 'Sec', 'Val', 'Trp', 'Xaa', 'Tyr', 'Glx',
                                                         '---'])
        self.assertTrue(reverse_map_check.all())
        del(alphabet.letters[-1])

    def test_three_letter_class_alphabet_with_gap2(self):
        alphabet = ThreeLetterProtein()
        alphabet.letters.append('...')
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet)
        self.assertEqual(size, 25)
        self.assertEqual(gap_characters, {'---', '***'})
        self.assertEqual(mapping, {'Ala': 0, 'Asx': 1, 'Cys': 2, 'Asp': 3, 'Glu': 4, 'Phe': 5, 'Gly': 6, 'His': 7,
                                   'Ile': 8, 'Lys': 9, 'Leu': 10, 'Met': 11, 'Asn': 12, 'Pro': 13, 'Gln': 14,
                                   'Arg': 15, 'Ser': 16, 'Thr': 17, 'Sec': 18, 'Val': 19, 'Trp': 20, 'Xaa': 21,
                                   'Tyr': 22, 'Glx': 23, '...': 24, '---': 25, '***': 25})
        reverse_map_check = reverse_mapping == np.array(['Ala', 'Asx', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His',
                                                         'Ile', 'Lys', 'Leu', 'Met', 'Asn', 'Pro', 'Gln', 'Arg',
                                                         'Ser', 'Thr', 'Sec', 'Val', 'Trp', 'Xaa', 'Tyr', 'Glx',
                                                         '...'])
        self.assertTrue(reverse_map_check.all())
        del (alphabet.letters[-1])

    def test_three_letter_class_alphabet_with_gap3(self):
        alphabet = ThreeLetterProtein()
        alphabet.letters.append('***')
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=alphabet)
        self.assertEqual(size, 25)
        self.assertEqual(gap_characters, {'---', '...'})
        self.assertEqual(mapping, {'Ala': 0, 'Asx': 1, 'Cys': 2, 'Asp': 3, 'Glu': 4, 'Phe': 5, 'Gly': 6, 'His': 7,
                                   'Ile': 8, 'Lys': 9, 'Leu': 10, 'Met': 11, 'Asn': 12, 'Pro': 13, 'Gln': 14,
                                   'Arg': 15, 'Ser': 16, 'Thr': 17, 'Sec': 18, 'Val': 19, 'Trp': 20, 'Xaa': 21,
                                   'Tyr': 22, 'Glx': 23, '***': 24, '---': 25, '...': 25})
        reverse_map_check = reverse_mapping == np.array(['Ala', 'Asx', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His',
                                                         'Ile', 'Lys', 'Leu', 'Met', 'Asn', 'Pro', 'Gln', 'Arg',
                                                         'Ser', 'Thr', 'Sec', 'Val', 'Trp', 'Xaa', 'Tyr', 'Glx',
                                                         '***'])
        self.assertTrue(reverse_map_check.all())
        del (alphabet.letters[-1])

    def test_three_letter_class_alphabet_with_skip_letter1(self):
        with self.assertRaises(ValueError):
            build_mapping(alphabet=ThreeLetterProtein(), skip_letters=['B'])

    def test_three_letter_class_alphabet_with_skip_letter2(self):
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=ThreeLetterProtein(),
                                                                       skip_letters=['BBB'])
        self.assertEqual(size, 24)
        self.assertEqual(gap_characters, {'---', '...', '***'})
        self.assertEqual(mapping, {'Ala': 0, 'Asx': 1, 'Cys': 2, 'Asp': 3, 'Glu': 4, 'Phe': 5, 'Gly': 6, 'His': 7,
                                   'Ile': 8, 'Lys': 9, 'Leu': 10, 'Met': 11, 'Asn': 12, 'Pro': 13, 'Gln': 14,
                                   'Arg': 15, 'Ser': 16, 'Thr': 17, 'Sec': 18, 'Val': 19, 'Trp': 20, 'Xaa': 21,
                                   'Tyr': 22, 'Glx': 23, '---': 24, '...': 24, '***': 24, 'BBB': 25})
        reverse_map_check = reverse_mapping == np.array(['Ala', 'Asx', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His',
                                                         'Ile', 'Lys', 'Leu', 'Met', 'Asn', 'Pro', 'Gln', 'Arg',
                                                         'Ser', 'Thr', 'Sec', 'Val', 'Trp', 'Xaa', 'Tyr', 'Glx'])
        self.assertTrue(reverse_map_check.all())

    def test_three_letter_class_alphabet_with_skip_letters1(self):
        with self.assertRaises(ValueError):
            build_mapping(alphabet=ThreeLetterProtein(), skip_letters=['B', 'J'])

    def test_three_letter_class_alphabet_with_skip_letters2(self):
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=ThreeLetterProtein(),
                                                                       skip_letters=['BBB', 'JJJ'])
        self.assertEqual(size, 24)
        self.assertEqual(gap_characters, {'---', '...', '***'})
        self.assertEqual(mapping, {'Ala': 0, 'Asx': 1, 'Cys': 2, 'Asp': 3, 'Glu': 4, 'Phe': 5, 'Gly': 6, 'His': 7,
                                   'Ile': 8, 'Lys': 9, 'Leu': 10, 'Met': 11, 'Asn': 12, 'Pro': 13, 'Gln': 14,
                                   'Arg': 15, 'Ser': 16, 'Thr': 17, 'Sec': 18, 'Val': 19, 'Trp': 20, 'Xaa': 21,
                                   'Tyr': 22, 'Glx': 23, '---': 24, '...': 24, '***': 24, 'BBB': 25, 'JJJ': 25})
        reverse_map_check = reverse_mapping == np.array(['Ala', 'Asx', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His',
                                                         'Ile', 'Lys', 'Leu', 'Met', 'Asn', 'Pro', 'Gln', 'Arg',
                                                         'Ser', 'Thr', 'Sec', 'Val', 'Trp', 'Xaa', 'Tyr', 'Glx'])
        self.assertTrue(reverse_map_check.all())

    def test_three_letter_class_alphabet_with_gap_skip_letter1(self):
        with self.assertRaises(ValueError):
            build_mapping(alphabet=ThreeLetterProtein(), skip_letters=['-'])

    def test_three_letter_class_alphabet_with_gap_skip_letter2(self):
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=ThreeLetterProtein(),
                                                                       skip_letters=['---'])
        self.assertEqual(size, 24)
        self.assertEqual(gap_characters, {'...', '***'})
        self.assertEqual(mapping, {'Ala': 0, 'Asx': 1, 'Cys': 2, 'Asp': 3, 'Glu': 4, 'Phe': 5, 'Gly': 6, 'His': 7,
                                   'Ile': 8, 'Lys': 9, 'Leu': 10, 'Met': 11, 'Asn': 12, 'Pro': 13, 'Gln': 14,
                                   'Arg': 15, 'Ser': 16, 'Thr': 17, 'Sec': 18, 'Val': 19, 'Trp': 20, 'Xaa': 21,
                                   'Tyr': 22, 'Glx': 23, '...': 24, '***': 24, '---': 25})
        reverse_map_check = reverse_mapping == np.array(['Ala', 'Asx', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His',
                                                         'Ile', 'Lys', 'Leu', 'Met', 'Asn', 'Pro', 'Gln', 'Arg',
                                                         'Ser', 'Thr', 'Sec', 'Val', 'Trp', 'Xaa', 'Tyr', 'Glx'])
        self.assertTrue(reverse_map_check.all())

    def test_three_letter_class_alphabet_with_gap_skip_letters1(self):
        with self.assertRaises(ValueError):
            build_mapping(alphabet=ThreeLetterProtein(), skip_letters=['.', '*'])

    def test_three_letter_class_alphabet_with_gap_skip_letters2(self):
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=ThreeLetterProtein(),
                                                                       skip_letters=['...', '***'])
        self.assertEqual(size, 24)
        self.assertEqual(gap_characters, {'---'})
        self.assertEqual(mapping, {'Ala': 0, 'Asx': 1, 'Cys': 2, 'Asp': 3, 'Glu': 4, 'Phe': 5, 'Gly': 6, 'His': 7,
                                   'Ile': 8, 'Lys': 9, 'Leu': 10, 'Met': 11, 'Asn': 12, 'Pro': 13, 'Gln': 14,
                                   'Arg': 15, 'Ser': 16, 'Thr': 17, 'Sec': 18, 'Val': 19, 'Trp': 20, 'Xaa': 21,
                                   'Tyr': 22, 'Glx': 23, '---': 24, '...': 25, '***': 25})
        reverse_map_check = reverse_mapping == np.array(['Ala', 'Asx', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His',
                                                         'Ile', 'Lys', 'Leu', 'Met', 'Asn', 'Pro', 'Gln', 'Arg',
                                                         'Ser', 'Thr', 'Sec', 'Val', 'Trp', 'Xaa', 'Tyr', 'Glx'])
        self.assertTrue(reverse_map_check.all())

    def test_three_letter_class_alphabet_skip_letter_overlap1(self):
        with self.assertRaises(ValueError):
            build_mapping(alphabet=ThreeLetterProtein(), skip_letters=['A'])

    def test_three_letter_class_alphabet_skip_letter_overlap2(self):
        size, gap_characters, mapping, reverse_mapping = build_mapping(alphabet=ThreeLetterProtein(),
                                                                       skip_letters=['Ala'])
        self.assertEqual(size, 23)
        self.assertEqual(gap_characters, {'---', '...', '***'})
        self.assertEqual(mapping, {'Asx': 0, 'Cys': 1, 'Asp': 2, 'Glu': 3, 'Phe': 4, 'Gly': 5, 'His': 6,
                                   'Ile': 7, 'Lys': 8, 'Leu': 9, 'Met': 10, 'Asn': 11, 'Pro': 12, 'Gln': 13,
                                   'Arg': 14, 'Ser': 15, 'Thr': 16, 'Sec': 17, 'Val': 18, 'Trp': 19, 'Xaa': 20,
                                   'Tyr': 21, 'Glx': 22, '---': 23, '...': 23, '***': 23, 'Ala': 24})
        reverse_map_check = reverse_mapping == np.array(['Asx', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His', 'Ile',
                                                         'Lys', 'Leu', 'Met', 'Asn', 'Pro', 'Gln', 'Arg', 'Ser',
                                                         'Thr', 'Sec', 'Val', 'Trp', 'Xaa', 'Tyr', 'Glx'])
        self.assertTrue(reverse_map_check.all())

# class TestUtils(TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.expected_gap_chars = {'-', '.', '*'}
#         cls.alphabet = ExtendedIUPACProtein()
#         cls.alphabet_str = cls.alphabet.letters
#         cls.alphabet_list = [char for char in cls.alphabet_str]
#
#     def test2_convert_seq_to_numeric(self):
#         size, gap_chars, mapping, reverse = build_mapping(alphabet=self.alphabet_str)
#         query_seq_7hvp = 'PQITLWQRPLVTIRIGGQLKEALLDTGADDTVLE--EMNL--PGKWK----PKMIGGIGGFIKVRQYDQIPVEI-GHKAIGTV---LVGPTP'\
#                          'VNIIGRNLLTQIG-TLNF'
#         expected_array = np.array([12, 13, 7, 16, 9, 18, 13, 14, 12, 9, 17, 16, 7, 14, 7, 5, 5, 13, 9, 8, 3, 0, 9, 9, 2,
#                                    16, 5, 0, 2, 2, 16, 17, 9, 3, 26, 26, 3, 10, 11, 9, 26, 26, 12, 5, 8, 18, 8, 26, 26,
#                                    26, 26, 12, 8, 10, 7, 5, 5, 7, 5, 5, 4, 7, 8, 17, 14, 13, 19, 2, 13, 7, 12, 17, 3, 7,
#                                    26, 5, 6, 8, 0, 7, 5, 16, 17, 26, 26, 26, 9, 17, 5, 12, 16, 12, 17, 11, 7, 7, 5, 14,
#                                    11, 9, 9, 16, 13, 7, 5, 26, 16, 9, 11, 4])
#         numeric_seq_7hvp = convert_seq_to_numeric(seq=query_seq_7hvp, mapping=mapping)
#         diff_in_conversion = expected_array - numeric_seq_7hvp
#         self.assertEqual(numeric_seq_7hvp.shape, expected_array.shape)
#         self.assertFalse(np.any(diff_in_conversion))
#
#     def evaluate_compute_rank_and_coverage(self, seq_length, scores, pos_size, rank_type):
#         if rank_type not in ['min', 'max']:
#             with self.assertRaises(ValueError):
#                 compute_rank_and_coverage(seq_length=seq_length, scores=scores, pos_size=pos_size, rank_type=rank_type)
#         elif len(scores.shape) != pos_size:
#             with self.assertRaises(ValueError):
#                 compute_rank_and_coverage(seq_length=seq_length, scores=scores, pos_size=pos_size, rank_type=rank_type)
#         else:
#             rank, coverage = compute_rank_and_coverage(seq_length=seq_length, scores=scores, pos_size=pos_size,
#                                                        rank_type=rank_type)
#             unique_scores = np.unique(scores)
#             unique_rank = np.unique(rank)
#             unique_coverage = np.unique(coverage)
#             self.assertEqual(unique_scores.shape, unique_rank.shape)
#             self.assertEqual(unique_scores.shape, unique_coverage.shape)
#             min_score = np.min(scores)
#             min_rank = np.min(rank)
#             min_coverage = np.min(coverage)
#             max_coverage = np.max(coverage)
#             max_score = np.max(scores)
#             max_rank = np.max(rank)
#             min_mask = scores == min_score
#             max_mask = scores == max_score
#             if rank_type == 'min':
#                 rank_mask = rank == min_rank
#                 rank_mask2 = rank == max_rank
#                 cov_mask = coverage == min_coverage
#                 cov_mask2 = coverage == max_coverage
#             else:
#                 rank_mask = rank == max_rank
#                 rank_mask2 = rank == min_rank
#                 cov_mask = coverage == max_coverage
#                 cov_mask2 = coverage == min_coverage
#             diff_min_ranks = min_mask ^ rank_mask
#             self.assertFalse(diff_min_ranks.any())
#             diff_min_cov = min_mask ^ cov_mask
#             self.assertFalse(diff_min_cov.any())
#             diff_max_ranks = max_mask ^ rank_mask2
#             self.assertFalse(diff_max_ranks.any())
#             diff_max_cov = max_mask ^ cov_mask2
#             self.assertFalse(diff_max_cov.any())
#
#     def test3a_compute_rank_and_coverage(self):
#         seq_len = 100
#         scores = np.random.rand(seq_len, seq_len)
#         scores[np.tril_indices(seq_len, 1)] = 0
#         scores += scores.T
#         self.evaluate_compute_rank_and_coverage(seq_length=seq_len, scores=scores, pos_size=2, rank_type='middle')
#
#     def test3b_compute_rank_and_coverage(self):
#         seq_len = 100
#         scores = np.random.rand(seq_len, seq_len)
#         scores[np.tril_indices(seq_len, 1)] = 0
#         scores += scores.T
#         self.evaluate_compute_rank_and_coverage(seq_length=seq_len, scores=scores, pos_size=3, rank_type='min')
#
#     def test3c_compute_rank_and_coverage(self):
#         seq_len = 100
#         scores = np.random.rand(seq_len)
#         self.evaluate_compute_rank_and_coverage(seq_length=seq_len, scores=scores, pos_size=1, rank_type='min')
#
#     def test3d_compute_rank_and_coverage(self):
#         seq_len = 100
#         scores = np.random.rand(seq_len)
#         self.evaluate_compute_rank_and_coverage(seq_length=seq_len, scores=scores, pos_size=1, rank_type='max')
#
#     def test3e_compute_rank_and_coverage(self):
#         seq_len = 100
#         scores = np.random.rand(seq_len, seq_len)
#         scores[np.tril_indices(seq_len, 1)] = 0
#         scores += scores.T
#         self.evaluate_compute_rank_and_coverage(seq_length=seq_len, scores=scores, pos_size=1, rank_type='min')
#
#     def test3f_compute_rank_and_coverage(self):
#         seq_len = 100
#         scores = np.random.rand(seq_len, seq_len)
#         scores[np.tril_indices(seq_len, 1)] = 0
#         scores += scores.T
#         self.evaluate_compute_rank_and_coverage(seq_length=seq_len, scores=scores, pos_size=1, rank_type='max')


if __name__ == '__main__':
    unittest.main()
