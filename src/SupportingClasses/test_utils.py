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


class TestConvertSeqToNumeric(TestCase):

    def test_only_alphabet_characters(self):
        sequence = 'ACDEFGHIKLMNPQRSTVWYYWVTSRQPNMLKIHGFEDCAAYCWDVETFSGRHQIPKNLM'
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19, 18, 17, 16, 15, 14, 13,
                    12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 19, 1, 18, 2, 17, 3, 16, 4, 15, 5, 14, 6, 13, 7, 12, 8,
                    11, 9, 10]
        _, _, mapping, _ = build_mapping(alphabet=IUPACProtein())
        numeric = convert_seq_to_numeric(seq=sequence, mapping=mapping)
        self.assertFalse((numeric - expected).any())

    def test_alphabet_characters_and_gaps1(self):
        sequence = 'ACDEFGHIKLMNPQRSTVWY--YWVTSRQPNMLKIHGFEDCAA-CYDWEVFTGSHRIQKPLNM'
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 19, 18, 17, 16, 15,
                    14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 20, 1, 19, 2, 18, 3, 17, 4, 16, 5, 15, 6, 14,
                    7, 13, 8, 12, 9, 11, 10]
        _, _, mapping, _ = build_mapping(alphabet=IUPACProtein())
        numeric = convert_seq_to_numeric(seq=sequence, mapping=mapping)
        self.assertFalse((numeric - expected).any())

    def test_alphabet_characters_and_gaps2(self):
        sequence = 'ACDEFGHIKLMNPQRSTVWY..YWVTSRQPNMLKIHGFEDCAA.CYDWEVFTGSHRIQKPLNM'
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 19, 18, 17, 16, 15,
                    14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 20, 1, 19, 2, 18, 3, 17, 4, 16, 5, 15, 6, 14,
                    7, 13, 8, 12, 9, 11, 10]
        _, _, mapping, _ = build_mapping(alphabet=IUPACProtein())
        numeric = convert_seq_to_numeric(seq=sequence, mapping=mapping)
        self.assertFalse((numeric - expected).any())

    def test_alphabet_characters_and_gaps3(self):
        sequence = 'ACDEFGHIKLMNPQRSTVWY**YWVTSRQPNMLKIHGFEDCAA*CYDWEVFTGSHRIQKPLNM'
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 19, 18, 17, 16, 15,
                    14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 20, 1, 19, 2, 18, 3, 17, 4, 16, 5, 15, 6, 14,
                    7, 13, 8, 12, 9, 11, 10]
        _, _, mapping, _ = build_mapping(alphabet=IUPACProtein())
        numeric = convert_seq_to_numeric(seq=sequence, mapping=mapping)
        self.assertFalse((numeric - expected).any())

    def test_alphabet_with_skip_letter(self):
        sequence = 'ABCDEFGHIKLMNPQRSTVWYYWVTSRQPNMLKIHGFEDCBAAYBWCVDTESFRGQHPINKML'
        expected = [0, 21, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19, 18, 17, 16, 15,
                    14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 21, 0, 0, 19, 21, 18, 1, 17, 2, 16, 3, 15, 4, 14,
                    5, 13, 6, 12, 7, 11, 8, 10, 9]
        _, _, mapping, _ = build_mapping(alphabet=IUPACProtein(), skip_letters=['B'])
        numeric = convert_seq_to_numeric(seq=sequence, mapping=mapping)
        self.assertFalse((numeric - expected).any())

    def test_alphabet_with_skip_letters(self):
        sequence = 'ABCDEFGHIJKLMNPQRSTVWYYWVTSRQPNMLKJIHGFEDCBAAYBWCVDTESFRGQHPINJMKL'
        expected = [0, 21, 1, 2, 3, 4, 5, 6, 7, 21, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19, 18, 17, 16, 15,
                    14, 13, 12, 11, 10, 9, 8, 21, 7, 6, 5, 4, 3, 2, 1, 21, 0, 0, 19, 21, 18, 1, 17, 2, 16, 3, 15, 4, 14,
                    5, 13, 6, 12, 7, 11, 21, 10, 8, 9]
        _, _, mapping, _ = build_mapping(alphabet=IUPACProtein(), skip_letters=['B', 'J'])
        numeric = convert_seq_to_numeric(seq=sequence, mapping=mapping)
        self.assertFalse((numeric - expected).any())

    def test_alphabet_with_skip_letters_and_gaps(self):
        sequence = 'ABCDEFGHIJKLMNPQRSTVWY--YWVTSRQPNMLKJIHGFEDCBAAYBWCVDTESFRGQHPINJMKL-'
        expected = [0, 21, 1, 2, 3, 4, 5, 6, 7, 21, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 19, 18, 17,
                    16, 15, 14, 13, 12, 11, 10, 9, 8, 21, 7, 6, 5, 4, 3, 2, 1, 21, 0, 0, 19, 21, 18, 1, 17, 2, 16, 3,
                    15, 4, 14, 5, 13, 6, 12, 7, 11, 21, 10, 8, 9, 20]
        _, _, mapping, _ = build_mapping(alphabet=IUPACProtein(), skip_letters=['B', 'J', '.', '*'])
        numeric = convert_seq_to_numeric(seq=sequence, mapping=mapping)
        self.assertFalse((numeric - expected).any())

    def test_missing_character_letter(self):
        sequence = 'ABCDEFGHIKLMNPQRSTVWYYWVTSRQPNMLKIHGFEDCBAAYBWCVDTESFRGQHPINKML'
        _, _, mapping, _ = build_mapping(alphabet=IUPACProtein())
        with self.assertRaises(KeyError):
            convert_seq_to_numeric(seq=sequence, mapping=mapping)

    def test_missing_characters(self):
        sequence = 'ABCDEFGHIJKLMNPQRSTVWYYWVTSRQPNMLKJIHGFEDCBAAYBWCVDTESFRGQHPINJMKL'
        _, _, mapping, _ = build_mapping(alphabet=IUPACProtein())
        with self.assertRaises(KeyError):
            convert_seq_to_numeric(seq=sequence, mapping=mapping)


class TestComputeRankAndCoverage(TestCase):

    def test_position_size_1_rank_type_min(self):
        scores = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ranks, coverages = compute_rank_and_coverage(seq_length=10, scores=scores, pos_size=1, rank_type='min')
        expected_ranks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_1_rank_type_min_disordered(self):
        scores = np.array([0, 9, 1, 8, 2, 7, 3, 6, 4, 5])
        ranks, coverages = compute_rank_and_coverage(seq_length=10, scores=scores, pos_size=1, rank_type='min')
        expected_ranks = np.array([1, 10, 2, 9, 3, 8, 4, 7, 5, 6])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([0.1, 1.0, 0.2, 0.9, 0.3, 0.8, 0.4, 0.7, 0.5, 0.6])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_1_rank_type_min_ties1(self):
        scores = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ranks, coverages = compute_rank_and_coverage(seq_length=11, scores=scores, pos_size=1, rank_type='min')
        expected_ranks = np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([2 / 11.0, 2 / 11.0, 3 / 11.0, 4 / 11.0, 5 / 11.0, 6 / 11.0, 7 / 11.0, 8 / 11.0,
                                       9 / 11.0, 10 / 11.0, 1.0])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_1_rank_type_min_ties2(self):
        scores = np.array([0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9])
        ranks, coverages = compute_rank_and_coverage(seq_length=11, scores=scores, pos_size=1, rank_type='min')
        expected_ranks = np.array([1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([1 / 11.0, 2 / 11.0, 3 / 11.0, 4 / 11.0, 5 / 11.0, 7 / 11.0, 7 / 11.0, 8 / 11.0,
                                       9 / 11.0, 10 / 11.0, 1.0])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_1_rank_type_min_ties3(self):
        scores = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9])
        ranks, coverages = compute_rank_and_coverage(seq_length=11, scores=scores, pos_size=1, rank_type='min')
        expected_ranks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([1 / 11.0, 2 / 11.0, 3 / 11.0, 4 / 11.0, 5 / 11.0, 6 / 11.0, 7 / 11.0, 8 / 11.0,
                                       9 / 11.0, 1.0, 1.0])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_1_rank_type_max(self):
        scores = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ranks, coverages = compute_rank_and_coverage(seq_length=10, scores=scores, pos_size=1, rank_type='max')
        expected_ranks = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_1_rank_type_max_disordered(self):
        scores = np.array([0, 9, 1, 8, 2, 7, 3, 6, 4, 5])
        ranks, coverages = compute_rank_and_coverage(seq_length=10, scores=scores, pos_size=1, rank_type='max')
        expected_ranks = np.array([10, 1, 9, 2, 8, 3, 7, 4, 6, 5])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([1.0, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_1_rank_type_max_ties1(self):
        scores = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ranks, coverages = compute_rank_and_coverage(seq_length=11, scores=scores, pos_size=1, rank_type='max')
        expected_ranks = np.array([10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([1.0, 1.0, 9 / 11.0, 8 / 11.0, 7 / 11.0, 6 / 11.0, 5 / 11.0, 4 / 11.0, 3 / 11.0,
                                       2 / 11.0, 1 / 11.0])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_1_rank_type_max_ties2(self):
        scores = np.array([0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9])
        ranks, coverages = compute_rank_and_coverage(seq_length=11, scores=scores, pos_size=1, rank_type='max')
        expected_ranks = np.array([10, 9, 8, 7, 6, 5, 5, 4, 3, 2, 1])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([1.0, 10 / 11.0, 9 / 11.0, 8 / 11.0, 7 / 11.0, 6 / 11.0, 6 / 11.0, 4 / 11.0,
                                       3 / 11.0, 2 / 11.0, 1 / 11.0])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_1_rank_type_max_ties3(self):
        scores = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9])
        ranks, coverages = compute_rank_and_coverage(seq_length=11, scores=scores, pos_size=1, rank_type='max')
        expected_ranks = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([1.0, 10 / 11.0, 9 / 11.0, 8 / 11.0, 7 / 11.0, 6 / 11.0, 5 / 11.0, 4 / 11.0,
                                       3 / 11.0, 2 / 11.0, 2 / 11.0])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_2_rank_type_min(self):
        scores = np.array([[0, 0, 1, 2, 3],
                           [0, 0, 4, 5, 6],
                           [0, 0, 0, 7, 8],
                           [0, 0, 0, 0, 9],
                           [0, 0, 0, 0, 0]])
        ranks, coverages = compute_rank_and_coverage(seq_length=5, scores=scores, pos_size=2, rank_type='min')
        expected_ranks = np.array([[0, 1, 2, 3, 4],
                                   [0, 0, 5, 6, 7],
                                   [0, 0, 0, 8, 9],
                                   [0, 0, 0, 0, 10],
                                   [0, 0, 0, 0, 0]])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([[0.0, 0.1, 0.2, 0.3, 0.4],
                                       [0.0, 0.0, 0.5, 0.6, 0.7],
                                       [0.0, 0.0, 0.0, 0.8, 0.9],
                                       [0.0, 0.0, 0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_2_rank_type_min_disordered(self):
        scores = np.array([[0, 0, 9, 1, 8],
                           [0, 0, 2, 7, 3],
                           [0, 0, 0, 6, 4],
                           [0, 0, 0, 0, 5],
                           [0, 0, 0, 0, 0]])
        ranks, coverages = compute_rank_and_coverage(seq_length=5, scores=scores, pos_size=2, rank_type='min')
        expected_ranks = np.array([[0, 1, 10, 2, 9],
                                   [0, 0, 3, 8, 4],
                                   [0, 0, 0, 7, 5],
                                   [0, 0, 0, 0, 6],
                                   [0, 0, 0, 0, 0]])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([[0.0, 0.1, 1.0, 0.2, 0.9],
                                       [0.0, 0.0, 0.3, 0.8, 0.4],
                                       [0.0, 0.0, 0.0, 0.7, 0.5],
                                       [0.0, 0.0, 0.0, 0.0, 0.6],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_2_rank_type_min_ties1(self):
        scores = np.array([[0, 0, 0, 1, 2],
                           [0, 0, 3, 4, 5],
                           [0, 0, 0, 6, 7],
                           [0, 0, 0, 0, 8],
                           [0, 0, 0, 0, 0]])
        ranks, coverages = compute_rank_and_coverage(seq_length=5, scores=scores, pos_size=2, rank_type='min')
        expected_ranks = np.array([[0, 1, 1, 2, 3],
                                   [0, 0, 4, 5, 6],
                                   [0, 0, 0, 7, 8],
                                   [0, 0, 0, 0, 9],
                                   [0, 0, 0, 0, 0]])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([[0.0, 0.2, 0.2, 0.3, 0.4],
                                       [0.0, 0.0, 0.5, 0.6, 0.7],
                                       [0.0, 0.0, 0.0, 0.8, 0.9],
                                       [0.0, 0.0, 0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_2_rank_type_min_ties2(self):
        scores = np.array([[0, 0, 1, 2, 3],
                           [0, 0, 4, 5, 5],
                           [0, 0, 0, 6, 7],
                           [0, 0, 0, 0, 8],
                           [0, 0, 0, 0, 0]])
        ranks, coverages = compute_rank_and_coverage(seq_length=5, scores=scores, pos_size=2, rank_type='min')
        expected_ranks = np.array([[0, 1, 2, 3, 4],
                                   [0, 0, 5, 6, 6],
                                   [0, 0, 0, 7, 8],
                                   [0, 0, 0, 0, 9],
                                   [0, 0, 0, 0, 0]])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([[0.0, 0.1, 0.2, 0.3, 0.4],
                                       [0.0, 0.0, 0.5, 0.7, 0.7],
                                       [0.0, 0.0, 0.0, 0.8, 0.9],
                                       [0.0, 0.0, 0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_2_rank_type_min_ties3(self):
        scores = np.array([[0, 0, 1, 2, 3],
                           [0, 0, 4, 5, 6],
                           [0, 0, 0, 7, 8],
                           [0, 0, 0, 0, 8],
                           [0, 0, 0, 0, 0]])
        ranks, coverages = compute_rank_and_coverage(seq_length=5, scores=scores, pos_size=2, rank_type='min')
        expected_ranks = np.array([[0, 1, 2, 3, 4],
                                   [0, 0, 5, 6, 7],
                                   [0, 0, 0, 8, 9],
                                   [0, 0, 0, 0, 9],
                                   [0, 0, 0, 0, 0]])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([[0.0, 0.1, 0.2, 0.3, 0.4],
                                       [0.0, 0.0, 0.5, 0.6, 0.7],
                                       [0.0, 0.0, 0.0, 0.8, 1.0],
                                       [0.0, 0.0, 0.0, 0.0, 1.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_2_rank_type_max(self):
        scores = np.array([[0, 0, 1, 2, 3],
                           [0, 0, 4, 5, 6],
                           [0, 0, 0, 7, 8],
                           [0, 0, 0, 0, 9],
                           [0, 0, 0, 0, 0]])
        ranks, coverages = compute_rank_and_coverage(seq_length=5, scores=scores, pos_size=2, rank_type='max')
        expected_ranks = np.array([[0, 10, 9, 8, 7],
                                   [0, 0, 6, 5, 4],
                                   [0, 0, 0, 3, 2],
                                   [0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0]])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([[0.0, 1.0, 0.9, 0.8, 0.7],
                                       [0.0, 0.0, 0.6, 0.5, 0.4],
                                       [0.0, 0.0, 0.0, 0.3, 0.2],
                                       [0.0, 0.0, 0.0, 0.0, 0.1],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_2_rank_type_max_disordered(self):
        scores = np.array([[0, 0, 9, 1, 8],
                           [0, 0, 2, 7, 3],
                           [0, 0, 0, 6, 4],
                           [0, 0, 0, 0, 5],
                           [0, 0, 0, 0, 0]])
        ranks, coverages = compute_rank_and_coverage(seq_length=5, scores=scores, pos_size=2, rank_type='max')
        expected_ranks = np.array([[0, 10, 1, 9, 2],
                                   [0, 0, 8, 3, 7],
                                   [0, 0, 0, 4, 6],
                                   [0, 0, 0, 0, 5],
                                   [0, 0, 0, 0, 0]])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([[0.0, 1.0, 0.1, 0.9, 0.2],
                                       [0.0, 0.0, 0.8, 0.3, 0.7],
                                       [0.0, 0.0, 0.0, 0.4, 0.6],
                                       [0.0, 0.0, 0.0, 0.0, 0.5],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_2_rank_type_max_ties1(self):
        scores = np.array([[0, 0, 0, 1, 2],
                           [0, 0, 3, 4, 5],
                           [0, 0, 0, 6, 7],
                           [0, 0, 0, 0, 8],
                           [0, 0, 0, 0, 0]])
        ranks, coverages = compute_rank_and_coverage(seq_length=5, scores=scores, pos_size=2, rank_type='max')
        expected_ranks = np.array([[0, 9, 9, 8, 7],
                                   [0, 0, 6, 5, 4],
                                   [0, 0, 0, 3, 2],
                                   [0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0]])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([[0.0, 1.0, 1.0, 0.8, 0.7],
                                       [0.0, 0.0, 0.6, 0.5, 0.4],
                                       [0.0, 0.0, 0.0, 0.3, 0.2],
                                       [0.0, 0.0, 0.0, 0.0, 0.1],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_2_rank_type_max_ties2(self):
        scores = np.array([[0, 0, 1, 2, 3],
                           [0, 0, 4, 5, 5],
                           [0, 0, 0, 6, 7],
                           [0, 0, 0, 0, 8],
                           [0, 0, 0, 0, 0]])
        ranks, coverages = compute_rank_and_coverage(seq_length=5, scores=scores, pos_size=2, rank_type='max')
        expected_ranks = np.array([[0, 9, 8, 7, 6],
                                   [0, 0, 5, 4, 4],
                                   [0, 0, 0, 3, 2],
                                   [0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0]])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([[0.0, 1.0, 0.9, 0.8, 0.7],
                                       [0.0, 0.0, 0.6, 0.5, 0.5],
                                       [0.0, 0.0, 0.0, 0.3, 0.2],
                                       [0.0, 0.0, 0.0, 0.0, 0.1],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_size_2_rank_type_max_ties3(self):
        scores = np.array([[0, 0, 1, 2, 3],
                           [0, 0, 4, 5, 6],
                           [0, 0, 0, 7, 8],
                           [0, 0, 0, 0, 8],
                           [0, 0, 0, 0, 0]])
        ranks, coverages = compute_rank_and_coverage(seq_length=5, scores=scores, pos_size=2, rank_type='max')
        expected_ranks = np.array([[0, 9, 8, 7, 6],
                                   [0, 0, 5, 4, 3],
                                   [0, 0, 0, 2, 1],
                                   [0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0]])
        self.assertFalse((ranks - expected_ranks).any())
        expected_coverages = np.array([[0.0, 1.0, 0.9, 0.8, 0.7],
                                       [0.0, 0.0, 0.6, 0.5, 0.4],
                                       [0.0, 0.0, 0.0, 0.3, 0.2],
                                       [0.0, 0.0, 0.0, 0.0, 0.2],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertFalse((coverages - expected_coverages).any())

    def test_position_bad_rank1(self):
        scores = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        with self.assertRaises(ValueError):
            compute_rank_and_coverage(seq_length=10, scores=scores, pos_size=1, rank_type='med')

    def test_position_bad_rank2(self):
        scores = np.array([[0, 0, 1, 2, 3],
                           [0, 0, 4, 5, 6],
                           [0, 0, 0, 7, 8],
                           [0, 0, 0, 0, 9],
                           [0, 0, 0, 0, 0]])
        with self.assertRaises(ValueError):
            compute_rank_and_coverage(seq_length=5, scores=scores, pos_size=2, rank_type='med')

    def test_position_bad_bad_position_size1(self):
        scores = np.array([[[0, 0, 1, 2, 3],
                            [0, 0, 4, 5, 6],
                            [0, 0, 0, 7, 8],
                            [0, 0, 0, 0, 9],
                            [0, 0, 0, 0, 0]]])
        with self.assertRaises(ValueError):
            compute_rank_and_coverage(seq_length=5, scores=scores, pos_size=3, rank_type='min')

    def test_position_bad_position_size2(self):
        scores = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        with self.assertRaises(ValueError):
            compute_rank_and_coverage(seq_length=5, scores=scores, pos_size=2, rank_type='min')

    def test_position_bad_position_size3(self):
        scores = np.array([[0, 0, 1, 2, 3],
                           [0, 0, 4, 5, 6],
                           [0, 0, 0, 7, 8],
                           [0, 0, 0, 0, 9],
                           [0, 0, 0, 0, 0]])
        with self.assertRaises(ValueError):
            compute_rank_and_coverage(seq_length=25, scores=scores, pos_size=1, rank_type='min')


if __name__ == '__main__':
    unittest.main()
