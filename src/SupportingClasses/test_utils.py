"""
Created on June 17, 2019

@author: daniel
"""
import unittest
import numpy as np
from unittest import TestCase
from Bio.Alphabet import Gapped
from Bio.Alphabet.IUPAC import ExtendedIUPACProtein
from utils import build_mapping, convert_seq_to_numeric, compute_rank_and_coverage


class TestUtils(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.expected_gap_chars = {'-', '.', '*'}
        cls.alphabet = ExtendedIUPACProtein()
        cls.alphabet_str = cls.alphabet.letters
        cls.alphabet_list = [char for char in cls.alphabet_str]

    def evaluate_build_mapping(self, alphabet, alphabet_elements, alphabet_size, expected_gaps):
        size, gap_chars, mapping, reverse = build_mapping(alphabet=alphabet)
        self.assertEqual(size, alphabet_size)
        self.assertEqual(len(gap_chars), len(expected_gaps))
        self.assertEqual(gap_chars, expected_gaps)
        for char in gap_chars:
            self.assertEqual(mapping[char], size)
        self.assertEqual(len(mapping), size + len(gap_chars))
        self.assertEqual(len(reverse), size)
        for i in range(size):
            self.assertEqual(mapping[alphabet_elements[i]], i)
            self.assertEqual(reverse[i], alphabet_elements[i])

    def test1a_build_mapping_from_str(self):
        self.evaluate_build_mapping(alphabet=self.alphabet_str, alphabet_elements=self.alphabet_str,
                                    alphabet_size=len(self.alphabet_str), expected_gaps=self.expected_gap_chars)

    def test1b_build_mapping_from_list(self):
        self.evaluate_build_mapping(alphabet=self.alphabet_list, alphabet_elements=self.alphabet_list,
                                    alphabet_size=len(self.alphabet_list), expected_gaps=self.expected_gap_chars)

    def test1c_build_mapping_from_alphabet(self):
        alphabet = ExtendedIUPACProtein()
        self.evaluate_build_mapping(alphabet=alphabet, alphabet_elements=alphabet.letters,
                                    alphabet_size=len(alphabet.letters), expected_gaps=self.expected_gap_chars)

    def test1d_build_mapping_from_gapped_alphabet(self):
        alphabet = Gapped(ExtendedIUPACProtein())
        self.evaluate_build_mapping(alphabet=alphabet, alphabet_elements=alphabet.letters,
                                    alphabet_size=len(alphabet.letters), expected_gaps={'.', '*'})

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

    def evaluate_compute_rank_and_coverage(self, seq_length, scores, pos_size, rank_type):
        if rank_type not in ['min', 'max']:
            with self.assertRaises(ValueError):
                compute_rank_and_coverage(seq_length=seq_length, scores=scores, pos_size=pos_size, rank_type=rank_type)
        elif len(scores.shape) != pos_size:
            with self.assertRaises(ValueError):
                compute_rank_and_coverage(seq_length=seq_length, scores=scores, pos_size=pos_size, rank_type=rank_type)
        else:
            rank, coverage = compute_rank_and_coverage(seq_length=seq_length, scores=scores, pos_size=pos_size,
                                                       rank_type=rank_type)
            unique_scores = np.unique(scores)
            unique_rank = np.unique(rank)
            unique_coverage = np.unique(coverage)
            self.assertEqual(unique_scores.shape, unique_rank.shape)
            self.assertEqual(unique_scores.shape, unique_coverage.shape)
            min_score = np.min(scores)
            min_rank = np.min(rank)
            min_coverage = np.min(coverage)
            max_coverage = np.max(coverage)
            max_score = np.max(scores)
            max_rank = np.max(rank)
            min_mask = scores == min_score
            max_mask = scores == max_score
            if rank_type == 'min':
                rank_mask = rank == min_rank
                rank_mask2 = rank == max_rank
                cov_mask = coverage == min_coverage
                cov_mask2 = coverage == max_coverage
            else:
                rank_mask = rank == max_rank
                rank_mask2 = rank == min_rank
                cov_mask = coverage == max_coverage
                cov_mask2 = coverage == min_coverage
            diff_min_ranks = min_mask ^ rank_mask
            self.assertFalse(diff_min_ranks.any())
            diff_min_cov = min_mask ^ cov_mask
            self.assertFalse(diff_min_cov.any())
            diff_max_ranks = max_mask ^ rank_mask2
            self.assertFalse(diff_max_ranks.any())
            diff_max_cov = max_mask ^ cov_mask2
            self.assertFalse(diff_max_cov.any())

    def test3a_compute_rank_and_coverage(self):
        seq_len = 100
        scores = np.random.rand(seq_len, seq_len)
        scores[np.tril_indices(seq_len, 1)] = 0
        scores += scores.T
        self.evaluate_compute_rank_and_coverage(seq_length=seq_len, scores=scores, pos_size=2, rank_type='middle')

    def test3b_compute_rank_and_coverage(self):
        seq_len = 100
        scores = np.random.rand(seq_len, seq_len)
        scores[np.tril_indices(seq_len, 1)] = 0
        scores += scores.T
        self.evaluate_compute_rank_and_coverage(seq_length=seq_len, scores=scores, pos_size=3, rank_type='min')

    def test3c_compute_rank_and_coverage(self):
        seq_len = 100
        scores = np.random.rand(seq_len)
        self.evaluate_compute_rank_and_coverage(seq_length=seq_len, scores=scores, pos_size=1, rank_type='min')

    def test3d_compute_rank_and_coverage(self):
        seq_len = 100
        scores = np.random.rand(seq_len)
        self.evaluate_compute_rank_and_coverage(seq_length=seq_len, scores=scores, pos_size=1, rank_type='max')

    def test3e_compute_rank_and_coverage(self):
        seq_len = 100
        scores = np.random.rand(seq_len, seq_len)
        scores[np.tril_indices(seq_len, 1)] = 0
        scores += scores.T
        self.evaluate_compute_rank_and_coverage(seq_length=seq_len, scores=scores, pos_size=1, rank_type='min')

    def test3f_compute_rank_and_coverage(self):
        seq_len = 100
        scores = np.random.rand(seq_len, seq_len)
        scores[np.tril_indices(seq_len, 1)] = 0
        scores += scores.T
        self.evaluate_compute_rank_and_coverage(seq_length=seq_len, scores=scores, pos_size=1, rank_type='max')


if __name__ == '__main__':
    unittest.main()
