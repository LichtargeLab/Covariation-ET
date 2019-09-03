"""
Created on June 17, 2019

@author: daniel
"""
import unittest
import numpy as np
from unittest import TestCase
from Bio.Alphabet import  Gapped
from Bio.Alphabet.IUPAC import ExtendedIUPACProtein
from utils import build_mapping, convert_seq_to_numeric, compute_rank_and_coverage


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

    def test3a_compute_rank_and_coverage(self):
        """
        Testing rank_type error.
        """
        scores = np.random.rand(100, 100)
        scores[np.tril_indices(100, 1)] = 0
        scores += scores.T
        with self.assertRaises(ValueError):
            compute_rank_and_coverage(seq_length=100, scores=scores, pos_size=1, rank_type='middle')

    def test3b_compute_rank_and_coverage(self):
        """
        Testing pos_size error.
        """
        scores = np.random.rand(100, 100)
        scores[np.tril_indices(100, 1)] = 0
        scores += scores.T
        with self.assertRaises(ValueError):
            compute_rank_and_coverage(seq_length=100, scores=scores, pos_size=3, rank_type='min')

    def test3c_compute_rank_and_coverage(self):
        """
        Testing single position min.
        """
        scores = np.random.rand(100)
        rank, coverage = compute_rank_and_coverage(seq_length=100, scores=scores, pos_size=1, rank_type='min')
        unique_scores = np.unique(scores)
        unique_rank = np.unique(rank)
        unique_covearge = np.unique(coverage)
        self.assertEqual(unique_scores.shape, unique_rank.shape)
        self.assertEqual(unique_scores.shape, unique_covearge.shape)
        min_score = np.min(scores)
        min_rank = np.min(rank)
        max_coverage = np.max(coverage)
        max_score = np.max(scores)
        max_rank = np.max(rank)
        min_coverage = np.min(coverage)
        min_mask = (scores == min_score) * 1
        rank_mask = (rank == min_rank) * 1
        diff_min_ranks = min_mask - rank_mask
        if diff_min_ranks.any():
            print(min_score)
            print(min_rank)
            print(np.sum(min_mask))
            print(np.sum(rank_mask))
        self.assertFalse(diff_min_ranks.any())
        cov_mask = coverage == min_coverage
        diff_min_cov = min_mask - cov_mask
        if diff_min_cov.any():
            print(min_score)
            print(min_coverage)
            print(np.sum(min_mask))
            print(np.sum(cov_mask))
        self.assertFalse(diff_min_cov.any())
        max_mask = scores == max_score
        rank_mask2 = rank == max_rank
        diff_max_ranks = max_mask ^ rank_mask2
        if diff_max_ranks.any():
            print(max_score)
            print(max_rank)
            print(np.sum(max_mask))
            print(np.sum(rank_mask2))
        self.assertFalse(diff_max_ranks.any())
        cov_mask2 = coverage == max_coverage
        diff_max_cov = max_mask ^ cov_mask2
        if diff_max_cov.any():
            print(max_score)
            print(max_coverage)
            print(np.sum(max_mask))
            print(np.sum(cov_mask2))
        self.assertFalse(diff_min_cov.any())

    def test3d_compute_rank_and_coverage(self):
        """
        Testing single positions max.
        """
        scores = np.random.rand(100)
        rank, coverage = compute_rank_and_coverage(seq_length=100, scores=scores, pos_size=1, rank_type='max')
        unique_scores = np.unique(scores)
        unique_rank = np.unique(rank)
        unique_covearge = np.unique(coverage)
        self.assertEqual(unique_scores.shape, unique_rank.shape)
        self.assertEqual(unique_scores.shape, unique_covearge.shape)
        max_score = np.max(scores)
        max_rank = np.max(rank)
        max_coverage = np.max(coverage)
        min_score = np.min(scores)
        min_rank = np.min(rank)
        min_coverage = np.min(coverage)
        min_mask = scores == min_score
        rank_mask = rank == max_rank
        cov_mask = coverage == max_coverage
        max_mask = scores == max_score
        rank_mask2 = rank == min_rank
        cov_mask2 = coverage == min_coverage
        diff_min_ranks = min_mask ^ rank_mask
        if diff_min_ranks.any():
            print(min_score)
            print(min_rank)
            print(np.sum(min_mask))
            print(np.sum(rank_mask))
        self.assertFalse(diff_min_ranks.any())
        diff_min_cov = min_mask ^ cov_mask
        if diff_min_cov.any():
            print(min_score)
            print(max_coverage)
            print(np.sum(min_mask))
            print(np.sum(cov_mask))
        self.assertFalse(diff_min_cov.any())
        diff_max_ranks = max_mask ^ rank_mask2
        if diff_max_ranks.any():
            print(max_score)
            print(max_rank)
            print(np.sum(max_mask))
            print(np.sum(rank_mask2))
        self.assertFalse(diff_max_ranks.any())
        diff_max_cov = max_mask ^ cov_mask2
        if diff_max_cov.any():
            print(max_score)
            print(max_coverage)
            print(np.sum(max_mask))
            print(np.sum(cov_mask2))
        self.assertFalse(diff_min_cov.any())

    def test3e_compute_rank_and_coverage(self):
        """
        Testing pair position min.
        """
        scores = np.random.rand(100, 100)
        scores[np.tril_indices(100, 1)] = 0
        scores += scores.T
        rank, coverage = compute_rank_and_coverage(seq_length=100, scores=scores, pos_size=2, rank_type='min')
        unique_scores = np.unique(scores[np.triu_indices(100, k=1)])
        unique_rank = np.unique(rank[np.triu_indices(100, k=1)])
        unique_covearge = np.unique(coverage[np.triu_indices(100, k=1)])
        self.assertEqual(unique_scores.shape, unique_rank.shape)
        self.assertEqual(unique_scores.shape, unique_covearge.shape)
        min_score = np.min(scores[np.triu_indices(100, k=1)])
        min_rank = np.min(rank[np.triu_indices(100, k=1)])
        max_coverage = np.max(coverage[np.triu_indices(100, k=1)])
        max_score = np.max(scores[np.triu_indices(100, k=1)])
        max_rank = np.max(rank[np.triu_indices(100, k=1)])
        min_coverage = np.min(coverage[np.triu_indices(100, k=1)])
        min_mask = np.triu((scores == min_score) * 1, k=1)
        rank_mask = np.triu((rank == min_rank) * 1, k=1)
        cov_mask = np.triu((coverage == min_coverage) * 1, k=1)
        max_mask = np.triu((scores == max_score) * 1, k=1)
        rank_mask2 = np.triu((rank == max_rank) * 1, k=1)
        cov_mask2 = np.triu((coverage == max_coverage) * 1, k=1)
        diff_min_ranks = min_mask - rank_mask
        if diff_min_ranks.any():
            print(min_score)
            print(min_rank)
            print(np.sum(min_mask))
            print(np.sum(rank_mask))
        self.assertFalse(diff_min_ranks.any())
        diff_min_cov = min_mask - cov_mask
        if diff_min_cov.any():
            print(min_score)
            print(min_coverage)
            print(np.sum(min_mask))
            print(np.sum(cov_mask))
        self.assertFalse(diff_min_cov.any())
        diff_max_ranks = max_mask ^ rank_mask2
        if diff_max_ranks.any():
            print(max_score)
            print(max_rank)
            print(np.sum(max_mask))
            print(np.sum(rank_mask2))
        self.assertFalse(diff_max_ranks.any())
        diff_max_cov = max_mask ^ cov_mask2
        if diff_max_cov.any():
            print(max_score)
            print(max_coverage)
            print(np.sum(max_mask))
            print(np.sum(cov_mask2))
        self.assertFalse(diff_min_cov.any())

    def test3f_compute_rank_and_coverage(self):
        """
        Testing pair position max.
        """
        scores = np.random.rand(100, 100)
        scores[np.tril_indices(100, 1)] = 0
        scores += scores.T
        rank, coverage = compute_rank_and_coverage(seq_length=100, scores=scores, pos_size=2, rank_type='max')
        unique_scores = np.unique(scores[np.triu_indices(100, k=1)])
        unique_rank = np.unique(rank[np.triu_indices(100, k=1)])
        unique_covearge = np.unique(coverage[np.triu_indices(100, k=1)])
        self.assertEqual(unique_scores.shape, unique_rank.shape)
        self.assertEqual(unique_scores.shape, unique_covearge.shape)
        max_score = np.max(scores[np.triu_indices(100, k=1)])
        max_rank = np.max(rank[np.triu_indices(100, k=1)])
        max_coverage = np.max(coverage[np.triu_indices(100, k=1)])
        min_score = np.min(scores[np.triu_indices(100, k=1)])
        min_rank = np.min(rank[np.triu_indices(100, k=1)])
        min_coverage = np.min(coverage[np.triu_indices(100, k=1)])
        min_mask = np.triu(scores == min_score, k=1)
        rank_mask = np.triu(rank == max_rank, k=1)
        cov_mask = np.triu(coverage == max_coverage, k=1)
        max_mask = np.triu(scores == max_score, k=1)
        rank_mask2 = np.triu(rank == min_rank, k=1)
        cov_mask2 = np.triu(coverage == min_coverage, k=1)
        diff_min_ranks = min_mask ^ rank_mask
        if diff_min_ranks.any():
            print(min_score)
            print(min_rank)
            print(np.sum(min_mask))
            print(np.sum(rank_mask))
        self.assertFalse(diff_min_ranks.any())
        diff_min_cov = min_mask ^ cov_mask
        if diff_min_cov.any():
            print(min_score)
            print(max_coverage)
            print(np.sum(min_mask))
            print(np.sum(cov_mask))
        self.assertFalse(diff_min_cov.any())
        diff_max_ranks = max_mask ^ rank_mask2
        if diff_max_ranks.any():
            print(max_score)
            print(max_rank)
            print(np.sum(max_mask))
            print(np.sum(rank_mask2))
        self.assertFalse(diff_max_ranks.any())
        diff_max_cov = max_mask ^ cov_mask2
        if diff_max_cov.any():
            print(max_score)
            print(max_coverage)
            print(np.sum(max_mask))
            print(np.sum(cov_mask2))
        self.assertFalse(diff_min_cov.any())


if __name__ == '__main__':
    unittest.main()
