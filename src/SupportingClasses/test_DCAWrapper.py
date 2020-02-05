"""
Created on February 5, 2020

@author: Daniel Konecki
"""
import os
import unittest
import numpy as np
from time import time
from shutil import rmtree
from test_Base import TestBase
from DCAWrapper import DCAWrapper
from utils import compute_rank_and_coverage


class TestDCAWrapper(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestDCAWrapper, cls).setUpClass()
        cls.small_fa_fn = cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln']
        cls.large_fa_fn = cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln']
        cls.out_small_dir = os.path.join(cls.testing_dir, cls.small_structure_id)
        rmtree(cls.out_small_dir, ignore_errors=True)
        cls.out_large_dir = os.path.join(cls.testing_dir, cls.large_structure_id)
        rmtree(cls.out_large_dir, ignore_errors=True)

    def evaluate_init(self, query, aln_file, out_dir, expected_length, expected_sequence):
        self.assertFalse(os.path.isdir(os.path.abspath(out_dir)))
        dca = DCAWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        self.assertEqual(dca.out_dir, os.path.abspath(out_dir))
        self.assertTrue(os.path.isdir(os.path.abspath(out_dir)))
        self.assertEqual(dca.query, query)
        self.assertIsNotNone(dca.original_aln)
        self.assertGreaterEqual(dca.original_aln.seq_length, expected_length)
        self.assertEqual(str(dca.original_aln.query_sequence).replace('-', ''), expected_sequence)
        self.assertEqual(dca.original_aln_fn, os.path.join(out_dir, 'Original_Alignment.fa'))
        self.assertTrue(os.path.isfile(dca.original_aln_fn))
        self.assertIsNotNone(dca.non_gapped_aln)
        self.assertEqual(dca.non_gapped_aln.seq_length, expected_length)
        self.assertEqual(dca.non_gapped_aln.query_sequence, expected_sequence)
        self.assertEqual(dca.non_gapped_aln_fn, os.path.join(out_dir, 'Non-Gapped_Alignment.fa'))
        self.assertTrue(os.path.isfile(dca.non_gapped_aln_fn))
        self.assertEqual(dca.method, 'DCA')
        self.assertIsNone(dca.scores)
        self.assertIsNone(dca.coverages)
        self.assertIsNone(dca.rankings)
        self.assertIsNone(dca.time)

    def test_1a_init(self):
        self.evaluate_init(query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
                           expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
                           expected_sequence=str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq))

    def test_1b_init(self):
        self.evaluate_init(query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
                           expected_length=self.data_set.protein_data[self.large_structure_id]['Length'],
                           expected_sequence=str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq))

    def evaluate_import_scores(self, query, aln_file, out_dir, expected_length):
        dca = DCAWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        scores = np.random.RandomState(1234567890).rand(expected_length, expected_length)
        scores[np.tril_indices(expected_length, 1)] = 0
        scores += scores.T
        indices = np.triu_indices(expected_length, 1)
        sorted_scores, sorted_x, sorted_y = zip(*sorted(zip(scores[indices], indices[0], indices[1])))
        with open(os.path.join(out_dir, 'DCA_predictions.tsv'), 'w') as handle:
            for i in range(len(sorted_scores)):
                handle.write('{} {} {}\n'.format(sorted_x[i] + 1, sorted_y[i] + 1, sorted_scores[i]))
        dca.import_covariance_scores(out_path=os.path.join(out_dir, 'DCA_predictions.tsv'))
        diff_scores = dca.scores - scores
        not_passing_scores = diff_scores > 1E15
        self.assertFalse(not_passing_scores.any())

    def test_2a_import_scores(self):
        self.evaluate_import_scores(
            query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
            expected_length=self.data_set.protein_data[self.small_structure_id]['Length'])

    def test_2b_import_scores(self):
        self.evaluate_import_scores(
            query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
            expected_length=self.data_set.protein_data[self.large_structure_id]['Length'])

    def evaluate_calculator_scores(self, query, aln_file, out_dir, expected_length, expected_sequence):
        dca = DCAWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        start = time()
        dca.calculate_scores(delete_file=False)
        end = time()
        expected_time = end - start
        self.assertEqual(dca.out_dir, os.path.abspath(out_dir))
        self.assertTrue(os.path.isdir(os.path.abspath(out_dir)))
        self.assertEqual(dca.query, query)
        self.assertIsNotNone(dca.original_aln)
        self.assertGreaterEqual(dca.original_aln.seq_length, expected_length)
        self.assertEqual(str(dca.original_aln.query_sequence).replace('-', ''), expected_sequence)
        self.assertEqual(dca.original_aln_fn, os.path.join(out_dir, 'Original_Alignment.fa'))
        self.assertTrue(os.path.isfile(dca.original_aln_fn))
        self.assertIsNotNone(dca.non_gapped_aln)
        self.assertEqual(dca.non_gapped_aln.seq_length, expected_length)
        self.assertEqual(dca.non_gapped_aln.query_sequence, expected_sequence)
        self.assertEqual(dca.non_gapped_aln_fn, os.path.join(out_dir, 'Non-Gapped_Alignment.fa'))
        self.assertTrue(os.path.isfile(dca.non_gapped_aln_fn))
        self.assertEqual(dca.method, 'DCA')
        self.assertIsNotNone(dca.scores)
        expected_ranks, expected_coverages = compute_rank_and_coverage(expected_length, dca.scores, 2, 'max')
        ranks_diff = dca.rankings - expected_ranks
        ranks_not_passing = ranks_diff > 0.0
        self.assertFalse(ranks_not_passing.any())
        coverages_diff = dca.coverages - expected_coverages
        coverages_not_passing = coverages_diff > 0.0
        self.assertFalse(coverages_not_passing.any())
        self.assertLessEqual(dca.time, expected_time)
        self.assertTrue(os.path.isfile(os.path.join(out_dir, 'DCA.npz')))

    def test_3a_calculate_scores(self):
        self.evaluate_calculator_scores(
            query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
            expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
            expected_sequence=str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq))

    def test_3b_calculate_scores(self):
        self.evaluate_calculator_scores(
            query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
            expected_length=self.data_set.protein_data[self.large_structure_id]['Length'],
            expected_sequence=str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq))


if __name__ == '__main__':
    unittest.main()