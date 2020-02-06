"""
Created on February 5, 2020

@author: Daniel Konecki
"""
import os
import unittest
import numpy as np
from time import time
from shutil import rmtree
from multiprocessing import cpu_count
from dotenv import find_dotenv, load_dotenv
from evcouplings.utils import read_config_file
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)
from test_Base import TestBase
from ETMIPWrapper import ETMIPWrapper
from utils import compute_rank_and_coverage


class TestEVCouplingsWrapper(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestEVCouplingsWrapper, cls).setUpClass()
        cls.small_fa_fn = cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln']
        cls.large_fa_fn = cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln']
        cls.out_small_dir = os.path.join(cls.testing_dir, cls.small_structure_id)
        rmtree(cls.out_small_dir, ignore_errors=True)
        cls.out_large_dir = os.path.join(cls.testing_dir, cls.large_structure_id)
        rmtree(cls.out_large_dir, ignore_errors=True)

    def evaluate_init(self, query, aln_file, out_dir, expected_length, expected_sequence):
        self.assertFalse(os.path.isdir(os.path.abspath(out_dir)))
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        self.assertEqual(wetc.out_dir, os.path.abspath(out_dir))
        self.assertTrue(os.path.isdir(os.path.abspath(out_dir)))
        self.assertEqual(wetc.query, query)
        self.assertIsNotNone(wetc.original_aln)
        self.assertGreaterEqual(wetc.original_aln.seq_length, expected_length)
        self.assertEqual(str(wetc.original_aln.query_sequence).replace('-', ''), expected_sequence)
        self.assertEqual(wetc.original_aln_fn, os.path.join(out_dir, 'Original_Alignment.fa'))
        self.assertTrue(os.path.isfile(wetc.original_aln_fn))
        self.assertIsNotNone(wetc.non_gapped_aln)
        self.assertEqual(wetc.non_gapped_aln.seq_length, expected_length)
        self.assertEqual(wetc.non_gapped_aln.query_sequence, expected_sequence)
        self.assertEqual(wetc.non_gapped_aln_fn, os.path.join(out_dir, 'Non-Gapped_Alignment.fa'))
        self.assertTrue(os.path.isfile(wetc.non_gapped_aln_fn))
        self.assertEqual(wetc.method, 'WETC')
        self.assertIsNone(wetc.scores)
        self.assertIsNone(wetc.coverages)
        self.assertIsNone(wetc.rankings)
        self.assertIsNone(wetc.time)
        self.assertIsNone(wetc.msf_aln_fn)
        self.assertIsNone(wetc.distance_matrix)
        self.assertIsNone(wetc.tree)
        self.assertIsNone(wetc.rank_group_assignments)
        self.assertIsNone(wetc.rank_scores)
        self.assertIsNone(wetc.entropy)

    # def test_1a_init(self):
    #     self.evaluate_init(query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
    #                        expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
    #                        expected_sequence=str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq))
    #
    # def test_1b_init(self):
    #     self.evaluate_init(query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
    #                        expected_length=self.data_set.protein_data[self.large_structure_id]['Length'],
    #                        expected_sequence=str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq))

    def evaluate_convert_alignment(self, query, aln_file, out_dir):
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        wetc.convert_alignment()
        self.assertTrue(os.path.isfile(os.path.join(out_dir, 'Non-Gapped_Alignment.msf')))

    # def test_2a_convert_alignment(self):
    #     self.evaluate_convert_alignment(query=self.small_structure_id, aln_file=self.small_fa_fn,
    #                                     out_dir=self.out_small_dir)
    #
    # def test_2b_convert_alignment(self):
    #     self.evaluate_convert_alignment(query=self.large_structure_id, aln_file=self.large_fa_fn,
    #                                     out_dir=self.out_large_dir)

    def evaluate_import_rank_scores(self, query, aln_file, out_dir, expected_length):
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        scores = np.random.RandomState(1234567890).rand(expected_length)
        ranks, _ = compute_rank_and_coverage(expected_length, scores, 1, 'min')
        expected_path = os.path.join(out_dir, 'etc_out.rank_id.tsv')
        with open(expected_path, 'w') as handle:
            handle.write('Position\tRank\n')
            for i in range(expected_length):
                handle.write('{}\t{}\n'.format(i, ranks[i]))
        wetc.import_rank_scores()
        diff_ranks = wetc.rank_scores - ranks
        not_passing_ranks = diff_ranks > 1E-15
        self.assertFalse(not_passing_ranks.any())

    # def test_3a_import_rank_scores(self):
    #     self.evaluate_import_rank_scores(
    #         query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
    #         expected_length=self.data_set.protein_data[self.small_structure_id]['Length'])
    #
    # def test_3b_import_rank_scores(self):
    #     self.evaluate_import_rank_scores(
    #         query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
    #         expected_length=self.data_set.protein_data[self.large_structure_id]['Length'])

    def evaluate_import_entropy_rank_scores(self, query, aln_file, out_dir, expected_length):
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        expected_path = os.path.join(out_dir, 'etc_out.rank_plain_entropy.tsv')
        expected_positions = np.array(range(1, wetc.non_gapped_aln.seq_length + 1))
        rand_state = np.random.RandomState(1234567890)
        expected_rho = rand_state.rand(expected_length)
        expected_ranks = {x: rand_state.rand(expected_length)
                          for x in range(1, wetc.non_gapped_aln.size)}
        with open(expected_path, 'w') as handle:
            handle.write('Position\t' +
                         '\t'.join(['Rank {} Entropy'.format(x) for x in range(1, wetc.non_gapped_aln.size)]) +
                         '\tRho\n')
            for i in range(expected_length):
                handle.write('{}\t'.format(expected_positions[i]) +
                             '\t'.join([str(expected_ranks[x][i]) for x in range(1, wetc.non_gapped_aln.size)]) +
                             '\t{}\n'.format(expected_rho[i]))
        wetc.import_entropy_rank_sores()
        diff_rho = wetc.rho - expected_rho
        not_passing_rho = diff_rho > 1E-15
        self.assertFalse(not_passing_rho.any())
        for i in range(1, wetc.non_gapped_aln.size):
            diff_rank_entropy = wetc.entropy[i] - expected_ranks[i]
            not_passing_rank_entropy = diff_rank_entropy > 1E-15
            self.assertFalse(not_passing_rank_entropy.any())

    # def test_4a_import_entropy_rank_scores(self):
    #     self.evaluate_import_entropy_rank_scores(
    #         query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
    #         expected_length=self.data_set.protein_data[self.small_structure_id]['Length'])
    #
    # def test_4b_import_entropy_rank_scores(self):
    #     self.evaluate_import_entropy_rank_scores(
    #         query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
    #         expected_length=self.data_set.protein_data[self.large_structure_id]['Length'])

    # def evaluate_import_scores(self, query, aln_file, out_dir, expected_length):
    #     evc = EVCouplingsWrapper(query=query, aln_file=aln_file, out_dir=out_dir, protocol='standard')
    #     scores = np.random.RandomState(1234567890).rand(expected_length, expected_length)
    #     scores[np.tril_indices(expected_length, 1)] = 0
    #     scores += scores.T
    #     _, coverages = compute_rank_and_coverage(expected_length, scores, 2, 'max')
    #     probabilities = 1 - coverages
    #     indices = np.triu_indices(expected_length, 1)
    #     sorted_scores, sorted_x, sorted_y, sorted_probability = zip(*sorted(zip(scores[indices], indices[0], indices[1],
    #                                                                             probabilities[indices])))
    #     expected_dir = os.path.join(out_dir, 'couplings')
    #     os.makedirs(expected_dir, exist_ok=True)
    #     expected_path = os.path.join(expected_dir, '_CouplingScores.csv')
    #     with open(expected_path, 'w') as handle:
    #         handle.write('i,A_i,j,A_j,fn,cn,segment_i,segment_j,probability\n')
    #         for i in range(len(sorted_scores)):
    #             handle.write('{},X,{},X,0,{},X,X,{}\n'.format(sorted_x[i] + 1, sorted_y[i] + 1, sorted_scores[i],
    #                                                           sorted_probability[i]))
    #     evc.import_covariance_scores(out_path=expected_path)
    #
    #     diff_expected_scores = scores - scores.T
    #     not_passing_expected_scores = diff_expected_scores > 1E-15
    #     self.assertFalse(not_passing_expected_scores.any())
    #     diff_computed_scores = evc.scores - evc.scores.T
    #     not_passing_computed_scores = diff_computed_scores > 1E-15
    #     self.assertFalse(not_passing_computed_scores.any())
    #
    #     diff_scores = evc.scores - scores
    #     not_passing_scores = diff_scores > 1E-15
    #     self.assertFalse(not_passing_scores.any())
    #     diff_probabilities = evc.probability - probabilities
    #     not_passing_protbabilities = diff_probabilities > 1E-15
    #     self.assertFalse(not_passing_protbabilities.any())
    #
    # def test_3a_import_scores(self):
    #     self.evaluate_import_scores(
    #         query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
    #         expected_length=self.data_set.protein_data[self.small_structure_id]['Length'])
    #
    # def test_3b_import_scores(self):
    #     self.evaluate_import_scores(
    #         query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
    #         expected_length=self.data_set.protein_data[self.large_structure_id]['Length'])
    #
    # def evaluate_calculator_scores(self, query, aln_file, out_dir, expected_length, expected_sequence):
    #     evc = EVCouplingsWrapper(query=query, aln_file=aln_file, out_dir=out_dir, protocol='standard')
    #     start = time()
    #     evc.calculate_scores(delete_files=False, cores=self.max_threads)
    #     end = time()
    #     expected_time = end - start
    #     self.assertEqual(evc.out_dir, os.path.abspath(out_dir))
    #     self.assertTrue(os.path.isdir(os.path.abspath(out_dir)))
    #     self.assertEqual(evc.query, query)
    #     self.assertIsNotNone(evc.original_aln)
    #     self.assertGreaterEqual(evc.original_aln.seq_length, expected_length)
    #     self.assertEqual(str(evc.original_aln.query_sequence).replace('-', ''), expected_sequence)
    #     self.assertEqual(evc.original_aln_fn, os.path.join(out_dir, 'Original_Alignment.fa'))
    #     self.assertTrue(os.path.isfile(evc.original_aln_fn))
    #     self.assertIsNotNone(evc.non_gapped_aln)
    #     self.assertEqual(evc.non_gapped_aln.seq_length, expected_length)
    #     self.assertEqual(evc.non_gapped_aln.query_sequence, expected_sequence)
    #     self.assertEqual(evc.non_gapped_aln_fn, os.path.join(out_dir, 'Non-Gapped_Alignment.fa'))
    #     self.assertTrue(os.path.isfile(evc.non_gapped_aln_fn))
    #     self.assertEqual(evc.method, 'EVCouplings')
    #     self.assertEqual(evc.protocol, 'standard')
    #     self.assertIsNotNone(evc.scores)
    #     self.assertIsNotNone(evc.probability)
    #     expected_ranks, expected_coverages = compute_rank_and_coverage(expected_length, evc.scores, 2, 'max')
    #     ranks_diff = evc.rankings - expected_ranks
    #     ranks_not_passing = ranks_diff > 0.0
    #     self.assertFalse(ranks_not_passing.any())
    #     coverages_diff = evc.coverages - expected_coverages
    #     coverages_not_passing = coverages_diff > 0.0
    #     self.assertFalse(coverages_not_passing.any())
    #     self.assertLessEqual(evc.time, expected_time)
    #     self.assertTrue(os.path.isfile(os.path.join(out_dir, 'EVCouplings.npz')))
    #
    # def test_4a_calculate_scores(self):
    #     self.evaluate_calculator_scores(
    #         query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
    #         expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
    #         expected_sequence=str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq))
    #
    # def test_4b_calculate_scores(self):
    #     self.evaluate_calculator_scores(
    #         query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
    #         expected_length=self.data_set.protein_data[self.large_structure_id]['Length'],
    #         expected_sequence=str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq))


if __name__ == '__main__':
    unittest.main()