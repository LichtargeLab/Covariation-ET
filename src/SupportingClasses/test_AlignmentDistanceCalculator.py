"""
Created on May 16, 2019

@author: Daniel Konecki
"""
import os
import unittest
import numpy as np
from copy import deepcopy
from time import time, sleep
from Bio.Phylo.TreeConstruction import DistanceCalculator
from test_Base import TestBase
from ETMIPWrapper import ETMIPWrapper
from SeqAlignment import SeqAlignment
from EvolutionaryTraceAlphabet import FullIUPACProtein
from AlignmentDistanceCalculator import (AlignmentDistanceCalculator, convert_array_to_distance_matrix, init_pairwise,
                                         pairwise)


class TestAlignmentDistanceCalculator(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestAlignmentDistanceCalculator, cls).setUpClass()
        cls.query_aln_fa_small = SeqAlignment(
            file_name=cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln'],
            query_id=cls.small_structure_id)
        cls.query_aln_fa_small.import_alignment()
        cls.out_small_dir = os.path.join(cls.testing_dir, cls.small_structure_id)
        cls.out_large_dir = os.path.join(cls.testing_dir, cls.large_structure_id)

    def setUp(self):
        self.query_aln_fa_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                               query_id=self.small_structure_id)
        self.query_aln_fa_small.import_alignment()
        self.query_aln_fa_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                               query_id=self.large_structure_id)
        self.query_aln_fa_large.import_alignment()
        self.query_aln_msf_small = deepcopy(self.query_aln_fa_small)
        self.query_aln_msf_small.file_name = self.data_set.protein_data[self.small_structure_id]['Final_MSF_Aln']
        self.query_aln_msf_large = deepcopy(self.query_aln_fa_large)
        self.query_aln_msf_large.file_name = self.data_set.protein_data[self.large_structure_id]['Final_MSF_Aln']

    def tearDown(self):
        if os.path.exists('./identity.pkl'):
            os.remove('./identity.pkl')
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test')
        # if os.path.exists(wetc_test_dir):
        #     rmtree(wetc_test_dir)

    def evaluate_identity_scoring_matrix(self, scoring_matrix, alpha):
        expected_matrix = np.zeros((len(alpha.letters) + 2, len(alpha.letters) + 2))
        expected_matrix[range(len(alpha.letters) + 1), range(len(alpha.letters) + 1)] = 1
        diff = scoring_matrix - expected_matrix
        self.assertFalse(diff.any())

    def evaluate_blosum62_scoring_matrix(self, scoring_matrix):
        old_matrix = np.array(DistanceCalculator(model='blosum62').scoring_matrix)
        expected_matrix = np.pad(old_matrix, mode='constant', pad_width=((0, 2), (0, 2)), constant_values=0)
        diff = scoring_matrix - expected_matrix
        self.assertFalse(diff.any())

    def evaluate__init(self, aln_type, model_type):
        alpha = FullIUPACProtein()
        curr_calc = AlignmentDistanceCalculator(protein=aln_type == 'protein', model=model_type)
        self.assertEqual(curr_calc.aln_type, aln_type)
        self.assertEqual(curr_calc.alphabet.letters, alpha.letters)
        self.assertEqual(curr_calc.model, model_type)
        self.assertEqual(curr_calc.alphabet_size, len(alpha.letters))
        self.assertEqual(curr_calc.gap_characters, {'-', '.', '*'})
        expected_mapping = {char: i for i, char in enumerate(alpha.letters)}
        expected_mapping['-'] = len(alpha.letters)
        expected_mapping['.'] = len(alpha.letters)
        expected_mapping['*'] = len(alpha.letters)
        self.assertEqual(curr_calc.mapping, expected_mapping)
        if model_type == 'identity':
            self.evaluate_identity_scoring_matrix(scoring_matrix=curr_calc.scoring_matrix, alpha=alpha)
        elif model_type == 'blosum62':
            self.evaluate_blosum62_scoring_matrix(scoring_matrix=curr_calc.scoring_matrix)
        else:
            raise ValueError('evaluate_init not implemented for models other than identity or blosum62.')

    def test1a__init(self):
        self.evaluate__init(aln_type='protein', model_type='identity')

    def test1b__init(self):
        self.evaluate__init(aln_type='protein', model_type='blosum62')

    def test2__build_identity_scoring_matrix(self):
        alpha = FullIUPACProtein()
        identity_calc_current = AlignmentDistanceCalculator()
        identity_scoring_matrix = identity_calc_current._build_identity_scoring_matrix()
        self.evaluate_identity_scoring_matrix(scoring_matrix=identity_scoring_matrix, alpha=alpha)

    def test3__rebuild_scoring_matrix(self):
        curr_calc = AlignmentDistanceCalculator(model='blosum62')
        curr_calc.scoring_matrix = DistanceCalculator(model='blosum62').scoring_matrix
        rebuilt_scoring_matrix = curr_calc._rebuild_scoring_matrix()
        self.evaluate_blosum62_scoring_matrix(scoring_matrix=rebuilt_scoring_matrix)

    def test4a__update_scoring_matrix(self):
        alpha = FullIUPACProtein()
        identity_calc_current = AlignmentDistanceCalculator()
        identity_scoring_matrix = identity_calc_current._update_scoring_matrix()
        self.evaluate_identity_scoring_matrix(scoring_matrix=identity_scoring_matrix, alpha=alpha)

    def test4b__update_scoring_matrix(self):
        blosum62_calc_current = AlignmentDistanceCalculator(model='blosum62')
        blosum62_calc_current.scoring_matrix = DistanceCalculator(model='blosum62').scoring_matrix
        rebuilt_scoring_matrix = blosum62_calc_current._update_scoring_matrix()
        self.evaluate_blosum62_scoring_matrix(scoring_matrix=rebuilt_scoring_matrix)

    def evaluate__pairwise(self, model, seq1, seq2):
        curr_calc = AlignmentDistanceCalculator(model=model)
        old_calc = DistanceCalculator(model=model)
        dist = curr_calc._pairwise(seq1=seq1, seq2=seq2)
        expected_dist = old_calc._pairwise(seq1=seq1, seq2=seq2)
        self.assertEqual(dist, expected_dist)

    def test5a__pairwise(self):
        query_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
        self.evaluate__pairwise(model='identity', seq1=self.query_aln_fa_small.alignment[query_index],
                                seq2=self.query_aln_fa_small.alignment[query_index - 1])

    def test5b__pairwise(self):
        query_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
        self.evaluate__pairwise(model='identity', seq1=self.query_aln_fa_small.alignment[query_index],
                                seq2=self.query_aln_fa_small.alignment[query_index - 2])

    def test5c__pariwise(self):
        query_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
        self.evaluate__pairwise(model='blosum62', seq1=self.query_aln_fa_small.alignment[query_index],
                                seq2=self.query_aln_fa_small.alignment[query_index - 1])

    def test5d__pariwise(self):
        query_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
        self.evaluate__pairwise(model='blosum62', seq1=self.query_aln_fa_small.alignment[query_index],
                                seq2=self.query_aln_fa_small.alignment[query_index - 2])

    def evaluate__pairwise_init(self, model, seq1s, seq2s):
        curr_calc = AlignmentDistanceCalculator(model=model)
        old_calc = DistanceCalculator(model=model)
        init_pairwise(curr_calc.mapping, curr_calc.alphabet_size, curr_calc.model,
                      curr_calc.scoring_matrix)
        for i in range(len(seq1s)):
            _, _, dist = pairwise(seq1=seq1s[i], seq2=seq2s[i])
            expected_dist = old_calc._pairwise(seq1=seq1s[i], seq2=seq2s[i])
            self.assertEqual(dist, expected_dist)

    def test5e_pairwise(self):
        query_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
        self.evaluate__pairwise_init(model='identity', seq1s=[self.query_aln_fa_small.alignment[query_index]] * 2,
                                     seq2s=[self.query_aln_fa_small.alignment[query_index - 2],
                                            self.query_aln_fa_small.alignment[query_index - 1]])

    def test5f_pairwise(self):
        query_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
        self.evaluate__pairwise_init(model='blosum62', seq1s=[self.query_aln_fa_small.alignment[query_index]] * 2,
                                     seq2s=[self.query_aln_fa_small.alignment[query_index - 2],
                                            self.query_aln_fa_small.alignment[query_index - 1]])

    def evaluate_get_dentity_distance(self, aln, processes):
        curr_calc = AlignmentDistanceCalculator()
        start = time()
        identity_dist_current = curr_calc.get_identity_distance(aln, processes=processes)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        identity_calc_official = DistanceCalculator()
        start = time()
        identity_dist_official = identity_calc_official.get_distance(aln)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(identity_dist_current.names == identity_dist_official.names)
        diff = np.array(identity_dist_current) - np.array(identity_dist_official)
        self.assertTrue(not diff.any())

    def test6a_get_identity_distance(self):
        self.evaluate_get_dentity_distance(aln=self.query_aln_fa_small.alignment, processes=1)

    def test6b_get_identity_distance(self):
        self.evaluate_get_dentity_distance(aln=self.query_aln_fa_small.alignment, processes=self.max_threads)

    def test6c_get_identity_distance(self):
        self.evaluate_get_dentity_distance(aln=self.query_aln_fa_large.alignment, processes=1)

    def test6d_get_identity_distance(self):
        self.evaluate_get_dentity_distance(aln=self.query_aln_fa_large.alignment, processes=self.max_threads)

    def evaluate_get_scoring_matrix_distance(self, model, aln, processes):
        identity_calc_current = AlignmentDistanceCalculator(model=model)
        start = time()
        identity_dist_current = identity_calc_current.get_scoring_matrix_distance(aln, processes=processes)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        identity_calc_official = DistanceCalculator(model=model)
        start = time()
        identity_dist_official = identity_calc_official.get_distance(aln)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(identity_dist_current.names == identity_dist_official.names)
        diff = np.array(identity_dist_current) - np.array(identity_dist_official)
        self.assertTrue(not diff.any())

    def test7a_get_scoring_matrix_distance(self):
        self.evaluate_get_scoring_matrix_distance(model='blosum62', aln=self.query_aln_fa_small.alignment, processes=1)

    def test7b_get_scoring_matrix_distance(self):
        self.evaluate_get_scoring_matrix_distance(model='blosum62', aln=self.query_aln_fa_small.alignment,
                                                  processes=self.max_threads)

    def test7c_get_scoring_matrix_distance(self):
        self.evaluate_get_scoring_matrix_distance(model='blosum62', aln=self.query_aln_fa_large.alignment, processes=1)

    def test7d_get_scoring_matrix_distance(self):
        self.evaluate_get_scoring_matrix_distance(model='blosum62', aln=self.query_aln_fa_large.alignment,
                                                  processes=self.max_threads)

    def evaluate_get_distance(self, model, aln):
        curr_calc = AlignmentDistanceCalculator(model=model)
        start = time()
        curr_dist = curr_calc.get_distance(aln, processes=self.max_threads)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        official_calc = DistanceCalculator(model=model)
        start = time()
        official_dist = official_calc.get_distance(aln)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(curr_dist.names == official_dist.names)
        diff = np.array(curr_dist) - np.array(official_dist)
        self.assertTrue(not diff.any())

    def test8a_get_distance(self):
        # small identity
        self.evaluate_get_distance(model='identity', aln=self.query_aln_fa_small.alignment)

    def test8b_get_distance(self):
        # large identity
        self.evaluate_get_distance(model='identity', aln=self.query_aln_fa_large.alignment)

    def test8c_get_distance(self):
        # small blosum62
        self.evaluate_get_distance(model='blosum62', aln=self.query_aln_fa_small.alignment)

    def test8d_get_distance(self):
        # large blosum62
        self.evaluate_get_distance(model='blosum62', aln=self.query_aln_fa_large.alignment)

    def evaluate_get_et_distance(self, query_id, aln_fn, aln, processes, out_dir):
        et_mip_obj = ETMIPWrapper(query=query_id, aln_file=aln_fn, out_dir=out_dir)
        et_mip_obj.calculate_scores(method='intET', delete_files=False)
        aln_dist_df, id_dist_df, intermediate_df1 = et_mip_obj.import_distance_matrices(prefix='etc_out_intET')
        aln_dist_array = np.asarray(aln_dist_df, dtype=float)
        id_dist_array = np.asarray(id_dist_df, dtype=float)
        aln_dist_dm1 = convert_array_to_distance_matrix(aln_dist_array, list(aln_dist_df.columns))
        id_dist_dm1 = convert_array_to_distance_matrix(id_dist_array.T, list(id_dist_df.columns))
        et_calc = AlignmentDistanceCalculator(model='blosum62')
        id_dist_dm2, aln_dist_dm2, intermediate_df2, threshold = et_calc.get_et_distance(aln, processes=processes)
        diff_aln_dist = np.abs(np.array(aln_dist_dm1) - np.array(aln_dist_dm2))
        diff_aln_dist_threshold = diff_aln_dist > 1e-3  # Differences may arise in the third decimal place.
        diff_id_dist = np.abs(np.array(id_dist_dm1) - np.array(id_dist_dm2))
        diff_id_threshold = diff_id_dist > 1e-3  # Differences may arise in the third decimal place.
        joined = intermediate_df1.merge(intermediate_df2, on=['Seq1', 'Seq2'], how='inner', suffixes=('ETC', 'Python'))
        self.assertTrue(joined['Min_Seq_LengthETC'].equals(joined['Min_Seq_LengthPython']))
        self.assertTrue(joined['Id_CountETC'].equals(joined['Id_CountPython']))
        self.assertTrue(joined['Threshold_CountETC'].equals(joined['Threshold_CountPython']))
        self.assertTrue(id_dist_dm1.names == id_dist_dm2.names)
        self.assertTrue(not diff_id_threshold.any())
        self.assertTrue(aln_dist_dm1.names == aln_dist_dm2.names)
        self.assertTrue(not diff_aln_dist_threshold.any())

    def test10a_get_et_distance_small(self):
        self.evaluate_get_et_distance(query_id=self.small_structure_id, aln_fn=self.query_aln_fa_small.file_name,
                                      aln=self.query_aln_fa_small.remove_gaps().alignment, processes=1,
                                      out_dir=self.out_small_dir)

    def test10b_get_et_distance_small(self):
        self.evaluate_get_et_distance(query_id=self.small_structure_id, aln_fn=self.query_aln_fa_small.file_name,
                                      aln=self.query_aln_fa_small.remove_gaps().alignment, processes=self.max_threads,
                                      out_dir=self.out_small_dir)

    def test10c_get_et_distance_small(self):
        self.evaluate_get_et_distance(query_id=self.large_structure_id, aln_fn=self.query_aln_fa_large.file_name,
                                      aln=self.query_aln_fa_large.remove_gaps().alignment, processes=1,
                                      out_dir=self.out_large_dir)

    def test10d_get_et_distance_small(self):
        self.evaluate_get_et_distance(query_id=self.large_structure_id, aln_fn=self.query_aln_fa_large.file_name,
                                      aln=self.query_aln_fa_large.remove_gaps().alignment, processes=self.max_threads,
                                      out_dir=self.out_large_dir)


if __name__ == '__main__':
    unittest.main()
