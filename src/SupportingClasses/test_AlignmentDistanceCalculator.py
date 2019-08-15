"""
Created on May 16, 2019

@author: Daniel Konecki
"""
import os
import numpy as np
from time import time
from copy import deepcopy
from Bio.Phylo.TreeConstruction import DistanceCalculator
from test_Base import TestBase
from ETMIPWrapper import ETMIPWrapper
from SeqAlignment import SeqAlignment
from EvolutionaryTraceAlphabet import FullIUPACProtein
from AlignmentDistanceCalculator import AlignmentDistanceCalculator, convert_array_to_distance_matrix


class TestAlignmentDistanceCalculator(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestAlignmentDistanceCalculator, cls).setUpClass()
        cls.query_aln_fa_small = SeqAlignment(
            file_name=cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln'],
            query_id=cls.small_structure_id)
        cls.query_aln_fa_small.import_alignment()

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

    def test1a_init(self):
        alpha = FullIUPACProtein()
        identity_calc_current = AlignmentDistanceCalculator()
        self.assertEqual(identity_calc_current.aln_type, 'protein')
        self.assertEqual(identity_calc_current.alphabet.letters, alpha.letters)
        self.assertEqual(identity_calc_current.model, 'identity')
        self.assertEqual(identity_calc_current.alphabet_size, len(alpha.letters))
        self.assertEqual(identity_calc_current.gap_characters, {'-', '.', '*'})
        expected_mapping = {char: i for i, char in enumerate(alpha.letters)}
        expected_mapping['-'] = len(alpha.letters)
        expected_mapping['.'] = len(alpha.letters)
        expected_mapping['*'] = len(alpha.letters)
        self.assertEqual(identity_calc_current.mapping, expected_mapping)
        expected_matrix = np.zeros((len(alpha.letters) + 2, len(alpha.letters) + 2))
        expected_matrix[range(len(alpha.letters) + 1), range(len(alpha.letters) + 1)] = 1
        diff = identity_calc_current.scoring_matrix - expected_matrix
        self.assertFalse(diff.any())

    def test1b_init(self):
        alpha = FullIUPACProtein()
        identity_calc_current = AlignmentDistanceCalculator(model='blosum62')
        self.assertEqual(identity_calc_current.aln_type, 'protein')
        self.assertEqual(identity_calc_current.alphabet.letters, alpha.letters)
        self.assertEqual(identity_calc_current.model, 'blosum62')
        self.assertEqual(identity_calc_current.alphabet_size, len(alpha.letters))
        self.assertEqual(identity_calc_current.gap_characters, {'-', '.', '*'})
        expected_mapping = {char: i for i, char in enumerate(alpha.letters)}
        expected_mapping['-'] = len(alpha.letters)
        expected_mapping['.'] = len(alpha.letters)
        expected_mapping['*'] = len(alpha.letters)
        self.assertEqual(identity_calc_current.mapping, expected_mapping)
        old_dist_calc = DistanceCalculator(model='blosum62').scoring_matrix
        diff = identity_calc_current.scoring_matrix[:len(alpha.letters), :len(alpha.letters)] - np.array(old_dist_calc)
        self.assertFalse(diff.any())

    def test2__build_identity_scoring_matrix(self):
        alpha = FullIUPACProtein()
        identity_calc_current = AlignmentDistanceCalculator()
        identity_scoring_matrix = identity_calc_current._build_identity_scoring_matrix()
        expected_matrix = np.zeros((len(alpha.letters) + 2, len(alpha.letters) + 2))
        expected_matrix[range(len(alpha.letters) + 1), range(len(alpha.letters) + 1)] = 1
        diff = identity_scoring_matrix - expected_matrix
        self.assertFalse(diff.any())

    def test3__rebuild_scoring_matrix(self):
        alpha = FullIUPACProtein()
        identity_calc_current = AlignmentDistanceCalculator(model='blosum62')
        rebuilt_scoring_matrix = identity_calc_current._rebuild_scoring_matrix()
        old_dist_calc = DistanceCalculator(model='blosum62').scoring_matrix
        diff = rebuilt_scoring_matrix[:len(alpha.letters), :len(alpha.letters)] - np.array(old_dist_calc)
        self.assertFalse(diff.any())
        self.assertFalse(rebuilt_scoring_matrix[len(alpha.letters):, :].any())
        self.assertFalse(rebuilt_scoring_matrix[:, len(alpha.letters):].any())

    def test4a__update_scoring_matrix(self):
        alpha = FullIUPACProtein()
        identity_calc_current = AlignmentDistanceCalculator()
        identity_scoring_matrix = identity_calc_current._update_scoring_matrix()
        expected_matrix = np.zeros((len(alpha.letters) + 2, len(alpha.letters) + 2))
        expected_matrix[range(len(alpha.letters) + 1), range(len(alpha.letters) + 1)] = 1
        diff = identity_scoring_matrix - expected_matrix
        self.assertFalse(diff.any())

    def test4b__update_scoring_matrix(self):
        alpha = FullIUPACProtein()
        identity_calc_current = AlignmentDistanceCalculator(model='blosum62')
        rebuilt_scoring_matrix = identity_calc_current._update_scoring_matrix()
        old_dist_calc = DistanceCalculator(model='blosum62').scoring_matrix
        diff = rebuilt_scoring_matrix[:len(alpha.letters), :len(alpha.letters)] - np.array(old_dist_calc)
        self.assertFalse(diff.any())
        self.assertFalse(rebuilt_scoring_matrix[len(alpha.letters):, :].any())
        self.assertFalse(rebuilt_scoring_matrix[:, len(alpha.letters):].any())

    def test5a__pairwise(self):
        identity_calc_current = AlignmentDistanceCalculator()
        old_calc = DistanceCalculator(model='identity')
        for i in range(self.query_aln_fa_small.size):
            dist = identity_calc_current._pairwise(seq1=self.query_aln_fa_small.alignment[i],
                                                   seq2=self.query_aln_fa_small.alignment[0])
            expected_dist = old_calc._pairwise(seq1=self.query_aln_fa_small.alignment[i],
                                               seq2=self.query_aln_fa_small.alignment[0])
            self.assertEqual(dist, expected_dist)

    def test5b__pairwise(self):
        identity_calc_current = AlignmentDistanceCalculator(model='blosum62')
        old_calc = DistanceCalculator(model='blosum62')
        for i in range(self.query_aln_fa_small.size):
            dist = identity_calc_current._pairwise(seq1=self.query_aln_fa_small.alignment[i],
                                                   seq2=self.query_aln_fa_small.alignment[0])
            expected_dist = old_calc._pairwise(seq1=self.query_aln_fa_small.alignment[i],
                                               seq2=self.query_aln_fa_small.alignment[0])
            self.assertEqual(dist, expected_dist)

    def test6a_get_identity_distance(self):
        identity_calc_current = AlignmentDistanceCalculator()
        start = time()
        identity_dist_current = identity_calc_current.get_identity_distance(self.query_aln_fa_small.alignment)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        identity_calc_official = DistanceCalculator()
        start = time()
        identity_dist_official = identity_calc_official.get_distance(self.query_aln_fa_small.alignment)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(identity_dist_current.names == identity_dist_official.names)
        diff = np.array(identity_dist_current) - np.array(identity_dist_official)
        self.assertTrue(not diff.any())

    def test6b_get_identity_distance(self):
        identity_calc_current = AlignmentDistanceCalculator()
        start = time()
        identity_dist_current = identity_calc_current.get_identity_distance(self.query_aln_fa_large.alignment)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        identity_calc_official = DistanceCalculator()
        start = time()
        identity_dist_official = identity_calc_official.get_distance(self.query_aln_fa_large.alignment)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(identity_dist_current.names == identity_dist_official.names)
        diff = np.array(identity_dist_current) - np.array(identity_dist_official)
        self.assertTrue(not diff.any())

    def test7a_get_scoring_matrix_distance(self):
        identity_calc_current = AlignmentDistanceCalculator(model='blosum62')
        start = time()
        identity_dist_current = identity_calc_current.get_scoring_matrix_distance(self.query_aln_fa_small.alignment)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        identity_calc_official = DistanceCalculator(model='blosum62')
        start = time()
        identity_dist_official = identity_calc_official.get_distance(self.query_aln_fa_small.alignment)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(identity_dist_current.names == identity_dist_official.names)
        diff = np.array(identity_dist_current) - np.array(identity_dist_official)
        self.assertTrue(not diff.any())

    def test7b_get_scoring_matrix_distance(self):
        identity_calc_current = AlignmentDistanceCalculator(model='blosum62')
        start = time()
        identity_dist_current = identity_calc_current.get_scoring_matrix_distance(self.query_aln_fa_large.alignment)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        identity_calc_official = DistanceCalculator(model='blosum62')
        start = time()
        identity_dist_official = identity_calc_official.get_distance(self.query_aln_fa_large.alignment)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(identity_dist_current.names == identity_dist_official.names)
        diff = np.array(identity_dist_current) - np.array(identity_dist_official)
        self.assertTrue(not diff.any())

    def test8a_get_distance_small_identity(self):
        identity_calc_current = AlignmentDistanceCalculator()
        start = time()
        identity_dist_current = identity_calc_current.get_distance(self.query_aln_fa_small.alignment)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        identity_calc_official = DistanceCalculator()
        start = time()
        identity_dist_official = identity_calc_official.get_distance(self.query_aln_fa_small.alignment)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(identity_dist_current.names == identity_dist_official.names)
        diff = np.array(identity_dist_current) - np.array(identity_dist_official)
        self.assertTrue(not diff.any())

    def test8b_get_distance_large_identity(self):
        identity_calc_current = AlignmentDistanceCalculator()
        start = time()
        identity_dist_current = identity_calc_current.get_distance(self.query_aln_fa_large.alignment)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        identity_calc_official = DistanceCalculator()
        start = time()
        identity_dist_official = identity_calc_official.get_distance(self.query_aln_fa_large.alignment)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(identity_dist_current.names == identity_dist_official.names)
        diff = np.array(identity_dist_current) - np.array(identity_dist_official)
        self.assertTrue(not diff.any())

    def test9a_get_distance_small_blosum62(self):
        blosum62_calc_current = AlignmentDistanceCalculator(model='blosum62')
        start = time()
        blosum62_dist_current = blosum62_calc_current.get_distance(self.query_aln_fa_small.alignment)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        blosum62_calc_official = DistanceCalculator(model='blosum62')
        start = time()
        blosum62_dist_official = blosum62_calc_official.get_distance(self.query_aln_fa_small.alignment)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(blosum62_dist_current.names == blosum62_dist_official.names)
        diff = np.array(blosum62_dist_current) - np.array(blosum62_dist_official)
        self.assertTrue(not diff.any())

    def test9b_get_distance_large_blosum62(self):
        blosum62_calc_current = AlignmentDistanceCalculator(model='blosum62')
        start = time()
        blosum62_dist_current = blosum62_calc_current.get_distance(self.query_aln_fa_large.alignment)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        blosum62_calc_official = DistanceCalculator(model='blosum62')
        start = time()
        blosum62_dist_official = blosum62_calc_official.get_distance(self.query_aln_fa_large.alignment)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(blosum62_dist_current.names == blosum62_dist_official.names)
        diff = np.array(blosum62_dist_current) - np.array(blosum62_dist_official)
        self.assertTrue(not diff.any())

    def test10a_get_et_distance_small(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id, 'intET')
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(method='intET', out_dir=wetc_test_dir, delete_files=False)
        aln_dist_df, id_dist_df, intermediate_df1 = et_mip_obj.import_distance_matrices(wetc_test_dir,
                                                                                        prefix='etc_out_intET')
        aln_dist_array = np.asarray(aln_dist_df, dtype=float)
        id_dist_array = np.asarray(id_dist_df, dtype=float)
        aln_dist_dm1 = convert_array_to_distance_matrix(aln_dist_array, list(aln_dist_df.columns))
        id_dist_dm1 = convert_array_to_distance_matrix(id_dist_array.T, list(id_dist_df.columns))
        et_calc = AlignmentDistanceCalculator(model='blosum62')
        id_dist_dm2, aln_dist_dm2, intermediate_df2, threshold = et_calc.get_et_distance(self.query_aln_fa_small.alignment)
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

    def test10b_get_et_distance_large(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id, 'intET')
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(method='intET', out_dir=wetc_test_dir, delete_files=False)
        aln_dist_df, id_dist_df, intermediate_df1 = et_mip_obj.import_distance_matrices(wetc_test_dir,
                                                                                        prefix='etc_out_intET')
        aln_dist_array = np.asarray(aln_dist_df, dtype=float)
        id_dist_array = np.asarray(id_dist_df, dtype=float)
        aln_dist_dm1 = convert_array_to_distance_matrix(aln_dist_array, list(aln_dist_df.columns))
        id_dist_dm1 = convert_array_to_distance_matrix(id_dist_array.T, list(id_dist_df.columns))
        et_calc = AlignmentDistanceCalculator(model='blosum62')
        id_dist_dm2, aln_dist_dm2, intermediate_df2, threshold = et_calc.get_et_distance(self.query_aln_fa_large.alignment)
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
