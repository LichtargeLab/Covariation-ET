"""
Created on May 16, 2019

@author: Daniel Konecki
"""
import os
import numpy as np
from time import time
from copy import deepcopy
from shutil import rmtree
from unittest import TestCase
from multiprocessing import cpu_count
from Bio.Phylo.TreeConstruction import DistanceCalculator
from ETMIPWrapper import ETMIPWrapper
from SeqAlignment import SeqAlignment
from DataSetGenerator import DataSetGenerator
from AlignmentDistanceCalculator import AlignmentDistanceCalculator


class TestAlignmentDistanceCalculator(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.max_threads = cpu_count()
        cls.max_target_seqs = 500
        cls.testing_dir = os.path.abspath('../Test/')
        cls.input_path = os.path.join(cls.testing_dir, 'Input')
        cls.protein_list_path = os.path.join(cls.input_path, 'ProteinLists')
        if not os.path.isdir(cls.protein_list_path):
            os.makedirs(cls.protein_list_path)
        cls.small_structure_id = '7hvp'
        cls.large_structure_id = '2zxe'
        cls.protein_list_fn = os.path.join(cls.protein_list_path, 'Test_Set.txt')
        structure_ids = [cls.small_structure_id, cls.large_structure_id]
        with open(cls.protein_list_fn, 'wb') as test_list_handle:
            for structure_id in structure_ids:
                test_list_handle.write('{}\n'.format(structure_id))
        cls.data_set = DataSetGenerator(input_path=cls.input_path)
        cls.data_set.build_pdb_alignment_dataset(protein_list_fn='Test_Set.txt', num_threads=cls.max_threads,
                                                 max_target_seqs=cls.max_target_seqs)

    @classmethod
    def tearDownClass(cls):
        # rmtree(cls.input_path)
        del cls.data_set
        del cls.protein_list_fn
        del cls.large_structure_id
        del cls.small_structure_id
        del cls.protein_list_path
        del cls.input_path
        del cls.testing_dir

    def setUp(self):
        self.query_aln_fa_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['FA_File'],
                                               query_id=self.small_structure_id)
        self.query_aln_fa_small.import_alignment()
        self.query_aln_fa_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['FA_File'],
                                               query_id=self.large_structure_id)
        self.query_aln_fa_large.import_alignment()
        self.query_aln_msf_small = deepcopy(self.query_aln_fa_small)
        self.query_aln_msf_small.file_name = self.data_set.protein_data[self.small_structure_id]['MSF_File']
        self.query_aln_msf_large = deepcopy(self.query_aln_fa_large)
        self.query_aln_msf_large.file_name = self.data_set.protein_data[self.large_structure_id]['MSF_File']

    def tearDown(self):
        if os.path.exists('./identity.pkl'):
            os.remove('./identity.pkl')
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test')
        # if os.path.exists(wetc_test_dir):
        #     rmtree(wetc_test_dir)

    def test1a_get_distance_small_identity(self):
        self.query_aln_fa_small.compute_distance_matrix(model='identity')
        identity_calc_current = AlignmentDistanceCalculator()
        start = time()
        identity_dist_current = identity_calc_current.get_distance(self.query_aln_fa_small.alignment)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(self.query_aln_fa_small.distance_matrix.names == identity_dist_current.names)
        diff = np.array(self.query_aln_fa_small.distance_matrix) - np.array(identity_dist_current)
        self.assertTrue(not diff.any())
        identity_calc_official = DistanceCalculator()
        start = time()
        identity_dist_official = identity_calc_official.get_distance(self.query_aln_fa_small.alignment)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(identity_dist_current.names == identity_dist_official.names)
        diff = np.array(identity_dist_current) - np.array(identity_dist_official)
        self.assertTrue(not diff.any())

    def test1b_get_distance_small_blosum62(self):
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

    def test2a_get_distance_large_identity(self):
        self.query_aln_fa_large.compute_distance_matrix(model='identity')
        identity_calc_current = AlignmentDistanceCalculator()
        start = time()
        identity_dist_current = identity_calc_current.get_distance(self.query_aln_fa_fa_large.alignment)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(self.query_aln_fa_large.distance_matrix.names == identity_dist_current.names)
        diff = np.array(self.query_aln_fa_large.distance_matrix) - np.array(identity_dist_current)
        self.assertTrue(not diff.any())
        identity_calc_official = DistanceCalculator()
        start = time()
        identity_dist_official = identity_calc_official.get_distance(self.query_aln_fa_large.alignment)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(identity_dist_current.names == identity_dist_official.names)
        diff = np.array(identity_dist_current) - np.array(identity_dist_official)
        self.assertTrue(not diff.any())

    def test2b_get_distance_large_blosum62(self):
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

    def test3a_get_et_distance_small(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        aln_dist_df, id_dist_df, intermediate_df1 = et_mip_obj.import_distance_matrices(wetc_test_dir)
        aln_dist_array = np.asarray(aln_dist_df, dtype=float)
        id_dist_array = np.asarray(id_dist_df, dtype=float)
        aln_dist_dm1 = SeqAlignment._convert_array_to_distance_matrix(aln_dist_array, list(aln_dist_df.columns))
        id_dist_dm1 = SeqAlignment._convert_array_to_distance_matrix(id_dist_array.T, list(id_dist_df.columns))
        et_calc = AlignmentDistanceCalculator(model='blosum62')
        id_dist_dm2, aln_dist_dm2, intermediate_df2, threshold = et_calc.get_et_distance(self.query_aln_fa_small.alignment)
        diff_aln_dist = np.abs(np.array(aln_dist_dm1) - np.array(aln_dist_dm2))
        diff_aln_dist_threshold = diff_aln_dist > 1e-3  # Differences may arise in the third decimal place.
        diff_id_dist = np.abs(np.array(id_dist_dm1) - np.array(id_dist_dm2))
        diff_id_threshold = diff_id_dist > 1e-3  # Differences may arise in the third decimal place.
        self.assertTrue(id_dist_dm1.names == id_dist_dm2.names)
        self.assertTrue(not diff_id_threshold.any())
        self.assertTrue(aln_dist_dm1.names == aln_dist_dm2.names)
        self.assertTrue(not diff_aln_dist_threshold.any())

    def test3b_get_et_distance_large(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        aln_dist_df, id_dist_df, intermediate_df1 = et_mip_obj.import_distance_matrices(wetc_test_dir)
        aln_dist_array = np.asarray(aln_dist_df, dtype=float)
        id_dist_array = np.asarray(id_dist_df, dtype=float)
        aln_dist_dm1 = SeqAlignment._convert_array_to_distance_matrix(aln_dist_array, list(aln_dist_df.columns))
        id_dist_dm1 = SeqAlignment._convert_array_to_distance_matrix(id_dist_array.T, list(id_dist_df.columns))
        et_calc = AlignmentDistanceCalculator(model='blosum62')
        id_dist_dm2, aln_dist_dm2, intermediate_df2, threshold = et_calc.get_et_distance(self.query_aln_fa_large.alignment)
        diff_aln_dist = np.abs(np.array(aln_dist_dm1) - np.array(aln_dist_dm2))
        diff_aln_dist_threshold = diff_aln_dist > 1e-3  # Differences may arise in the third decimal place.
        diff_id_dist = np.abs(np.array(id_dist_dm1) - np.array(id_dist_dm2))
        diff_id_threshold = diff_id_dist > 1e-3  # Differences may arise in the third decimal place.
        self.assertTrue(id_dist_dm1.names == id_dist_dm2.names)
        self.assertTrue(not diff_id_threshold.any())
        self.assertTrue(aln_dist_dm1.names == aln_dist_dm2.names)
        self.assertTrue(not diff_aln_dist_threshold.any())
