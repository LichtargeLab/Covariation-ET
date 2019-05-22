"""
Created on May 16, 2019

@author: Daniel Konecki
"""
import os
import numpy as np
from time import time
from shutil import rmtree
from unittest import TestCase
from Bio.Phylo.TreeConstruction import DistanceCalculator
from SeqAlignment import SeqAlignment
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from ETMIPWrapper import  ETMIPWrapper

class TestSeqAlignment(TestCase):

    def setUp(self):
        self.testing_dir = '../Test/'
        msa_file_small = os.path.join(self.testing_dir, '7hvpA.fa')
        query_small = 'query_7hvpA'
        self.query_aln_small = SeqAlignment(file_name=msa_file_small, query_id=query_small)
        self.query_aln_small.import_alignment()
        msa_file_big = os.path.join(self.testing_dir, '2zxeA.fa')
        query_big = 'query_2zxeA'
        self.query_aln_big = SeqAlignment(file_name=msa_file_big, query_id=query_big)
        self.query_aln_big.import_alignment()

    def tearDown(self):
        if os.path.exists('./identity.pkl'):
            os.remove('./identity.pkl')
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test')
        if os.path.exists(wetc_test_dir):
            rmtree(wetc_test_dir)

    def test_get_distance_small_identity(self):
        self.query_aln_small.compute_distance_matrix(model='identity')
        identity_calc_current = AlignmentDistanceCalculator()
        start = time()
        identity_dist_current = identity_calc_current.get_distance(self.query_aln_small.alignment)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(self.query_aln_small.distance_matrix.names == identity_dist_current.names)
        diff = np.array(self.query_aln_small.distance_matrix) - np.array(identity_dist_current)
        self.assertTrue(not diff.any())
        identity_calc_official = DistanceCalculator()
        start = time()
        identity_dist_official = identity_calc_official.get_distance(self.query_aln_small.alignment)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(identity_dist_current.names == identity_dist_official.names)
        diff = np.array(identity_dist_current) - np.array(identity_dist_official)
        self.assertTrue(not diff.any())

    def test_get_distance_small_blosum62(self):
        blosum62_calc_current = AlignmentDistanceCalculator(model='blosum62')
        start = time()
        blosum62_dist_current = blosum62_calc_current.get_distance(self.query_aln_small.alignment)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        blosum62_calc_official = DistanceCalculator(model='blosum62')
        start = time()
        blosum62_dist_official = blosum62_calc_official.get_distance(self.query_aln_small.alignment)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(blosum62_dist_current.names == blosum62_dist_official.names)
        diff = np.array(blosum62_dist_current) - np.array(blosum62_dist_official)
        self.assertTrue(not diff.any())

    def test_get_distance_big_identity(self):
        self.query_aln_big.compute_distance_matrix(model='identity')
        identity_calc_current = AlignmentDistanceCalculator()
        start = time()
        identity_dist_current = identity_calc_current.get_distance(self.query_aln_big.alignment)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(self.query_aln_big.distance_matrix.names == identity_dist_current.names)
        diff = np.array(self.query_aln_big.distance_matrix) - np.array(identity_dist_current)
        self.assertTrue(not diff.any())
        identity_calc_official = DistanceCalculator()
        start = time()
        identity_dist_official = identity_calc_official.get_distance(self.query_aln_big.alignment)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(identity_dist_current.names == identity_dist_official.names)
        diff = np.array(identity_dist_current) - np.array(identity_dist_official)
        self.assertTrue(not diff.any())

    def test_get_distance_big_blosum62(self):
        blosum62_calc_current = AlignmentDistanceCalculator(model='blosum62')
        start = time()
        blosum62_dist_current = blosum62_calc_current.get_distance(self.query_aln_big.alignment)
        print('Current implementation took {} min'.format((time() - start) / 60.0))
        blosum62_calc_official = DistanceCalculator(model='blosum62')
        start = time()
        blosum62_dist_official = blosum62_calc_official.get_distance(self.query_aln_big.alignment)
        print('Official implementation took {} min'.format((time() - start) / 60.0))
        self.assertTrue(blosum62_dist_current.names == blosum62_dist_official.names)
        diff = np.array(blosum62_dist_current) - np.array(blosum62_dist_official)
        self.assertTrue(not diff.any())

    def test_get_et_distance_small(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test')
        if not os.path.isdir(wetc_test_dir):
            os.mkdir(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        aln_dist_df, id_dist_df = et_mip_obj.import_distance_matrices(wetc_test_dir)
        aln_dist_array = np.asarray(aln_dist_df, dtype=float)
        id_dist_array = np.asarray(id_dist_df, dtype=float)
        aln_dist_dm1 = SeqAlignment._convert_array_to_distance_matrix(aln_dist_array, list(aln_dist_df.columns))
        id_dist_dm1 = SeqAlignment._convert_array_to_distance_matrix(id_dist_array.T, list(id_dist_df.columns))
        et_calc = AlignmentDistanceCalculator(model='blosum62')
        id_dist_dm2, aln_dist_dm2 = et_calc.get_et_distance(self.query_aln_small.alignment)
        diff_aln_dist = np.array(aln_dist_dm1) - np.array(aln_dist_dm2)
        # from IPython import embed
        # embed()
        # exit()
        # self.assertTrue(not diff_aln_dist.any())
        diff_id_dist = np.abs(np.array(id_dist_dm1) - np.array(id_dist_dm2))
        print(diff_id_dist)
        diff_id_threshold = diff_id_dist < 0.01 # Differences may arise in the third decimal place.
        print(diff_id_threshold)
        from IPython import embed
        embed()
        self.assertTrue(not diff_id_threshold.any())