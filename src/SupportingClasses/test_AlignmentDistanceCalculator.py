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

class TestSeqAlignment(TestCase):

    def setUp(self):
        msa_file_small = '/media/daniel/ExtraDrive1/DataForProjects/ETMIPC/23TestGenes/7hvpA.fa'
        query_small = 'query_7hvpA'
        self.query_aln_small = SeqAlignment(file_name=msa_file_small, query_id=query_small)
        self.query_aln_small.import_alignment()
        msa_file_big = '/media/daniel/ExtraDrive1/DataForProjects/ETMIPC/23TestGenes/2zxeA.fa'
        query_big = 'query_2zxeA'
        self.query_aln_big = SeqAlignment(file_name=msa_file_big, query_id=query_big)
        self.query_aln_big.import_alignment()

    def tearDown(self):
        if os.path.exists('./identity.pkl'):
            os.remove('./identity.pkl')

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
