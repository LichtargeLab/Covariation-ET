import os
import sys
import math
import datetime
import unittest
import numpy as np
import pandas as pd
from time import time
from math import floor
from shutil import rmtree
from random import shuffle
from unittest import TestCase
from scipy.stats import rankdata
from Bio.PDB.Polypeptide import one_to_three
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve
from test_Base import TestBase
from SeqAlignment import SeqAlignment
from ContactScorer import (ContactScorer, surface_plot, heatmap_plot, plot_z_scores, init_compute_w2_ave_sub,
                           compute_w2_ave_sub, init_clustering_z_score, clustering_z_score)
sys.path.append(os.path.abspath('..'))
from EvolutionaryTrace import EvolutionaryTrace


class TestContactScorer(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestContactScorer, cls).setUpClass()
        cls.CONTACT_DISTANCE2 = 16
        cls.query1 = cls.small_structure_id
        cls.aln_file1 = cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln']
        cls.pdb_file1 = cls.data_set.protein_data[cls.small_structure_id]['PDB']
        cls.pdb_chain1 = cls.data_set.protein_data[cls.small_structure_id]['Chain']
        cls.seq_len1 = cls.data_set.protein_data[cls.small_structure_id]['Length']
        cls.query2 = cls.large_structure_id
        cls.aln_file2 = cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln']
        cls.pdb_file2 = cls.data_set.protein_data[cls.large_structure_id]['PDB']
        cls.pdb_chain2 = cls.data_set.protein_data[cls.large_structure_id]['Chain']
        cls.seq_len2 = cls.data_set.protein_data[cls.large_structure_id]['Length']

    def setUp(self):
        # self.query1 = '1c17A'
        # self.aln_file1 = '../Test/1c17A.fa'
        # self.aln_obj1 = SeqAlignment(file_name=self.aln_file1, query_id=self.query1)
        # self.pdb_file1 = '../Test/query_1c17A.pdb'
        # self.pdb_obj1 = PDBReference(pdb_file=self.pdb_file1)
        self.scorer1 = ContactScorer(query=self.query1, seq_alignment=self.aln_file1, pdb_reference=self.pdb_file1,
                                     cutoff=8.0, chain=self.pdb_chain1)
        # self.query2 = '1h1vA'
        # self.aln_file2 = '../Test/1h1vA.fa'
        # self.aln_obj2 = SeqAlignment(file_name=self.aln_file2, query_id=self.query2)
        # self.pdb_file2 = '../Test/query_1h1vA.pdb'
        # self.pdb_obj2 = PDBReference(pdb_file=self.pdb_file2)
        self.scorer2 = ContactScorer(query=self.query2, seq_alignment=self.aln_file2, pdb_reference=self.pdb_file2,
                                     cutoff=8.0, chain=self.pdb_chain2)

    def tearDown(self):
        # del self.query1
        # del self.aln_file1
        # del self.aln_obj1
        # del self.pdb_file1
        # del self.pdb_obj1
        del self.scorer1
        # del self.query2
        # del self.aln_file2
        # del self.aln_obj2
        # del self.pdb_file2
        # del self.pdb_obj2
        del self.scorer2
        # del self.CONTACT_DISTANCE2

    @staticmethod
    def check_precision(mapped_scores, mapped_dists, threshold=0.5, count=None):
        if count is None:
            count = mapped_dists.shape[0]
        ranked_scores = rankdata(-1 * np.array(mapped_scores), method='dense')
        ind = np.where(ranked_scores <= count)
        mapped_scores = mapped_scores[ind]
        preds = (mapped_scores > threshold) * 1.0
        mapped_dists = mapped_dists[ind]
        truth = (mapped_dists <= 8.0) * 1.0
        precision = precision_score(truth, preds)
        return precision

    @staticmethod
    def check_recall(mapped_scores, mapped_dists, threshold=0.5, count=None):
        if count is None:
            count = mapped_dists.shape[0]
        ranked_scores = rankdata(-1 * np.array(mapped_scores), method='dense')
        ind = np.where(ranked_scores <= count)
        mapped_scores = mapped_scores[ind]
        preds = (mapped_scores > threshold) * 1.0
        mapped_dists = mapped_dists[ind]
        truth = (mapped_dists <= 8.0) * 1.0
        precision = recall_score(truth, preds)
        return precision

    @staticmethod
    def check_f1(mapped_scores, mapped_dists, threshold=0.5, count=None):
        if count is None:
            count = mapped_dists.shape[0]
        ranked_scores = rankdata(-1 * np.array(mapped_scores), method='dense')
        ind = np.where(ranked_scores <= count)
        mapped_scores = mapped_scores[ind]
        preds = (mapped_scores > threshold) * 1.0
        mapped_dists = mapped_dists[ind]
        truth = (mapped_dists <= 8.0) * 1.0
        precision = f1_score(truth, preds)
        return precision

    def _et_calcDist(self, atoms1, atoms2):
        """return smallest distance (squared) between two groups of atoms"""
        # (not distant by more than ~100 A)
        # mind2=CONTACT_DISTANCE2+100
        c1 = atoms1[0]  # atoms must not be empty
        c2 = atoms2[0]
        mind2 = (c1[0] - c2[0]) * (c1[0] - c2[0]) + \
                (c1[1] - c2[1]) * (c1[1] - c2[1]) + \
                (c1[2] - c2[2]) * (c1[2] - c2[2])
        for c1 in atoms1:
            for c2 in atoms2:
                d2 = (c1[0] - c2[0]) * (c1[0] - c2[0]) + \
                     (c1[1] - c2[1]) * (c1[1] - c2[1]) + \
                     (c1[2] - c2[2]) * (c1[2] - c2[2])
                if d2 < mind2:
                    mind2 = d2
        return mind2  # Square of distance between most proximate atoms

    def _et_computeAdjacency(self, chain, mapping):
        """Compute the pairs of contacting residues
        A(i,j) implemented as a hash of hash of residue numbers"""
        three2one = {
            "ALA": 'A',
            "ARG": 'R',
            "ASN": 'N',
            "ASP": 'D',
            "CYS": 'C',
            "GLN": 'Q',
            "GLU": 'E',
            "GLY": 'G',
            "HIS": 'H',
            "ILE": 'I',
            "LEU": 'L',
            "LYS": 'K',
            "MET": 'M',
            "PHE": 'F',
            "PRO": 'P',
            "SER": 'S',
            "THR": 'T',
            "TRP": 'W',
            "TYR": 'Y',
            "VAL": 'V',
            "A": "A",
            "G": "G",
            "T": "T",
            "U": "U",
            "C": "C", }

        ResAtoms = {}
        for residue in chain:
            try:
                aa = three2one[residue.get_resname()]
            except KeyError:
                continue
            # resi = residue.get_id()[1]
            resi = mapping[residue.get_id()[1]]
            for atom in residue:
                try:
                    # ResAtoms[resi - 1].append(atom.coord)
                    ResAtoms[resi].append(atom.coord)
                except KeyError:
                    # ResAtoms[resi - 1] = [atom.coord]
                    ResAtoms[resi] = [atom.coord]
        A = {}
        for resi in ResAtoms.keys():
            for resj in ResAtoms.keys():
                if resi < resj:
                    curr_dist = self._et_calcDist(ResAtoms[resi], ResAtoms[resj])
                    if curr_dist < self.CONTACT_DISTANCE2:
                        try:
                            A[resi][resj] = 1
                        except KeyError:
                            A[resi] = {resj: 1}
        return A, ResAtoms

    @staticmethod
    def _et_calc_w2_sub_problems(A, bias=1):
        """Calculate w2_ave components for calculation z-score (z_S) for residue selection reslist=[1,2,...]
        z_S = (w-<w>_S)/sigma_S
        The steps are:
        1. Calculate Selection Clustering Weight (SCW) 'w'
        2. Calculate mean SCW (<w>_S) in the ensemble of random
        selections of len(reslist) residues
        3. Calculate mean square SCW (<w^2>_S) and standard deviation (sigma_S)
        Reference: Mihalek, Res, Yao, Lichtarge (2003)

        reslist - a list of int's of protein residue numbers, e.g. ET residues
        L - length of protein
        A - the adjacency matrix implemented as a dictionary. The first key is related to the second key by resi<resj.
        bias - option to calculate with bias or nobias (j-i factor)"""
        part1 = 0.0
        part2 = 0.0
        part3 = 0.0
        if bias == 1:
            for resi, neighborsj in A.items():
                for resj in neighborsj:
                    for resk, neighborsl in A.items():
                        for resl in neighborsl:
                            if (resi == resk and resj == resl) or \
                                    (resi == resl and resj == resk):
                                part1 += (resj - resi) * (resl - resk)
                            elif (resi == resk) or (resj == resl) or \
                                    (resi == resl) or (resj == resk):
                                part2 += (resj - resi) * (resl - resk)
                            else:
                                part3 += (resj - resi) * (resl - resk)
        elif bias == 0:
            for resi, neighborsj in A.items():
                for resj in neighborsj:
                    for resk, neighborsl in A.items():
                        for resl in neighborsl:
                            if (resi == resk and resj == resl) or \
                                    (resi == resl and resj == resk):
                                part1 += 1
                            elif (resi == resk) or (resj == resl) or \
                                    (resi == resl) or (resj == resk):
                                part2 += 1
                            else:
                                part3 += 1
        return part1, part2, part3

    @staticmethod
    def _et_calcZScore(reslist, L, A, bias=1):
        """Calculate z-score (z_S) for residue selection reslist=[1,2,...]
        z_S = (w-<w>_S)/sigma_S
        The steps are:
        1. Calculate Selection Clustering Weight (SCW) 'w'
        2. Calculate mean SCW (<w>_S) in the ensemble of random
        selections of len(reslist) residues
        3. Calculate mean square SCW (<w^2>_S) and standard deviation (sigma_S)
        Reference: Mihalek, Res, Yao, Lichtarge (2003)

        reslist - a list of int's of protein residue numbers, e.g. ET residues
        L - length of protein
        A - the adjacency matrix implemented as a dictionary. The first key is related to the second key by resi<resj.
        bias - option to calculate with bias or nobias (j-i factor)"""
        w = 0
        if bias == 1:
            for resi in reslist:
                for resj in reslist:
                    if resi < resj:
                        try:
                            Aij = A[resi][resj]  # A(i,j)==1
                            w += (resj - resi)
                        except KeyError:
                            pass
        elif bias == 0:
            for resi in reslist:
                for resj in reslist:
                    if resi < resj:
                        try:
                            Aij = A[resi][resj]  # A(i,j)==1
                            w += 1
                        except KeyError:
                            pass
        M = len(reslist)
        pi1 = M * (M - 1.0) / (L * (L - 1.0))
        pi2 = pi1 * (M - 2.0) / (L - 2.0)
        pi3 = pi2 * (M - 3.0) / (L - 3.0)
        w_ave = 0
        w2_ave = 0
        if bias == 1:
            for resi, neighborsj in A.items():
                for resj in neighborsj:
                    w_ave += (resj - resi)
                    for resk, neighborsl in A.items():
                        for resl in neighborsl:
                            if (resi == resk and resj == resl) or \
                                    (resi == resl and resj == resk):
                                w2_ave += pi1 * (resj - resi) * (resl - resk)
                            elif (resi == resk) or (resj == resl) or \
                                    (resi == resl) or (resj == resk):
                                w2_ave += pi2 * (resj - resi) * (resl - resk)
                            else:
                                w2_ave += pi3 * (resj - resi) * (resl - resk)
        elif bias == 0:
            for resi, neighborsj in A.items():
                w_ave += len(neighborsj)
                for resj in neighborsj:
                    for resk, neighborsl in A.items():
                        for resl in neighborsl:
                            if (resi == resk and resj == resl) or \
                                    (resi == resl and resj == resk):
                                w2_ave += pi1
                            elif (resi == resk) or (resj == resl) or \
                                    (resi == resl) or (resj == resk):
                                w2_ave += pi2
                            else:
                                w2_ave += pi3
        w_ave = w_ave * pi1
        sigma = math.sqrt(w2_ave - w_ave * w_ave)
        if sigma == 0:
            return M, L, pi1, pi2, pi3, 'NA', w, w_ave, w2_ave, sigma
        return M, L, pi1, pi2, pi3, (w - w_ave) / sigma, w, w_ave, w2_ave, sigma

    def all_z_scores(self, A, L, bias, res_i, res_j, scores):
        data = {'Res_i': res_i, 'Res_j': res_j, 'Covariance_Score': scores, 'Z-Score': [], 'W': [], 'W_Ave': [],
                'W2_Ave': [], 'Sigma': [], 'Num_Residues': []}
        res_list = []
        res_set = set()
        prev_size = 0
        prev_score = None
        for i in range(len(scores)):
            curr_i = res_i[i]
            if curr_i not in res_set:
                res_list.append(curr_i)
                res_set.add(curr_i)
            curr_j = res_j[i]
            if curr_j not in res_set:
                res_list.append(curr_j)
                res_set.add(curr_j)
            if len(res_set) == prev_size:
                score_data = prev_score
            else:
                score_data = self._et_calcZScore(reslist=res_list, L=L, A=A, bias=bias)
            data['Z-Score'].append(score_data[0])
            data['W'].append(score_data[1])
            data['W_Ave'].append(score_data[2])
            data['W2_Ave'].append(score_data[3])
            data['Sigma'].append(score_data[4])
            data['Num_Residues'].append(len(res_list))
            prev_size = len(res_set)
            prev_score = score_data
        return pd.DataFrame(data)

    # def test_init(self):
    #     aln = SeqAlignment(query_id=self.query1, file_name=self.aln_file1)
    #     scorer = ContactScorer(query=self.query1, seq_alignment=aln, pdb_reference=self.pdb_file1,
    #                            cutoff=8.0, chain=self.pdb_chain1)
    #
    # def test_1a___init(self):
    #     with self.assertRaises(TypeError):
    #         ContactScorer()
    #     self.assertEqual(self.scorer1.query_alignment, os.path.abspath(self.aln_file1))
    #     self.assertEqual(self.scorer1.query_structure, os.path.abspath(self.pdb_file1))
    #     self.assertEqual(self.scorer1.cutoff, 8.0)
    #     self.assertEqual(self.scorer1.best_chain, self.pdb_chain1)
    #     self.assertIsNone(self.scorer1.query_pdb_mapping)
    #     self.assertIsNone(self.scorer1._specific_mapping)
    #     self.assertIsNone(self.scorer1.distances)
    #     self.assertIsNone(self.scorer1.dist_type)
    #
    # def test_1b___init(self):
    #     with self.assertRaises(TypeError):
    #         ContactScorer()
    #     self.assertEqual(self.scorer2.query_alignment, os.path.abspath(self.aln_file2))
    #     self.assertEqual(self.scorer2.query_structure, os.path.abspath(self.pdb_file2))
    #     self.assertEqual(self.scorer2.cutoff, 8.0)
    #     self.assertEqual(self.scorer2.best_chain, self.pdb_chain2)
    #     self.assertIsNone(self.scorer2.query_pdb_mapping)
    #     self.assertIsNone(self.scorer2._specific_mapping)
    #     self.assertIsNone(self.scorer2.distances)
    #     self.assertIsNone(self.scorer2.dist_type)
    #
    # def test_1c___init(self):
    #     with self.assertRaises(TypeError):
    #         ContactScorer()
    #     eval1 = ContactScorer(query=self.query1, seq_alignment=self.aln_file1, pdb_reference=self.pdb_file1, cutoff=8.0)
    #     self.assertEqual(eval1.query_alignment, os.path.abspath(self.aln_file1))
    #     self.assertEqual(eval1.query_structure, os.path.abspath(self.pdb_file1))
    #     self.assertEqual(eval1.cutoff, 8.0)
    #     self.assertIsNone(eval1.best_chain)
    #     self.assertIsNone(eval1.query_pdb_mapping)
    #     self.assertIsNone(eval1._specific_mapping)
    #     self.assertIsNone(eval1.distances)
    #     self.assertIsNone(eval1.dist_type)
    #
    # def test_1d___init(self):
    #     with self.assertRaises(TypeError):
    #         ContactScorer()
    #     eval2 = ContactScorer(query=self.query2, seq_alignment=self.aln_file2, pdb_reference=self.pdb_file2, cutoff=8.0)
    #     self.assertEqual(eval2.query_alignment, os.path.abspath(self.aln_file2))
    #     self.assertEqual(eval2.query_structure, os.path.abspath(self.pdb_file2))
    #     self.assertEqual(eval2.cutoff, 8.0)
    #     self.assertIsNone(eval2.best_chain)
    #     self.assertIsNone(eval2.query_pdb_mapping)
    #     self.assertIsNone(eval2._specific_mapping)
    #     self.assertIsNone(eval2.distances)
    #     self.assertIsNone(eval2.dist_type)

    # def test_2a___str(self):
    #     with self.assertRaises(ValueError):
    #         str(self.scorer1)
    #     self.scorer1.fit()
    #     expected_str1 = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
    #         self.seq_len1, 1, self.pdb_chain1)
    #     self.assertEqual(str(self.scorer1), expected_str1)
    #
    # def test_2b___str(self):
    #     with self.assertRaises(ValueError):
    #         str(self.scorer2)
    #     self.scorer2.fit()
    #     expected_str2 = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
    #         self.seq_len2, 1, self.pdb_chain2)
    #     self.assertEqual(str(self.scorer2), expected_str2)
    #
    # def test_2c___str(self):
    #     eval1 = ContactScorer(query=self.query1, seq_alignment=self.aln_file1, pdb_reference=self.pdb_file1, cutoff=8.0)
    #     with self.assertRaises(ValueError):
    #         str(eval1)
    #     eval1.fit()
    #     expected_str1 = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
    #         self.seq_len1, 1, self.pdb_chain1)
    #     self.assertEqual(str(eval1), expected_str1)
    #
    # def test_2d___str(self):
    #     eval2 = ContactScorer(query=self.query2, seq_alignment=self.aln_file2, pdb_reference=self.pdb_file2, cutoff=8.0)
    #     with self.assertRaises(ValueError):
    #         str(eval2)
    #     eval2.fit()
    #     expected_str2 = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
    #         self.seq_len2, 1, self.pdb_chain2)
    #     self.assertEqual(str(eval2), expected_str2)

    # def test_3a_fit(self):
    #     self.assertEqual(self.scorer1.query_alignment, os.path.abspath(self.aln_file1))
    #     self.assertEqual(self.scorer1.query_structure, os.path.abspath(self.pdb_file1))
    #     self.scorer1.fit()
    #     self.assertNotEqual(self.scorer1.query_alignment, self.aln_file1)
    #     self.assertNotEqual(self.scorer1.query_structure, self.pdb_file1)
    #     self.assertEqual(self.scorer1.best_chain, 'A')
    #     self.assertEqual(self.scorer1.query_pdb_mapping,
    #                      {i + 18: i for i in range(len(self.scorer1.query_structure.seq[self.scorer1.best_chain]))})
    #     self.scorer1.best_chain = None
    #     self.scorer1.fit()
    #     self.assertNotEqual(self.scorer1.query_alignment, self.aln_file1)
    #     self.assertNotEqual(self.scorer1.query_structure, self.pdb_file1)
    #     self.assertEqual(self.scorer1.best_chain, 'A')
    #     self.assertEqual(self.scorer1.query_pdb_mapping,
    #                      {i + 18: i for i in range(len(self.scorer1.query_structure.seq[self.scorer1.best_chain]))})
    #     self.scorer1.query_pdb_mapping = None
    #     self.scorer1.fit()
    #     self.assertNotEqual(self.scorer1.query_alignment, self.aln_file1)
    #     self.assertNotEqual(self.scorer1.query_structure, self.pdb_file1)
    #     self.assertEqual(self.scorer1.best_chain, 'A')
    #     self.assertEqual(self.scorer1.query_pdb_mapping,
    #                      {i + 18: i for i in range(len(self.scorer1.query_structure.seq[self.scorer1.best_chain]))})
    #
    # def test_3b_fit(self):
    #     self.assertEqual(self.scorer2.query_alignment, os.path.abspath(self.aln_file2))
    #     self.assertEqual(self.scorer2.query_structure, os.path.abspath(self.pdb_file2))
    #     self.scorer2.fit()
    #     self.assertNotEqual(self.scorer2.query_alignment, self.aln_file2)
    #     self.assertNotEqual(self.scorer2.query_structure, self.pdb_file2)
    #     self.assertEqual(self.scorer2.best_chain, 'A')
    #     self.assertEqual(self.scorer2.query_pdb_mapping,
    #                      {i + 16: i for i in range(len(self.scorer2.query_structure.seq[self.scorer2.best_chain]))})
    #     self.scorer2.best_chain = None
    #     self.scorer2.fit()
    #     self.assertNotEqual(self.scorer2.query_alignment, self.aln_file2)
    #     self.assertNotEqual(self.scorer2.query_structure, self.pdb_file2)
    #     self.assertEqual(self.scorer2.best_chain, 'A')
    #     self.assertEqual(self.scorer2.query_pdb_mapping,
    #                      {i + 16: i for i in range(len(self.scorer2.query_structure.seq[self.scorer2.best_chain]))})
    #     self.scorer2.query_pdb_mapping = None
    #     self.scorer2.fit()
    #     self.assertNotEqual(self.scorer2.query_alignment, self.aln_file2)
    #     self.assertNotEqual(self.scorer2.query_structure, self.pdb_file2)
    #     self.assertEqual(self.scorer2.best_chain, 'A')
    #     self.assertEqual(self.scorer2.query_pdb_mapping,
    #                      {i + 16: i for i in range(len(self.scorer2.query_structure.seq[self.scorer2.best_chain]))})
    #
    # def test_3c_fit(self):
    #     eval1 = ContactScorer(query=self.query1, seq_alignment=self.aln_file1, pdb_reference=self.pdb_file1, cutoff=8.0)
    #     self.assertEqual(eval1.query_alignment, os.path.abspath(self.aln_file1))
    #     self.assertEqual(eval1.query_structure, os.path.abspath(self.pdb_file1))
    #     eval1.fit()
    #     self.assertNotEqual(eval1.query_alignment, self.aln_file1)
    #     self.assertNotEqual(eval1.query_structure, self.pdb_file1)
    #     self.assertEqual(eval1.best_chain, 'A')
    #     self.assertEqual(eval1.query_pdb_mapping,
    #                      {i + 18: i for i in range(len(eval1.query_structure.seq[eval1.best_chain]))})
    #     eval1.best_chain = None
    #     eval1.fit()
    #     self.assertNotEqual(eval1.query_alignment, self.aln_file1)
    #     self.assertNotEqual(eval1.query_structure, self.pdb_file1)
    #     self.assertEqual(eval1.best_chain, 'A')
    #     self.assertEqual(eval1.query_pdb_mapping,
    #                      {i + 18: i for i in range(len(eval1.query_structure.seq[eval1.best_chain]))})
    #     eval1.query_pdb_mapping = None
    #     eval1.fit()
    #     self.assertNotEqual(eval1.query_alignment, self.aln_file1)
    #     self.assertNotEqual(eval1.query_structure, self.pdb_file1)
    #     self.assertEqual(eval1.best_chain, 'A')
    #     self.assertEqual(eval1.query_pdb_mapping,
    #                      {i + 18: i for i in range(len(eval1.query_structure.seq[eval1.best_chain]))})
    #
    # def test_3d_fit(self):
    #     eval2 = ContactScorer(query=self.query2, seq_alignment=self.aln_file2, pdb_reference=self.pdb_file2, cutoff=8.0)
    #     self.assertEqual(eval2.query_alignment, os.path.abspath(self.aln_file2))
    #     self.assertEqual(eval2.query_structure, os.path.abspath(self.pdb_file2))
    #     eval2.fit()
    #     self.assertNotEqual(eval2.query_alignment, self.aln_file2)
    #     self.assertNotEqual(eval2.query_structure, self.pdb_file2)
    #     self.assertEqual(eval2.best_chain, 'A')
    #     self.assertEqual(eval2.query_pdb_mapping,
    #                      {i + 16: i for i in range(len(eval2.query_structure.seq[eval2.best_chain]))})
    #     eval2.best_chain = None
    #     eval2.fit()
    #     self.assertNotEqual(eval2.query_alignment, self.aln_file2)
    #     self.assertNotEqual(eval2.query_structure, self.pdb_file2)
    #     self.assertEqual(eval2.best_chain, 'A')
    #     self.assertEqual(eval2.query_pdb_mapping,
    #                      {i + 16: i for i in range(len(eval2.query_structure.seq[eval2.best_chain]))})
    #     eval2.query_pdb_mapping = None
    #     eval2.fit()
    #     self.assertNotEqual(eval2.query_alignment, self.aln_file2)
    #     self.assertNotEqual(eval2.query_structure, self.pdb_file2)
    #     self.assertEqual(eval2.best_chain, 'A')
    #     self.assertEqual(eval2.query_pdb_mapping,
    #                      {i + 16: i for i in range(len(eval2.query_structure.seq[eval2.best_chain]))})

    # def test_4a__get_all_coords(self):
    #     self.scorer1.fit()
    #     residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
    #     expected1 = np.vstack([[24.704, 20.926, 27.944], [25.408, 20.195, 26.922], [24.487, 19.147, 26.324],
    #                            [23.542, 18.689, 26.993], [26.589, 19.508, 27.519], [26.344, 18.392, 28.442],
    #                            [27.689, 17.685, 28.514], [27.941, 16.866, 27.267], [29.154, 16.092, 27.419]])
    #     measured1 = np.vstack(ContactScorer._get_all_coords(residue1))
    #     diff = measured1 - expected1
    #     not_passing = diff > 1E-5
    #     self.assertFalse(not_passing.any())
    # 
    # def test_4b__get_all_coords(self):
    #     self.scorer2.fit()
    #     residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][1]
    #     expected2 = np.vstack([[26.432, 44.935, 26.052], [25.921, 43.597, 25.862], [25.159, 43.203, 24.568],
    #                            [23.936, 43.424, 24.593], [25.050, 43.281, 27.093], [25.777, 43.092, 28.306]])
    #     measured2 = np.vstack(ContactScorer._get_all_coords(residue2))
    #     diff = measured2 - expected2
    #     not_passing = diff > 1E-5
    #     self.assertFalse(not_passing.any())
    # 
    # def test_4c__get_c_alpha_coords(self):
    #     self.scorer1.fit()
    #     residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
    #     expected1 = np.vstack([[25.408, 20.195, 26.922]])
    #     measured1 = np.vstack(ContactScorer._get_c_alpha_coords(residue1))
    #     diff = measured1 - expected1
    #     not_passing = diff > 1E-5
    #     self.assertFalse(not_passing.any())
    # 
    # def test_4d__get_c_alpha_coords(self):
    #     self.scorer2.fit()
    #     residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][1]
    #     expected2 = np.vstack([[25.921, 43.597, 25.862]])
    #     measured2 = np.vstack(ContactScorer._get_c_alpha_coords(residue2))
    #     diff = measured2 - expected2
    #     not_passing = diff > 1E-5
    #     self.assertFalse(not_passing.any())
    # 
    # def test_4e__get_c_beta_coords(self):
    #     self.scorer1.fit()
    #     residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
    #     expected1 = np.vstack([[26.589, 19.508, 27.519]])
    #     measured1 = np.vstack(ContactScorer._get_c_beta_coords(residue1))
    #     diff = measured1 - expected1
    #     not_passing = diff > 1E-5
    #     self.assertFalse(not_passing.any())
    # 
    # def test_4f__get_c_beta_coords(self):
    #     self.scorer2.fit()
    #     residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][1]
    #     expected2 = np.vstack([[25.050, 43.281, 27.093]])
    #     measured2 = np.vstack(ContactScorer._get_c_beta_coords(residue2))
    #     diff = measured2 - expected2
    #     not_passing = diff > 1E-5
    #     self.assertFalse(not_passing.any())
    # 
    # def test_4g__get_coords(self):
    #     self.scorer1.fit()
    #     residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
    #     expected1a = np.vstack([[24.704, 20.926, 27.944], [25.408, 20.195, 26.922], [24.487, 19.147, 26.324],
    #                            [23.542, 18.689, 26.993], [26.589, 19.508, 27.519], [26.344, 18.392, 28.442],
    #                            [27.689, 17.685, 28.514], [27.941, 16.866, 27.267], [29.154, 16.092, 27.419]])
    #     measured1a = np.vstack(ContactScorer._get_coords(residue1, method='Any'))
    #     self.assertFalse(((measured1a - expected1a) > 1E-5).any())
    #     expected1b = np.vstack([[25.408, 20.195, 26.922]])
    #     measured1b = np.vstack(ContactScorer._get_coords(residue1, method='CA'))
    #     self.assertFalse(((measured1b - expected1b) > 1E-5).any())
    #     expected1c = np.vstack([[26.589, 19.508, 27.519]])
    #     measured1c = np.vstack(ContactScorer._get_coords(residue1, method='CB'))
    #     self.assertFalse(((measured1c - expected1c) > 1E-5).any())
    # 
    # def test_4h__get_coords(self):
    #     self.scorer2.fit()
    #     residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][1]
    #     expected2a = np.vstack([[26.432, 44.935, 26.052], [25.921, 43.597, 25.862], [25.159, 43.203, 24.568],
    #                            [23.936, 43.424, 24.593], [25.050, 43.281, 27.093], [25.777, 43.092, 28.306]])
    #     measured2a = np.vstack(ContactScorer._get_coords(residue2, method='Any'))
    #     self.assertFalse(((measured2a - expected2a) > 1E-5).any())
    #     expected2b = np.vstack([[25.921, 43.597, 25.862]])
    #     measured2b = np.vstack(ContactScorer._get_c_alpha_coords(residue2))
    #     self.assertFalse(((measured2b - expected2b) > 1E-5).any())
    #     expected2c = np.vstack([[25.050, 43.281, 27.093]])
    #     measured2c = np.vstack(ContactScorer._get_c_beta_coords(residue2))
    #     self.assertFalse(((measured2c - expected2c) > 1E-5).any())

    # def test_5a_measure_distance(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     self.assertEqual(self.scorer1.dist_type, 'Any')
    #     residue1a = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
    #     residue1b = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][2]
    #     pos1a = ContactScorer._get_all_coords(residue1a)
    #     pos1b = ContactScorer._get_all_coords(residue1b)
    #     expected1a = None
    #     for i in range(len(pos1a)):
    #         for j in range(len(pos1b)):
    #             curr_dist = np.sqrt(np.power(pos1a[i][0] - pos1b[j][0], 2) + np.power(pos1a[i][1] - pos1b[j][1], 2) +
    #                                 np.power(pos1a[i][2] - pos1b[j][2], 2))
    #             if (expected1a is None) or (curr_dist < expected1a):
    #                 expected1a = curr_dist
    #     self.assertLess(expected1a - self.scorer1.distances[0, 1], 1E-5)
    #     self.scorer1.measure_distance(method='CA')
    #     self.assertEqual(self.scorer1.dist_type, 'CA')
    #     ca_atom1a = residue1a['CA'].get_coord()
    #     ca_atom1b = residue1b['CA'].get_coord()
    #     expected1b = np.sqrt(np.power(ca_atom1a[0] - ca_atom1b[0], 2) + np.power(ca_atom1a[1] - ca_atom1b[1], 2) +
    #                          np.power(ca_atom1a[2] - ca_atom1b[2], 2))
    #     self.assertLess(expected1b - self.scorer1.distances[0, 1], 1E-5)
    #     self.scorer1.measure_distance(method='CB')
    #     self.assertEqual(self.scorer1.dist_type, 'CB')
    #     cb_atom1a = residue1a['CB'].get_coord()
    #     cb_atom1b = residue1b['CB'].get_coord()
    #     expected1c = np.sqrt(np.power(cb_atom1a[0] - cb_atom1b[0], 2) + np.power(cb_atom1a[1] - cb_atom1b[1], 2) +
    #                          np.power(cb_atom1a[2] - cb_atom1b[2], 2))
    #     self.assertLess(expected1c - self.scorer1.distances[0, 1], 1E-5)
    #
    # def test_5b_measure_distance(self):
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     self.assertEqual(self.scorer2.dist_type, 'Any')
    #     residue2a = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][1]
    #     residue2b = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][2]
    #     pos2a = ContactScorer._get_all_coords(residue2a)
    #     pos2b = ContactScorer._get_all_coords(residue2b)
    #     expected2a = None
    #     for i in range(len(pos2a)):
    #         for j in range(len(pos2b)):
    #             curr_dist = np.sqrt(np.power(pos2a[i][0] - pos2b[j][0], 2) + np.power(pos2a[i][1] - pos2b[j][1], 2) +
    #                                 np.power(pos2a[i][2] - pos2b[j][2], 2))
    #             if (expected2a is None) or (curr_dist < expected2a):
    #                 expected2a = curr_dist
    #     self.assertLess(expected2a - self.scorer2.distances[0, 1], 1e-6)
    #     self.scorer2.measure_distance(method='CA')
    #     self.assertEqual(self.scorer2.dist_type, 'CA')
    #     ca_atom2a = residue2a['CA'].get_coord()
    #     ca_atom2b = residue2b['CA'].get_coord()
    #     expected2b = np.sqrt(np.power(ca_atom2a[0] - ca_atom2b[0], 2) + np.power(ca_atom2a[1] - ca_atom2b[1], 2) +
    #                          np.power(ca_atom2a[2] - ca_atom2b[2], 2))
    #     self.assertLess(expected2b - self.scorer2.distances[0, 1], 1e-6)
    #     self.scorer2.measure_distance(method='CB')
    #     self.assertEqual(self.scorer2.dist_type, 'CB')
    #     cb_atom2a = residue2a['CB'].get_coord()
    #     cb_atom2b = residue2b['CB'].get_coord()
    #     expected2c = np.sqrt(np.power(cb_atom2a[0] - cb_atom2b[0], 2) + np.power(cb_atom2a[1] - cb_atom2b[1], 2) +
    #                          np.power(cb_atom2a[2] - cb_atom2b[2], 2))
    #     self.assertLess(expected2c - self.scorer2.distances[0, 1], 1e-6)
    #
    # def test_5c_measure_distance(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     self.assertEqual(self.scorer1.dist_type, 'Any')
    #     residue_coords = {}
    #     size1 = len(self.scorer1.query_structure.seq[self.scorer1.best_chain])
    #     dists = np.zeros((size1, size1))
    #     dists2 = np.zeros((size1, size1))
    #     counter = 0
    #     for res_num in self.scorer1.query_structure.residue_pos[self.scorer1.best_chain]:
    #         residue = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][res_num]
    #         coords = self.scorer1._get_all_coords(residue)
    #         residue_coords[counter] = coords
    #         for residue2 in residue_coords:
    #             if residue2 == counter:
    #                 continue
    #             else:
    #                 dist = self._et_calcDist(coords, residue_coords[residue2])
    #                 dist2 = np.sqrt(dist)
    #                 dists[counter, residue2] = dist
    #                 dists[residue2, counter] = dist
    #                 dists2[counter, residue2] = dist2
    #                 dists2[residue2, counter] = dist2
    #         counter += 1
    #     distance_diff = np.square(self.scorer1.distances) - dists
    #     self.assertLess(np.max(distance_diff), 1e-3)
    #     adj_diff = ((np.square(self.scorer1.distances)[np.nonzero(distance_diff)] < self.CONTACT_DISTANCE2) ^
    #                 (dists[np.nonzero(distance_diff)] < self.CONTACT_DISTANCE2))
    #     self.assertEqual(np.sum(adj_diff), 0)
    #     self.assertEqual(len(np.nonzero(adj_diff)[0]), 0)
    #     distance_diff2 = self.scorer1.distances - dists2
    #     self.assertEqual(np.sum(distance_diff2), 0.0)
    #     self.assertEqual(len(np.nonzero(distance_diff2)[0]), 0.0)
    #
    # def test_5d_measure_distance(self):
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     self.assertEqual(self.scorer2.dist_type, 'Any')
    #     residue_coords = {}
    #     size2 = len(self.scorer2.query_structure.seq[self.scorer2.best_chain])
    #     dists = np.zeros((size2, size2))
    #     dists2 = np.zeros((size2, size2))
    #     counter = 0
    #     for res_num in self.scorer2.query_structure.residue_pos[self.scorer2.best_chain]:
    #         residue = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][res_num]
    #         coords = self.scorer2._get_all_coords(residue)
    #         residue_coords[counter] = coords
    #         for residue2 in residue_coords:
    #             if residue2 == counter:
    #                 continue
    #             else:
    #                 dist = self._et_calcDist(coords, residue_coords[residue2])
    #                 dist2 = np.sqrt(dist)
    #                 dists[counter, residue2] = dist
    #                 dists[residue2, counter] = dist
    #                 dists2[counter, residue2] = dist2
    #                 dists2[residue2, counter] = dist2
    #         counter += 1
    #     distance_diff = np.square(self.scorer2.distances) - dists
    #     self.assertLess(np.max(distance_diff), 2E-3)
    #     adj_diff = ((np.square(self.scorer2.distances)[np.nonzero(distance_diff)] < self.CONTACT_DISTANCE2) ^
    #                 (dists[np.nonzero(distance_diff)] < self.CONTACT_DISTANCE2))
    #     self.assertEqual(np.sum(adj_diff), 0)
    #     self.assertEqual(len(np.nonzero(adj_diff)[0]), 0)
    #     distance_diff2 = self.scorer2.distances - dists2
    #     self.assertEqual(np.sum(distance_diff2), 0.0)
    #     self.assertEqual(len(np.nonzero(distance_diff2)[0]), 0.0)

    # def test_6a_find_pairs_by_separation(self):
    #     self.scorer1.fit()
    #     with self.assertRaises(ValueError):
    #         self.scorer1.find_pairs_by_separation(category='Wide')
    #     expected1 = {'Any': [], 'Neighbors': [], 'Short': [], 'Medium': [], 'Long': []}
    #     for i in range(self.seq_len1):
    #         for j in range(i + 1, self.seq_len1):
    #             pair = (i, j)
    #             separation = j - i
    #             if (separation >= 1) and (separation < 6):
    #                 expected1['Neighbors'].append(pair)
    #             if (separation >= 6) and (separation < 13):
    #                 expected1['Short'].append(pair)
    #             if (separation >= 13) and (separation < 24):
    #                 expected1['Medium'].append(pair)
    #             if separation >= 24:
    #                 expected1['Long'].append(pair)
    #             expected1['Any'].append(pair)
    #     self.assertEqual(self.scorer1.find_pairs_by_separation(category='Any'), expected1['Any'])
    #     self.assertEqual(self.scorer1.find_pairs_by_separation(category='Neighbors'), expected1['Neighbors'])
    #     self.assertEqual(self.scorer1.find_pairs_by_separation(category='Short'), expected1['Short'])
    #     self.assertEqual(self.scorer1.find_pairs_by_separation(category='Medium'), expected1['Medium'])
    #     self.assertEqual(self.scorer1.find_pairs_by_separation(category='Long'), expected1['Long'])
    #
    # def test_6b_find_pairs_by_separation(self):
    #     self.scorer2.fit()
    #     with self.assertRaises(ValueError):
    #         self.scorer2.find_pairs_by_separation(category='Small')
    #     expected2 = {'Any': [], 'Neighbors': [], 'Short': [], 'Medium': [], 'Long': []}
    #     for i in range(self.seq_len2):
    #         for j in range(i + 1, self.seq_len2):
    #             pair = (i, j)
    #             separation = j - i
    #             if (separation >= 1) and (separation < 6):
    #                 expected2['Neighbors'].append(pair)
    #             if (separation >= 6) and (separation < 13):
    #                 expected2['Short'].append(pair)
    #             if (separation >= 13) and (separation < 24):
    #                 expected2['Medium'].append(pair)
    #             if separation >= 24:
    #                 expected2['Long'].append(pair)
    #             expected2['Any'].append(pair)
    #     self.assertEqual(self.scorer2.find_pairs_by_separation(category='Any'), expected2['Any'])
    #     self.assertEqual(self.scorer2.find_pairs_by_separation(category='Neighbors'), expected2['Neighbors'])
    #     self.assertEqual(self.scorer2.find_pairs_by_separation(category='Short'), expected2['Short'])
    #     self.assertEqual(self.scorer2.find_pairs_by_separation(category='Medium'), expected2['Medium'])
    #     self.assertEqual(self.scorer2.find_pairs_by_separation(category='Long'), expected2['Long'])

    # def test_7a__map_predictions_to_pdb(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='CB')
    #     scores1 = np.random.rand(self.seq_len1, self.seq_len1)
    #     scores1[np.tril_indices(self.seq_len1, 1)] = 0
    #     scores1 += scores1.T
    #     scores_mapped1a, dists_mapped1a = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Any')
    #     pairs1a = self.scorer1.find_pairs_by_separation(category='Any')
    #     expected_scores1a = scores1[[x[0] for x in pairs1a if x[0] in self.scorer1.query_pdb_mapping and
    #                                  x[1] in self.scorer1.query_pdb_mapping],
    #                                 [x[1] for x in pairs1a if x[0] in self.scorer1.query_pdb_mapping and
    #                                  x[1] in self.scorer1.query_pdb_mapping]]
    #     expected_dists1a = self.scorer1.distances[[self.scorer1.query_pdb_mapping[x[0]] for x in pairs1a
    #                                                if x[0] in self.scorer1.query_pdb_mapping and
    #                                                x[1] in self.scorer1.query_pdb_mapping],
    #                                               [self.scorer1.query_pdb_mapping[x[1]] for x in pairs1a
    #                                                if x[0] in self.scorer1.query_pdb_mapping and
    #                                                x[1] in self.scorer1.query_pdb_mapping]]
    #     self.assertLess(np.sum(expected_scores1a - scores_mapped1a), 1e-5)
    #     self.assertLess(np.sum(expected_dists1a - dists_mapped1a), 1e-5)
    #     scores_mapped1b, dists_mapped1b = self.scorer1._map_predictions_to_pdb(predictions=scores1,
    #                                                                            category='Neighbors')
    #     pairs1b = self.scorer1.find_pairs_by_separation(category='Neighbors')
    #     expected_scores1b = scores1[[x[0] for x in pairs1b if x[0] in self.scorer1.query_pdb_mapping and
    #                                  x[1] in self.scorer1.query_pdb_mapping],
    #                                 [x[1] for x in pairs1b if x[0] in self.scorer1.query_pdb_mapping and
    #                                  x[1] in self.scorer1.query_pdb_mapping]]
    #     expected_dists1b = self.scorer1.distances[[self.scorer1.query_pdb_mapping[x[0]] for x in pairs1b
    #                                                if x[0] in self.scorer1.query_pdb_mapping and
    #                                                x[1] in self.scorer1.query_pdb_mapping],
    #                                               [self.scorer1.query_pdb_mapping[x[1]] for x in pairs1b
    #                                                if x[0] in self.scorer1.query_pdb_mapping and
    #                                                x[1] in self.scorer1.query_pdb_mapping]]
    #     self.assertLess(np.sum(expected_scores1b - scores_mapped1b), 1e-5)
    #     self.assertLess(np.sum(expected_dists1b - dists_mapped1b), 1e-5)
    #     scores_mapped1c, dists_mapped1c = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Short')
    #     pairs1c = self.scorer1.find_pairs_by_separation(category='Short')
    #     expected_scores1c = scores1[[x[0] for x in pairs1c if x[0] in self.scorer1.query_pdb_mapping and
    #                                  x[1] in self.scorer1.query_pdb_mapping],
    #                                 [x[1] for x in pairs1c if x[0] in self.scorer1.query_pdb_mapping and
    #                                  x[1] in self.scorer1.query_pdb_mapping]]
    #     expected_dists1c = self.scorer1.distances[[self.scorer1.query_pdb_mapping[x[0]] for x in pairs1c
    #                                                if x[0] in self.scorer1.query_pdb_mapping and
    #                                                x[1] in self.scorer1.query_pdb_mapping],
    #                                               [self.scorer1.query_pdb_mapping[x[1]] for x in pairs1c
    #                                                if x[0] in self.scorer1.query_pdb_mapping and
    #                                                x[1] in self.scorer1.query_pdb_mapping]]
    #     self.assertLess(np.sum(expected_scores1c - scores_mapped1c), 1e-5)
    #     self.assertLess(np.sum(expected_dists1c - dists_mapped1c), 1e-5)
    #     scores_mapped1d, dists_mapped1d = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Medium')
    #     pairs1d = self.scorer1.find_pairs_by_separation(category='Medium')
    #     expected_scores1d = scores1[[x[0] for x in pairs1d if x[0] in self.scorer1.query_pdb_mapping and
    #                                  x[1] in self.scorer1.query_pdb_mapping],
    #                                 [x[1] for x in pairs1d if x[0] in self.scorer1.query_pdb_mapping and
    #                                  x[1] in self.scorer1.query_pdb_mapping]]
    #     expected_dists1d = self.scorer1.distances[[self.scorer1.query_pdb_mapping[x[0]] for x in pairs1d
    #                                                if x[0] in self.scorer1.query_pdb_mapping and
    #                                                x[1] in self.scorer1.query_pdb_mapping],
    #                                               [self.scorer1.query_pdb_mapping[x[1]] for x in pairs1d
    #                                                if x[0] in self.scorer1.query_pdb_mapping and
    #                                                x[1] in self.scorer1.query_pdb_mapping]]
    #     self.assertLess(np.sum(expected_scores1d - scores_mapped1d), 1e-5)
    #     self.assertLess(np.sum(expected_dists1d - dists_mapped1d), 1e-5)
    #     scores_mapped1e, dists_mapped1e = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Long')
    #     pairs1e = self.scorer1.find_pairs_by_separation(category='Long')
    #     expected_scores1e = scores1[[x[0] for x in pairs1e if x[0] in self.scorer1.query_pdb_mapping and
    #                                  x[1] in self.scorer1.query_pdb_mapping],
    #                                 [x[1] for x in pairs1e if x[0] in self.scorer1.query_pdb_mapping and
    #                                  x[1] in self.scorer1.query_pdb_mapping]]
    #     expected_dists1e = self.scorer1.distances[[self.scorer1.query_pdb_mapping[x[0]] for x in pairs1e if
    #                                                x[0] in self.scorer1.query_pdb_mapping and
    #                                                x[1] in self.scorer1.query_pdb_mapping],
    #                                               [self.scorer1.query_pdb_mapping[x[1]] for x in pairs1e if
    #                                                x[0] in self.scorer1.query_pdb_mapping and
    #                                                x[1] in self.scorer1.query_pdb_mapping]]
    #     self.assertLess(np.sum(expected_scores1e - scores_mapped1e), 1e-5)
    #     self.assertLess(np.sum(expected_dists1e - dists_mapped1e), 1e-5)
    #
    # def test_7b__map_predictions_to_pdb(self):
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='CB')
    #     scores2 = np.random.rand(self.seq_len2, self.seq_len2)
    #     scores2[np.tril_indices(self.seq_len2, 1)] = 0
    #     scores2 += scores2.T
    #     scores_mapped2a, dists_mapped2a = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Any')
    #     pairs2a = self.scorer2.find_pairs_by_separation(category='Any')
    #     expected_scores2a = scores2[[x[0] for x in pairs2a if x[0] in self.scorer2.query_pdb_mapping and
    #                                  x[1] in self.scorer2.query_pdb_mapping],
    #                                 [x[1] for x in pairs2a if x[0] in self.scorer2.query_pdb_mapping and
    #                                  x[1] in self.scorer2.query_pdb_mapping]]
    #     expected_dists2a = self.scorer2.distances[[self.scorer2.query_pdb_mapping[x[0]] for x in pairs2a
    #                                                if x[0] in self.scorer2.query_pdb_mapping and
    #                                                x[1] in self.scorer2.query_pdb_mapping],
    #                                               [self.scorer2.query_pdb_mapping[x[1]] for x in pairs2a
    #                                                if x[0] in self.scorer2.query_pdb_mapping and
    #                                                x[1] in self.scorer2.query_pdb_mapping]]
    #     self.assertLess(np.sum(expected_scores2a - scores_mapped2a), 1e-5)
    #     self.assertLess(np.sum(expected_dists2a - dists_mapped2a), 1e-5)
    #     scores_mapped2b, dists_mapped2b = self.scorer2._map_predictions_to_pdb(predictions=scores2,
    #                                                                            category='Neighbors')
    #     pairs2b = self.scorer2.find_pairs_by_separation(category='Neighbors')
    #     expected_scores2b = scores2[[x[0] for x in pairs2b if x[0] in self.scorer2.query_pdb_mapping and
    #                                  x[1] in self.scorer2.query_pdb_mapping],
    #                                 [x[1] for x in pairs2b if x[0] in self.scorer2.query_pdb_mapping and
    #                                  x[1] in self.scorer2.query_pdb_mapping]]
    #     expected_dists2b = self.scorer2.distances[[self.scorer2.query_pdb_mapping[x[0]] for x in pairs2b
    #                                                if x[0] in self.scorer2.query_pdb_mapping and
    #                                                x[1] in self.scorer2.query_pdb_mapping],
    #                                               [self.scorer2.query_pdb_mapping[x[1]] for x in pairs2b
    #                                                if x[0] in self.scorer2.query_pdb_mapping and
    #                                                x[1] in self.scorer2.query_pdb_mapping]]
    #     self.assertLess(np.sum(expected_scores2b - scores_mapped2b), 1e-5)
    #     self.assertLess(np.sum(expected_dists2b - dists_mapped2b), 1e-5)
    #     scores_mapped2c, dists_mapped2c = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Short')
    #     pairs2c = self.scorer2.find_pairs_by_separation(category='Short')
    #     expected_scores2c = scores2[[x[0] for x in pairs2c if x[0] in self.scorer2.query_pdb_mapping and
    #                                  x[1] in self.scorer2.query_pdb_mapping],
    #                                 [x[1] for x in pairs2c if x[0] in self.scorer2.query_pdb_mapping and
    #                                  x[1] in self.scorer2.query_pdb_mapping]]
    #     expected_dists2c = self.scorer2.distances[[self.scorer2.query_pdb_mapping[x[0]] for x in pairs2c
    #                                                if x[0] in self.scorer2.query_pdb_mapping and
    #                                                x[1] in self.scorer2.query_pdb_mapping],
    #                                               [self.scorer2.query_pdb_mapping[x[1]] for x in pairs2c
    #                                                if x[0] in self.scorer2.query_pdb_mapping and
    #                                                x[1] in self.scorer2.query_pdb_mapping]]
    #     self.assertLess(np.sum(expected_scores2c - scores_mapped2c), 1e-5)
    #     self.assertLess(np.sum(expected_dists2c - dists_mapped2c), 1e-5)
    #     scores_mapped2d, dists_mapped2d = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Medium')
    #     pairs2d = self.scorer2.find_pairs_by_separation(category='Medium')
    #     expected_scores2d = scores2[[x[0] for x in pairs2d if x[0] in self.scorer2.query_pdb_mapping and
    #                                  x[1] in self.scorer2.query_pdb_mapping],
    #                                 [x[1] for x in pairs2d if x[0] in self.scorer2.query_pdb_mapping and
    #                                  x[1] in self.scorer2.query_pdb_mapping]]
    #     expected_dists2d = self.scorer2.distances[[self.scorer2.query_pdb_mapping[x[0]] for x in pairs2d
    #                                                if x[0] in self.scorer2.query_pdb_mapping and
    #                                                x[1] in self.scorer2.query_pdb_mapping],
    #                                               [self.scorer2.query_pdb_mapping[x[1]] for x in pairs2d
    #                                                if x[0] in self.scorer2.query_pdb_mapping and
    #                                                x[1] in self.scorer2.query_pdb_mapping]]
    #     self.assertLess(np.sum(expected_scores2d - scores_mapped2d), 1e-5)
    #     self.assertLess(np.sum(expected_dists2d - dists_mapped2d), 1e-5)
    #     scores_mapped2e, dists_mapped2e = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Long')
    #     pairs2e = self.scorer2.find_pairs_by_separation(category='Long')
    #     expected_scores2e = scores2[[x[0] for x in pairs2e if x[0] in self.scorer2.query_pdb_mapping and
    #                                  x[1] in self.scorer2.query_pdb_mapping],
    #                                 [x[1] for x in pairs2e if x[0] in self.scorer2.query_pdb_mapping and
    #                                  x[1] in self.scorer2.query_pdb_mapping]]
    #     expected_dists2e = self.scorer2.distances[[self.scorer2.query_pdb_mapping[x[0]] for x in pairs2e
    #                                                if x[0] in self.scorer2.query_pdb_mapping and
    #                                                x[1] in self.scorer2.query_pdb_mapping],
    #                                               [self.scorer2.query_pdb_mapping[x[1]] for x in pairs2e
    #                                                if x[0] in self.scorer2.query_pdb_mapping and
    #                                                x[1] in self.scorer2.query_pdb_mapping]]
    #     self.assertLess(np.sum(expected_scores2e - scores_mapped2e), 1e-5)
    #     self.assertLess(np.sum(expected_dists2e - dists_mapped2e), 1e-5)

    # def test_8a_score_auc(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='CB')
    #     scores1 = np.random.rand(self.seq_len1, self.seq_len1)
    #     scores1[np.tril_indices(self.seq_len1, 1)] = 0
    #     scores1 += scores1.T
    #     scores_mapped1a, dists_mapped1a = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Any')
    #     fpr_expected1a, tpr_expected1a, _ = roc_curve(dists_mapped1a <= 8.0, scores_mapped1a, pos_label=True)
    #     auroc_expected1a = auc(fpr_expected1a, tpr_expected1a)
    #     tpr1a, fpr1a, auroc1a = self.scorer1.score_auc(scores1, category='Any')
    #     self.assertEqual(np.sum(fpr_expected1a - fpr1a), 0)
    #     self.assertEqual(np.sum(tpr_expected1a - tpr1a), 0)
    #     self.assertEqual(auroc_expected1a, auroc1a)
    #     scores_mapped1b, dists_mapped1b = self.scorer1._map_predictions_to_pdb(predictions=scores1,
    #                                                                            category='Neighbors')
    #     fpr_expected1b, tpr_expected1b, _ = roc_curve(dists_mapped1b <= 8.0, scores_mapped1b, pos_label=True)
    #     auroc_expected1b = auc(fpr_expected1b, tpr_expected1b)
    #     tpr1b, fpr1b, auroc1b = self.scorer1.score_auc(scores1, category='Neighbors')
    #     self.assertEqual(np.sum(fpr_expected1b - fpr1b), 0)
    #     self.assertEqual(np.sum(tpr_expected1b - tpr1b), 0)
    #     self.assertEqual(auroc_expected1b, auroc1b)
    #     scores_mapped1c, dists_mapped1c = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Short')
    #     fpr_expected1c, tpr_expected1c, _ = roc_curve(dists_mapped1c <= 8.0, scores_mapped1c, pos_label=True)
    #     auroc_expected1c = auc(fpr_expected1c, tpr_expected1c)
    #     tpr1c, fpr1c, auroc1c = self.scorer1.score_auc(scores1, category='Short')
    #     self.assertEqual(np.sum(fpr_expected1c - fpr1c), 0)
    #     self.assertEqual(np.sum(tpr_expected1c - tpr1c), 0)
    #     self.assertEqual(auroc_expected1c, auroc1c)
    #     scores_mapped1d, dists_mapped1d = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Medium')
    #     fpr_expected1d, tpr_expected1d, _ = roc_curve(dists_mapped1d <= 8.0, scores_mapped1d, pos_label=True)
    #     auroc_expected1d = auc(fpr_expected1d, tpr_expected1d)
    #     tpr1d, fpr1d, auroc1d = self.scorer1.score_auc(scores1, category='Medium')
    #     self.assertEqual(np.sum(fpr_expected1d - fpr1d), 0)
    #     self.assertEqual(np.sum(tpr_expected1d - tpr1d), 0)
    #     self.assertEqual(auroc_expected1d, auroc1d)
    #     scores_mapped1e, dists_mapped1e = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Long')
    #     fpr_expected1e, tpr_expected1e, _ = roc_curve(dists_mapped1e <= 8.0, scores_mapped1e, pos_label=True)
    #     auroc_expected1e = auc(fpr_expected1e, tpr_expected1e)
    #     tpr1e, fpr1e, auroc1e = self.scorer1.score_auc(scores1, category='Long')
    #     self.assertEqual(np.sum(fpr_expected1e - fpr1e), 0)
    #     self.assertEqual(np.sum(tpr_expected1e - tpr1e), 0)
    #     self.assertEqual(auroc_expected1e, auroc1e)
    #
    # def test_8b_score_auc(self):
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='CB')
    #     scores2 = np.random.rand(self.seq_len2, self.seq_len2)
    #     scores2[np.tril_indices(self.seq_len2, 1)] = 0
    #     scores2 += scores2.T
    #     scores_mapped2a, dists_mapped2a = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Any')
    #     fpr_expected2a, tpr_expected2a, _ = roc_curve(dists_mapped2a <= 8.0, scores_mapped2a, pos_label=True)
    #     auroc_expected2a = auc(fpr_expected2a, tpr_expected2a)
    #     tpr2a, fpr2a, auroc2a = self.scorer2.score_auc(scores2, category='Any')
    #     self.assertEqual(np.sum(fpr_expected2a - fpr2a), 0)
    #     self.assertEqual(np.sum(tpr_expected2a - tpr2a), 0)
    #     self.assertEqual(auroc_expected2a, auroc2a)
    #     scores_mapped2b, dists_mapped2b = self.scorer2._map_predictions_to_pdb(predictions=scores2,
    #                                                                            category='Neighbors')
    #     fpr_expected2b, tpr_expected2b, _ = roc_curve(dists_mapped2b <= 8.0, scores_mapped2b, pos_label=True)
    #     auroc_expected2b = auc(fpr_expected2b, tpr_expected2b)
    #     tpr2b, fpr2b, auroc2b = self.scorer2.score_auc(scores2, category='Neighbors')
    #     self.assertEqual(np.sum(fpr_expected2b - fpr2b), 0)
    #     self.assertEqual(np.sum(tpr_expected2b - tpr2b), 0)
    #     self.assertEqual(auroc_expected2b, auroc2b)
    #     scores_mapped2c, dists_mapped2c = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Short')
    #     fpr_expected2c, tpr_expected2c, _ = roc_curve(dists_mapped2c <= 8.0, scores_mapped2c, pos_label=True)
    #     auroc_expected2c = auc(fpr_expected2c, tpr_expected2c)
    #     tpr2c, fpr2c, auroc2c = self.scorer2.score_auc(scores2, category='Short')
    #     self.assertEqual(np.sum(fpr_expected2c - fpr2c), 0)
    #     self.assertEqual(np.sum(tpr_expected2c - tpr2c), 0)
    #     self.assertEqual(auroc_expected2c, auroc2c)
    #     scores_mapped2d, dists_mapped2d = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Medium')
    #     fpr_expected2d, tpr_expected2d, _ = roc_curve(dists_mapped2d <= 8.0, scores_mapped2d, pos_label=True)
    #     auroc_expected2d = auc(fpr_expected2d, tpr_expected2d)
    #     tpr2d, fpr2d, auroc2d = self.scorer2.score_auc(scores2, category='Medium')
    #     self.assertEqual(np.sum(fpr_expected2d - fpr2d), 0)
    #     self.assertEqual(np.sum(tpr_expected2d - tpr2d), 0)
    #     self.assertEqual(auroc_expected2d, auroc2d)
    #     scores_mapped2e, dists_mapped2e = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Long')
    #     fpr_expected2e, tpr_expected2e, _ = roc_curve(dists_mapped2e <= 8.0, scores_mapped2e, pos_label=True)
    #     auroc_expected2e = auc(fpr_expected2e, tpr_expected2e)
    #     tpr2e, fpr2e, auroc2e = self.scorer2.score_auc(scores2, category='Long')
    #     self.assertEqual(np.sum(fpr_expected2e - fpr2e), 0)
    #     self.assertEqual(np.sum(tpr_expected2e - tpr2e), 0)
    #     self.assertEqual(auroc_expected2e, auroc2e)

    # def test_9a_plot_auc(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='CB')
    #     scores1 = np.random.rand(self.seq_len1, self.seq_len1)
    #     scores1[np.tril_indices(self.seq_len1, 1)] = 0
    #     scores1 += scores1.T
    #     auroc1a = self.scorer1.score_auc(scores1, category='Any')
    #     self.scorer1.plot_auc(auc_data=auroc1a, title='{} AUROC for All Pairs'.format(self.small_structure_id),
    #                           file_name='{}_Any_AUROC'.format(self.small_structure_id), output_dir=self.testing_dir)
    #     expected_path1 = os.path.abspath(os.path.join(self.testing_dir,
    #                                                   '{}_Any_AUROC.png'.format(self.small_structure_id)))
    #     self.assertTrue(os.path.isfile(expected_path1))
    #     os.remove(expected_path1)
    #
    # def test_9b_plot_auc(self):
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='CB')
    #     scores2 = np.random.rand(self.seq_len2, self.seq_len2)
    #     scores2[np.tril_indices(self.seq_len2, 1)] = 0
    #     scores2 += scores2.T
    #     auroc2a = self.scorer2.score_auc(scores2, category='Any')
    #     self.scorer2.plot_auc(auc_data=auroc2a, title='{} AUROC for All Pairs'.format(self.large_structure_id),
    #                           file_name='{}_Any_AUROC'.format(self.large_structure_id), output_dir=self.testing_dir)
    #     expected_path2 = os.path.abspath(os.path.join(self.testing_dir,
    #                                                   '{}_Any_AUROC.png'.format(self.large_structure_id)))
    #     self.assertTrue(os.path.isfile(expected_path2))
    #     os.remove(expected_path2)

    # def test_10a_score_precision_recall(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='CB')
    #     scores1 = np.random.rand(self.seq_len1, self.seq_len1)
    #     scores1[np.tril_indices(self.seq_len1, 1)] = 0
    #     scores1 += scores1.T
    #     scores_mapped1a, dists_mapped1a = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Any')
    #     precision_expected1a, recall_expected1a, _ = precision_recall_curve(dists_mapped1a <= 8.0, scores_mapped1a,
    #                                                                         pos_label=True)
    #     recall_expected1a, precision_expected1a = zip(*sorted(zip(recall_expected1a, precision_expected1a)))
    #     recall_expected1a, precision_expected1a = np.array(recall_expected1a), np.array(precision_expected1a)
    #     auprc_expected1a = auc(recall_expected1a, precision_expected1a)
    #     precision1a, recall1a, auprc1a = self.scorer1.score_precision_recall(scores1, category='Any')
    #     self.assertEqual(np.sum(precision_expected1a - precision1a), 0)
    #     self.assertEqual(np.sum(recall_expected1a - recall1a), 0)
    #     self.assertEqual(auprc_expected1a, auprc1a)
    #     scores_mapped1b, dists_mapped1b = self.scorer1._map_predictions_to_pdb(predictions=scores1,
    #                                                                            category='Neighbors')
    #     precision_expected1b, recall_expected1b, _ = precision_recall_curve(dists_mapped1b <= 8.0, scores_mapped1b,
    #                                                                         pos_label=True)
    #     recall_expected1b, precision_expected1b = zip(*sorted(zip(recall_expected1b, precision_expected1b)))
    #     recall_expected1b, precision_expected1b = np.array(recall_expected1b), np.array(precision_expected1b)
    #     auprc_expected1b = auc(recall_expected1b, precision_expected1b)
    #     precision1b, recall1b, auprc1b = self.scorer1.score_precision_recall(scores1, category='Neighbors')
    #
    #     self.assertEqual(np.sum(precision_expected1b - precision1b), 0)
    #     self.assertEqual(np.sum(recall_expected1b - recall1b), 0)
    #     self.assertEqual(auprc_expected1b, auprc1b)
    #     scores_mapped1c, dists_mapped1c = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Short')
    #     precision_expected1c, recall_expected1c, _ = precision_recall_curve(dists_mapped1c <= 8.0, scores_mapped1c,
    #                                                                         pos_label=True)
    #     recall_expected1c, precision_expected1c = zip(*sorted(zip(recall_expected1c, precision_expected1c)))
    #     recall_expected1c, precision_expected1c = np.array(recall_expected1c), np.array(precision_expected1c)
    #     auprc_expected1c = auc(recall_expected1c, precision_expected1c)
    #     precision1c, recall1c, auprc1c = self.scorer1.score_precision_recall(scores1, category='Short')
    #     self.assertEqual(np.sum(precision_expected1c - precision1c), 0)
    #     self.assertEqual(np.sum(recall_expected1c - recall1c), 0)
    #     self.assertEqual(auprc_expected1c, auprc1c)
    #     scores_mapped1d, dists_mapped1d = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Medium')
    #     precision_expected1d, recall_expected1d, _ = precision_recall_curve(dists_mapped1d <= 8.0, scores_mapped1d,
    #                                                                         pos_label=True)
    #     recall_expected1d, precision_expected1d = zip(*sorted(zip(recall_expected1d, precision_expected1d)))
    #     recall_expected1d, precision_expected1d = np.array(recall_expected1d), np.array(precision_expected1d)
    #     auprc_expected1d = auc(recall_expected1d, precision_expected1d)
    #     precision1d, recall1d, auprc1d = self.scorer1.score_precision_recall(scores1, category='Medium')
    #     self.assertEqual(np.sum(precision_expected1d - precision1d), 0)
    #     self.assertEqual(np.sum(recall_expected1d - recall1d), 0)
    #     self.assertEqual(auprc_expected1d, auprc1d)
    #     scores_mapped1e, dists_mapped1e = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Long')
    #     precision_expected1e, recall_expected1e, _ = precision_recall_curve(dists_mapped1e <= 8.0, scores_mapped1e,
    #                                                                         pos_label=True)
    #     recall_expected1e, precision_expected1e = zip(*sorted(zip(recall_expected1e, precision_expected1e)))
    #     recall_expected1e, precision_expected1e = np.array(recall_expected1e), np.array(precision_expected1e)
    #     auprc_expected1e = auc(recall_expected1e, precision_expected1e)
    #     precision1e, recall1e, auprc1e = self.scorer1.score_precision_recall(scores1, category='Long')
    #     self.assertEqual(np.sum(precision_expected1e - precision1e), 0)
    #     self.assertEqual(np.sum(recall_expected1e - recall1e), 0)
    #     self.assertEqual(auprc_expected1e, auprc1e)
    #
    # def test_10b_score_precision_recall(self):
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='CB')
    #     scores2 = np.random.rand(self.seq_len2, self.seq_len2)
    #     scores2[np.tril_indices(self.seq_len2, 1)] = 0
    #     scores2 += scores2.T
    #     scores_mapped2a, dists_mapped2a = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Any')
    #     precision_expected2a, recall_expected2a, _ = precision_recall_curve(dists_mapped2a <= 8.0, scores_mapped2a,
    #                                                                         pos_label=True)
    #     recall_expected2a, precision_expected2a = zip(*sorted(zip(recall_expected2a, precision_expected2a)))
    #     recall_expected2a, precision_expected2a = np.array(recall_expected2a), np.array(precision_expected2a)
    #     auprc_expected2a = auc(recall_expected2a, precision_expected2a)
    #     precision2a, recall2a, auprc2a = self.scorer2.score_precision_recall(scores2, category='Any')
    #     self.assertEqual(np.sum(precision_expected2a - precision2a), 0)
    #     self.assertEqual(np.sum(recall_expected2a - recall2a), 0)
    #     self.assertEqual(auprc_expected2a, auprc2a)
    #     scores_mapped2b, dists_mapped2b = self.scorer2._map_predictions_to_pdb(predictions=scores2,
    #                                                                            category='Neighbors')
    #     precision_expected2b, recall_expected2b, _ = precision_recall_curve(dists_mapped2b <= 8.0, scores_mapped2b,
    #                                                                         pos_label=True)
    #     recall_expected2b, precision_expected2b = zip(*sorted(zip(recall_expected2b, precision_expected2b)))
    #     recall_expected2b, precision_expected2b = np.array(recall_expected2b), np.array(precision_expected2b)
    #     auprc_expected2b = auc(recall_expected2b, precision_expected2b)
    #     precision2b, recall2b, auprc2b = self.scorer2.score_precision_recall(scores2, category='Neighbors')
    #     self.assertEqual(np.sum(precision_expected2b - precision2b), 0)
    #     self.assertEqual(np.sum(recall_expected2b - recall2b), 0)
    #     self.assertEqual(auprc_expected2b, auprc2b)
    #     scores_mapped2c, dists_mapped2c = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Short')
    #     precision_expected2c, recall_expected2c, _ = precision_recall_curve(dists_mapped2c <= 8.0, scores_mapped2c,
    #                                                                         pos_label=True)
    #     recall_expected2c, precision_expected2c = zip(*sorted(zip(recall_expected2c, precision_expected2c)))
    #     recall_expected2c, precision_expected2c = np.array(recall_expected2c), np.array(precision_expected2c)
    #     auprc_expected2c = auc(recall_expected2c, precision_expected2c)
    #     precision2c, recall2c, auprc2c = self.scorer2.score_precision_recall(scores2, category='Short')
    #     self.assertEqual(np.sum(precision_expected2c - precision2c), 0)
    #     self.assertEqual(np.sum(recall_expected2c - recall2c), 0)
    #     self.assertEqual(auprc_expected2c, auprc2c)
    #     scores_mapped2d, dists_mapped2d = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Medium')
    #     precision_expected2d, recall_expected2d, _ = precision_recall_curve(dists_mapped2d <= 8.0, scores_mapped2d,
    #                                                                         pos_label=True)
    #     recall_expected2d, precision_expected2d = zip(*sorted(zip(recall_expected2d, precision_expected2d)))
    #     recall_expected2d, precision_expected2d = np.array(recall_expected2d), np.array(precision_expected2d)
    #     auprc_expected2d = auc(recall_expected2d, precision_expected2d)
    #     precision2d, recall2d, auprc2d = self.scorer2.score_precision_recall(scores2, category='Medium')
    #     self.assertEqual(np.sum(precision_expected2d - precision2d), 0)
    #     self.assertEqual(np.sum(recall_expected2d - recall2d), 0)
    #     self.assertEqual(auprc_expected2d, auprc2d)
    #     scores_mapped2e, dists_mapped2e = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Long')
    #     precision_expected2e, recall_expected2e, _ = precision_recall_curve(dists_mapped2e <= 8.0, scores_mapped2e,
    #                                                                         pos_label=True)
    #     recall_expected2e, precision_expected2e = zip(*sorted(zip(recall_expected2e, precision_expected2e)))
    #     recall_expected2e, precision_expected2e = np.array(recall_expected2e), np.array(precision_expected2e)
    #     auprc_expected2e = auc(recall_expected2e, precision_expected2e)
    #     precision2e, recall2e, auprc2e = self.scorer2.score_precision_recall(scores2, category='Long')
    #     self.assertEqual(np.sum(precision_expected2e - precision2e), 0)
    #     self.assertEqual(np.sum(recall_expected2e - recall2e), 0)
    #     self.assertEqual(auprc_expected2e, auprc2e)

    # def test_11a_plot_auprc(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='CB')
    #     scores1 = np.random.rand(self.seq_len1, self.seq_len1)
    #     scores1[np.tril_indices(self.seq_len1, 1)] = 0
    #     scores1 += scores1.T
    #     auprc1a = self.scorer1.score_precision_recall(scores1, category='Any')
    #     self.scorer1.plot_auprc(auprc_data=auprc1a, title='{} AUPRC for All Pairs'.format(self.small_structure_id),
    #                             file_name='{}_Any_AUPRC'.format(self.small_structure_id), output_dir=self.testing_dir)
    #     expected_path1 = os.path.abspath(os.path.join(self.testing_dir,
    #                                                   '{}_Any_AUPRC.png'.format(self.small_structure_id)))
    #     self.assertTrue(os.path.isfile(expected_path1))
    #     os.remove(expected_path1)
    #
    # def test_11b_plot_auprc(self):
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='CB')
    #     scores2 = np.random.rand(self.seq_len2, self.seq_len2)
    #     scores2[np.tril_indices(self.seq_len2, 1)] = 0
    #     scores2 += scores2.T
    #     auprc2a = self.scorer2.score_precision_recall(scores2, category='Any')
    #     self.scorer2.plot_auprc(auprc_data=auprc2a, title='{} AUPRC for All Pairs'.format(self.large_structure_id),
    #                             file_name='{}_Any_AUPRC'.format(self.large_structure_id), output_dir=self.testing_dir)
    #     expected_path2 = os.path.abspath(os.path.join(self.testing_dir,
    #                                                   '{}_Any_AUPRC.png'.format(self.large_structure_id)))
    #     self.assertTrue(os.path.isfile(expected_path2))
    #     os.remove(expected_path2)

#     # def test_12a_score_tpr_fdr(self):
#     #     self.scorer1.fit()
#     #     self.scorer1.measure_distance(method='CB')
#     #     scores1 = np.random.rand(self.seq_len1, self.seq_len1)
#     #     scores1[np.tril_indices(self.seq_len1, 1)] = 0
#     #     scores1 += scores1.T
#     #     scores_mapped1a, dists_mapped1a = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Any')
#     #     _, tpr_expected1a, _ = roc_curve(dists_mapped1a <= 8.0, scores_mapped1a, pos_label=True)
#     #     # autprfdrc_expected1a = auc(, )
#     #     tpr1a, fdr1a, autprfdrc1a = self.scorer1.score_tpr_fdr(scores1, category='Any')
#     #     self.assertEqual(np.sum(tpr_expected1a - tpr1a), 0)
#     #     # self.assertEqual(np.sum(fdr_expected1a - fdr1a), 0)
#     #     # self.assertEqual(autprfdrc_expected1a, autprfdrc1a)
#     #     scores_mapped1b, dists_mapped1b = self.scorer1._map_predictions_to_pdb(predictions=scores1,
#     #                                                                            category='Neighbors')
#     #     _, tpr_expected1b, _ = roc_curve(dists_mapped1b <= 8.0, scores_mapped1b, pos_label=True)
#     #     # autprfdrc_expected1b = auc(, )
#     #     tpr1b, fdr1b, autprfdrc1b = self.scorer1.score_tpr_fdr(scores1, category='Neighbors')
#     #     self.assertEqual(np.sum(tpr_expected1b - tpr1b), 0)
#     #     # self.assertEqual(np.sum(fdr_expected1b - fdr1b), 0)
#     #     # self.assertEqual(autprfdrc_expected1b, autprfdrc1b)
#     #     scores_mapped1c, dists_mapped1c = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Short')
#     #     _, tpr_expected1c, _ = roc_curve(dists_mapped1c <= 8.0, scores_mapped1c, pos_label=True)
#     #     # autprfdrc_expected1c = auc(, )
#     #     tpr1c, fdr1c, autprfdrc1c = self.scorer1.score_tpr_fdr(scores1, category='Short')
#     #     self.assertEqual(np.sum(tpr_expected1c - tpr1c), 0)
#     #     # self.assertEqual(np.sum(fdr_expected1c - fdr1c), 0)
#     #     # self.assertEqual(autprfdrc_expected1c, autprfdrc1c)
#     #     scores_mapped1d, dists_mapped1d = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Medium')
#     #     _, tpr_expected1d, _ = roc_curve(dists_mapped1d <= 8.0, scores_mapped1d, pos_label=True)
#     #     # autprfdrc_expected1d = auc(, )
#     #     tpr1d, fdr1d, autprfdrc1d = self.scorer1.score_tpr_fdr(scores1, category='Medium')
#     #     self.assertEqual(np.sum(tpr_expected1d - tpr1d), 0)
#     #     # self.assertEqual(np.sum(fdr_expected1d - fdr1d), 0)
#     #     # self.assertEqual(autprfdrc_expected1d, autprfdrc1d)
#     #     scores_mapped1e, dists_mapped1e = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Long')
#     #     _, tpr_expected1e, _ = roc_curve(dists_mapped1e <= 8.0, scores_mapped1e, pos_label=True)
#     #     # autprfdrc_expected1e = auc(, )
#     #     tpr1e, fdr1e, autprfdrc1e = self.scorer1.score_tpr_fdr(scores1, category='Long')
#     #     self.assertEqual(np.sum(tpr_expected1e - tpr1e), 0)
#     #     # self.assertEqual(np.sum(fdr_expected1e - fdr1e), 0)
#     #     # self.assertEqual(autprfdrc_expected1e, autprfdrc1e)
#     #
#     # def test_12b_score_tpr_fdr(self):
#     #     self.scorer2.fit()
#     #     self.scorer2.measure_distance(method='CB')
#     #     scores2 = np.random.rand(self.seq_len2, self.seq_len2)
#     #     scores2[np.tril_indices(self.seq_len2, 1)] = 0
#     #     scores2 += scores2.T
#     #     scores_mapped2a, dists_mapped2a = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Any')
#     #     _, tpr_expected2a, _ = roc_curve(dists_mapped2a <= 8.0, scores_mapped2a, pos_label=True)
#     #     # autprfdrc_expected2a = auc(, )
#     #     tpr2a, fdr2a, autprfdrc2a = self.scorer2.score_tpr_fdr(scores2, category='Any')
#     #     self.assertEqual(np.sum(tpr_expected2a - tpr2a), 0)
#     #     # self.assertEqual(np.sum(fdr_expected2a - fdr2a), 0)
#     #     # self.assertEqual(autprfdrc_expected2a, autprfdrc2a)
#     #     scores_mapped2b, dists_mapped2b = self.scorer2._map_predictions_to_pdb(predictions=scores2,
#     #                                                                            category='Neighbors')
#     #     _, tpr_expected2b, _ = roc_curve(dists_mapped2b <= 8.0, scores_mapped2b, pos_label=True)
#     #     # autprfdrc_expected2b = auc(, )
#     #     tpr2b, fdr2b, autprfdrc2b = self.scorer2.score_tpr_fdr(scores2, category='Neighbors')
#     #     self.assertEqual(np.sum(tpr_expected2b - tpr2b), 0)
#     #     # self.assertEqual(np.sum(fdr_expected2b - fdr2b), 0)
#     #     # self.assertEqual(autprfdrc_expected2b, autprfdrc2b)
#     #     scores_mapped2c, dists_mapped2c = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Short')
#     #     _, tpr_expected2c, _ = roc_curve(dists_mapped2c <= 8.0, scores_mapped2c, pos_label=True)
#     #     # autprfdrc_expected2c = auc(, )
#     #     tpr2c, fdr2c, autprfdrc2c = self.scorer2.score_tpr_fdr(scores2, category='Short')
#     #     self.assertEqual(np.sum(tpr_expected2c - tpr2c), 0)
#     #     # self.assertEqual(np.sum(fdr_expected2c - fdr2c), 0)
#     #     # self.assertEqual(autprfdrc_expected2c, autprfdrc2c)
#     #     scores_mapped2d, dists_mapped2d = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Medium')
#     #     _, tpr_expected2d, _ = roc_curve(dists_mapped2d <= 8.0, scores_mapped2d, pos_label=True)
#     #     # autprfdrc_expected2d = auc(, )
#     #     tpr2d, fdr2d, autprfdrc2d = self.scorer2.score_tpr_fdr(scores2, category='Medium')
#     #     self.assertEqual(np.sum(tpr_expected2d - tpr2d), 0)
#     #     # self.assertEqual(np.sum(fdr_expected2d - fdr2d), 0)
#     #     # self.assertEqual(autprfdrc_expected2d, autprfdrc2d)
#     #     scores_mapped2e, dists_mapped2e = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Long')
#     #     _, tpr_expected2e, _ = roc_curve(dists_mapped2e <= 8.0, scores_mapped2e, pos_label=True)
#     #     # autprfdrc_expected2e = auc(, )
#     #     tpr2e, fdr2e, autprfdrc2e = self.scorer2.score_tpr_fdr(scores2, category='Long')
#     #     self.assertEqual(np.sum(tpr_expected2e - tpr2e), 0)
#     #     # self.assertEqual(np.sum(fdr_expected2e - fdr2e), 0)
#     #     # self.assertEqual(autprfdrc_expected2e, autprfdrc2e)

    # def test_13a_plot_autprfdrc(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='CB')
    #     scores1 = np.random.rand(self.seq_len1, self.seq_len1)
    #     scores1[np.tril_indices(self.seq_len1, 1)] = 0
    #     scores1 += scores1.T
    #     autprfdrc1a = self.scorer1.score_tpr_fdr(scores1, category='Any')
    #     self.scorer1.plot_autprfdrc(autprfdrc_data=autprfdrc1a,
    #                                 title='{} AUTPRFDRC for All Pairs'.format(self.small_structure_id),
    #                                 file_name='{}_Any_AUTPRFDRC'.format(self.small_structure_id),
    #                                 output_dir=self.testing_dir)
    #     expected_path1 = os.path.abspath(os.path.join(self.testing_dir,
    #                                                   '{}_Any_AUTPRFDRC.png'.format(self.small_structure_id)))
    #     self.assertTrue(os.path.isfile(expected_path1))
    #     os.remove(expected_path1)
    #
    # def test_13b_plot_autprfdrc(self):
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='CB')
    #     scores2 = np.random.rand(self.seq_len2, self.seq_len2)
    #     scores2[np.tril_indices(self.seq_len2, 1)] = 0
    #     scores2 += scores2.T
    #     autprfdrc2a = self.scorer2.score_tpr_fdr(scores2, category='Any')
    #     self.scorer2.plot_autprfdrc(autprfdrc_data=autprfdrc2a,
    #                                 title='{} AUTPRFDRC for All Pairs'.format(self.large_structure_id),
    #                                 file_name='{}_Any_AUTPRFDRC'.format(self.large_structure_id),
    #                                 output_dir=self.testing_dir)
    #     expected_path2 = os.path.abspath(os.path.join(self.testing_dir,
    #                                                   '{}_Any_AUTPRFDRC.png'.format(self.large_structure_id)))
    #     self.assertTrue(os.path.isfile(expected_path2))
    #     os.remove(expected_path2)

    # def test_14a_score_precision(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='CB')
    #     scores1 = np.random.rand(self.seq_len1, self.seq_len1)
    #     scores1[np.tril_indices(self.seq_len1, 1)] = 0
    #     scores1 += scores1.T
    #     scores_mapped1a, dists_mapped1a = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Any')
    #     expected_precision1a_all = self.check_precision(mapped_scores=scores_mapped1a, mapped_dists=dists_mapped1a)
    #     precision1a_all = self.scorer1.score_precision(predictions=scores1, category='Any')
    #     self.assertEqual(expected_precision1a_all, precision1a_all)
    #     expected_precision1a_k10 = self.check_precision(mapped_scores=scores_mapped1a, mapped_dists=dists_mapped1a,
    #                                                     count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     precision1a_k10 = self.scorer1.score_precision(predictions=scores1, k=10, category='Any')
    #     self.assertEqual(expected_precision1a_k10, precision1a_k10)
    #     expected_precision1a_n10 = self.check_precision(mapped_scores=scores_mapped1a, mapped_dists=dists_mapped1a,
    #                                                     count=10.0)
    #     precision1a_n10 = self.scorer1.score_precision(predictions=scores1, n=10, category='Any')
    #     self.assertEqual(expected_precision1a_n10, precision1a_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_precision(predictions=scores1, k=10, n=10, category='Any')
    #     scores_mapped1b, dists_mapped1b = self.scorer1._map_predictions_to_pdb(predictions=scores1,
    #                                                                            category='Neighbors')
    #     expected_precision1b_all = self.check_precision(mapped_scores=scores_mapped1b, mapped_dists=dists_mapped1b)
    #     precision1b_all = self.scorer1.score_precision(predictions=scores1, category='Neighbors')
    #     self.assertEqual(expected_precision1b_all, precision1b_all)
    #     expected_precision1b_k10 = self.check_precision(mapped_scores=scores_mapped1b, mapped_dists=dists_mapped1b,
    #                                                     count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     precision1b_k10 = self.scorer1.score_precision(predictions=scores1, k=10, category='Neighbors')
    #     self.assertEqual(expected_precision1b_k10, precision1b_k10)
    #     expected_precision1b_n10 = self.check_precision(mapped_scores=scores_mapped1b, mapped_dists=dists_mapped1b,
    #                                                     count=10.0)
    #     precision1b_n10 = self.scorer1.score_precision(predictions=scores1, n=10, category='Neighbors')
    #     self.assertEqual(expected_precision1b_n10, precision1b_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_precision(predictions=scores1, k=10, n=10, category='Neighbors')
    #     scores_mapped1c, dists_mapped1c = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Short')
    #     expected_precision1c_all = self.check_precision(mapped_scores=scores_mapped1c, mapped_dists=dists_mapped1c)
    #     precision1c_all = self.scorer1.score_precision(predictions=scores1, category='Short')
    #     self.assertEqual(expected_precision1c_all, precision1c_all)
    #     precision1c_k10 = self.scorer1.score_precision(predictions=scores1, k=10, category='Short')
    #     expected_precision1c_k10 = self.check_precision(mapped_scores=scores_mapped1c, mapped_dists=dists_mapped1c,
    #                                                     count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     self.assertEqual(expected_precision1c_k10, precision1c_k10)
    #     expected_precision1c_n10 = self.check_precision(mapped_scores=scores_mapped1c, mapped_dists=dists_mapped1c,
    #                                                     count=10.0)
    #     precision1c_n10 = self.scorer1.score_precision(predictions=scores1, n=10, category='Short')
    #     self.assertEqual(expected_precision1c_n10, precision1c_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_precision(predictions=scores1, k=10, n=10, category='Short')
    #     scores_mapped1d, dists_mapped1d = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Medium')
    #     expected_precision1d_all = self.check_precision(mapped_scores=scores_mapped1d, mapped_dists=dists_mapped1d)
    #     precision1d_all = self.scorer1.score_precision(predictions=scores1, category='Medium')
    #     self.assertEqual(expected_precision1d_all, precision1d_all)
    #     expected_precision1d_k10 = self.check_precision(mapped_scores=scores_mapped1d, mapped_dists=dists_mapped1d,
    #                                                     count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     precision1d_k10 = self.scorer1.score_precision(predictions=scores1, k=10, category='Medium')
    #     self.assertEqual(expected_precision1d_k10, precision1d_k10)
    #     expected_precision1d_n10 = self.check_precision(mapped_scores=scores_mapped1d, mapped_dists=dists_mapped1d,
    #                                                     count=10.0)
    #     precision1d_n10 = self.scorer1.score_precision(predictions=scores1, n=10, category='Medium')
    #     self.assertEqual(expected_precision1d_n10, precision1d_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_precision(predictions=scores1, k=10, n=10, category='Medium')
    #     scores_mapped1e, dists_mapped1e = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Long')
    #     expected_precision1e_all = self.check_precision(mapped_scores=scores_mapped1e, mapped_dists=dists_mapped1e)
    #     precision1e_all = self.scorer1.score_precision(predictions=scores1, category='Long')
    #     self.assertEqual(expected_precision1e_all, precision1e_all)
    #     expected_precision1e_k10 = self.check_precision(mapped_scores=scores_mapped1e, mapped_dists=dists_mapped1e,
    #                                                     count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     precision1e_k10 = self.scorer1.score_precision(predictions=scores1, k=10, category='Long')
    #     self.assertEqual(expected_precision1e_k10, precision1e_k10)
    #     expected_precision1e_n10 = self.check_precision(mapped_scores=scores_mapped1e, mapped_dists=dists_mapped1e,
    #                                                     count=10.0)
    #     precision1e_n10 = self.scorer1.score_precision(predictions=scores1, n=10, category='Long')
    #     self.assertEqual(expected_precision1e_n10, precision1e_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_precision(predictions=scores1, k=10, n=10, category='Long')
    #
    # def test_14b_score_precision(self):
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='CB')
    #     scores2 = np.random.rand(self.seq_len2, self.seq_len2)
    #     scores2[np.tril_indices(self.seq_len2, 1)] = 0
    #     scores2 += scores2.T
    #     scores_mapped2a, dists_mapped2a = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Any')
    #     expected_precision2a_all = self.check_precision(mapped_scores=scores_mapped2a, mapped_dists=dists_mapped2a)
    #     precision2a_all = self.scorer2.score_precision(predictions=scores2, category='Any')
    #     self.assertEqual(expected_precision2a_all, precision2a_all)
    #     expected_precision2a_k10 = self.check_precision(mapped_scores=scores_mapped2a, mapped_dists=dists_mapped2a,
    #                                                     count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     precision2a_k10 = self.scorer2.score_precision(predictions=scores2, k=10, category='Any')
    #     self.assertEqual(expected_precision2a_k10, precision2a_k10)
    #     expected_precision2a_n10 = self.check_precision(mapped_scores=scores_mapped2a, mapped_dists=dists_mapped2a,
    #                                                     count=10.0)
    #     precision2a_n10 = self.scorer2.score_precision(predictions=scores2, n=10, category='Any')
    #     self.assertEqual(expected_precision2a_n10, precision2a_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_precision(predictions=scores2, k=10, n=10, category='Any')
    #     scores_mapped2b, dists_mapped2b = self.scorer2._map_predictions_to_pdb(predictions=scores2,
    #                                                                            category='Neighbors')
    #     expected_precision2b_all = self.check_precision(mapped_scores=scores_mapped2b, mapped_dists=dists_mapped2b)
    #     precision2b_all = self.scorer2.score_precision(predictions=scores2, category='Neighbors')
    #     self.assertEqual(expected_precision2b_all, precision2b_all)
    #     expected_precision2b_k10 = self.check_precision(mapped_scores=scores_mapped2b, mapped_dists=dists_mapped2b,
    #                                                     count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     precision2b_k10 = self.scorer2.score_precision(predictions=scores2, k=10, category='Neighbors')
    #     self.assertEqual(expected_precision2b_k10, precision2b_k10)
    #     expected_precision2b_n10 = self.check_precision(mapped_scores=scores_mapped2b, mapped_dists=dists_mapped2b,
    #                                                     count=10.0)
    #     precision2b_n10 = self.scorer2.score_precision(predictions=scores2, n=10, category='Neighbors')
    #     self.assertEqual(expected_precision2b_n10, precision2b_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_precision(predictions=scores2, k=10, n=10, category='Neighbors')
    #     scores_mapped2c, dists_mapped2c = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Short')
    #     expected_precision2c_all = self.check_precision(mapped_scores=scores_mapped2c, mapped_dists=dists_mapped2c)
    #     precision2c_all = self.scorer2.score_precision(predictions=scores2, category='Short')
    #     self.assertEqual(expected_precision2c_all, precision2c_all)
    #     precision2c_k10 = self.scorer2.score_precision(predictions=scores2, k=10, category='Short')
    #     expected_precision2c_k10 = self.check_precision(mapped_scores=scores_mapped2c, mapped_dists=dists_mapped2c,
    #                                                     count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     self.assertEqual(expected_precision2c_k10, precision2c_k10)
    #     expected_precision2c_n10 = self.check_precision(mapped_scores=scores_mapped2c, mapped_dists=dists_mapped2c,
    #                                                     count=10.0)
    #     precision2c_n10 = self.scorer2.score_precision(predictions=scores2, n=10, category='Short')
    #     self.assertEqual(expected_precision2c_n10, precision2c_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_precision(predictions=scores2, k=10, n=10, category='Short')
    #     scores_mapped2d, dists_mapped2d = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Medium')
    #     expected_precision2d_all = self.check_precision(mapped_scores=scores_mapped2d, mapped_dists=dists_mapped2d)
    #     precision2d_all = self.scorer2.score_precision(predictions=scores2, category='Medium')
    #     self.assertEqual(expected_precision2d_all, precision2d_all)
    #     expected_precision2d_k10 = self.check_precision(mapped_scores=scores_mapped2d, mapped_dists=dists_mapped2d,
    #                                                     count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     precision2d_k10 = self.scorer2.score_precision(predictions=scores2, k=10, category='Medium')
    #     self.assertEqual(expected_precision2d_k10, precision2d_k10)
    #     expected_precision2d_n10 = self.check_precision(mapped_scores=scores_mapped2d, mapped_dists=dists_mapped2d,
    #                                                     count=10.0)
    #     precision2d_n10 = self.scorer2.score_precision(predictions=scores2, n=10, category='Medium')
    #     self.assertEqual(expected_precision2d_n10, precision2d_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_precision(predictions=scores2, k=10, n=10, category='Medium')
    #     scores_mapped2e, dists_mapped2e = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Long')
    #     expected_precision2e_all = self.check_precision(mapped_scores=scores_mapped2e, mapped_dists=dists_mapped2e)
    #     precision2e_all = self.scorer2.score_precision(predictions=scores2, category='Long')
    #     self.assertEqual(expected_precision2e_all, precision2e_all)
    #     expected_precision2e_k10 = self.check_precision(mapped_scores=scores_mapped2e, mapped_dists=dists_mapped2e,
    #                                                     count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     precision2e_k10 = self.scorer2.score_precision(predictions=scores2, k=10, category='Long')
    #     self.assertEqual(expected_precision2e_k10, precision2e_k10)
    #     expected_precision2e_n10 = self.check_precision(mapped_scores=scores_mapped2e, mapped_dists=dists_mapped2e,
    #                                                     count=10.0)
    #     precision2e_n10 = self.scorer2.score_precision(predictions=scores2, n=10, category='Long')
    #     self.assertEqual(expected_precision2e_n10, precision2e_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_precision(predictions=scores2, k=10, n=10, category='Long')

    # def test_15a_score_recall(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='CB')
    #     scores1 = np.random.rand(self.seq_len1, self.seq_len1)
    #     scores1[np.tril_indices(self.seq_len1, 1)] = 0
    #     scores1 += scores1.T
    #     scores_mapped1a, dists_mapped1a = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Any')
    #     expected_recall1a_all = self.check_recall(mapped_scores=scores_mapped1a, mapped_dists=dists_mapped1a)
    #     recall1a_all = self.scorer1.score_recall(predictions=scores1, category='Any')
    #     self.assertEqual(expected_recall1a_all, recall1a_all)
    #     expected_recall1a_k10 = self.check_recall(mapped_scores=scores_mapped1a, mapped_dists=dists_mapped1a,
    #                                               count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     recall1a_k10 = self.scorer1.score_recall(predictions=scores1, k=10, category='Any')
    #     self.assertEqual(expected_recall1a_k10, recall1a_k10)
    #     expected_recall1a_n10 = self.check_recall(mapped_scores=scores_mapped1a, mapped_dists=dists_mapped1a,
    #                                               count=10.0)
    #     precision1a_n10 = self.scorer1.score_recall(predictions=scores1, n=10, category='Any')
    #     self.assertEqual(expected_recall1a_n10, precision1a_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_recall(predictions=scores1, k=10, n=10, category='Any')
    #     scores_mapped1b, dists_mapped1b = self.scorer1._map_predictions_to_pdb(predictions=scores1,
    #                                                                            category='Neighbors')
    #     expected_recall1b_all = self.check_recall(mapped_scores=scores_mapped1b, mapped_dists=dists_mapped1b)
    #     recall1b_all = self.scorer1.score_recall(predictions=scores1, category='Neighbors')
    #     self.assertEqual(expected_recall1b_all, recall1b_all)
    #     expected_recall1b_k10 = self.check_recall(mapped_scores=scores_mapped1b, mapped_dists=dists_mapped1b,
    #                                               count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     recall1b_k10 = self.scorer1.score_recall(predictions=scores1, k=10, category='Neighbors')
    #     self.assertEqual(expected_recall1b_k10, recall1b_k10)
    #     expected_recall1b_n10 = self.check_recall(mapped_scores=scores_mapped1b, mapped_dists=dists_mapped1b,
    #                                               count=10.0)
    #     recall1b_n10 = self.scorer1.score_recall(predictions=scores1, n=10, category='Neighbors')
    #     self.assertEqual(expected_recall1b_n10, recall1b_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_recall(predictions=scores1, k=10, n=10, category='Neighbors')
    #     scores_mapped1c, dists_mapped1c = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Short')
    #     expected_recall1c_all = self.check_recall(mapped_scores=scores_mapped1c, mapped_dists=dists_mapped1c)
    #     recall1c_all = self.scorer1.score_recall(predictions=scores1, category='Short')
    #     self.assertEqual(expected_recall1c_all, recall1c_all)
    #     recall1c_k10 = self.scorer1.score_recall(predictions=scores1, k=10, category='Short')
    #     expected_recall1c_k10 = self.check_recall(mapped_scores=scores_mapped1c, mapped_dists=dists_mapped1c,
    #                                               count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     self.assertEqual(expected_recall1c_k10, recall1c_k10)
    #     expected_recall1c_n10 = self.check_recall(mapped_scores=scores_mapped1c, mapped_dists=dists_mapped1c,
    #                                               count=10.0)
    #     recall1c_n10 = self.scorer1.score_recall(predictions=scores1, n=10, category='Short')
    #     self.assertEqual(expected_recall1c_n10, recall1c_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_recall(predictions=scores1, k=10, n=10, category='Short')
    #     scores_mapped1d, dists_mapped1d = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Medium')
    #     expected_recall1d_all = self.check_recall(mapped_scores=scores_mapped1d, mapped_dists=dists_mapped1d)
    #     recall1d_all = self.scorer1.score_recall(predictions=scores1, category='Medium')
    #     self.assertEqual(expected_recall1d_all, recall1d_all)
    #     expected_recall1d_k10 = self.check_recall(mapped_scores=scores_mapped1d, mapped_dists=dists_mapped1d,
    #                                               count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     recall1d_k10 = self.scorer1.score_recall(predictions=scores1, k=10, category='Medium')
    #     self.assertEqual(expected_recall1d_k10, recall1d_k10)
    #     expected_recall1d_n10 = self.check_recall(mapped_scores=scores_mapped1d, mapped_dists=dists_mapped1d,
    #                                               count=10.0)
    #     recall1d_n10 = self.scorer1.score_recall(predictions=scores1, n=10, category='Medium')
    #     self.assertEqual(expected_recall1d_n10, recall1d_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_recall(predictions=scores1, k=10, n=10, category='Medium')
    #     scores_mapped1e, dists_mapped1e = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Long')
    #     expected_recall1e_all = self.check_recall(mapped_scores=scores_mapped1e, mapped_dists=dists_mapped1e)
    #     recall1e_all = self.scorer1.score_recall(predictions=scores1, category='Long')
    #     self.assertEqual(expected_recall1e_all, recall1e_all)
    #     expected_recall1e_k10 = self.check_recall(mapped_scores=scores_mapped1e, mapped_dists=dists_mapped1e,
    #                                               count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     recall1e_k10 = self.scorer1.score_recall(predictions=scores1, k=10, category='Long')
    #     self.assertEqual(expected_recall1e_k10, recall1e_k10)
    #     expected_recall1e_n10 = self.check_recall(mapped_scores=scores_mapped1e, mapped_dists=dists_mapped1e,
    #                                               count=10.0)
    #     recall1e_n10 = self.scorer1.score_recall(predictions=scores1, n=10, category='Long')
    #     self.assertEqual(expected_recall1e_n10, recall1e_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_recall(predictions=scores1, k=10, n=10, category='Long')
    #
    # def test_15b_score_recall(self):
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='CB')
    #     scores2 = np.random.rand(self.seq_len2, self.seq_len2)
    #     scores2[np.tril_indices(self.seq_len2, 1)] = 0
    #     scores2 += scores2.T
    #     scores_mapped2a, dists_mapped2a = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Any')
    #     expected_recall2a_all = self.check_recall(mapped_scores=scores_mapped2a, mapped_dists=dists_mapped2a)
    #     recall2a_all = self.scorer2.score_recall(predictions=scores2, category='Any')
    #     self.assertEqual(expected_recall2a_all, recall2a_all)
    #     expected_recall2a_k10 = self.check_recall(mapped_scores=scores_mapped2a, mapped_dists=dists_mapped2a,
    #                                               count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     recall2a_k10 = self.scorer2.score_recall(predictions=scores2, k=10, category='Any')
    #     self.assertEqual(expected_recall2a_k10, recall2a_k10)
    #     expected_recall2a_n10 = self.check_recall(mapped_scores=scores_mapped2a, mapped_dists=dists_mapped2a,
    #                                               count=10.0)
    #     recall2a_n10 = self.scorer2.score_recall(predictions=scores2, n=10, category='Any')
    #     self.assertEqual(expected_recall2a_n10, recall2a_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_recall(predictions=scores2, k=10, n=10, category='Any')
    #     scores_mapped2b, dists_mapped2b = self.scorer2._map_predictions_to_pdb(predictions=scores2,
    #                                                                            category='Neighbors')
    #     expected_recall2b_all = self.check_recall(mapped_scores=scores_mapped2b, mapped_dists=dists_mapped2b)
    #     recall2b_all = self.scorer2.score_recall(predictions=scores2, category='Neighbors')
    #     self.assertEqual(expected_recall2b_all, recall2b_all)
    #     expected_recall2b_k10 = self.check_recall(mapped_scores=scores_mapped2b, mapped_dists=dists_mapped2b,
    #                                               count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     recall2b_k10 = self.scorer2.score_recall(predictions=scores2, k=10, category='Neighbors')
    #     self.assertEqual(expected_recall2b_k10, recall2b_k10)
    #     expected_recall2b_n10 = self.check_recall(mapped_scores=scores_mapped2b, mapped_dists=dists_mapped2b,
    #                                               count=10.0)
    #     recall2b_n10 = self.scorer2.score_recall(predictions=scores2, n=10, category='Neighbors')
    #     self.assertEqual(expected_recall2b_n10, recall2b_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_recall(predictions=scores2, k=10, n=10, category='Neighbors')
    #     scores_mapped2c, dists_mapped2c = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Short')
    #     expected_recall2c_all = self.check_recall(mapped_scores=scores_mapped2c, mapped_dists=dists_mapped2c)
    #     recall2c_all = self.scorer2.score_recall(predictions=scores2, category='Short')
    #     self.assertEqual(expected_recall2c_all, recall2c_all)
    #     recall2c_k10 = self.scorer2.score_recall(predictions=scores2, k=10, category='Short')
    #     expected_recall2c_k10 = self.check_recall(mapped_scores=scores_mapped2c, mapped_dists=dists_mapped2c,
    #                                               count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     self.assertEqual(expected_recall2c_k10, recall2c_k10)
    #     expected_recall2c_n10 = self.check_recall(mapped_scores=scores_mapped2c, mapped_dists=dists_mapped2c,
    #                                               count=10.0)
    #     recall2c_n10 = self.scorer2.score_recall(predictions=scores2, n=10, category='Short')
    #     self.assertEqual(expected_recall2c_n10, recall2c_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_recall(predictions=scores2, k=10, n=10, category='Short')
    #     scores_mapped2d, dists_mapped2d = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Medium')
    #     expected_recall2d_all = self.check_recall(mapped_scores=scores_mapped2d, mapped_dists=dists_mapped2d)
    #     recall2d_all = self.scorer2.score_recall(predictions=scores2, category='Medium')
    #     self.assertEqual(expected_recall2d_all, recall2d_all)
    #     expected_recall2d_k10 = self.check_recall(mapped_scores=scores_mapped2d, mapped_dists=dists_mapped2d,
    #                                               count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     recall2d_k10 = self.scorer2.score_recall(predictions=scores2, k=10, category='Medium')
    #     self.assertEqual(expected_recall2d_k10, recall2d_k10)
    #     expected_recall2d_n10 = self.check_recall(mapped_scores=scores_mapped2d, mapped_dists=dists_mapped2d,
    #                                               count=10.0)
    #     recall2d_n10 = self.scorer2.score_recall(predictions=scores2, n=10, category='Medium')
    #     self.assertEqual(expected_recall2d_n10, recall2d_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_recall(predictions=scores2, k=10, n=10, category='Medium')
    #     scores_mapped2e, dists_mapped2e = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Long')
    #     expected_recall2e_all = self.check_recall(mapped_scores=scores_mapped2e, mapped_dists=dists_mapped2e)
    #     recall2e_all = self.scorer2.score_recall(predictions=scores2, category='Long')
    #     self.assertEqual(expected_recall2e_all, recall2e_all)
    #     expected_recall2e_k10 = self.check_recall(mapped_scores=scores_mapped2e, mapped_dists=dists_mapped2e,
    #                                               count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     recall2e_k10 = self.scorer2.score_recall(predictions=scores2, k=10, category='Long')
    #     self.assertEqual(expected_recall2e_k10, recall2e_k10)
    #     expected_recall2e_n10 = self.check_recall(mapped_scores=scores_mapped2e, mapped_dists=dists_mapped2e,
    #                                               count=10.0)
    #     recall2e_n10 = self.scorer2.score_recall(predictions=scores2, n=10, category='Long')
    #     self.assertEqual(expected_recall2e_n10, recall2e_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_recall(predictions=scores2, k=10, n=10, category='Long')

    # def test_16a_score_f1(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='CB')
    #     scores1 = np.random.rand(self.seq_len1, self.seq_len1)
    #     scores1[np.tril_indices(self.seq_len1, 1)] = 0
    #     scores1 += scores1.T
    #     scores_mapped1a, dists_mapped1a = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Any')
    #     expected_f1_1a_all = self.check_f1(mapped_scores=scores_mapped1a, mapped_dists=dists_mapped1a)
    #     f1_1a_all = self.scorer1.score_f1(predictions=scores1, category='Any')
    #     self.assertEqual(expected_f1_1a_all, f1_1a_all)
    #     expected_f1_1a_k10 = self.check_f1(mapped_scores=scores_mapped1a, mapped_dists=dists_mapped1a,
    #                                        count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     f1_1a_k10 = self.scorer1.score_f1(predictions=scores1, k=10, category='Any')
    #     self.assertEqual(expected_f1_1a_k10, f1_1a_k10)
    #     expected_f1_1a_n10 = self.check_f1(mapped_scores=scores_mapped1a, mapped_dists=dists_mapped1a,
    #                                        count=10.0)
    #     f1_1a_n10 = self.scorer1.score_f1(predictions=scores1, n=10, category='Any')
    #     self.assertEqual(expected_f1_1a_n10, f1_1a_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_f1(predictions=scores1, k=10, n=10, category='Any')
    #     scores_mapped1b, dists_mapped1b = self.scorer1._map_predictions_to_pdb(predictions=scores1,
    #                                                                            category='Neighbors')
    #     expected_f1_1b_all = self.check_f1(mapped_scores=scores_mapped1b, mapped_dists=dists_mapped1b)
    #     f1_1b_all = self.scorer1.score_f1(predictions=scores1, category='Neighbors')
    #     self.assertEqual(expected_f1_1b_all, f1_1b_all)
    #     expected_f1_1b_k10 = self.check_f1(mapped_scores=scores_mapped1b, mapped_dists=dists_mapped1b,
    #                                        count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     f1_1b_k10 = self.scorer1.score_f1(predictions=scores1, k=10, category='Neighbors')
    #     self.assertEqual(expected_f1_1b_k10, f1_1b_k10)
    #     expected_f1_1b_n10 = self.check_f1(mapped_scores=scores_mapped1b, mapped_dists=dists_mapped1b,
    #                                        count=10.0)
    #     f1_1b_n10 = self.scorer1.score_f1(predictions=scores1, n=10, category='Neighbors')
    #     self.assertEqual(expected_f1_1b_n10, f1_1b_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_f1(predictions=scores1, k=10, n=10, category='Neighbors')
    #     scores_mapped1c, dists_mapped1c = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Short')
    #     expected_f1_1c_all = self.check_f1(mapped_scores=scores_mapped1c, mapped_dists=dists_mapped1c)
    #     f1_1c_all = self.scorer1.score_f1(predictions=scores1, category='Short')
    #     self.assertEqual(expected_f1_1c_all, f1_1c_all)
    #     f1_1c_k10 = self.scorer1.score_f1(predictions=scores1, k=10, category='Short')
    #     expected_f1_1c_k10 = self.check_f1(mapped_scores=scores_mapped1c, mapped_dists=dists_mapped1c,
    #                                        count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     self.assertEqual(expected_f1_1c_k10, f1_1c_k10)
    #     expected_f1_1c_n10 = self.check_f1(mapped_scores=scores_mapped1c, mapped_dists=dists_mapped1c,
    #                                        count=10.0)
    #     f1_1c_n10 = self.scorer1.score_f1(predictions=scores1, n=10, category='Short')
    #     self.assertEqual(expected_f1_1c_n10, f1_1c_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_f1(predictions=scores1, k=10, n=10, category='Short')
    #     scores_mapped1d, dists_mapped1d = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Medium')
    #     expected_f1_1d_all = self.check_f1(mapped_scores=scores_mapped1d, mapped_dists=dists_mapped1d)
    #     f1_1d_all = self.scorer1.score_f1(predictions=scores1, category='Medium')
    #     self.assertEqual(expected_f1_1d_all, f1_1d_all)
    #     expected_f1_1d_k10 = self.check_f1(mapped_scores=scores_mapped1d, mapped_dists=dists_mapped1d,
    #                                        count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     f1_1d_k10 = self.scorer1.score_f1(predictions=scores1, k=10, category='Medium')
    #     self.assertEqual(expected_f1_1d_k10, f1_1d_k10)
    #     expected_f1_1d_n10 = self.check_f1(mapped_scores=scores_mapped1d, mapped_dists=dists_mapped1d,
    #                                        count=10.0)
    #     f1_1d_n10 = self.scorer1.score_f1(predictions=scores1, n=10, category='Medium')
    #     self.assertEqual(expected_f1_1d_n10, f1_1d_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_f1(predictions=scores1, k=10, n=10, category='Medium')
    #     scores_mapped1e, dists_mapped1e = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Long')
    #     expected_f1_1e_all = self.check_f1(mapped_scores=scores_mapped1e, mapped_dists=dists_mapped1e)
    #     f1_1e_all = self.scorer1.score_f1(predictions=scores1, category='Long')
    #     self.assertEqual(expected_f1_1e_all, f1_1e_all)
    #     expected_f1_1e_k10 = self.check_f1(mapped_scores=scores_mapped1e, mapped_dists=dists_mapped1e,
    #                                        count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     f1_1e_k10 = self.scorer1.score_f1(predictions=scores1, k=10, category='Long')
    #     self.assertEqual(expected_f1_1e_k10, f1_1e_k10)
    #     expected_f1_1e_n10 = self.check_f1(mapped_scores=scores_mapped1e, mapped_dists=dists_mapped1e,
    #                                        count=10.0)
    #     f1_1e_n10 = self.scorer1.score_f1(predictions=scores1, n=10, category='Long')
    #     self.assertEqual(expected_f1_1e_n10, f1_1e_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_f1(predictions=scores1, k=10, n=10, category='Long')
    #
    # def test_16b_score_f1(self):
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='CB')
    #     scores2 = np.random.rand(self.seq_len2, self.seq_len2)
    #     scores2[np.tril_indices(self.seq_len2, 1)] = 0
    #     scores2 += scores2.T
    #     scores_mapped2a, dists_mapped2a = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Any')
    #     expected_f1_2a_all = self.check_f1(mapped_scores=scores_mapped2a, mapped_dists=dists_mapped2a)
    #     f1_2a_all = self.scorer2.score_f1(predictions=scores2, category='Any')
    #     self.assertEqual(expected_f1_2a_all, f1_2a_all)
    #     expected_f1_2a_k10 = self.check_f1(mapped_scores=scores_mapped2a, mapped_dists=dists_mapped2a,
    #                                        count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     f1_2a_k10 = self.scorer2.score_f1(predictions=scores2, k=10, category='Any')
    #     self.assertEqual(expected_f1_2a_k10, f1_2a_k10)
    #     expected_f1_2a_n10 = self.check_f1(mapped_scores=scores_mapped2a, mapped_dists=dists_mapped2a,
    #                                        count=10.0)
    #     f1_2a_n10 = self.scorer2.score_f1(predictions=scores2, n=10, category='Any')
    #     self.assertEqual(expected_f1_2a_n10, f1_2a_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_f1(predictions=scores2, k=10, n=10, category='Any')
    #     scores_mapped2b, dists_mapped2b = self.scorer2._map_predictions_to_pdb(predictions=scores2,
    #                                                                            category='Neighbors')
    #     expected_f1_2b_all = self.check_f1(mapped_scores=scores_mapped2b, mapped_dists=dists_mapped2b)
    #     f1_2b_all = self.scorer2.score_f1(predictions=scores2, category='Neighbors')
    #     self.assertEqual(expected_f1_2b_all, f1_2b_all)
    #     expected_f1_2b_k10 = self.check_f1(mapped_scores=scores_mapped2b, mapped_dists=dists_mapped2b,
    #                                        count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     f1_2b_k10 = self.scorer2.score_f1(predictions=scores2, k=10, category='Neighbors')
    #     self.assertEqual(expected_f1_2b_k10, f1_2b_k10)
    #     expected_f1_2b_n10 = self.check_f1(mapped_scores=scores_mapped2b, mapped_dists=dists_mapped2b,
    #                                        count=10.0)
    #     f1_2b_n10 = self.scorer2.score_f1(predictions=scores2, n=10, category='Neighbors')
    #     self.assertEqual(expected_f1_2b_n10, f1_2b_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_f1(predictions=scores2, k=10, n=10, category='Neighbors')
    #     scores_mapped2c, dists_mapped2c = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Short')
    #     expected_f1_2c_all = self.check_f1(mapped_scores=scores_mapped2c, mapped_dists=dists_mapped2c)
    #     f1_2c_all = self.scorer2.score_f1(predictions=scores2, category='Short')
    #     self.assertEqual(expected_f1_2c_all, f1_2c_all)
    #     f1_2c_k10 = self.scorer2.score_f1(predictions=scores2, k=10, category='Short')
    #     expected_f1_2c_k10 = self.check_f1(mapped_scores=scores_mapped2c, mapped_dists=dists_mapped2c,
    #                                        count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     self.assertEqual(expected_f1_2c_k10, f1_2c_k10)
    #     expected_f1_2c_n10 = self.check_f1(mapped_scores=scores_mapped2c, mapped_dists=dists_mapped2c,
    #                                        count=10.0)
    #     f1_2c_n10 = self.scorer2.score_f1(predictions=scores2, n=10, category='Short')
    #     self.assertEqual(expected_f1_2c_n10, f1_2c_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_f1(predictions=scores2, k=10, n=10, category='Short')
    #     scores_mapped2d, dists_mapped2d = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Medium')
    #     expected_f1_2d_all = self.check_f1(mapped_scores=scores_mapped2d, mapped_dists=dists_mapped2d)
    #     f1_2d_all = self.scorer2.score_f1(predictions=scores2, category='Medium')
    #     self.assertEqual(expected_f1_2d_all, f1_2d_all)
    #     expected_f1_2d_k10 = self.check_f1(mapped_scores=scores_mapped2d, mapped_dists=dists_mapped2d,
    #                                        count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     f1_2d_k10 = self.scorer2.score_f1(predictions=scores2, k=10, category='Medium')
    #     self.assertEqual(expected_f1_2d_k10, f1_2d_k10)
    #     expected_f1_2d_n10 = self.check_f1(mapped_scores=scores_mapped2d, mapped_dists=dists_mapped2d,
    #                                        count=10.0)
    #     f1_2d_n10 = self.scorer2.score_f1(predictions=scores2, n=10, category='Medium')
    #     self.assertEqual(expected_f1_2d_n10, f1_2d_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_f1(predictions=scores2, k=10, n=10, category='Medium')
    #     scores_mapped2e, dists_mapped2e = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Long')
    #     expected_f1_2e_all = self.check_f1(mapped_scores=scores_mapped2e, mapped_dists=dists_mapped2e)
    #     f1_2e_all = self.scorer2.score_f1(predictions=scores2, category='Long')
    #     self.assertEqual(expected_f1_2e_all, f1_2e_all)
    #     expected_f1_2e_k10 = self.check_f1(mapped_scores=scores_mapped2e, mapped_dists=dists_mapped2e,
    #                                        count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     f1_2e_k10 = self.scorer2.score_f1(predictions=scores2, k=10, category='Long')
    #     self.assertEqual(expected_f1_2e_k10, f1_2e_k10)
    #     expected_f1_2e_n10 = self.check_f1(mapped_scores=scores_mapped2e, mapped_dists=dists_mapped2e,
    #                                        count=10.0)
    #     f1_2e_n10 = self.scorer2.score_f1(predictions=scores2, n=10, category='Long')
    #     self.assertEqual(expected_f1_2e_n10, f1_2e_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_f1(predictions=scores2, k=10, n=10, category='Long')

    # def test_17a_compute_w2_ave_sub(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     recip_map = {v: k for k, v in self.scorer1.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer1.query_structure.pdb_residue_list[self.scorer1.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     expected_adjacency, res_atoms = self._et_computeAdjacency(
    #         self.scorer1.query_structure.structure[0][self.scorer1.best_chain], mapping=final_map)
    #     init_compute_w2_ave_sub(dists=self.scorer1.distances, bias_bool=True)
    #     cases_biased = {}
    #     for i in range(self.scorer1.distances.shape[0]):
    #         curr_cases = compute_w2_ave_sub(i)
    #         for k in curr_cases:
    #             if k not in cases_biased:
    #                 cases_biased[k] = 0
    #             cases_biased[k] += curr_cases[k]
    #     expected_w2_biased = self._et_calc_w2_sub_problems(A=expected_adjacency, bias=1)
    #     self.assertEqual(cases_biased['Case1'], expected_w2_biased[0])
    #     self.assertEqual(cases_biased['Case2'], expected_w2_biased[1])
    #     self.assertEqual(cases_biased['Case3'], expected_w2_biased[2])
    #
    #     init_compute_w2_ave_sub(dists=self.scorer1.distances, bias_bool=False)
    #     cases_unbiased = {}
    #     for i in range(self.scorer1.distances.shape[0]):
    #         curr_cases = compute_w2_ave_sub(i)
    #         for k in curr_cases:
    #             if k not in cases_unbiased:
    #                 cases_unbiased[k] = 0
    #             cases_unbiased[k] += curr_cases[k]
    #     expected_w2_unbiased = self._et_calc_w2_sub_problems(A=expected_adjacency, bias=0)
    #     self.assertEqual(cases_unbiased['Case1'], expected_w2_unbiased[0])
    #     self.assertEqual(cases_unbiased['Case2'], expected_w2_unbiased[1])
    #     self.assertEqual(cases_unbiased['Case3'], expected_w2_unbiased[2])
    #
    # def test_17b_compute_w2_ave_sub(self):
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     recip_map = {v: k for k, v in self.scorer2.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer2.query_structure.pdb_residue_list[self.scorer2.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     expected_adjacency, res_atoms = self._et_computeAdjacency(
    #         self.scorer2.query_structure.structure[0][self.scorer2.best_chain], mapping=final_map)
    #     init_compute_w2_ave_sub(dists=self.scorer2.distances, bias_bool=True)
    #     cases_biased = {}
    #     for i in range(self.scorer2.distances.shape[0]):
    #         curr_cases = compute_w2_ave_sub(i)
    #         for k in curr_cases:
    #             if k not in cases_biased:
    #                 cases_biased[k] = 0
    #             cases_biased[k] += curr_cases[k]
    #     expected_w2_biased = self._et_calc_w2_sub_problems(A=expected_adjacency, bias=1)
    #     self.assertEqual(cases_biased['Case1'], expected_w2_biased[0])
    #     self.assertEqual(cases_biased['Case2'], expected_w2_biased[1])
    #     self.assertEqual(cases_biased['Case3'], expected_w2_biased[2])
    #     init_compute_w2_ave_sub(dists=self.scorer2.distances, bias_bool=False)
    #     cases_unbiased = {}
    #     for i in range(self.scorer2.distances.shape[0]):
    #         curr_cases = compute_w2_ave_sub(i)
    #         for k in curr_cases:
    #             if k not in cases_unbiased:
    #                 cases_unbiased[k] = 0
    #             cases_unbiased[k] += curr_cases[k]
    #     expected_w2_unbiased = self._et_calc_w2_sub_problems(A=expected_adjacency, bias=0)
    #     self.assertEqual(cases_unbiased['Case1'], expected_w2_unbiased[0])
    #     self.assertEqual(cases_unbiased['Case2'], expected_w2_unbiased[1])
    #     self.assertEqual(cases_unbiased['Case3'], expected_w2_unbiased[2])

    def test_18a_clustering_z_scores(self):
        self.scorer1.fit()
        self.scorer1.measure_distance(method='Any')
        recip_map = {v: k for k, v in self.scorer1.query_pdb_mapping.items()}
        struc_seq_map = {k: i for i, k in
                         enumerate(self.scorer1.query_structure.pdb_residue_list[self.scorer1.best_chain])}
        final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
        expected_adjacency, res_atoms = self._et_computeAdjacency(
            self.scorer1.query_structure.structure[0][self.scorer1.best_chain], mapping=final_map)
        init_compute_w2_ave_sub(dists=self.scorer1.distances, bias_bool=True)
        cases_biased = {}
        for i in range(self.scorer1.distances.shape[0]):
            curr_cases = compute_w2_ave_sub(i)
            for k in curr_cases:
                if k not in cases_biased:
                    cases_biased[k] = 0
                cases_biased[k] += curr_cases[k]
        print(self.scorer1.query_pdb_mapping.keys())
        print(list(self.scorer1.query_pdb_mapping.keys()))
        print(shuffle(list(self.scorer1.query_pdb_mapping.keys())))
        residue_list = list(self.scorer1.query_pdb_mapping.keys())
        print(residue_list)
        shuffle(residue_list)
        print(residue_list)
        init_clustering_z_score(bias_bool=True, w2_ave_sub_dict=cases_biased, curr_pdb=self.scorer1.query_structure,
                                map_to_structure=self.scorer1.query_pdb_mapping, residue_dists=self.scorer1.distances,
                                seq_aln=self.scorer1.query_alignment)
        print('#' * 100)
        for i in range(len(residue_list)):
            curr_residues = residue_list[:(i + 1)]
            a, m, l, pi1, pi2, pi3, z_score, w, w_ave, w2_ave, sigma, num_residues = clustering_z_score(curr_residues)
            em, el, epi1, epi2, epi3, expected_z_score, expected_w, expected_w_ave, expected_w2_ave, expected_sigma = self._et_calcZScore(
                reslist=curr_residues, L=self.scorer1.query_alignment.seq_length, A=expected_adjacency, bias=True)
            print('Z-Scores: {} vs {} - {}'.format(z_score, expected_z_score, (z_score == expected_z_score) if isinstance(z_score, str) else np.abs(z_score - expected_z_score) < 1E-9))
            for res_i in expected_adjacency:
                for res_j in expected_adjacency[res_i]:
                    self.assertEqual(a[self.scorer1.query_pdb_mapping[res_i], self.scorer1.query_pdb_mapping[res_j]],
                                     expected_adjacency[res_i][res_j])
                    a[self.scorer1.query_pdb_mapping[res_i], self.scorer1.query_pdb_mapping[res_j]] = 0
            self.assertEqual(np.sum(a), 0)
            self.assertEqual(m, em)
            self.assertEqual(l, el)
            self.assertLess(np.abs(pi1 - epi1), 1E-16)
            self.assertLess(np.abs(pi2 - epi2), 1E-16)
            self.assertLess(np.abs(pi3 - epi3), 1E-16)
            # if isinstance(z_score, str):
            #     self.assertTrue(isinstance(expected_z_score, str))
            #     self.assertEqual(z_score, expected_z_score, '{} vs {}'.format(z_score, expected_z_score))
            # else:
            #     if z_score < 0:
            #         self.assertTrue(expected_z_score < 0)
            #     else:
            #         self.assertFalse(expected_z_score < 0)
            #     self.assertLess(np.abs(z_score - expected_z_score), 1E-7, '{} vs {}'.format(z_score, expected_z_score))
            # self.assertLess(np.abs(w - expected_w), 1E-9, '{} vs {}'.format(w, expected_w))
            # self.assertLess(np.abs(w_ave - expected_w_ave), 1E-9, '{} vs {}'.format(w_ave, expected_w_ave))
            # self.assertLess(np.abs(w2_ave - expected_w2_ave), 1E-7, '{} vs {}'.format(w2_ave, expected_w2_ave))
            # self.assertLess(np.abs(sigma - expected_sigma), 1E-9, '{} vs {}'.format(sigma, expected_sigma))
            # self.assertEqual(num_residues, len(curr_residues))

        init_compute_w2_ave_sub(dists=self.scorer1.distances, bias_bool=False)
        cases_unbiased = {}
        for i in range(self.scorer1.distances.shape[0]):
            curr_cases = compute_w2_ave_sub(i)
            for k in curr_cases:
                if k not in cases_unbiased:
                    cases_unbiased[k] = 0
                cases_unbiased[k] += curr_cases[k]
        init_clustering_z_score(bias_bool=False, w2_ave_sub_dict=cases_unbiased, curr_pdb=self.scorer1.query_structure,
                                map_to_structure=self.scorer1.query_pdb_mapping, residue_dists=self.scorer1.distances,
                                seq_aln=self.scorer1.query_alignment)
        print('#' * 100)
        for i in range(len(residue_list)):
            curr_residues = residue_list[:(i + 1)]
            a, m, l, pi1, pi2, pi3, z_score, w, w_ave, w2_ave, sigma, num_residues = clustering_z_score(curr_residues)
            em, el, epi1, epi2, epi3, expected_z_score, expected_w, expected_w_ave, expected_w2_ave, expected_sigma = self._et_calcZScore(
                reslist=curr_residues, L=self.scorer1.query_alignment.seq_length, A=expected_adjacency, bias=False)
            print('Z-Scores: {} vs {} - {}'.format(z_score, expected_z_score, (z_score == expected_z_score) if isinstance(z_score, str) else np.abs(z_score - expected_z_score) < 1E-9))
            for res_i in expected_adjacency:
                for res_j in expected_adjacency[res_i]:
                    self.assertEqual(a[self.scorer1.query_pdb_mapping[res_i], self.scorer1.query_pdb_mapping[res_j]],
                                     expected_adjacency[res_i][res_j])
                    a[self.scorer1.query_pdb_mapping[res_i], self.scorer1.query_pdb_mapping[res_j]] = 0
            self.assertEqual(np.sum(a), 0)
            self.assertEqual(m, em)
            self.assertEqual(l, el)
            self.assertLess(np.abs(pi1 - epi1), 1E-16)
            self.assertLess(np.abs(pi2 - epi2), 1E-16)
            self.assertLess(np.abs(pi3 - epi3), 1E-16)
            if isinstance(z_score, str):
                self.assertTrue(isinstance(expected_z_score, str))
                self.assertEqual(z_score, expected_z_score, '{} vs {}'.format(z_score, expected_z_score))
            else:
                if z_score < 0:
                    self.assertTrue(expected_z_score < 0)
                else:
                    self.assertFalse(expected_z_score < 0)
                self.assertLess(np.abs(z_score - expected_z_score), 1E-7, '{} vs {}'.format(z_score, expected_z_score))
            self.assertLess(np.abs(w - expected_w), 1E-9, '{} vs {}'.format(w, expected_w))
            self.assertLess(np.abs(w_ave - expected_w_ave), 1E-9, '{} vs {}'.format(w_ave, expected_w_ave))
            self.assertLess(np.abs(w2_ave - expected_w2_ave), 1E-7, '{} vs {}'.format(w2_ave, expected_w2_ave))
            self.assertLess(np.abs(sigma - expected_sigma), 1E-9, '{} vs {}'.format(sigma, expected_sigma))
            self.assertEqual(num_residues, len(curr_residues))

    # def test_18b_clustering_z_scores(self):
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     recip_map = {v: k for k, v in self.scorer2.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer2.query_structure.pdb_residue_list[self.scorer2.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     expected_adjacency, res_atoms = self._et_computeAdjacency(
    #         self.scorer2.query_structure.structure[0][self.scorer2.best_chain], mapping=final_map)
    #     init_compute_w2_ave_sub(dists=self.scorer2.distances, bias_bool=True)
    #     cases_biased = {}
    #     for i in range(self.scorer2.distances.shape[0]):
    #         curr_cases = compute_w2_ave_sub(i)
    #         for k in curr_cases:
    #             if k not in cases_biased:
    #                 cases_biased[k] = 0
    #             cases_biased[k] += curr_cases[k]
    #     residue_list = shuffle(list(self.scorer2.query_pdb_mapping.keys()))
    #     init_clustering_z_score(bias_bool=True, w2_ave_sub_dict=cases_biased, curr_pdb=self.scorer2.query_structure,
    #                             map_to_structure=self.scorer2.query_pdb_mapping, residue_dists=self.scorer2.distances,
    #                             seq_aln=self.scorer2.query_alignment)
    #     for i in range(len(residue_list)):
    #         curr_residues = residue_list[:(i + 1)]
    #         z_score, w, w_ave, w2_ave, sigma, num_residues = clustering_z_score(curr_residues)
    #         expected_z_score, expected_w, expected_w_ave, expecteed_w2_ave, expected_sigma = self._et_calcZScore(
    #             reslist=curr_residues, L=self.scorer2.query_alignment.seq_length, A=expected_adjacency, bias=True)
    #         self.assertEqual(z_score, expected_z_score)
    #         self.assertEqual(w, expected_w)
    #         self.assertEqual(w_ave, expected_w_ave)
    #         self.assertEqual(w2_ave, expecteed_w2_ave)
    #         self.assertEqual(sigma, expected_sigma)
    #         self.assertEqual(num_residues, len(curr_residues))
    #
    #
    #     init_compute_w2_ave_sub(dists=self.scorer2.distances, bias_bool=False)
    #     cases_unbiased = {}
    #     for i in range(self.scorer2.distances.shape[0]):
    #         curr_cases = compute_w2_ave_sub(i)
    #         for k in curr_cases:
    #             if k not in cases_unbiased:
    #                 cases_unbiased[k] = 0
    #             cases_unbiased[k] += curr_cases[k]
    #     init_clustering_z_score(bias_bool=False, w2_ave_sub_dict=cases_unbiased, curr_pdb=self.scorer2.query_structure,
    #                             map_to_structure=self.scorer2.query_pdb_mapping, residue_dists=self.scorer2.distances,
    #                             seq_aln=self.scorer2.query_alignment)
    #     for i in range(len(residue_list)):
    #         curr_residues = residue_list[:(i + 1)]
    #         z_score, w, w_ave, w2_ave, sigma, num_residues = clustering_z_score(curr_residues)
    #         expected_z_score, expected_w, expected_w_ave, expecteed_w2_ave, expected_sigma = self._et_calcZScore(
    #             reslist=curr_residues, L=self.scorer2.query_alignment.seq_length, A=expected_adjacency, bias=False)
    #         self.assertEqual(z_score, expected_z_score)
    #         self.assertEqual(w, expected_w)
    #         self.assertEqual(w_ave, expected_w_ave)
    #         self.assertEqual(w2_ave, expecteed_w2_ave)
    #         self.assertEqual(sigma, expected_sigma)
    #         self.assertEqual(num_residues, len(curr_residues))

    # def test_17a_score_clustering_of_contact_predictions(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     scores1 = np.random.RandomState(1234567890).rand(79, 79)
    #     scores1[np.tril_indices(79, 1)] = 0
    #     scores1 += scores1.T
    #     recip_map = {v: k for k, v in self.scorer1.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer1.query_structure.pdb_residue_list[self.scorer1.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     A, res_atoms = self._et_computeAdjacency(self.scorer1.query_structure.structure[0][self.scorer1.best_chain],
    #                                              mapping=final_map)
    #     start1 = time()
    #     zscore_df_1b, _ = self.scorer1.score_clustering_of_contact_predictions(
    #         predictions=scores1, bias=True, file_path=os.path.join(os.path.abspath('../Test/'), 'z_score1b.tsv'),
    #         w2_ave_sub=None)
    #     end1 = time()
    #     print('Time for ContactScorer to compute SCW: {}'.format((end1 - start1) / 60.0))
    #     self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'z_score1b.tsv')))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'z_score1b.tsv'))
    #     start2 = time()
    #     zscore_df_1be = self.all_z_scores(A=A, L=self.scorer1.query_alignment.seq_length, bias=1,
    #                                       res_i=list(zscore_df_1b['Res_i']), res_j=list(zscore_df_1b['Res_j']),
    #                                       scores=list(zscore_df_1b['Covariance_Score']))
    #     end2 = time()
    #     print('Time for Rhonalds method to compute SCW: {}'.format((end2 - start2) / 60.0))
    #     # Covariance score comparison
    #     cov_score_diff = np.abs(np.array(zscore_df_1b['Covariance_Score']) -
    #                             np.array(zscore_df_1be['Covariance_Score']))
    #     cov_score_diff = cov_score_diff > 1e-10
    #     self.assertEqual(len(np.nonzero(cov_score_diff)[0]), 0)
    #     # Num residues comparison
    #     self.assertEqual(len(np.nonzero(np.array(zscore_df_1b['Num_Residues']) -
    #                                     np.array(zscore_df_1be['Num_Residues']))[0]), 0)
    #     # Res I comparison
    #     self.assertEqual(len(np.nonzero(np.array(zscore_df_1b['Res_i']) -
    #                                     np.array(zscore_df_1be['Res_i']))[0]), 0)
    #     # Res J comparison
    #     self.assertEqual(len(np.nonzero(np.array(zscore_df_1b['Res_j']) -
    #                                     np.array(zscore_df_1be['Res_j']))[0]), 0)
    #     # W Comparison
    #     w_diff = np.abs(np.array(zscore_df_1b['W']) - np.array(zscore_df_1be['W']))
    #     w_diff = w_diff > 1e-10
    #     self.assertEqual(len(np.nonzero(w_diff)[0]), 0)
    #     # W_Ave Comaparison
    #     w_ave_diff = np.abs(np.array(zscore_df_1b['W_Ave']) - np.array(zscore_df_1be['W_Ave']))
    #     w_ave_diff = w_ave_diff > 1e-7
    #     self.assertEqual(len(np.nonzero(w_ave_diff)[0]), 0)
    #     # W2_Ave Comparison
    #     w2_ave_diff = np.abs(np.array(zscore_df_1b['W2_Ave']) - np.array(zscore_df_1be['W2_Ave']))
    #     w2_ave_diff = w2_ave_diff > 1e-3
    #     self.assertEqual(len(np.nonzero(w2_ave_diff)[0]), 0)
    #     # Sigma Comparison
    #     sigma_diff = np.abs(np.array(zscore_df_1b['Sigma']) - np.array(zscore_df_1be['Sigma']))
    #     sigma_diff = sigma_diff > 1e-6
    #     self.assertEqual(len(np.nonzero(sigma_diff)[0]), 0)
    #     # Z-Score Comparison
    #     z_score_diff = np.abs(np.array(zscore_df_1b['Z-Score'].replace('NA', np.nan)) -\
    #                    np.array(zscore_df_1be['Z-Score'].replace('NA', np.nan)))
    #     z_score_diff = z_score_diff > 1e-7
    #     self.assertEqual(len(np.nonzero(z_score_diff)[0]), 0)
    #     start3 = time()
    #     zscore_df_1u, _ = self.scorer1.score_clustering_of_contact_predictions(
    #         predictions=scores1, bias=False, file_path=os.path.join(os.path.abspath('../Test/'), 'z_score1u.tsv'),
    #         w2_ave_sub=None)
    #     end3 = time()
    #     print('Time for ContactScorer to compute SCW: {}'.format((end3 - start3) / 60.0))
    #     self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'z_score1u.tsv')))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'z_score1u.tsv'))
    #     start4 = time()
    #     zscore_df_1ue = self.all_z_scores(A=A, L=self.scorer1.query_alignment.seq_length, bias=0,
    #                                       res_i=list(zscore_df_1u['Res_i']), res_j=list(zscore_df_1u['Res_j']),
    #                                       scores=list(zscore_df_1u['Covariance_Score']))
    #     end4 = time()
    #     print('Time for Rhonalds method to compute SCW: {}'.format((end4 - start4) / 60.0))
    #     # Covariance score comparison
    #     cov_score_diff = np.abs(np.array(zscore_df_1u['Covariance_Score']) -
    #                             np.array(zscore_df_1ue['Covariance_Score']))
    #     cov_score_diff = cov_score_diff > 1e-10
    #     self.assertEqual(len(np.nonzero(cov_score_diff)[0]), 0)
    #     # Num residues comparison
    #     self.assertEqual(len(np.nonzero(np.array(zscore_df_1u['Num_Residues']) -
    #                                     np.array(zscore_df_1ue['Num_Residues']))[0]), 0)
    #     # Res I comparison
    #     self.assertEqual(len(np.nonzero(np.array(zscore_df_1u['Res_i']) -
    #                                     np.array(zscore_df_1ue['Res_i']))[0]), 0)
    #     # Res J comparison
    #     self.assertEqual(len(np.nonzero(np.array(zscore_df_1u['Res_j']) -
    #                                     np.array(zscore_df_1ue['Res_j']))[0]), 0)
    #     # W Comparison
    #     w_diff = np.abs(np.array(zscore_df_1u['W']) - np.array(zscore_df_1ue['W']))
    #     w_diff = w_diff > 1e-10
    #     self.assertEqual(len(np.nonzero(w_diff)[0]), 0)
    #     # W_Ave Comaparison
    #     w_ave_diff = np.abs(np.array(zscore_df_1u['W_Ave']) - np.array(zscore_df_1ue['W_Ave']))
    #     w_ave_diff = w_ave_diff > 1e-7
    #     self.assertEqual(len(np.nonzero(w_ave_diff)[0]), 0)
    #     # W2_Ave Comparison
    #     w2_ave_diff = np.abs(np.array(zscore_df_1u['W2_Ave']) - np.array(zscore_df_1ue['W2_Ave']))
    #     w2_ave_diff = w2_ave_diff > 1e-3
    #     self.assertEqual(len(np.nonzero(w2_ave_diff)[0]), 0)
    #     # Sigma Comparison
    #     sigma_diff = np.abs(np.array(zscore_df_1u['Sigma']) - np.array(zscore_df_1ue['Sigma']))
    #     sigma_diff = sigma_diff > 1e-6
    #     self.assertEqual(len(np.nonzero(sigma_diff)[0]), 0)
    #     # Z-Score Comparison
    #     z_score_diff = np.abs(np.array(zscore_df_1u['Z-Score'].replace('NA', np.nan)) -\
    #                    np.array(zscore_df_1ue['Z-Score'].replace('NA', np.nan)))
    #     z_score_diff = z_score_diff > 1e-7
    #     self.assertEqual(len(np.nonzero(z_score_diff)[0]), 0)
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     scores2 = np.random.RandomState(1234567890).rand(368, 368)
    #     scores2[np.tril_indices(368, 1)] = 0
    #     scores2 += scores2.T
    #     recip_map = {v: k for k, v in self.scorer2.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer2.query_structure.pdb_residue_list[self.scorer2.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     A, res_atoms = self._et_computeAdjacency(self.scorer2.query_structure.structure[0][self.scorer2.best_chain],
    #                                              mapping=final_map)
    #     start5 = time()
    #     zscore_df_2b, _ = self.scorer2.score_clustering_of_contact_predictions(
    #         predictions=scores2, bias=True, file_path=os.path.join(os.path.abspath('../Test/'), 'z_score2b.tsv'),
    #         w2_ave_sub=None)
    #     end5 = time()
    #     print('Time for ContactScorer to compute SCW: {}'.format((end5 - start5) / 60.0))
    #     self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'z_score2b.tsv')))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'z_score2b.tsv'))
    #     start6 = time()
    #     zscore_df_2be = self.all_z_scores(A=A, L=self.scorer2.query_alignment.seq_length, bias=1,
    #                                       res_i=list(zscore_df_2b['Res_i']), res_j=list(zscore_df_2b['Res_j']),
    #                                       scores=list(zscore_df_2b['Covariance_Score']))
    #     end6 = time()
    #     print('Time for Rhonalds method to compute SCW: {}'.format((end6 - start6) / 60.0))
    #     # Covariance score comparison
    #     cov_score_diff = np.abs(np.array(zscore_df_2b['Covariance_Score']) -
    #                             np.array(zscore_df_2be['Covariance_Score']))
    #     cov_score_diff = cov_score_diff > 1e-10
    #     self.assertEqual(len(np.nonzero(cov_score_diff)[0]), 0)
    #     # Num residues comparison
    #     self.assertEqual(len(np.nonzero(np.array(zscore_df_2b['Num_Residues']) -
    #                                     np.array(zscore_df_2be['Num_Residues']))[0]), 0)
    #     # Res I comparison
    #     self.assertEqual(len(np.nonzero(np.array(zscore_df_2b['Res_i']) -
    #                                     np.array(zscore_df_2be['Res_i']))[0]), 0)
    #     # Res J comparison
    #     self.assertEqual(len(np.nonzero(np.array(zscore_df_2b['Res_j']) -
    #                                     np.array(zscore_df_2be['Res_j']))[0]), 0)
    #     # W Comparison
    #     w_diff = np.abs(np.array(zscore_df_2b['W']) - np.array(zscore_df_2be['W']))
    #     w_diff = w_diff > 1e-10
    #     self.assertEqual(len(np.nonzero(w_diff)[0]), 0)
    #     # W_Ave Comaparison
    #     w_ave_diff = np.abs(np.array(zscore_df_2b['W_Ave']) - np.array(zscore_df_2be['W_Ave']))
    #     w_ave_diff = w_ave_diff > 1e-6
    #     self.assertEqual(len(np.nonzero(w_ave_diff)[0]), 0)
    #     # W2_Ave Comparison
    #     w2_ave_diff = np.abs(np.array(zscore_df_2b['W2_Ave']) - np.array(zscore_df_2be['W2_Ave']))
    #     w2_ave_diff = w2_ave_diff > 5
    #     self.assertEqual(len(np.nonzero(w2_ave_diff)[0]), 0)
    #     # Sigma Comparison
    #     sigma_diff = np.abs(np.array(zscore_df_2b['Sigma']) - np.array(zscore_df_2be['Sigma']))
    #     sigma_diff = sigma_diff > 1e-3
    #     self.assertEqual(len(np.nonzero(sigma_diff)[0]), 0)
    #     # Z-Score Comparison
    #     z_score_diff = np.abs(np.array(zscore_df_2b['Z-Score'].replace('NA', np.nan)) -\
    #                           np.array(zscore_df_2be['Z-Score'].replace('NA', np.nan)))
    #     z_score_diff = z_score_diff > 1e-1
    #     self.assertEqual(len(np.nonzero(z_score_diff)[0]), 0)
    #     start7 = time()
    #     zscore_df_2u, _ = self.scorer2.score_clustering_of_contact_predictions(
    #         predictions=scores2, bias=False, file_path=os.path.join(os.path.abspath('../Test/'), 'z_score2u.tsv'),
    #         w2_ave_sub=None)
    #     end7 = time()
    #     print('Time for ContactScorer to compute SCW: {}'.format((end7 - start7) / 60.0))
    #     self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'z_score2u.tsv')))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'z_score2u.tsv'))
    #     start8 = time()
    #     zscore_df_2ue = self.all_z_scores(A=A, L=self.scorer2.query_alignment.seq_length, bias=0,
    #                                       res_i=list(zscore_df_2u['Res_i']), res_j=list(zscore_df_2u['Res_j']),
    #                                       scores=list(zscore_df_2u['Covariance_Score']))
    #     end8 = time()
    #     print('Time for Rhonalds method to compute SCW: {}'.format((end8 - start8) / 60.0))
    #     # Covariance score comparison
    #     cov_score_diff = np.abs(np.array(zscore_df_2u['Covariance_Score']) -
    #                             np.array(zscore_df_2ue['Covariance_Score']))
    #     cov_score_diff = cov_score_diff > 1e-10
    #     self.assertEqual(len(np.nonzero(cov_score_diff)[0]), 0)
    #     # Num residues comparison
    #     self.assertEqual(len(np.nonzero(np.array(zscore_df_2u['Num_Residues']) -
    #                                     np.array(zscore_df_2ue['Num_Residues']))[0]), 0)
    #     # Res I comparison
    #     self.assertEqual(len(np.nonzero(np.array(zscore_df_2u['Res_i']) -
    #                                     np.array(zscore_df_2ue['Res_i']))[0]), 0)
    #     # Res J comparison
    #     self.assertEqual(len(np.nonzero(np.array(zscore_df_2u['Res_j']) -
    #                                     np.array(zscore_df_2ue['Res_j']))[0]), 0)
    #     # W Comparison
    #     w_diff = np.abs(np.array(zscore_df_2u['W']) - np.array(zscore_df_2ue['W']))
    #     w_diff = w_diff > 1e-10
    #     self.assertEqual(len(np.nonzero(w_diff)[0]), 0)
    #     # W_Ave Comaparison
    #     w_ave_diff = np.abs(np.array(zscore_df_2u['W_Ave']) - np.array(zscore_df_2ue['W_Ave']))
    #     w_ave_diff = w_ave_diff > 1e-7
    #     self.assertEqual(len(np.nonzero(w_ave_diff)[0]), 0)
    #     # W2_Ave Comparison
    #     w2_ave_diff = np.abs(np.array(zscore_df_2u['W2_Ave']) - np.array(zscore_df_2ue['W2_Ave']))
    #     w2_ave_diff = w2_ave_diff > 1e-1
    #     self.assertEqual(len(np.nonzero(w2_ave_diff)[0]), 0)
    #     # Sigma Comparison
    #     sigma_diff = np.abs(np.array(zscore_df_2u['Sigma']) - np.array(zscore_df_2ue['Sigma']))
    #     sigma_diff = sigma_diff > 1e-3
    #     self.assertEqual(len(np.nonzero(sigma_diff)[0]), 0)
    #     # Z-Score Comparison
    #     z_score_diff = np.abs(np.array(zscore_df_2u['Z-Score'].replace('NA', np.nan)) - \
    #                           np.array(zscore_df_2ue['Z-Score'].replace('NA', np.nan)))
    #     z_score_diff = z_score_diff > 1e-3
    #     self.assertEqual(len(np.nonzero(z_score_diff)[0]), 0)
    # 
    # def test__clustering_z_score(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     recip_map = {v: k for k, v in self.scorer1.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer1.query_structure.pdb_residue_list[self.scorer1.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     A, res_atoms = self._et_computeAdjacency(self.scorer1.query_structure.structure[0][self.scorer1.best_chain],
    #                                              mapping=final_map)
    #     z_scores_e1b, w_e1b, w_ave_e1b, w2_ave_e1b, sigma_e1b = self._et_calcZScore(
    #         reslist=range(self.scorer1.query_alignment.seq_length), L=self.scorer1.query_alignment.seq_length, A=A,
    #         bias=1)
    #     z_scores_1b, w_1b, w_ave_1b, w2_ave_1b, sigma_1b, _ = self.scorer1._clustering_z_score(
    #         res_list=range(self.scorer1.query_alignment.seq_length), bias=True, w2_ave_sub=None)
    #     if isinstance(z_scores_1b, str):
    #         self.assertEqual(z_scores_1b, 'NA')
    #         self.assertEqual(z_scores_e1b, 'NA')
    #     else:
    #         self.assertEqual(z_scores_1b, z_scores_e1b)
    #     self.assertEqual(w_1b, w_e1b)
    #     self.assertEqual(w_ave_1b, w_ave_e1b)
    #     self.assertEqual(w2_ave_1b, w2_ave_e1b)
    #     self.assertEqual(sigma_1b, sigma_e1b)
    #     z_scores_e1u, w_e1u, w_ave_e1u, w2_ave_e1u, sigma_e1u = self._et_calcZScore(
    #         reslist=range(self.scorer1.query_alignment.seq_length), L=self.scorer1.query_alignment.seq_length, A=A,
    #         bias=0)
    #     z_scores_1u, w_1u, w_ave_1u, w2_ave_1u, sigma_1u, _ = self.scorer1._clustering_z_score(
    #         res_list=range(self.scorer1.query_alignment.seq_length), bias=False, w2_ave_sub=None)
    #     if isinstance(z_scores_1u, str):
    #         self.assertEqual(z_scores_1u, 'NA')
    #         self.assertEqual(z_scores_e1u, 'NA')
    #     else:
    #         self.assertEqual(z_scores_1u, z_scores_e1u)
    #     self.assertEqual(w_1u, w_e1u)
    #     self.assertEqual(w_ave_1u, w_ave_e1u)
    #     self.assertEqual(w2_ave_1u, w2_ave_e1u)
    #     self.assertEqual(sigma_1u, sigma_e1u)
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     recip_map = {v: k for k, v in self.scorer2.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer2.query_structure.pdb_residue_list[self.scorer2.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     A, res_atoms = self._et_computeAdjacency(self.scorer2.query_structure.structure[0][self.scorer2.best_chain],
    #                                              mapping=final_map)
    #     z_scores_e2b, w_e2b, w_ave_e2b, w2_ave_e2b, sigma_e2b = self._et_calcZScore(
    #         reslist=range(self.scorer2.query_alignment.seq_length), L=self.scorer2.query_alignment.seq_length, A=A,
    #         bias=1)
    #     z_scores_2b, w_2b, w_ave_2b, w2_ave_2b, sigma_2b, _ = self.scorer2._clustering_z_score(
    #         res_list=range(self.scorer2.query_alignment.seq_length), bias=True, w2_ave_sub=None)
    #     if isinstance(z_scores_2b, str):
    #         self.assertEqual(z_scores_2b, 'NA')
    #         self.assertEqual(z_scores_e2b, 'NA')
    #     else:
    #         self.assertEqual(z_scores_2b, z_scores_e2b)
    #     self.assertEqual(w_2b, w_e2b)
    #     self.assertEqual(w_ave_2b, w_ave_e2b)
    #     self.assertEqual(w2_ave_2b, w2_ave_e2b)
    #     self.assertEqual(sigma_2b, sigma_e2b)
    #     z_scores_e2u, w_e2u, w_ave_e2u, w2_ave_e2u, sigma_e2u = self._et_calcZScore(
    #         reslist=range(self.scorer2.query_alignment.seq_length), L=self.scorer2.query_alignment.seq_length, A=A,
    #         bias=0)
    #     z_scores_2u, w_2u, w_ave_2u, w2_ave_2u, sigma_2u, _ = self.scorer2._clustering_z_score(
    #         res_list=range(self.scorer2.query_alignment.seq_length), bias=False, w2_ave_sub=None)
    #     if isinstance(z_scores_2u, str):
    #         self.assertEqual(z_scores_2u, 'NA')
    #         self.assertEqual(z_scores_e2u, 'NA')
    #     else:
    #         self.assertEqual(z_scores_2u, z_scores_e2u)
    #     self.assertEqual(w_2u, w_e2u)
    #     self.assertEqual(w_ave_2u, w_ave_e2u)
    #     self.assertEqual(w2_ave_2u, w2_ave_e2u)
    #     self.assertEqual(sigma_2u, sigma_e2u)
    # 
    # def test_write_out_clustering_results(self):
    #     def comp_function(df, q_ind_map, q_to_s_map, seq_pdb_map, seq, scores, coverages, distances, adjacencies):
    #         for i in df.index:
    #             # Mapping to Structure
    #             pos1 = q_to_s_map[q_ind_map[df.loc[i, 'Pos1']]]
    #             pos2 = q_to_s_map[q_ind_map[df.loc[i, 'Pos2']]]
    #             self.assertEqual(df.loc[i, '(AA1)'], '({})'.format(one_to_three(seq[seq_pdb_map[pos1]])),
    #                              'Positions: {}\t{}'.format(pos1, pos2))
    #             self.assertEqual(df.loc[i, '(AA2)'], '({})'.format(one_to_three(seq[seq_pdb_map[pos2]])),
    #                              'Positions: {}\t{}'.format(pos1, pos2))
    #             # Scores
    #             self.assertLess(np.abs(df.loc[i, 'Raw_Score'] - scores[pos1, pos2]), 1e-3,
    #                             'Positions: {}\t{}'.format(pos1, pos2))
    #             # Coverages
    #             self.assertLess(np.abs(df.loc[i, 'Coverage_Score'] - coverages[pos1, pos2]), 1e-4,
    #                             'Positions: {}\t{}'.format(pos1, pos2))
    #             # Distances
    #             self.assertLess(np.abs(df.loc[i, 'Residue_Dist'] - distances[pos1, pos2]), 1e-4,
    #                             'Positions: {}\t{}'.format(pos1, pos2))
    #             # Contacts
    #             if df.loc[i, 'Within_Threshold'] == 1:
    #                 try:
    #                     self.assertEqual(df.loc[i, 'Within_Threshold'], adjacencies[pos1][pos2],
    #                                      'Positions: {}\t{}'.format(pos1, pos2))
    #                 except KeyError:
    #                     self.assertEqual(df.loc[i, 'Within_Threshold'], adjacencies[pos2][pos1],
    #                                      'Positions: {}\t{}'.format(pos1, pos2))
    #             else:
    #                 with self.assertRaises(KeyError):
    #                     adjacencies[pos2][pos1]
    # 
    #     today = str(datetime.date.today())
    #     header = ['Pos1', '(AA1)', 'Pos2', '(AA2)', 'Raw_Score', 'Coverage_Score', 'Residue_Dist', 'Within_Threshold']
    #     save_dir = os.path.abspath('../Test')
    #     #
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     recip_map = {v: k for k, v in self.scorer1.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer1.query_structure.pdb_residue_list[self.scorer1.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     A, res_atoms = self._et_computeAdjacency(self.scorer1.query_structure.structure[0][self.scorer1.best_chain],
    #                                              mapping=final_map)
    #     pdb_query_mapping = {v: k for k, v in self.scorer1.query_pdb_mapping.items()}
    #     pdb_index_mapping = {k: i for i, k in enumerate(self.scorer1.query_structure.pdb_residue_list[self.scorer1.best_chain])}
    #     scores1 = np.random.RandomState(1234567890).rand(79, 79)
    #     scores1[np.tril_indices(79, 1)] = 0
    #     scores1 += scores1.T
    #     coverages1 = np.random.RandomState(179424691).rand(79, 79)
    #     coverages1[np.tril_indices(79, 1)] = 0
    #     coverages1 += coverages1.T
    #     self.scorer1.write_out_clustering_results(today=today, raw_scores=scores1, coverage_scores=coverages1,
    #                                               file_name='Contact_1a_Scores.tsv', output_dir=save_dir)
    #     curr_path = os.path.join(save_dir, 'Contact_1a_Scores.tsv')
    #     self.assertTrue(os.path.isfile(curr_path))
    #     test_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    #     self.assertEqual(list(test_df.columns), header)
    #     comp_function(df=test_df, q_ind_map=pdb_index_mapping, q_to_s_map=pdb_query_mapping,
    #                   seq_pdb_map=self.scorer1.query_pdb_mapping,
    #                   seq=self.scorer1.query_alignment.query_sequence, scores=scores1, coverages=coverages1,
    #                   distances=self.scorer1.distances, adjacencies=A)
    #     os.remove(curr_path)
    #     self.scorer1.write_out_clustering_results(today=None, raw_scores=scores1, coverage_scores=coverages1,
    #                                               file_name='Contact_1b_Scores.tsv', output_dir=save_dir)
    #     curr_path = os.path.join(save_dir, 'Contact_1b_Scores.tsv')
    #     self.assertTrue(os.path.isfile(curr_path))
    #     test_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    #     self.assertEqual(list(test_df.columns), header)
    #     comp_function(df=test_df, q_ind_map=pdb_index_mapping, q_to_s_map=pdb_query_mapping,
    #                   seq_pdb_map=self.scorer1.query_pdb_mapping,
    #                   seq=self.scorer1.query_alignment.query_sequence, scores=scores1, coverages=coverages1,
    #                   distances=self.scorer1.distances, adjacencies=A)
    #     os.remove(curr_path)
    # 
    #     self.scorer1.write_out_clustering_results(today=today, raw_scores=scores1, coverage_scores=coverages1,
    #                                               file_name=None, output_dir=save_dir)
    #     curr_path = os.path.join(save_dir, "{}_{}.Covariance_vs_Structure.txt".format(today, self.scorer1.query))
    #     self.assertTrue(os.path.isfile(curr_path))
    #     test_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    #     self.assertEqual(list(test_df.columns), header)
    #     comp_function(df=test_df, q_ind_map=pdb_index_mapping, q_to_s_map=pdb_query_mapping,
    #                   seq_pdb_map=self.scorer1.query_pdb_mapping,
    #                   seq=self.scorer1.query_alignment.query_sequence, scores=scores1,coverages=coverages1,
    #                   distances=self.scorer1.distances, adjacencies=A)
    #     os.remove(curr_path)
    #     #
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     recip_map = {v: k for k, v in self.scorer2.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer2.query_structure.pdb_residue_list[self.scorer2.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     A, res_atoms = self._et_computeAdjacency(self.scorer2.query_structure.structure[0][self.scorer2.best_chain],
    #                                              mapping=final_map)
    #     pdb_query_mapping = {v: k for k, v in self.scorer2.query_pdb_mapping.items()}
    #     pdb_index_mapping = {k: i for i, k in
    #                          enumerate(self.scorer2.query_structure.pdb_residue_list[self.scorer2.best_chain])}
    #     scores2 = np.random.RandomState(1234567890).rand(368, 368)
    #     scores2[np.tril_indices(368, 1)] = 0
    #     scores2 += scores2.T
    #     coverages2 = np.random.RandomState(179424691).rand(368, 368)
    #     coverages2[np.tril_indices(368, 1)] = 0
    #     coverages2 += coverages2.T
    #     self.scorer2.write_out_clustering_results(today=today, raw_scores=scores2, coverage_scores=coverages2,
    #                                               file_name='Contact_2a_Scores.tsv', output_dir=save_dir)
    #     curr_path = os.path.join(save_dir, 'Contact_2a_Scores.tsv')
    #     self.assertTrue(os.path.isfile(curr_path))
    #     test_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    #     self.assertEqual(list(test_df.columns), header)
    #     comp_function(df=test_df, q_ind_map=pdb_index_mapping, q_to_s_map=pdb_query_mapping,
    #                   seq_pdb_map=self.scorer2.query_pdb_mapping,
    #                   seq=self.scorer2.query_alignment.query_sequence, scores=scores2, coverages=coverages2,
    #                   distances=self.scorer2.distances, adjacencies=A)
    #     os.remove(curr_path)
    #     self.scorer2.write_out_clustering_results(today=None, raw_scores=scores2, coverage_scores=coverages2,
    #                                               file_name='Contact_2b_Scores.tsv', output_dir=save_dir)
    #     curr_path = os.path.join(save_dir, 'Contact_2b_Scores.tsv')
    #     self.assertTrue(os.path.isfile(curr_path))
    #     test_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    #     self.assertEqual(list(test_df.columns), header)
    #     comp_function(df=test_df, q_ind_map=pdb_index_mapping, q_to_s_map=pdb_query_mapping,
    #                   seq_pdb_map=self.scorer2.query_pdb_mapping,
    #                   seq=self.scorer2.query_alignment.query_sequence, scores=scores2, coverages=coverages2,
    #                   distances=self.scorer2.distances, adjacencies=A)
    #     os.remove(curr_path)
    #     self.scorer2.write_out_clustering_results(today=today, raw_scores=scores2, coverage_scores=coverages2,
    #                                               file_name=None, output_dir=save_dir)
    #     curr_path = os.path.join(save_dir, "{}_{}.Covariance_vs_Structure.txt".format(today, self.scorer2.query))
    #     self.assertTrue(os.path.isfile(curr_path))
    #     test_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    #     self.assertEqual(list(test_df.columns), header)
    #     comp_function(df=test_df, q_ind_map=pdb_index_mapping, q_to_s_map=pdb_query_mapping,
    #                   seq_pdb_map=self.scorer2.query_pdb_mapping,
    #                   seq=self.scorer2.query_alignment.query_sequence, scores=scores2, coverages=coverages2,
    #                   distances=self.scorer2.distances, adjacencies=A)
    #     os.remove(curr_path)
    # 
    # def test_evaluate_predictor(self):
    #     out_dir = os.path.abspath('../Test')
    #     today = str(datetime.date.today())
    #     aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    #                '-']
    #     aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
    #     #
    #     path1 = os.path.join(out_dir, '{}.fa'.format(self.query1))
    #     etmipc1 = ETMIPC(path1)
    #     time1 = etmipc1.calculate_scores(curr_date=today, query=self.query1, tree_depth=(2, 5),
    #                                      out_dir=out_dir, processes=1, ignore_alignment_size=True,
    #                                      clustering='agglomerative', clustering_args={'affinity': 'euclidean',
    #                                                                                   'linkage': 'ward'},
    #                                      aa_mapping=aa_dict, combine_clusters='sum', combine_branches='sum',
    #                                      del_intermediate=False, low_mem=False)
    #     print(time1)
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     score_df = None
    #     coverage_df = None
    #     for v in range(1, 6):
    #         self.scorer1.evaluate_predictor(predictor=etmipc1, verbosity=v, out_dir=out_dir, dist='Any',
    #                                         biased_w2_ave=None, unbiased_w2_ave=None, today=today)
    #         for c in etmipc1.tree_depth:
    #             c_out_dir = os.path.join(out_dir, str(c))
    #             self.assertTrue(os.path.isdir(c_out_dir))
    #             if v >= 1:
    #                 fn1 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) +
    #                                    "{}_{}.Covariance_vs_Structure.txt".format(today, self.query1))
    #                 self.assertTrue(os.path.isfile(fn1))
    #                 os.remove(fn1)
    #             if v >= 2:
    #                 fn2 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) + 'Dist-Any_Biased_ZScores.tsv')
    #                 self.assertTrue(os.path.isfile(fn2))
    #                 os.remove(fn2)
    #                 fn3 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) + 'Dist-Any_Biased_ZScores.eps')
    #                 self.assertTrue(os.path.isfile(fn3))
    #                 os.remove(fn3)
    #                 fn4 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) + 'Dist-Any_Unbiased_ZScores.tsv')
    #                 self.assertTrue(os.path.isfile(fn4))
    #                 os.remove(fn4)
    #                 fn5 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) + 'Dist-Any_Unbiased_ZScores.eps')
    #                 self.assertTrue(os.path.isfile(fn5))
    #                 os.remove(fn5)
    #             if v >= 3:
    #                 fn6 = os.path.join(out_dir, 'Score_Evaluation_Dist-Any.txt')
    #                 self.assertTrue(os.path.isfile(fn6))
    #                 score_df = pd.read_csv(fn6, delimiter='\t', header=0, index_col=False)
    #                 self.assertTrue('K' in score_df.columns)
    #                 self.assertTrue('Time' in score_df.columns)
    #                 self.assertTrue('AUROC' in score_df.columns)
    #                 self.assertTrue('Distance' in score_df.columns)
    #                 self.assertTrue('Sequence_Separation' in score_df.columns)
    #                 fn7 = os.path.join(out_dir, 'Coverage_Evaluation_Dist-Any.txt')
    #                 self.assertTrue(os.path.isfile(fn7))
    #                 coverage_df = pd.read_csv(fn7, delimiter='\t', header=0, index_col=False)
    #                 self.assertTrue('K' in coverage_df.columns)
    #                 self.assertTrue('Time' in coverage_df.columns)
    #                 self.assertTrue('AUROC' in coverage_df.columns)
    #                 self.assertTrue('Distance' in coverage_df.columns)
    #                 self.assertTrue('Sequence_Separation' in coverage_df.columns)
    #                 fn8 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) +
    #                                    'AUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('Any', 'Any'))
    #                 self.assertTrue(os.path.isfile(fn8))
    #                 os.remove(fn8)
    #                 fn9 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) +
    #                                    'AUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('Any', 'Neighbors'))
    #                 self.assertTrue(os.path.isfile(fn9))
    #                 os.remove(fn9)
    #                 fn10 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) +
    #                                     'AUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('Any', 'Short'))
    #                 self.assertTrue(os.path.isfile(fn10))
    #                 os.remove(fn10)
    #                 fn11 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) +
    #                                     'AUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('Any', 'Medium'))
    #                 self.assertTrue(os.path.isfile(fn11))
    #                 os.remove(fn11)
    #                 fn12 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) +
    #                                     'AUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('Any', 'Long'))
    #                 self.assertTrue(os.path.isfile(fn12))
    #                 os.remove(fn12)
    #             else:
    #                 self.assertTrue(score_df is None or 'K' not in score_df.columns)
    #                 self.assertTrue(score_df is None or 'Time' not in score_df.columns)
    #                 self.assertTrue(score_df is None or 'AUROC' not in score_df.columns)
    #                 self.assertTrue(score_df is None or 'Distance' not in score_df.columns)
    #                 self.assertTrue(score_df is None or 'Sequence_Separation' not in score_df.columns)
    #                 self.assertTrue(coverage_df is None or 'K' not in coverage_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Time' not in coverage_df.columns)
    #                 self.assertTrue(coverage_df is None or 'AUROC' not in coverage_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Distance' not in coverage_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Sequence_Separation' not in coverage_df.columns)
    #             if v >= 4:
    #                 self.assertTrue('Precision (L)' in score_df.columns)
    #                 self.assertTrue('Precision (L)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/2)' in score_df.columns)
    #                 self.assertTrue('Precision (L/2)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/3)' in score_df.columns)
    #                 self.assertTrue('Precision (L/3)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/4)' in score_df.columns)
    #                 self.assertTrue('Precision (L/4)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/5)' in score_df.columns)
    #                 self.assertTrue('Precision (L/5)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/6)' in score_df.columns)
    #                 self.assertTrue('Precision (L/6)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/7)' in score_df.columns)
    #                 self.assertTrue('Precision (L/7)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/8)' in score_df.columns)
    #                 self.assertTrue('Precision (L/8)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/9)' in score_df.columns)
    #                 self.assertTrue('Precision (L/9)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/10)' in score_df.columns)
    #                 self.assertTrue('Precision (L/10)' in coverage_df.columns)
    #             else:
    #                 self.assertTrue(score_df is None or 'Precision (L)' not in score_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Precision (L)' not in coverage_df.columns)
    #                 self.assertTrue(score_df is None or 'Precision (L/2)' not in score_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/2)' not in coverage_df.columns)
    #                 self.assertTrue(score_df is None or 'Precision (L/3)' not in score_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/3)' not in coverage_df.columns)
    #                 self.assertTrue(score_df is None or 'Precision (L/4)' not in score_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/4)' not in coverage_df.columns)
    #                 self.assertTrue(score_df is None or 'Precision (L/5)' not in score_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/5)' not in coverage_df.columns)
    #                 self.assertTrue(score_df is None or 'Precision (L/6)' not in score_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/6)' not in coverage_df.columns)
    #                 self.assertTrue(score_df is None or 'Precision (L/7)' not in score_df.columns or score_df is None)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/7)' not in coverage_df.columns or score_df is None)
    #                 self.assertTrue(score_df is None or 'Precision (L/8)' not in score_df.columns or score_df is None)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/8)' not in coverage_df.columns or score_df is None)
    #                 self.assertTrue(score_df is None or 'Precision (L/9)' not in score_df.columns or score_df is None)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/9)' not in coverage_df.columns or score_df is None)
    #                 self.assertTrue(score_df is None or 'Precision (L/10)' not in score_df.columns or score_df is None)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/10)' not in coverage_df.columns or score_df is None)
    #             if v == 5:
    #                 fn13 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) + 'Dist-Any_Heatmap.eps')
    #                 self.assertTrue(os.path.isfile(fn13))
    #                 os.remove(fn13)
    #                 fn14 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) + 'Dist-Any_Surface.eps')
    #                 self.assertTrue(os.path.isfile(fn14))
    #                 os.remove(fn14)
    #         if v >= 3:
    #             os.remove(fn6)
    #             os.remove(fn7)
    #     os.remove(os.path.join(out_dir, 'alignment.pkl'))
    #     os.remove(os.path.join(out_dir, 'ungapped_alignment.pkl'))
    #     os.remove(os.path.join(out_dir, 'UngappedAlignment.fa'))
    #     os.remove(os.path.join(out_dir, 'X.npz'))
    #     os.remove(os.path.join(out_dir, '1c17A_cET-MIp.npz'))
    #     os.remove(os.path.join(out_dir, '1c17A_cET-MIp.pkl'))
    #     rmtree(os.path.join(out_dir, 'joblib'))
    #     rmtree(os.path.join(out_dir, str(1)))
    #     rmtree(os.path.join(out_dir, str(2)))
    #     rmtree(os.path.join(out_dir, str(3)))
    #     rmtree(os.path.join(out_dir, str(4)))
    #     #
    #     path2 = os.path.join(out_dir, '{}.fa'.format(self.query2))
    #     etmipc2 = ETMIPC(path2)
    #     time2 = etmipc2.calculate_scores(curr_date=today, query=self.query2, tree_depth=(2, 5),
    #                                      out_dir=out_dir, processes=1, ignore_alignment_size=True,
    #                                      clustering='agglomerative', clustering_args={'affinity': 'euclidean',
    #                                                                                   'linkage': 'ward'},
    #                                      aa_mapping=aa_dict, combine_clusters='sum', combine_branches='sum',
    #                                      del_intermediate=False, low_mem=False)
    #     print(time2)
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     score_df = None
    #     coverage_df = None
    #     for v in range(1, 6):
    #         self.scorer2.evaluate_predictor(predictor=etmipc2, verbosity=v, out_dir=out_dir, dist='Any',
    #                                         biased_w2_ave=None, unbiased_w2_ave=None, today=today)
    #         for c in etmipc2.tree_depth:
    #             c_out_dir = os.path.join(out_dir, str(c))
    #             self.assertTrue(os.path.isdir(c_out_dir))
    #             if v >= 1:
    #                 fn1 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) +
    #                                    "{}_{}.Covariance_vs_Structure.txt".format(today, self.query2))
    #                 self.assertTrue(os.path.isfile(fn1))
    #                 os.remove(fn1)
    #             if v >= 2:
    #                 fn2 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) + 'Dist-Any_Biased_ZScores.tsv')
    #                 self.assertTrue(os.path.isfile(fn2))
    #                 os.remove(fn2)
    #                 fn3 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) + 'Dist-Any_Biased_ZScores.eps')
    #                 self.assertTrue(os.path.isfile(fn3))
    #                 os.remove(fn3)
    #                 fn4 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) + 'Dist-Any_Unbiased_ZScores.tsv')
    #                 self.assertTrue(os.path.isfile(fn4))
    #                 os.remove(fn4)
    #                 fn5 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) + 'Dist-Any_Unbiased_ZScores.eps')
    #                 self.assertTrue(os.path.isfile(fn5))
    #                 os.remove(fn5)
    #             if v >= 3:
    #                 fn6 = os.path.join(out_dir, 'Score_Evaluation_Dist-Any.txt')
    #                 self.assertTrue(os.path.isfile(fn6))
    #                 score_df = pd.read_csv(fn6, delimiter='\t', header=0, index_col=False)
    #                 self.assertTrue('K' in score_df.columns)
    #                 self.assertTrue('Time' in score_df.columns)
    #                 self.assertTrue('AUROC' in score_df.columns)
    #                 self.assertTrue('Distance' in score_df.columns)
    #                 self.assertTrue('Sequence_Separation' in score_df.columns)
    #                 fn7 = os.path.join(out_dir, 'Coverage_Evaluation_Dist-Any.txt')
    #                 self.assertTrue(os.path.isfile(fn7))
    #                 coverage_df = pd.read_csv(fn7, delimiter='\t', header=0, index_col=False)
    #                 self.assertTrue('K' in coverage_df.columns)
    #                 self.assertTrue('Time' in coverage_df.columns)
    #                 self.assertTrue('AUROC' in coverage_df.columns)
    #                 self.assertTrue('Distance' in coverage_df.columns)
    #                 self.assertTrue('Sequence_Separation' in coverage_df.columns)
    #                 fn8 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) +
    #                                    'AUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('Any', 'Any'))
    #                 self.assertTrue(os.path.isfile(fn8))
    #                 os.remove(fn8)
    #                 fn9 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) +
    #                                    'AUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('Any', 'Neighbors'))
    #                 self.assertTrue(os.path.isfile(fn9))
    #                 os.remove(fn9)
    #                 fn10 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) +
    #                                     'AUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('Any', 'Short'))
    #                 self.assertTrue(os.path.isfile(fn10))
    #                 os.remove(fn10)
    #                 fn11 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) +
    #                                     'AUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('Any', 'Medium'))
    #                 self.assertTrue(os.path.isfile(fn11))
    #                 os.remove(fn11)
    #                 fn12 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) +
    #                                     'AUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('Any', 'Long'))
    #                 self.assertTrue(os.path.isfile(fn12))
    #                 os.remove(fn12)
    #             else:
    #                 self.assertTrue(score_df is None or 'K' not in score_df.columns)
    #                 self.assertTrue(score_df is None or 'Time' not in score_df.columns)
    #                 self.assertTrue(score_df is None or 'AUROC' not in score_df.columns)
    #                 self.assertTrue(score_df is None or 'Distance' not in score_df.columns)
    #                 self.assertTrue(score_df is None or 'Sequence_Separation' not in score_df.columns)
    #                 self.assertTrue(coverage_df is None or 'K' not in coverage_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Time' not in coverage_df.columns)
    #                 self.assertTrue(coverage_df is None or 'AUROC' not in coverage_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Distance' not in coverage_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Sequence_Separation' not in coverage_df.columns)
    #             if v >= 4:
    #                 self.assertTrue('Precision (L)' in score_df.columns)
    #                 self.assertTrue('Precision (L)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/2)' in score_df.columns)
    #                 self.assertTrue('Precision (L/2)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/3)' in score_df.columns)
    #                 self.assertTrue('Precision (L/3)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/4)' in score_df.columns)
    #                 self.assertTrue('Precision (L/4)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/5)' in score_df.columns)
    #                 self.assertTrue('Precision (L/5)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/6)' in score_df.columns)
    #                 self.assertTrue('Precision (L/6)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/7)' in score_df.columns)
    #                 self.assertTrue('Precision (L/7)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/8)' in score_df.columns)
    #                 self.assertTrue('Precision (L/8)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/9)' in score_df.columns)
    #                 self.assertTrue('Precision (L/9)' in coverage_df.columns)
    #                 self.assertTrue('Precision (L/10)' in score_df.columns)
    #                 self.assertTrue('Precision (L/10)' in coverage_df.columns)
    #             else:
    #                 self.assertTrue(score_df is None or 'Precision (L)' not in score_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Precision (L)' not in coverage_df.columns)
    #                 self.assertTrue(score_df is None or 'Precision (L/2)' not in score_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/2)' not in coverage_df.columns)
    #                 self.assertTrue(score_df is None or 'Precision (L/3)' not in score_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/3)' not in coverage_df.columns)
    #                 self.assertTrue(score_df is None or 'Precision (L/4)' not in score_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/4)' not in coverage_df.columns)
    #                 self.assertTrue(score_df is None or 'Precision (L/5)' not in score_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/5)' not in coverage_df.columns)
    #                 self.assertTrue(score_df is None or 'Precision (L/6)' not in score_df.columns)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/6)' not in coverage_df.columns)
    #                 self.assertTrue(score_df is None or 'Precision (L/7)' not in score_df.columns or score_df is None)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/7)' not in coverage_df.columns or score_df is None)
    #                 self.assertTrue(score_df is None or 'Precision (L/8)' not in score_df.columns or score_df is None)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/8)' not in coverage_df.columns or score_df is None)
    #                 self.assertTrue(score_df is None or 'Precision (L/9)' not in score_df.columns or score_df is None)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/9)' not in coverage_df.columns or score_df is None)
    #                 self.assertTrue(score_df is None or 'Precision (L/10)' not in score_df.columns or score_df is None)
    #                 self.assertTrue(coverage_df is None or 'Precision (L/10)' not in coverage_df.columns or score_df is None)
    #             if v == 5:
    #                 fn13 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) + 'Dist-Any_Heatmap.eps')
    #                 self.assertTrue(os.path.isfile(fn13))
    #                 os.remove(fn13)
    #                 fn14 = os.path.join(c_out_dir, 'Scores_K-{}_'.format(c) + 'Dist-Any_Surface.eps')
    #                 self.assertTrue(os.path.isfile(fn14))
    #                 os.remove(fn14)
    #         if v >= 3:
    #             os.remove(fn6)
    #             os.remove(fn7)
    #     os.remove(os.path.join(out_dir, 'alignment.pkl'))
    #     os.remove(os.path.join(out_dir, 'ungapped_alignment.pkl'))
    #     os.remove(os.path.join(out_dir, 'UngappedAlignment.fa'))
    #     os.remove(os.path.join(out_dir, 'X.npz'))
    #     os.remove(os.path.join(out_dir, '1h1vA_cET-MIp.npz'))
    #     os.remove(os.path.join(out_dir, '1h1vA_cET-MIp.pkl'))
    #     rmtree(os.path.join(out_dir, 'joblib'))
    #     rmtree(os.path.join(out_dir, str(1)))
    #     rmtree(os.path.join(out_dir, str(2)))
    #     rmtree(os.path.join(out_dir, str(3)))
    #     rmtree(os.path.join(out_dir, str(4)))
    # 
    # def test_evaluate_predictions(self):
    #     out_dir = os.path.abspath('../Test')
    #     today = str(datetime.date.today())
    #     #
    #     scores1 = np.random.RandomState(1234567890).rand(79, 79)
    #     scores1[np.tril_indices(79, 1)] = 0
    #     scores1 += scores1.T
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     #
    #     prev_stats = None
    #     prev_b_w2_ave = None
    #     prev_u_w2_ave = None
    #     full_len = None
    #     for v in range(1, 6):
    #         curr_stats, curr_b_w2_ave, curr_u_w2_ave = self.scorer1.evaluate_predictions(verbosity=v, out_dir=out_dir,
    #                                                                                      scores=scores1,  dist='CB',
    #                                                                                      file_prefix='SCORER1_TEST',
    #                                                                                      stats=prev_stats,
    #                                                                                      biased_w2_ave=prev_b_w2_ave,
    #                                                                                      unbiased_w2_ave=prev_u_w2_ave,
    #                                                                                      today=today)
    #         # Tests
    #         # Check that the correct data is in the dataframe according to the verbosity
    #         if v >= 1:
    #             fn1 = os.path.join(out_dir, "SCORER1_TEST{}_{}.Covariance_vs_Structure.txt".format(today, self.query1))
    #             self.assertTrue(os.path.isfile(fn1))
    #             os.remove(fn1)
    #             if v == 1:
    #                 self.assertTrue(curr_stats == {})
    #                 self.assertTrue(curr_b_w2_ave is None)
    #                 self.assertTrue(curr_u_w2_ave is None)
    #         if v >= 2:
    #             fn2 = os.path.join(out_dir, 'SCORER1_TEST' + 'Dist-CB_Biased_ZScores.tsv')
    #             self.assertTrue(os.path.isfile(fn2))
    #             os.remove(fn2)
    #             fn3 = os.path.join(out_dir, 'SCORER1_TEST' + 'Dist-CB_Biased_ZScores.eps')
    #             self.assertTrue(os.path.isfile(fn3))
    #             os.remove(fn3)
    #             fn4 = os.path.join(out_dir, 'SCORER1_TEST' + 'Dist-CB_Unbiased_ZScores.tsv')
    #             self.assertTrue(os.path.isfile(fn4))
    #             os.remove(fn4)
    #             fn5 = os.path.join(out_dir, 'SCORER1_TEST' + 'Dist-CB_Unbiased_ZScores.eps')
    #             self.assertTrue(os.path.isfile(fn5))
    #             os.remove(fn5)
    #             if v == 2:
    #                 self.assertTrue(curr_stats == {})
    #             self.assertTrue(curr_b_w2_ave is not None)
    #             self.assertTrue(curr_u_w2_ave is not None)
    #         if v >= 3:
    #             self.assertTrue('AUROC' in curr_stats)
    #             self.assertTrue('Distance' in curr_stats)
    #             self.assertTrue('Sequence_Separation' in curr_stats)
    #             # Check that lengths are even multiples of previous runs
    #             if full_len is None and curr_stats != {}:
    #                 full_len = len(curr_stats['AUROC'])
    #             self.assertTrue(len(curr_stats['AUROC']) % full_len == 0)
    #             for key in curr_stats:
    #                 # print(curr_stats['AUROC'])
    #                 # print(curr_stats[key])
    #                 self.assertEqual(len(curr_stats[key]), len(curr_stats['AUROC']),
    #                                  '{} does not match AUROC length'.format(key))
    #             fn6 = os.path.join(out_dir, 'SCORER1_TESTAUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('CB',
    #                                                                                                         'Any'))
    #             self.assertTrue(os.path.isfile(fn6))
    #             os.remove(fn6)
    #             fn7 = os.path.join(out_dir, 'SCORER1_TESTAUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('CB',
    #                                                                                                         'Neighbors'))
    #             self.assertTrue(os.path.isfile(fn7))
    #             os.remove(fn7)
    #             fn8 = os.path.join(out_dir, 'SCORER1_TESTAUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('CB',
    #                                                                                                         'Short'))
    #             self.assertTrue(os.path.isfile(fn8))
    #             os.remove(fn8)
    #             fn9 = os.path.join(out_dir, 'SCORER1_TESTAUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('CB',
    #                                                                                                         'Medium'))
    #             self.assertTrue(os.path.isfile(fn9))
    #             os.remove(fn9)
    #             fn10 = os.path.join(out_dir, 'SCORER1_TESTAUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('CB',
    #                                                                                                         'Long'))
    #             self.assertTrue(os.path.isfile(fn10))
    #             os.remove(fn10)
    #         if v >= 4:
    #             precision_labels = ['Precision (L)', 'Precision (L/2)', 'Precision (L/3)', 'Precision (L/4)',
    #                                 'Precision (L/5)', 'Precision (L/6)', 'Precision (L/7)', 'Precision (L/8)',
    #                                 'Precision (L/9)', 'Precision (L/10)']
    #             for l in precision_labels:
    #                 self.assertTrue(l in curr_stats)
    #                 self.assertEqual(len(curr_stats[l]), len(curr_stats['AUROC']))
    #         if v == 5:
    #             fn11 = os.path.join(out_dir, 'SCORER1_TESTDist-CB_Heatmap.eps')
    #             self.assertTrue(os.path.isfile(fn11))
    #             os.remove(fn11)
    #             fn12 = os.path.join(out_dir, 'SCORER1_TESTDist-CB_Surface.eps')
    #             self.assertTrue(os.path.isfile(fn12))
    #             os.remove(fn12)
    #         # Update
    #         prev_stats = curr_stats
    #         prev_b_w2_ave = curr_b_w2_ave
    #         prev_u_w2_ave = curr_u_w2_ave
    #     #
    #     scores2 = np.random.RandomState(1234567890).rand(368, 368)
    #     scores2[np.tril_indices(368, 1)] = 0
    #     scores2 += scores2.T
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     #
    #     prev_stats = None
    #     prev_b_w2_ave = None
    #     prev_u_w2_ave = None
    #     full_len = None
    #     for v in range(1, 6):
    #         curr_stats, curr_b_w2_ave, curr_u_w2_ave = self.scorer2.evaluate_predictions(verbosity=v, out_dir=out_dir,
    #                                                                                      scores=scores2, dist='CB',
    #                                                                                      file_prefix='SCORER2_TEST',
    #                                                                                      stats=prev_stats,
    #                                                                                      biased_w2_ave=prev_b_w2_ave,
    #                                                                                      unbiased_w2_ave=prev_u_w2_ave,
    #                                                                                      today=today)
    #         # Tests
    #         # Check that the correct data is in the dataframe according to the verbosity
    #         if v >= 1:
    #             fn1 = os.path.join(out_dir, "SCORER2_TEST{}_{}.Covariance_vs_Structure.txt".format(today, self.query2))
    #             self.assertTrue(os.path.isfile(fn1))
    #             os.remove(fn1)
    #             if v == 1:
    #                 self.assertTrue(curr_stats == {})
    #                 self.assertTrue(curr_b_w2_ave is None)
    #                 self.assertTrue(curr_u_w2_ave is None)
    #         if v >= 2:
    #             fn2 = os.path.join(out_dir, 'SCORER2_TEST' + 'Dist-CB_Biased_ZScores.tsv')
    #             self.assertTrue(os.path.isfile(fn2))
    #             os.remove(fn2)
    #             fn3 = os.path.join(out_dir, 'SCORER2_TEST' + 'Dist-CB_Biased_ZScores.eps')
    #             self.assertTrue(os.path.isfile(fn3))
    #             os.remove(fn3)
    #             fn4 = os.path.join(out_dir, 'SCORER2_TEST' + 'Dist-CB_Unbiased_ZScores.tsv')
    #             self.assertTrue(os.path.isfile(fn4))
    #             os.remove(fn4)
    #             fn5 = os.path.join(out_dir, 'SCORER2_TEST' + 'Dist-CB_Unbiased_ZScores.eps')
    #             self.assertTrue(os.path.isfile(fn5))
    #             os.remove(fn5)
    #             if v == 2:
    #                 self.assertTrue(curr_stats == {})
    #             self.assertTrue(curr_b_w2_ave is not None)
    #             self.assertTrue(curr_u_w2_ave is not None)
    #         if v >= 3:
    #             self.assertTrue('AUROC' in curr_stats)
    #             self.assertTrue('Distance' in curr_stats)
    #             self.assertTrue('Sequence_Separation' in curr_stats)
    #             # Check that lengths are even multiples of previous runs
    #             if full_len is None and curr_stats != {}:
    #                 full_len = len(curr_stats['AUROC'])
    #             self.assertTrue(len(curr_stats['AUROC']) % full_len == 0)
    #             for key in curr_stats:
    #                 # print(curr_stats['AUROC'])
    #                 # print(curr_stats[key])
    #                 self.assertEqual(len(curr_stats[key]), len(curr_stats['AUROC']),
    #                                  '{} does not match AUROC length'.format(key))
    #             fn6 = os.path.join(out_dir, 'SCORER2_TESTAUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('CB',
    #                                                                                                         'Any'))
    #             self.assertTrue(os.path.isfile(fn6))
    #             os.remove(fn6)
    #             fn7 = os.path.join(out_dir, 'SCORER2_TESTAUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('CB',
    #                                                                                                         'Neighbors'))
    #             self.assertTrue(os.path.isfile(fn7))
    #             os.remove(fn7)
    #             fn8 = os.path.join(out_dir, 'SCORER2_TESTAUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('CB',
    #                                                                                                         'Short'))
    #             self.assertTrue(os.path.isfile(fn8))
    #             os.remove(fn8)
    #             fn9 = os.path.join(out_dir, 'SCORER2_TESTAUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('CB',
    #                                                                                                         'Medium'))
    #             self.assertTrue(os.path.isfile(fn9))
    #             os.remove(fn9)
    #             fn10 = os.path.join(out_dir, 'SCORER2_TESTAUROC_Evaluation_Dist-{}_Separation-{}.eps'.format('CB',
    #                                                                                                          'Long'))
    #             self.assertTrue(os.path.isfile(fn10))
    #             os.remove(fn10)
    #         if v >= 4:
    #             precision_labels = ['Precision (L)', 'Precision (L/2)', 'Precision (L/3)', 'Precision (L/4)',
    #                                 'Precision (L/5)', 'Precision (L/6)', 'Precision (L/7)', 'Precision (L/8)',
    #                                 'Precision (L/9)', 'Precision (L/10)']
    #             for l in precision_labels:
    #                 self.assertTrue(l in curr_stats)
    #                 self.assertEqual(len(curr_stats[l]), len(curr_stats['AUROC']))
    #         if v == 5:
    #             fn11 = os.path.join(out_dir, 'SCORER2_TESTDist-CB_Heatmap.eps')
    #             self.assertTrue(os.path.isfile(fn11))
    #             os.remove(fn11)
    #             fn12 = os.path.join(out_dir, 'SCORER2_TESTDist-CB_Surface.eps')
    #             self.assertTrue(os.path.isfile(fn12))
    #             os.remove(fn12)
    #         # Update
    #         prev_stats = curr_stats
    #         prev_b_w2_ave = curr_b_w2_ave
    #         prev_u_w2_ave = curr_u_w2_ave
    # 
    # def test_write_out_contact_scoring(self):
    #     def comp_function(df, seq, clusters, branches, scores, coverages):
    #         for i in df.index:
    #             # Mapping to Structure
    #             pos1 = df.loc[i, 'Pos1'] - 1
    #             pos2 = df.loc[i, 'Pos2'] - 1
    #             self.assertEqual(df.loc[i, 'AA1'], '{}'.format(one_to_three(seq[pos1])),
    #                              'Positions: {}\t{}'.format(pos1, pos2))
    #             self.assertEqual(df.loc[i, 'AA2'], '{}'.format(one_to_three(seq[pos2])),
    #                              'Positions: {}\t{}'.format(pos1, pos2))
    #             # Cluster Scores
    #             for c in clusters:
    #                 self.assertLess(np.abs(df.loc[i, 'Raw_Score_{}'.format(c + 1)] - clusters[c][pos1, pos2]), 1e-4)
    #             # Branch Scores
    #             self.assertLess(np.abs(df.loc[i, 'Integrated_Score'] - branches[pos1, pos2]), 1e-4)
    #             # Scores
    #             self.assertLess(np.abs(df.loc[i, 'Final_Score'] - scores[pos1, pos2]), 1e-3,
    #                             'Positions: {}\t{}'.format(pos1, pos2))
    #             # Coverages
    #             self.assertLess(np.abs(df.loc[i, 'Coverage_Score'] - coverages[pos1, pos2]), 1e-4,
    #                             'Positions: {}\t{}'.format(pos1, pos2))
    # 
    #     def comp_nonunique_cluster_files(df1, df2, cluster1, cluster2):
    #         index1 = 'Raw_Score_{}'.format(cluster1 + 1)
    #         index2 = 'Raw_Score_{}'.format(cluster2 + 1)
    #         col1 = np.array(df1.loc[:, index1])
    #         col2 = np.array(df2.loc[:, index2])
    #         diff = np.sum(np.abs(col1 - col2))
    #         self.assertLess(diff, 1e-10)
    # 
    #     aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    #                '-']
    #     aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
    #     out_dir = os.path.abspath('../Test/')
    #     today = str(datetime.date.today())
    #     headers = {1: ['Pos1', 'AA1', 'Pos2', 'AA2', 'Raw_Score_1', 'Integrated_Score', 'Final_Score', 'Coverage_Score'],
    #                2: ['Pos1', 'AA1', 'Pos2', 'AA2', 'Raw_Score_1', 'Raw_Score_2', 'Integrated_Score', 'Final_Score', 'Coverage_Score'],
    #                3: ['Pos1', 'AA1', 'Pos2', 'AA2', 'Raw_Score_1', 'Raw_Score_2', 'Raw_Score_3', 'Integrated_Score', 'Final_Score', 'Coverage_Score'],
    #                4: ['Pos1', 'AA1', 'Pos2', 'AA2', 'Raw_Score_1', 'Raw_Score_2', 'Raw_Score_3', 'Raw_Score_4', 'Integrated_Score', 'Final_Score', 'Coverage_Score']}
    #     #
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     path1 = os.path.join(out_dir, '1c17A.fa')
    #     etmipc1 = ETMIPC(path1)
    #     start1 = time()
    #     time1 = etmipc1.calculate_scores(curr_date=today, query='1c17A', tree_depth=(2, 5),
    #                                      out_dir=out_dir, processes=1, ignore_alignment_size=True,
    #                                      clustering='agglomerative', clustering_args={'affinity': 'euclidean',
    #                                                                                   'linkage': 'ward'},
    #                                      aa_mapping=aa_dict, combine_clusters='sum', combine_branches='sum',
    #                                      del_intermediate=False, low_mem=False)
    #     end1 = time()
    #     print(time1)
    #     print(end1 - start1)
    #     self.assertLessEqual(time1, end1 - start1)
    #     for branch1 in etmipc1.tree_depth:
    #         branch_dir = os.path.join(etmipc1.output_dir, str(branch1))
    #         self.assertTrue(os.path.isdir(branch_dir))
    #         score_path = os.path.join(branch_dir,
    #                                   "{}_{}_{}.all_scores.txt".format(today, self.scorer1.query, branch1))
    #         self.assertTrue(os.path.isfile(score_path))
    #         test_df = pd.read_csv(score_path, index_col=None, delimiter='\t')
    #         self.assertEqual(list(test_df.columns), headers[branch1])
    #         comp_function(df=test_df, seq=self.scorer1.query_alignment.query_sequence,
    #                       clusters=etmipc1.get_cluster_scores(branch=branch1),
    #                       branches=etmipc1.get_branch_scores(branch=branch1), scores=etmipc1.get_scores(branch=branch1),
    #                       coverages=etmipc1.get_coverage(branch=branch1))
    #     for curr_pos, mapped_pos in etmipc1.cluster_mapping.items():
    #         curr_path = os.path.join(etmipc1.output_dir, str(curr_pos[0]),
    #                                  "{}_{}_{}.all_scores.txt".format(today, self.scorer1.query, curr_pos[0]))
    #         mapped_path = os.path.join(etmipc1.output_dir, str(mapped_pos[0]),
    #                                    "{}_{}_{}.all_scores.txt".format(today, self.scorer1.query, mapped_pos[0]))
    #         curr_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    #         mapped_df = pd.read_csv(mapped_path, index_col=None, delimiter='\t')
    #         comp_nonunique_cluster_files(df1=curr_df, df2=mapped_df, cluster1=curr_pos[1], cluster2=mapped_pos[1])
    #     for branch1 in etmipc1.tree_depth:
    #         rmtree(os.path.join(etmipc1.output_dir, str(branch1)))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), '{}_cET-MIp.npz'.format('1c17A')))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), '{}_cET-MIp.pkl'.format('1c17A')))
    #     rmtree(os.path.join(out_dir, 'joblib'))
    #     #
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     path2 = os.path.join(out_dir, '1h1vA.fa')
    #     etmipc2 = ETMIPC(path2)
    #     start2 = time()
    #     time2 = etmipc2.calculate_scores(curr_date=today, query='1h1vA', tree_depth=(2, 5),
    #                                      out_dir=out_dir, processes=1, ignore_alignment_size=True,
    #                                      clustering='agglomerative', clustering_args={'affinity': 'euclidean',
    #                                                                                   'linkage': 'ward'},
    #                                      aa_mapping=aa_dict, combine_clusters='sum', combine_branches='sum',
    #                                      del_intermediate=False, low_mem=False)
    #     end2 = time()
    #     print(time2)
    #     print(end2 - start2)
    #     self.assertLessEqual(time2, end2 - start2)
    #     for branch2 in etmipc2.tree_depth:
    #         branch_dir = os.path.join(etmipc2.output_dir, str(branch2))
    #         self.assertTrue(os.path.isdir(branch_dir))
    #         score_path = os.path.join(branch_dir,
    #                                   "{}_{}_{}.all_scores.txt".format(today, self.scorer2.query, branch2))
    #         self.assertTrue(os.path.isfile(score_path))
    #         test_df = pd.read_csv(score_path, index_col=None, delimiter='\t')
    #         self.assertEqual(list(test_df.columns), headers[branch2])
    #         comp_function(df=test_df, seq=self.scorer2.query_alignment.query_sequence,
    #                       clusters=etmipc2.get_cluster_scores(branch=branch2),
    #                       branches=etmipc2.get_branch_scores(branch=branch2), scores=etmipc2.get_scores(branch=branch2),
    #                       coverages=etmipc2.get_coverage(branch=branch2))
    #     for curr_pos, mapped_pos in etmipc2.cluster_mapping.items():
    #         curr_path = os.path.join(etmipc2.output_dir, str(curr_pos[0]),
    #                                  "{}_{}_{}.all_scores.txt".format(today, self.scorer2.query, curr_pos[0]))
    #         mapped_path = os.path.join(etmipc2.output_dir, str(mapped_pos[0]),
    #                                    "{}_{}_{}.all_scores.txt".format(today, self.scorer2.query, mapped_pos[0]))
    #         curr_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    #         mapped_df = pd.read_csv(mapped_path, index_col=None, delimiter='\t')
    #         comp_nonunique_cluster_files(df1=curr_df, df2=mapped_df, cluster1=curr_pos[1], cluster2=mapped_pos[1])
    #     for branch2 in etmipc2.tree_depth:
    #         rmtree(os.path.join(etmipc2.output_dir, str(branch2)))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), '{}_cET-MIp.npz'.format('1h1vA')))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), '{}_cET-MIp.pkl'.format('1h1vA')))
    #     rmtree(os.path.join(out_dir, 'joblib'))
    # 
    # def test_plot_z_score(self):
    #     save_dir = os.path.abspath('../Test')
    #     #
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     scores1 = np.random.RandomState(1234567890).rand(79, 79)
    #     scores1[np.tril_indices(79, 1)] = 0
    #     scores1 += scores1.T
    #     start1 = time()
    #     curr_path1a = os.path.join(save_dir, 'z_score1.tsv')
    #     zscore_df_1, _ = self.scorer1.score_clustering_of_contact_predictions(
    #         predictions=scores1, bias=True, file_path=curr_path1a, w2_ave_sub=None)
    #     end1 = time()
    #     print('Time for ContactScorer to compute SCW: {}'.format((end1 - start1) / 60.0))
    #     self.assertTrue(os.path.isfile(curr_path1a))
    #     os.remove(curr_path1a)
    #     curr_path1b = os.path.join(save_dir, 'z_score1.eps')
    #     plot_z_scores(zscore_df_1, file_path=curr_path1b)
    #     self.assertTrue(os.path.isfile(curr_path1b))
    #     os.remove(curr_path1b)
    #     #
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     scores2 = np.random.RandomState(1234567890).rand(368, 368)
    #     scores2[np.tril_indices(368, 1)] = 0
    #     scores2 += scores2.T
    #     start2 = time()
    #     curr_path2a = os.path.join(save_dir, 'z_score2.tsv')
    #     zscore_df_2, _ = self.scorer2.score_clustering_of_contact_predictions(
    #         predictions=scores2, bias=True, file_path=curr_path2a, w2_ave_sub=None)
    #     end2 = time()
    #     print('Time for ContactScorer to compute SCW: {}'.format((end2 - start2) / 60.0))
    #     self.assertTrue(os.path.isfile(curr_path2a))
    #     os.remove(curr_path2a)
    #     curr_path2b = os.path.join(save_dir, 'z_score2.eps')
    #     plot_z_scores(zscore_df_2, file_path=curr_path2b)
    #     self.assertTrue(os.path.isfile(curr_path2b))
    #     os.remove(curr_path2b)
    # 
    # def test_heatmap_plot(self):
    #     save_dir = os.path.abspath('../Test')
    #     #
    #     scores1 = np.random.rand(79, 79)
    #     scores1[np.tril_indices(79, 1)] = 0
    #     scores1 += scores1.T
    #     heatmap_plot(name='Score 1 Heatmap Plot', data_mat=scores1, output_dir=save_dir)
    #     expected_path1 = os.path.abspath(os.path.join(save_dir, 'Score_1_Heatmap_Plot.eps'))
    #     print(expected_path1)
    #     self.assertTrue(os.path.isfile(expected_path1))
    #     os.remove(expected_path1)
    # 
    # def test_surface_plot(self):
    #     save_dir = os.path.abspath('../Test')
    #     #
    #     scores1 = np.random.rand(79, 79)
    #     scores1[np.tril_indices(79, 1)] = 0
    #     scores1 += scores1.T
    #     surface_plot(name='Score 1 Surface Plot', data_mat=scores1, output_dir=save_dir)
    #     expected_path1 = os.path.abspath(os.path.join(save_dir, 'Score_1_Surface_Plot.eps'))
    #     print(expected_path1)
    #     self.assertTrue(os.path.isfile(expected_path1))
    #     os.remove(expected_path1)
    # 
    # def test_adjacency_determination(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     scorer1_a = self.scorer1.distances < self.scorer1.cutoff
    #     scorer1_a[range(scorer1_a.shape[0]), range(scorer1_a.shape[1])] = 0
    #     recip_map = {v: k for k, v in self.scorer1.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer1.query_structure.pdb_residue_list[self.scorer1.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     A, res_atoms = self._et_computeAdjacency(self.scorer1.query_structure.structure[0][self.scorer1.best_chain],
    #                                              mapping=final_map)
    #     scorer1_a_pos = np.nonzero(scorer1_a)
    #     scorer1_in_dict = 0
    #     scorer1_not_in_dict = 0
    #     scorer1_missed_pos = ([], [])
    #     for i in range(len(scorer1_a_pos[0])):
    #         if scorer1_a_pos[0][i] < scorer1_a_pos[1][i]:
    #             try:
    #                 scorer1_in_dict += A[scorer1_a_pos[0][i]][scorer1_a_pos[1][i]]
    #             except KeyError:
    #                 try:
    #                     scorer1_in_dict += A[scorer1_a_pos[1][i]][scorer1_a_pos[0][i]]
    #                 except KeyError:
    #                     scorer1_not_in_dict += 1
    #                     scorer1_missed_pos[0].append(scorer1_a_pos[0][i])
    #                     scorer1_missed_pos[1].append(scorer1_a_pos[1][i])
    #     rhonald1_in_dict = 0
    #     rhonald1_not_in_dict = 0
    #     rhonald1_not_mapped = 0
    #     for i in A.keys():
    #         for j in A[i].keys():
    #             if A[i][j] == scorer1_a[i, j]:
    #                 rhonald1_in_dict += 1
    #             else:
    #                 rhonald1_not_in_dict += 1
    #     print('ContactScorer Check - In Dict: {} |Not In Dict: {}'.format(scorer1_in_dict, scorer1_not_in_dict))
    #     print('Rhonald Check - In Dict: {} |Not In Dict: {}'.format(rhonald1_in_dict, rhonald1_not_in_dict))
    #     print('{} residues not mapped from Rhonald to ContactScorer'.format(rhonald1_not_mapped))
    #     self.assertEqual(scorer1_in_dict, rhonald1_in_dict)
    #     self.assertEqual(scorer1_not_in_dict, 0)
    #     self.assertEqual(rhonald1_not_in_dict, 0)
    #     #
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     scorer2_a = self.scorer2.distances < self.scorer2.cutoff
    #     scorer2_a[range(scorer2_a.shape[0]), range(scorer2_a.shape[1])] = 0
    #     recip_map = {v: k for k, v in self.scorer2.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer2.query_structure.pdb_residue_list[self.scorer2.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     A, res_atoms = self._et_computeAdjacency(self.scorer2.query_structure.structure[0][self.scorer2.best_chain],
    #                                              mapping=final_map)
    #     scorer2_a_pos = np.nonzero(scorer2_a)
    #     scorer2_in_dict = 0
    #     scorer2_not_in_dict = 0
    #     scorer2_missed_pos = ([], [])
    #     for i in range(len(scorer2_a_pos[0])):
    #         if scorer2_a_pos[0][i] < scorer2_a_pos[1][i]:
    #             try:
    #                 scorer2_in_dict += A[scorer2_a_pos[0][i]][scorer2_a_pos[1][i]]
    #             except KeyError:
    #                 try:
    #                     scorer2_in_dict += A[scorer2_a_pos[1][i]][scorer2_a_pos[0][i]]
    #                 except KeyError:
    #                     scorer2_not_in_dict += 1
    #                     scorer2_missed_pos[0].append(scorer2_a_pos[0][i])
    #                     scorer2_missed_pos[1].append(scorer2_a_pos[1][i])
    #     rhonald2_in_dict = 0
    #     rhonald2_not_in_dict = 0
    #     for i in A.keys():
    #         for j in A[i].keys():
    #             if A[i][j] == scorer2_a[i, j]:
    #                 rhonald2_in_dict += 1
    #             else:
    #                 rhonald2_not_in_dict += 1
    #     print('ContactScorer Check - In Dict: {} |Not In Dict: {}'.format(scorer2_in_dict, scorer2_not_in_dict))
    #     print('Rhonald Check - In Dict: {} |Not In Dict: {}'.format(rhonald2_in_dict, rhonald2_not_in_dict))
    #     self.assertEqual(scorer2_in_dict, rhonald2_in_dict)
    #     self.assertEqual(scorer2_not_in_dict, 0)
    #     self.assertEqual(rhonald2_not_in_dict, 0)


if __name__ == '__main__':
    unittest.main()
