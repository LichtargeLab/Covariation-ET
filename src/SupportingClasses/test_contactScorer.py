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
from PDBReference import PDBReference
from utils import compute_rank_and_coverage
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
        cls.seq1 = cls.data_set.protein_data[cls.small_structure_id]['Sequence']
        cls.seq_len1 = cls.data_set.protein_data[cls.small_structure_id]['Length']
        cls.query2 = cls.large_structure_id
        cls.aln_file2 = cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln']
        cls.pdb_file2 = cls.data_set.protein_data[cls.large_structure_id]['PDB']
        cls.pdb_chain2 = cls.data_set.protein_data[cls.large_structure_id]['Chain']
        cls.seq2 = cls.data_set.protein_data[cls.large_structure_id]['Sequence']
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

    def comp_function(self, df, q_ind_map, q_to_s_map, seq_pdb_map, seq, scores, coverages, distances, adjacencies):
        for i in df.index:
            # Mapping to Structure
            if not isinstance(df.loc[i, 'Pos1'], int) and not isinstance(df.loc[i, 'Pos2'], int):
                continue
            pos1 = q_to_s_map[q_ind_map[df.loc[i, 'Pos1']]]
            pos2 = q_to_s_map[q_ind_map[df.loc[i, 'Pos2']]]
            self.assertEqual(df.loc[i, '(AA1)'], '({})'.format(one_to_three(seq[seq_pdb_map[pos1]])),
                             'Positions: {}\t{}'.format(pos1, pos2))
            self.assertEqual(df.loc[i, '(AA2)'], '({})'.format(one_to_three(seq[seq_pdb_map[pos2]])),
                             'Positions: {}\t{}'.format(pos1, pos2))
            # Scores
            self.assertLess(np.abs(df.loc[i, 'Raw_Score'] - scores[pos1, pos2]), 1e-3,
                            'Positions: {}\t{}'.format(pos1, pos2))
            # Coverages
            self.assertLess(np.abs(df.loc[i, 'Coverage_Score'] - coverages[pos1, pos2]), 1e-4,
                            'Positions: {}\t{}'.format(pos1, pos2))
            # Distances
            self.assertLess(np.abs(df.loc[i, 'Residue_Dist'] - distances[pos1, pos2]), 1e-4,
                            'Positions: {}\t{}'.format(pos1, pos2))
            # Contacts
            if df.loc[i, 'Within_Threshold'] == 1:
                try:
                    self.assertEqual(df.loc[i, 'Within_Threshold'], adjacencies[pos1][pos2],
                                     'Positions: {}\t{}'.format(pos1, pos2))
                except KeyError:
                    self.assertEqual(df.loc[i, 'Within_Threshold'], adjacencies[pos2][pos1],
                                     'Positions: {}\t{}'.format(pos1, pos2))
            else:
                with self.assertRaises(KeyError):
                    adjacencies[pos2][pos1]

    def score_comp_function(self, df, seq, clusters, branches, scores, coverages):
        for i in df.index:
            # Mapping to Structure
            pos1 = df.loc[i, 'Pos1'] - 1
            pos2 = df.loc[i, 'Pos2'] - 1
            self.assertEqual(df.loc[i, 'AA1'], '{}'.format(one_to_three(seq[pos1])),
                             'Positions: {}\t{}'.format(pos1, pos2))
            self.assertEqual(df.loc[i, 'AA2'], '{}'.format(one_to_three(seq[pos2])),
                             'Positions: {}\t{}'.format(pos1, pos2))
            # Cluster Scores
            for c in clusters:
                self.assertLess(np.abs(df.loc[i, 'Raw_Score_{}'.format(c + 1)] - clusters[c][pos1, pos2]), 1e-4)
            # Branch Scores
            self.assertLess(np.abs(df.loc[i, 'Integrated_Score'] - branches[pos1, pos2]), 1e-4)
            # Scores
            self.assertLess(np.abs(df.loc[i, 'Final_Score'] - scores[pos1, pos2]), 1e-3,
                            'Positions: {}\t{}'.format(pos1, pos2))
            # Coverages
            self.assertLess(np.abs(df.loc[i, 'Coverage_Score'] - coverages[pos1, pos2]), 1e-4,
                            'Positions: {}\t{}'.format(pos1, pos2))

    def score_comp_nonunique_cluster_files(self, df1, df2, cluster1, cluster2):
        index1 = 'Raw_Score_{}'.format(cluster1 + 1)
        index2 = 'Raw_Score_{}'.format(cluster2 + 1)
        col1 = np.array(df1.loc[:, index1])
        col2 = np.array(df2.loc[:, index2])
        diff = np.sum(np.abs(col1 - col2))
        self.assertLess(diff, 1e-10)

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
        cases = {'Case1': 0, 'Case2': 0, 'Case3': 0}
        if bias == 1:
            for resi, neighborsj in A.items():
                for resj in neighborsj:
                    w_ave += (resj - resi)
                    for resk, neighborsl in A.items():
                        for resl in neighborsl:
                            if (resi == resk and resj == resl) or \
                                    (resi == resl and resj == resk):
                                w2_ave += pi1 * (resj - resi) * (resl - resk)
                                cases['Case1'] += (resj - resi) * (resl - resk)
                            elif (resi == resk) or (resj == resl) or \
                                    (resi == resl) or (resj == resk):
                                w2_ave += pi2 * (resj - resi) * (resl - resk)
                                cases['Case2'] += (resj - resi) * (resl - resk)
                            else:
                                w2_ave += pi3 * (resj - resi) * (resl - resk)
                                cases['Case3'] += (resj - resi) * (resl - resk)
        elif bias == 0:
            for resi, neighborsj in A.items():
                w_ave += len(neighborsj)
                for resj in neighborsj:
                    for resk, neighborsl in A.items():
                        for resl in neighborsl:
                            if (resi == resk and resj == resl) or \
                                    (resi == resl and resj == resk):
                                w2_ave += pi1
                                cases['Case1'] += 1
                            elif (resi == resk) or (resj == resl) or \
                                    (resi == resl) or (resj == resk):
                                w2_ave += pi2
                                cases['Case2'] += 1
                            else:
                                w2_ave += pi3
                                cases['Case3'] += 1
        w_ave = w_ave * pi1
        # print('EXPECTED M: ', M)
        # print('EXPECTED L: ', L)
        # print('EXPECTED W: ', w)
        # print('EXPECTED RES LIST: ', sorted(reslist))
        # print('EXPECTED W_AVE: ', w_ave)
        # print('EXPECTED W_AVE^2: ', (w_ave * w_ave))
        # print('EXPECTED W^2_AVE: ', w2_ave)
        # print('EXPECTED DIFF: ', w2_ave - w_ave * w_ave)
        # print('EXPECTED DIFF2: ', w2_ave - (w_ave * w_ave))
        sigma = math.sqrt(w2_ave - w_ave * w_ave)
        if sigma == 0:
            return M, L, pi1, pi2, pi3, 'NA', w, w_ave, w2_ave, sigma, cases
        return M, L, pi1, pi2, pi3, (w - w_ave) / sigma, w, w_ave, w2_ave, sigma, cases

    def all_z_scores(self, mapping, special_mapping, A, L, bias, res_i, res_j, scores):
        data = {'Res_i': res_i, 'Res_j': res_j, 'Covariance_Score': scores, 'Z-Score': [], 'W': [], 'W_Ave': [],
                'W2_Ave': [], 'Sigma': [], 'Num_Residues': []}
        res_list = []
        res_set = set()
        prev_size = 0
        prev_score = None
        for i in range(len(scores)):
            # curr_i = res_i[i] + 1
            curr_i = res_i[i]
            # curr_j = res_j[i] + 1
            curr_j = res_j[i]
            if (curr_i not in A) or (curr_j not in A):
                score_data = (None, None, None, None, None, None, '-', None, None, None, None, None)
            else:
                if curr_i not in res_set:
                    res_list.append(curr_i)
                    res_set.add(curr_i)
                if curr_j not in res_set:
                    res_list.append(curr_j)
                    res_set.add(curr_j)
                if len(res_set) == prev_size:
                    score_data = prev_score
                else:
                    # score_data = self._et_calcZScore(reslist=[special_mapping[res] for res in res_list], L=L, A=A, bias=bias)
                    score_data = self._et_calcZScore(reslist=res_list, L=L, A=A, bias=bias)
            data['Z-Score'].append(score_data[5])
            data['W'].append(score_data[6])
            data['W_Ave'].append(score_data[7])
            data['W2_Ave'].append(score_data[8])
            data['Sigma'].append(score_data[9])
            data['Num_Residues'].append(len(res_list))
            prev_size = len(res_set)
            prev_score = score_data
        return pd.DataFrame(data)

    # def test_init(self):
    #     aln = SeqAlignment(query_id=self.query1, file_name=self.aln_file1)
    #     scorer = ContactScorer(query=self.query1, seq_alignment=aln, pdb_reference=self.pdb_file1,
    #                            cutoff=8.0, chain=self.pdb_chain1)
    #
    # def evaluate_init(self, scorer, expected_aln, expected_structure, expected_cutoff, expected_chain):
    #     self.assertEqual(scorer.query_alignment, expected_aln)
    #     self.assertEqual(scorer.query_structure, expected_structure)
    #     self.assertEqual(scorer.cutoff, expected_cutoff)
    #     if expected_chain:
    #         self.assertEqual(scorer.best_chain, expected_chain)
    #     else:
    #         self.assertIsNone(scorer.best_chain)
    #     self.assertIsNone(scorer.query_pdb_mapping)
    #     self.assertIsNone(scorer._specific_mapping)
    #     self.assertIsNone(scorer.distances)
    #     self.assertIsNone(scorer.dist_type)
    #     self.assertIsNone(scorer.data)
    #
    # def test_1a___init(self):
    #     with self.assertRaises(TypeError):
    #         ContactScorer()
    #     self.evaluate_init(scorer=self.scorer1, expected_aln=os.path.abspath(self.aln_file1),
    #                        expected_structure=os.path.abspath(self.pdb_file1), expected_cutoff=8.0,
    #                        expected_chain=self.pdb_chain1)
    #
    # def test_1b___init(self):
    #     with self.assertRaises(TypeError):
    #         ContactScorer()
    #     self.evaluate_init(scorer=self.scorer2, expected_aln=os.path.abspath(self.aln_file2),
    #                        expected_structure=os.path.abspath(self.pdb_file2), expected_cutoff=8.0,
    #                        expected_chain=self.pdb_chain2)
    #
    # def test_1c___init(self):
    #     with self.assertRaises(TypeError):
    #         ContactScorer()
    #     eval1 = ContactScorer(query=self.query1, seq_alignment=self.aln_file1, pdb_reference=self.pdb_file1, cutoff=8.0)
    #     self.evaluate_init(scorer=eval1, expected_aln=os.path.abspath(self.aln_file1),
    #                        expected_structure=os.path.abspath(self.pdb_file1), expected_cutoff=8.0,
    #                        expected_chain=None)
    #
    # def test_1d___init(self):
    #     with self.assertRaises(TypeError):
    #         ContactScorer()
    #     eval2 = ContactScorer(query=self.query2, seq_alignment=self.aln_file2, pdb_reference=self.pdb_file2, cutoff=8.0)
    #     self.evaluate_init(scorer=eval2, expected_aln=os.path.abspath(self.aln_file2),
    #                        expected_structure=os.path.abspath(self.pdb_file2), expected_cutoff=8.0,
    #                        expected_chain=None)
    #
    # def evaluate_str(self, scorer, seq_len, chain_count, expected_chain):
    #     with self.assertRaises(ValueError):
    #         str(scorer)
    #     scorer.fit()
    #     expected_str = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
    #         seq_len, chain_count, expected_chain)
    #     self.assertEqual(str(scorer), expected_str)
    #
    # def test_2a___str(self):
    #     self.evaluate_str(self.scorer1, self.seq_len1, 1, self.pdb_chain1)
    #
    # def test_2b___str(self):
    #     self.evaluate_str(self.scorer2, self.seq_len2, 1, self.pdb_chain2)
    #
    # def test_2c___str(self):
    #     eval1 = ContactScorer(query=self.query1, seq_alignment=self.aln_file1, pdb_reference=self.pdb_file1, cutoff=8.0)
    #     self.evaluate_str(eval1, self.seq_len1, 1, self.pdb_chain1)
    #
    # def test_2d___str(self):
    #     eval2 = ContactScorer(query=self.query2, seq_alignment=self.aln_file2, pdb_reference=self.pdb_file2, cutoff=8.0)
    #     self.evaluate_str(eval2, self.seq_len2, 1, self.pdb_chain2)
    #
    # def evaluate_fit(self, scorer, expected_aln_fn, expected_struct_fn, expected_chain, expected_mapping, expected_seq,
    #                  expected_struct):
    #     self.assertEqual(scorer.query_alignment, expected_aln_fn)
    #     self.assertEqual(scorer.query_structure, expected_struct_fn)
    #     scorer.fit()
    #     self.assertNotEqual(scorer.query_alignment, expected_aln_fn)
    #     self.assertNotEqual(scorer.query_structure, expected_struct_fn)
    #     self.assertEqual(scorer.best_chain, expected_chain)
    #     self.assertEqual(scorer.query_pdb_mapping, expected_mapping)
    #     self.assertIsNotNone(scorer.data)
    #     scorer.best_chain = None
    #     scorer.fit()
    #     self.assertNotEqual(scorer.query_alignment, expected_aln_fn)
    #     self.assertNotEqual(scorer.query_structure, expected_struct_fn)
    #     self.assertEqual(scorer.best_chain, expected_chain)
    #     self.assertEqual(scorer.query_pdb_mapping, expected_mapping)
    #     self.assertIsNotNone(scorer.data)
    #     scorer.query_pdb_mapping = None
    #     scorer.fit()
    #     self.assertNotEqual(scorer.query_alignment, expected_aln_fn)
    #     self.assertNotEqual(scorer.query_structure, expected_struct_fn)
    #     self.assertEqual(scorer.best_chain, expected_chain)
    #     self.assertEqual(scorer.query_pdb_mapping, expected_mapping)
    #     self.assertIsNotNone(scorer.data)
    #     for i in scorer.data.index:
    #         self.assertEqual(scorer.data.loc[i, 'Seq AA 1'], expected_seq.seq[scorer.data.loc[i, 'Seq Pos 1']])
    #         self.assertEqual(scorer.data.loc[i, 'Seq AA 2'], expected_seq.seq[scorer.data.loc[i, 'Seq Pos 2']])
    #         self.assertEqual(scorer.data.loc[i, 'Seq Separation'],
    #                          scorer.data.loc[i, 'Seq Pos 2'] - scorer.data.loc[i, 'Seq Pos 1'])
    #         if scorer.data.loc[i, 'Seq Separation'] < 6:
    #             self.assertEqual(scorer.data.loc[i, 'Seq Separation Category'], 'Neighbors')
    #         elif scorer.data.loc[i, 'Seq Separation'] < 12:
    #             self.assertEqual(scorer.data.loc[i, 'Seq Separation Category'], 'Short')
    #         elif scorer.data.loc[i, 'Seq Separation'] < 24:
    #             self.assertEqual(scorer.data.loc[i, 'Seq Separation Category'], 'Medium')
    #         else:
    #             self.assertEqual(scorer.data.loc[i, 'Seq Separation Category'], 'Long')
    #         if scorer.data.loc[i, 'Struct Pos 1'] == '-':
    #             self.assertFalse(scorer.data.loc[i, 'Seq AA 1'] in scorer.query_pdb_mapping)
    #             self.assertEqual(scorer.data.loc[i, 'Struct AA 1'], '-')
    #         else:
    #             self.assertEqual(scorer.data.loc[i, 'Struct Pos 1'],
    #                              expected_struct.pdb_residue_list[expected_chain][scorer.query_pdb_mapping[scorer.data.loc[i, 'Seq Pos 1']]])
    #             self.assertEqual(scorer.data.loc[i, 'Struct AA 1'],
    #                              expected_struct.seq[expected_chain][scorer.query_pdb_mapping[scorer.data.loc[i, 'Seq Pos 1']]])
    #         if scorer.data.loc[i, 'Struct Pos 2'] == '-':
    #             self.assertFalse(scorer.data.loc[i, 'Seq AA 2'] in scorer.query_pdb_mapping)
    #             self.assertEqual(scorer.data.loc[i, 'Struct AA 2'], '-')
    #         else:
    #             self.assertEqual(scorer.data.loc[i, 'Struct Pos 2'],
    #                              expected_struct.pdb_residue_list[expected_chain][scorer.query_pdb_mapping[scorer.data.loc[i, 'Seq Pos 2']]])
    #             self.assertEqual(scorer.data.loc[i, 'Struct AA 2'],
    #                              expected_struct.seq[expected_chain][scorer.query_pdb_mapping[scorer.data.loc[i, 'Seq Pos 2']]])
    #
    # def test_3a_fit(self):
    #     struct1 = PDBReference(pdb_file=self.pdb_file1)
    #     struct1.import_pdb(structure_id=self.query1)
    #     self.evaluate_fit(self.scorer1, os.path.abspath(self.aln_file1), os.path.abspath(self.pdb_file1), 'A',
    #                       {i + 18: i for i in range(len(struct1.seq['A']))}, self.seq1, struct1)
    #
    # def test_3b_fit(self):
    #     struct2 = PDBReference(pdb_file=self.pdb_file2)
    #     struct2.import_pdb(structure_id=self.query2)
    #     self.evaluate_fit(self.scorer2, os.path.abspath(self.aln_file2), os.path.abspath(self.pdb_file2), 'A',
    #                       {i + 16: i for i in range(len(struct2.seq['A']))}, self.seq2, struct2)
    #
    # def test_3c_fit(self):
    #     struct1 = PDBReference(pdb_file=self.pdb_file1)
    #     struct1.import_pdb(structure_id=self.query1)
    #     eval1 = ContactScorer(query=self.query1, seq_alignment=self.aln_file1, pdb_reference=self.pdb_file1, cutoff=8.0)
    #     self.evaluate_fit(eval1, os.path.abspath(self.aln_file1), os.path.abspath(self.pdb_file1), 'A',
    #                       {i + 18: i for i in range(len(struct1.seq['A']))}, self.seq1, struct1)
    #
    # def test_3d_fit(self):
    #     struct2 = PDBReference(pdb_file=self.pdb_file2)
    #     struct2.import_pdb(structure_id=self.query2)
    #     eval2 = ContactScorer(query=self.query2, seq_alignment=self.aln_file2, pdb_reference=self.pdb_file2, cutoff=8.0)
    #     self.evaluate_fit(eval2, os.path.abspath(self.aln_file2), os.path.abspath(self.pdb_file2), 'A',
    #                       {i + 16: i for i in range(len(struct2.seq['A']))}, self.seq2, struct2)
    #
    # def evaluate_get_coords(self, method, residue, expected_coordinates, options={}):
    #     measured_coordinates = np.vstack(method(residue, **options))
    #     diff = measured_coordinates - expected_coordinates
    #     not_passing = diff > 1E-5
    #     self.assertFalse(not_passing.any())
    #
    # def test_4a__get_all_coords(self):
    #     self.scorer1.fit()
    #     residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
    #     expected1 = np.vstack([[24.704, 20.926, 27.944], [25.408, 20.195, 26.922], [24.487, 19.147, 26.324],
    #                            [23.542, 18.689, 26.993], [26.589, 19.508, 27.519], [26.344, 18.392, 28.442],
    #                            [27.689, 17.685, 28.514], [27.941, 16.866, 27.267], [29.154, 16.092, 27.419]])
    #     self.evaluate_get_coords(method=ContactScorer._get_all_coords, residue=residue1, expected_coordinates=expected1)
    #
    # def test_4b__get_all_coords(self):
    #     self.scorer2.fit()
    #     residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][1]
    #     expected2 = np.vstack([[26.432, 44.935, 26.052], [25.921, 43.597, 25.862], [25.159, 43.203, 24.568],
    #                            [23.936, 43.424, 24.593], [25.050, 43.281, 27.093], [25.777, 43.092, 28.306]])
    #     self.evaluate_get_coords(method=ContactScorer._get_all_coords, residue=residue2, expected_coordinates=expected2)
    #
    # def test_4c__get_c_alpha_coords(self):
    #     self.scorer1.fit()
    #     residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
    #     expected1 = np.vstack([[25.408, 20.195, 26.922]])
    #     self.evaluate_get_coords(method=ContactScorer._get_c_alpha_coords, residue=residue1,
    #                              expected_coordinates=expected1)
    #
    # def test_4d__get_c_alpha_coords(self):
    #     self.scorer2.fit()
    #     residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][1]
    #     expected2 = np.vstack([[25.921, 43.597, 25.862]])
    #     self.evaluate_get_coords(method=ContactScorer._get_c_alpha_coords, residue=residue2,
    #                              expected_coordinates=expected2)
    #
    # def test_4e__get_c_beta_coords(self):
    #     self.scorer1.fit()
    #     residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
    #     expected1 = np.vstack([[26.589, 19.508, 27.519]])
    #     self.evaluate_get_coords(method=ContactScorer._get_c_beta_coords, residue=residue1,
    #                              expected_coordinates=expected1)
    #
    # def test_4f__get_c_beta_coords(self):
    #     self.scorer2.fit()
    #     residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][1]
    #     expected2 = np.vstack([[25.050, 43.281, 27.093]])
    #     self.evaluate_get_coords(method=ContactScorer._get_c_beta_coords, residue=residue2,
    #                              expected_coordinates=expected2)
    #
    # def test_4g__get_coords(self):
    #     self.scorer1.fit()
    #     residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
    #     expected1a = np.vstack([[24.704, 20.926, 27.944], [25.408, 20.195, 26.922], [24.487, 19.147, 26.324],
    #                            [23.542, 18.689, 26.993], [26.589, 19.508, 27.519], [26.344, 18.392, 28.442],
    #                            [27.689, 17.685, 28.514], [27.941, 16.866, 27.267], [29.154, 16.092, 27.419]])
    #     self.evaluate_get_coords(method=ContactScorer._get_coords, residue=residue1,
    #                              expected_coordinates=expected1a, options={'method': 'Any'})
    #     expected1b = np.vstack([[25.408, 20.195, 26.922]])
    #     self.evaluate_get_coords(method=ContactScorer._get_coords, residue=residue1,
    #                              expected_coordinates=expected1b, options={'method': 'CA'})
    #     expected1c = np.vstack([[26.589, 19.508, 27.519]])
    #     self.evaluate_get_coords(method=ContactScorer._get_coords, residue=residue1,
    #                              expected_coordinates=expected1c, options={'method': 'CB'})
    #
    # def test_4h__get_coords(self):
    #     self.scorer2.fit()
    #     residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][1]
    #     expected2a = np.vstack([[26.432, 44.935, 26.052], [25.921, 43.597, 25.862], [25.159, 43.203, 24.568],
    #                            [23.936, 43.424, 24.593], [25.050, 43.281, 27.093], [25.777, 43.092, 28.306]])
    #     self.evaluate_get_coords(method=ContactScorer._get_coords, residue=residue2,
    #                              expected_coordinates=expected2a, options={'method': 'Any'})
    #     expected2b = np.vstack([[25.921, 43.597, 25.862]])
    #     self.evaluate_get_coords(method=ContactScorer._get_coords, residue=residue2,
    #                              expected_coordinates=expected2b, options={'method': 'CA'})
    #     expected2c = np.vstack([[25.050, 43.281, 27.093]])
    #     self.evaluate_get_coords(method=ContactScorer._get_coords, residue=residue2,
    #                              expected_coordinates=expected2c, options={'method': 'CB'})
    #
    # def evaulate_measure_distance(self, scorer):
    #     scorer.fit()
    #     scorer.measure_distance(method='Any')
    #     self.assertEqual(scorer.dist_type, 'Any')
    #     residue_coords = {}
    #     size1 = len(scorer.query_structure.seq[scorer.best_chain])
    #     dists = np.zeros((size1, size1))
    #     dists2 = np.zeros((size1, size1))
    #     counter = 0
    #     counter_map = {}
    #     for res_num in scorer.query_structure.residue_pos[scorer.best_chain]:
    #         counter_map[counter] = res_num
    #         residue = scorer.query_structure.structure[0][scorer.best_chain][res_num]
    #         coords = scorer._get_all_coords(residue)
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
    #                 self.assertLess(np.abs(dist - np.square(
    #                     scorer.data.loc[(scorer.data['Struct Pos 1'] == counter_map[residue2]) &
    #                                     (scorer.data['Struct Pos 2'] == res_num), 'Distance'].values[0])), 1E-3)
    #                 self.assertLess(np.abs(dist2 -
    #                                        scorer.data.loc[(scorer.data['Struct Pos 1'] == counter_map[residue2]) &
    #                                                        (scorer.data['Struct Pos 2'] == res_num),
    #                                                        'Distance'].values[0]), 1E-5)
    #         counter += 1
    #     distance_diff = np.square(scorer.distances) - dists
    #     self.assertLess(np.max(distance_diff), 1e-3)
    #     adj_diff = ((np.square(scorer.distances)[np.nonzero(distance_diff)] < self.CONTACT_DISTANCE2) ^
    #                 (dists[np.nonzero(distance_diff)] < self.CONTACT_DISTANCE2))
    #     self.assertEqual(np.sum(adj_diff), 0)
    #     self.assertEqual(len(np.nonzero(adj_diff)[0]), 0)
    #     distance_diff2 = scorer.distances - dists2
    #     self.assertEqual(np.sum(distance_diff2), 0.0)
    #     self.assertEqual(len(np.nonzero(distance_diff2)[0]), 0.0)
    #     residue1a = scorer.query_structure.structure[0][scorer.best_chain][1]
    #     residue1b = scorer.query_structure.structure[0][scorer.best_chain][2]
    #     pos1a = ContactScorer._get_all_coords(residue1a)
    #     pos1b = ContactScorer._get_all_coords(residue1b)
    #     expected1a = None
    #     for i in range(len(pos1a)):
    #         for j in range(len(pos1b)):
    #             curr_dist = np.sqrt(np.power(pos1a[i][0] - pos1b[j][0], 2) + np.power(pos1a[i][1] - pos1b[j][1], 2) +
    #                                 np.power(pos1a[i][2] - pos1b[j][2], 2))
    #             if (expected1a is None) or (curr_dist < expected1a):
    #                 expected1a = curr_dist
    #     self.assertLess(expected1a - scorer.distances[0, 1], 1E-5)
    #     scorer.measure_distance(method='CA')
    #     self.assertEqual(scorer.dist_type, 'CA')
    #     ca_atom1a = residue1a['CA'].get_coord()
    #     ca_atom1b = residue1b['CA'].get_coord()
    #     expected1b = np.sqrt(np.power(ca_atom1a[0] - ca_atom1b[0], 2) + np.power(ca_atom1a[1] - ca_atom1b[1], 2) +
    #                          np.power(ca_atom1a[2] - ca_atom1b[2], 2))
    #     self.assertLess(expected1b - scorer.distances[0, 1], 1E-5)
    #     scorer.measure_distance(method='CB')
    #     self.assertEqual(scorer.dist_type, 'CB')
    #     cb_atom1a = residue1a['CB'].get_coord()
    #     cb_atom1b = residue1b['CB'].get_coord()
    #     expected1c = np.sqrt(np.power(cb_atom1a[0] - cb_atom1b[0], 2) + np.power(cb_atom1a[1] - cb_atom1b[1], 2) +
    #                          np.power(cb_atom1a[2] - cb_atom1b[2], 2))
    #     self.assertLess(expected1c - scorer.distances[0, 1], 1E-5)
    #
    # def test_5a_measure_distance(self):
    #     self.evaulate_measure_distance(scorer=self.scorer1)
    #
    # def test_5b_measure_distance(self):
    #     self.evaulate_measure_distance(scorer=self.scorer2)
    #
    # def evaluate_find_pairs_by_separation(self, scorer, seq_len):
    #     scorer.fit()
    #     with self.assertRaises(ValueError):
    #         scorer.find_pairs_by_separation(category='Wide')
    #     expected1 = {'Any': [], 'Neighbors': [], 'Short': [], 'Medium': [], 'Long': []}
    #     for i in range(seq_len):
    #         for j in range(i + 1, seq_len):
    #             pair = (i, j)
    #             separation = j - i
    #             if (separation >= 1) and (separation < 6):
    #                 expected1['Neighbors'].append(pair)
    #             if (separation >= 6) and (separation < 12):
    #                 expected1['Short'].append(pair)
    #             if (separation >= 12) and (separation < 24):
    #                 expected1['Medium'].append(pair)
    #             if separation >= 24:
    #                 expected1['Long'].append(pair)
    #             expected1['Any'].append(pair)
    #     self.assertEqual(scorer.find_pairs_by_separation(category='Any'), expected1['Any'])
    #     self.assertEqual(scorer.find_pairs_by_separation(category='Neighbors'), expected1['Neighbors'])
    #     self.assertEqual(scorer.find_pairs_by_separation(category='Short'), expected1['Short'])
    #     self.assertEqual(scorer.find_pairs_by_separation(category='Medium'), expected1['Medium'])
    #     self.assertEqual(scorer.find_pairs_by_separation(category='Long'), expected1['Long'])
    #
    # def test_6a_find_pairs_by_separation(self):
    #     self.evaluate_find_pairs_by_separation(self.scorer1, self.seq_len1)
    #
    # def test_6b_find_pairs_by_separation(self):
    #     self.evaluate_find_pairs_by_separation(self.scorer2, self.seq_len2)
    #
    # def evaluate_map_prediction_to_pdb(self, scorer, seq_len):
    #     scorer.fit()
    #     scorer.measure_distance(method='CB')
    #     scores = np.random.rand(seq_len, seq_len)
    #     scores[np.tril_indices(seq_len, 1)] = 0
    #     scores += scores.T
    #     ranks, coverages = compute_rank_and_coverage(seq_len, scores, 2, 'min')
    #     scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
    #     self.assertIsNotNone(scorer.data)
    #     self.assertTrue(pd.Series(['Rank', 'Score', 'Coverage', 'True Prediction']).isin(scorer.data.columns).all())
    #     for i in scorer.data.index:
    #         pos1 = scorer.data.loc[i, 'Seq Pos 1']
    #         pos2 = scorer.data.loc[i, 'Seq Pos 2']
    #         self.assertEqual(ranks[pos1, pos2], scorer.data.loc[i, 'Rank'])
    #         self.assertEqual(scores[pos1, pos2], scorer.data.loc[i, 'Score'])
    #         self.assertEqual(coverages[pos1, pos2], scorer.data.loc[i, 'Coverage'])
    #         if coverages[pos1, pos2] <= 0.5:
    #             self.assertEqual(scorer.data.loc[i, 'True Prediction'], 1)
    #         else:
    #             self.assertEqual(scorer.data.loc[i, 'True Prediction'], 0)
    #
    # def test_7a_map_prediction_to_pdb(self):
    #     self.evaluate_map_prediction_to_pdb(scorer=self.scorer1, seq_len=self.seq_len1)
    #
    # def test_7b_map_prediction_to_pdb(self):
    #     self.evaluate_map_prediction_to_pdb(scorer=self.scorer2, seq_len=self.seq_len2)

    @staticmethod
    def identify_expected_scores_and_distances(scorer, scores, coverages, ranks, distances, category='Any', n=None,
                                               k=None, cutoff=8.0, threshold=0.5):
        seq_sep_ind = scorer.find_pairs_by_separation(category=category, mappable_only=True)
        converted_ind = list(zip(*seq_sep_ind))
        dist_ind = [(scorer.query_pdb_mapping[x[0]], scorer.query_pdb_mapping[x[1]]) for x in seq_sep_ind]
        converted_dist_ind = list(zip(*dist_ind))
        if n and k:
            raise ValueError('Both n and k cannot be defined when identifying data for testing.')
        elif n is None and k is None:
            n = len(converted_ind[0])
        elif k is not None:
            n = int(floor(scorer.query_alignment.seq_length / float(k)))
        else:
            pass
        scores_subset = scores[converted_ind]
        coverage_subset = coverages[converted_ind]
        ranks_subset = ranks[converted_ind]
        preds_subset = coverage_subset <= threshold
        distance_subset = distances[converted_dist_ind]
        contact_subset = distance_subset <= cutoff
        df = pd.DataFrame({'Seq Pos 1': converted_ind[0], 'Seq Pos 2': converted_ind[1],
                           'Struct Pos 1': [scorer.query_structure.pdb_residue_list[scorer.best_chain][x]
                                            for x in converted_dist_ind[0]],
                           'Struct Pos 2': [scorer.query_structure.pdb_residue_list[scorer.best_chain][x]
                                            for x in converted_dist_ind[1]],
                           'Score': scores_subset, 'Coverage': coverage_subset, 'Rank': ranks_subset,
                           'Predictions': preds_subset, 'Distance': distance_subset, 'Contact': contact_subset})
        df_sorted = df.sort_values(by='Coverage')
        df_sorted['Top Predictions'] = rankdata(df_sorted['Coverage'], method='dense')
        n_index = df_sorted['Top Predictions'] <= n
        df_final = df_sorted.loc[n_index, :]
        return df_final

    # def evaluate__identify_relevant_data(self, scorer, seq_len):
    #     scorer.fit()
    #     scorer.measure_distance(method='CB')
    #     scores = np.random.rand(seq_len, seq_len)
    #     scores[np.tril_indices(self.seq_len1, 1)] = 0
    #     scores += scores.T
    #     ranks, coverages = compute_rank_and_coverage(seq_len, scores, 2, 'min')
    #     scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
    #     with self.assertRaises(ValueError):
    #         scorer._identify_relevant_data(category='Any', n=10, k=10)
    #     for category in ['Any', 'Neighbors', 'Short', 'Medium', 'Long']:
    #         print('Category: {}'.format(category))
    #         for n, k in [(None, None), (10, None), (100, None), (None, 1), (None, 2), (None, 3), (None, 4), (None, 5),
    #                      (None, 6), (None, 7), (None, 8), (None, 9), (None, 10)]:
    #             print('Subset: n:{} k:{}'.format(n, k))
    #             curr_subset = scorer._identify_relevant_data(category=category, n=n, k=k)
    #             expected_subset = TestContactScorer.identify_expected_scores_and_distances(
    #                 scorer, scores, coverages, ranks, scorer.distances, category, n=n, k=k)
    #             seq_1_pos_diff = np.abs(curr_subset['Seq Pos 1'].values - expected_subset['Seq Pos 1'].values)
    #             seq_1_pos_not_passing = seq_1_pos_diff > 0
    #             self.assertFalse(seq_1_pos_not_passing.any())
    #             seq_2_pos_diff = np.abs(curr_subset['Seq Pos 2'].values - expected_subset['Seq Pos 2'].values)
    #             seq_2_pos_not_passing = seq_2_pos_diff > 0
    #             self.assertFalse(seq_2_pos_not_passing.any())
    #             struct_1_pos_diff = np.abs(curr_subset['Struct Pos 1'].values - expected_subset['Struct Pos 1'].values)
    #             struct_1_not_passing = struct_1_pos_diff > 0
    #             self.assertFalse(struct_1_not_passing.any())
    #             struct_2_pos_diff = np.abs(curr_subset['Struct Pos 2'].values - expected_subset['Struct Pos 2'].values)
    #             struct_2_not_passing = struct_2_pos_diff > 0
    #             self.assertFalse(struct_2_not_passing.any())
    #             if k and (n is None):
    #                 n = int(floor(scorer.query_alignment.seq_length / float(k)))
    #             if n:
    #                 self.assertEqual(len(curr_subset['Rank'].unique()), n)
    #                 self.assertEqual(len(expected_subset['Rank'].unique()), n)
    #                 self.assertEqual(len(curr_subset['Score'].unique()), n)
    #                 self.assertEqual(len(expected_subset['Score'].unique()), n)
    #                 self.assertEqual(len(curr_subset['Coverage'].unique()), n)
    #                 self.assertEqual(len(expected_subset['Coverage'].unique()), n)
    #             else:
    #                 self.assertEqual(len(curr_subset['Rank'].unique()), len(expected_subset['Rank'].unique()))
    #                 self.assertEqual(len(curr_subset['Score'].unique()), len(expected_subset['Score'].unique()))
    #                 self.assertEqual(len(curr_subset['Coverage'].unique()), len(expected_subset['Coverage'].unique()))
    #             self.assertEqual(len(curr_subset['Distance'].unique()), len(expected_subset['Distance'].unique()))
    #             self.assertEqual(len(curr_subset['Contact (within {}A cutoff)'.format(scorer.cutoff)].unique()),
    #                              len(expected_subset['Contact'].unique()))
    #             self.assertEqual(len(curr_subset['True Prediction'].unique()),
    #                              len(expected_subset['Predictions'].unique()))
    #             diff_ranks = np.abs(curr_subset['Rank'].values - expected_subset['Rank'].values)
    #             not_passing_ranks = diff_ranks > 1E-12
    #             self.assertFalse(not_passing_ranks.any())
    #             diff_scores = np.abs(curr_subset['Score'].values - expected_subset['Score'].values)
    #             not_passing_scores = diff_scores > 1E-12
    #             self.assertFalse(not_passing_scores.any())
    #             diff_coverages = np.abs(curr_subset['Coverage'].values - expected_subset['Coverage'].values)
    #             not_passing_coverages = diff_coverages > 1E-12
    #             self.assertFalse(not_passing_coverages.any())
    #             diff_preds = curr_subset['True Prediction'].values ^ expected_subset['Predictions'].values
    #             not_passing_preds = diff_preds > 1E-12
    #             self.assertFalse(not_passing_preds.any())
    #             diff_contacts = (curr_subset['Contact (within {}A cutoff)'.format(scorer.cutoff)].values ^
    #                              expected_subset['Contact'].values)
    #             not_passing_contacts = diff_contacts > 1E-12
    #             self.assertFalse(not_passing_contacts.any())
    #             diff_distances = np.abs(curr_subset['Distance'].values - expected_subset['Distance'].values)
    #             not_passing_distances = diff_distances > 1E-12
    #             self.assertFalse(not_passing_distances.any())
    #
    # def test_8a__identify_relevant_data(self):
    #     self.evaluate__identify_relevant_data(scorer=self.scorer1, seq_len=self.seq_len1)
    #
    # def test_8b__identify_relevant_data(self):
    #     self.evaluate__identify_relevant_data(scorer=self.scorer2, seq_len=self.seq_len2)

    # def evaluate_score_auc(self, scorer, seq_len):
    #     scorer.fit()
    #     scorer.measure_distance(method='CB')
    #     scores = np.random.rand(seq_len, seq_len)
    #     scores[np.tril_indices(seq_len, 1)] = 0
    #     scores += scores.T
    #     ranks, coverages = compute_rank_and_coverage(seq_len, scores, 2, 'min')
    #     scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
    #     expected_df_a = TestContactScorer.identify_expected_scores_and_distances(
    #         scorer=scorer, scores=scores, coverages=coverages, ranks=ranks, distances=scorer.distances, category='Any')
    #     fpr_expected1a, tpr_expected1a, _ = roc_curve(expected_df_a['Distance'] <= 8.0,
    #                                                   1.0 - expected_df_a['Coverage'], pos_label=True)
    #     auroc_expected1a = auc(fpr_expected1a, tpr_expected1a)
    #     tpr1a, fpr1a, auroc1a = scorer.score_auc(category='Any')
    #     self.assertEqual(np.sum(fpr_expected1a - fpr1a), 0)
    #     self.assertEqual(np.sum(tpr_expected1a - tpr1a), 0)
    #     self.assertEqual(auroc_expected1a, auroc1a)
    #     expected_df_b = TestContactScorer.identify_expected_scores_and_distances(
    #         scorer=scorer, scores=scores, coverages=coverages, ranks=ranks, distances=scorer.distances,
    #         category='Neighbors')
    #     fpr_expected1b, tpr_expected1b, _ = roc_curve(expected_df_b['Distance'] <= 8.0,
    #                                                   1.0 - expected_df_b['Coverage'], pos_label=True)
    #     auroc_expected1b = auc(fpr_expected1b, tpr_expected1b)
    #     tpr1b, fpr1b, auroc1b = scorer.score_auc(category='Neighbors')
    #     self.assertEqual(np.sum(fpr_expected1b - fpr1b), 0)
    #     self.assertEqual(np.sum(tpr_expected1b - tpr1b), 0)
    #     self.assertEqual(auroc_expected1b, auroc1b)
    #     expected_df_c = TestContactScorer.identify_expected_scores_and_distances(
    #         scorer=scorer, scores=scores, coverages=coverages, ranks=ranks, distances=scorer.distances,
    #         category='Short')
    #     fpr_expected1c, tpr_expected1c, _ = roc_curve(expected_df_c['Distance'] <= 8.0,
    #                                                   1.0 - expected_df_c['Coverage'], pos_label=True)
    #     auroc_expected1c = auc(fpr_expected1c, tpr_expected1c)
    #     tpr1c, fpr1c, auroc1c = scorer.score_auc(category='Short')
    #     self.assertEqual(np.sum(fpr_expected1c - fpr1c), 0)
    #     self.assertEqual(np.sum(tpr_expected1c - tpr1c), 0)
    #     self.assertEqual(auroc_expected1c, auroc1c)
    #     expected_df_d = TestContactScorer.identify_expected_scores_and_distances(
    #         scorer=scorer, scores=scores, coverages=coverages, ranks=ranks, distances=scorer.distances,
    #         category='Medium')
    #     fpr_expected1d, tpr_expected1d, _ = roc_curve(expected_df_d['Distance'] <= 8.0,
    #                                                   1.0 - expected_df_d['Coverage'], pos_label=True)
    #     auroc_expected1d = auc(fpr_expected1d, tpr_expected1d)
    #     tpr1d, fpr1d, auroc1d = scorer.score_auc(category='Medium')
    #     self.assertEqual(np.sum(fpr_expected1d - fpr1d), 0)
    #     self.assertEqual(np.sum(tpr_expected1d - tpr1d), 0)
    #     self.assertEqual(auroc_expected1d, auroc1d)
    #     expected_df_e = TestContactScorer.identify_expected_scores_and_distances(
    #         scorer=scorer, scores=scores, coverages=coverages, ranks=ranks, distances=scorer.distances, category='Long')
    #     fpr_expected1e, tpr_expected1e, _ = roc_curve(expected_df_e['Distance'] <= 8.0,
    #                                                   1.0 - expected_df_e['Coverage'], pos_label=True)
    #     auroc_expected1e = auc(fpr_expected1e, tpr_expected1e)
    #     tpr1e, fpr1e, auroc1e = scorer.score_auc(category='Long')
    #     self.assertEqual(np.sum(fpr_expected1e - fpr1e), 0)
    #     self.assertEqual(np.sum(tpr_expected1e - tpr1e), 0)
    #     self.assertEqual(auroc_expected1e, auroc1e)
    #
    # def test_9a_score_auc(self):
    #     self.evaluate_score_auc(self.scorer1, self.seq_len1)
    #
    # def test_9b_score_auc(self):
    #     self.evaluate_score_auc(self.scorer2, self.seq_len2)

    # def evaluate_plot_auc(self, scorer, seq_len, structure_id, dir):
    #     scorer.fit()
    #     scorer.measure_distance(method='CB')
    #     scores = np.random.rand(seq_len, seq_len)
    #     scores[np.tril_indices(seq_len, 1)] = 0
    #     scores += scores.T
    #     ranks, coverages = compute_rank_and_coverage(seq_len, scores, 2, 'min')
    #     scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
    #     auroc1a = scorer.score_auc(category='Any')
    #     scorer.plot_auc(auc_data=auroc1a, title='{} AUROC for All Pairs'.format(structure_id),
    #                     file_name='{}_Any_AUROC'.format(structure_id), output_dir=dir)
    #     expected_path1 = os.path.abspath(os.path.join(dir, '{}_Any_AUROC.png'.format(structure_id)))
    #     self.assertTrue(os.path.isfile(expected_path1))
    #     os.remove(expected_path1)
    #
    # def test_10a_plot_auc(self):
    #     self.evaluate_plot_auc(scorer=self.scorer1, seq_len=self.seq_len1, structure_id=self.small_structure_id,
    #                            dir=self.testing_dir)
    #
    # def test_10b_plot_auc(self):
    #     self.evaluate_plot_auc(scorer=self.scorer2, seq_len=self.seq_len2, structure_id=self.large_structure_id,
    #                            dir=self.testing_dir)

    # def evaluate_score_precision_recall(self, scorer, seq_len):
    #     scorer.fit()
    #     scorer.measure_distance(method='CB')
    #     scores = np.random.rand(seq_len, seq_len)
    #     scores[np.tril_indices(seq_len, 1)] = 0
    #     scores += scores.T
    #     ranks, coverages = compute_rank_and_coverage(seq_len, scores, 2, 'min')
    #     scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
    #     expected_df_a = TestContactScorer.identify_expected_scores_and_distances(
    #         scorer=scorer, scores=scores, coverages=coverages, ranks=ranks, distances=scorer.distances, category='Any')
    #     precision_expected1a, recall_expected1a, _ = precision_recall_curve(expected_df_a['Distance'] <= 8.0,
    #                                                                         1.0 - expected_df_a['Coverage'],
    #                                                                         pos_label=True)
    #     recall_expected1a, precision_expected1a = zip(*sorted(zip(recall_expected1a, precision_expected1a)))
    #     recall_expected1a, precision_expected1a = np.array(recall_expected1a), np.array(precision_expected1a)
    #     auprc_expected1a = auc(recall_expected1a, precision_expected1a)
    #     precision1a, recall1a, auprc1a = scorer.score_precision_recall(category='Any')
    #     self.assertEqual(np.sum(precision_expected1a - precision1a), 0)
    #     self.assertEqual(np.sum(recall_expected1a - recall1a), 0)
    #     self.assertEqual(auprc_expected1a, auprc1a)
    #     expected_df_b = TestContactScorer.identify_expected_scores_and_distances(
    #         scorer=scorer, scores=scores, coverages=coverages, ranks=ranks, distances=scorer.distances,
    #         category='Neighbors')
    #     precision_expected1b, recall_expected1b, _ = precision_recall_curve(expected_df_b['Distance'] <= 8.0,
    #                                                                         1.0 - expected_df_b['Coverage'],
    #                                                                         pos_label=True)
    #     recall_expected1b, precision_expected1b = zip(*sorted(zip(recall_expected1b, precision_expected1b)))
    #     recall_expected1b, precision_expected1b = np.array(recall_expected1b), np.array(precision_expected1b)
    #     auprc_expected1b = auc(recall_expected1b, precision_expected1b)
    #     precision1b, recall1b, auprc1b = scorer.score_precision_recall(category='Neighbors')
    #     self.assertEqual(np.sum(precision_expected1b - precision1b), 0)
    #     self.assertEqual(np.sum(recall_expected1b - recall1b), 0)
    #     self.assertEqual(auprc_expected1b, auprc1b)
    #     expected_df_c = TestContactScorer.identify_expected_scores_and_distances(
    #         scorer=scorer, scores=scores, coverages=coverages, ranks=ranks, distances=scorer.distances,
    #         category='Short')
    #     precision_expected1c, recall_expected1c, _ = precision_recall_curve(expected_df_c['Distance'] <= 8.0,
    #                                                                         1.0 - expected_df_c['Coverage'],
    #                                                                         pos_label=True)
    #     recall_expected1c, precision_expected1c = zip(*sorted(zip(recall_expected1c, precision_expected1c)))
    #     recall_expected1c, precision_expected1c = np.array(recall_expected1c), np.array(precision_expected1c)
    #     auprc_expected1c = auc(recall_expected1c, precision_expected1c)
    #     precision1c, recall1c, auprc1c = scorer.score_precision_recall(category='Short')
    #     self.assertEqual(np.sum(precision_expected1c - precision1c), 0)
    #     self.assertEqual(np.sum(recall_expected1c - recall1c), 0)
    #     self.assertEqual(auprc_expected1c, auprc1c)
    #     expected_df_d = TestContactScorer.identify_expected_scores_and_distances(
    #         scorer=scorer, scores=scores, coverages=coverages, ranks=ranks, distances=scorer.distances,
    #         category='Medium')
    #     precision_expected1d, recall_expected1d, _ = precision_recall_curve(expected_df_d['Distance'] <= 8.0,
    #                                                                         1.0 - expected_df_d['Coverage'],
    #                                                                         pos_label=True)
    #     recall_expected1d, precision_expected1d = zip(*sorted(zip(recall_expected1d, precision_expected1d)))
    #     recall_expected1d, precision_expected1d = np.array(recall_expected1d), np.array(precision_expected1d)
    #     auprc_expected1d = auc(recall_expected1d, precision_expected1d)
    #     precision1d, recall1d, auprc1d = scorer.score_precision_recall(category='Medium')
    #     self.assertEqual(np.sum(precision_expected1d - precision1d), 0)
    #     self.assertEqual(np.sum(recall_expected1d - recall1d), 0)
    #     self.assertEqual(auprc_expected1d, auprc1d)
    #     expected_df_e = TestContactScorer.identify_expected_scores_and_distances(
    #         scorer=scorer, scores=scores, coverages=coverages, ranks=ranks, distances=scorer.distances, category='Long')
    #     precision_expected1e, recall_expected1e, _ = precision_recall_curve(expected_df_e['Distance'] <= 8.0,
    #                                                                         1.0 - expected_df_e['Coverage'],
    #                                                                         pos_label=True)
    #     recall_expected1e, precision_expected1e = zip(*sorted(zip(recall_expected1e, precision_expected1e)))
    #     recall_expected1e, precision_expected1e = np.array(recall_expected1e), np.array(precision_expected1e)
    #     auprc_expected1e = auc(recall_expected1e, precision_expected1e)
    #     precision1e, recall1e, auprc1e = scorer.score_precision_recall(category='Long')
    #     self.assertEqual(np.sum(precision_expected1e - precision1e), 0)
    #     self.assertEqual(np.sum(recall_expected1e - recall1e), 0)
    #     self.assertEqual(auprc_expected1e, auprc1e)
    #
    # def test_11a_score_precision_recall(self):
    #     self.evaluate_score_precision_recall(scorer=self.scorer1, seq_len=self.seq_len1)
    #
    # def test_11b_score_precision_recall(self):
    #     self.evaluate_score_precision_recall(scorer=self.scorer2, seq_len=self.seq_len2)
    #
    # def evaluate_plot_auprc(self, scorer, seq_len, structure_id, dir):
    #     scorer.fit()
    #     scorer.measure_distance(method='CB')
    #     scores = np.random.rand(seq_len, seq_len)
    #     scores[np.tril_indices(seq_len, 1)] = 0
    #     scores += scores.T
    #     ranks, coverages = compute_rank_and_coverage(seq_len, scores, 2, 'min')
    #     scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
    #     auprc1a = scorer.score_precision_recall(category='Any')
    #     scorer.plot_auprc(auprc_data=auprc1a, title='{} AUPRC for All Pairs'.format(structure_id),
    #                       file_name='{}_Any_AUPRC'.format(structure_id), output_dir=dir)
    #     expected_path1 = os.path.abspath(os.path.join(dir, '{}_Any_AUPRC.png'.format(structure_id)))
    #     self.assertTrue(os.path.isfile(expected_path1))
    #     os.remove(expected_path1)
    #
    # def test_12a_plot_auprc(self):
    #     self.evaluate_plot_auprc(scorer=self.scorer1, seq_len=self.seq_len1, structure_id=self.small_structure_id,
    #                              dir=self.testing_dir)
    #
    # def test_12b_plot_auprc(self):
    #     self.evaluate_plot_auprc(scorer=self.scorer2, seq_len=self.seq_len2, structure_id=self.large_structure_id,
    #                              dir=self.testing_dir)

    # def evaluate_score_precision(self, scorer, seq_len):
    #     scorer.fit()
    #     scorer.measure_distance(method='CB')
    #     scores = np.random.rand(seq_len, seq_len)
    #     scores[np.tril_indices(self.seq_len1, 1)] = 0
    #     scores += scores.T
    #     ranks, coverages = compute_rank_and_coverage(seq_len, scores, 2, 'min')
    #     scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
    #     with self.assertRaises(ValueError):
    #         scorer.score_precision(category='Any', n=10, k=10)
    #     for category in ['Any', 'Neighbors', 'Short', 'Medium', 'Long']:
    #         print('Category: {}'.format(category))
    #         for n, k in [(None, None), (10, None), (100, None), (None, 1), (None, 2), (None, 3), (None, 4), (None, 5),
    #                      (None, 6), (None, 7), (None, 8), (None, 9), (None, 10)]:
    #             print('N: {}, K: {}'.format(n, k))
    #             precision = scorer.score_precision(category=category, n=n, k=k)
    #             expected_df = TestContactScorer.identify_expected_scores_and_distances(
    #                 scorer=scorer, scores=scores, coverages=coverages, ranks=ranks, distances=scorer.distances,
    #                 category=category, n=n, k=k)
    #             expected_precision = precision_score(expected_df['Distance'] <= scorer.cutoff,
    #                                                  expected_df['Coverage'] <= 0.5, pos_label=True)
    #             self.assertEqual(precision, expected_precision)
    #
    # def test_13a_score_precision(self):
    #     self.evaluate_score_precision(scorer=self.scorer1, seq_len=self.seq_len1)
    #
    # def test_13b_score_precision(self):
    #     self.evaluate_score_precision(scorer=self.scorer2, seq_len=self.seq_len2)

    # def evaluate_score_recall(self, scorer, seq_len):
    #     scorer.fit()
    #     scorer.measure_distance(method='CB')
    #     scores = np.random.rand(seq_len, seq_len)
    #     scores[np.tril_indices(self.seq_len1, 1)] = 0
    #     scores += scores.T
    #     ranks, coverages = compute_rank_and_coverage(seq_len, scores, 2, 'min')
    #     scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
    #     with self.assertRaises(ValueError):
    #         scorer.score_recall(category='Any', n=10, k=10)
    #     for category in ['Any', 'Neighbors', 'Short', 'Medium', 'Long']:
    #         print('Category: {}'.format(category))
    #         for n, k in [(None, None), (10, None), (100, None), (None, 1), (None, 2), (None, 3), (None, 4), (None, 5),
    #                      (None, 6), (None, 7), (None, 8), (None, 9), (None, 10)]:
    #             print('N: {}, K: {}'.format(n, k))
    #             precision = scorer.score_precision(category=category, n=n, k=k)
    #             expected_df = TestContactScorer.identify_expected_scores_and_distances(
    #                 scorer=scorer, scores=scores, coverages=coverages, ranks=ranks, distances=scorer.distances,
    #                 category=category, n=n, k=k)
    #             expected_precision = precision_score(expected_df['Distance'] <= scorer.cutoff,
    #                                                  expected_df['Coverage'] <= 0.5, pos_label=True)
    #             self.assertEqual(precision, expected_precision)
    #
    # def test_14a_score_precision(self):
    #     self.evaluate_score_recall(scorer=self.scorer1, seq_len=self.seq_len1)
    #
    # def test_14b_score_precision(self):
    #     self.evaluate_score_recall(scorer=self.scorer2, seq_len=self.seq_len2)

    def evaluate_score_f1(self, scorer, seq_len):
        scorer.fit()
        scorer.measure_distance(method='CB')
        scores = np.random.rand(seq_len, seq_len)
        scores[np.tril_indices(self.seq_len1, 1)] = 0
        scores += scores.T
        ranks, coverages = compute_rank_and_coverage(seq_len, scores, 2, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        with self.assertRaises(ValueError):
            scorer.score_recall(category='Any', n=10, k=10)
        for category in ['Any', 'Neighbors', 'Short', 'Medium', 'Long']:
            print('Category: {}'.format(category))
            for n, k in [(None, None), (10, None), (100, None), (None, 1), (None, 2), (None, 3), (None, 4), (None, 5),
                         (None, 6), (None, 7), (None, 8), (None, 9), (None, 10)]:
                print('N: {}, K: {}'.format(n, k))
                f1 = scorer.score_f1(category=category, n=n, k=k)
                expected_df = TestContactScorer.identify_expected_scores_and_distances(
                    scorer=scorer, scores=scores, coverages=coverages, ranks=ranks, distances=scorer.distances,
                    category=category, n=n, k=k)
                expected_f1 = f1_score(expected_df['Distance'] <= scorer.cutoff, expected_df['Coverage'] <= 0.5,
                                       pos_label=True)
                self.assertEqual(f1, expected_f1)

    def test_15a_score_f1(self):
        self.evaluate_score_f1(scorer=self.scorer1, seq_len=self.seq_len1)

    def test_15b_score_f1(self):
        self.evaluate_score_f1(scorer=self.scorer2, seq_len=self.seq_len2)

    ####################################################################################################################

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
    #     ################################################################################################################
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
    #     ################################################################################################################
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
    #
    # def test_18a_clustering_z_scores(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     recip_map = {v: k for k, v in self.scorer1.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer1.query_structure.pdb_residue_list[self.scorer1.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     expected_adjacency, res_atoms = self._et_computeAdjacency(
    #         self.scorer1.query_structure.structure[0][self.scorer1.best_chain], mapping=final_map)
    #     residue_list = list(self.scorer1.query_pdb_mapping.keys())
    #     shuffle(residue_list)
    #     ################################################################################################################
    #     init_compute_w2_ave_sub(dists=self.scorer1.distances, bias_bool=False)
    #     cases_unbiased = {}
    #     for i in range(self.scorer1.distances.shape[0]):
    #         curr_cases = compute_w2_ave_sub(i)
    #         for k in curr_cases:
    #             if k not in cases_unbiased:
    #                 cases_unbiased[k] = 0
    #             cases_unbiased[k] += curr_cases[k]
    #     init_clustering_z_score(bias_bool=False, w2_ave_sub_dict=cases_unbiased, curr_pdb=self.scorer1.query_structure,
    #                             map_to_structure=self.scorer1.query_pdb_mapping, residue_dists=self.scorer1.distances,
    #                             best_chain=self.scorer1.best_chain)
    #     for i in range(len(residue_list)):
    #         curr_residues = residue_list[:(i + 1)]
    #         a, m, l, pi1, pi2, pi3, z_score, w, w_ave, w2_ave, sigma, num_residues = clustering_z_score(curr_residues)
    #         em, el, epi1, epi2, epi3, e_z_score, e_w, e_w_ave, e_w2_ave, e_sigma, e_cases = self._et_calcZScore(
    #             reslist=curr_residues, L=len(self.scorer1.query_structure.seq[self.scorer1.best_chain]),
    #             A=expected_adjacency, bias=False)
    #         for res_i in expected_adjacency:
    #             for res_j in expected_adjacency[res_i]:
    #                 self.assertEqual(a[self.scorer1.query_pdb_mapping[res_i], self.scorer1.query_pdb_mapping[res_j]],
    #                                  expected_adjacency[res_i][res_j])
    #                 a[self.scorer1.query_pdb_mapping[res_i], self.scorer1.query_pdb_mapping[res_j]] = 0
    #         self.assertEqual(m, em)
    #         self.assertEqual(l, el)
    #         self.assertLess(np.abs(pi1 - epi1), 1E-16)
    #         self.assertLess(np.abs(pi2 - epi2), 1E-16)
    #         self.assertLess(np.abs(pi3 - epi3), 1E-16)
    #         self.assertEqual(num_residues, len(curr_residues))
    #         self.assertLess(np.abs(w - e_w), 1E-16, '{} vs {}'.format(w, e_w))
    #         self.assertLess(np.abs(w_ave - e_w_ave), 1E-16, '{} vs {}'.format(w_ave, e_w_ave))
    #         for case in e_cases:
    #             self.assertEqual(cases_unbiased[case], e_cases[case])
    #         self.assertLess(np.abs(w2_ave - e_w2_ave), 1E-5, '{} vs {}'.format(w2_ave, e_w2_ave))
    #         composed_w2_ave = ((pi1 * cases_unbiased['Case1']) + (pi2 * cases_unbiased['Case2']) +
    #                            (pi3 * cases_unbiased['Case3']))
    #         expected_composed_w2_ave = ((epi1 * e_cases['Case1']) + (epi2 * e_cases['Case2']) +
    #                                     (epi3 * e_cases['Case3']))
    #         self.assertLess(np.abs(composed_w2_ave - expected_composed_w2_ave), 1E-16)
    #         self.assertLess(np.abs(sigma - e_sigma), 1E-6, '{} vs {}'.format(sigma, e_sigma))
    #         expected_composed_sigma = math.sqrt(expected_composed_w2_ave - e_w_ave * e_w_ave)
    #         self.assertLess(np.abs(sigma - expected_composed_sigma), 1E-16)
    #         if isinstance(z_score, str):
    #             self.assertTrue(isinstance(e_z_score, str))
    #             self.assertEqual(z_score, e_z_score, '{} vs {}'.format(z_score, e_z_score))
    #         else:
    #             if z_score < 0:
    #                 self.assertTrue(e_z_score < 0)
    #             else:
    #                 self.assertFalse(e_z_score < 0)
    #             self.assertLess(np.abs(z_score - e_z_score), 1E-6, '{} vs {}'.format(z_score, e_z_score))
    #             expected_composed_z_score = (e_w - e_w_ave) / expected_composed_sigma
    #             self.assertLess(np.abs(z_score - expected_composed_z_score), 1E-16)
    #     ################################################################################################################
    #     init_compute_w2_ave_sub(dists=self.scorer1.distances, bias_bool=True)
    #     cases_biased = {}
    #     for i in range(self.scorer1.distances.shape[0]):
    #         curr_cases = compute_w2_ave_sub(i)
    #         for k in curr_cases:
    #             if k not in cases_biased:
    #                 cases_biased[k] = 0
    #             cases_biased[k] += curr_cases[k]
    #     init_clustering_z_score(bias_bool=True, w2_ave_sub_dict=cases_biased, curr_pdb=self.scorer1.query_structure,
    #                             map_to_structure=self.scorer1.query_pdb_mapping, residue_dists=self.scorer1.distances,
    #                             best_chain=self.scorer1.best_chain)
    #     for i in range(len(residue_list)):
    #         curr_residues = residue_list[:(i + 1)]
    #         a, m, l, pi1, pi2, pi3, z_score, w, w_ave, w2_ave, sigma, num_residues = clustering_z_score(curr_residues)
    #         em, el, epi1, epi2, epi3, e_z_score, e_w, e_w_ave, e_w2_ave, e_sigma, e_cases = self._et_calcZScore(
    #             reslist=curr_residues, L=len(self.scorer1.query_structure.seq[self.scorer1.best_chain]),
    #             A=expected_adjacency, bias=True)
    #         for res_i in expected_adjacency:
    #             for res_j in expected_adjacency[res_i]:
    #                 self.assertEqual(a[self.scorer1.query_pdb_mapping[res_i], self.scorer1.query_pdb_mapping[res_j]],
    #                                  expected_adjacency[res_i][res_j])
    #                 a[self.scorer1.query_pdb_mapping[res_i], self.scorer1.query_pdb_mapping[res_j]] = 0
    #         self.assertEqual(m, em)
    #         self.assertEqual(l, el)
    #         self.assertLess(np.abs(pi1 - epi1), 1E-16)
    #         self.assertLess(np.abs(pi2 - epi2), 1E-16)
    #         self.assertLess(np.abs(pi3 - epi3), 1E-16)
    #         self.assertEqual(num_residues, len(curr_residues))
    #         self.assertLess(np.abs(w - e_w), 1E-16, '{} vs {}'.format(w, e_w))
    #         self.assertLess(np.abs(w_ave - e_w_ave), 1E-16, '{} vs {}'.format(w_ave, e_w_ave))
    #         for case in e_cases:
    #             self.assertEqual(cases_biased[case], e_cases[case])
    #         self.assertLess(np.abs(w2_ave - e_w2_ave), 1E-3, '{} vs {}'.format(w2_ave, e_w2_ave))
    #         composed_w2_ave = ((pi1 * cases_biased['Case1']) + (pi2 * cases_biased['Case2']) +
    #                            (pi3 * cases_biased['Case3']))
    #         expected_composed_w2_ave = ((epi1 * e_cases['Case1']) + (epi2 * e_cases['Case2']) +
    #                                     (epi3 * e_cases['Case3']))
    #         self.assertLess(np.abs(composed_w2_ave - expected_composed_w2_ave), 1E-16)
    #         self.assertLess(np.abs(sigma - e_sigma), 1E-4, '{} vs {}'.format(sigma, e_sigma))
    #         expected_composed_sigma = math.sqrt(expected_composed_w2_ave - e_w_ave * e_w_ave)
    #         self.assertLess(np.abs(sigma - expected_composed_sigma), 1E-16)
    #         if isinstance(z_score, str):
    #             self.assertTrue(isinstance(e_z_score, str))
    #             self.assertEqual(z_score, e_z_score, '{} vs {}'.format(z_score, e_z_score))
    #         else:
    #             if z_score < 0:
    #                 self.assertTrue(e_z_score < 0)
    #             else:
    #                 self.assertFalse(e_z_score < 0)
    #             self.assertLess(np.abs(z_score - e_z_score), 1E-4, '{} vs {}'.format(z_score, e_z_score))
    #             expected_composed_z_score = (e_w - e_w_ave) / expected_composed_sigma
    #             self.assertLess(np.abs(z_score - expected_composed_z_score), 1E-16)
    #
    # def test_18b_clustering_z_scores(self):
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     recip_map = {v: k for k, v in self.scorer2.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer2.query_structure.pdb_residue_list[self.scorer2.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     expected_adjacency, res_atoms = self._et_computeAdjacency(
    #         self.scorer2.query_structure.structure[0][self.scorer2.best_chain], mapping=final_map)
    #     residue_list = list(self.scorer2.query_pdb_mapping.keys())
    #     shuffle(residue_list)
    #     ################################################################################################################
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
    #                             best_chain=self.scorer2.best_chain)
    #     for i in range(len(residue_list)):
    #         curr_residues = residue_list[:(i + 1)]
    #         a, m, l, pi1, pi2, pi3, z_score, w, w_ave, w2_ave, sigma, num_residues = clustering_z_score(curr_residues)
    #         em, el, epi1, epi2, epi3, e_z_score, e_w, e_w_ave, e_w2_ave, e_sigma, e_cases = self._et_calcZScore(
    #             reslist=curr_residues, L=len(self.scorer2.query_structure.seq[self.scorer2.best_chain]),
    #             A=expected_adjacency, bias=False)
    #         for res_i in expected_adjacency:
    #             for res_j in expected_adjacency[res_i]:
    #                 self.assertEqual(a[self.scorer2.query_pdb_mapping[res_i], self.scorer2.query_pdb_mapping[res_j]],
    #                                  expected_adjacency[res_i][res_j])
    #                 a[self.scorer2.query_pdb_mapping[res_i], self.scorer2.query_pdb_mapping[res_j]] = 0
    #         self.assertEqual(m, em)
    #         self.assertEqual(l, el)
    #         self.assertLess(np.abs(pi1 - epi1), 1E-16)
    #         self.assertLess(np.abs(pi2 - epi2), 1E-16)
    #         self.assertLess(np.abs(pi3 - epi3), 1E-16)
    #         self.assertEqual(num_residues, len(curr_residues))
    #         self.assertLess(np.abs(w - e_w), 1E-16, '{} vs {}'.format(w, e_w))
    #         self.assertLess(np.abs(w_ave - e_w_ave), 1E-16, '{} vs {}'.format(w_ave, e_w_ave))
    #         for case in e_cases:
    #             self.assertEqual(cases_unbiased[case], e_cases[case])
    #         self.assertLess(np.abs(w2_ave - e_w2_ave), 1E-4, '{} vs {}'.format(w2_ave, e_w2_ave))
    #         composed_w2_ave = ((pi1 * cases_unbiased['Case1']) + (pi2 * cases_unbiased['Case2']) +
    #                            (pi3 * cases_unbiased['Case3']))
    #         expected_composed_w2_ave = ((epi1 * e_cases['Case1']) + (epi2 * e_cases['Case2']) +
    #                                     (epi3 * e_cases['Case3']))
    #         self.assertLess(np.abs(composed_w2_ave - expected_composed_w2_ave), 1E-16)
    #         self.assertLess(np.abs(sigma - e_sigma), 1E-5, '{} vs {}'.format(sigma, e_sigma))
    #         expected_composed_sigma = math.sqrt(expected_composed_w2_ave - e_w_ave * e_w_ave)
    #         self.assertLess(np.abs(sigma - expected_composed_sigma), 1E-16)
    #         if isinstance(z_score, str):
    #             self.assertTrue(isinstance(e_z_score, str))
    #             self.assertEqual(z_score, e_z_score, '{} vs {}'.format(z_score, e_z_score))
    #         else:
    #             if z_score < 0:
    #                 self.assertTrue(e_z_score < 0)
    #             else:
    #                 self.assertFalse(e_z_score < 0)
    #             self.assertLess(np.abs(z_score - e_z_score), 1E-5, '{} vs {}'.format(z_score, e_z_score))
    #             expected_composed_z_score = (e_w - e_w_ave) / expected_composed_sigma
    #             self.assertLess(np.abs(z_score - expected_composed_z_score), 1E-16)
    #     ################################################################################################################
    #     init_compute_w2_ave_sub(dists=self.scorer2.distances, bias_bool=True)
    #     cases_biased = {}
    #     for i in range(self.scorer2.distances.shape[0]):
    #         curr_cases = compute_w2_ave_sub(i)
    #         for k in curr_cases:
    #             if k not in cases_biased:
    #                 cases_biased[k] = 0
    #             cases_biased[k] += curr_cases[k]
    #     init_clustering_z_score(bias_bool=True, w2_ave_sub_dict=cases_biased, curr_pdb=self.scorer2.query_structure,
    #                             map_to_structure=self.scorer2.query_pdb_mapping, residue_dists=self.scorer2.distances,
    #                             best_chain=self.scorer2.best_chain)
    #     for i in range(len(residue_list)):
    #         curr_residues = residue_list[:(i + 1)]
    #         a, m, l, pi1, pi2, pi3, z_score, w, w_ave, w2_ave, sigma, num_residues = clustering_z_score(curr_residues)
    #         em, el, epi1, epi2, epi3, e_z_score, e_w, e_w_ave, e_w2_ave, e_sigma, e_cases = self._et_calcZScore(
    #             reslist=curr_residues, L=len(self.scorer2.query_structure.seq[self.scorer2.best_chain]),
    #             A=expected_adjacency, bias=True)
    #         for res_i in expected_adjacency:
    #             for res_j in expected_adjacency[res_i]:
    #                 self.assertEqual(a[self.scorer2.query_pdb_mapping[res_i], self.scorer2.query_pdb_mapping[res_j]],
    #                                  expected_adjacency[res_i][res_j])
    #                 a[self.scorer2.query_pdb_mapping[res_i], self.scorer2.query_pdb_mapping[res_j]] = 0
    #         self.assertEqual(m, em)
    #         self.assertEqual(l, el)
    #         self.assertLess(np.abs(pi1 - epi1), 1E-16)
    #         self.assertLess(np.abs(pi2 - epi2), 1E-16)
    #         self.assertLess(np.abs(pi3 - epi3), 1E-16)
    #         self.assertEqual(num_residues, len(curr_residues))
    #         self.assertLess(np.abs(w - e_w), 1E-16, '{} vs {}'.format(w, e_w))
    #         self.assertLess(np.abs(w_ave - e_w_ave), 1E-16, '{} vs {}'.format(w_ave, e_w_ave))
    #         for case in e_cases:
    #             self.assertEqual(cases_biased[case], e_cases[case])
    #         self.assertLess(np.abs(w2_ave - e_w2_ave), 1E-2, '{} vs {}'.format(w2_ave, e_w2_ave))
    #         composed_w2_ave = ((pi1 * cases_biased['Case1']) + (pi2 * cases_biased['Case2']) +
    #                            (pi3 * cases_biased['Case3']))
    #         expected_composed_w2_ave = ((epi1 * e_cases['Case1']) + (epi2 * e_cases['Case2']) +
    #                                     (epi3 * e_cases['Case3']))
    #         self.assertLess(np.abs(composed_w2_ave - expected_composed_w2_ave), 1E-16)
    #         self.assertLess(np.abs(sigma - e_sigma), 1E-5, '{} vs {}'.format(sigma, e_sigma))
    #         expected_composed_sigma = math.sqrt(expected_composed_w2_ave - e_w_ave * e_w_ave)
    #         self.assertLess(np.abs(sigma - expected_composed_sigma), 1E-16)
    #         if isinstance(z_score, str):
    #             self.assertTrue(isinstance(e_z_score, str))
    #             self.assertEqual(z_score, e_z_score, '{} vs {}'.format(z_score, e_z_score))
    #         else:
    #             if z_score < 0:
    #                 self.assertTrue(e_z_score < 0)
    #             else:
    #                 self.assertFalse(e_z_score < 0)
    #             self.assertLess(np.abs(z_score - e_z_score), 1E-5, '{} vs {}'.format(z_score, e_z_score))
    #             expected_composed_z_score = (e_w - e_w_ave) / expected_composed_sigma
    #             self.assertLess(np.abs(z_score - expected_composed_z_score), 1E-16)
    #
    # def test_19a_score_clustering_of_contact_predictions(self):
    #     # Initialize scorer and scores
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     scores1 = np.random.RandomState(1234567890).rand(self.scorer1.query_alignment.seq_length,
    #                                                      self.scorer1.query_alignment.seq_length)
    #     scores1[np.tril_indices(self.scorer1.query_alignment.seq_length, 1)] = 0
    #     scores1 += scores1.T
    #     # Calculate biased Z-scores for the small structure
    #     start1 = time()
    #     output_fn_1b = os.path.join(self.testing_dir, 'z_score1b.tsv')
    #     zscore_df_1b, _, _ = self.scorer1.score_clustering_of_contact_predictions(
    #         predictions=scores1, bias=True, file_path=output_fn_1b, w2_ave_sub=None)
    #     end1 = time()
    #     print('Time for ContactScorer to compute SCW: {}'.format((end1 - start1) / 60.0))
    #     # Check that the scoring file was written out to the expected file.
    #     self.assertTrue(os.path.isfile(output_fn_1b))
    #     os.remove(output_fn_1b)
    #     # Generate data for calculating expected values
    #     recip_map = {v: k for k, v in self.scorer1.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer1.query_structure.pdb_residue_list[self.scorer1.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     A, res_atoms = self._et_computeAdjacency(self.scorer1.query_structure.structure[0][self.scorer1.best_chain],
    #                                              mapping=final_map)
    #     # Iterate over the returned data frame row by row and test whether the results are correct
    #     visited_scorable_residues = set()
    #     prev_len = 0
    #     prev_stats = None
    #     prev_composed_w2_ave = None
    #     prev_composed_sigma = None
    #     prev_composed_z_score = None
    #     for ind in zscore_df_1b.index:
    #         print('{}:{}'.format(ind, np.max(zscore_df_1b.index)))
    #         res_i = zscore_df_1b.loc[ind, 'Res_i']
    #         res_j = zscore_df_1b.loc[ind, 'Res_j']
    #         if (res_i in self.scorer1.query_pdb_mapping) and (res_j in self.scorer1.query_pdb_mapping):
    #             visited_scorable_residues.add(res_i)
    #             visited_scorable_residues.add(res_j)
    #             if len(visited_scorable_residues) > prev_len:
    #                 curr_stats = self._et_calcZScore(
    #                     reslist=sorted(visited_scorable_residues),
    #                     L=len(self.scorer1.query_structure.seq[self.scorer1.best_chain]), A=A, bias=True)
    #                 expected_composed_w2_ave = ((curr_stats[2] * curr_stats[10]['Case1']) +
    #                                             (curr_stats[3] * curr_stats[10]['Case2']) +
    #                                             (curr_stats[4] * curr_stats[10]['Case3']))
    #                 expected_composed_sigma = math.sqrt(expected_composed_w2_ave - curr_stats[7] * curr_stats[7])
    #                 if expected_composed_sigma == 0.0:
    #                     expected_composed_z_score = 'NA'
    #                 else:
    #                     expected_composed_z_score = (curr_stats[6] - curr_stats[7]) / expected_composed_sigma
    #                 prev_len = len(visited_scorable_residues)
    #                 prev_stats = curr_stats
    #                 prev_composed_w2_ave = expected_composed_w2_ave
    #                 prev_composed_sigma = expected_composed_sigma
    #                 prev_composed_z_score = expected_composed_z_score
    #             else:
    #                 curr_stats = prev_stats
    #                 expected_composed_w2_ave = prev_composed_w2_ave
    #                 expected_composed_sigma = prev_composed_sigma
    #                 expected_composed_z_score = prev_composed_z_score
    #             error_message = '\nW: {}\nExpected W: {}\nW Ave: {}\nExpected W Ave: {}\nW2 Ave: {}\nExpected W2 Ave: '\
    #                             '{}\nComposed Expected W2 Ave: {}\nSigma: {}\nExpected Sigma: {}\nComposed Expected '\
    #                             'Sigma: {}\nZ-Score: {}\nExpected Z-Score: {}\nComposed Expected Z-Score: {}'.format(
    #                 zscore_df_1b.loc[ind, 'W'], curr_stats[6], zscore_df_1b.loc[ind, 'W_Ave'], curr_stats[7],
    #                 zscore_df_1b.loc[ind, 'W2_Ave'], curr_stats[8], expected_composed_w2_ave,
    #                 zscore_df_1b.loc[ind, 'Sigma'], curr_stats[9], expected_composed_sigma,
    #                 zscore_df_1b.loc[ind, 'Z-Score'], curr_stats[5], expected_composed_z_score)
    #             self.assertEqual(zscore_df_1b.loc[ind, 'Num_Residues'], len(visited_scorable_residues))
    #             self.assertLessEqual(np.abs(zscore_df_1b.loc[ind, 'W'] - curr_stats[6]), 1E-16, error_message)
    #             self.assertLessEqual(np.abs(zscore_df_1b.loc[ind, 'W_Ave'] - curr_stats[7]), 1E-16, error_message)
    #             self.assertLessEqual(np.abs(zscore_df_1b.loc[ind, 'W2_Ave'] - expected_composed_w2_ave), 1E-16,
    #                                  error_message)
    #             self.assertLessEqual(np.abs(zscore_df_1b.loc[ind, 'W2_Ave'] - curr_stats[8]), 1E-3, error_message)
    #             self.assertLessEqual(np.abs(zscore_df_1b.loc[ind, 'Sigma'] - expected_composed_sigma), 1E-16,
    #                                  error_message)
    #             self.assertLessEqual(np.abs(zscore_df_1b.loc[ind, 'Sigma'] - curr_stats[9]), 1E-5, error_message)
    #             if expected_composed_sigma == 0.0:
    #                 self.assertEqual(zscore_df_1b.loc[ind, 'Z-Score'], expected_composed_z_score)
    #                 self.assertEqual(zscore_df_1b.loc[ind, 'Z-Score'], curr_stats[5])
    #             else:
    #                 self.assertLessEqual(np.abs(zscore_df_1b.loc[ind, 'Z-Score'] - expected_composed_z_score), 1E-16,
    #                                      error_message)
    #                 self.assertLessEqual(np.abs(zscore_df_1b.loc[ind, 'Z-Score'] - curr_stats[5]), 1E-5, error_message)
    #         else:
    #             self.assertEqual(zscore_df_1b.loc[ind, 'Z-Score'], '-')
    #             self.assertTrue(np.isnan(zscore_df_1b.loc[ind, 'W']))
    #             self.assertTrue(np.isnan(zscore_df_1b.loc[ind, 'W_Ave']))
    #             self.assertTrue(np.isnan(zscore_df_1b.loc[ind, 'W2_Ave']))
    #             self.assertTrue(np.isnan(zscore_df_1b.loc[ind, 'Sigma']))
    #             self.assertTrue(np.isnan(zscore_df_1b.loc[ind, 'Num_Residues']))
    #         self.assertEqual(zscore_df_1b.loc[ind, 'Covariance_Score'], scores1[res_i, res_j])
    #
    # def test_19b_score_clustering_of_contact_predictions(self):
    #     # Initialize scorer and scores
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     scores1 = np.random.RandomState(1234567890).rand(self.scorer1.query_alignment.seq_length,
    #                                                      self.scorer1.query_alignment.seq_length)
    #     scores1[np.tril_indices(self.scorer1.query_alignment.seq_length, 1)] = 0
    #     scores1 += scores1.T
    #     # Calculate biased Z-scores for the small structure
    #     start1 = time()
    #     output_fn_1u = os.path.join(self.testing_dir, 'z_score1u.tsv')
    #     zscore_df_1u, _, _ = self.scorer1.score_clustering_of_contact_predictions(
    #         predictions=scores1, bias=False, file_path=output_fn_1u, w2_ave_sub=None)
    #     end1 = time()
    #     print('Time for ContactScorer to compute SCW: {}'.format((end1 - start1) / 60.0))
    #     # Check that the scoring file was written out to the expected file.
    #     self.assertTrue(os.path.isfile(output_fn_1u))
    #     os.remove(output_fn_1u)
    #     # Generate data for calculating expected values
    #     recip_map = {v: k for k, v in self.scorer1.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer1.query_structure.pdb_residue_list[self.scorer1.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     A, res_atoms = self._et_computeAdjacency(self.scorer1.query_structure.structure[0][self.scorer1.best_chain],
    #                                              mapping=final_map)
    #     # Iterate over the returned data frame row by row and test whether the results are correct
    #     visited_scorable_residues = set()
    #     prev_len = 0
    #     prev_stats = None
    #     prev_composed_w2_ave = None
    #     prev_composed_sigma = None
    #     prev_composed_z_score = None
    #     for ind in zscore_df_1u.index:
    #         print('{}:{}'.format(ind, np.max(zscore_df_1u.index)))
    #         res_i = zscore_df_1u.loc[ind, 'Res_i']
    #         res_j = zscore_df_1u.loc[ind, 'Res_j']
    #         if (res_i in self.scorer1.query_pdb_mapping) and (res_j in self.scorer1.query_pdb_mapping):
    #             visited_scorable_residues.add(res_i)
    #             visited_scorable_residues.add(res_j)
    #             if len(visited_scorable_residues) > prev_len:
    #                 curr_stats = self._et_calcZScore(
    #                     reslist=sorted(visited_scorable_residues),
    #                     L=len(self.scorer1.query_structure.seq[self.scorer1.best_chain]), A=A, bias=False)
    #                 expected_composed_w2_ave = ((curr_stats[2] * curr_stats[10]['Case1']) +
    #                                             (curr_stats[3] * curr_stats[10]['Case2']) +
    #                                             (curr_stats[4] * curr_stats[10]['Case3']))
    #                 expected_composed_sigma = math.sqrt(expected_composed_w2_ave - curr_stats[7] * curr_stats[7])
    #                 if expected_composed_sigma == 0.0:
    #                     expected_composed_z_score = 'NA'
    #                 else:
    #                     expected_composed_z_score = (curr_stats[6] - curr_stats[7]) / expected_composed_sigma
    #                 prev_len = len(visited_scorable_residues)
    #                 prev_stats = curr_stats
    #                 prev_composed_w2_ave = expected_composed_w2_ave
    #                 prev_composed_sigma = expected_composed_sigma
    #                 prev_composed_z_score = expected_composed_z_score
    #             else:
    #                 curr_stats = prev_stats
    #                 expected_composed_w2_ave = prev_composed_w2_ave
    #                 expected_composed_sigma = prev_composed_sigma
    #                 expected_composed_z_score = prev_composed_z_score
    #             error_message = '\nW: {}\nExpected W: {}\nW Ave: {}\nExpected W Ave: {}\nW2 Ave: {}\nExpected W2 Ave: '\
    #                             '{}\nComposed Expected W2 Ave: {}\nSigma: {}\nExpected Sigma: {}\nComposed Expected '\
    #                             'Sigma: {}\nZ-Score: {}\nExpected Z-Score: {}\nComposed Expected Z-Score: {}'.format(
    #                 zscore_df_1u.loc[ind, 'W'], curr_stats[6], zscore_df_1u.loc[ind, 'W_Ave'], curr_stats[7],
    #                 zscore_df_1u.loc[ind, 'W2_Ave'], curr_stats[8], expected_composed_w2_ave,
    #                 zscore_df_1u.loc[ind, 'Sigma'], curr_stats[9], expected_composed_sigma,
    #                 zscore_df_1u.loc[ind, 'Z-Score'], curr_stats[5], expected_composed_z_score)
    #             self.assertEqual(zscore_df_1u.loc[ind, 'Num_Residues'], len(visited_scorable_residues))
    #             self.assertLessEqual(np.abs(zscore_df_1u.loc[ind, 'W'] - curr_stats[6]), 1E-16, error_message)
    #             self.assertLessEqual(np.abs(zscore_df_1u.loc[ind, 'W_Ave'] - curr_stats[7]), 1E-16, error_message)
    #             self.assertLessEqual(np.abs(zscore_df_1u.loc[ind, 'W2_Ave'] - expected_composed_w2_ave), 1E-16,
    #                                  error_message)
    #             self.assertLessEqual(np.abs(zscore_df_1u.loc[ind, 'W2_Ave'] - curr_stats[8]), 1E-5, error_message)
    #             self.assertLessEqual(np.abs(zscore_df_1u.loc[ind, 'Sigma'] - expected_composed_sigma), 1E-16,
    #                                  error_message)
    #             self.assertLessEqual(np.abs(zscore_df_1u.loc[ind, 'Sigma'] - curr_stats[9]), 1E-5, error_message)
    #             if expected_composed_sigma == 0.0:
    #                 self.assertEqual(zscore_df_1u.loc[ind, 'Z-Score'], expected_composed_z_score)
    #                 self.assertEqual(zscore_df_1u.loc[ind, 'Z-Score'], curr_stats[5])
    #             else:
    #                 self.assertLessEqual(np.abs(zscore_df_1u.loc[ind, 'Z-Score'] - expected_composed_z_score), 1E-16,
    #                                      error_message)
    #                 self.assertLessEqual(np.abs(zscore_df_1u.loc[ind, 'Z-Score'] - curr_stats[5]), 1E-6, error_message)
    #         else:
    #             self.assertEqual(zscore_df_1u.loc[ind, 'Z-Score'], '-')
    #             self.assertTrue(np.isnan(zscore_df_1u.loc[ind, 'W']))
    #             self.assertTrue(np.isnan(zscore_df_1u.loc[ind, 'W_Ave']))
    #             self.assertTrue(np.isnan(zscore_df_1u.loc[ind, 'W2_Ave']))
    #             self.assertTrue(np.isnan(zscore_df_1u.loc[ind, 'Sigma']))
    #             self.assertTrue(np.isnan(zscore_df_1u.loc[ind, 'Num_Residues']))
    #         self.assertEqual(zscore_df_1u.loc[ind, 'Covariance_Score'], scores1[res_i, res_j])
    #
    # def test_19c_score_clustering_of_contact_predictions(self):
    #     # Initialize scorer and scores
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     scores2 = np.random.RandomState(1234567890).rand(self.scorer2.query_alignment.seq_length,
    #                                                      self.scorer2.query_alignment.seq_length)
    #     scores2[np.tril_indices(self.scorer2.query_alignment.seq_length, 1)] = 0
    #     scores2 += scores2.T
    #     # Calculate biased Z-scores for the small structure
    #     start1 = time()
    #     output_fn_2b = os.path.join(self.testing_dir, 'z_score2b.tsv')
    #     zscore_df_2b, _, _ = self.scorer2.score_clustering_of_contact_predictions(
    #         predictions=scores2, bias=True, file_path=output_fn_2b, w2_ave_sub=None)
    #     end1 = time()
    #     print('Time for ContactScorer to compute SCW: {}'.format((end1 - start1) / 60.0))
    #     # Check that the scoring file was written out to the expected file.
    #     self.assertTrue(os.path.isfile(output_fn_2b))
    #     os.remove(output_fn_2b)
    #     # Generate data for calculating expected values
    #     recip_map = {v: k for k, v in self.scorer2.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer2.query_structure.pdb_residue_list[self.scorer2.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     A, res_atoms = self._et_computeAdjacency(self.scorer2.query_structure.structure[0][self.scorer2.best_chain],
    #                                              mapping=final_map)
    #     # Iterate over the returned data frame row by row and test whether the results are correct
    #     visited_scorable_residues = set()
    #     prev_len = 0
    #     prev_stats = None
    #     prev_composed_w2_ave = None
    #     prev_composed_sigma = None
    #     prev_composed_z_score = None
    #     for ind in zscore_df_2b.index:
    #         print('{}:{}'.format(ind, np.max(zscore_df_2b.index)))
    #         res_i = zscore_df_2b.loc[ind, 'Res_i']
    #         res_j = zscore_df_2b.loc[ind, 'Res_j']
    #         if (res_i in self.scorer2.query_pdb_mapping) and (res_j in self.scorer2.query_pdb_mapping):
    #             visited_scorable_residues.add(res_i)
    #             visited_scorable_residues.add(res_j)
    #             if len(visited_scorable_residues) > prev_len:
    #                 curr_stats = self._et_calcZScore(
    #                     reslist=sorted(visited_scorable_residues),
    #                     L=len(self.scorer2.query_structure.seq[self.scorer2.best_chain]), A=A, bias=True)
    #                 expected_composed_w2_ave = ((curr_stats[2] * curr_stats[10]['Case1']) +
    #                                             (curr_stats[3] * curr_stats[10]['Case2']) +
    #                                             (curr_stats[4] * curr_stats[10]['Case3']))
    #                 expected_composed_sigma = math.sqrt(expected_composed_w2_ave - curr_stats[7] * curr_stats[7])
    #                 if expected_composed_sigma == 0.0:
    #                     expected_composed_z_score = 'NA'
    #                 else:
    #                     expected_composed_z_score = (curr_stats[6] - curr_stats[7]) / expected_composed_sigma
    #                 prev_len = len(visited_scorable_residues)
    #                 prev_stats = curr_stats
    #                 prev_composed_w2_ave = expected_composed_w2_ave
    #                 prev_composed_sigma = expected_composed_sigma
    #                 prev_composed_z_score = expected_composed_z_score
    #             else:
    #                 curr_stats = prev_stats
    #                 expected_composed_w2_ave = prev_composed_w2_ave
    #                 expected_composed_sigma = prev_composed_sigma
    #                 expected_composed_z_score = prev_composed_z_score
    #             error_message = '\nW: {}\nExpected W: {}\nW Ave: {}\nExpected W Ave: {}\nW2 Ave: {}\nExpected W2 Ave: '\
    #                             '{}\nComposed Expected W2 Ave: {}\nSigma: {}\nExpected Sigma: {}\nComposed Expected '\
    #                             'Sigma: {}\nZ-Score: {}\nExpected Z-Score: {}\nComposed Expected Z-Score: {}'.format(
    #                 zscore_df_2b.loc[ind, 'W'], curr_stats[6], zscore_df_2b.loc[ind, 'W_Ave'], curr_stats[7],
    #                 zscore_df_2b.loc[ind, 'W2_Ave'], curr_stats[8], expected_composed_w2_ave,
    #                 zscore_df_2b.loc[ind, 'Sigma'], curr_stats[9], expected_composed_sigma,
    #                 zscore_df_2b.loc[ind, 'Z-Score'], curr_stats[5], expected_composed_z_score)
    #             self.assertEqual(zscore_df_2b.loc[ind, 'Num_Residues'], len(visited_scorable_residues))
    #             self.assertLessEqual(np.abs(zscore_df_2b.loc[ind, 'W'] - curr_stats[6]), 1E-16, error_message)
    #             self.assertLessEqual(np.abs(zscore_df_2b.loc[ind, 'W_Ave'] - curr_stats[7]), 1E-16, error_message)
    #             self.assertLessEqual(np.abs(zscore_df_2b.loc[ind, 'W2_Ave'] - expected_composed_w2_ave), 1E-16,
    #                                  error_message)
    #             self.assertLessEqual(np.abs(zscore_df_2b.loc[ind, 'W2_Ave'] - curr_stats[8]), 1E-2, error_message)
    #             self.assertLessEqual(np.abs(zscore_df_2b.loc[ind, 'Sigma'] - expected_composed_sigma), 1E-16,
    #                                  error_message)
    #             self.assertLessEqual(np.abs(zscore_df_2b.loc[ind, 'Sigma'] - curr_stats[9]), 1E-5, error_message)
    #             if expected_composed_sigma == 0.0:
    #                 self.assertEqual(zscore_df_2b.loc[ind, 'Z-Score'], expected_composed_z_score)
    #                 self.assertEqual(zscore_df_2b.loc[ind, 'Z-Score'], curr_stats[5])
    #             else:
    #                 self.assertLessEqual(np.abs(zscore_df_2b.loc[ind, 'Z-Score'] - expected_composed_z_score), 1E-16,
    #                                      error_message)
    #                 self.assertLessEqual(np.abs(zscore_df_2b.loc[ind, 'Z-Score'] - curr_stats[5]), 1E-5, error_message)
    #         else:
    #             self.assertEqual(zscore_df_2b.loc[ind, 'Z-Score'], '-')
    #             self.assertTrue(np.isnan(zscore_df_2b.loc[ind, 'W']))
    #             self.assertTrue(np.isnan(zscore_df_2b.loc[ind, 'W_Ave']))
    #             self.assertTrue(np.isnan(zscore_df_2b.loc[ind, 'W2_Ave']))
    #             self.assertTrue(np.isnan(zscore_df_2b.loc[ind, 'Sigma']))
    #             self.assertTrue(np.isnan(zscore_df_2b.loc[ind, 'Num_Residues']))
    #         self.assertEqual(zscore_df_2b.loc[ind, 'Covariance_Score'], scores2[res_i, res_j])
    #
    # def test_19d_score_clustering_of_contact_predictions(self):
    #     # Initialize scorer and scores
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     scores2 = np.random.RandomState(1234567890).rand(self.scorer2.query_alignment.seq_length,
    #                                                      self.scorer2.query_alignment.seq_length)
    #     scores2[np.tril_indices(self.scorer2.query_alignment.seq_length, 1)] = 0
    #     scores2 += scores2.T
    #     # Calculate biased Z-scores for the small structure
    #     start1 = time()
    #     output_fn_2u = os.path.join(self.testing_dir, 'z_score2u.tsv')
    #     zscore_df_2u, _, _ = self.scorer2.score_clustering_of_contact_predictions(
    #         predictions=scores2, bias=False, file_path=output_fn_2u, w2_ave_sub=None)
    #     end1 = time()
    #     print('Time for ContactScorer to compute SCW: {}'.format((end1 - start1) / 60.0))
    #     # Check that the scoring file was written out to the expected file.
    #     self.assertTrue(os.path.isfile(output_fn_2u))
    #     os.remove(output_fn_2u)
    #     # Generate data for calculating expected values
    #     recip_map = {v: k for k, v in self.scorer2.query_pdb_mapping.items()}
    #     struc_seq_map = {k: i for i, k in
    #                      enumerate(self.scorer2.query_structure.pdb_residue_list[self.scorer2.best_chain])}
    #     final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
    #     A, res_atoms = self._et_computeAdjacency(self.scorer2.query_structure.structure[0][self.scorer2.best_chain],
    #                                              mapping=final_map)
    #     # Iterate over the returned data frame row by row and test whether the results are correct
    #     visited_scorable_residues = set()
    #     prev_len = 0
    #     prev_stats = None
    #     prev_composed_w2_ave = None
    #     prev_composed_sigma = None
    #     prev_composed_z_score = None
    #     for ind in zscore_df_2u.index:
    #         print('{}:{}'.format(ind, np.max(zscore_df_2u.index)))
    #         res_i = zscore_df_2u.loc[ind, 'Res_i']
    #         res_j = zscore_df_2u.loc[ind, 'Res_j']
    #         if (res_i in self.scorer2.query_pdb_mapping) and (res_j in self.scorer2.query_pdb_mapping):
    #             visited_scorable_residues.add(res_i)
    #             visited_scorable_residues.add(res_j)
    #             if len(visited_scorable_residues) > prev_len:
    #                 curr_stats = self._et_calcZScore(
    #                     reslist=sorted(visited_scorable_residues),
    #                     L=len(self.scorer2.query_structure.seq[self.scorer2.best_chain]), A=A, bias=False)
    #                 expected_composed_w2_ave = ((curr_stats[2] * curr_stats[10]['Case1']) +
    #                                             (curr_stats[3] * curr_stats[10]['Case2']) +
    #                                             (curr_stats[4] * curr_stats[10]['Case3']))
    #                 expected_composed_sigma = math.sqrt(expected_composed_w2_ave - curr_stats[7] * curr_stats[7])
    #                 if expected_composed_sigma == 0.0:
    #                     expected_composed_z_score = 'NA'
    #                 else:
    #                     expected_composed_z_score = (curr_stats[6] - curr_stats[7]) / expected_composed_sigma
    #                 prev_len = len(visited_scorable_residues)
    #                 prev_stats = curr_stats
    #                 prev_composed_w2_ave = expected_composed_w2_ave
    #                 prev_composed_sigma = expected_composed_sigma
    #                 prev_composed_z_score = expected_composed_z_score
    #             else:
    #                 curr_stats = prev_stats
    #                 expected_composed_w2_ave = prev_composed_w2_ave
    #                 expected_composed_sigma = prev_composed_sigma
    #                 expected_composed_z_score = prev_composed_z_score
    #             error_message = '\nW: {}\nExpected W: {}\nW Ave: {}\nExpected W Ave: {}\nW2 Ave: {}\nExpected W2 Ave: '\
    #                             '{}\nComposed Expected W2 Ave: {}\nSigma: {}\nExpected Sigma: {}\nComposed Expected '\
    #                             'Sigma: {}\nZ-Score: {}\nExpected Z-Score: {}\nComposed Expected Z-Score: {}'.format(
    #                 zscore_df_2u.loc[ind, 'W'], curr_stats[6], zscore_df_2u.loc[ind, 'W_Ave'], curr_stats[7],
    #                 zscore_df_2u.loc[ind, 'W2_Ave'], curr_stats[8], expected_composed_w2_ave,
    #                 zscore_df_2u.loc[ind, 'Sigma'], curr_stats[9], expected_composed_sigma,
    #                 zscore_df_2u.loc[ind, 'Z-Score'], curr_stats[5], expected_composed_z_score)
    #             self.assertEqual(zscore_df_2u.loc[ind, 'Num_Residues'], len(visited_scorable_residues))
    #             self.assertLessEqual(np.abs(zscore_df_2u.loc[ind, 'W'] - curr_stats[6]), 1E-16, error_message)
    #             self.assertLessEqual(np.abs(zscore_df_2u.loc[ind, 'W_Ave'] - curr_stats[7]), 1E-16, error_message)
    #             self.assertLessEqual(np.abs(zscore_df_2u.loc[ind, 'W2_Ave'] - expected_composed_w2_ave), 1E-16,
    #                                  error_message)
    #             self.assertLessEqual(np.abs(zscore_df_2u.loc[ind, 'W2_Ave'] - curr_stats[8]), 1E-4, error_message)
    #             self.assertLessEqual(np.abs(zscore_df_2u.loc[ind, 'Sigma'] - expected_composed_sigma), 1E-16,
    #                                  error_message)
    #             self.assertLessEqual(np.abs(zscore_df_2u.loc[ind, 'Sigma'] - curr_stats[9]), 1E-5, error_message)
    #             if expected_composed_sigma == 0.0:
    #                 self.assertEqual(zscore_df_2u.loc[ind, 'Z-Score'], expected_composed_z_score)
    #                 self.assertEqual(zscore_df_2u.loc[ind, 'Z-Score'], curr_stats[5])
    #             else:
    #                 self.assertLessEqual(np.abs(zscore_df_2u.loc[ind, 'Z-Score'] - expected_composed_z_score), 1E-16,
    #                                      error_message)
    #                 self.assertLessEqual(np.abs(zscore_df_2u.loc[ind, 'Z-Score'] - curr_stats[5]), 1E-6, error_message)
    #         else:
    #             self.assertEqual(zscore_df_2u.loc[ind, 'Z-Score'], '-')
    #             self.assertTrue(np.isnan(zscore_df_2u.loc[ind, 'W']))
    #             self.assertTrue(np.isnan(zscore_df_2u.loc[ind, 'W_Ave']))
    #             self.assertTrue(np.isnan(zscore_df_2u.loc[ind, 'W2_Ave']))
    #             self.assertTrue(np.isnan(zscore_df_2u.loc[ind, 'Sigma']))
    #             self.assertTrue(np.isnan(zscore_df_2u.loc[ind, 'Num_Residues']))
    #         self.assertEqual(zscore_df_2u.loc[ind, 'Covariance_Score'], scores2[res_i, res_j])
    #
    # def test_20a_write_out_clustering_results(self):
    #     today = str(datetime.date.today())
    #     header = ['Pos1', '(AA1)', 'Pos2', '(AA2)', 'Raw_Score', 'Coverage_Score', 'Residue_Dist', 'Within_Threshold']
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
    #     scores1 = np.random.RandomState(1234567890).rand(self.scorer1.query_alignment.seq_length,
    #                                                      self.scorer1.query_alignment.seq_length)
    #     scores1[np.tril_indices(self.scorer1.query_alignment.seq_length, 1)] = 0
    #     scores1 += scores1.T
    #     coverages1 = np.random.RandomState(179424691).rand(self.scorer1.query_alignment.seq_length,
    #                                                        self.scorer1.query_alignment.seq_length)
    #     coverages1[np.tril_indices(self.scorer1.query_alignment.seq_length, 1)] = 0
    #     coverages1 += coverages1.T
    #     self.scorer1.write_out_clustering_results(today=today, raw_scores=scores1, coverage_scores=coverages1,
    #                                               file_name='Contact_1a_Scores.tsv', output_dir=self.testing_dir)
    #     curr_path = os.path.join(self.testing_dir, 'Contact_1a_Scores.tsv')
    #     self.assertTrue(os.path.isfile(curr_path))
    #     test_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    #     self.assertEqual(list(test_df.columns), header)
    #     self.comp_function(df=test_df, q_ind_map=pdb_index_mapping, q_to_s_map=pdb_query_mapping,
    #                        seq_pdb_map=self.scorer1.query_pdb_mapping,
    #                        seq=self.scorer1.query_alignment.query_sequence, scores=scores1, coverages=coverages1,
    #                        distances=self.scorer1.distances, adjacencies=A)
    #     os.remove(curr_path)
    #     self.scorer1.write_out_clustering_results(today=None, raw_scores=scores1, coverage_scores=coverages1,
    #                                               file_name='Contact_1b_Scores.tsv', output_dir=self.testing_dir)
    #     curr_path = os.path.join(self.testing_dir, 'Contact_1b_Scores.tsv')
    #     self.assertTrue(os.path.isfile(curr_path))
    #     test_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    #     self.assertEqual(list(test_df.columns), header)
    #     self.comp_function(df=test_df, q_ind_map=pdb_index_mapping, q_to_s_map=pdb_query_mapping,
    #                        seq_pdb_map=self.scorer1.query_pdb_mapping,
    #                        seq=self.scorer1.query_alignment.query_sequence, scores=scores1, coverages=coverages1,
    #                        distances=self.scorer1.distances, adjacencies=A)
    #     os.remove(curr_path)
    #
    #     self.scorer1.write_out_clustering_results(today=today, raw_scores=scores1, coverage_scores=coverages1,
    #                                               file_name=None, output_dir=self.testing_dir)
    #     curr_path = os.path.join(self.testing_dir, "{}_{}.Covariance_vs_Structure.txt".format(today, self.scorer1.query))
    #     self.assertTrue(os.path.isfile(curr_path))
    #     test_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    #     self.assertEqual(list(test_df.columns), header)
    #     self.comp_function(df=test_df, q_ind_map=pdb_index_mapping, q_to_s_map=pdb_query_mapping,
    #                        seq_pdb_map=self.scorer1.query_pdb_mapping,
    #                        seq=self.scorer1.query_alignment.query_sequence, scores=scores1,coverages=coverages1,
    #                        distances=self.scorer1.distances, adjacencies=A)
    #     os.remove(curr_path)
    #
    # def test_20b_write_out_clustering_results(self):
    #     today = str(datetime.date.today())
    #     header = ['Pos1', '(AA1)', 'Pos2', '(AA2)', 'Raw_Score', 'Coverage_Score', 'Residue_Dist', 'Within_Threshold']
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
    #     scores2 = np.random.RandomState(1234567890).rand(self.scorer2.query_alignment.seq_length,
    #                                                      self.scorer2.query_alignment.seq_length)
    #     scores2[np.tril_indices(self.scorer2.query_alignment.seq_length, 1)] = 0
    #     scores2 += scores2.T
    #     coverages2 = np.random.RandomState(179424691).rand(self.scorer2.query_alignment.seq_length,
    #                                                        self.scorer2.query_alignment.seq_length)
    #     coverages2[np.tril_indices(self.scorer2.query_alignment.seq_length, 1)] = 0
    #     coverages2 += coverages2.T
    #     self.scorer2.write_out_clustering_results(today=today, raw_scores=scores2, coverage_scores=coverages2,
    #                                               file_name='Contact_2a_Scores.tsv', output_dir=self.testing_dir)
    #     curr_path = os.path.join(self.testing_dir, 'Contact_2a_Scores.tsv')
    #     self.assertTrue(os.path.isfile(curr_path))
    #     test_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    #     self.assertEqual(list(test_df.columns), header)
    #     self.comp_function(df=test_df, q_ind_map=pdb_index_mapping, q_to_s_map=pdb_query_mapping,
    #                        seq_pdb_map=self.scorer2.query_pdb_mapping,
    #                        seq=self.scorer2.query_alignment.query_sequence, scores=scores2, coverages=coverages2,
    #                        distances=self.scorer2.distances, adjacencies=A)
    #     os.remove(curr_path)
    #     self.scorer2.write_out_clustering_results(today=None, raw_scores=scores2, coverage_scores=coverages2,
    #                                               file_name='Contact_2b_Scores.tsv', output_dir=self.testing_dir)
    #     curr_path = os.path.join(self.testing_dir, 'Contact_2b_Scores.tsv')
    #     self.assertTrue(os.path.isfile(curr_path))
    #     test_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    #     self.assertEqual(list(test_df.columns), header)
    #     self.comp_function(df=test_df, q_ind_map=pdb_index_mapping, q_to_s_map=pdb_query_mapping,
    #                        seq_pdb_map=self.scorer2.query_pdb_mapping,
    #                        seq=self.scorer2.query_alignment.query_sequence, scores=scores2, coverages=coverages2,
    #                        distances=self.scorer2.distances, adjacencies=A)
    #     os.remove(curr_path)
    #     self.scorer2.write_out_clustering_results(today=today, raw_scores=scores2, coverage_scores=coverages2,
    #                                               file_name=None, output_dir=self.testing_dir)
    #     curr_path = os.path.join(self.testing_dir, "{}_{}.Covariance_vs_Structure.txt".format(today, self.scorer2.query))
    #     self.assertTrue(os.path.isfile(curr_path))
    #     test_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    #     self.assertEqual(list(test_df.columns), header)
    #     self.comp_function(df=test_df, q_ind_map=pdb_index_mapping, q_to_s_map=pdb_query_mapping,
    #                        seq_pdb_map=self.scorer2.query_pdb_mapping,
    #                        seq=self.scorer2.query_alignment.query_sequence, scores=scores2, coverages=coverages2,
    #                        distances=self.scorer2.distances, adjacencies=A)
    #     os.remove(curr_path)
    #
    # def test_21a_evaluate_predictions(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     scores1 = np.random.RandomState(1234567890).rand(self.scorer1.query_alignment.seq_length,
    #                                                      self.scorer1.query_alignment.seq_length)
    #     scores1[np.tril_indices(self.scorer1.query_alignment.seq_length, 1)] = 0
    #     scores1 += scores1.T
    #     #
    #     prev_b_w2_ave = None
    #     prev_u_w2_ave = None
    #     for v in range(1, 4):
    #         curr_stats, curr_b_w2_ave, curr_u_w2_ave = self.scorer1.evaluate_predictions(
    #             verbosity=v, out_dir=self.testing_dir, scores=scores1, dist='CB', file_prefix='SCORER1_TEST',
    #             biased_w2_ave=prev_b_w2_ave, unbiased_w2_ave=prev_u_w2_ave, processes=1, threshold=0.5, plots=True)
    #         # Tests
    #         # Check that the correct data is in the dataframe according to the verbosity
    #         column_length = None
    #         for key in curr_stats:
    #             if column_length is None:
    #                 column_length = len(curr_stats[key])
    #             else:
    #                 self.assertEqual(len(curr_stats[key]), column_length)
    #         if v >= 1:
    #             self.assertTrue('Distance' in curr_stats)
    #             self.assertTrue('Sequence_Separation' in curr_stats)
    #             self.assertTrue('AUROC' in curr_stats)
    #             self.assertTrue('AUPRC' in curr_stats)
    #             self.assertTrue('AUTPRFDRC' in curr_stats)
    #             for sep in ['Any', 'Neighbors', 'Short', 'Medium', 'Long']:
    #                 fn1 = os.path.join(self.testing_dir, 'SCORER1_TESTAUROC_Evaluation_Dist-{}_Separation-{}.png'.format(
    #                     'CB', sep))
    #                 self.assertTrue(os.path.isfile(fn1))
    #                 os.remove(fn1)
    #                 fn2 = os.path.join(self.testing_dir, 'SCORER1_TESTAUPRC_Evaluation_Dist-{}_Separation-{}.png'.format(
    #                                 'CB', sep))
    #                 self.assertTrue(os.path.isfile(fn2))
    #                 os.remove(fn2)
    #                 fn3 = os.path.join(self.testing_dir,
    #                                    'SCORER1_TESTAUTPRFDRC_Evaluation_Dist-{}_Separation-{}.png'.format('CB', sep))
    #                 self.assertTrue(os.path.isfile(fn3))
    #                 os.remove(fn3)
    #             if v == 1:
    #                 self.assertTrue(curr_b_w2_ave is None)
    #                 self.assertTrue(curr_u_w2_ave is None)
    #         if v >= 2:
    #             self.assertTrue('Top K Predictions' in curr_stats)
    #             self.assertTrue('F1 Score' in curr_stats)
    #             self.assertTrue('Precision' in curr_stats)
    #             self.assertTrue('Recall' in curr_stats)
    #             if v == 2:
    #                 self.assertTrue(curr_b_w2_ave is None)
    #                 self.assertTrue(curr_u_w2_ave is None)
    #         if v >= 3:
    #             self.assertTrue('Max Biased Z-Score' in curr_stats)
    #             self.assertTrue('AUC Biased Z-Score' in curr_stats)
    #             self.assertTrue('Max Unbiased Z-Score' in curr_stats)
    #             self.assertTrue('AUC Unbiased Z-Score' in curr_stats)
    #             fn4 = os.path.join(self.testing_dir, 'SCORER1_TEST' + 'Dist-CB_Biased_ZScores.tsv')
    #             self.assertTrue(os.path.isfile(fn4))
    #             os.remove(fn4)
    #             fn5 = os.path.join(self.testing_dir, 'SCORER1_TEST' + 'Dist-CB_Biased_ZScores.png')
    #             self.assertTrue(os.path.isfile(fn5))
    #             os.remove(fn5)
    #             fn6 = os.path.join(self.testing_dir, 'SCORER1_TEST' + 'Dist-CB_Unbiased_ZScores.tsv')
    #             self.assertTrue(os.path.isfile(fn6))
    #             os.remove(fn6)
    #             fn7 = os.path.join(self.testing_dir, 'SCORER1_TEST' + 'Dist-CB_Unbiased_ZScores.png')
    #             self.assertTrue(os.path.isfile(fn7))
    #             os.remove(fn7)
    #             self.assertTrue(curr_b_w2_ave is not None)
    #             self.assertTrue(curr_u_w2_ave is not None)
    #         # Update
    #         prev_b_w2_ave = curr_b_w2_ave
    #         prev_u_w2_ave = curr_u_w2_ave
    #
    # def test_21b_evaluate_predictions(self):
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     scores2 = np.random.RandomState(1234567890).rand(self.scorer2.query_alignment.seq_length,
    #                                                      self.scorer2.query_alignment.seq_length)
    #     scores2[np.tril_indices(self.scorer2.query_alignment.seq_length, 1)] = 0
    #     scores2 += scores2.T
    #     #
    #     prev_b_w2_ave = None
    #     prev_u_w2_ave = None
    #     for v in range(1, 4):
    #         curr_stats, curr_b_w2_ave, curr_u_w2_ave = self.scorer2.evaluate_predictions(
    #             verbosity=v, out_dir=self.testing_dir, scores=scores2, dist='CB', file_prefix='SCORER2_TEST',
    #             biased_w2_ave=prev_b_w2_ave, unbiased_w2_ave=prev_u_w2_ave, processes=1, threshold=0.5, plots=True)
    #         # Tests
    #         # Check that the correct data is in the dataframe according to the verbosity
    #         column_length = None
    #         for key in curr_stats:
    #             if column_length is None:
    #                 column_length = len(curr_stats[key])
    #             else:
    #                 self.assertEqual(len(curr_stats[key]), column_length)
    #         if v >= 1:
    #             self.assertTrue('Distance' in curr_stats)
    #             self.assertTrue('Sequence_Separation' in curr_stats)
    #             self.assertTrue('AUROC' in curr_stats)
    #             self.assertTrue('AUPRC' in curr_stats)
    #             self.assertTrue('AUTPRFDRC' in curr_stats)
    #             for sep in ['Any', 'Neighbors', 'Short', 'Medium', 'Long']:
    #                 fn1 = os.path.join(self.testing_dir, 'SCORER2_TESTAUROC_Evaluation_Dist-{}_Separation-{}.png'.format(
    #                     'CB', sep))
    #                 self.assertTrue(os.path.isfile(fn1))
    #                 os.remove(fn1)
    #                 fn2 = os.path.join(self.testing_dir, 'SCORER2_TESTAUPRC_Evaluation_Dist-{}_Separation-{}.png'.format(
    #                     'CB', sep))
    #                 self.assertTrue(os.path.isfile(fn2))
    #                 os.remove(fn2)
    #                 fn3 = os.path.join(self.testing_dir,
    #                                    'SCORER2_TESTAUTPRFDRC_Evaluation_Dist-{}_Separation-{}.png'.format('CB', sep))
    #                 self.assertTrue(os.path.isfile(fn3))
    #                 os.remove(fn3)
    #             if v == 1:
    #                 self.assertTrue(curr_b_w2_ave is None)
    #                 self.assertTrue(curr_u_w2_ave is None)
    #         if v >= 2:
    #             self.assertTrue('Top K Predictions' in curr_stats)
    #             self.assertTrue('F1 Score' in curr_stats)
    #             self.assertTrue('Precision' in curr_stats)
    #             self.assertTrue('Recall' in curr_stats)
    #             if v == 2:
    #                 self.assertTrue(curr_b_w2_ave is None)
    #                 self.assertTrue(curr_u_w2_ave is None)
    #         if v >= 3:
    #             self.assertTrue('Max Biased Z-Score' in curr_stats)
    #             self.assertTrue('AUC Biased Z-Score' in curr_stats)
    #             self.assertTrue('Max Unbiased Z-Score' in curr_stats)
    #             self.assertTrue('AUC Unbiased Z-Score' in curr_stats)
    #             fn4 = os.path.join(self.testing_dir, 'SCORER2_TEST' + 'Dist-CB_Biased_ZScores.tsv')
    #             self.assertTrue(os.path.isfile(fn4))
    #             os.remove(fn4)
    #             fn5 = os.path.join(self.testing_dir, 'SCORER2_TEST' + 'Dist-CB_Biased_ZScores.png')
    #             self.assertTrue(os.path.isfile(fn5))
    #             os.remove(fn5)
    #             fn6 = os.path.join(self.testing_dir, 'SCORER2_TEST' + 'Dist-CB_Unbiased_ZScores.tsv')
    #             self.assertTrue(os.path.isfile(fn6))
    #             os.remove(fn6)
    #             fn7 = os.path.join(self.testing_dir, 'SCORER2_TEST' + 'Dist-CB_Unbiased_ZScores.png')
    #             self.assertTrue(os.path.isfile(fn7))
    #             os.remove(fn7)
    #             self.assertTrue(curr_b_w2_ave is not None)
    #             self.assertTrue(curr_u_w2_ave is not None)
    #         # Update
    #         prev_b_w2_ave = curr_b_w2_ave
    #         prev_u_w2_ave = curr_u_w2_ave
    #
    # def test_22a_evaluate_predictor(self):
    #     etmip1 = EvolutionaryTrace(query_id=self.query1, polymer_type='Protein', aln_fn=self.aln_file1,
    #                                et_distance=True, distance_model='blosum62', tree_building_method='et',
    #                                tree_building_options={}, ranks=None, position_type='pair',
    #                                scoring_metric='filtered_average_product_corrected_mutual_information',
    #                                gap_correction=None, out_dir=os.path.join(self.testing_dir, self.query1),
    #                                output_files=set(), processors=self.max_threads, low_memory=True)
    #     etmip1.import_and_process_aln()
    #     etmip1.compute_distance_matrix_tree_and_assignments()
    #     etmip1.perform_trace()
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     prev_b_w2_ave = None
    #     prev_u_w2_ave = None
    #     for v in range(1, 4):
    #         score_df, curr_b_w2_ave, curr_u_w2_ave = self.scorer1.evaluate_predictor(
    #             predictor=etmip1, verbosity=v, out_dir=self.testing_dir, dist='Any', biased_w2_ave=prev_b_w2_ave,
    #             unbiased_w2_ave=prev_u_w2_ave, processes=self.max_threads, threshold=0.5, pos_size=2, rank_type='min',
    #             file_prefix='SCORER1_TEST', plots=True)
    #         if v >= 1:
    #             self.assertTrue('Distance' in score_df.columns)
    #             self.assertTrue('Sequence_Separation' in score_df.columns)
    #             self.assertTrue('AUROC' in score_df.columns)
    #             self.assertTrue('AUPRC' in score_df.columns)
    #             self.assertTrue('AUTPRFDRC' in score_df.columns)
    #             fn1 = os.path.join(self.testing_dir, '{}_Evaluation_Dist-{}.txt'.format('SCORER1_TEST', 'Any'))
    #             self.assertTrue(os.path.isfile(fn1))
    #             os.remove(fn1)
    #             for sep in ['Any', 'Neighbors', 'Short', 'Medium', 'Long']:
    #                 fn2 = os.path.join(self.testing_dir, 'SCORER1_TESTAUROC_Evaluation_Dist-{}_Separation-{}.png'.format(
    #                     'Any', sep))
    #                 self.assertTrue(os.path.isfile(fn2))
    #                 os.remove(fn2)
    #                 fn3 = os.path.join(self.testing_dir, 'SCORER1_TESTAUPRC_Evaluation_Dist-{}_Separation-{}.png'.format(
    #                     'Any', sep))
    #                 self.assertTrue(os.path.isfile(fn3))
    #                 os.remove(fn3)
    #                 fn4 = os.path.join(self.testing_dir,
    #                                    'SCORER1_TESTAUTPRFDRC_Evaluation_Dist-{}_Separation-{}.png'.format('Any', sep))
    #                 self.assertTrue(os.path.isfile(fn4))
    #                 os.remove(fn4)
    #             if v == 1:
    #                 self.assertTrue(curr_b_w2_ave is None)
    #                 self.assertTrue(curr_u_w2_ave is None)
    #         if v >= 2:
    #             self.assertTrue('Top K Predictions' in score_df.columns)
    #             self.assertTrue('F1 Score' in score_df.columns)
    #             self.assertTrue('Precision' in score_df.columns)
    #             self.assertTrue('Recall' in score_df.columns)
    #             if v == 2:
    #                 self.assertTrue(curr_b_w2_ave is None)
    #                 self.assertTrue(curr_u_w2_ave is None)
    #         if v >= 3:
    #             self.assertTrue('Max Biased Z-Score' in score_df.columns)
    #             self.assertTrue('AUC Biased Z-Score' in score_df.columns)
    #             self.assertTrue('Max Unbiased Z-Score' in score_df.columns)
    #             self.assertTrue('AUC Unbiased Z-Score' in score_df.columns)
    #             fn5 = os.path.join(self.testing_dir, 'SCORER1_TEST' + 'Dist-{}_Biased_ZScores.tsv'.format('Any'))
    #             self.assertTrue(os.path.isfile(fn5))
    #             os.remove(fn5)
    #             fn6 = os.path.join(self.testing_dir, 'SCORER1_TEST' + 'Dist-{}_Biased_ZScores.png'.format('Any'))
    #             self.assertTrue(os.path.isfile(fn6))
    #             os.remove(fn6)
    #             fn7 = os.path.join(self.testing_dir, 'SCORER1_TEST' + 'Dist-{}_Unbiased_ZScores.tsv'.format('Any'))
    #             self.assertTrue(os.path.isfile(fn7))
    #             os.remove(fn7)
    #             fn8 = os.path.join(self.testing_dir, 'SCORER1_TEST' + 'Dist-{}_Unbiased_ZScores.png'.format('Any'))
    #             self.assertTrue(os.path.isfile(fn8))
    #             os.remove(fn8)
    #             self.assertTrue(curr_b_w2_ave is not None)
    #             self.assertTrue(curr_u_w2_ave is not None)
    #         prev_b_w2_ave = curr_b_w2_ave
    #         prev_u_w2_ave = curr_u_w2_ave
    #
    # def test_22b_evaluate_predictor(self):
    #     etmip2 = EvolutionaryTrace(query_id=self.query2, polymer_type='Protein', aln_fn=self.aln_file2,
    #                                et_distance=True, distance_model='blosum62', tree_building_method='et',
    #                                tree_building_options={}, ranks=None, position_type='pair',
    #                                scoring_metric='filtered_average_product_corrected_mutual_information',
    #                                gap_correction=None, out_dir=os.path.join(self.testing_dir, self.query2),
    #                                output_files=set(), processors=self.max_threads, low_memory=True)
    #     etmip2.import_and_process_aln()
    #     etmip2.compute_distance_matrix_tree_and_assignments()
    #     etmip2.perform_trace()
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     prev_b_w2_ave = None
    #     prev_u_w2_ave = None
    #     for v in range(1, 4):
    #         score_df, curr_b_w2_ave, curr_u_w2_ave = self.scorer2.evaluate_predictor(
    #             predictor=etmip2, verbosity=v, out_dir=self.testing_dir, dist='Any', biased_w2_ave=prev_b_w2_ave,
    #             unbiased_w2_ave=prev_u_w2_ave, processes=self.max_threads, threshold=0.5, pos_size=2, rank_type='min',
    #             file_prefix='SCORER2_TEST', plots=True)
    #         if v >= 1:
    #             self.assertTrue('Distance' in score_df.columns)
    #             self.assertTrue('Sequence_Separation' in score_df.columns)
    #             self.assertTrue('AUROC' in score_df.columns)
    #             self.assertTrue('AUPRC' in score_df.columns)
    #             self.assertTrue('AUTPRFDRC' in score_df.columns)
    #             fn1 = os.path.join(self.testing_dir, '{}_Evaluation_Dist-{}.txt'.format('SCORER2_TEST', 'Any'))
    #             self.assertTrue(os.path.isfile(fn1))
    #             os.remove(fn1)
    #             for sep in ['Any', 'Neighbors', 'Short', 'Medium', 'Long']:
    #                 fn2 = os.path.join(self.testing_dir,
    #                                    'SCORER2_TESTAUROC_Evaluation_Dist-{}_Separation-{}.png'.format(
    #                                        'Any', sep))
    #                 self.assertTrue(os.path.isfile(fn2))
    #                 os.remove(fn2)
    #                 fn3 = os.path.join(self.testing_dir,
    #                                    'SCORER2_TESTAUPRC_Evaluation_Dist-{}_Separation-{}.png'.format(
    #                                        'Any', sep))
    #                 self.assertTrue(os.path.isfile(fn3))
    #                 os.remove(fn3)
    #                 fn4 = os.path.join(self.testing_dir,
    #                                    'SCORER2_TESTAUTPRFDRC_Evaluation_Dist-{}_Separation-{}.png'.format('Any', sep))
    #                 self.assertTrue(os.path.isfile(fn4))
    #                 os.remove(fn4)
    #             if v == 1:
    #                 self.assertTrue(curr_b_w2_ave is None)
    #                 self.assertTrue(curr_u_w2_ave is None)
    #         if v >= 2:
    #             self.assertTrue('Top K Predictions' in score_df.columns)
    #             self.assertTrue('F1 Score' in score_df.columns)
    #             self.assertTrue('Precision' in score_df.columns)
    #             self.assertTrue('Recall' in score_df.columns)
    #             if v == 2:
    #                 self.assertTrue(curr_b_w2_ave is None)
    #                 self.assertTrue(curr_u_w2_ave is None)
    #         if v >= 3:
    #             self.assertTrue('Max Biased Z-Score' in score_df.columns)
    #             self.assertTrue('AUC Biased Z-Score' in score_df.columns)
    #             self.assertTrue('Max Unbiased Z-Score' in score_df.columns)
    #             self.assertTrue('AUC Unbiased Z-Score' in score_df.columns)
    #             fn5 = os.path.join(self.testing_dir, 'SCORER2_TEST' + 'Dist-{}_Biased_ZScores.tsv'.format('Any'))
    #             self.assertTrue(os.path.isfile(fn5))
    #             os.remove(fn5)
    #             fn6 = os.path.join(self.testing_dir, 'SCORER2_TEST' + 'Dist-{}_Biased_ZScores.png'.format('Any'))
    #             self.assertTrue(os.path.isfile(fn6))
    #             os.remove(fn6)
    #             fn7 = os.path.join(self.testing_dir, 'SCORER2_TEST' + 'Dist-{}_Unbiased_ZScores.tsv'.format('Any'))
    #             self.assertTrue(os.path.isfile(fn7))
    #             os.remove(fn7)
    #             fn8 = os.path.join(self.testing_dir, 'SCORER2_TEST' + 'Dist-{}_Unbiased_ZScores.png'.format('Any'))
    #             self.assertTrue(os.path.isfile(fn8))
    #             os.remove(fn8)
    #             self.assertTrue(curr_b_w2_ave is not None)
    #             self.assertTrue(curr_u_w2_ave is not None)
    #         prev_b_w2_ave = curr_b_w2_ave
    #         prev_u_w2_ave = curr_u_w2_ave
    #
    # # def test_write_out_contact_scoring(self):
    # #     aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    # #                '-']
    # #     aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
    # #     out_dir = os.path.abspath('../Test/')
    # #     today = str(datetime.date.today())
    # #     headers = {1: ['Pos1', 'AA1', 'Pos2', 'AA2', 'Raw_Score_1', 'Integrated_Score', 'Final_Score', 'Coverage_Score'],
    # #                2: ['Pos1', 'AA1', 'Pos2', 'AA2', 'Raw_Score_1', 'Raw_Score_2', 'Integrated_Score', 'Final_Score', 'Coverage_Score'],
    # #                3: ['Pos1', 'AA1', 'Pos2', 'AA2', 'Raw_Score_1', 'Raw_Score_2', 'Raw_Score_3', 'Integrated_Score', 'Final_Score', 'Coverage_Score'],
    # #                4: ['Pos1', 'AA1', 'Pos2', 'AA2', 'Raw_Score_1', 'Raw_Score_2', 'Raw_Score_3', 'Raw_Score_4', 'Integrated_Score', 'Final_Score', 'Coverage_Score']}
    # #     #
    # #     self.scorer1.fit()
    # #     self.scorer1.measure_distance(method='Any')
    # #     path1 = os.path.join(out_dir, '1c17A.fa')
    # #     etmipc1 = ETMIPC(path1)
    # #     start1 = time()
    # #     time1 = etmipc1.calculate_scores(curr_date=today, query='1c17A', tree_depth=(2, 5),
    # #                                      out_dir=out_dir, processes=1, ignore_alignment_size=True,
    # #                                      clustering='agglomerative', clustering_args={'affinity': 'euclidean',
    # #                                                                                   'linkage': 'ward'},
    # #                                      aa_mapping=aa_dict, combine_clusters='sum', combine_branches='sum',
    # #                                      del_intermediate=False, low_mem=False)
    # #     end1 = time()
    # #     print(time1)
    # #     print(end1 - start1)
    # #     self.assertLessEqual(time1, end1 - start1)
    # #     for branch1 in etmipc1.tree_depth:
    # #         branch_dir = os.path.join(etmipc1.output_dir, str(branch1))
    # #         self.assertTrue(os.path.isdir(branch_dir))
    # #         score_path = os.path.join(branch_dir,
    # #                                   "{}_{}_{}.all_scores.txt".format(today, self.scorer1.query, branch1))
    # #         self.assertTrue(os.path.isfile(score_path))
    # #         test_df = pd.read_csv(score_path, index_col=None, delimiter='\t')
    # #         self.assertEqual(list(test_df.columns), headers[branch1])
    # #         self.score_comp_function(df=test_df, seq=self.scorer1.query_alignment.query_sequence,
    # #                                  clusters=etmipc1.get_cluster_scores(branch=branch1),
    # #                                  branches=etmipc1.get_branch_scores(branch=branch1),
    # #                                  scores=etmipc1.get_scores(branch=branch1),
    # #                                  coverages=etmipc1.get_coverage(branch=branch1))
    # #     for curr_pos, mapped_pos in etmipc1.cluster_mapping.items():
    # #         curr_path = os.path.join(etmipc1.output_dir, str(curr_pos[0]),
    # #                                  "{}_{}_{}.all_scores.txt".format(today, self.scorer1.query, curr_pos[0]))
    # #         mapped_path = os.path.join(etmipc1.output_dir, str(mapped_pos[0]),
    # #                                    "{}_{}_{}.all_scores.txt".format(today, self.scorer1.query, mapped_pos[0]))
    # #         curr_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    # #         mapped_df = pd.read_csv(mapped_path, index_col=None, delimiter='\t')
    # #         self.score_comp_nonunique_cluster_files(df1=curr_df, df2=mapped_df, cluster1=curr_pos[1],
    # #                                                 cluster2=mapped_pos[1])
    # #     for branch1 in etmipc1.tree_depth:
    # #         rmtree(os.path.join(etmipc1.output_dir, str(branch1)))
    # #     os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
    # #     os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
    # #     os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
    # #     os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
    # #     os.remove(os.path.join(os.path.abspath('../Test/'), '{}_cET-MIp.npz'.format('1c17A')))
    # #     os.remove(os.path.join(os.path.abspath('../Test/'), '{}_cET-MIp.pkl'.format('1c17A')))
    # #     rmtree(os.path.join(out_dir, 'joblib'))
    # #     #
    # #     self.scorer2.fit()
    # #     self.scorer2.measure_distance(method='Any')
    # #     path2 = os.path.join(out_dir, '1h1vA.fa')
    # #     etmipc2 = ETMIPC(path2)
    # #     start2 = time()
    # #     time2 = etmipc2.calculate_scores(curr_date=today, query='1h1vA', tree_depth=(2, 5),
    # #                                      out_dir=out_dir, processes=1, ignore_alignment_size=True,
    # #                                      clustering='agglomerative', clustering_args={'affinity': 'euclidean',
    # #                                                                                   'linkage': 'ward'},
    # #                                      aa_mapping=aa_dict, combine_clusters='sum', combine_branches='sum',
    # #                                      del_intermediate=False, low_mem=False)
    # #     end2 = time()
    # #     print(time2)
    # #     print(end2 - start2)
    # #     self.assertLessEqual(time2, end2 - start2)
    # #     for branch2 in etmipc2.tree_depth:
    # #         branch_dir = os.path.join(etmipc2.output_dir, str(branch2))
    # #         self.assertTrue(os.path.isdir(branch_dir))
    # #         score_path = os.path.join(branch_dir,
    # #                                   "{}_{}_{}.all_scores.txt".format(today, self.scorer2.query, branch2))
    # #         self.assertTrue(os.path.isfile(score_path))
    # #         test_df = pd.read_csv(score_path, index_col=None, delimiter='\t')
    # #         self.assertEqual(list(test_df.columns), headers[branch2])
    # #         self.score_comp_function(df=test_df, seq=self.scorer2.query_alignment.query_sequence,
    # #                                  clusters=etmipc2.get_cluster_scores(branch=branch2),
    # #                                  branches=etmipc2.get_branch_scores(branch=branch2),
    # #                                  scores=etmipc2.get_scores(branch=branch2),
    # #                                  coverages=etmipc2.get_coverage(branch=branch2))
    # #     for curr_pos, mapped_pos in etmipc2.cluster_mapping.items():
    # #         curr_path = os.path.join(etmipc2.output_dir, str(curr_pos[0]),
    # #                                  "{}_{}_{}.all_scores.txt".format(today, self.scorer2.query, curr_pos[0]))
    # #         mapped_path = os.path.join(etmipc2.output_dir, str(mapped_pos[0]),
    # #                                    "{}_{}_{}.all_scores.txt".format(today, self.scorer2.query, mapped_pos[0]))
    # #         curr_df = pd.read_csv(curr_path, index_col=None, delimiter='\t')
    # #         mapped_df = pd.read_csv(mapped_path, index_col=None, delimiter='\t')
    # #         self.score_comp_nonunique_cluster_files(df1=curr_df, df2=mapped_df, cluster1=curr_pos[1],
    # #                                                 cluster2=mapped_pos[1])
    # #     for branch2 in etmipc2.tree_depth:
    # #         rmtree(os.path.join(etmipc2.output_dir, str(branch2)))
    # #     os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
    # #     os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
    # #     os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
    # #     os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
    # #     os.remove(os.path.join(os.path.abspath('../Test/'), '{}_cET-MIp.npz'.format('1h1vA')))
    # #     os.remove(os.path.join(os.path.abspath('../Test/'), '{}_cET-MIp.pkl'.format('1h1vA')))
    # #     rmtree(os.path.join(out_dir, 'joblib'))
    #
    # # def test_heatmap_plot(self):
    # #     save_dir = os.path.abspath('../Test')
    # #     #
    # #     scores1 = np.random.rand(79, 79)
    # #     scores1[np.tril_indices(79, 1)] = 0
    # #     scores1 += scores1.T
    # #     heatmap_plot(name='Score 1 Heatmap Plot', data_mat=scores1, output_dir=save_dir)
    # #     expected_path1 = os.path.abspath(os.path.join(save_dir, 'Score_1_Heatmap_Plot.eps'))
    # #     print(expected_path1)
    # #     self.assertTrue(os.path.isfile(expected_path1))
    # #     os.remove(expected_path1)
    # #
    # # def test_surface_plot(self):
    # #     save_dir = os.path.abspath('../Test')
    # #     #
    # #     scores1 = np.random.rand(79, 79)
    # #     scores1[np.tril_indices(79, 1)] = 0
    # #     scores1 += scores1.T
    # #     surface_plot(name='Score 1 Surface Plot', data_mat=scores1, output_dir=save_dir)
    # #     expected_path1 = os.path.abspath(os.path.join(save_dir, 'Score_1_Surface_Plot.eps'))
    # #     print(expected_path1)
    # #     self.assertTrue(os.path.isfile(expected_path1))
    # #     os.remove(expected_path1)


if __name__ == '__main__':
    unittest.main()
