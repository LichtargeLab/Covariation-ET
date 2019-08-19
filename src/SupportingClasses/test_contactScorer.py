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
from unittest import TestCase
from scipy.stats import rankdata
from Bio.PDB.Polypeptide import one_to_three
from sklearn.metrics import auc, roc_curve, precision_score
from test_Base import TestBase
from ContactScorer import ContactScorer, surface_plot, heatmap_plot, plot_z_scores
sys.path.append(os.path.abspath('..'))
from EvolutionaryTrace import EvolutionaryTrace


class TestContactScorer(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestContactScorer, cls).setUpClass()
        cls.CONTACT_DISTANCE2 = 64
        cls.query1 = cls.small_structure_id
        cls.aln_file1 = cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln']
        cls.pdb_file1 = cls.data_set.protein_data[cls.small_structure_id]['PDB']
        cls.pdb_chain1 = cls.data_set.protein_data[cls.small_structure_id]['Chain']
        cls.pdb_len1 = cls.data_set.protein_data[cls.small_structure_id]['Length']
        cls.query2 = cls.large_structure_id
        cls.aln_file2 = cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln']
        cls.pdb_file2 = cls.data_set.protein_data[cls.large_structure_id]['PDB']
        cls.pdb_chain2 = cls.data_set.protein_data[cls.large_structure_id]['Chain']
        cls.pdb_len2 = cls.data_set.protein_data[cls.large_structure_id]['Length']

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
    def check_precision(mapped_scores, mapped_dists, count=None):
        if count is None:
            count = mapped_dists.shape[0]
        ranked_scores = rankdata(-1 * np.array(mapped_scores), method='dense')
        ind = np.where(ranked_scores <= count)
        mapped_scores = mapped_scores[ind]
        preds = (mapped_scores > 0.0) * 1.0
        mapped_dists = mapped_dists[ind]
        truth = (mapped_dists <= 8.0) * 1.0
        precision = precision_score(truth, preds)
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
                    if self._et_calcDist(ResAtoms[resi], ResAtoms[resj]) < self.CONTACT_DISTANCE2:
                        try:
                            A[resi][resj] = 1
                        except KeyError:
                            A[resi] = {resj: 1}
        return A, ResAtoms

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
            return 'NA', w, w_ave, w2_ave, sigma
        return (w - w_ave) / sigma, w, w_ave, w2_ave, sigma

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

    def test_1a___init(self):
        with self.assertRaises(TypeError):
            ContactScorer()
        self.assertEqual(self.scorer1.query_alignment, os.path.abspath(self.aln_file1))
        self.assertEqual(self.scorer1.query_structure, os.path.abspath(self.pdb_file1))
        self.assertEqual(self.scorer1.cutoff, 8.0)
        self.assertEqual(self.scorer1.best_chain, self.pdb_chain1)
        self.assertIsNone(self.scorer1.query_pdb_mapping)
        self.assertIsNone(self.scorer1._specific_mapping)
        self.assertIsNone(self.scorer1.distances)
        self.assertIsNone(self.scorer1.dist_type)

    def test_1b___init(self):
        with self.assertRaises(TypeError):
            ContactScorer()
        self.assertEqual(self.scorer2.query_alignment, os.path.abspath(self.aln_file2))
        self.assertEqual(self.scorer2.query_structure, os.path.abspath(self.pdb_file2))
        self.assertEqual(self.scorer2.cutoff, 8.0)
        self.assertEqual(self.scorer2.best_chain, self.pdb_chain2)
        self.assertIsNone(self.scorer2.query_pdb_mapping)
        self.assertIsNone(self.scorer2._specific_mapping)
        self.assertIsNone(self.scorer2.distances)
        self.assertIsNone(self.scorer2.dist_type)

    def test_1c___init(self):
        with self.assertRaises(TypeError):
            ContactScorer()
        eval1 = ContactScorer(query=self.query1, seq_alignment=self.aln_file1, pdb_reference=self.pdb_file1, cutoff=8.0)
        self.assertEqual(eval1.query_alignment, os.path.abspath(self.aln_file1))
        self.assertEqual(eval1.query_structure, os.path.abspath(self.pdb_file1))
        self.assertEqual(eval1.cutoff, 8.0)
        self.assertIsNone(eval1.best_chain)
        self.assertIsNone(eval1.query_pdb_mapping)
        self.assertIsNone(eval1._specific_mapping)
        self.assertIsNone(eval1.distances)
        self.assertIsNone(eval1.dist_type)

    def test_1d___init(self):
        with self.assertRaises(TypeError):
            ContactScorer()
        eval2 = ContactScorer(query=self.query2, seq_alignment=self.aln_file2, pdb_reference=self.pdb_file2, cutoff=8.0)
        self.assertEqual(eval2.query_alignment, os.path.abspath(self.aln_file2))
        self.assertEqual(eval2.query_structure, os.path.abspath(self.pdb_file2))
        self.assertEqual(eval2.cutoff, 8.0)
        self.assertIsNone(eval2.best_chain)
        self.assertIsNone(eval2.query_pdb_mapping)
        self.assertIsNone(eval2._specific_mapping)
        self.assertIsNone(eval2.distances)
        self.assertIsNone(eval2.dist_type)

    def test_2a___str(self):
        with self.assertRaises(ValueError):
            str(self.scorer1)
        self.scorer1.fit()
        expected_str1 = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            self.pdb_len1, 3, self.pdb_chain1)
        self.assertEqual(str(self.scorer1), expected_str1)

    def test_2b___str(self):
        with self.assertRaises(ValueError):
            str(self.scorer2)
        self.scorer2.fit()
        expected_str2 = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            self.pdb_len2, 3, self.pdb_chain2)
        self.assertEqual(str(self.scorer2), expected_str2)

    def test_2c___str(self):
        eval1 = ContactScorer(query=self.query1, seq_alignment=self.aln_file1, pdb_reference=self.pdb_file1, cutoff=8.0)
        with self.assertRaises(ValueError):
            str(eval1)
        eval1.fit()
        expected_str1 = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            self.pdb_len1, 3, self.pdb_chain1)
        self.assertEqual(str(eval1), expected_str1)

    def test_2d___str(self):
        eval2 = ContactScorer(query=self.query2, seq_alignment=self.aln_file2, pdb_reference=self.pdb_file2, cutoff=8.0)
        with self.assertRaises(ValueError):
            str(eval2)
        eval2.fit()
        expected_str2 = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            self.pdb_len2, 3, self.pdb_chain2)
        self.assertEqual(str(eval2), expected_str2)

    def test_3a_fit(self):
        self.assertEqual(self.scorer1.query_alignment, os.path.abspath(self.aln_file1))
        self.assertEqual(self.scorer1.query_structure, os.path.abspath(self.pdb_file1))
        self.scorer1.fit()
        self.assertNotEqual(self.scorer1.query_alignment, self.aln_file1)
        self.assertNotEqual(self.scorer1.query_structure, self.pdb_file1)
        self.assertEqual(self.scorer1.best_chain, 'A')
        self.assertEqual(self.scorer1.query_pdb_mapping, {i: i for i in range(self.pdb_len1)})
        self.scorer1.best_chain = None
        self.scorer1.fit()
        self.assertNotEqual(self.scorer1.query_alignment, self.aln_file1)
        self.assertNotEqual(self.scorer1.query_structure, self.pdb_file1)
        self.assertEqual(self.scorer1.best_chain, 'A')
        self.assertEqual(self.scorer1.query_pdb_mapping, {i: i for i in range(self.pdb_len1)})
        self.scorer1.query_pdb_mapping = None
        self.scorer1.fit()
        self.assertNotEqual(self.scorer1.query_alignment, self.aln_file1)
        self.assertNotEqual(self.scorer1.query_structure, self.pdb_file1)
        self.assertEqual(self.scorer1.best_chain, 'A')
        self.assertEqual(self.scorer1.query_pdb_mapping, {i: i for i in range(self.pdb_len1)})

    def test_3b_fit(self):
        self.assertEqual(self.scorer2.query_alignment, os.path.abspath(self.aln_file2))
        self.assertEqual(self.scorer2.query_structure, os.path.abspath(self.pdb_file2))
        self.scorer2.fit()
        self.assertNotEqual(self.scorer2.query_alignment, self.aln_file2)
        self.assertNotEqual(self.scorer2.query_structure, self.pdb_file2)
        self.assertEqual(self.scorer2.best_chain, 'A')
        self.assertEqual(self.scorer2.query_pdb_mapping, {i: i for i in range(self.pdb_len2)})
        self.scorer2.best_chain = None
        self.scorer2.fit()
        self.assertNotEqual(self.scorer2.query_alignment, self.aln_file2)
        self.assertNotEqual(self.scorer2.query_structure, self.pdb_file2)
        self.assertEqual(self.scorer2.best_chain, 'A')
        self.assertEqual(self.scorer2.query_pdb_mapping, {i: i for i in range(self.pdb_len2)})
        self.scorer2.query_pdb_mapping = None
        self.scorer2.fit()
        self.assertNotEqual(self.scorer2.query_alignment, self.aln_file2)
        self.assertNotEqual(self.scorer2.query_structure, self.pdb_file2)
        self.assertEqual(self.scorer2.best_chain, 'A')
        self.assertEqual(self.scorer2.query_pdb_mapping, {i: i for i in range(self.pdb_len2)})

    def test_3c_fit(self):
        eval1 = ContactScorer(query=self.query1, seq_alignment=self.aln_file1, pdb_reference=self.pdb_file1, cutoff=8.0)
        self.assertEqual(eval1.query_alignment, os.path.abspath(self.aln_file1))
        self.assertEqual(eval1.query_structure, os.path.abspath(self.pdb_file1))
        eval1.fit()
        self.assertNotEqual(eval1.query_alignment, self.aln_file1)
        self.assertNotEqual(eval1.query_structure, self.pdb_file1)
        self.assertEqual(eval1.best_chain, 'A')
        self.assertEqual(eval1.query_pdb_mapping, {i: i for i in range(self.pdb_len1)})
        eval1.best_chain = None
        eval1.fit()
        self.assertNotEqual(eval1.query_alignment, self.aln_file1)
        self.assertNotEqual(eval1.query_structure, self.pdb_file1)
        self.assertEqual(eval1.best_chain, 'A')
        self.assertEqual(eval1.query_pdb_mapping, {i: i for i in range(self.pdb_len1)})
        eval1.query_pdb_mapping = None
        eval1.fit()
        self.assertNotEqual(eval1.query_alignment, self.aln_file1)
        self.assertNotEqual(eval1.query_structure, self.pdb_file1)
        self.assertEqual(eval1.best_chain, 'A')
        self.assertEqual(eval1.query_pdb_mapping, {i: i for i in range(self.pdb_len1)})

    def test_3d_fit(self):
        eval2 = ContactScorer(query=self.query2, seq_alignment=self.aln_file2, pdb_reference=self.pdb_file2, cutoff=8.0)
        self.assertEqual(eval2.query_alignment, os.path.abspath(self.aln_file2))
        self.assertEqual(eval2.query_structure, os.path.abspath(self.pdb_file2))
        eval2.fit()
        self.assertNotEqual(eval2.query_alignment, self.aln_file2)
        self.assertNotEqual(eval2.query_structure, self.pdb_file2)
        self.assertEqual(eval2.best_chain, 'A')
        self.assertEqual(eval2.query_pdb_mapping, {i: i for i in range(self.pdb_len2)})
        eval2.best_chain = None
        eval2.fit()
        self.assertNotEqual(eval2.query_alignment, self.aln_file2)
        self.assertNotEqual(eval2.query_structure, self.pdb_file2)
        self.assertEqual(eval2.best_chain, 'A')
        self.assertEqual(eval2.query_pdb_mapping, {i: i for i in range(self.pdb_len2)})
        eval2.query_pdb_mapping = None
        eval2.fit()
        self.assertNotEqual(eval2.query_alignment, self.aln_file2)
        self.assertNotEqual(eval2.query_structure, self.pdb_file2)
        self.assertEqual(eval2.best_chain, 'A')
        self.assertEqual(eval2.query_pdb_mapping, {i: i for i in range(self.pdb_len2)})

    def test_4a__get_all_coords(self):
        self.scorer1.fit()
        residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
        expected1 = np.vstack([[-3.260, 7.392, 33.952], [-2.317, 6.655, 34.795], [-0.919, 6.658, 34.208],
                               [-0.802, 7.111, 33.058], [-2.897, 5.256, 34.804], [-4.336, 5.353, 34.377],
                               [-4.607, 6.783, 33.948]])
        measured1 = np.vstack(ContactScorer._get_all_coords(residue1))
        diff = measured1 - expected1
        not_passing = diff > 1E-5
        self.assertFalse(not_passing.any())

    def test_4b__get_all_coords(self):
        self.scorer2.fit()
        residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][32]
        expected2 = np.vstack([[156.841, 21.422, 49.694], [157.088, 22.877, 49.918], [156.067, 23.458, 50.893],
                               [156.426, 24.143, 51.855], [157.033, 23.649, 48.591], [157.983, 23.277, 47.444],
                               [157.455, 23.836, 46.128], [159.408, 23.761, 47.709]])
        measured2 = np.vstack(ContactScorer._get_all_coords(residue2))
        diff = measured2 - expected2
        not_passing = diff > 1E-5
        self.assertFalse(not_passing.any())

    def test_4c__get_c_alpha_coords(self):
        self.scorer1.fit()
        residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
        expected1 = np.vstack([[-2.317, 6.655, 34.795]])
        measured1 = np.vstack(ContactScorer._get_c_alpha_coords(residue1))
        diff = measured1 - expected1
        not_passing = diff > 1E-5
        self.assertFalse(not_passing.any())

    def test_4d__get_c_alpha_coords(self):
        self.scorer2.fit()
        residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][32]
        expected2 = np.vstack([[157.088, 22.877, 49.918]])
        measured2 = np.vstack(ContactScorer._get_c_alpha_coords(residue2))
        diff = measured2 - expected2
        not_passing = diff > 1E-5
        self.assertFalse(not_passing.any())

    def test_4e__get_c_beta_coords(self):
        self.scorer1.fit()
        residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
        expected1 = np.vstack([[-2.897, 5.256, 34.804]])
        measured1 = np.vstack(ContactScorer._get_c_beta_coords(residue1))
        diff = measured1 - expected1
        not_passing = diff > 1E-5
        self.assertFalse(not_passing.any())

    def test_4f__get_c_beta_coords(self):
        self.scorer2.fit()
        residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][32]
        expected2 = np.vstack([[157.033, 23.649, 48.591]])
        measured2 = np.vstack(ContactScorer._get_c_beta_coords(residue2))
        diff = measured2 - expected2
        not_passing = diff > 1E-5
        self.assertFalse(not_passing.any())

    def test_4g__get_coords(self):
        self.scorer1.fit()
        residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
        expected1a = np.vstack([[-3.260, 7.392, 33.952], [-2.317, 6.655, 34.795], [-0.919, 6.658, 34.208],
                               [-0.802, 7.111, 33.058], [-2.897, 5.256, 34.804], [-4.336, 5.353, 34.377],
                               [-4.607, 6.783, 33.948]])
        measured1a = np.vstack(ContactScorer._get_coords(residue1, method='Any'))
        self.assertFalse(((measured1a - expected1a) > 1E-5).any())
        expected1b = np.vstack([[-2.317, 6.655, 34.795]])
        measured1b = np.vstack(ContactScorer._get_coords(residue1, method='CA'))
        self.assertFalse(((measured1b - expected1b) > 1E-5).any())
        expected1c = np.vstack([[-2.897, 5.256, 34.804]])
        measured1c = np.vstack(ContactScorer._get_coords(residue1, method='CB'))
        self.assertFalse(((measured1c - expected1c) > 1E-5).any())

    def test_4h__get_coords(self):
        self.scorer2.fit()
        residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][32]
        expected2a = np.vstack([[156.841, 21.422, 49.694], [157.088, 22.877, 49.918], [156.067, 23.458, 50.893],
                               [156.426, 24.143, 51.855], [157.033, 23.649, 48.591], [157.983, 23.277, 47.444],
                               [157.455, 23.836, 46.128], [159.408, 23.761, 47.709]])
        measured2a = np.vstack(ContactScorer._get_coords(residue2, method='Any'))
        self.assertFalse(((measured2a - expected2a) > 1E-5).any())
        expected2b = np.vstack([[157.088, 22.877, 49.918]])
        measured2b = np.vstack(ContactScorer._get_c_alpha_coords(residue2))
        self.assertFalse(((measured2b - expected2b) > 1E-5).any())
        expected2c = np.vstack([[157.033, 23.649, 48.591]])
        measured2c = np.vstack(ContactScorer._get_c_beta_coords(residue2))
        self.assertFalse(((measured2c - expected2c) > 1E-5).any())

    # def test_measure_distance(self):
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
    #     self.assertLess(expected1a - self.scorer1.distances[0, 1], 1e-5)
    #     self.scorer1.measure_distance(method='CA')
    #     self.assertEqual(self.scorer1.dist_type, 'CA')
    #     expected1b = np.sqrt(np.power(31.313 - 31.462, 2) + np.power(-5.089 - -7.593, 2) + np.power(-7.616 - -4.746, 2))
    #     self.assertLess(expected1b - self.scorer1.distances[0, 1], 1e-5)
    #     self.scorer1.measure_distance(method='CB')
    #     self.assertEqual(self.scorer1.dist_type, 'CB')
    #     expected1c = np.sqrt(np.power(32.271 - 32.746, 2) + np.power(-5.871 - -8.085, 2) + np.power(-8.520 - -4.071, 2))
    #     self.assertLess(expected1c - self.scorer1.distances[0, 1], 1e-5)
    # 
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='Any')
    #     self.assertEqual(self.scorer2.dist_type, 'Any')
    #     residue2a = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][5]
    #     residue2b = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][6]
    #     pos2a = ContactScorer._get_all_coords(residue2a)
    #     pos2b = ContactScorer._get_all_coords(residue2b)
    #     expected2a = None
    #     for i in range(len(pos2a)):
    #         for j in range(len(pos2b)):
    #             curr_dist = np.sqrt(np.power(pos2a[i][0] - pos2b[j][0], 2) + np.power(pos2a[i][1] - pos2b[j][1], 2) +
    #                                 np.power(pos2a[i][2] - pos2b[j][2], 2))
    #             if (expected2a is None) or (curr_dist < expected2a):
    #                 expected2a = curr_dist
    #     print(expected2a)
    #     print(self.scorer2.distances[0, 1])
    #     self.assertLess(expected2a - self.scorer2.distances[0, 1], 1e-5)
    #     self.scorer2.measure_distance(method='CA')
    #     self.assertEqual(self.scorer2.dist_type, 'CA')
    #     expected2b = np.sqrt(np.power(33.929 - 31.582, 2) + np.power(20.460 - 22.092, 2) + np.power(39.036 - 41.510, 2))
    #     self.assertLess(expected2b - self.scorer2.distances[0, 1], 1e-5)
    #     self.scorer2.measure_distance(method='CB')
    #     self.assertEqual(self.scorer2.dist_type, 'CB')
    #     expected2c = np.sqrt(np.power(33.683 - 32.070, 2) + np.power(20.497 - 22.322, 2) + np.power(37.525 - 42.924, 2))
    #     self.assertLess(expected2c - self.scorer2.distances[0, 1], 1e-5)
    # 
    # def test_measure_distance_2(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='Any')
    #     self.assertEqual(self.scorer1.dist_type, 'Any')
    #     residue_coords = {}
    #     dists = np.zeros((79, 79))
    #     dists2 = np.zeros((79, 79))
    #     for residue in self.scorer1.query_structure.structure[0][self.scorer1.best_chain]:
    #         res_num = residue.id[1] - 1
    #         coords = self.scorer1._get_all_coords(residue)
    #         residue_coords[res_num] = coords
    #         for residue2 in residue_coords:
    #             if residue2 == res_num:
    #                 continue
    #             else:
    #                 dist = self._et_calcDist(coords, residue_coords[residue2])
    #                 dist2 = np.sqrt(dist)
    #                 dists[res_num, residue2] = dist
    #                 dists[residue2, res_num] = dist
    #                 dists2[res_num, residue2] = dist2
    #                 dists2[residue2, res_num] = dist2
    #     distance_diff = np.square(self.scorer1.distances) - dists
    #     # self.assertEqual(len(np.nonzero(distance_diff)[0]), 0)
    #     self.assertLess(np.max(distance_diff), 1e-3)
    #     adj_diff = ((np.square(self.scorer1.distances)[np.nonzero(distance_diff)] < 64) -
    #                 (dists[np.nonzero(distance_diff)] < 64))
    #     self.assertEqual(np.sum(adj_diff), 0)
    #     self.assertEqual(len(np.nonzero(adj_diff)[0]), 0)
    #     distance_diff2 = self.scorer1.distances - dists2
    #     self.assertEqual(np.sum(distance_diff2), 0.0)
    #     self.assertEqual(len(np.nonzero(distance_diff2)[0]), 0.0)
    # 
    # def test_find_pairs_by_separation(self):
    #     self.scorer1.fit()
    #     with self.assertRaises(ValueError):
    #         self.scorer1.find_pairs_by_separation(category='Wide')
    #     expected1 = {'Any': [], 'Neighbors': [], 'Short': [], 'Medium': [], 'Long': []}
    #     for i in range(79):
    #         for j in range(i + 1, 79):
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
    #     self.scorer2.fit()
    #     with self.assertRaises(ValueError):
    #         self.scorer2.find_pairs_by_separation(category='Small')
    #     expected2 = {'Any': [], 'Neighbors': [], 'Short': [], 'Medium': [], 'Long': []}
    #     for i in range(368):
    #         for j in range(i + 1, 368):
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
    # 
    # def test__map_predictions_to_pdb(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='CB')
    #     scores1 = np.random.rand(79, 79)
    #     scores1[np.tril_indices(79, 1)] = 0
    #     scores1 += scores1.T
    #     scores_mapped1a, dists_mapped1a = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Any')
    #     pairs1a = self.scorer1.find_pairs_by_separation(category='Any')
    #     expected_scores1a = scores1[[x[0] for x in pairs1a], [x[1] for x in pairs1a]]
    #     expected_dists1a = self.scorer1.distances[[x[0] for x in pairs1a], [x[1] for x in pairs1a]]
    #     self.assertLess(np.sum(expected_scores1a - scores_mapped1a), 1e-5)
    #     self.assertLess(np.sum(expected_dists1a - dists_mapped1a), 1e-5)
    #     scores_mapped1b, dists_mapped1b = self.scorer1._map_predictions_to_pdb(predictions=scores1,
    #                                                                            category='Neighbors')
    #     pairs1b = self.scorer1.find_pairs_by_separation(category='Neighbors')
    #     expected_scores1b = scores1[[x[0] for x in pairs1b], [x[1] for x in pairs1b]]
    #     expected_dists1b = self.scorer1.distances[[x[0] for x in pairs1b], [x[1] for x in pairs1b]]
    #     self.assertLess(np.sum(expected_scores1b - scores_mapped1b), 1e-5)
    #     self.assertLess(np.sum(expected_dists1b - dists_mapped1b), 1e-5)
    #     scores_mapped1c, dists_mapped1c = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Short')
    #     pairs1c = self.scorer1.find_pairs_by_separation(category='Short')
    #     expected_scores1c = scores1[[x[0] for x in pairs1c], [x[1] for x in pairs1c]]
    #     expected_dists1c = self.scorer1.distances[[x[0] for x in pairs1c], [x[1] for x in pairs1c]]
    #     self.assertLess(np.sum(expected_scores1c - scores_mapped1c), 1e-5)
    #     self.assertLess(np.sum(expected_dists1c - dists_mapped1c), 1e-5)
    #     scores_mapped1d, dists_mapped1d = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Medium')
    #     pairs1d = self.scorer1.find_pairs_by_separation(category='Medium')
    #     expected_scores1d = scores1[[x[0] for x in pairs1d], [x[1] for x in pairs1d]]
    #     expected_dists1d = self.scorer1.distances[[x[0] for x in pairs1d], [x[1] for x in pairs1d]]
    #     self.assertLess(np.sum(expected_scores1d - scores_mapped1d), 1e-5)
    #     self.assertLess(np.sum(expected_dists1d - dists_mapped1d), 1e-5)
    #     scores_mapped1e, dists_mapped1e = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Long')
    #     pairs1e = self.scorer1.find_pairs_by_separation(category='Long')
    #     expected_scores1e = scores1[[x[0] for x in pairs1e], [x[1] for x in pairs1e]]
    #     expected_dists1e = self.scorer1.distances[[x[0] for x in pairs1e], [x[1] for x in pairs1e]]
    #     self.assertLess(np.sum(expected_scores1e - scores_mapped1e), 1e-5)
    #     self.assertLess(np.sum(expected_dists1e - dists_mapped1e), 1e-5)
    #     #
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='CB')
    #     scores2 = np.random.rand(368, 368)
    #     scores2[np.tril_indices(368, 1)] = 0
    #     scores2 += scores2.T
    #     scores_mapped2a, dists_mapped2a = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Any')
    #     pairs2a = self.scorer2.find_pairs_by_separation(category='Any')
    #     expected_scores2a = scores2[[x[0] for x in pairs2a], [x[1] for x in pairs2a]]
    #     expected_dists2a = self.scorer2.distances[[x[0] for x in pairs2a], [x[1] for x in pairs2a]]
    #     self.assertLess(np.sum(expected_scores2a - scores_mapped2a), 1e-5)
    #     self.assertLess(np.sum(expected_dists2a - dists_mapped2a), 1e-5)
    #     scores_mapped2b, dists_mapped2b = self.scorer2._map_predictions_to_pdb(predictions=scores2,
    #                                                                            category='Neighbors')
    #     pairs2b = self.scorer2.find_pairs_by_separation(category='Neighbors')
    #     expected_scores2b = scores2[[x[0] for x in pairs2b], [x[1] for x in pairs2b]]
    #     expected_dists2b = self.scorer2.distances[[x[0] for x in pairs2b], [x[1] for x in pairs2b]]
    #     self.assertLess(np.sum(expected_scores2b - scores_mapped2b), 1e-5)
    #     self.assertLess(np.sum(expected_dists2b - dists_mapped2b), 1e-5)
    #     scores_mapped2c, dists_mapped2c = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Short')
    #     pairs2c = self.scorer2.find_pairs_by_separation(category='Short')
    #     expected_scores2c = scores2[[x[0] for x in pairs2c], [x[1] for x in pairs2c]]
    #     expected_dists2c = self.scorer2.distances[[x[0] for x in pairs2c], [x[1] for x in pairs2c]]
    #     self.assertLess(np.sum(expected_scores2c - scores_mapped2c), 1e-5)
    #     self.assertLess(np.sum(expected_dists2c - dists_mapped2c), 1e-5)
    #     scores_mapped2d, dists_mapped2d = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Medium')
    #     pairs2d = self.scorer2.find_pairs_by_separation(category='Medium')
    #     expected_scores2d = scores2[[x[0] for x in pairs2d], [x[1] for x in pairs2d]]
    #     expected_dists2d = self.scorer2.distances[[x[0] for x in pairs2d], [x[1] for x in pairs2d]]
    #     self.assertLess(np.sum(expected_scores2d - scores_mapped2d), 1e-5)
    #     self.assertLess(np.sum(expected_dists2d - dists_mapped2d), 1e-5)
    #     scores_mapped2e, dists_mapped2e = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Long')
    #     pairs2e = self.scorer2.find_pairs_by_separation(category='Long')
    #     expected_scores2e = scores2[[x[0] for x in pairs2e], [x[1] for x in pairs2e]]
    #     expected_dists2e = self.scorer2.distances[[x[0] for x in pairs2e], [x[1] for x in pairs2e]]
    #     self.assertLess(np.sum(expected_scores2e - scores_mapped2e), 1e-5)
    #     self.assertLess(np.sum(expected_dists2e - dists_mapped2e), 1e-5)
    # 
    # def test_score_auc(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='CB')
    #     scores1 = np.random.rand(79, 79)
    #     scores1[np.tril_indices(79, 1)] = 0
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
    #     #
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='CB')
    #     scores2 = np.random.rand(368, 368)
    #     scores2[np.tril_indices(368, 1)] = 0
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
    # 
    # def test_plot_auc(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='CB')
    #     scores1 = np.random.rand(79, 79)
    #     scores1[np.tril_indices(79, 1)] = 0
    #     scores1 += scores1.T
    #     auroc1a = self.scorer1.score_auc(scores1, category='Any')
    #     self.scorer1.plot_auc(auc_data=auroc1a, title='1c17A AUROC for All Pairs',
    #                           file_name='1c17A_Any_AUROC', output_dir=os.path.abspath('../Test'))
    #     expected_path1 = os.path.abspath(os.path.join('../Test', '1c17A_Any_AUROC.eps'))
    #     print(expected_path1)
    #     self.assertTrue(os.path.isfile(expected_path1))
    #     os.remove(expected_path1)
    #     #
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='CB')
    #     scores2 = np.random.rand(368, 368)
    #     scores2[np.tril_indices(368, 1)] = 0
    #     scores2 += scores2.T
    #     auroc2a = self.scorer2.score_auc(scores2, category='Any')
    #     self.scorer2.plot_auc(auc_data=auroc2a, title='1h1vA AUROC for All Pairs',
    #                           file_name='1h1vA_Any_AUROC', output_dir=os.path.abspath('../Test'))
    #     expected_path2 = os.path.abspath(os.path.join('../Test', '1h1vA_Any_AUROC.eps'))
    #     self.assertTrue(os.path.isfile(expected_path2))
    #     os.remove(expected_path2)
    # 
    # def test_score_precision(self):
    #     self.scorer1.fit()
    #     self.scorer1.measure_distance(method='CB')
    #     scores1 = np.random.rand(79, 79)
    #     scores1[np.tril_indices(79, 1)] = 0
    #     scores1 += scores1.T
    #     scores_mapped1a, dists_mapped1a = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Any')
    #     expected_precision1a_all = self.check_precision(mapped_scores=scores_mapped1a, mapped_dists=dists_mapped1a)
    #     precision1a_all = self.scorer1.score_precision(predictions=scores1, category='Any')
    #     print(expected_precision1a_all)
    #     print(precision1a_all)
    #     self.assertEqual(expected_precision1a_all, precision1a_all)
    #     expected_precision1a_k10 = self.check_precision(mapped_scores=scores_mapped1a, mapped_dists=dists_mapped1a,
    #                                                count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     precision1a_k10 = self.scorer1.score_precision(predictions=scores1, k=10, category='Any')
    #     self.assertEqual(expected_precision1a_k10, precision1a_k10)
    #     expected_precision1a_n10 = self.check_precision(mapped_scores=scores_mapped1a, mapped_dists=dists_mapped1a,
    #                                                count=10.0)
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
    #                                                count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     precision1b_k10 = self.scorer1.score_precision(predictions=scores1, k=10, category='Neighbors')
    #     self.assertEqual(expected_precision1b_k10, precision1b_k10)
    #     expected_precision1b_n10 = self.check_precision(mapped_scores=scores_mapped1b, mapped_dists=dists_mapped1b,
    #                                                count=10.0)
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
    #                                                count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     self.assertEqual(expected_precision1c_k10, precision1c_k10)
    #     expected_precision1c_n10 = self.check_precision(mapped_scores=scores_mapped1c, mapped_dists=dists_mapped1c,
    #                                                count=10.0)
    #     precision1c_n10 = self.scorer1.score_precision(predictions=scores1, n=10, category='Short')
    #     self.assertEqual(expected_precision1c_n10, precision1c_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_precision(predictions=scores1, k=10, n=10, category='Short')
    #     scores_mapped1d, dists_mapped1d = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Medium')
    #     expected_precision1d_all = self.check_precision(mapped_scores=scores_mapped1d, mapped_dists=dists_mapped1d)
    #     precision1d_all = self.scorer1.score_precision(predictions=scores1, category='Medium')
    #     self.assertEqual(expected_precision1d_all, precision1d_all)
    #     expected_precision1d_k10 = self.check_precision(mapped_scores=scores_mapped1d, mapped_dists=dists_mapped1d,
    #                                                count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     precision1d_k10 = self.scorer1.score_precision(predictions=scores1, k=10, category='Medium')
    #     self.assertEqual(expected_precision1d_k10, precision1d_k10)
    #     expected_precision1d_n10 = self.check_precision(mapped_scores=scores_mapped1d, mapped_dists=dists_mapped1d,
    #                                                count=10.0)
    #     precision1d_n10 = self.scorer1.score_precision(predictions=scores1, n=10, category='Medium')
    #     self.assertEqual(expected_precision1d_n10, precision1d_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_precision(predictions=scores1, k=10, n=10, category='Medium')
    #     scores_mapped1e, dists_mapped1e = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Long')
    #     expected_precision1e_all = self.check_precision(mapped_scores=scores_mapped1e, mapped_dists=dists_mapped1e)
    #     precision1e_all = self.scorer1.score_precision(predictions=scores1, category='Long')
    #     self.assertEqual(expected_precision1e_all, precision1e_all)
    #     expected_precision1e_k10 = self.check_precision(mapped_scores=scores_mapped1e, mapped_dists=dists_mapped1e,
    #                                                count=floor(self.scorer1.query_alignment.seq_length / 10.0))
    #     precision1e_k10 = self.scorer1.score_precision(predictions=scores1, k=10, category='Long')
    #     self.assertEqual(expected_precision1e_k10, precision1e_k10)
    #     expected_precision1e_n10 = self.check_precision(mapped_scores=scores_mapped1e, mapped_dists=dists_mapped1e,
    #                                                count=10.0)
    #     precision1e_n10 = self.scorer1.score_precision(predictions=scores1, n=10, category='Long')
    #     self.assertEqual(expected_precision1e_n10, precision1e_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer1.score_precision(predictions=scores1, k=10, n=10, category='Long')
    #     #
    #     self.scorer2.fit()
    #     self.scorer2.measure_distance(method='CB')
    #     scores2 = np.random.rand(368, 368)
    #     scores2[np.tril_indices(368, 1)] = 0
    #     scores2 += scores2.T
    #     scores_mapped2a, dists_mapped2a = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Any')
    #     expected_precision2a_all = self.check_precision(mapped_scores=scores_mapped2a, mapped_dists=dists_mapped2a)
    #     precision2a_all = self.scorer2.score_precision(predictions=scores2, category='Any')
    #     self.assertEqual(expected_precision2a_all, precision2a_all)
    #     expected_precision2a_k10 = self.check_precision(mapped_scores=scores_mapped2a, mapped_dists=dists_mapped2a,
    #                                                count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     precision2a_k10 = self.scorer2.score_precision(predictions=scores2, k=10, category='Any')
    #     self.assertEqual(expected_precision2a_k10, precision2a_k10)
    #     expected_precision2a_n10 = self.check_precision(mapped_scores=scores_mapped2a, mapped_dists=dists_mapped2a,
    #                                                count=10.0)
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
    #                                                count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     precision2b_k10 = self.scorer2.score_precision(predictions=scores2, k=10, category='Neighbors')
    #     self.assertEqual(expected_precision2b_k10, precision2b_k10)
    #     expected_precision2b_n10 = self.check_precision(mapped_scores=scores_mapped2b, mapped_dists=dists_mapped2b,
    #                                                count=10.0)
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
    #                                                count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     self.assertEqual(expected_precision2c_k10, precision2c_k10)
    #     expected_precision2c_n10 = self.check_precision(mapped_scores=scores_mapped2c, mapped_dists=dists_mapped2c,
    #                                                count=10.0)
    #     precision2c_n10 = self.scorer2.score_precision(predictions=scores2, n=10, category='Short')
    #     self.assertEqual(expected_precision2c_n10, precision2c_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_precision(predictions=scores2, k=10, n=10, category='Short')
    #     scores_mapped2d, dists_mapped2d = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Medium')
    #     expected_precision2d_all = self.check_precision(mapped_scores=scores_mapped2d, mapped_dists=dists_mapped2d)
    #     precision2d_all = self.scorer2.score_precision(predictions=scores2, category='Medium')
    #     self.assertEqual(expected_precision2d_all, precision2d_all)
    #     expected_precision2d_k10 = self.check_precision(mapped_scores=scores_mapped2d, mapped_dists=dists_mapped2d,
    #                                                count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     precision2d_k10 = self.scorer2.score_precision(predictions=scores2, k=10, category='Medium')
    #     self.assertEqual(expected_precision2d_k10, precision2d_k10)
    #     expected_precision2d_n10 = self.check_precision(mapped_scores=scores_mapped2d, mapped_dists=dists_mapped2d,
    #                                                count=10.0)
    #     precision2d_n10 = self.scorer2.score_precision(predictions=scores2, n=10, category='Medium')
    #     self.assertEqual(expected_precision2d_n10, precision2d_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_precision(predictions=scores2, k=10, n=10, category='Medium')
    #     scores_mapped2e, dists_mapped2e = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Long')
    #     expected_precision2e_all = self.check_precision(mapped_scores=scores_mapped2e, mapped_dists=dists_mapped2e)
    #     precision2e_all = self.scorer2.score_precision(predictions=scores2, category='Long')
    #     self.assertEqual(expected_precision2e_all, precision2e_all)
    #     expected_precision2e_k10 = self.check_precision(mapped_scores=scores_mapped2e, mapped_dists=dists_mapped2e,
    #                                                count=floor(self.scorer2.query_alignment.seq_length / 10.0))
    #     precision2e_k10 = self.scorer2.score_precision(predictions=scores2, k=10, category='Long')
    #     self.assertEqual(expected_precision2e_k10, precision2e_k10)
    #     expected_precision2e_n10 = self.check_precision(mapped_scores=scores_mapped2e, mapped_dists=dists_mapped2e,
    #                                                count=10.0)
    #     precision2e_n10 = self.scorer2.score_precision(predictions=scores2, n=10, category='Long')
    #     self.assertEqual(expected_precision2e_n10, precision2e_n10)
    #     with self.assertRaises(ValueError):
    #         self.scorer2.score_precision(predictions=scores2, k=10, n=10, category='Long')
    # 
    # def test_score_clustering_of_contact_predictions(self):
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
