import os
import math
import numpy as np
import pandas as pd
from math import floor
from unittest import TestCase
from scipy.stats import rankdata
from sklearn.metrics import auc, roc_curve, precision_score
from SeqAlignment import SeqAlignment
from PDBReference import PDBReference
from ContactScorer import ContactScorer


class TestContactScorer(TestCase):

    def setUp(self):
        self.query1 = '1c17A'
        self.aln_file1 = '../Test/1c17A.fa'
        # self.aln_obj1 = SeqAlignment(file_name=self.aln_file1, query_id=self.query1)
        self.pdb_file1 = '../Test/query_1c17A.pdb'
        # self.pdb_obj1 = PDBReference(pdb_file=self.pdb_file1)
        self.scorer1 = ContactScorer(query= self.query1, seq_alignment=self.aln_file1, pdb_reference=self.pdb_file1,
                                     cutoff=8.0)
        self.query2 = '1h1vA'
        self.aln_file2 = '../Test/1h1vA.fa'
        # self.aln_obj2 = SeqAlignment(file_name=self.aln_file2, query_id=self.query2)
        self.pdb_file2 = '../Test/query_1h1vA.pdb'
        # self.pdb_obj2 = PDBReference(pdb_file=self.pdb_file2)
        self.scorer2 = ContactScorer(query=self.query2, seq_alignment=self.aln_file2, pdb_reference=self.pdb_file2,
                                     cutoff=8.0)
        self.CONTACT_DISTANCE2 = 64

    def tearDown(self):
        del self.query1
        del self.aln_file1
        # del self.aln_obj1
        del self.pdb_file1
        # del self.pdb_obj1
        del self.scorer1
        del self.query2
        del self.aln_file2
        # del self.aln_obj2
        del self.pdb_file2
        # del self.pdb_obj2
        del self.scorer2
        del self.CONTACT_DISTANCE2

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
        #
        # print(A.keys())
        bias_mat = np.zeros((L, L))
        #
        if bias == 1:
            for resi in reslist:
                for resj in reslist:
                    if resi < resj:
                        try:
                            Aij = A[resi][resj]  # A(i,j)==1
                            # w += (resj - resi)
                            #
                            b_curr = (resj - resi)
                            # bias_mat[resi - 1, resj - 1] = b_curr
                            # bias_mat[resj - 1, resi - 1] = b_curr
                            bias_mat[resi, resj] = b_curr
                            bias_mat[resj, resi] = b_curr
                            w += b_curr
                            #
                        except KeyError:
                            pass
        elif bias == 0:
            for resi in reslist:
                for resj in reslist:
                    if resi < resj:
                        try:
                            Aij = A[resi][resj]  # A(i,j)==1
                            w += 1
                            #
                            # bias_mat[resi - 1, resj - 1] = 1
                            # bias_mat[resj - 1, resi - 1] = 1
                            bias_mat[resi, resj] = 1
                            bias_mat[resj, resi] = 1
                            #
                        except KeyError:
                            pass
        #
        np.savetxt('/home/daniel/Documents/git/ETMIP/src/Test/Rhonald_Bias_{}_Mat.csv'.format(len(reslist)), bias_mat,
                   delimiter='\t')
        #
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

    def _et_computeAdjacency(self, chain):
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
        count = 0
        count2 = 0
        for residue in chain:
            count += 1
            try:
                aa = three2one[residue.get_resname()]
                count2 += 1
            except KeyError:
                continue
            resi = residue.get_id()[1]
            for atom in residue:
                try:
                    # ResAtoms[int(atom.resi)].append(atom.coord)
                    ResAtoms[resi - 1].append(atom.coord)
                except KeyError:
                    # ResAtoms[int(atom.resi)] = [atom.coord]
                    ResAtoms[resi - 1] = [atom.coord]
        print('Residue Counts: {} and {}'.format(count, count2))
        A = {}
        A_recip = {}
        #
        a = np.zeros((len(ResAtoms), len(ResAtoms)))
        dists = np.zeros(a.shape)
        #
        count = 0
        count2 = 0
        count3 = 0
        for resi in ResAtoms.keys():
            count += 1
            for resj in ResAtoms.keys():
                count2 += 1
                if resi < resj:
                    count3 += 1
                    #
                    curr_dist = self._et_calcDist(ResAtoms[resi], ResAtoms[resj])
                    dists[resi, resj] = curr_dist
                    #
                    # if self._et_calcDist(ResAtoms[resi], ResAtoms[resj]) < self.CONTACT_DISTANCE2:
                    if curr_dist < self.CONTACT_DISTANCE2:
                        #
                        a[resi, resj] = 1
                        #
                        try:
                            A[resi][resj] = 1
                            A_recip[resj][resi] = 1
                        except KeyError:
                            A[resi] = {resj: 1}
                            A_recip[resj] = {resi: 1}
                if (resi == 0) and (resj == 1):
                    print('###########################################################################################')
                    print('resi < resj: {}'.format(resi < resj))
                    print('curr_dist: {}'.format(self._et_calcDist(ResAtoms[resi], ResAtoms[resj])))
                    print('curr_dist < self.CONTACT_DISTANCE2: {}'.format(
                        self._et_calcDist(ResAtoms[resi], ResAtoms[resj])< self.CONTACT_DISTANCE2))
                    print('A[resi][resj]: {}'.format(A[resi][resj]))
                    print('###########################################################################################')
        #
        print('Second set of residue count: {} and {} and {}'.format(count, count2, count3))
        np.savetxt('/home/daniel/Documents/git/ETMIP/src/Test/Rhonald_A_Mat.csv', a, delimiter='\t')
        np.savetxt('/home/daniel/Documents/git/ETMIP/src/Test/Rhonald_Dist_Mat.csv', dists, delimiter='\t')
        #
        return A, ResAtoms, A_recip

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

    def test__init__(self):
        with self.assertRaises(TypeError):
            ContactScorer()
        self.assertEqual(self.scorer1.query_alignment, os.path.abspath(self.aln_file1))
        self.assertEqual(self.scorer1.query_structure, os.path.abspath(self.pdb_file1))
        self.assertEqual(self.scorer1.cutoff, 8.0)
        self.assertIsNone(self.scorer1.best_chain)
        self.assertIsNone(self.scorer1.query_pdb_mapping)
        self.assertIsNone(self.scorer1._specific_mapping)
        self.assertIsNone(self.scorer1.distances)
        self.assertIsNone(self.scorer1.dist_type)
        self.assertEqual(self.scorer2.query_alignment, os.path.abspath(self.aln_file2))
        self.assertEqual(self.scorer2.query_structure, os.path.abspath(self.pdb_file2))
        self.assertEqual(self.scorer2.cutoff, 8.0)
        self.assertIsNone(self.scorer2.best_chain)
        self.assertIsNone(self.scorer2.query_pdb_mapping)
        self.assertIsNone(self.scorer2._specific_mapping)
        self.assertIsNone(self.scorer2.distances)
        self.assertIsNone(self.scorer2.dist_type)

    def test_fit(self):
        self.assertEqual(self.scorer1.query_alignment, os.path.abspath(self.aln_file1))
        self.assertEqual(self.scorer1.query_structure, os.path.abspath(self.pdb_file1))
        self.scorer1.fit()
        self.assertNotEqual(self.scorer1.query_alignment, self.aln_file1)
        self.assertNotEqual(self.scorer1.query_structure, self.pdb_file1)
        self.assertEqual(self.scorer1.best_chain, 'A')
        self.assertEqual(self.scorer1.query_pdb_mapping, {i: i for i in range(79)})
        self.scorer1.best_chain = None
        self.scorer1.fit()
        self.assertNotEqual(self.scorer1.query_alignment, self.aln_file1)
        self.assertNotEqual(self.scorer1.query_structure, self.pdb_file1)
        self.assertEqual(self.scorer1.best_chain, 'A')
        self.assertEqual(self.scorer1.query_pdb_mapping, {i: i for i in range(79)})
        self.scorer1.query_pdb_mapping = None
        self.scorer1.fit()
        self.assertNotEqual(self.scorer1.query_alignment, self.aln_file1)
        self.assertNotEqual(self.scorer1.query_structure, self.pdb_file1)
        self.assertEqual(self.scorer1.best_chain, 'A')
        self.assertEqual(self.scorer1.query_pdb_mapping, {i: i for i in range(79)})
        self.assertEqual(self.scorer2.query_alignment, os.path.abspath(self.aln_file2))
        self.assertEqual(self.scorer2.query_structure, os.path.abspath(self.pdb_file2))
        self.scorer2.fit()
        self.assertNotEqual(self.scorer2.query_alignment, self.aln_file2)
        self.assertNotEqual(self.scorer2.query_structure, self.pdb_file2)
        self.assertEqual(self.scorer2.best_chain, 'A')

        self.assertEqual(self.scorer2.query_pdb_mapping, {i: i for i in range(368)})
        self.scorer2.best_chain = None
        self.scorer2.fit()
        self.assertNotEqual(self.scorer2.query_alignment, self.aln_file2)
        self.assertNotEqual(self.scorer2.query_structure, self.pdb_file2)
        self.assertEqual(self.scorer2.best_chain, 'A')

        self.assertEqual(self.scorer2.query_pdb_mapping, {i: i for i in range(368)})
        self.scorer2.query_pdb_mapping = None
        self.scorer2.fit()
        self.assertNotEqual(self.scorer2.query_alignment, self.aln_file2)
        self.assertNotEqual(self.scorer2.query_structure, self.pdb_file2)
        self.assertEqual(self.scorer2.best_chain, 'A')

        self.assertEqual(self.scorer2.query_pdb_mapping, {i: i for i in range(368)})

    def test__str__(self):
        with self.assertRaises(ValueError):
            str(self.scorer1)
        self.scorer1.fit()
        self.assertEqual(str(self.scorer1),
                         'Query Sequence of Length: 79\nPDB with 1 Chains\nBest Sequence Match to Chain: A')
        with self.assertRaises(ValueError):
            str(self.scorer2)
        self.scorer2.fit()
        self.assertEqual(str(self.scorer2),
                         'Query Sequence of Length: 368\nPDB with 1 Chains\nBest Sequence Match to Chain: A')

    def test__get_all_coords(self):
        self.scorer1.fit()
        residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
        expected1 = np.vstack([[32.009, -3.867, -7.132], [31.313, -5.089, -7.616], [30.848, -5.958, -6.453],
                               [29.671, -5.946, -6.094], [32.271, -5.871, -8.520], [32.697, -5.107, -9.764],
                               [33.817, -6.055, -10.812], [35.183, -6.331, -9.687], [32.971, -4.141, -6.850],
                               [31.470, -3.497, -6.322], [32.029, -3.183, -7.914], [30.450, -4.786, -8.190],
                               [33.159, -6.118, -7.958], [31.788, -6.784, -8.833], [31.815, -4.858, -10.336],
                               [33.194, -4.198, -9.459], [36.108, -6.051, -10.169], [35.220, -7.375, -9.415],
                               [35.044, -5.732, -8.799]])
        measured1 = np.vstack(ContactScorer._get_all_coords(residue1))
        self.assertLess(np.sum(measured1 - expected1), 1e-5)
        self.scorer2.fit()
        residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][5]
        expected2 = np.vstack([[34.379, 19.087, 39.460], [33.929, 20.460, 39.036], [32.641, 20.827, 39.757],
                              [31.587, 20.266, 39.470], [33.683, 20.497, 37.525], [34.857, 20.070, 36.827],
                              [33.502, 21.908, 37.040]])
        measured2 = np.vstack(ContactScorer._get_all_coords(residue2))
        self.assertLess(np.sum(measured2 - expected2), 1e-5)

    def test__get_c_alpha_coords(self):
        self.scorer1.fit()
        residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
        expected1 = np.vstack([[31.313, -5.089, -7.616]])
        measured1 = np.vstack(ContactScorer._get_c_alpha_coords(residue1))
        self.assertLess(np.sum(measured1 - expected1), 1e-5)
        self.scorer2.fit()
        residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][5]
        expected2 = np.vstack([[33.929, 20.460, 39.036]])
        measured2 = np.vstack(ContactScorer._get_c_alpha_coords(residue2))
        self.assertLess(np.sum(measured2 - expected2), 1e-5)

    def test__get_c_beta_coords(self):
        self.scorer1.fit()
        residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
        expected1 = np.vstack([[32.271, -5.871, -8.520]])
        measured1 = np.vstack(ContactScorer._get_c_beta_coords(residue1))
        self.assertLess(np.sum(measured1 - expected1), 1e-5)
        self.scorer2.fit()
        residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][5]
        expected2 = np.vstack([[33.683, 20.497, 37.525]])
        measured2 = np.vstack(ContactScorer._get_c_beta_coords(residue2))
        self.assertLess(np.sum(measured2 - expected2), 1e-5)

    def test__get_coords(self):
        self.scorer1.fit()
        residue1 = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
        expected1a = np.vstack([[32.009, -3.867, -7.132], [31.313, -5.089, -7.616], [30.848, -5.958, -6.453],
                                [29.671, -5.946, -6.094], [32.271, -5.871, -8.520], [32.697, -5.107, -9.764],
                                [33.817, -6.055, -10.812], [35.183, -6.331, -9.687], [32.971, -4.141, -6.850],
                                [31.470, -3.497, -6.322], [32.029, -3.183, -7.914], [30.450, -4.786, -8.190],
                                [33.159, -6.118, -7.958], [31.788, -6.784, -8.833], [31.815, -4.858, -10.336],
                                [33.194, -4.198, -9.459], [36.108, -6.051, -10.169], [35.220, -7.375, -9.415],
                                [35.044, -5.732, -8.799]])
        measured1a = np.vstack(ContactScorer._get_coords(residue1, method='Any'))
        self.assertLess(np.sum(measured1a - expected1a), 1e-5)
        expected1b = np.vstack([[31.313, -5.089, -7.616]])
        measured1b = np.vstack(ContactScorer._get_coords(residue1, method='CA'))
        self.assertLess(np.sum(measured1b - expected1b), 1e-5)
        expected1c = np.vstack([[32.271, -5.871, -8.520]])
        measured1c = np.vstack(ContactScorer._get_coords(residue1, method='CB'))
        self.assertLess(np.sum(measured1c - expected1c), 1e-5)
        self.scorer2.fit()
        residue2 = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][5]
        expected2a = np.vstack([[34.379, 19.087, 39.460], [33.929, 20.460, 39.036], [32.641, 20.827, 39.757],
                                [31.587, 20.266, 39.470], [33.683, 20.497, 37.525], [34.857, 20.070, 36.827],
                                [33.502, 21.908, 37.040]])
        measured2a = np.vstack(ContactScorer._get_coords(residue2, method='Any'))
        self.assertLess(np.sum(measured2a - expected2a), 1e-5)
        expected2b = np.vstack([[33.929, 20.460, 39.036]])
        measured2b = np.vstack(ContactScorer._get_c_alpha_coords(residue2))
        self.assertLess(np.sum(measured2b - expected2b), 1e-5)
        expected2c = np.vstack([[33.683, 20.497, 37.525]])
        measured2c = np.vstack(ContactScorer._get_c_beta_coords(residue2))
        self.assertLess(np.sum(measured2c - expected2c), 1e-5)

    def test_measure_distance(self):
        self.scorer1.fit()
        self.scorer1.measure_distance(method='Any')
        self.assertEqual(self.scorer1.dist_type, 'Any')
        residue1a = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][1]
        residue1b = self.scorer1.query_structure.structure[0][self.scorer1.best_chain][2]
        pos1a = ContactScorer._get_all_coords(residue1a)
        pos1b = ContactScorer._get_all_coords(residue1b)
        expected1a = None
        for i in range(len(pos1a)):
            for j in range(len(pos1b)):
                curr_dist = np.sqrt(np.power(pos1a[i][0] - pos1b[j][0], 2) + np.power(pos1a[i][1] - pos1b[j][1], 2) +
                                    np.power(pos1a[i][2] - pos1b[j][2], 2))
                if (expected1a is None) or (curr_dist < expected1a):
                    expected1a = curr_dist
        self.assertLess(expected1a - self.scorer1.distances[0, 1], 1e-5)
        self.scorer1.measure_distance(method='CA')
        self.assertEqual(self.scorer1.dist_type, 'CA')
        expected1b = np.sqrt(np.power(31.313 - 31.462, 2) + np.power(-5.089 - -7.593, 2) + np.power(-7.616 - -4.746, 2))
        self.assertLess(expected1b - self.scorer1.distances[0, 1], 1e-5)
        self.scorer1.measure_distance(method='CB')
        self.assertEqual(self.scorer1.dist_type, 'CB')
        expected1c = np.sqrt(np.power(32.271 - 32.746, 2) + np.power(-5.871 - -8.085, 2) + np.power(-8.520 - -4.071, 2))
        self.assertLess(expected1c - self.scorer1.distances[0, 1], 1e-5)

        self.scorer2.fit()
        self.scorer2.measure_distance(method='Any')
        self.assertEqual(self.scorer2.dist_type, 'Any')
        residue2a = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][5]
        residue2b = self.scorer2.query_structure.structure[0][self.scorer2.best_chain][6]
        pos2a = ContactScorer._get_all_coords(residue2a)
        pos2b = ContactScorer._get_all_coords(residue2b)
        expected2a = None
        for i in range(len(pos2a)):
            for j in range(len(pos2b)):
                curr_dist = np.sqrt(np.power(pos2a[i][0] - pos2b[j][0], 2) + np.power(pos2a[i][1] - pos2b[j][1], 2) +
                                    np.power(pos2a[i][2] - pos2b[j][2], 2))
                if (expected2a is None) or (curr_dist < expected2a):
                    expected2a = curr_dist
        print(expected2a)
        print(self.scorer2.distances[0, 1])
        self.assertLess(expected2a - self.scorer2.distances[0, 1], 1e-5)
        self.scorer2.measure_distance(method='CA')
        self.assertEqual(self.scorer2.dist_type, 'CA')
        expected2b = np.sqrt(np.power(33.929 - 31.582, 2) + np.power(20.460 - 22.092, 2) + np.power(39.036 - 41.510, 2))
        self.assertLess(expected2b - self.scorer2.distances[0, 1], 1e-5)
        self.scorer2.measure_distance(method='CB')
        self.assertEqual(self.scorer2.dist_type, 'CB')
        expected2c = np.sqrt(np.power(33.683 - 32.070, 2) + np.power(20.497 - 22.322, 2) + np.power(37.525 - 42.924, 2))
        self.assertLess(expected2c - self.scorer2.distances[0, 1], 1e-5)

    def test_find_pairs_by_separation(self):
        self.scorer1.fit()
        with self.assertRaises(ValueError):
            self.scorer1.find_pairs_by_separation(category='Wide')
        expected1 = {'Any': [], 'Neighbors': [], 'Short': [], 'Medium': [], 'Long': []}
        for i in range(79):
            for j in range(i + 1, 79):
                pair = (i, j)
                separation = j - i
                if (separation >= 1) and (separation < 6):
                    expected1['Neighbors'].append(pair)
                if (separation >= 6) and (separation < 13):
                    expected1['Short'].append(pair)
                if (separation >= 13) and (separation < 24):
                    expected1['Medium'].append(pair)
                if separation >= 24:
                    expected1['Long'].append(pair)
                expected1['Any'].append(pair)
        self.assertEqual(self.scorer1.find_pairs_by_separation(category='Any'), expected1['Any'])
        self.assertEqual(self.scorer1.find_pairs_by_separation(category='Neighbors'), expected1['Neighbors'])
        self.assertEqual(self.scorer1.find_pairs_by_separation(category='Short'), expected1['Short'])
        self.assertEqual(self.scorer1.find_pairs_by_separation(category='Medium'), expected1['Medium'])
        self.assertEqual(self.scorer1.find_pairs_by_separation(category='Long'), expected1['Long'])
        self.scorer2.fit()
        with self.assertRaises(ValueError):
            self.scorer2.find_pairs_by_separation(category='Small')
        expected2 = {'Any': [], 'Neighbors': [], 'Short': [], 'Medium': [], 'Long': []}
        for i in range(368):
            for j in range(i + 1, 368):
                pair = (i, j)
                separation = j - i
                if (separation >= 1) and (separation < 6):
                    expected2['Neighbors'].append(pair)
                if (separation >= 6) and (separation < 13):
                    expected2['Short'].append(pair)
                if (separation >= 13) and (separation < 24):
                    expected2['Medium'].append(pair)
                if separation >= 24:
                    expected2['Long'].append(pair)
                expected2['Any'].append(pair)
        self.assertEqual(self.scorer2.find_pairs_by_separation(category='Any'), expected2['Any'])
        self.assertEqual(self.scorer2.find_pairs_by_separation(category='Neighbors'), expected2['Neighbors'])
        self.assertEqual(self.scorer2.find_pairs_by_separation(category='Short'), expected2['Short'])
        self.assertEqual(self.scorer2.find_pairs_by_separation(category='Medium'), expected2['Medium'])
        self.assertEqual(self.scorer2.find_pairs_by_separation(category='Long'), expected2['Long'])

    def test__map_predictions_to_pdb(self):
        self.scorer1.fit()
        self.scorer1.measure_distance(method='CB')
        scores1 = np.random.rand(79, 79)
        scores1[np.tril_indices(79, 1)] = 0
        scores1 += scores1.T
        scores_mapped1a, dists_mapped1a = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Any')
        pairs1a = self.scorer1.find_pairs_by_separation(category='Any')
        expected_scores1a = scores1[[x[0] for x in pairs1a], [x[1] for x in pairs1a]]
        expected_dists1a = self.scorer1.distances[[x[0] for x in pairs1a], [x[1] for x in pairs1a]]
        self.assertLess(np.sum(expected_scores1a - scores_mapped1a), 1e-5)
        self.assertLess(np.sum(expected_dists1a - dists_mapped1a), 1e-5)
        scores_mapped1b, dists_mapped1b = self.scorer1._map_predictions_to_pdb(predictions=scores1,
                                                                               category='Neighbors')
        pairs1b = self.scorer1.find_pairs_by_separation(category='Neighbors')
        expected_scores1b = scores1[[x[0] for x in pairs1b], [x[1] for x in pairs1b]]
        expected_dists1b = self.scorer1.distances[[x[0] for x in pairs1b], [x[1] for x in pairs1b]]
        self.assertLess(np.sum(expected_scores1b - scores_mapped1b), 1e-5)
        self.assertLess(np.sum(expected_dists1b - dists_mapped1b), 1e-5)
        scores_mapped1c, dists_mapped1c = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Short')
        pairs1c = self.scorer1.find_pairs_by_separation(category='Short')
        expected_scores1c = scores1[[x[0] for x in pairs1c], [x[1] for x in pairs1c]]
        expected_dists1c = self.scorer1.distances[[x[0] for x in pairs1c], [x[1] for x in pairs1c]]
        self.assertLess(np.sum(expected_scores1c - scores_mapped1c), 1e-5)
        self.assertLess(np.sum(expected_dists1c - dists_mapped1c), 1e-5)
        scores_mapped1d, dists_mapped1d = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Medium')
        pairs1d = self.scorer1.find_pairs_by_separation(category='Medium')
        expected_scores1d = scores1[[x[0] for x in pairs1d], [x[1] for x in pairs1d]]
        expected_dists1d = self.scorer1.distances[[x[0] for x in pairs1d], [x[1] for x in pairs1d]]
        self.assertLess(np.sum(expected_scores1d - scores_mapped1d), 1e-5)
        self.assertLess(np.sum(expected_dists1d - dists_mapped1d), 1e-5)
        scores_mapped1e, dists_mapped1e = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Long')
        pairs1e = self.scorer1.find_pairs_by_separation(category='Long')
        expected_scores1e = scores1[[x[0] for x in pairs1e], [x[1] for x in pairs1e]]
        expected_dists1e = self.scorer1.distances[[x[0] for x in pairs1e], [x[1] for x in pairs1e]]
        self.assertLess(np.sum(expected_scores1e - scores_mapped1e), 1e-5)
        self.assertLess(np.sum(expected_dists1e - dists_mapped1e), 1e-5)
        #
        self.scorer2.fit()
        self.scorer2.measure_distance(method='CB')
        scores2 = np.random.rand(368, 368)
        scores2[np.tril_indices(368, 1)] = 0
        scores2 += scores2.T
        scores_mapped2a, dists_mapped2a = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Any')
        pairs2a = self.scorer2.find_pairs_by_separation(category='Any')
        expected_scores2a = scores2[[x[0] for x in pairs2a], [x[1] for x in pairs2a]]
        expected_dists2a = self.scorer2.distances[[x[0] for x in pairs2a], [x[1] for x in pairs2a]]
        self.assertLess(np.sum(expected_scores2a - scores_mapped2a), 1e-5)
        self.assertLess(np.sum(expected_dists2a - dists_mapped2a), 1e-5)
        scores_mapped2b, dists_mapped2b = self.scorer2._map_predictions_to_pdb(predictions=scores2,
                                                                               category='Neighbors')
        pairs2b = self.scorer2.find_pairs_by_separation(category='Neighbors')
        expected_scores2b = scores2[[x[0] for x in pairs2b], [x[1] for x in pairs2b]]
        expected_dists2b = self.scorer2.distances[[x[0] for x in pairs2b], [x[1] for x in pairs2b]]
        self.assertLess(np.sum(expected_scores2b - scores_mapped2b), 1e-5)
        self.assertLess(np.sum(expected_dists2b - dists_mapped2b), 1e-5)
        scores_mapped2c, dists_mapped2c = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Short')
        pairs2c = self.scorer2.find_pairs_by_separation(category='Short')
        expected_scores2c = scores2[[x[0] for x in pairs2c], [x[1] for x in pairs2c]]
        expected_dists2c = self.scorer2.distances[[x[0] for x in pairs2c], [x[1] for x in pairs2c]]
        self.assertLess(np.sum(expected_scores2c - scores_mapped2c), 1e-5)
        self.assertLess(np.sum(expected_dists2c - dists_mapped2c), 1e-5)
        scores_mapped2d, dists_mapped2d = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Medium')
        pairs2d = self.scorer2.find_pairs_by_separation(category='Medium')
        expected_scores2d = scores2[[x[0] for x in pairs2d], [x[1] for x in pairs2d]]
        expected_dists2d = self.scorer2.distances[[x[0] for x in pairs2d], [x[1] for x in pairs2d]]
        self.assertLess(np.sum(expected_scores2d - scores_mapped2d), 1e-5)
        self.assertLess(np.sum(expected_dists2d - dists_mapped2d), 1e-5)
        scores_mapped2e, dists_mapped2e = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Long')
        pairs2e = self.scorer2.find_pairs_by_separation(category='Long')
        expected_scores2e = scores2[[x[0] for x in pairs2e], [x[1] for x in pairs2e]]
        expected_dists2e = self.scorer2.distances[[x[0] for x in pairs2e], [x[1] for x in pairs2e]]
        self.assertLess(np.sum(expected_scores2e - scores_mapped2e), 1e-5)
        self.assertLess(np.sum(expected_dists2e - dists_mapped2e), 1e-5)

    def test_score_auc(self):
        self.scorer1.fit()
        self.scorer1.measure_distance(method='CB')
        scores1 = np.random.rand(79, 79)
        scores1[np.tril_indices(79, 1)] = 0
        scores1 += scores1.T
        scores_mapped1a, dists_mapped1a = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Any')
        fpr_expected1a, tpr_expected1a, _ = roc_curve(dists_mapped1a <= 8.0, scores_mapped1a, pos_label=True)
        auroc_expected1a = auc(fpr_expected1a, tpr_expected1a)
        tpr1a, fpr1a, auroc1a = self.scorer1.score_auc(scores1, category='Any')
        self.assertEqual(np.sum(fpr_expected1a - fpr1a), 0)
        self.assertEqual(np.sum(tpr_expected1a - tpr1a), 0)
        self.assertEqual(auroc_expected1a, auroc1a)
        scores_mapped1b, dists_mapped1b = self.scorer1._map_predictions_to_pdb(predictions=scores1,
                                                                               category='Neighbors')
        fpr_expected1b, tpr_expected1b, _ = roc_curve(dists_mapped1b <= 8.0, scores_mapped1b, pos_label=True)
        auroc_expected1b = auc(fpr_expected1b, tpr_expected1b)
        tpr1b, fpr1b, auroc1b = self.scorer1.score_auc(scores1, category='Neighbors')
        self.assertEqual(np.sum(fpr_expected1b - fpr1b), 0)
        self.assertEqual(np.sum(tpr_expected1b - tpr1b), 0)
        self.assertEqual(auroc_expected1b, auroc1b)
        scores_mapped1c, dists_mapped1c = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Short')
        fpr_expected1c, tpr_expected1c, _ = roc_curve(dists_mapped1c <= 8.0, scores_mapped1c, pos_label=True)
        auroc_expected1c = auc(fpr_expected1c, tpr_expected1c)
        tpr1c, fpr1c, auroc1c = self.scorer1.score_auc(scores1, category='Short')
        self.assertEqual(np.sum(fpr_expected1c - fpr1c), 0)
        self.assertEqual(np.sum(tpr_expected1c - tpr1c), 0)
        self.assertEqual(auroc_expected1c, auroc1c)
        scores_mapped1d, dists_mapped1d = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Medium')
        fpr_expected1d, tpr_expected1d, _ = roc_curve(dists_mapped1d <= 8.0, scores_mapped1d, pos_label=True)
        auroc_expected1d = auc(fpr_expected1d, tpr_expected1d)
        tpr1d, fpr1d, auroc1d = self.scorer1.score_auc(scores1, category='Medium')
        self.assertEqual(np.sum(fpr_expected1d - fpr1d), 0)
        self.assertEqual(np.sum(tpr_expected1d - tpr1d), 0)
        self.assertEqual(auroc_expected1d, auroc1d)
        scores_mapped1e, dists_mapped1e = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Long')
        fpr_expected1e, tpr_expected1e, _ = roc_curve(dists_mapped1e <= 8.0, scores_mapped1e, pos_label=True)
        auroc_expected1e = auc(fpr_expected1e, tpr_expected1e)
        tpr1e, fpr1e, auroc1e = self.scorer1.score_auc(scores1, category='Long')
        self.assertEqual(np.sum(fpr_expected1e - fpr1e), 0)
        self.assertEqual(np.sum(tpr_expected1e - tpr1e), 0)
        self.assertEqual(auroc_expected1e, auroc1e)
        #
        self.scorer2.fit()
        self.scorer2.measure_distance(method='CB')
        scores2 = np.random.rand(368, 368)
        scores2[np.tril_indices(368, 1)] = 0
        scores2 += scores2.T
        scores_mapped2a, dists_mapped2a = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Any')
        fpr_expected2a, tpr_expected2a, _ = roc_curve(dists_mapped2a <= 8.0, scores_mapped2a, pos_label=True)
        auroc_expected2a = auc(fpr_expected2a, tpr_expected2a)
        tpr2a, fpr2a, auroc2a = self.scorer2.score_auc(scores2, category='Any')
        self.assertEqual(np.sum(fpr_expected2a - fpr2a), 0)
        self.assertEqual(np.sum(tpr_expected2a - tpr2a), 0)
        self.assertEqual(auroc_expected2a, auroc2a)
        scores_mapped2b, dists_mapped2b = self.scorer2._map_predictions_to_pdb(predictions=scores2,
                                                                               category='Neighbors')
        fpr_expected2b, tpr_expected2b, _ = roc_curve(dists_mapped2b <= 8.0, scores_mapped2b, pos_label=True)
        auroc_expected2b = auc(fpr_expected2b, tpr_expected2b)
        tpr2b, fpr2b, auroc2b = self.scorer2.score_auc(scores2, category='Neighbors')
        self.assertEqual(np.sum(fpr_expected2b - fpr2b), 0)
        self.assertEqual(np.sum(tpr_expected2b - tpr2b), 0)
        self.assertEqual(auroc_expected2b, auroc2b)
        scores_mapped2c, dists_mapped2c = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Short')
        fpr_expected2c, tpr_expected2c, _ = roc_curve(dists_mapped2c <= 8.0, scores_mapped2c, pos_label=True)
        auroc_expected2c = auc(fpr_expected2c, tpr_expected2c)
        tpr2c, fpr2c, auroc2c = self.scorer2.score_auc(scores2, category='Short')
        self.assertEqual(np.sum(fpr_expected2c - fpr2c), 0)
        self.assertEqual(np.sum(tpr_expected2c - tpr2c), 0)
        self.assertEqual(auroc_expected2c, auroc2c)
        scores_mapped2d, dists_mapped2d = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Medium')
        fpr_expected2d, tpr_expected2d, _ = roc_curve(dists_mapped2d <= 8.0, scores_mapped2d, pos_label=True)
        auroc_expected2d = auc(fpr_expected2d, tpr_expected2d)
        tpr2d, fpr2d, auroc2d = self.scorer2.score_auc(scores2, category='Medium')
        self.assertEqual(np.sum(fpr_expected2d - fpr2d), 0)
        self.assertEqual(np.sum(tpr_expected2d - tpr2d), 0)
        self.assertEqual(auroc_expected2d, auroc2d)
        scores_mapped2e, dists_mapped2e = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Long')
        fpr_expected2e, tpr_expected2e, _ = roc_curve(dists_mapped2e <= 8.0, scores_mapped2e, pos_label=True)
        auroc_expected2e = auc(fpr_expected2e, tpr_expected2e)
        tpr2e, fpr2e, auroc2e = self.scorer2.score_auc(scores2, category='Long')
        self.assertEqual(np.sum(fpr_expected2e - fpr2e), 0)
        self.assertEqual(np.sum(tpr_expected2e - tpr2e), 0)
        self.assertEqual(auroc_expected2e, auroc2e)


    def test_plot_auc(self):
        self.scorer1.fit()
        self.scorer1.measure_distance(method='CB')
        scores1 = np.random.rand(79, 79)
        scores1[np.tril_indices(79, 1)] = 0
        scores1 += scores1.T
        auroc1a = self.scorer1.score_auc(scores1, category='Any')
        self.scorer1.plot_auc(query_name='1c17A', auc_data=auroc1a, title='1c17A AUROC for All Pairs',
                              file_name='1c17A_Any_AUROC', output_dir=os.path.abspath('../Test'))
        expected_path1 = os.path.abspath(os.path.join('../Test', '1c17A_Any_AUROC.eps'))
        print(expected_path1)
        self.assertTrue(os.path.isfile(expected_path1))
        os.remove(expected_path1)
        #
        self.scorer2.fit()
        self.scorer2.measure_distance(method='CB')
        scores2 = np.random.rand(368, 368)
        scores2[np.tril_indices(368, 1)] = 0
        scores2 += scores2.T
        auroc2a = self.scorer2.score_auc(scores2, category='Any')
        self.scorer2.plot_auc(query_name='1h1vA', auc_data=auroc2a, title='1h1vA AUROC for All Pairs',
                              file_name='1h1vA_Any_AUROC', output_dir=os.path.abspath('../Test'))
        expected_path2 = os.path.abspath(os.path.join('../Test', '1h1vA_Any_AUROC.eps'))
        self.assertTrue(os.path.isfile(expected_path2))
        os.remove(expected_path2)

    def test_score_precision(self):
        self.scorer1.fit()
        self.scorer1.measure_distance(method='CB')
        scores1 = np.random.rand(79, 79)
        scores1[np.tril_indices(79, 1)] = 0
        scores1 += scores1.T
        scores_mapped1a, dists_mapped1a = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Any')
        expected_precision1a_all = self.check_precision(mapped_scores=scores_mapped1a, mapped_dists=dists_mapped1a)
        precision1a_all = self.scorer1.score_precision(predictions=scores1, category='Any')
        print(expected_precision1a_all)
        print(precision1a_all)
        self.assertEqual(expected_precision1a_all, precision1a_all)
        expected_precision1a_k10 = self.check_precision(mapped_scores=scores_mapped1a, mapped_dists=dists_mapped1a,
                                                   count=floor(self.scorer1.query_alignment.seq_length / 10.0))
        precision1a_k10 = self.scorer1.score_precision(predictions=scores1, k=10, category='Any')
        self.assertEqual(expected_precision1a_k10, precision1a_k10)
        expected_precision1a_n10 = self.check_precision(mapped_scores=scores_mapped1a, mapped_dists=dists_mapped1a,
                                                   count=10.0)
        precision1a_n10 = self.scorer1.score_precision(predictions=scores1, n=10, category='Any')
        self.assertEqual(expected_precision1a_n10, precision1a_n10)
        with self.assertRaises(ValueError):
            self.scorer1.score_precision(predictions=scores1, k=10, n=10, category='Any')
        scores_mapped1b, dists_mapped1b = self.scorer1._map_predictions_to_pdb(predictions=scores1,
                                                                               category='Neighbors')
        expected_precision1b_all = self.check_precision(mapped_scores=scores_mapped1b, mapped_dists=dists_mapped1b)
        precision1b_all = self.scorer1.score_precision(predictions=scores1, category='Neighbors')
        self.assertEqual(expected_precision1b_all, precision1b_all)
        expected_precision1b_k10 = self.check_precision(mapped_scores=scores_mapped1b, mapped_dists=dists_mapped1b,
                                                   count=floor(self.scorer1.query_alignment.seq_length / 10.0))
        precision1b_k10 = self.scorer1.score_precision(predictions=scores1, k=10, category='Neighbors')
        self.assertEqual(expected_precision1b_k10, precision1b_k10)
        expected_precision1b_n10 = self.check_precision(mapped_scores=scores_mapped1b, mapped_dists=dists_mapped1b,
                                                   count=10.0)
        precision1b_n10 = self.scorer1.score_precision(predictions=scores1, n=10, category='Neighbors')
        self.assertEqual(expected_precision1b_n10, precision1b_n10)
        with self.assertRaises(ValueError):
            self.scorer1.score_precision(predictions=scores1, k=10, n=10, category='Neighbors')
        scores_mapped1c, dists_mapped1c = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Short')
        expected_precision1c_all = self.check_precision(mapped_scores=scores_mapped1c, mapped_dists=dists_mapped1c)
        precision1c_all = self.scorer1.score_precision(predictions=scores1, category='Short')
        self.assertEqual(expected_precision1c_all, precision1c_all)
        precision1c_k10 = self.scorer1.score_precision(predictions=scores1, k=10, category='Short')
        expected_precision1c_k10 = self.check_precision(mapped_scores=scores_mapped1c, mapped_dists=dists_mapped1c,
                                                   count=floor(self.scorer1.query_alignment.seq_length / 10.0))
        self.assertEqual(expected_precision1c_k10, precision1c_k10)
        expected_precision1c_n10 = self.check_precision(mapped_scores=scores_mapped1c, mapped_dists=dists_mapped1c,
                                                   count=10.0)
        precision1c_n10 = self.scorer1.score_precision(predictions=scores1, n=10, category='Short')
        self.assertEqual(expected_precision1c_n10, precision1c_n10)
        with self.assertRaises(ValueError):
            self.scorer1.score_precision(predictions=scores1, k=10, n=10, category='Short')
        scores_mapped1d, dists_mapped1d = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Medium')
        expected_precision1d_all = self.check_precision(mapped_scores=scores_mapped1d, mapped_dists=dists_mapped1d)
        precision1d_all = self.scorer1.score_precision(predictions=scores1, category='Medium')
        self.assertEqual(expected_precision1d_all, precision1d_all)
        expected_precision1d_k10 = self.check_precision(mapped_scores=scores_mapped1d, mapped_dists=dists_mapped1d,
                                                   count=floor(self.scorer1.query_alignment.seq_length / 10.0))
        precision1d_k10 = self.scorer1.score_precision(predictions=scores1, k=10, category='Medium')
        self.assertEqual(expected_precision1d_k10, precision1d_k10)
        expected_precision1d_n10 = self.check_precision(mapped_scores=scores_mapped1d, mapped_dists=dists_mapped1d,
                                                   count=10.0)
        precision1d_n10 = self.scorer1.score_precision(predictions=scores1, n=10, category='Medium')
        self.assertEqual(expected_precision1d_n10, precision1d_n10)
        with self.assertRaises(ValueError):
            self.scorer1.score_precision(predictions=scores1, k=10, n=10, category='Medium')
        scores_mapped1e, dists_mapped1e = self.scorer1._map_predictions_to_pdb(predictions=scores1, category='Long')
        expected_precision1e_all = self.check_precision(mapped_scores=scores_mapped1e, mapped_dists=dists_mapped1e)
        precision1e_all = self.scorer1.score_precision(predictions=scores1, category='Long')
        self.assertEqual(expected_precision1e_all, precision1e_all)
        expected_precision1e_k10 = self.check_precision(mapped_scores=scores_mapped1e, mapped_dists=dists_mapped1e,
                                                   count=floor(self.scorer1.query_alignment.seq_length / 10.0))
        precision1e_k10 = self.scorer1.score_precision(predictions=scores1, k=10, category='Long')
        self.assertEqual(expected_precision1e_k10, precision1e_k10)
        expected_precision1e_n10 = self.check_precision(mapped_scores=scores_mapped1e, mapped_dists=dists_mapped1e,
                                                   count=10.0)
        precision1e_n10 = self.scorer1.score_precision(predictions=scores1, n=10, category='Long')
        self.assertEqual(expected_precision1e_n10, precision1e_n10)
        with self.assertRaises(ValueError):
            self.scorer1.score_precision(predictions=scores1, k=10, n=10, category='Long')
        #
        self.scorer2.fit()
        self.scorer2.measure_distance(method='CB')
        scores2 = np.random.rand(368, 368)
        scores2[np.tril_indices(368, 1)] = 0
        scores2 += scores2.T
        scores_mapped2a, dists_mapped2a = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Any')
        expected_precision2a_all = self.check_precision(mapped_scores=scores_mapped2a, mapped_dists=dists_mapped2a)
        precision2a_all = self.scorer2.score_precision(predictions=scores2, category='Any')
        self.assertEqual(expected_precision2a_all, precision2a_all)
        expected_precision2a_k10 = self.check_precision(mapped_scores=scores_mapped2a, mapped_dists=dists_mapped2a,
                                                   count=floor(self.scorer2.query_alignment.seq_length / 10.0))
        precision2a_k10 = self.scorer2.score_precision(predictions=scores2, k=10, category='Any')
        self.assertEqual(expected_precision2a_k10, precision2a_k10)
        expected_precision2a_n10 = self.check_precision(mapped_scores=scores_mapped2a, mapped_dists=dists_mapped2a,
                                                   count=10.0)
        precision2a_n10 = self.scorer2.score_precision(predictions=scores2, n=10, category='Any')
        self.assertEqual(expected_precision2a_n10, precision2a_n10)
        with self.assertRaises(ValueError):
            self.scorer2.score_precision(predictions=scores2, k=10, n=10, category='Any')
        scores_mapped2b, dists_mapped2b = self.scorer2._map_predictions_to_pdb(predictions=scores2,
                                                                               category='Neighbors')
        expected_precision2b_all = self.check_precision(mapped_scores=scores_mapped2b, mapped_dists=dists_mapped2b)
        precision2b_all = self.scorer2.score_precision(predictions=scores2, category='Neighbors')
        self.assertEqual(expected_precision2b_all, precision2b_all)
        expected_precision2b_k10 = self.check_precision(mapped_scores=scores_mapped2b, mapped_dists=dists_mapped2b,
                                                   count=floor(self.scorer2.query_alignment.seq_length / 10.0))
        precision2b_k10 = self.scorer2.score_precision(predictions=scores2, k=10, category='Neighbors')
        self.assertEqual(expected_precision2b_k10, precision2b_k10)
        expected_precision2b_n10 = self.check_precision(mapped_scores=scores_mapped2b, mapped_dists=dists_mapped2b,
                                                   count=10.0)
        precision2b_n10 = self.scorer2.score_precision(predictions=scores2, n=10, category='Neighbors')
        self.assertEqual(expected_precision2b_n10, precision2b_n10)
        with self.assertRaises(ValueError):
            self.scorer2.score_precision(predictions=scores2, k=10, n=10, category='Neighbors')
        scores_mapped2c, dists_mapped2c = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Short')
        expected_precision2c_all = self.check_precision(mapped_scores=scores_mapped2c, mapped_dists=dists_mapped2c)
        precision2c_all = self.scorer2.score_precision(predictions=scores2, category='Short')
        self.assertEqual(expected_precision2c_all, precision2c_all)
        precision2c_k10 = self.scorer2.score_precision(predictions=scores2, k=10, category='Short')
        expected_precision2c_k10 = self.check_precision(mapped_scores=scores_mapped2c, mapped_dists=dists_mapped2c,
                                                   count=floor(self.scorer2.query_alignment.seq_length / 10.0))
        self.assertEqual(expected_precision2c_k10, precision2c_k10)
        expected_precision2c_n10 = self.check_precision(mapped_scores=scores_mapped2c, mapped_dists=dists_mapped2c,
                                                   count=10.0)
        precision2c_n10 = self.scorer2.score_precision(predictions=scores2, n=10, category='Short')
        self.assertEqual(expected_precision2c_n10, precision2c_n10)
        with self.assertRaises(ValueError):
            self.scorer2.score_precision(predictions=scores2, k=10, n=10, category='Short')
        scores_mapped2d, dists_mapped2d = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Medium')
        expected_precision2d_all = self.check_precision(mapped_scores=scores_mapped2d, mapped_dists=dists_mapped2d)
        precision2d_all = self.scorer2.score_precision(predictions=scores2, category='Medium')
        self.assertEqual(expected_precision2d_all, precision2d_all)
        expected_precision2d_k10 = self.check_precision(mapped_scores=scores_mapped2d, mapped_dists=dists_mapped2d,
                                                   count=floor(self.scorer2.query_alignment.seq_length / 10.0))
        precision2d_k10 = self.scorer2.score_precision(predictions=scores2, k=10, category='Medium')
        self.assertEqual(expected_precision2d_k10, precision2d_k10)
        expected_precision2d_n10 = self.check_precision(mapped_scores=scores_mapped2d, mapped_dists=dists_mapped2d,
                                                   count=10.0)
        precision2d_n10 = self.scorer2.score_precision(predictions=scores2, n=10, category='Medium')
        self.assertEqual(expected_precision2d_n10, precision2d_n10)
        with self.assertRaises(ValueError):
            self.scorer2.score_precision(predictions=scores2, k=10, n=10, category='Medium')
        scores_mapped2e, dists_mapped2e = self.scorer2._map_predictions_to_pdb(predictions=scores2, category='Long')
        expected_precision2e_all = self.check_precision(mapped_scores=scores_mapped2e, mapped_dists=dists_mapped2e)
        precision2e_all = self.scorer2.score_precision(predictions=scores2, category='Long')
        self.assertEqual(expected_precision2e_all, precision2e_all)
        expected_precision2e_k10 = self.check_precision(mapped_scores=scores_mapped2e, mapped_dists=dists_mapped2e,
                                                   count=floor(self.scorer2.query_alignment.seq_length / 10.0))
        precision2e_k10 = self.scorer2.score_precision(predictions=scores2, k=10, category='Long')
        self.assertEqual(expected_precision2e_k10, precision2e_k10)
        expected_precision2e_n10 = self.check_precision(mapped_scores=scores_mapped2e, mapped_dists=dists_mapped2e,
                                                   count=10.0)
        precision2e_n10 = self.scorer2.score_precision(predictions=scores2, n=10, category='Long')
        self.assertEqual(expected_precision2e_n10, precision2e_n10)
        with self.assertRaises(ValueError):
            self.scorer2.score_precision(predictions=scores2, k=10, n=10, category='Long')

    def test_measure_distance_2(self):
        self.scorer1.fit()
        self.scorer1.measure_distance(method='Any')
        self.assertEqual(self.scorer1.dist_type, 'Any')
        residue_coords = {}
        dists = np.zeros((79, 79))
        dists2 = np.zeros((79, 79))
        for residue in self.scorer1.query_structure.structure[0][self.scorer1.best_chain]:
            res_num = residue.id[1] - 1
            coords = self.scorer1._get_all_coords(residue)
            residue_coords[res_num] = coords
            for residue2 in residue_coords:
                if residue2 == res_num:
                    continue
                else:
                    dist = self._et_calcDist(coords, residue_coords[residue2])
                    dist2 = np.sqrt(dist)
                    dists[res_num, residue2] = dist
                    dists[residue2, res_num] = dist
                    dists2[res_num, residue2] = dist2
                    dists2[residue2, res_num] = dist2
        distance_diff = np.square(self.scorer1.distances) - dists
        # self.assertEqual(len(np.nonzero(distance_diff)[0]), 0)
        self.assertLess(np.max(distance_diff), 1e-3)
        adj_diff = ((np.square(self.scorer1.distances)[np.nonzero(distance_diff)] < 64) -
                    (dists[np.nonzero(distance_diff)] < 64))
        self.assertEqual(np.sum(adj_diff), 0)
        self.assertEqual(len(np.nonzero(adj_diff)[0]), 0)
        distance_diff2 = self.scorer1.distances - dists2
        self.assertEqual(np.sum(distance_diff2), 0.0)
        self.assertEqual(len(np.nonzero(distance_diff2)[0]), 0.0)

    def test_adjacency_determination(self):
        self.scorer1.fit()
        self.scorer1.measure_distance(method='Any')
        scorer1_a = self.scorer1.distances < self.scorer1.cutoff
        scorer1_a[range(scorer1_a.shape[0]), range(scorer1_a.shape[1])] = 0
        A, _, _ = self._et_computeAdjacency(self.scorer1.query_structure.structure[0][self.scorer1.best_chain])
        scorer1_a_pos = np.nonzero(scorer1_a)
        scorer1_in_dict = 0
        scorer1_not_in_dict = 0
        scorer1_missed_pos = ([], [])
        for i in range(len(scorer1_a_pos[0])):
            if scorer1_a_pos[0][i] < scorer1_a_pos[1][i]:
                try:
                    scorer1_in_dict += A[scorer1_a_pos[0][i]][scorer1_a_pos[1][i]]
                except KeyError:
                    try:
                        scorer1_in_dict += A[scorer1_a_pos[1][i]][scorer1_a_pos[0][i]]
                    except KeyError:
                        scorer1_not_in_dict += 1
                        scorer1_missed_pos[0].append(scorer1_a_pos[0][i])
                        scorer1_missed_pos[1].append(scorer1_a_pos[1][i])
        rhonald1_in_dict = 0
        rhonald1_not_in_dict = 0
        for i in A.keys():
            for j in A[i].keys():
                if A[i][j] == scorer1_a[i, j]:
                    rhonald1_in_dict += 1
                else:
                    rhonald1_not_in_dict += 1
        print('ContactScorer Check - In Dict: {} |Not In Dict: {}'.format(scorer1_in_dict, scorer1_not_in_dict))
        print('Rhonald Check - In Dict: {} |Not In Dict: {}'.format(rhonald1_in_dict, rhonald1_not_in_dict))
        print(scorer1_a[scorer1_missed_pos])
        print(self.scorer1.distances[scorer1_missed_pos])
        #
        residue_coords = {}
        dists = np.zeros((79, 79))
        dists2 = np.zeros((79, 79))
        for residue in self.scorer1.query_structure.structure[0][self.scorer1.best_chain]:
            res_num = residue.id[1] - 1
            coords = self.scorer1._get_all_coords(residue)
            residue_coords[res_num] = coords
            for residue2 in residue_coords:
                if residue2 == res_num:
                    continue
                else:
                    dist = self._et_calcDist(coords, residue_coords[residue2])
                    dist2 = np.sqrt(dist)
                    dists[res_num, residue2] = dist
                    dists[residue2, res_num] = dist
                    dists2[res_num, residue2] = dist2
                    dists2[residue2, res_num] = dist2
        print(dists2[scorer1_missed_pos])
        #
        print(scorer1_missed_pos[0])
        print(scorer1_missed_pos[1])
        final_check = []
        for i in range(len(scorer1_missed_pos[0])):
            pos1 = scorer1_missed_pos[0][i]
            pos2 = scorer1_missed_pos[1][i]
            curr_check = False
            if pos1 in A:
                if pos2 in A[pos1]:
                    curr_check = True
            if pos2 in A:
                if pos1 in A[pos2]:
                    curr_check = True
            final_check.append(curr_check)
        print(final_check)
        self.assertEqual(scorer1_in_dict, rhonald1_in_dict)
        self.assertEqual(scorer1_not_in_dict, 0)
        self.assertEqual(rhonald1_not_in_dict, 0)

    def test_score_clustering_of_contact_predictions(self):
        def all_z_scores(A, L, bias, res_i, res_j, scores):
            # res_i = list(np.array(res_i) + 1)
            # res_j = list(np.array(res_j) + 1)
            data = {'Res_i': res_i, 'Res_j': res_j, 'Covariance_Score': scores, 'Z-Score': [], 'W': [], 'W_Ave': [],
                    'W2_Ave': [], 'Sigma': [], 'Num_Residues': []}
            res_list = []
            res_set = set()
            for i in range(len(scores)):
                curr_i = res_i[i]
                if curr_i not in res_set:
                    res_list.append(curr_i)
                    res_set.add(curr_i)
                curr_j = res_j[i]
                if curr_j not in res_set:
                    res_list.append(curr_j)
                    res_set.add(curr_j)
                z_score, w, w_ave, w2_ave, sigma = self._et_calcZScore(reslist=res_list, L=L, A=A, bias=bias)
                data['Z-Score'].append(z_score)
                data['W'].append(w)
                data['W_Ave'].append(w_ave)
                data['W2_Ave'].append(w2_ave)
                data['Sigma'].append(sigma)
                data['Num_Residues'].append(len(res_list))
            return pd.DataFrame(data)
        self.scorer1.fit()
        self.scorer1.measure_distance(method='Any')
        scores1 = np.random.rand(79, 79)
        scores1[np.tril_indices(79, 1)] = 0
        scores1 += scores1.T
        A, _, _ = self._et_computeAdjacency(self.scorer1.query_structure.structure[0][self.scorer1.best_chain])
        #
        scorer1_a = self.scorer1.distances < self.scorer1.cutoff
        scorer1_a[range(scorer1_a.shape[0]), range(scorer1_a.shape[0])] = 0
        scorer1_a = np.triu(scorer1_a, k=1)
        scorer1_a_pos = np.nonzero(scorer1_a)
        print(scorer1_a_pos)
        scorer1_a_size = len(scorer1_a_pos[0])
        print(A)
        expected1_a_size = np.sum([len(A[k]) for k in A.keys()])
        self.assertEqual(scorer1_a_size, expected1_a_size)
        content_comp = np.sum([A[scorer1_a_pos[0][i]][scorer1_a_pos[1][i]] for i in range(scorer1_a_size)])
        self.assertEqual(scorer1_a_size, content_comp)
        #
        zscore_df_1b, _ = self.scorer1.score_clustering_of_contact_predictions(
            predictions=scores1, bias=True, file_path=os.path.join(os.path.abspath('../Test/'), 'z_score.tsv'),
            w2_ave_sub=None)
        zscore_df_1be = all_z_scores(A=A, L=self.scorer1.query_alignment.seq_length, bias=1,
                                     res_i=list(zscore_df_1b['Res_i']), res_j=list(zscore_df_1b['Res_j']),
                                     scores=list(zscore_df_1b['Covariance_Score']))
        self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'z_score.tsv')))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'z_score.tsv'))
        zscore_df_1b.to_csv(os.path.join(os.path.abspath('../Test/'), 'ContactScorer_DF.csv'))
        zscore_df_1be.to_csv(os.path.join(os.path.abspath('../Test/'), 'Rhonald_DF.csv'))
        pd.testing.assert_frame_equal(zscore_df_1b, zscore_df_1be)
        zscore_df_1u, _ = self.scorer1.score_clustering_of_contact_predictions(
            predictions=scores1, bias=False, file_path=os.path.join(os.path.abspath('../Test/'), 'z_score.tsv'),
            w2_ave_sub=None)
        zscore_df_1ue = all_z_scores(A=A, L=self.scorer1.query_alignment.seq_length, bias=0,
                                     res_i=list(zscore_df_1u['Res_i']), res_j=list(zscore_df_1u['Res_j']),
                                     scores=list(zscore_df_1u['Covariance_Score']))
        self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'z_score.tsv')))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'z_score.tsv'))
        pd.testing.assert_frame_equal(zscore_df_1u, zscore_df_1ue)
        self.scorer2.fit()
        self.scorer2.measure_distance(method='Any')
        scores2 = np.random.rand(368, 368)
        scores2[np.tril_indices(368, 1)] = 0
        scores2 += scores2.T
        A, _, _ = self._et_computeAdjacency(self.scorer2.query_structure.structure[0][self.scorer2.best_chain])
        zscore_df_2b, _ = self.scorer2.score_clustering_of_contact_predictions(
            predictions=scores2, bias=True, file_path=os.path.join(os.path.abspath('../Test/'), 'z_score.tsv'),
            w2_ave_sub=None)
        zscore_df_2be = all_z_scores(A=A, L=self.scorer2.query_alignment.seq_length, bias=1,
                                     res_i=list(zscore_df_2b['Res_i']), res_j=list(zscore_df_2b['Res_j']),
                                     scores=list(zscore_df_2b['Covariance_Score']))
        self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'z_score.tsv')))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'z_score.tsv'))
        pd.testing.assert_frame_equal(zscore_df_2b, zscore_df_2be)
        zscore_df_2u, _ = self.scorer2.score_clustering_of_contact_predictions(
            predictions=scores2, bias=False, file_path=os.path.join(os.path.abspath('../Test/'), 'z_score.tsv'),
            w2_ave_sub=None)
        zscore_df_2ue = all_z_scores(A=A, L=self.scorer2.query_alignment.seq_length, bias=0,
                                     res_i=list(zscore_df_2b['Res_i']), res_j=list(zscore_df_2b['Res_j']),
                                     scores=list(zscore_df_2b['Covariance_Score']))
        self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'z_score.tsv')))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'z_score.tsv'))
        pd.testing.assert_frame_equal(zscore_df_2u, zscore_df_2ue)

    def test__clustering_z_score(self):
        self.scorer1.fit()
        self.scorer1.measure_distance(method='Any')
        A, _, _ = self._et_computeAdjacency(self.scorer1.query_structure.structure[0][self.scorer1.best_chain])
        z_scores_e1b, w_e1b, w_ave_e1b, w2_ave_e1b, sigma_e1b = self._et_calcZScore(
            reslist=self.scorer1.query_structure.pdb_residue_list, L=self.scorer1.query_alignment.seq_length, A=A,
            bias=1)
        z_scores_1b, w_1b, w_ave_1b, w2_ave_1b, sigma_1b, _ = self.scorer1._clustering_z_score(
            res_list=self.scorer1.query_structure.pdb_residue_list, bias=True, w2_ave_sub=None)
        if isinstance(z_scores_1b, str):
            self.assertEqual(z_scores_1b, '-')
            self.assertEqual(z_scores_e1b, 'NA')
            self.assertEqual(w_1b, None)
            self.assertEqual(w_e1b, 0)
            self.assertEqual(w_ave_1b, None)
            self.assertEqual(w_ave_e1b, 0)
            self.assertEqual(w2_ave_1b, None)
            self.assertEqual(w2_ave_e1b, 0)
            self.assertEqual(sigma_1b, None)
            self.assertEqual(sigma_e1b, 0)
        else:
            self.assertEqual(z_scores_1b, z_scores_e1b)
            self.assertEqual(w_1b, w_e1b)
            self.assertEqual(w_ave_1b, w_ave_e1b)
            self.assertEqual(w2_ave_1b, w2_ave_e1b)
            self.assertEqual(sigma_1b, sigma_e1b)
        z_scores_e1u, w_e1u, w_ave_e1u, w2_ave_e1u, sigma_e1u = self._et_calcZScore(
            reslist=self.scorer1.query_structure.pdb_residue_list, L=self.scorer1.query_alignment.seq_length, A=A,
            bias=0)
        z_scores_1u, w_1u, w_ave_1u, w2_ave_1u, sigma_1u, _ = self.scorer1._clustering_z_score(
            res_list=self.scorer1.query_structure.pdb_residue_list, bias=False, w2_ave_sub=None)
        if isinstance(z_scores_1u, str):
            self.assertEqual(z_scores_1u, '-')
            self.assertEqual(z_scores_e1u, 'NA')
            self.assertEqual(w_1u, None)
            self.assertEqual(w_e1u, 0)
            self.assertEqual(w_ave_1u, None)
            self.assertEqual(w_ave_e1u, 0)
            self.assertEqual(w2_ave_1u, None)
            self.assertEqual(w2_ave_e1u, 0)
            self.assertEqual(sigma_1u, None)
            self.assertEqual(sigma_e1u, 0)
        else:
            self.assertEqual(z_scores_1u, z_scores_e1u)
            self.assertEqual(w_1u, w_e1u)
            self.assertEqual(w_ave_1u, w_ave_e1u)
            self.assertEqual(w2_ave_1u, w2_ave_e1u)
            self.assertEqual(sigma_1u, sigma_e1u)
        self.scorer2.fit()
        self.scorer2.measure_distance(method='Any')
        A, _, _ = self._et_computeAdjacency(self.scorer2.query_structure.structure[0][self.scorer2.best_chain])
        z_scores_e2b, w_e2b, w_ave_e2b, w2_ave_e2b, sigma_e2b = self._et_calcZScore(
            reslist=self.scorer2.query_structure.pdb_residue_list, L=self.scorer2.query_alignment.seq_length, A=A,
            bias=1)
        z_scores_2b, w_2b, w_ave_2b, w2_ave_2b, sigma_2b, _ = self.scorer2._clustering_z_score(
            res_list=self.scorer2.query_structure.pdb_residue_list, bias=True, w2_ave_sub=None)
        if isinstance(z_scores_2b, str):
            self.assertEqual(z_scores_2b, '-')
            self.assertEqual(z_scores_e2b, 'NA')
            self.assertEqual(w_2b, None)
            self.assertEqual(w_e2b, 0)
            self.assertEqual(w_ave_2b, None)
            self.assertEqual(w_ave_e2b, 0)
            self.assertEqual(w2_ave_2b, None)
            self.assertEqual(w2_ave_e2b, 0)
            self.assertEqual(sigma_2b, None)
            self.assertEqual(sigma_e2b, 0)
        else:
            self.assertEqual(z_scores_2b, z_scores_e2b)
            self.assertEqual(w_2b, w_e2b)
            self.assertEqual(w_ave_2b, w_ave_e2b)
            self.assertEqual(w2_ave_2b, w2_ave_e2b)
            self.assertEqual(sigma_2b, sigma_e2b)
        z_scores_e2u, w_e2u, w_ave_e2u, w2_ave_e2u, sigma_e2u = self._et_calcZScore(
            reslist=self.scorer2.query_structure.pdb_residue_list, L=self.scorer2.query_alignment.seq_length, A=A,
            bias=0)
        z_scores_2u, w_2u, w_ave_2u, w2_ave_2u, sigma_2u, _ = self.scorer2._clustering_z_score(
            res_list=self.scorer2.query_structure.pdb_residue_list, bias=False, w2_ave_sub=None)
        if isinstance(z_scores_2u, str):
            self.assertEqual(z_scores_2u, '-')
            self.assertEqual(z_scores_e2u, 'NA')
            self.assertEqual(w_2u, None)
            self.assertEqual(w_e2u, 0)
            self.assertEqual(w_ave_2u, None)
            self.assertEqual(w_ave_e2u, 0)
            self.assertEqual(w2_ave_2u, None)
            self.assertEqual(w2_ave_e2u, 0)
            self.assertEqual(sigma_2u, None)
            self.assertEqual(sigma_e2u, 0)
        else:
            self.assertEqual(z_scores_2u, z_scores_e2u)
            self.assertEqual(w_2u, w_e2u)
            self.assertEqual(w_ave_2u, w_ave_e2u)
            self.assertEqual(w2_ave_2u, w2_ave_e2u)
            self.assertEqual(sigma_2u, sigma_e2u)


    # def test_write_out_clustering_results(self):
    #     self.fail()
    #
    # def test_evaluate_predictor(self):
    #     self.fail()
    #
    # def test_evaluate_predictions(self):
    #     self.fail()


import numpy as np


def check_adjacency(total_size):
    contact_adjacency = np.loadtxt('ContactScorer_A_Mat.csv', delimiter='\t')
    rhonald_adjacency = np.loadtxt('Rhonald_A_Mat.csv', delimiter='\t')
    diff_adjacency = contact_adjacency - rhonald_adjacency
    check = True
    for i in range(total_size):
        try:
            contact_adjacency_curr = np.loadtxt('Contact_Scorer_A_{}_Mat.csv'.format(i), delimiter='\t')
        except IOError:
            continue
        diff_adjacency2 = contact_adjacency - contact_adjacency_curr
        if len(np.nonzero(diff_adjacency2)) > 0:
            check = False
            break
    if np.sum(diff_adjacency) == 0 and len(np.nonzero(diff_adjacency)[0]) == 0 and check:
        return True
    else:
        return False


def check_bias(size, contact_bias=None):
    if contact_bias is None:
        contact_bias = np.loadtxt('ContactScorer_Bias_Mat.csv', delimiter='\t')
    contact_relevant_curr = np.loadtxt('ContactScorer_Relevant_{}_Mat.csv'.format(size), delimiter='\t')
    contact_adjacency = np.loadtxt('ContactScorer_A_Mat.csv', delimiter='\t')
    contact_bias_curr = contact_bias * contact_relevant_curr
    contact_intermediate_curr = np.loadtxt('ContactScorer_Intermediate_{}_Mat.csv'.format(size), delimiter='\t')
    if len(np.nonzero((contact_adjacency * contact_bias_curr) - contact_intermediate_curr)[0]) != 0:
        raise ValueError('Intermediate Contact Values do not match')
    rhonald_bias_curr = np.loadtxt('Rhonald_Bias_{}_Mat.csv'.format(size), delimiter='\t')
    diff_bias = contact_intermediate_curr - rhonald_bias_curr
    if np.sum(diff_bias) == 0 and len(np.nonzero(diff_bias)[0]) == 0:
        return True, (contact_bias, contact_bias_curr, contact_relevant_curr, rhonald_bias_curr)
    else:
        return False, (contact_bias, contact_bias_curr, contact_relevant_curr, rhonald_bias_curr)
