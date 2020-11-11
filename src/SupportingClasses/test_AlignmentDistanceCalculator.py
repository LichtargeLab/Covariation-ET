"""
Created on May 16, 2019

@author: Daniel Konecki
"""
import os
import unittest
from unittest import TestCase
import numpy as np
from copy import deepcopy
from time import time, sleep
from Bio.SubsMat.MatrixInfo import blosum62
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceMatrix
from test_Base import protein_seq1, protein_seq2, protein_seq3, protein_msa, dna_seq1, dna_seq2, dna_seq3, dna_msa
from utils import build_mapping, convert_seq_to_numeric
from ETMIPWrapper import ETMIPWrapper
from SeqAlignment import SeqAlignment
from EvolutionaryTraceAlphabet import FullIUPACProtein, FullIUPACDNA
from AlignmentDistanceCalculator import (AlignmentDistanceCalculator, convert_array_to_distance_matrix, init_pairwise,
                                         pairwise, init_identity, identity, init_characterize_sequence,
                                         characterize_sequence, init_similarity, similarity)


class TestAlignmentDistanceCalculatorInit(TestCase):

    def setUp(self):
        self.protein_alpha = FullIUPACProtein()
        self.protein_alpha_size, self.protein_gap_chars, self.protein_mapping, self.protein_reverse = build_mapping(
            alphabet=self.protein_alpha, skip_letters=None)
        self.protein_identity_mat = np.eye(self.protein_alpha_size + 2)
        self.protein_identity_mat[-1, -1] = 0
        self.dna_alpha = FullIUPACDNA()
        self.dna_alpha_size, self.dna_gap_chars, self.dna_mapping, self.dna_reverse = build_mapping(
            alphabet=self.dna_alpha, skip_letters=None)
        self.dna_identity_mat = np.eye(self.dna_alpha_size + 2)
        self.dna_identity_mat[-1, -1] = 0

    def evaluate_init(self, adc, q_type, expected_letters, dist_model, expected_alpha_size, expected_gap_chars,
                      expected_mapping, expected_reverse, expected_scoring_mat):
        self.assertEqual(adc.aln_type, q_type)
        self.assertEqual(adc.alphabet.letters, expected_letters)
        self.assertEqual(adc.model, dist_model)
        self.assertEqual(adc.alphabet_size, expected_alpha_size)
        self.assertEqual(adc.gap_characters, expected_gap_chars)
        self.assertEqual(adc.mapping, expected_mapping)
        if dist_model == 'identity':
            self.assertFalse((adc.scoring_matrix - expected_scoring_mat).any())
        elif dist_model == 'blosum62':
            for i in range(expected_alpha_size):
                for j in range(expected_alpha_size):
                    try:
                        self.assertEqual(adc.scoring_matrix[i, j], blosum62[(expected_reverse[i], expected_reverse[j])])
                    except KeyError:
                        self.assertEqual(adc.scoring_matrix[i, j], blosum62[(expected_reverse[j], expected_reverse[i])])
        elif dist_model == 'blastn':
            for i in range(expected_alpha_size):
                for j in range(expected_alpha_size):
                    try:
                        self.assertEqual(adc.scoring_matrix[i, j], DistanceCalculator.blastn[i][j])
                    except IndexError:
                        self.assertEqual(adc.scoring_matrix[i, j], DistanceCalculator.blastn[j][i])
        else:
            raise ValueError('Unexpected model in evaluate!')

    def test_init(self):
        adc = AlignmentDistanceCalculator()
        self.evaluate_init(adc=adc, q_type='protein', expected_letters=self.protein_alpha.letters,
                           dist_model='identity', expected_alpha_size=self.protein_alpha_size,
                           expected_gap_chars=self.protein_gap_chars, expected_mapping=self.protein_mapping,
                           expected_reverse=None, expected_scoring_mat=self.protein_identity_mat)

    def test_init_protein_identity_no_skip_letters(self):
        adc = AlignmentDistanceCalculator(protein=True, model='identity', skip_letters=None)
        self.evaluate_init(adc=adc, q_type='protein', expected_letters=self.protein_alpha.letters,
                           dist_model='identity', expected_alpha_size=self.protein_alpha_size,
                           expected_gap_chars=self.protein_gap_chars, expected_mapping=self.protein_mapping,
                           expected_reverse=None, expected_scoring_mat=self.protein_identity_mat)

    def test_init_protein_blosum62_no_skip_letters(self):
        adc = AlignmentDistanceCalculator(protein=True, model='blosum62', skip_letters=None)
        self.evaluate_init(adc=adc, q_type='protein', expected_letters=self.protein_alpha.letters,
                           dist_model='blosum62', expected_alpha_size=self.protein_alpha_size,
                           expected_gap_chars=self.protein_gap_chars, expected_mapping=self.protein_mapping,
                           expected_reverse=self.protein_reverse, expected_scoring_mat=None)

    def test_init_protein_identity_skip_letters(self):
        expected_alpha = FullIUPACProtein()
        expected_alpha_size, expected_gap_chars, expected_mapping, _ = build_mapping(alphabet=expected_alpha,
                                                                                     skip_letters=['.', '*'])
        adc = AlignmentDistanceCalculator(protein=True, model='identity', skip_letters=['.', '*'])
        self.evaluate_init(adc=adc, q_type='protein', expected_letters=expected_alpha.letters, dist_model='identity',
                           expected_alpha_size=expected_alpha_size, expected_gap_chars=expected_gap_chars,
                           expected_mapping=expected_mapping, expected_reverse=None,
                           expected_scoring_mat=self.protein_identity_mat)

    def test_init_dna_identity_no_skip_letters(self):
        adc = AlignmentDistanceCalculator(protein=False, model='identity', skip_letters=None)
        self.evaluate_init(adc=adc, q_type='dna', expected_letters=self.dna_alpha.letters, dist_model='identity',
                           expected_alpha_size=self.dna_alpha_size, expected_gap_chars=self.dna_gap_chars,
                           expected_mapping=self.dna_mapping, expected_reverse=None,
                           expected_scoring_mat=self.dna_identity_mat)

    def test_init_dna_blastn_no_skip_letters(self):
        adc = AlignmentDistanceCalculator(protein=False, model='blastn', skip_letters=None)
        self.evaluate_init(adc=adc, q_type='dna', expected_letters=self.dna_alpha.letters, dist_model='blastn',
                           expected_alpha_size=self.dna_alpha_size, expected_gap_chars=self.dna_gap_chars,
                           expected_mapping=self.dna_mapping, expected_reverse=None, expected_scoring_mat=None)

    def test_init_dna_identity_skip_letters(self):
        expected_alpha = FullIUPACDNA()
        expected_alpha_size, expected_gap_chars, expected_mapping, _ = build_mapping(alphabet=expected_alpha,
                                                                                     skip_letters=['.', '*'])
        adc = AlignmentDistanceCalculator(protein=False, model='identity', skip_letters=['.', '*'])
        self.evaluate_init(adc=adc, q_type='dna', expected_letters=expected_alpha.letters, dist_model='identity',
                           expected_alpha_size=expected_alpha_size, expected_gap_chars=expected_gap_chars,
                           expected_mapping=expected_mapping, expected_reverse=None,
                           expected_scoring_mat=self.dna_identity_mat)

    def test_init_fail_type_error_protein(self):
        with self.assertRaises(AssertionError):
            AlignmentDistanceCalculator(protein='Protein')

    def test_init_fail_value_error_model_protein(self):
        with self.assertRaises(ValueError):
            AlignmentDistanceCalculator(model='blastn')

    def test_init_fail_value_error_model_dna(self):
        with self.assertRaises(ValueError):
            AlignmentDistanceCalculator(protein=False, model='blosum62')

    def test_init_fail_type_error_skip_letters(self):
        with self.assertRaises(TypeError):
            AlignmentDistanceCalculator(skip_letters=1)


class TestAlignmentDistanceCalculatorMatrixConstruction(TestCase):

    def test_build_identity_scoring_matrix_protein(self):
        expected_alpha = FullIUPACProtein()
        expected_alpha_size, expected_gap_chars, expected_mapping, _ = build_mapping(alphabet=expected_alpha,
                                                                                     skip_letters=None)
        adc = AlignmentDistanceCalculator()
        id_mat = adc._build_identity_scoring_matrix()
        self.assertEqual(id_mat.shape, (adc.alphabet_size + 2, adc.alphabet_size + 2))
        self.assertTrue(id_mat[range(adc.alphabet_size + 1), range(adc.alphabet_size + 1)].all())
        id_mat[range(adc.alphabet_size + 1), range(adc.alphabet_size + 1)] -= 1
        self.assertFalse(id_mat.any())

    def test_build_identity_scoring_matrix_dna(self):
        expected_alpha = FullIUPACDNA()
        expected_alpha_size, expected_gap_chars, expected_mapping, _ = build_mapping(alphabet=expected_alpha,
                                                                                     skip_letters=None)
        adc = AlignmentDistanceCalculator(protein=False)
        id_mat = adc._build_identity_scoring_matrix()
        self.assertEqual(id_mat.shape, (adc.alphabet_size + 2, adc.alphabet_size + 2))
        self.assertTrue(id_mat[range(adc.alphabet_size + 1), range(adc.alphabet_size + 1)].all())
        id_mat[range(adc.alphabet_size + 1), range(adc.alphabet_size + 1)] -= 1
        self.assertFalse(id_mat.any())

    def test_rebuild_scoring_matrix_protein(self):
        dist_calc = DistanceCalculator(model='blosum62')
        adc = AlignmentDistanceCalculator(model='blosum62')
        adc.scoring_matrix = dist_calc.scoring_matrix
        scoring_mat = adc._rebuild_scoring_matrix()
        expected_reverse = np.array(list(adc.alphabet.letters))
        for i in range(adc.alphabet_size):
            for j in range(adc.alphabet_size):
                try:
                    self.assertEqual(scoring_mat[i, j], blosum62[(expected_reverse[i], expected_reverse[j])])
                except KeyError:
                    self.assertEqual(scoring_mat[i, j], blosum62[(expected_reverse[j], expected_reverse[i])])
                finally:
                    scoring_mat[i, j] = 0
        self.assertFalse(scoring_mat.any())

    def test_rebuild_scoring_matrix_dna(self):
        dist_calc = DistanceCalculator(model='blastn')
        adc = AlignmentDistanceCalculator(protein=False, model='blastn')
        adc.scoring_matrix = dist_calc.scoring_matrix
        scoring_mat = adc._rebuild_scoring_matrix()
        for i in range(adc.alphabet_size):
            for j in range(adc.alphabet_size):
                try:
                    self.assertEqual(scoring_mat[i, j], DistanceCalculator.blastn[i][j])
                except IndexError:
                    self.assertEqual(scoring_mat[i, j], DistanceCalculator.blastn[j][i])
                finally:
                    scoring_mat[i, j] = 0
        self.assertFalse(scoring_mat.any())

    def test_update_scoring_matrix_protein_identity(self):
        expected_alpha = FullIUPACProtein()
        expected_alpha_size, expected_gap_chars, expected_mapping, _ = build_mapping(alphabet=expected_alpha,
                                                                                     skip_letters=None)
        adc = AlignmentDistanceCalculator()
        id_mat = adc._update_scoring_matrix()
        self.assertEqual(id_mat.shape, (adc.alphabet_size + 2, adc.alphabet_size + 2))
        self.assertTrue(id_mat[range(adc.alphabet_size + 1), range(adc.alphabet_size + 1)].all())
        id_mat[range(adc.alphabet_size + 1), range(adc.alphabet_size + 1)] -= 1
        self.assertFalse(id_mat.any())

    def test_update_scoring_matrix_dna_identity(self):
        expected_alpha = FullIUPACDNA()
        expected_alpha_size, expected_gap_chars, expected_mapping, _ = build_mapping(alphabet=expected_alpha,
                                                                                     skip_letters=None)
        adc = AlignmentDistanceCalculator(protein=False)
        id_mat = adc._update_scoring_matrix()
        self.assertEqual(id_mat.shape, (adc.alphabet_size + 2, adc.alphabet_size + 2))
        self.assertTrue(id_mat[range(adc.alphabet_size + 1), range(adc.alphabet_size + 1)].all())
        id_mat[range(adc.alphabet_size + 1), range(adc.alphabet_size + 1)] -= 1
        self.assertFalse(id_mat.any())

    def test_update_scoring_matrix_protein_blosum62(self):
        dist_calc = DistanceCalculator(model='blosum62')
        adc = AlignmentDistanceCalculator(model='blosum62')
        adc.scoring_matrix = dist_calc.scoring_matrix
        scoring_mat = adc._rebuild_scoring_matrix()
        expected_reverse = np.array(list(adc.alphabet.letters))
        for i in range(adc.alphabet_size):
            for j in range(adc.alphabet_size):
                try:
                    self.assertEqual(scoring_mat[i, j], blosum62[(expected_reverse[i], expected_reverse[j])])
                except KeyError:
                    self.assertEqual(scoring_mat[i, j], blosum62[(expected_reverse[j], expected_reverse[i])])
                finally:
                    scoring_mat[i, j] = 0
        self.assertFalse(scoring_mat.any())

    def test_update_scoring_matrix_dna_blastn(self):
        dist_calc = DistanceCalculator(model='blastn')
        adc = AlignmentDistanceCalculator(protein=False, model='blastn')
        adc.scoring_matrix = dist_calc.scoring_matrix
        scoring_mat = adc._rebuild_scoring_matrix()
        for i in range(adc.alphabet_size):
            for j in range(adc.alphabet_size):
                try:
                    self.assertEqual(scoring_mat[i, j], DistanceCalculator.blastn[i][j])
                except IndexError:
                    self.assertEqual(scoring_mat[i, j], DistanceCalculator.blastn[j][i])
                finally:
                    scoring_mat[i, j] = 0
        self.assertFalse(scoring_mat.any())

    def test_update_scoring_matrix_fail(self):
        dist_calc = DistanceCalculator(model='blastn')
        adc = AlignmentDistanceCalculator(protein=False, model='blastn')
        adc.scoring_matrix = dist_calc.scoring_matrix
        adc.aln_type = 'rna'
        with self.assertRaises(ValueError):
            adc._update_scoring_matrix()


class TestAlignmentDistanceCalculatorPairwise(TestCase):

    # init_pairwise must be tested by proxy for now, I am not sure how to test the global variables, and testing
    # variable assignment is not essential.

    def test_pairwise_protein_identity_full(self):
        adc = AlignmentDistanceCalculator()
        init_pairwise(alpha_map=adc.mapping, alpha_size=adc.alphabet_size, mod=adc.model,
                      scoring_mat=adc.scoring_matrix)
        s1, s2, score = pairwise(seq1=protein_seq1, seq2=protein_seq1)
        self.assertEqual(s1, 'seq1')
        self.assertEqual(s2, 'seq1')
        self.assertEqual(score, 0)

    def test_pairwise_protein_identity_partial(self):
        adc = AlignmentDistanceCalculator()
        init_pairwise(alpha_map=adc.mapping, alpha_size=adc.alphabet_size, mod=adc.model,
                      scoring_mat=adc.scoring_matrix)
        s1, s2, score = pairwise(seq1=protein_seq1, seq2=protein_seq2)
        self.assertEqual(s1, 'seq1')
        self.assertEqual(s2, 'seq2')
        self.assertLess((score - (4.0 / 6.0)), 1E-15)

    def test_pairwise_dna_identity_full(self):
        adc = AlignmentDistanceCalculator(protein=False)
        init_pairwise(alpha_map=adc.mapping, alpha_size=adc.alphabet_size, mod=adc.model,
                      scoring_mat=adc.scoring_matrix)
        s1, s2, score = pairwise(seq1=dna_seq1, seq2=dna_seq1)
        self.assertEqual(s1, 'seq1')
        self.assertEqual(s2, 'seq1')
        self.assertEqual(score, 0)

    def test_pairwise_dna_identity_partial(self):
        adc = AlignmentDistanceCalculator(protein=False)
        init_pairwise(alpha_map=adc.mapping, alpha_size=adc.alphabet_size, mod=adc.model,
                      scoring_mat=adc.scoring_matrix)
        s1, s2, score = pairwise(seq1=dna_seq1, seq2=dna_seq2)
        self.assertEqual(s1, 'seq1')
        self.assertEqual(s2, 'seq2')
        self.assertLess((score - (12.0 / 18.0)), 1E-15)

    def test_pairwise_protein_blosum62_full(self):
        adc = AlignmentDistanceCalculator(model='blosum62')
        init_pairwise(alpha_map=adc.mapping, alpha_size=adc.alphabet_size, mod=adc.model,
                      scoring_mat=adc.scoring_matrix)
        s1, s2, score = pairwise(seq1=protein_seq1, seq2=protein_seq1)
        self.assertEqual(s1, 'seq1')
        self.assertEqual(s2, 'seq1')
        self.assertEqual(score, 0)

    def test_pairwise_protein_blosum62_partial(self):
        adc = AlignmentDistanceCalculator(model='blosum62')
        init_pairwise(alpha_map=adc.mapping, alpha_size=adc.alphabet_size, mod=adc.model,
                      scoring_mat=adc.scoring_matrix)
        s1, s2, score = pairwise(seq1=protein_seq1, seq2=protein_seq2)
        self.assertEqual(s1, 'seq1')
        self.assertEqual(s2, 'seq2')
        max_score = 0
        for char in protein_seq2:
            ind = adc.mapping[char]
            max_score += adc.scoring_matrix[ind, ind]
        actual_score = 0
        for i in range(len(protein_seq1)):
            ind1 = adc.mapping[protein_seq1[i]]
            ind2 = adc.mapping[protein_seq2[i]]
            actual_score += adc.scoring_matrix[ind1, ind2]
        self.assertLess((score - (actual_score / max_score)), 1E-15)

    def test_pairwise_dna_blastn_full(self):
        adc = AlignmentDistanceCalculator(protein=False, model='blastn')
        init_pairwise(alpha_map=adc.mapping, alpha_size=adc.alphabet_size, mod=adc.model,
                      scoring_mat=adc.scoring_matrix)
        s1, s2, score = pairwise(seq1=dna_seq1, seq2=dna_seq1)
        self.assertEqual(s1, 'seq1')
        self.assertEqual(s2, 'seq1')
        self.assertEqual(score, 0)

    def test_pairwise_dna_blastn_partial(self):
        adc = AlignmentDistanceCalculator(protein=False, model='blastn')
        init_pairwise(alpha_map=adc.mapping, alpha_size=adc.alphabet_size, mod=adc.model,
                      scoring_mat=adc.scoring_matrix)
        s1, s2, score = pairwise(seq1=dna_seq1, seq2=dna_seq2)
        self.assertEqual(s1, 'seq1')
        self.assertEqual(s2, 'seq2')
        max_score = 0
        for char in dna_seq2:
            ind = adc.mapping[char]
            max_score += adc.scoring_matrix[ind, ind]
        actual_score = 0
        for i in range(len(dna_seq1)):
            ind1 = adc.mapping[dna_seq1[i]]
            ind2 = adc.mapping[dna_seq2[i]]
            actual_score += adc.scoring_matrix[ind1, ind2]
        self.assertLess((score - (actual_score / max_score)), 1E-15)

    def test__pairwise_protein_identity_full(self):
        adc = AlignmentDistanceCalculator()
        score = adc._pairwise(protein_seq1, protein_seq1)
        self.assertEqual(score, 0)

    def test__pairwise_protein_identity_partial(self):
        adc = AlignmentDistanceCalculator()
        score = adc._pairwise(seq1=protein_seq1, seq2=protein_seq2)
        self.assertLess((score - (4.0 / 6.0)), 1E-15)

    def test__pairwise_dna_identity_full(self):
        adc = AlignmentDistanceCalculator(protein=False)
        score = adc._pairwise(seq1=dna_seq1, seq2=dna_seq1)
        self.assertEqual(score, 0)

    def test__pairwise_dna_identity_partial(self):
        adc = AlignmentDistanceCalculator(protein=False)
        score = adc._pairwise(seq1=dna_seq1, seq2=dna_seq2)
        self.assertLess((score - (12.0 / 18.0)), 1E-15)

    def test__pairwise_protein_blosum62_full(self):
        adc = AlignmentDistanceCalculator(model='blosum62')
        score = adc._pairwise(seq1=protein_seq1, seq2=protein_seq1)
        self.assertEqual(score, 0)

    def test__pairwise_protein_blosum62_partial(self):
        adc = AlignmentDistanceCalculator(model='blosum62')
        score = adc._pairwise(seq1=protein_seq1, seq2=protein_seq2)
        max_score = 0
        for char in protein_seq2:
            ind = adc.mapping[char]
            max_score += adc.scoring_matrix[ind, ind]
        actual_score = 0
        for i in range(len(protein_seq1)):
            ind1 = adc.mapping[protein_seq1[i]]
            ind2 = adc.mapping[protein_seq2[i]]
            actual_score += adc.scoring_matrix[ind1, ind2]
        self.assertLess((score - (actual_score / max_score)), 1E-15)

    def test__pairwise_dna_blastn_full(self):
        adc = AlignmentDistanceCalculator(protein=False, model='blastn')
        score = adc._pairwise(seq1=dna_seq1, seq2=dna_seq1)
        self.assertEqual(score, 0)

    def test__pairwise_dna_blastn_partial(self):
        adc = AlignmentDistanceCalculator(protein=False, model='blastn')
        score = adc._pairwise(seq1=dna_seq1, seq2=dna_seq2)
        max_score = 0
        for char in dna_seq2:
            ind = adc.mapping[char]
            max_score += adc.scoring_matrix[ind, ind]
        actual_score = 0
        for i in range(len(dna_seq1)):
            ind1 = adc.mapping[dna_seq1[i]]
            ind2 = adc.mapping[dna_seq2[i]]
            actual_score += adc.scoring_matrix[ind1, ind2]
        self.assertLess((score - (actual_score / max_score)), 1E-15)


class TestAlignmentDistanceCalculatorIdentityDistance(TestCase):

    # init_identity must be tested by proxy for now, I am not sure how to test the global variables, and testing
    # variable assignment is not essential.

    def test_identity_protein(self):
        adc = AlignmentDistanceCalculator()
        numerical_alignment = np.vstack([convert_seq_to_numeric(seq, mapping=adc.mapping) for seq in protein_msa])
        init_identity(numerical_alignment)
        ind, scores = identity(i=0)
        self.assertEqual(ind, 0)
        self.assertEqual(scores.shape, (len(protein_msa), ))
        for i in range(len(protein_msa)):
            score = adc._pairwise(protein_msa[0], protein_msa[i])
            self.assertEqual(scores[i], score)

    def test_identity_dna(self):
        adc = AlignmentDistanceCalculator(protein=False)
        numerical_alignment = np.vstack([convert_seq_to_numeric(seq, mapping=adc.mapping) for seq in dna_msa])
        init_identity(numerical_alignment)
        ind, scores = identity(i=0)
        self.assertEqual(ind, 0)
        self.assertEqual(scores.shape, (len(dna_msa), ))
        for i in range(len(dna_msa)):
            score = adc._pairwise(dna_msa[0], dna_msa[i])
            self.assertEqual(scores[i], score)

    def test_get_identity_distance_protein_single_process(self):
        adc = AlignmentDistanceCalculator()
        dm = adc.get_identity_distance(msa=protein_msa, processes=1)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in protein_msa])
        self.assertEqual(len(dm.matrix), len(protein_msa))
        for i in range(len(protein_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(protein_msa[i], protein_msa[j])
                self.assertEqual(dm[i][j], score)

    def test_get_identity_distance_protein_multi_process(self):
        adc = AlignmentDistanceCalculator()
        dm = adc.get_identity_distance(msa=protein_msa, processes=2)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in protein_msa])
        self.assertEqual(len(dm.matrix), len(protein_msa))
        for i in range(len(protein_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(protein_msa[i], protein_msa[j])
                self.assertEqual(dm[i][j], score)

    def test_get_identity_distance_dna_single_process(self):
        adc = AlignmentDistanceCalculator(protein=False)
        dm = adc.get_identity_distance(msa=dna_msa, processes=1)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in dna_msa])
        self.assertEqual(len(dm.matrix), len(dna_msa))
        for i in range(len(dna_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(dna_msa[i], dna_msa[j])
                self.assertEqual(dm[i][j], score)

    def test_get_identity_distance_dna_multi_process(self):
        adc = AlignmentDistanceCalculator(protein=False)
        dm = adc.get_identity_distance(msa=dna_msa, processes=2)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in dna_msa])
        self.assertEqual(len(dm.matrix), len(dna_msa))
        for i in range(len(dna_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(dna_msa[i], dna_msa[j])
                self.assertEqual(dm[i][j], score)


class TestAlignmentDistanceCalculatorGetScoringMatrixDistance(TestCase):

    # init_pairwise and pairwise were already tested, so only the get_identity_distance method will be tested here.

    def test_get_scoring_matrix_distance_protein_single_process(self):
        adc = AlignmentDistanceCalculator(model='blosum62')
        dm = adc.get_scoring_matrix_distance(msa=protein_msa, processes=1)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in protein_msa])
        self.assertEqual(len(dm.matrix), len(protein_msa))
        for i in range(len(protein_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(protein_msa[i], protein_msa[j])
                self.assertEqual(dm[i][j], score)

    def test_get_scoring_matrix_distance_protein_multi_process(self):
        adc = AlignmentDistanceCalculator(model='blosum62')
        dm = adc.get_scoring_matrix_distance(msa=protein_msa, processes=2)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in protein_msa])
        self.assertEqual(len(dm.matrix), len(protein_msa))
        for i in range(len(protein_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(protein_msa[i], protein_msa[j])
                self.assertEqual(dm[i][j], score)

    def test_get_scoring_matrix_distance_dna_single_process(self):
        adc = AlignmentDistanceCalculator(protein=False, model='blastn')
        dm = adc.get_scoring_matrix_distance(msa=dna_msa, processes=1)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in dna_msa])
        self.assertEqual(len(dm.matrix), len(dna_msa))
        for i in range(len(dna_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(dna_msa[i], dna_msa[j])
                self.assertEqual(dm[i][j], score)

    def test_get_scoring_matrix_distance_dna_multi_process(self):
        adc = AlignmentDistanceCalculator(protein=False, model='blastn')
        dm = adc.get_scoring_matrix_distance(msa=dna_msa, processes=2)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in dna_msa])
        self.assertEqual(len(dm.matrix), len(dna_msa))
        for i in range(len(dna_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(dna_msa[i], dna_msa[j])
                self.assertEqual(dm[i][j], score)


class TestAlignmentDistanceCalculatorGetDistance(TestCase):

    def test_get_distance_protein_identity_single_process(self):
        adc = AlignmentDistanceCalculator()
        dm = adc.get_distance(msa=protein_msa, processes=1)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in protein_msa])
        self.assertEqual(len(dm.matrix), len(protein_msa))
        for i in range(len(protein_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(protein_msa[i], protein_msa[j])
                self.assertEqual(dm[i][j], score)

    def test_get_distance_protein_identity_multi_process(self):
        adc = AlignmentDistanceCalculator()
        dm = adc.get_distance(msa=protein_msa, processes=2)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in protein_msa])
        self.assertEqual(len(dm.matrix), len(protein_msa))
        for i in range(len(protein_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(protein_msa[i], protein_msa[j])
                self.assertEqual(dm[i][j], score)

    def test_get_distance_protein_blosum62_single_process(self):
        adc = AlignmentDistanceCalculator(model='blosum62')
        dm = adc.get_distance(msa=protein_msa, processes=1)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in protein_msa])
        self.assertEqual(len(dm.matrix), len(protein_msa))
        for i in range(len(protein_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(protein_msa[i], protein_msa[j])
                self.assertEqual(dm[i][j], score)

    def test_get_distance_protein_blosum62_multi_process(self):
        adc = AlignmentDistanceCalculator(model='blosum62')
        dm = adc.get_distance(msa=protein_msa, processes=2)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in protein_msa])
        self.assertEqual(len(dm.matrix), len(protein_msa))
        for i in range(len(protein_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(protein_msa[i], protein_msa[j])
                self.assertEqual(dm[i][j], score)

    def test_get_distance_dna_identity_single_process(self):
        adc = AlignmentDistanceCalculator(protein=False)
        dm = adc.get_distance(msa=dna_msa, processes=1)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in dna_msa])
        self.assertEqual(len(dm.matrix), len(dna_msa))
        for i in range(len(dna_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(dna_msa[i], dna_msa[j])
                self.assertEqual(dm[i][j], score)

    def test_get_distance_dna_identity_multi_process(self):
        adc = AlignmentDistanceCalculator(protein=False)
        dm = adc.get_distance(msa=dna_msa, processes=2)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in dna_msa])
        self.assertEqual(len(dm.matrix), len(dna_msa))
        for i in range(len(dna_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(dna_msa[i], dna_msa[j])
                self.assertEqual(dm[i][j], score)

    def test_get_distance_dna_blastn_single_process(self):
        adc = AlignmentDistanceCalculator(protein=False, model='blastn')
        dm = adc.get_distance(msa=dna_msa, processes=1)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in dna_msa])
        self.assertEqual(len(dm.matrix), len(dna_msa))
        for i in range(len(dna_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(dna_msa[i], dna_msa[j])
                self.assertEqual(dm[i][j], score)

    def test_get_distance_dna_blastn_multi_process(self):
        adc = AlignmentDistanceCalculator(protein=False, model='blastn')
        dm = adc.get_distance(msa=dna_msa, processes=2)
        self.assertTrue(isinstance(dm, DistanceMatrix))
        self.assertEqual(dm.names, [seq_rec.id for seq_rec in dna_msa])
        self.assertEqual(len(dm.matrix), len(dna_msa))
        for i in range(len(dna_msa)):
            self.assertEqual(len(dm.matrix[i]), i + 1)
            for j in range(i + 1):
                score = adc._pairwise(dna_msa[i], dna_msa[j])
                self.assertEqual(dm[i][j], score)

    def test_et_distance_fail_not_msa(self):
        adc = AlignmentDistanceCalculator()
        with self.assertRaises(TypeError):
            adc.get_distance(msa=[protein_seq1, protein_seq2, protein_seq3], processes=1)


class TestAlignmentDistanceCalculatorGetETDistance(TestCase):

    # init_characterize_sequence and init_similarity must be tested by proxy for now, I am not sure how to test the
    # global variables, and testing variable assignment is not essential.

    def test_characterize_sequence_protein_contiguous(self):
        alpha_size, _, mapping, _ = build_mapping(alphabet=FullIUPACProtein(), skip_letters=None)
        init_characterize_sequence(aln=protein_msa, alpha_map=mapping, alpha_size=alpha_size)
        seq_id, non_gap_length, non_gap_pos, num_repr = characterize_sequence(i=0)
        self.assertEqual(seq_id, protein_msa[0].id)
        self.assertEqual(non_gap_length, 3)
        self.assertEqual(list(non_gap_pos), [True if x != '-' else False for x in protein_msa[0].seq])
        self.assertFalse((num_repr - convert_seq_to_numeric(protein_msa[0].seq, mapping=mapping)).any())

    def test_characterize_sequence_protein_noncontiguous(self):
        alpha_size, _, mapping, _ = build_mapping(alphabet=FullIUPACProtein(), skip_letters=None)
        init_characterize_sequence(aln=protein_msa, alpha_map=mapping, alpha_size=alpha_size)
        seq_id, non_gap_length, non_gap_pos, num_repr = characterize_sequence(i=1)
        self.assertEqual(seq_id, protein_msa[1].id)
        self.assertEqual(non_gap_length, 5)
        self.assertEqual(list(non_gap_pos), [True if x != '-' else False for x in protein_msa[1].seq])
        self.assertFalse((num_repr - convert_seq_to_numeric(protein_msa[1].seq, mapping=mapping)).any())

    def test_characterize_sequence_dna_contiguous(self):
        alpha_size, _, mapping, _ = build_mapping(alphabet=FullIUPACDNA(), skip_letters=None)
        init_characterize_sequence(aln=dna_msa, alpha_map=mapping, alpha_size=alpha_size)
        seq_id, non_gap_length, non_gap_pos, num_repr = characterize_sequence(i=0)
        self.assertEqual(seq_id, dna_msa[0].id)
        self.assertEqual(non_gap_length, 9)
        self.assertEqual(list(non_gap_pos), [True if x != '-' else False for x in dna_msa[0].seq])
        self.assertFalse((num_repr - convert_seq_to_numeric(dna_msa[0].seq, mapping=mapping)).any())

    def test_characterize_sequence_dna_noncontiguous(self):
        alpha_size, _, mapping, _ = build_mapping(alphabet=FullIUPACDNA(), skip_letters=None)
        init_characterize_sequence(aln=dna_msa, alpha_map=mapping, alpha_size=alpha_size)
        seq_id, non_gap_length, non_gap_pos, num_repr = characterize_sequence(i=1)
        self.assertEqual(seq_id, dna_msa[1].id)
        self.assertEqual(non_gap_length, 15)
        self.assertEqual(list(non_gap_pos), [True if x != '-' else False for x in dna_msa[1].seq])
        self.assertFalse((num_repr - convert_seq_to_numeric(dna_msa[1].seq, mapping=mapping)).any())

    def test_similarity_protein(self):
        alpha_size, _, mapping, _ = build_mapping(alphabet=FullIUPACProtein(), skip_letters=None)
        init_characterize_sequence(aln=protein_msa, alpha_map=mapping, alpha_size=alpha_size)
        seq_id1, non_gap_length1, non_gap_pos1, num_repr1 = characterize_sequence(i=0)
        seq_id2, non_gap_length2, non_gap_pos2, num_repr2 = characterize_sequence(i=1)
        seq_con = {seq_id1: {'non_gap_length': non_gap_length1,
                            'non_gap_pos': non_gap_pos1,
                            'num_repr': num_repr1},
                   seq_id2: {'non_gap_length': non_gap_length2,
                            'non_gap_pos': non_gap_pos2,
                            'num_repr': num_repr2}}
        adc = AlignmentDistanceCalculator(model='blosum62')
        init_similarity(seq_con=seq_con, cutoff=1.0, score_mat=adc.scoring_matrix)
        res = similarity(seq_id1, seq_id2)
        expected_id_count = 0
        expected_score_count = 0
        for i in range(non_gap_pos1.shape[0]):
            if non_gap_pos1[i] and non_gap_pos2[i]:
                if num_repr1[i] == num_repr2[i]:
                    expected_id_count += 1
                if adc.scoring_matrix[num_repr1[i], num_repr2[i]] >= 1.0:
                    expected_score_count += 1
        expected_seq_len = min(non_gap_length1, non_gap_length2)
        expected_id_score = expected_id_count / float(expected_seq_len)
        expected_sim_score = 1 - (expected_score_count / float(expected_seq_len))
        self.assertEqual(res[0], seq_id1)
        self.assertEqual(res[1], seq_id2)
        self.assertEqual(res[2], expected_id_score)
        self.assertEqual(res[3], expected_sim_score)
        self.assertEqual(res[4], expected_seq_len)
        self.assertEqual(res[5], expected_id_count)
        self.assertEqual(res[6], expected_score_count)

    def test_similarity_dna(self):
        alpha_size, _, mapping, _ = build_mapping(alphabet=FullIUPACDNA(), skip_letters=None)
        init_characterize_sequence(aln=dna_msa, alpha_map=mapping, alpha_size=alpha_size)
        seq_id1, non_gap_length1, non_gap_pos1, num_repr1 = characterize_sequence(i=0)
        seq_id2, non_gap_length2, non_gap_pos2, num_repr2 = characterize_sequence(i=1)
        seq_con = {seq_id1: {'non_gap_length': non_gap_length1,
                             'non_gap_pos': non_gap_pos1,
                             'num_repr': num_repr1},
                   seq_id2: {'non_gap_length': non_gap_length2,
                             'non_gap_pos': non_gap_pos2,
                             'num_repr': num_repr2}}
        adc = AlignmentDistanceCalculator(protein=False, model='blastn')
        init_similarity(seq_con=seq_con, cutoff=1.0, score_mat=adc.scoring_matrix)
        res = similarity(seq_id1, seq_id2)
        expected_id_count = 0
        expected_score_count = 0
        for i in range(non_gap_pos1.shape[0]):
            if non_gap_pos1[i] and non_gap_pos2[i]:
                if num_repr1[i] == num_repr2[i]:
                    expected_id_count += 1
                if adc.scoring_matrix[num_repr1[i], num_repr2[i]] >= 1.0:
                    expected_score_count += 1
        expected_seq_len = min(non_gap_length1, non_gap_length2)
        expected_id_score = expected_id_count / float(expected_seq_len)
        expected_sim_score = 1 - (expected_score_count / float(expected_seq_len))
        self.assertEqual(res[0], seq_id1)
        self.assertEqual(res[1], seq_id2)
        self.assertEqual(res[2], expected_id_score)
        self.assertEqual(res[3], expected_sim_score)
        self.assertEqual(res[4], expected_seq_len)
        self.assertEqual(res[5], expected_id_count)
        self.assertEqual(res[6], expected_score_count)

    def test_get_et_distance_protein_single_process(self):
        adc = AlignmentDistanceCalculator(model='blosum62')
        plain_id, similarity, df, threshold = adc.get_et_distance(msa=protein_msa, processes=1)
        unique_score_mat = np.tril(adc.scoring_matrix, k=-1)
        expected_thresh = np.floor(np.average(unique_score_mat[unique_score_mat > 0]) + 0.5)
        self.assertEqual(threshold, expected_thresh)
        self.assertTrue(isinstance(plain_id, DistanceMatrix))
        self.assertTrue(isinstance(similarity, DistanceMatrix))
        self.assertEqual(plain_id.names, [seq_rec.id for seq_rec in protein_msa])
        self.assertEqual(similarity.names, [seq_rec.id for seq_rec in protein_msa])
        self.assertEqual(len(plain_id.matrix), len(dna_msa))
        self.assertEqual(len(similarity.matrix), len(dna_msa))
        pair_count = 0
        for i in range(len(protein_msa)):
            self.assertEqual(len(plain_id.matrix[i]), i + 1)
            self.assertEqual(len(similarity.matrix[i]), i + 1)
            for j in range(i):
                init_characterize_sequence(aln=protein_msa, alpha_map=adc.mapping, alpha_size=adc.alphabet_size)
                seq_id1, non_gap_length1, non_gap_pos1, num_repr1 = characterize_sequence(i=i)
                seq_id2, non_gap_length2, non_gap_pos2, num_repr2 = characterize_sequence(i=j)
                expected_id_count = 0
                expected_score_count = 0
                for x in range(non_gap_pos1.shape[0]):
                    if non_gap_pos1[x] and non_gap_pos2[x]:
                        if num_repr1[x] == num_repr2[x]:
                            expected_id_count += 1
                        if adc.scoring_matrix[num_repr1[x], num_repr2[x]] >= expected_thresh:
                            expected_score_count += 1
                expected_seq_len = min(non_gap_length1, non_gap_length2)
                seq1_ind = df['Seq1'] == protein_msa[i].id
                seq2_ind = df['Seq2'] == protein_msa[j].id
                joint_ind = seq1_ind & seq2_ind
                if not np.sum(joint_ind) > 0:
                    seq1_ind = df['Seq1'] == protein_msa[j].id
                    seq2_ind = df['Seq2'] == protein_msa[i].id
                    joint_ind = seq1_ind & seq2_ind
                if np.sum(joint_ind) > 0:
                    df_row = df.iloc[df.index[joint_ind][0], :]
                    self.assertEqual(df_row['Min_Seq_Length'], expected_seq_len)
                    self.assertEqual(df_row['Id_Count'], expected_id_count)
                    self.assertEqual(df_row['Threshold_Count'], expected_score_count)
                    pair_count += 1
                expected_id_score = expected_id_count / float(expected_seq_len)
                expected_sim_score = 1 - (expected_score_count / float(expected_seq_len))
                self.assertEqual(plain_id[i][j], expected_id_score)
                self.assertEqual(similarity[i][j], expected_sim_score)
        self.assertEqual(len(df), pair_count)

    def test_get_et_distance_protein_multi_process(self):
        adc = AlignmentDistanceCalculator(model='blosum62')
        plain_id, similarity, df, threshold = adc.get_et_distance(msa=protein_msa, processes=2)
        unique_score_mat = np.tril(adc.scoring_matrix, k=-1)
        expected_thresh = np.floor(np.average(unique_score_mat[unique_score_mat > 0]) + 0.5)
        self.assertEqual(threshold, expected_thresh)
        self.assertTrue(isinstance(plain_id, DistanceMatrix))
        self.assertTrue(isinstance(similarity, DistanceMatrix))
        self.assertEqual(plain_id.names, [seq_rec.id for seq_rec in protein_msa])
        self.assertEqual(similarity.names, [seq_rec.id for seq_rec in protein_msa])
        self.assertEqual(len(plain_id.matrix), len(dna_msa))
        self.assertEqual(len(similarity.matrix), len(dna_msa))
        pair_count = 0
        for i in range(len(protein_msa)):
            self.assertEqual(len(plain_id.matrix[i]), i + 1)
            self.assertEqual(len(similarity.matrix[i]), i + 1)
            for j in range(i):
                init_characterize_sequence(aln=protein_msa, alpha_map=adc.mapping, alpha_size=adc.alphabet_size)
                seq_id1, non_gap_length1, non_gap_pos1, num_repr1 = characterize_sequence(i=i)
                seq_id2, non_gap_length2, non_gap_pos2, num_repr2 = characterize_sequence(i=j)
                expected_id_count = 0
                expected_score_count = 0
                for x in range(non_gap_pos1.shape[0]):
                    if non_gap_pos1[x] and non_gap_pos2[x]:
                        if num_repr1[x] == num_repr2[x]:
                            expected_id_count += 1
                        if adc.scoring_matrix[num_repr1[x], num_repr2[x]] >= expected_thresh:
                            expected_score_count += 1
                expected_seq_len = min(non_gap_length1, non_gap_length2)
                seq1_ind = df['Seq1'] == protein_msa[i].id
                seq2_ind = df['Seq2'] == protein_msa[j].id
                joint_ind = seq1_ind & seq2_ind
                if not np.sum(joint_ind) > 0:
                    seq1_ind = df['Seq1'] == protein_msa[j].id
                    seq2_ind = df['Seq2'] == protein_msa[i].id
                    joint_ind = seq1_ind & seq2_ind
                if np.sum(joint_ind) > 0:
                    df_row = df.iloc[df.index[joint_ind][0], :]
                    self.assertEqual(df_row['Min_Seq_Length'], expected_seq_len)
                    self.assertEqual(df_row['Id_Count'], expected_id_count)
                    self.assertEqual(df_row['Threshold_Count'], expected_score_count)
                    pair_count += 1
                expected_id_score = expected_id_count / float(expected_seq_len)
                expected_sim_score = 1 - (expected_score_count / float(expected_seq_len))
                self.assertEqual(plain_id[i][j], expected_id_score)
                self.assertEqual(similarity[i][j], expected_sim_score)
        self.assertEqual(len(df), pair_count)

    def test_get_et_distance_dna_single_process(self):
        adc = AlignmentDistanceCalculator(protein=False, model='blastn')
        plain_id, similarity, df, threshold = adc.get_et_distance(msa=dna_msa, processes=1)
        expected_thresh = 1
        self.assertEqual(threshold, expected_thresh)
        self.assertTrue(isinstance(plain_id, DistanceMatrix))
        self.assertTrue(isinstance(similarity, DistanceMatrix))
        self.assertEqual(plain_id.names, [seq_rec.id for seq_rec in dna_msa])
        self.assertEqual(similarity.names, [seq_rec.id for seq_rec in dna_msa])
        self.assertEqual(len(plain_id.matrix), len(dna_msa))
        self.assertEqual(len(similarity.matrix), len(dna_msa))
        pair_count = 0
        for i in range(len(dna_msa)):
            self.assertEqual(len(plain_id.matrix[i]), i + 1)
            self.assertEqual(len(similarity.matrix[i]), i + 1)
            for j in range(i):
                init_characterize_sequence(aln=dna_msa, alpha_map=adc.mapping, alpha_size=adc.alphabet_size)
                seq_id1, non_gap_length1, non_gap_pos1, num_repr1 = characterize_sequence(i=i)
                seq_id2, non_gap_length2, non_gap_pos2, num_repr2 = characterize_sequence(i=j)
                expected_id_count = 0
                expected_score_count = 0
                for x in range(non_gap_pos1.shape[0]):
                    if non_gap_pos1[x] and non_gap_pos2[x]:
                        if num_repr1[x] == num_repr2[x]:
                            expected_id_count += 1
                        if adc.scoring_matrix[num_repr1[x], num_repr2[x]] >= expected_thresh:
                            expected_score_count += 1
                expected_seq_len = min(non_gap_length1, non_gap_length2)
                seq1_ind = df['Seq1'] == dna_msa[i].id
                seq2_ind = df['Seq2'] == dna_msa[j].id
                joint_ind = seq1_ind & seq2_ind
                if not np.sum(joint_ind) > 0:
                    seq1_ind = df['Seq1'] == dna_msa[j].id
                    seq2_ind = df['Seq2'] == dna_msa[i].id
                    joint_ind = seq1_ind & seq2_ind
                if np.sum(joint_ind) > 0:
                    df_row = df.iloc[df.index[joint_ind][0], :]
                    self.assertEqual(df_row['Min_Seq_Length'], expected_seq_len)
                    self.assertEqual(df_row['Id_Count'], expected_id_count)
                    self.assertEqual(df_row['Threshold_Count'], expected_score_count)
                    pair_count += 1
                expected_id_score = expected_id_count / float(expected_seq_len)
                expected_sim_score = 1 - (expected_score_count / float(expected_seq_len))
                self.assertEqual(plain_id[i][j], expected_id_score)
                self.assertEqual(similarity[i][j], expected_sim_score)
        self.assertEqual(len(df), pair_count)

    def test_get_et_distance_dna_multi_process(self):
        adc = AlignmentDistanceCalculator(protein=False, model='blastn')
        plain_id, similarity, df, threshold = adc.get_et_distance(msa=dna_msa, processes=2)
        expected_thresh = 1
        self.assertEqual(threshold, expected_thresh)
        self.assertTrue(isinstance(plain_id, DistanceMatrix))
        self.assertTrue(isinstance(similarity, DistanceMatrix))
        self.assertEqual(plain_id.names, [seq_rec.id for seq_rec in dna_msa])
        self.assertEqual(similarity.names, [seq_rec.id for seq_rec in dna_msa])
        self.assertEqual(len(plain_id.matrix), len(dna_msa))
        self.assertEqual(len(similarity.matrix), len(dna_msa))
        pair_count = 0
        for i in range(len(dna_msa)):
            self.assertEqual(len(plain_id.matrix[i]), i + 1)
            self.assertEqual(len(similarity.matrix[i]), i + 1)
            for j in range(i):
                init_characterize_sequence(aln=dna_msa, alpha_map=adc.mapping, alpha_size=adc.alphabet_size)
                seq_id1, non_gap_length1, non_gap_pos1, num_repr1 = characterize_sequence(i=i)
                seq_id2, non_gap_length2, non_gap_pos2, num_repr2 = characterize_sequence(i=j)
                expected_id_count = 0
                expected_score_count = 0
                for x in range(non_gap_pos1.shape[0]):
                    if non_gap_pos1[x] and non_gap_pos2[x]:
                        if num_repr1[x] == num_repr2[x]:
                            expected_id_count += 1
                        if adc.scoring_matrix[num_repr1[x], num_repr2[x]] >= expected_thresh:
                            expected_score_count += 1
                expected_seq_len = min(non_gap_length1, non_gap_length2)
                seq1_ind = df['Seq1'] == dna_msa[i].id
                seq2_ind = df['Seq2'] == dna_msa[j].id
                joint_ind = seq1_ind & seq2_ind
                if not np.sum(joint_ind) > 0:
                    seq1_ind = df['Seq1'] == dna_msa[j].id
                    seq2_ind = df['Seq2'] == dna_msa[i].id
                    joint_ind = seq1_ind & seq2_ind
                if np.sum(joint_ind) > 0:
                    df_row = df.iloc[df.index[joint_ind][0], :]
                    self.assertEqual(df_row['Min_Seq_Length'], expected_seq_len)
                    self.assertEqual(df_row['Id_Count'], expected_id_count)
                    self.assertEqual(df_row['Threshold_Count'], expected_score_count)
                    pair_count += 1
                expected_id_score = expected_id_count / float(expected_seq_len)
                expected_sim_score = 1 - (expected_score_count / float(expected_seq_len))
                self.assertEqual(plain_id[i][j], expected_id_score)
                self.assertEqual(similarity[i][j], expected_sim_score)
        self.assertEqual(len(df), pair_count)

    def test_get_et_distance_fail_not_msa(self):
        adc = AlignmentDistanceCalculator()
        with self.assertRaises(TypeError):
            adc.get_et_distance(msa=[protein_seq1, protein_seq2, protein_seq3], processes=1)


class TestAlignmentDistanceCalculatorConvertArrayToDistanceMatrix(TestCase):

    def test_convert_array_to_distance_matrix_protein(self):
        adc = AlignmentDistanceCalculator(model='blosum62')
        dm = adc.get_distance(msa=protein_msa, processes=2)
        arr = np.array(dm)
        expected_list = []
        for i in range(arr.shape[0]):
            curr_list = []
            for j in range(i + 1):
                curr_list.append(arr[i, j])
            expected_list.append(curr_list)
        lit_of_lists = convert_array_to_distance_matrix(arr, dm.names)
        self.assertEqual(lit_of_lists.names, dm.names)
        self.assertEqual(lit_of_lists.matrix, expected_list)

    def test_convert_array_to_distance_matrix_dna(self):
        adc = AlignmentDistanceCalculator(protein=False, model='blastn')
        dm = adc.get_distance(msa=dna_msa, processes=2)
        arr = np.array(dm)
        expected_list = []
        for i in range(arr.shape[0]):
            curr_list = []
            for j in range(i + 1):
                curr_list.append(arr[i, j])
            expected_list.append(curr_list)
        lit_of_lists = convert_array_to_distance_matrix(arr, dm.names)
        self.assertEqual(lit_of_lists.names, dm.names)
        self.assertEqual(lit_of_lists.matrix, expected_list)


if __name__ == '__main__':
    unittest.main()
