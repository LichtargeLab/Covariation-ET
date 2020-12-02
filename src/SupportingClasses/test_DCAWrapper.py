"""
Created on February 5, 2020

@author: Daniel Konecki
"""
import os
import unittest
from unittest import TestCase
import numpy as np
from time import time
from shutil import rmtree
from DCAWrapper import DCAWrapper
from utils import compute_rank_and_coverage
from test_Base import protein_seq1, protein_seq2, protein_seq3, dna_seq1, dna_seq2, dna_seq3, write_out_temp_fn


pro_str = f'>{protein_seq1.id}\n{protein_seq1.seq}\n>{protein_seq2.id}\n{protein_seq2.seq}\n>{protein_seq3.id}\n{protein_seq3.seq}'
dna_str = f'>{dna_seq1.id}\n{dna_seq1.seq}\n>{dna_seq2.id}\n{dna_seq2.seq}\n>{dna_seq3.id}\n{dna_seq3.seq}'
test_dir = os.path.join(os.getcwd(), 'TestCase')


class TestDCAWrapperInit(TestCase):

    def evaluate_init(self, query, aln_file, out_dir, expected_length, expected_sequence, polymer_type='Protein'):
        self.assertFalse(os.path.isdir(os.path.abspath(out_dir)))
        dca = DCAWrapper(query=query, aln_file=aln_file, out_dir=out_dir, polymer_type=polymer_type)
        self.assertEqual(dca.out_dir, os.path.abspath(out_dir))
        self.assertTrue(os.path.isdir(os.path.abspath(out_dir)))
        self.assertEqual(dca.query, query)
        self.assertIsNotNone(dca.original_aln)
        self.assertGreaterEqual(dca.original_aln.seq_length, expected_length)
        self.assertEqual(str(dca.original_aln.query_sequence).replace('-', ''), expected_sequence)
        self.assertEqual(dca.original_aln_fn, os.path.join(out_dir, 'Original_Alignment.fa'))
        self.assertTrue(os.path.isfile(dca.original_aln_fn))
        self.assertIsNotNone(dca.non_gapped_aln)
        self.assertEqual(dca.non_gapped_aln.seq_length, expected_length)
        self.assertEqual(dca.non_gapped_aln.query_sequence, expected_sequence)
        self.assertEqual(dca.non_gapped_aln_fn, os.path.join(out_dir, 'Non-Gapped_Alignment.fa'))
        self.assertTrue(os.path.isfile(dca.non_gapped_aln_fn))
        self.assertEqual(dca.method, 'DCA')
        self.assertIsNone(dca.scores)
        self.assertIsNone(dca.coverages)
        self.assertIsNone(dca.rankings)
        self.assertIsNone(dca.time)

    def test_dcawrapper_init_protein1(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_init(query='seq1', aln_file=protein_aln_fn, out_dir=test_dir, expected_length=3,
                           expected_sequence='MET')
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_dcawrapper_init_protein2(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_init(query='seq2', aln_file=protein_aln_fn, out_dir=test_dir, expected_length=5,
                           expected_sequence='MTREE')
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_dcawrapper_init_protein3(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_init(query='seq3', aln_file=protein_aln_fn, out_dir=test_dir, expected_length=5,
                           expected_sequence='MFREE')
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_dcawrapper_init_dna1(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_init(query='seq1', aln_file=dna_aln_fn, out_dir=test_dir, expected_length=9,
                           expected_sequence='ATGGAGACT', polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_dcawrapper_init_dna2(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_init(query='seq2', aln_file=dna_aln_fn, out_dir=test_dir, expected_length=15,
                           expected_sequence='ATGACTAGAGAGGAG', polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_dcawrapper_init_dna3(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_init(query='seq3', aln_file=dna_aln_fn, out_dir=test_dir, expected_length=15,
                           expected_sequence='ATGTTTAGAGAGGAG', polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)


class TestDCAWrapperImportScores(TestCase):

    def evaluate_import_scores(self, query, aln_file, out_dir, expected_length, polymer_type='Protein'):
        dca = DCAWrapper(query=query, aln_file=aln_file, out_dir=out_dir, polymer_type=polymer_type)
        scores = np.random.RandomState(1234567890).rand(expected_length, expected_length)
        scores[np.tril_indices(expected_length, 1)] = 0
        scores += scores.T
        indices = np.triu_indices(expected_length, 1)
        sorted_scores, sorted_x, sorted_y = zip(*sorted(zip(scores[indices], indices[0], indices[1])))
        with open(os.path.join(out_dir, 'DCA_predictions.tsv'), 'w') as handle:
            for i in range(len(sorted_scores)):
                handle.write('{} {} {}\n'.format(sorted_x[i] + 1, sorted_y[i] + 1, sorted_scores[i]))
        dca.import_covariance_scores(out_path=os.path.join(out_dir, 'DCA_predictions.tsv'))
        diff_scores = dca.scores - scores
        not_passing_scores = diff_scores > 1E-15
        self.assertFalse(not_passing_scores.any())

    def test_dcawrapper_import_scores_protein1(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_import_scores(query='seq1', aln_file=protein_aln_fn, out_dir=test_dir, expected_length=3)
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_dcawrapper_import_scores_protein2(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_import_scores(query='seq2', aln_file=protein_aln_fn, out_dir=test_dir, expected_length=5)
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_dcawrapper_import_scores_protein3(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_import_scores(query='seq3', aln_file=protein_aln_fn, out_dir=test_dir, expected_length=5)
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_dcawrapper_import_scores_dna1(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_import_scores(query='seq1', aln_file=dna_aln_fn, out_dir=test_dir, expected_length=9,
                                    polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_dcawrapper_import_scores_dna2(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_import_scores(query='seq2', aln_file=dna_aln_fn, out_dir=test_dir, expected_length=15,
                                    polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_dcawrapper_import_scores_dna3(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_import_scores(query='seq3', aln_file=dna_aln_fn, out_dir=test_dir, expected_length=15,
                                    polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)


class TestDCAWrapperCalculateScores(TestCase):

    def evaluate_calculate_scores(self, query, aln_file, out_dir, expected_length, polymer_type='Protein'):
        dca = DCAWrapper(query=query, aln_file=aln_file, out_dir=out_dir, polymer_type=polymer_type)
        start = time()
        dca.calculate_scores(delete_file=False)
        end = time()
        expected_time = end - start
        self.assertIsNotNone(dca.scores)
        expected_ranks, expected_coverages = compute_rank_and_coverage(expected_length, dca.scores, 2, 'max')
        ranks_diff = dca.rankings - expected_ranks
        ranks_not_passing = ranks_diff > 0.0
        self.assertFalse(ranks_not_passing.any())
        coverages_diff = dca.coverages - expected_coverages
        coverages_not_passing = coverages_diff > 0.0
        self.assertFalse(coverages_not_passing.any())
        self.assertLessEqual(dca.time, expected_time)
        self.assertTrue(os.path.isfile(os.path.join(out_dir, 'DCA.npz')))

    def test_dcawrapper_calculate_scores_protein1(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_calculate_scores(query='seq1', aln_file=protein_aln_fn, out_dir=test_dir, expected_length=3)
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_dcawrapper_calculate_scores_protein2(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_calculate_scores(query='seq2', aln_file=protein_aln_fn, out_dir=test_dir, expected_length=5)
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_dcawrapper_calculate_scores_protein3(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_calculate_scores( query='seq3', aln_file=protein_aln_fn, out_dir=test_dir, expected_length=5)
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_dcawrapper_calculate_scores_dna1(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_calculate_scores(query='seq1', aln_file=dna_aln_fn, out_dir=test_dir, expected_length=9,
                                       polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_dcawrapper_calculate_scores_dna2(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_calculate_scores(query='seq2', aln_file=dna_aln_fn, out_dir=test_dir, expected_length=15,
                                       polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_dcawrapper_calculate_scores_dna3(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_calculate_scores(query='seq3', aln_file=dna_aln_fn, out_dir=test_dir, expected_length=15,
                                       polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)


if __name__ == '__main__':
    unittest.main()