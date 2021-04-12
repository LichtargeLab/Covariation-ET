"""
Created on February 5, 2020

@author: Daniel Konecki
"""
import os
import sys
import unittest
from shutil import rmtree
from unittest import TestCase

#
from dotenv import find_dotenv, load_dotenv
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)
source_code_path = os.path.join(os.environ.get('PROJECT_PATH'), 'src')
# Add the project path to the python path so the required clases can be imported
if source_code_path not in sys.path:
    sys.path.append(os.path.join(os.environ.get('PROJECT_PATH'), 'src'))
#

from SupportingClasses.Predictor import Predictor
from Testing.test_Base import protein_seq1, protein_seq2, protein_seq3, dna_seq1, dna_seq2, dna_seq3, write_out_temp_fn


pro_str = f'>{protein_seq1.id}\n{protein_seq1.seq}\n>{protein_seq2.id}\n{protein_seq2.seq}\n>{protein_seq3.id}\n{protein_seq3.seq}'
dna_str = f'>{dna_seq1.id}\n{dna_seq1.seq}\n>{dna_seq2.id}\n{dna_seq2.seq}\n>{dna_seq3.id}\n{dna_seq3.seq}'
test_dir = os.path.join(os.getcwd(), 'TestCase')


class TestPredictorInit(TestCase):

    def evaluate_init(self, query, aln_file, out_dir, expected_length, expected_sequence, polymer_type='Protein'):
        self.assertFalse(os.path.isdir(os.path.abspath(out_dir)))
        pred = Predictor(query=query, aln_file=aln_file, polymer_type=polymer_type, out_dir=out_dir)
        self.assertEqual(pred.out_dir, os.path.abspath(out_dir))
        self.assertTrue(os.path.isdir(os.path.abspath(out_dir)))
        self.assertEqual(pred.query, query)
        self.assertEqual(pred.polymer_type, polymer_type)
        self.assertIsNotNone(pred.original_aln)
        self.assertGreaterEqual(pred.original_aln.seq_length, expected_length)
        self.assertEqual(str(pred.original_aln.query_sequence).replace('-', ''), expected_sequence)
        self.assertEqual(pred.original_aln_fn, os.path.join(out_dir, 'Original_Alignment.fa'))
        self.assertTrue(os.path.isfile(pred.original_aln_fn))
        self.assertIsNotNone(pred.non_gapped_aln)
        self.assertEqual(pred.non_gapped_aln.seq_length, expected_length)
        self.assertEqual(pred.non_gapped_aln.query_sequence, expected_sequence)
        self.assertEqual(pred.non_gapped_aln_fn, os.path.join(out_dir, 'Non-Gapped_Alignment.fa'))
        self.assertTrue(os.path.isfile(pred.non_gapped_aln_fn))
        self.assertEqual(pred.method, 'Base')
        self.assertIsNone(pred.scores)
        self.assertIsNone(pred.coverages)
        self.assertIsNone(pred.rankings)
        self.assertIsNone(pred.time)

    def test_predictor_init_protein_aln_1(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_init(query='seq1', aln_file=protein_aln_fn, out_dir=test_dir, expected_length=3,
                           expected_sequence='MET')
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_predictor_init_protein_aln_2(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_init(query='seq2', aln_file=protein_aln_fn, out_dir=test_dir, expected_length=5,
                           expected_sequence='MTREE')
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_predictor_init_protein_aln_3(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_init(query='seq3', aln_file=protein_aln_fn, out_dir=test_dir, expected_length=5,
                           expected_sequence='MFREE')
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_predictor_init_dna_aln_1(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_init(query='seq1', aln_file=dna_aln_fn, out_dir=test_dir, expected_length=9,
                           expected_sequence='ATGGAGACT', polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_predictor_init_dna_aln_2(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_init(query='seq2', aln_file=dna_aln_fn, out_dir=test_dir, expected_length=15,
                           expected_sequence='ATGACTAGAGAGGAG', polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)

    def test_predictor_init_dna_aln_3(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_init(query='seq3', aln_file=dna_aln_fn, out_dir=test_dir, expected_length=15,
                           expected_sequence='ATGTTTAGAGAGGAG', polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)


class TestPredictorCalculateScores(TestCase):

    def evaluate_calculator_scores(self, query, aln_file, out_dir, polymer_type='Protein'):
        pred = Predictor(query=query, aln_file=aln_file, out_dir=out_dir, polymer_type=polymer_type)
        with self.assertRaises(NotImplementedError):
            pred.calculate_scores()

    def test_predictor_calculate_scores_protein(self):
        protein_aln_fn = write_out_temp_fn(suffix='protein.fasta', out_str=pro_str)
        self.evaluate_calculator_scores(query='seq1', aln_file=protein_aln_fn, out_dir=test_dir)
        os.remove(protein_aln_fn)
        rmtree(test_dir)

    def test_predictor_calculate_scores_dna(self):
        dna_aln_fn = write_out_temp_fn(suffix='dna.fasta', out_str=dna_str)
        self.evaluate_calculator_scores(query='seq1', aln_file=dna_aln_fn, out_dir=test_dir, polymer_type='DNA')
        os.remove(dna_aln_fn)
        rmtree(test_dir)


if __name__ == '__main__':
    unittest.main()