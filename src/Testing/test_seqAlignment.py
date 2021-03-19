"""
Created on Nov 9, 2018

@author: daniel
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd
from shutil import rmtree
from datetime import datetime
from Bio.Alphabet import Gapped
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
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

from SupportingClasses.utils import build_mapping
from SupportingClasses.SeqAlignment import SeqAlignment
from SupportingClasses.FrequencyTable import FrequencyTable
from SupportingClasses.AlignmentDistanceCalculator import AlignmentDistanceCalculator
from SupportingClasses.EvolutionaryTraceAlphabet import FullIUPACProtein, FullIUPACDNA, MultiPositionAlphabet
from Testing.test_Base import (generate_temp_fn, write_out_temp_fn, protein_short_seq, protein_seq1, protein_seq2,
                               protein_seq3, protein_seq4, dna_seq1, dna_seq2, dna_seq3)


# single_protein_seq = '>test\nMET\n'
single_protein_seq = f'>test\n{protein_seq1[0:3].seq}\n'
# two_protein_seqs = '>test\nMET---\n>seq_1\nM-TREE\n'
two_protein_seqs = f'>test\n{protein_seq1.seq}\n>seq_1\n{protein_seq2.seq}\n'
# third_protein_seq = '>seq_2\nM-FREE\n'
third_protein_seq = f'>seq_2\n{protein_seq3.seq}\n'
# fully_gapped_protein_seqs = '>test\n------\n>seq_1\nM-TREE\n'
fully_gapped_protein_seqs = f'>test\n{protein_seq4.seq}\n>seq_1\n{protein_seq2.seq}\n'
# single_dna_seq = '>test\nATGGAGACT\n'
single_dna_seq = f'>test\n{dna_seq1.seq[0:9]}\n'
# two_dna_seqs = '>test\nATGGAGACT---------\n>seq_1\nATG---ACTAGAGAGGAG\n'
two_dna_seqs = f'>test\n{dna_seq1.seq}\n>seq_1\n{dna_seq2.seq}\n'


class TestSeqAlignmentInit(TestCase):

    def evaluate_init(self, aln, q_fn, q_id, q_type, q_alpha):
        self.assertTrue(aln.file_name.endswith(q_fn))
        self.assertEqual(aln.query_id, q_id)
        self.assertIsNone(aln.alignment)
        self.assertIsNone(aln.seq_order)
        self.assertIsNone(aln.query_sequence)
        self.assertIsNone(aln.seq_length)
        self.assertIsNone(aln.size)
        self.assertIsNone(aln.marked)
        self.assertEqual(aln.polymer_type, q_type)
        self.assertEqual(aln.alphabet.size, q_alpha.size)
        self.assertEqual(aln.alphabet.letters, q_alpha.letters)

    def test_init_polymer_type_protein(self):
        fn = generate_temp_fn(suffix='fasta')
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        self.evaluate_init(aln=aln, q_fn=fn, q_id='test', q_type='Protein', q_alpha=FullIUPACProtein())

    def test_init_polymer_type_dna(self):
        fn = generate_temp_fn(suffix='fasta')
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        self.evaluate_init(aln=aln, q_fn=fn, q_id='test', q_type='DNA', q_alpha=FullIUPACDNA())

    def test_init_polymer_type_failure(self):
        fn = generate_temp_fn(suffix='fasta')
        with self.assertRaises(ValueError):
            SeqAlignment(file_name=fn, query_id='test', polymer_type='RNA')


class TestSeqAlignmentAlignmentIO(TestCase):

    def evaluate_import_alignment(self, aln, expected_seqs, expected_ids, expected_query, expected_seq_len,
                                  expected_size, expected_marks):
        self.assertIsNotNone(aln.alignment)
        for ind, seq in enumerate(expected_seqs):
            self.assertEqual(aln.alignment[ind].seq, seq)
        self.assertEqual(aln.seq_order, expected_ids)
        self.assertEqual(aln.query_sequence, expected_query)
        self.assertEqual(aln.seq_length, expected_seq_len)
        self.assertEqual(aln.size, expected_size)
        self.assertEqual(aln.marked, expected_marks)

    def test_import_single_protein_seq(self):
        fn = generate_temp_fn(suffix='fasta')
        write_out_temp_fn(suffix='fasta', out_str=single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        self.evaluate_import_alignment(aln=aln, expected_seqs=[protein_seq1.seq[0:3]], expected_ids=['test'],
                                       expected_query=protein_seq1.seq[0:3], expected_seq_len=3, expected_size=1,
                                       expected_marks=[False])
        os.remove(fn)

    def test_import_multiple_protein_seqs(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        self.evaluate_import_alignment(aln=aln, expected_seqs=[protein_seq1.seq, protein_seq2.seq],
                                       expected_ids=['test', 'seq_1'], expected_query=protein_seq1.seq,
                                       expected_seq_len=6, expected_size=2, expected_marks=[False, False])
        os.remove(fn)

    def test_import_multiple_protein_seqs_save(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        save_fn = f'temp_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.pkl'
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment(save_fn)
        os.remove(fn)
        aln2 = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln2.import_alignment(save_fn)
        self.evaluate_import_alignment(aln=aln2, expected_seqs=[protein_seq1.seq, protein_seq2.seq],
                                       expected_ids=['test', 'seq_1'], expected_query=protein_seq1.seq,
                                       expected_seq_len=6, expected_size=2, expected_marks=[False, False])
        os.remove(save_fn)

    def test_import_single_dna_seq(self):
        fn = generate_temp_fn(suffix='fasta')
        write_out_temp_fn(suffix='fasta', out_str=single_dna_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        self.evaluate_import_alignment(aln=aln, expected_seqs=[dna_seq1.seq[0:9]], expected_ids=['test'],
                                       expected_query=dna_seq1.seq[0:9], expected_seq_len=9, expected_size=1,
                                       expected_marks=[False])
        os.remove(fn)

    def test_import_multiple_dna_seqs(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        self.evaluate_import_alignment(aln=aln, expected_seqs=[dna_seq1.seq, dna_seq2.seq],
                                       expected_ids=['test', 'seq_1'], expected_query=dna_seq1.seq, expected_seq_len=18,
                                       expected_size=2, expected_marks=[False, False])
        os.remove(fn)

    def test_import_multiple_dna_seqs_save(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_dna_seqs)
        save_fn = f'temp_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.pkl'
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment(save_fn)
        os.remove(fn)
        aln2 = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln2.import_alignment(save_fn)
        self.evaluate_import_alignment(aln=aln2, expected_seqs=[dna_seq1.seq, dna_seq2.seq],
                                       expected_ids=['test', 'seq_1'], expected_query=dna_seq1.seq, expected_seq_len=18,
                                       expected_size=2, expected_marks=[False, False])
        os.remove(save_fn)

    def test_write_multiple_protein_seqs(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        os.remove(fn)
        aln.write_out_alignment(fn)
        aln2 = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln2.import_alignment()
        self.evaluate_import_alignment(aln=aln2, expected_seqs=[protein_seq1.seq, protein_seq2.seq],
                                       expected_ids=['test', 'seq_1'], expected_query=protein_seq1.seq,
                                       expected_seq_len=6, expected_size=2, expected_marks=[False, False])
        os.remove(fn)

    def test_write_multiple_dna_seqs(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        os.remove(fn)
        aln.write_out_alignment(fn)
        aln2 = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln2.import_alignment()
        self.evaluate_import_alignment(aln=aln2, expected_seqs=[dna_seq1.seq, dna_seq2.seq],
                                       expected_ids=['test', 'seq_1'], expected_query=dna_seq1.seq,
                                       expected_seq_len=18, expected_size=2, expected_marks=[False, False])
        os.remove(fn)


class TestSeqAlignmentSubAlignmentMethods(TestSeqAlignmentAlignmentIO):

    def test_generate_sub_aln_1(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        sub_aln = aln.generate_sub_alignment(['test'])
        self.evaluate_import_alignment(aln=sub_aln, expected_seqs=[protein_seq1.seq], expected_ids=['test'],
                                       expected_query=protein_seq1.seq, expected_seq_len=6, expected_size=1,
                                       expected_marks=[False])
        os.remove(fn)

    def test_generate_sub_aln_0(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        sub_aln = aln.generate_sub_alignment([])
        self.evaluate_import_alignment(aln=sub_aln, expected_seqs=[], expected_ids=[], expected_query=protein_seq1.seq,
                                       expected_seq_len=6, expected_size=0, expected_marks=[])
        os.remove(fn)

    def test_subset_columns(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        sub_aln = aln._subset_columns([0, 1, 2])
        self.assertIsNotNone(sub_aln)
        self.assertEqual(sub_aln[0].seq, 'MET')
        self.assertEqual(sub_aln[1].seq, 'M-T')
        self.assertEqual(sub_aln[0].id, 'test')
        self.assertEqual(sub_aln[1].id, 'seq_1')
        os.remove(fn)

    def test_subset_columns_fail_negative(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        sub_aln = aln._subset_columns([-3, -2, -1])
        self.assertIsNone(sub_aln)
        os.remove(fn)

    def test_subset_columns_fail_positive(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        sub_aln = aln._subset_columns([10, 11, 12])
        self.assertIsNone(sub_aln)
        os.remove(fn)

    def test_remove_gaps_ungapped(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        sub_aln = aln.remove_gaps()
        self.evaluate_import_alignment(aln=sub_aln, expected_seqs=[protein_seq1.seq[0:3]], expected_ids=['test'],
                                       expected_query=protein_seq1.seq[0:3], expected_seq_len=3, expected_size=1,
                                       expected_marks=[False])
        os.remove(fn)

    def test_remove_gaps_gapped(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        sub_aln = aln.remove_gaps()
        self.evaluate_import_alignment(aln=sub_aln, expected_seqs=[protein_seq1.seq[0:3], protein_seq2.seq[0:3]],
                                       expected_ids=['test', 'seq_1'], expected_query=protein_seq1.seq[0:3],
                                       expected_seq_len=3, expected_size=2, expected_marks=[False, False])
        os.remove(fn)

    def test_remove_gaps_fully_gapped(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=fully_gapped_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        sub_aln = aln.remove_gaps()
        self.evaluate_import_alignment(aln=sub_aln, expected_seqs=[protein_seq4.seq, protein_seq2.seq],
                                       expected_ids=['test', 'seq_1'], expected_query=protein_seq4.seq,
                                       expected_seq_len=6, expected_size=2, expected_marks=[False, False])
        os.remove(fn)

    def test_remove_bad_sequences_all_allowed(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        sub_aln = aln.remove_bad_sequences()
        self.evaluate_import_alignment(aln=sub_aln, expected_seqs=[dna_seq1.seq, dna_seq2.seq],
                                       expected_ids=['test', 'seq_1'], expected_query=dna_seq1.seq,
                                       expected_seq_len=18, expected_size=2, expected_marks=[False, False])
        os.remove(fn)

    def test_remove_bad_sequences_one_removed(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        aln.alphabet.letters = ''.join(list(set(aln.alphabet.letters) - set('R')))
        sub_aln = aln.remove_bad_sequences()
        self.evaluate_import_alignment(aln=sub_aln, expected_seqs=[protein_seq1.seq],
                                       expected_ids=['test'], expected_query=protein_seq1.seq,
                                       expected_seq_len=6, expected_size=1, expected_marks=[False])
        os.remove(fn)

    def test_remove_bad_sequences_all_removed(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        sub_aln = aln.remove_bad_sequences()
        self.evaluate_import_alignment(aln=sub_aln, expected_seqs=[],
                                       expected_ids=[], expected_query=protein_seq1.seq,
                                       expected_seq_len=6, expected_size=0, expected_marks=[])
        os.remove(fn)

    def test_generate_positional_sub_alignment_all(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        aln2 = aln.generate_positional_sub_alignment([0, 1, 2, 3, 4, 5])
        self.evaluate_import_alignment(aln=aln2, expected_seqs=[protein_seq1.seq, protein_seq2.seq],
                                       expected_ids=['test', 'seq_1'], expected_query=protein_seq1.seq,
                                       expected_seq_len=6, expected_size=2, expected_marks=[False, False])
        os.remove(fn)

    def test_generate_positional_sub_alignment_some(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        aln2 = aln.generate_positional_sub_alignment([0, 3])

        self.evaluate_import_alignment(aln=aln2, expected_seqs=[protein_seq1.seq[0] + protein_seq1.seq[3],
                                                                protein_seq2.seq[0] + protein_seq2.seq[3]],
                                       expected_ids=['test', 'seq_1'],
                                       expected_query=protein_seq1.seq[0] + protein_seq1.seq[3],
                                       expected_seq_len=2, expected_size=2, expected_marks=[False, False])
        os.remove(fn)

    def test_generate_positional_sub_alignment_none(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        aln2 = aln.generate_positional_sub_alignment([])
        self.assertIsNone(aln2.alignment)
        self.assertEqual(aln2.seq_order, ['test', 'seq_1'])
        self.assertEqual(aln2.query_sequence, '')
        self.assertEqual(aln2.seq_length, 0)
        self.assertEqual(aln2.size, 2)
        self.assertEqual(aln2.marked, [False, False])
        os.remove(fn)


class TestSeqAlignmentAlignmentToNumeric(TestCase):

    def evaluate_numerical_aln(self, out_str, q_type, q_alpha, expected_array):
        fn = write_out_temp_fn(suffix='fasta', out_str=out_str)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type=q_type)
        aln.import_alignment()
        _, _, mapping, _ = build_mapping(q_alpha)
        num_aln = aln._alignment_to_num(mapping)
        self.assertFalse((num_aln - expected_array).any())
        os.remove(fn)

    def test_single_DNA_seq(self):
        self.evaluate_numerical_aln(out_str=single_dna_seq, q_type='DNA', q_alpha=FullIUPACDNA(),
                                    expected_array=np.array([[0, 1, 3, 3, 0, 3, 0, 2, 1]]))

    def test_two_DNA_seqs(self):
        self.evaluate_numerical_aln(out_str=two_dna_seqs, q_type='DNA', q_alpha=FullIUPACDNA(),
                                    expected_array=np.array([[0, 1, 3, 3, 0, 3, 0, 2, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                                                             [0, 1, 3, 4, 4, 4, 0, 2, 1, 0, 3, 0, 3, 0, 3, 3, 0, 3]]))

    def test_single_protein_seq(self):
        self.evaluate_numerical_aln(out_str=single_protein_seq, q_type='Protein', q_alpha=FullIUPACProtein(),
                                    expected_array=np.array([[11, 4, 17]]))

    def test_two_protein_seqs(self):
        self.evaluate_numerical_aln(out_str=two_protein_seqs, q_type='Protein', q_alpha=FullIUPACProtein(),
                                    expected_array=np.array([[11, 4, 17, 23, 23, 23], [11, 23, 17, 15, 4, 4]]))

    def test_no_alignment_failure(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        _, _, mapping, _ = build_mapping(aln.alphabet)
        with self.assertRaises(TypeError):
            aln._alignment_to_num(mapping)
        os.remove(fn)

    def test_no_mapping_failure(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        with self.assertRaises(TypeError):
            aln._alignment_to_num(None)
        os.remove(fn)

    def test_bad_alphabet_failure(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        _, _, mapping, _ = build_mapping(FullIUPACDNA())
        with self.assertRaises(KeyError):
            aln._alignment_to_num(mapping)
        os.remove(fn)

    def test_bad_alphabet_success(self):
        self.evaluate_numerical_aln(out_str=single_dna_seq, q_type='DNA', q_alpha=FullIUPACProtein(),
                                    expected_array=np.array([[0, 17, 6, 6, 0, 6, 0, 2, 17]]))


class TestSeqAlignmentComputeEffectiveAlignmentSize(TestCase):

    def test_compute_effective_aln_size_permissive_threshold(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        effective_size = aln.compute_effective_alignment_size(identity_threshold=1.0)
        self.assertEqual(effective_size, 2)
        os.remove(fn)

    def test_compute_effective_aln_size_equal_threshold(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        effective_size = aln.compute_effective_alignment_size(identity_threshold=(2 / float(6)) + np.finfo(float).min)
        self.assertEqual(effective_size, 1)
        os.remove(fn)

    def test_compute_effective_aln_size_restrictive_threshold(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        effective_size = aln.compute_effective_alignment_size(identity_threshold=np.finfo(float).min)
        self.assertEqual(effective_size, 1)
        os.remove(fn)

    def test_compute_effective_aln_size_multiple_processors(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        effective_size = aln.compute_effective_alignment_size(identity_threshold=np.finfo(float).min, processes=2)
        self.assertEqual(effective_size, 1)
        os.remove(fn)

    def test_compute_effective_aln_size_with_distance_matrix(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        calculator = AlignmentDistanceCalculator(protein=(aln.polymer_type == 'Protein'))
        distance_matrix = calculator.get_distance(aln.alignment)
        effective_size = aln.compute_effective_alignment_size(identity_threshold=np.finfo(float).min,
                                                              distance_matrix=distance_matrix)
        self.assertEqual(effective_size, 1)
        os.remove(fn)

    def test_compute_effective_aln_size_with_distance_matrix_and_multiple_processors(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        calculator = AlignmentDistanceCalculator(protein=(aln.polymer_type == 'Protein'))
        distance_matrix = calculator.get_distance(aln.alignment)
        effective_size = aln.compute_effective_alignment_size(identity_threshold=np.finfo(float).min,
                                                              distance_matrix=distance_matrix, processes=2)
        self.assertEqual(effective_size, 1)
        os.remove(fn)

    def test_compute_effective_aln_size_none_threshold(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        with self.assertRaises(TypeError):
            aln.compute_effective_alignment_size(identity_threshold=None)
        os.remove(fn)

    def test_compute_effective_aln_size_bad_threshold(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        with self.assertRaises(TypeError):
            aln.compute_effective_alignment_size(identity_threshold='A')
        os.remove(fn)


class TestSeqAlignmentDetermineUsablePositions(TestCase):

    def evaluate_determine_usable_positions(self, out_str, threshold, expected_pos, expected_counts):
        fn = write_out_temp_fn(suffix='fasta', out_str=out_str)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        pos, count = aln.determine_usable_positions(ratio=threshold)
        self.assertFalse((pos - expected_pos).any())
        self.assertFalse((count - expected_counts).any())
        os.remove(fn)

    def test_determine_usable_positions_permissive_ratio(self):
        self.evaluate_determine_usable_positions(out_str=two_protein_seqs, threshold=1.0,
                                                 expected_pos=np.array([0, 1, 2, 3, 4, 5]),
                                                 expected_counts=np.array([2, 1, 2, 1, 1, 1]))

    def test_determine_usable_positions_equal_ratio(self):
        self.evaluate_determine_usable_positions(out_str=two_protein_seqs, threshold=0.5,
                                                 expected_pos=np.array([0, 1, 2, 3, 4, 5]),
                                                 expected_counts=np.array([2, 1, 2, 1, 1, 1]))

    def test_determine_usable_positions_restrictive_ratio(self):
        self.evaluate_determine_usable_positions(out_str=two_protein_seqs, threshold=np.finfo(float).min,
                                                 expected_pos=np.array([]),
                                                 expected_counts=np.array([2, 1, 2, 1, 1, 1]))


class TestSeqAlignmentIdentifyComparableSequences(TestCase):

    def evaluate_identify_comparable_sequences(self, out_str, pos1, pos2, expected_shape1, expected_array1,
                                               expected_shape2, expected_array2, expected_indices, expected_count):
        fn = write_out_temp_fn(suffix='fasta', out_str=out_str)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        col1, col2, indices, count = aln.identify_comparable_sequences(pos1=pos1, pos2=pos2)
        self.assertEqual(col1.shape, (expected_shape1, ))
        self.assertTrue(all([col1[i] == expected_array1[i] for i in range(expected_shape1)]))
        self.assertEqual(col2.shape, (expected_shape2,))
        self.assertTrue(all([col2[i] == expected_array2[i] for i in range(expected_shape2)]))
        self.assertFalse((indices - expected_indices).any())
        self.assertEqual(count, expected_count)
        os.remove(fn)

    def test_comparable_positions_all_comparable(self):
        self.evaluate_identify_comparable_sequences(out_str=two_protein_seqs, pos1=0, pos2=2, expected_shape1=2,
                                                    expected_array1=np.array(['M', 'M']), expected_shape2=2,
                                                    expected_array2=np.array(['T', 'T']),
                                                    expected_indices=np.array([0, 1]), expected_count=2)

    def test_comparable_positions_some_comparable1(self):
        self.evaluate_identify_comparable_sequences(out_str=two_protein_seqs, pos1=0, pos2=1, expected_shape1=1,
                                                    expected_array1=np.array(['M']), expected_shape2=1,
                                                    expected_array2=np.array(['E']), expected_indices=np.array([0]),
                                                    expected_count=1)

    def test_comparable_positions_some_comparable2(self):
        self.evaluate_identify_comparable_sequences(out_str=two_protein_seqs, pos1=0, pos2=5, expected_shape1=1,
                                                    expected_array1=np.array(['M']), expected_shape2=1,
                                                    expected_array2=np.array(['E']), expected_indices=np.array([1]),
                                                    expected_count=1)

    def test_comparable_positions_none(self):
        self.evaluate_identify_comparable_sequences(out_str=two_protein_seqs, pos1=1, pos2=3, expected_shape1=0,
                                                    expected_array1=np.array([]), expected_shape2=0,
                                                    expected_array2=np.array([]), expected_indices=np.array([]),
                                                    expected_count=0)

    def test_comparable_positions_out_of_range(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        with self.assertRaises(IndexError):
            aln.identify_comparable_sequences(pos1=1, pos2=6)
        os.remove(fn)

    def test_comparable_flipped_positions_out_of_range(self):
        self.evaluate_identify_comparable_sequences(out_str=two_protein_seqs, pos1=2, pos2=0, expected_shape1=2,
                                                    expected_array1=np.array(['T', 'T']), expected_shape2=2,
                                                    expected_array2=np.array(['M', 'M']),
                                                    expected_indices=np.array([0, 1]), expected_count=2)


class TestSeqAlignmentConsensusSequence(TestCase):

    def evaluate_consensus_sequence(self, out_str, expected_consensus):
        fn = write_out_temp_fn(suffix='fasta', out_str=out_str)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        consensus_rec = aln.consensus_sequence(method='majority')
        self.assertEqual(consensus_rec.id, 'Consensus Sequence')
        self.assertEqual(consensus_rec.seq, expected_consensus)
        os.remove(fn)

    def test_consensus_sequence_trivial(self):
        self.evaluate_consensus_sequence(out_str=single_protein_seq + single_protein_seq, expected_consensus='MET')

    def test_consensus_sequence(self):
        self.evaluate_consensus_sequence(out_str=two_protein_seqs + third_protein_seq, expected_consensus='M-TREE')

    def test_consensus_sequence_ties(self):
        self.evaluate_consensus_sequence(out_str=two_protein_seqs + two_protein_seqs, expected_consensus='METREE')

    def test_consensus_sequence_method_failure(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        with self.assertRaises(ValueError):
            aln.consensus_sequence(method='average')
        os.remove(fn)


class TestSeqAlignmentGapZScoreCheck(TestCase):

    def evaluate_gap_z_score_check(self, out_str, q_type, cutoff, expected_passing):
        fn = write_out_temp_fn(suffix='fasta', out_str=out_str)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type=q_type)
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        overly_gapped_seqs = aln._gap_z_score_check(cutoff, num_aln, alpha_size)
        self.assertFalse(((1 * overly_gapped_seqs) - (1 * expected_passing)).any())
        os.remove(fn)

    def test_gap_z_score_single_seq(self):
        self.evaluate_gap_z_score_check(out_str=single_dna_seq, q_type='DNA', cutoff=2.0,
                                        expected_passing=np.array([True]))

    def test_gap_z_score_two_seq(self):
        self.evaluate_gap_z_score_check(out_str=two_dna_seqs, q_type='DNA', cutoff=0.5,
                                        expected_passing=np.array([False, True]))

    def test_gap_z_score_three_seq(self):
        self.evaluate_gap_z_score_check(out_str=two_protein_seqs + third_protein_seq, q_type='Protein', cutoff=1.0,
                                        expected_passing=[False, True, True])

    def test_gap_z_score_no_cutoff_failure(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        with self.assertRaises(ValueError):
            aln._gap_z_score_check(None, num_aln, alpha_size)
        os.remove(fn)

    def test_gap_z_score_no_num_aln_failure(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        with self.assertRaises(ValueError):
            aln._gap_z_score_check(2.0, None, alpha_size)
        os.remove(fn)

    def test_gap_z_score_no_gap_num(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        with self.assertRaises(ValueError):
            aln._gap_z_score_check(2.0, num_aln, None)
        os.remove(fn)


class TestSeqAlignmentGapPercentileCheck(TestCase):

    def evaluate_gap_percentile_check(self, out_str, q_type, cutoff, expected_passing):
        fn = write_out_temp_fn(suffix='fasta', out_str=out_str)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type=q_type)
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        overly_gapped_seqs = aln._gap_percentile_check(cutoff, num_aln, alpha_size, mapping)
        self.assertFalse(((1 * overly_gapped_seqs) - (1 * expected_passing)).any())
        os.remove(fn)

    def test_gap_percentile_check_single_seq(self):
        self.evaluate_gap_percentile_check(out_str=single_dna_seq, q_type='DNA', cutoff=0.5,
                                           expected_passing=np.array([True]))

    def test_gap_percentile_check_two_seq(self):
        self.evaluate_gap_percentile_check(out_str=two_dna_seqs, q_type='DNA', cutoff=0.5,
                                           expected_passing=np.array([False, True]))

    def test_gap_percentile_check_three_seq(self):
        self.evaluate_gap_percentile_check(out_str=two_protein_seqs + third_protein_seq, q_type='Protein', cutoff=0.5,
                                           expected_passing=np.array([False, True, True]))

    def test_gap_percentile_check_no_cutoff_failure(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        with self.assertRaises(ValueError):
            aln._gap_percentile_check(None, num_aln, alpha_size, mapping)
        os.remove(fn)

    def test_gap_percentile_check_no_num_aln_failure(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        with self.assertRaises(ValueError):
            aln._gap_percentile_check(2.0, None, alpha_size, mapping)
        os.remove(fn)

    def test_gap_percentile_check_no_gap_num(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        with self.assertRaises(ValueError):
            aln._gap_percentile_check(2.0, num_aln, None, mapping)
        os.remove(fn)

    def test_gap_percentile_check_no_mapping(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        with self.assertRaises(ValueError):
            aln._gap_percentile_check(2.0, num_aln, alpha_size, None)
        os.remove(fn)


class TestSeqAlignmentGapEvaluation(TestCase):

    def evaluate_gap_evaluation(self, out_str, q_type, size_cutoff, z_score_cutoff, percentile_cutoff, expected_passing,
                                expected_failing):
        fn = write_out_temp_fn(suffix='fasta', out_str=out_str)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type=q_type)
        aln.import_alignment()
        passing, not_passing = aln.gap_evaluation(size_cutoff=size_cutoff, z_score_cutoff=z_score_cutoff,
                                                  percentile_cutoff=percentile_cutoff)
        self.assertEqual(passing, expected_passing)
        self.assertEqual(not_passing, expected_failing)
        os.remove(fn)

    def test_gap_evaluation_z_score_single_seq(self):
        self.evaluate_gap_evaluation(out_str=single_dna_seq, q_type='DNA', size_cutoff=0, z_score_cutoff=2.0,
                                     percentile_cutoff=None, expected_passing=['test'], expected_failing=[])

    def test_gap_evaluation_z_score_two_seq(self):
        self.evaluate_gap_evaluation(out_str=two_dna_seqs, q_type='DNA', size_cutoff=1, z_score_cutoff=0.5,
                                     percentile_cutoff=None, expected_passing=['seq_1'], expected_failing=['test'])

    def test_gap_evaluation_z_score_three_seq(self):
        self.evaluate_gap_evaluation(out_str=two_protein_seqs + third_protein_seq, q_type='Protein', size_cutoff=2,
                                     z_score_cutoff=1.0, percentile_cutoff=None, expected_passing=['seq_1', 'seq_2'],
                                     expected_failing=['test'])

    def test_gap_evaluation_percentile_check_single_seq(self):
        self.evaluate_gap_evaluation(out_str=single_dna_seq, q_type='DNA', size_cutoff=1, z_score_cutoff=None,
                                     percentile_cutoff=0.5, expected_passing=['test'], expected_failing=[])

    def test_gap_evaluation_percentile_check_two_seq(self):
        self.evaluate_gap_evaluation(out_str=two_dna_seqs, q_type='DNA', size_cutoff=2, z_score_cutoff=None,
                                     percentile_cutoff=0.5, expected_passing=['seq_1'], expected_failing=['test'])

    def test_gap_evaluation_percentile_check_three_seq(self):
        self.evaluate_gap_evaluation(out_str=two_protein_seqs + third_protein_seq, q_type='Protein', size_cutoff=3,
                                     z_score_cutoff=None, percentile_cutoff=0.5, expected_passing=['seq_1', 'seq_2'],
                                     expected_failing=['test'])

    def test_gap_evaluation_no_size_cutoff_failure(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        with self.assertRaises(ValueError):
            aln.gap_evaluation(None, z_score_cutoff=2.0, percentile_cutoff=0.5)
        os.remove(fn)

    def test_gap_evaluation_no_z_score_cutoff_failure(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        with self.assertRaises(ValueError):
            aln.gap_evaluation(size_cutoff=1, z_score_cutoff=None, percentile_cutoff=0.5)
        os.remove(fn)

    def test_gap_evaluation_no_percentile_cutoff_failure(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        with self.assertRaises(ValueError):
            aln.gap_evaluation(size_cutoff=2, z_score_cutoff=2.0, percentile_cutoff=None)
        os.remove(fn)


class TestSeqAlignmentHeatmapPlot(TestCase):

    def evaluate_heatmap_plot(self, out_str, q_type, name, out_dir, save, ax, expected_values, expected_labels):
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            save_fn = os.path.join(os.getcwd(), out_dir, 'Visualization_Test.eps')
        else:
            save_fn = os.path.join(os.getcwd(), 'Visualization_Test.eps')
        fn = write_out_temp_fn(suffix='fasta', out_str=out_str)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type=q_type)
        aln.import_alignment()
        df, ret_ax = aln.heatmap_plot(name=name, out_dir=out_dir, save=save, ax=ax)
        self.assertTrue(df.equals(pd.DataFrame(expected_values, index=aln.seq_order,
                                               columns=expected_labels)))
        self.assertEqual([label.get_text() for label in ret_ax.get_xticklabels()], expected_labels)
        self.assertEqual([label.get_text() for label in ret_ax.get_yticklabels()], aln.seq_order)
        self.assertEqual(ret_ax.title.get_text(), name)
        if ax:
            self.assertTrue(ret_ax is ax)
        plt.clf()
        os.remove(fn)
        if save:
            self.assertTrue(os.path.isfile(save_fn))
            os.remove(save_fn)
        else:
            self.assertFalse(os.path.isfile(save_fn))
        if out_dir:
            rmtree(out_dir)

    def test_heatmap_plot_only_name(self):
        self.evaluate_heatmap_plot(out_str=single_protein_seq, q_type='Protein', name='Visualization Test',
                                   out_dir=None, save=False, ax=None, expected_values=np.array([[11, 4, 17]]),
                                   expected_labels=['0:M', '1:E', '2:T'])

    def test_heatmap_plot_dir_no_save(self):
        self.evaluate_heatmap_plot(out_str=single_protein_seq, q_type='Protein', name='Visualization Test',
                                   out_dir='plot_test', save=False, ax=None, expected_values=np.array([[11, 4, 17]]),
                                   expected_labels=['0:M', '1:E', '2:T'])

    def test_heatmap_plot_dir_save(self):
        self.evaluate_heatmap_plot(out_str=single_protein_seq, q_type='Protein', name='Visualization Test',
                                   out_dir='plot_test', save=True, ax=None, expected_values=np.array([[11, 4, 17]]),
                                   expected_labels=['0:M', '1:E', '2:T'])

    def test_heatmap_plot_no_dir_save(self):
        self.evaluate_heatmap_plot(out_str=single_protein_seq, q_type='Protein', name='Visualization Test',
                                   out_dir=None, save=True, ax=None, expected_values=np.array([[11, 4, 17]]),
                                   expected_labels=['0:M', '1:E', '2:T'])

    def test_heatmap_plot_dir_no_save_custom_ax(self):
        _, original_ax = plt.subplots(1)
        self.evaluate_heatmap_plot(out_str=single_protein_seq, q_type='Protein', name='Visualization Test',
                                   out_dir='plot_test', save=False, ax=original_ax,
                                   expected_values=np.array([[11, 4, 17]]), expected_labels=['0:M', '1:E', '2:T'])

    def test_heatmap_plot_dir_save_custom_ax(self):
        _, original_ax = plt.subplots(1)
        self.evaluate_heatmap_plot(out_str=single_protein_seq, q_type='Protein', name='Visualization Test',
                                   out_dir='plot_test', save=True, ax=original_ax,
                                   expected_values=np.array([[11, 4, 17]]), expected_labels=['0:M', '1:E', '2:T'])

    def test_heatmap_plot_no_dir_save_custom_ax(self):
        _, original_ax = plt.subplots(1)
        self.evaluate_heatmap_plot(out_str=single_protein_seq, q_type='Protein', name='Visualization Test',
                                   out_dir=None, save=True, ax=original_ax, expected_values=np.array([[11, 4, 17]]),
                                   expected_labels=['0:M', '1:E', '2:T'])

    def test_heatmap_plot_save_no_overwrite(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        df, ax = aln.heatmap_plot(name='Visualization Test', out_dir=None, save=True, ax=None)
        self.assertTrue(os.path.isfile(os.path.join(os.getcwd(), 'Visualization_Test.eps')))
        df2, ax2 = aln.heatmap_plot(name='Visualization Test', out_dir=None, save=True, ax=None)
        self.assertIsNone(df2)
        self.assertIsNone(ax2)
        plt.clf()
        os.remove(fn)
        os.remove('Visualization_Test.eps')

    def test_heatmap_plot_no_save_overwrite(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        df, ax = aln.heatmap_plot(name='Visualization Test', out_dir=None, save=False, ax=None)
        self.assertTrue(df.equals(pd.DataFrame(np.array([[11, 4, 17]]), index=aln.seq_order,
                                               columns=['0:M', '1:E', '2:T'])))
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ['0:M', '1:E', '2:T'])
        self.assertEqual([label.get_text() for label in ax.get_yticklabels()], aln.seq_order)
        self.assertEqual(ax.title.get_text(), 'Visualization Test')
        df2, ax2 = aln.heatmap_plot(name='Visualization Test', out_dir=None, save=True, ax=None)
        self.assertTrue(df2.equals(pd.DataFrame(np.array([[11, 4, 17]]), index=aln.seq_order,
                                                columns=['0:M', '1:E', '2:T'])))
        self.assertEqual([label.get_text() for label in ax2.get_xticklabels()], ['0:M', '1:E', '2:T'])
        self.assertEqual([label.get_text() for label in ax2.get_yticklabels()], aln.seq_order)
        self.assertEqual(ax2.title.get_text(), 'Visualization Test')
        plt.clf()
        os.remove(fn)
        os.remove('Visualization_Test.eps')

    def test_heatmap_plot_no_name_failure(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=single_dna_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        with self.assertRaises(ValueError):
            aln.heatmap_plot(name=None, out_dir=None, save=False, ax=None)
        os.remove(fn)


class TestCharacterization(TestCase):

    def setUp(self):
        self.single_alpha_size, _, self.single_mapping, self.single_reverse = build_mapping(Gapped(FullIUPACProtein()))
        self.protein_single_alpha_one_seq_pos = np.array([0, 1, 2])
        self.protein_single_alpha_one_seq_chars = np.array([['M'], ['E'], ['T']])
        self.protein_single_alpha_one_seq_counts = csc_matrix(([1, 1, 1], (self.protein_single_alpha_one_seq_pos,
                                                               [self.single_mapping[x[0]]
                                                                for x in self.protein_single_alpha_one_seq_chars])),
                                                              shape=(3, self.single_alpha_size))
        self.pair_alpha_size, _, self.pair_mapping, self.pair_reverse = build_mapping(MultiPositionAlphabet(Gapped(
            FullIUPACProtein()), size=2))

        self.protein_pair_alpha_one_seq_pos = np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])
        self.protein_pair_alpha_one_seq_chars = np.array([['MM'], ['ME'], ['MT'], ['EE'], ['ET'], ['TT']])
        self.protein_pair_alpha_one_seq_counts = csc_matrix(([1, 1, 1, 1, 1, 1],
                                                             (list(range(len(self.protein_pair_alpha_one_seq_pos))),
                                                              [self.pair_mapping[x[0]] for x in
                                                               self.protein_pair_alpha_one_seq_chars])),
                                                            shape=(len(self.protein_pair_alpha_one_seq_pos),
                                                                   self.pair_alpha_size))
        self.s_to_p = np.zeros((self.single_alpha_size + 1, self.single_alpha_size + 1))
        for char in self.pair_mapping:
            self.s_to_p[self.single_mapping[char[0]], self.single_mapping[char[1]]] = self.pair_mapping[char]

    def evaluate_characterize_positions_1_and_2(self, out_str, q_type, single, pair, single_size, single_mapping,
                                                single_reverse, pair_size, pair_mapping, pair_reverse,
                                                expected_single_positions, expected_single_chars,
                                                expected_single_counts, expected_pair_positions, expected_pair_chars,
                                                expected_pair_counts, method, single_to_pair=None):
        fn = write_out_temp_fn(suffix='fasta', out_str=out_str)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type=q_type)
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(Gapped(aln.alphabet))
        if method == 1:
            s_res, p_res = aln.characterize_positions(single=single, pair=pair, single_size=single_size,
                                                      single_mapping=single_mapping, single_reverse=single_reverse,
                                                      pair_size=pair_size, pair_mapping=pair_mapping,
                                                      pair_reverse=pair_reverse)
        elif method == 2:
            s_res, p_res = aln.characterize_positions2(single=single, pair=pair, single_size=single_size,
                                                       single_mapping=single_mapping, single_reverse=single_reverse,
                                                       pair_size=pair_size, pair_mapping=pair_mapping,
                                                       pair_reverse=pair_reverse, single_to_pair=single_to_pair)
        else:
            raise ValueError('Incorrect method specified!')
        if single:
            self.assertIsInstance(s_res, FrequencyTable)
            self.assertFalse((s_res.get_positions() - expected_single_positions).any())
            for i in expected_single_positions:
                self.assertEqual(s_res.get_chars(i), expected_single_chars[i])
            self.assertFalse((s_res.get_count_matrix() - expected_single_counts).any())
        else:
            self.assertIsNone(s_res)
        if pair:
            self.assertIsInstance(p_res, FrequencyTable)
            self.assertFalse((p_res.get_positions() - expected_pair_positions).any())
            for i, curr_pos in enumerate(expected_pair_positions):
                self.assertEqual(p_res.get_chars((curr_pos[0], curr_pos[1])), expected_pair_chars[i])
            self.assertFalse((p_res.get_count_matrix() - expected_pair_counts).any())
        else:
            self.assertIsNone(p_res)
        os.remove(fn)

    def test_single_and_pair_false(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        s_res, p_res = aln.characterize_positions(single=False, pair=False, single_size=None, single_mapping=None,
                                                  single_reverse=None, pair_size=None, pair_mapping=None,
                                                  pair_reverse=None)
        self.assertIsNone(s_res)
        self.assertIsNone(p_res)
        os.remove(fn)

    def test_single_no_inputs(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=False, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=None, pair_mapping=None,
                                                     pair_reverse=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=None, expected_pair_chars=None,
                                                     expected_pair_counts=None, method=1)

    def test_single_size_only(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=False, single_size=self.single_alpha_size,
                                                     single_mapping=None, single_reverse=None, pair_size=None,
                                                     pair_mapping=None, pair_reverse=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=None, expected_pair_chars=None,
                                                     expected_pair_counts=None, method=1)

    def test_single_mapping_only(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=False, single_size=None, single_mapping=self.single_mapping,
                                                     single_reverse=None, pair_size=None, pair_mapping=None,
                                                     pair_reverse=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=None, expected_pair_chars=None,
                                                     expected_pair_counts=None, method=1)

    def test_single_reverse_only(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=False, single_size=None, single_mapping=None,
                                                     single_reverse=self.single_reverse, pair_size=None,
                                                     pair_mapping=None, pair_reverse=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=None, expected_pair_chars=None,
                                                     expected_pair_counts=None, method=1)

    def test_single_all_inputs(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=False, single_size=self.single_alpha_size,
                                                     single_mapping=self.single_mapping,
                                                     single_reverse=self.single_reverse,
                                                     pair_size=None, pair_mapping=None, pair_reverse=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=None, expected_pair_chars=None,
                                                     expected_pair_counts=None, method=1)

    def test_pair_no_inputs(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=False,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=None, pair_mapping=None,
                                                     pair_reverse=None, expected_single_positions=None,
                                                     expected_single_chars=None, expected_single_counts=None,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=1)

    def test_pair_size_only(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=False,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=self.pair_alpha_size,
                                                     pair_mapping=None, pair_reverse=None,
                                                     expected_single_positions=None, expected_single_chars=None,
                                                     expected_single_counts=None,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=1)

    def test_pair_mapping_only(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=False,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=None,
                                                     pair_mapping=self.pair_mapping, pair_reverse=None,
                                                     expected_single_positions=None,
                                                     expected_single_chars=None, expected_single_counts=None,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=1)

    def test_pair_reverse_only(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=False,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=None, pair_mapping=None,
                                                     pair_reverse=self.pair_reverse, expected_single_positions=None,
                                                     expected_single_chars=None, expected_single_counts=None,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=1)

    def test_pair_all_inputs(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=False,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=self.pair_alpha_size,
                                                     pair_mapping=self.pair_mapping, pair_reverse=self.pair_reverse,
                                                     expected_single_positions=None, expected_single_chars=None,
                                                     expected_single_counts=None,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=1)

    def test_single_and_pair_no_inputs(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=None, pair_mapping=None,
                                                     pair_reverse=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=1)

    def test_single_and_pair_size_only(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=True, single_size=self.single_alpha_size, single_mapping=None,
                                                     single_reverse=None, pair_size=self.pair_alpha_size,
                                                     pair_mapping=None, pair_reverse=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=1)

    def test_single_and_pair_mapping_only(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=True, single_size=None, single_mapping=self.single_mapping,
                                                     single_reverse=None, pair_size=None,
                                                     pair_mapping=self.pair_mapping, pair_reverse=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=1)

    def test_single_and_pair_reverse_only(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=self.single_reverse, pair_size=None,
                                                     pair_mapping=None, pair_reverse=self.pair_reverse,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=1)

    def test_single_and_pair_all_inputs(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=True, single_size=self.single_alpha_size,
                                                     single_mapping=self.single_mapping,
                                                     single_reverse=self.single_reverse, pair_size=self.pair_alpha_size,
                                                     pair_mapping=self.pair_mapping, pair_reverse=self.pair_reverse,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=1)

    ####################################################################################################################

    def test_single_and_pair_false2(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        s_res, p_res = aln.characterize_positions2(single=False, pair=False, single_size=None, single_mapping=None,
                                                   single_reverse=None, pair_size=None, pair_mapping=None,
                                                   pair_reverse=None, single_to_pair=None)
        self.assertIsNone(s_res)
        self.assertIsNone(p_res)
        os.remove(fn)

    def test_single_no_inputs2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=False, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=None, pair_mapping=None,
                                                     pair_reverse=None, single_to_pair=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=None, expected_pair_chars=None,
                                                     expected_pair_counts=None, method=2)

    def test_single_size_only2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=False, single_size=self.single_alpha_size,
                                                     single_mapping=None, single_reverse=None, pair_size=None,
                                                     pair_mapping=None, pair_reverse=None, single_to_pair=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=None, expected_pair_chars=None,
                                                     expected_pair_counts=None, method=2)

    def test_single_mapping_only2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=False, single_size=None, single_mapping=self.single_mapping,
                                                     single_reverse=None, pair_size=None, pair_mapping=None,
                                                     pair_reverse=None, single_to_pair=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=None, expected_pair_chars=None,
                                                     expected_pair_counts=None, method=2)

    def test_single_reverse_only2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=False, single_size=None, single_mapping=None,
                                                     single_reverse=self.single_reverse, pair_size=None,
                                                     pair_mapping=None, pair_reverse=None, single_to_pair=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=None, expected_pair_chars=None,
                                                     expected_pair_counts=None, method=2)

    def test_single_all_inputs2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=False, single_size=self.single_alpha_size,
                                                     single_mapping=self.single_mapping,
                                                     single_reverse=self.single_reverse, pair_size=None,
                                                     pair_mapping=None, pair_reverse=None, single_to_pair=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=None, expected_pair_chars=None,
                                                     expected_pair_counts=None, method=2)

    def test_pair_no_inputs2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=False,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=None, pair_mapping=None,
                                                     pair_reverse=None, single_to_pair=None,
                                                     expected_single_positions=None, expected_single_chars=None,
                                                     expected_single_counts=None,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=2)

    def test_pair_size_only2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=False,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=self.pair_alpha_size,
                                                     pair_mapping=None, pair_reverse=None, single_to_pair=None,
                                                     expected_single_positions=None, expected_single_chars=None,
                                                     expected_single_counts=None,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=2)

    def test_pair_mapping_only2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=False,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=None,
                                                     pair_mapping=self.pair_mapping, pair_reverse=None,
                                                     single_to_pair=None, expected_single_positions=None,
                                                     expected_single_chars=None, expected_single_counts=None,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=2)

    def test_pair_reverse_only2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=False,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=None, pair_mapping=None,
                                                     pair_reverse=self.pair_reverse, single_to_pair=None,
                                                     expected_single_positions=None, expected_single_chars=None,
                                                     expected_single_counts=None,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=2)

    def test_pair_single_to_pair_only2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=False,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=None, pair_mapping=None,
                                                     pair_reverse=None, single_to_pair=self.s_to_p,
                                                     expected_single_positions=None, expected_single_chars=None,
                                                     expected_single_counts=None,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=2)

    def test_pair_all_inputs2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=False,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=self.pair_alpha_size,
                                                     pair_mapping=self.pair_mapping, pair_reverse=self.pair_reverse,
                                                     single_to_pair=self.s_to_p, expected_single_positions=None,
                                                     expected_single_chars=None, expected_single_counts=None,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=2)

    def test_single_and_pair_no_inputs2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=None, pair_mapping=None,
                                                     pair_reverse=None, single_to_pair=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=2)

    def test_single_and_pair_size_only2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=True, single_size=self.single_alpha_size, single_mapping=None,
                                                     single_reverse=None, pair_size=self.pair_alpha_size,
                                                     pair_mapping=None, pair_reverse=None, single_to_pair=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=2)

    def test_single_and_pair_mapping_only2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=True, single_size=None, single_mapping=self.single_mapping,
                                                     single_reverse=None, pair_size=None,
                                                     pair_mapping=self.pair_mapping, pair_reverse=None,
                                                     single_to_pair=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=2)

    def test_single_and_pair_reverse_only2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=self.single_reverse, pair_size=None,
                                                     pair_mapping=None, pair_reverse=self.pair_reverse,
                                                     single_to_pair=None,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=2)

    def test_single_and_pair_single_to_pair_only2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=True, single_size=None, single_mapping=None,
                                                     single_reverse=None, pair_size=None, pair_mapping=None,
                                                     pair_reverse=None, single_to_pair=self.s_to_p,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=2)

    def test_single_and_pair_all_inputs2(self):
        self.evaluate_characterize_positions_1_and_2(out_str=single_protein_seq, q_type='Protein', single=True,
                                                     pair=True, single_size=self.single_alpha_size,
                                                     single_mapping=self.single_mapping,
                                                     single_reverse=self.single_reverse, pair_size=self.pair_alpha_size,
                                                     pair_mapping=self.pair_mapping, pair_reverse=self.pair_reverse,
                                                     single_to_pair=self.s_to_p,
                                                     expected_single_positions=self.protein_single_alpha_one_seq_pos,
                                                     expected_single_chars=self.protein_single_alpha_one_seq_chars,
                                                     expected_single_counts=self.protein_single_alpha_one_seq_counts,
                                                     expected_pair_positions=self.protein_pair_alpha_one_seq_pos,
                                                     expected_pair_chars=self.protein_pair_alpha_one_seq_chars,
                                                     expected_pair_counts=self.protein_pair_alpha_one_seq_counts,
                                                     method=2)


if __name__ == '__main__':
    unittest.main()
