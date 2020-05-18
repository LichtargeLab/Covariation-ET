"""
Created on Nov 9, 2018

@author: daniel
"""
import os
import unittest
import numpy as np
from copy import deepcopy
from shutil import rmtree
from datetime import datetime
from Bio.Seq import Seq
from Bio.Alphabet import Gapped
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from unittest import TestCase
from test_Base import TestBase
from utils import build_mapping
from SeqAlignment import SeqAlignment
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from EvolutionaryTraceAlphabet import FullIUPACProtein, FullIUPACDNA, MultiPositionAlphabet


single_protein_seq = '>test\nMET\n'
two_protein_seqs = '>test\nMET---\n>seq_1\nM-TREE\n'
third_protein_seq = '>seq_2\nM-FREE\n'
fully_gapped_protein_seqs = '>test\n------\n>seq_1\nM-TREE\n'
single_dna_seq = '>test\nATGGAGACT\n'
two_dna_seqs = '>test\nATGGAGACT---------\n>seq_1\nATG---ACTAGAGAGGAG\n'


def generate_temp_fn():
    return f'temp_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.fasta'


def write_out_temp_fasta(out_str=None):
    fn = generate_temp_fn()
    with open(fn, 'a') as handle:
        os.utime(fn)
        if out_str:
            handle.write(out_str)
    return fn


class TestSeqAlignmentInputOutput(TestCase):

    def test_init_polymer_type_protein(self):
        fn = generate_temp_fn()
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        self.assertTrue(aln.file_name.endswith(fn))
        self.assertEqual(aln.query_id, 'test')
        self.assertIsNone(aln.alignment)
        self.assertIsNone(aln.seq_order)
        self.assertIsNone(aln.query_sequence)
        self.assertIsNone(aln.seq_length)
        self.assertIsNone(aln.size)
        self.assertIsNone(aln.marked)
        self.assertEqual(aln.polymer_type, 'Protein')
        expected_alpha = FullIUPACProtein()
        self.assertEqual(aln.alphabet.size, expected_alpha.size)
        self.assertEqual(aln.alphabet.letters, expected_alpha.letters)

    def test_init_polymer_type_dna(self):
        fn = generate_temp_fn()
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        self.assertTrue(aln.file_name.endswith(fn))
        self.assertEqual(aln.query_id, 'test')
        self.assertIsNone(aln.alignment)
        self.assertIsNone(aln.seq_order)
        self.assertIsNone(aln.query_sequence)
        self.assertIsNone(aln.seq_length)
        self.assertIsNone(aln.size)
        self.assertIsNone(aln.marked)
        self.assertEqual(aln.polymer_type, 'DNA')
        expected_alpha = FullIUPACDNA()
        self.assertEqual(aln.alphabet.size, expected_alpha.size)
        self.assertEqual(aln.alphabet.letters, expected_alpha.letters)

    def test_init_polymer_type_failure(self):
        fn = generate_temp_fn()
        with self.assertRaises(ValueError):
            SeqAlignment(file_name=fn, query_id='test', polymer_type='RNA')

    def test_import_single_protein_seq(self):
        fn = generate_temp_fn()
        write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        self.assertIsNotNone(aln.alignment)
        self.assertEqual(aln.alignment[0].seq, 'MET')
        self.assertEqual(aln.seq_order, ['test'])
        self.assertEqual(aln.query_sequence, 'MET')
        self.assertEqual(aln.seq_length, 3)
        self.assertEqual(aln.size, 1)
        self.assertEqual(aln.marked, [False])
        os.remove(fn)

    def test_import_multiple_protein_seqs(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        self.assertIsNotNone(aln.alignment)
        self.assertEqual(aln.alignment[0].seq, 'MET---')
        self.assertEqual(aln.alignment[1].seq, 'M-TREE')
        self.assertEqual(aln.seq_order, ['test', 'seq_1'])
        self.assertEqual(aln.query_sequence, 'MET---')
        self.assertEqual(aln.seq_length, 6)
        self.assertEqual(aln.size, 2)
        self.assertEqual(aln.marked, [False, False])
        os.remove(fn)

    def test_import_multiple_protein_seqs_save(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        save_fn = f'temp_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.pkl'
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment(save_fn)
        os.remove(fn)
        aln2 = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln2.import_alignment(save_fn)
        self.assertIsNotNone(aln.alignment)
        self.assertEqual(aln2.alignment[0].seq, 'MET---')
        self.assertEqual(aln2.alignment[1].seq, 'M-TREE')
        self.assertEqual(aln2.seq_order, ['test', 'seq_1'])
        self.assertEqual(aln2.query_sequence, 'MET---')
        self.assertEqual(aln2.seq_length, 6)
        self.assertEqual(aln2.size, 2)
        self.assertEqual(aln2.marked, [False, False])
        os.remove(save_fn)

    def test_import_single_dna_seq(self):
        fn = generate_temp_fn()
        write_out_temp_fasta(single_dna_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        self.assertIsNotNone(aln.alignment)
        self.assertEqual(aln.alignment[0].seq, 'ATGGAGACT')
        self.assertEqual(aln.seq_order, ['test'])
        self.assertEqual(aln.query_sequence, 'ATGGAGACT')
        self.assertEqual(aln.seq_length, 9)
        self.assertEqual(aln.size, 1)
        self.assertEqual(aln.marked, [False])
        os.remove(fn)

    def test_import_multiple_dna_seqs(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        self.assertIsNotNone(aln.alignment)
        self.assertEqual(aln.alignment[0].seq, 'ATGGAGACT---------')
        self.assertEqual(aln.alignment[1].seq, 'ATG---ACTAGAGAGGAG')
        self.assertEqual(aln.seq_order, ['test', 'seq_1'])
        self.assertEqual(aln.query_sequence, 'ATGGAGACT---------')
        self.assertEqual(aln.seq_length, 18)
        self.assertEqual(aln.size, 2)
        self.assertEqual(aln.marked, [False, False])
        os.remove(fn)

    def test_import_multiple_dna_seqs_save(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        save_fn = f'temp_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.pkl'
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment(save_fn)
        os.remove(fn)
        aln2 = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln2.import_alignment(save_fn)
        self.assertIsNotNone(aln.alignment)
        self.assertEqual(aln2.alignment[0].seq, 'ATGGAGACT---------')
        self.assertEqual(aln2.alignment[1].seq, 'ATG---ACTAGAGAGGAG')
        self.assertEqual(aln2.seq_order, ['test', 'seq_1'])
        self.assertEqual(aln2.query_sequence, 'ATGGAGACT---------')
        self.assertEqual(aln2.seq_length, 18)
        self.assertEqual(aln2.size, 2)
        self.assertEqual(aln2.marked, [False, False])
        os.remove(save_fn)

    def test_write_multiple_protein_seqs(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        os.remove(fn)
        aln.write_out_alignment(fn)
        aln2 = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln2.import_alignment()
        self.assertIsNotNone(aln2.alignment)
        self.assertEqual(aln2.alignment[0].seq, 'MET---')
        self.assertEqual(aln2.alignment[1].seq, 'M-TREE')
        self.assertEqual(aln2.seq_order, ['test', 'seq_1'])
        self.assertEqual(aln2.query_sequence, 'MET---')
        self.assertEqual(aln2.seq_length, 6)
        self.assertEqual(aln2.size, 2)
        self.assertEqual(aln2.marked, [False, False])
        os.remove(fn)

    def test_write_multiple_dna_seqs(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        os.remove(fn)
        aln.write_out_alignment(fn)
        aln2 = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln2.import_alignment()
        self.assertIsNotNone(aln.alignment)
        self.assertEqual(aln2.alignment[0].seq, 'ATGGAGACT---------')
        self.assertEqual(aln2.alignment[1].seq, 'ATG---ACTAGAGAGGAG')
        self.assertEqual(aln2.seq_order, ['test', 'seq_1'])
        self.assertEqual(aln2.query_sequence, 'ATGGAGACT---------')
        self.assertEqual(aln2.seq_length, 18)
        self.assertEqual(aln2.size, 2)
        self.assertEqual(aln2.marked, [False, False])
        os.remove(fn)


class TestSubAlignmentMethods(TestCase):

    def test_generate_sub_aln_1(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        sub_aln = aln.generate_sub_alignment(['test'])
        self.assertIsNotNone(sub_aln.alignment)
        self.assertEqual(sub_aln.alignment[0].seq, 'MET---')
        self.assertEqual(sub_aln.seq_order, ['test'])
        self.assertEqual(sub_aln.query_sequence, 'MET---')
        self.assertEqual(sub_aln.seq_length, 6)
        self.assertEqual(sub_aln.size, 1)
        self.assertEqual(sub_aln.marked, [False])
        os.remove(fn)

    def test_generate_sub_aln_0(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        sub_aln = aln.generate_sub_alignment([])
        self.assertIsNotNone(sub_aln.alignment)
        self.assertEqual(sub_aln.seq_order, [])
        self.assertEqual(sub_aln.query_sequence, 'MET---')
        self.assertEqual(sub_aln.seq_length, 6)
        self.assertEqual(sub_aln.size, 0)
        self.assertEqual(sub_aln.marked, [])
        os.remove(fn)

    def test_subset_columns(self):
        fn = write_out_temp_fasta(two_protein_seqs)
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
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        sub_aln = aln._subset_columns([-3, -2, -1])
        self.assertIsNone(sub_aln)
        os.remove(fn)

    def test_subset_columns_fail_positive(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        sub_aln = aln._subset_columns([10, 11, 12])
        self.assertIsNone(sub_aln)
        os.remove(fn)

    def test_remove_gaps_ungapped(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        sub_aln = aln.remove_gaps()
        self.assertIsNotNone(sub_aln.alignment)
        self.assertEqual(sub_aln.alignment[0].seq, 'MET')
        self.assertEqual(sub_aln.seq_order, ['test'])
        self.assertEqual(sub_aln.query_sequence, 'MET')
        self.assertEqual(sub_aln.seq_length, 3)
        self.assertEqual(sub_aln.size, 1)
        self.assertEqual(sub_aln.marked, [False])
        os.remove(fn)

    def test_remove_gaps_gapped(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        sub_aln = aln.remove_gaps()
        self.assertIsNotNone(sub_aln.alignment)
        self.assertEqual(sub_aln.alignment[0].seq, 'MET')
        self.assertEqual(sub_aln.alignment[1].seq, 'M-T')
        self.assertEqual(sub_aln.seq_order, ['test', 'seq_1'])
        self.assertEqual(sub_aln.query_sequence, 'MET')
        self.assertEqual(sub_aln.seq_length, 3)
        self.assertEqual(sub_aln.size, 2)
        self.assertEqual(sub_aln.marked, [False, False])
        os.remove(fn)

    def test_remove_gaps_fully_gapped(self):
        fn = write_out_temp_fasta(fully_gapped_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        sub_aln = aln.remove_gaps()
        self.assertIsNotNone(sub_aln.alignment)
        self.assertEqual(sub_aln.alignment[0].seq, '------')
        self.assertEqual(sub_aln.alignment[1].seq, 'M-TREE')
        self.assertEqual(sub_aln.seq_order, ['test', 'seq_1'])
        self.assertEqual(sub_aln.query_sequence, '------')
        self.assertEqual(sub_aln.seq_length, 6)
        self.assertEqual(sub_aln.size, 2)
        self.assertEqual(sub_aln.marked, [False, False])
        os.remove(fn)

    def test_remove_bad_sequences_all_allowed(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        sub_aln = aln.remove_bad_sequences()
        self.assertIsNotNone(sub_aln.alignment)
        self.assertEqual(sub_aln.alignment[0].seq, 'ATGGAGACT---------')
        self.assertEqual(sub_aln.alignment[1].seq, 'ATG---ACTAGAGAGGAG')
        self.assertEqual(sub_aln.seq_order, ['test', 'seq_1'])
        self.assertEqual(sub_aln.query_sequence, 'ATGGAGACT---------')
        self.assertEqual(sub_aln.seq_length, 18)
        self.assertEqual(sub_aln.size, 2)
        self.assertEqual(sub_aln.marked, [False, False])
        os.remove(fn)

    def test_remove_bad_sequences_one_removed(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        aln.alphabet.letters = ''.join(list(set(aln.alphabet.letters) - set('R')))
        sub_aln = aln.remove_bad_sequences()
        self.assertIsNotNone(sub_aln.alignment)
        self.assertEqual(sub_aln.alignment[0].seq, 'MET---')
        self.assertEqual(sub_aln.seq_order, ['test'])
        self.assertEqual(sub_aln.query_sequence, 'MET---')
        self.assertEqual(sub_aln.seq_length, 6)
        self.assertEqual(sub_aln.size, 1)
        self.assertEqual(sub_aln.marked, [False])
        os.remove(fn)

    def test_remove_bad_sequences_all_removed(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        sub_aln = aln.remove_bad_sequences()
        self.assertIsNotNone(sub_aln.alignment)
        self.assertEqual(sub_aln.seq_order, [])
        self.assertEqual(sub_aln.query_sequence, 'MET---')
        self.assertEqual(sub_aln.seq_length, 6)
        self.assertEqual(sub_aln.size, 0)
        self.assertEqual(sub_aln.marked, [])
        os.remove(fn)

    def test_generate_positional_sub_alignment_all(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        aln2 = aln.generate_positional_sub_alignment([0, 1, 2, 3, 4, 5])
        self.assertIsNotNone(aln2.alignment)
        self.assertEqual(aln2.alignment[0].seq, 'MET---')
        self.assertEqual(aln2.alignment[1].seq, 'M-TREE')
        self.assertEqual(aln2.seq_order, ['test', 'seq_1'])
        self.assertEqual(aln2.query_sequence, 'MET---')
        self.assertEqual(aln2.seq_length, 6)
        self.assertEqual(aln2.size, 2)
        self.assertEqual(aln2.marked, [False, False])
        os.remove(fn)

    def test_generate_positional_sub_alignment_some(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        aln2 = aln.generate_positional_sub_alignment([0, 3])
        self.assertIsNotNone(aln2.alignment)
        self.assertEqual(aln2.alignment[0].seq, 'M-')
        self.assertEqual(aln2.alignment[1].seq, 'MR')
        self.assertEqual(aln2.seq_order, ['test', 'seq_1'])
        self.assertEqual(aln2.query_sequence, 'M-')
        self.assertEqual(aln2.seq_length, 2)
        self.assertEqual(aln2.size, 2)
        self.assertEqual(aln2.marked, [False, False])
        os.remove(fn)

    def test_generate_positional_sub_alignment_none(self):
        fn = write_out_temp_fasta(two_protein_seqs)
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


class TestAlignmentToNumericRepresentation(TestCase):

    def test_no_alignment_failure(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        _, _, mapping, _ = build_mapping(aln.alphabet)
        with self.assertRaises(TypeError):
            aln._alignment_to_num(mapping)
        os.remove(fn)

    def test_no_mapping_failure(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        with self.assertRaises(TypeError):
            aln._alignment_to_num(None)
        os.remove(fn)

    def test_bad_alphabet_failure(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        _, _, mapping, _ = build_mapping(FullIUPACDNA())
        with self.assertRaises(KeyError):
            aln._alignment_to_num(mapping)
        os.remove(fn)

    def test_bad_alphabet_success(self):
        fn = write_out_temp_fasta(single_dna_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        _, _, mapping, _ = build_mapping(FullIUPACProtein())
        num_aln = aln._alignment_to_num(mapping)
        expected_array = np.array([[0, 17, 6, 6, 0, 6, 0, 2, 17]])
        self.assertFalse((num_aln - expected_array).any())
        os.remove(fn)

    def test_single_DNA_seq(self):
        fn = write_out_temp_fasta(single_dna_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        _, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        expected_array = np.array([[0, 1, 3, 3, 0, 3, 0, 2, 1]])
        self.assertFalse((num_aln - expected_array).any())
        os.remove(fn)

    def test_two_DNA_seqs(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        _, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        expected_array = np.array([[0, 1, 3, 3, 0, 3, 0, 2, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                                   [0, 1, 3, 4, 4, 4, 0, 2, 1, 0, 3, 0, 3, 0, 3, 3, 0, 3]])
        self.assertFalse((num_aln - expected_array).any())
        os.remove(fn)

    def test_single_protein_seq(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        _, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        expected_array = np.array([[11, 4, 17]])
        self.assertFalse((num_aln - expected_array).any())
        os.remove(fn)

    def test_two_protein_seqs(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        _, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        expected_array = np.array([[11, 4, 17, 23, 23, 23],
                                   [11, 23, 17, 15, 4, 4]])
        self.assertFalse((num_aln - expected_array).any())
        os.remove(fn)


class TestAlignmentAndPositionMetrics(TestCase):

    def test_compute_effective_aln_size_permissive_threshold(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        effective_size = aln.compute_effective_alignment_size(identity_threshold=1.0)
        self.assertEqual(effective_size, 2)
        os.remove(fn)

    def test_compute_effective_aln_size_equal_threshold(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        effective_size = aln.compute_effective_alignment_size(identity_threshold=(2 / float(6)) + np.finfo(float).min)
        self.assertEqual(effective_size, 1)
        os.remove(fn)

    def test_compute_effective_aln_size_restrictive_threshold(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        effective_size = aln.compute_effective_alignment_size(identity_threshold=np.finfo(float).min)
        self.assertEqual(effective_size, 1)
        os.remove(fn)

    def test_compute_effective_aln_size_multiple_processors(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        effective_size = aln.compute_effective_alignment_size(identity_threshold=np.finfo(float).min, processes=2)
        self.assertEqual(effective_size, 1)
        os.remove(fn)

    def test_compute_effective_aln_size_with_distance_matrix(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        calculator = AlignmentDistanceCalculator(protein=(aln.polymer_type == 'Protein'))
        distance_matrix = calculator.get_distance(aln.alignment)
        effective_size = aln.compute_effective_alignment_size(identity_threshold=np.finfo(float).min,
                                                              distance_matrix=distance_matrix)
        self.assertEqual(effective_size, 1)
        os.remove(fn)

    def test_compute_effective_aln_size_with_distance_matrix_and_multiple_processors(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        calculator = AlignmentDistanceCalculator(protein=(aln.polymer_type == 'Protein'))
        distance_matrix = calculator.get_distance(aln.alignment)
        effective_size = aln.compute_effective_alignment_size(identity_threshold=np.finfo(float).min,
                                                              distance_matrix=distance_matrix, processes=2)
        self.assertEqual(effective_size, 1)
        os.remove(fn)

    def test_compute_effective_aln_size_none_threshold(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        with self.assertRaises(TypeError):
            aln.compute_effective_alignment_size(identity_threshold=None)
        os.remove(fn)

    def test_compute_effective_aln_size_bad_threshold(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        with self.assertRaises(TypeError):
            aln.compute_effective_alignment_size(identity_threshold='A')
        os.remove(fn)

    def test_determine_usable_positions_permissive_ratio(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        pos, count = aln.determine_usable_positions(ratio=1.0)
        self.assertFalse((pos - np.array([0, 1, 2, 3, 4, 5])).any())
        self.assertFalse((count - np.array([2, 1, 2, 1, 1, 1])).any())
        os.remove(fn)

    def test_determine_usable_positions_equal_ratio(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        pos, count = aln.determine_usable_positions(ratio=0.5)
        self.assertFalse((pos - np.array([0, 1, 2, 3, 4, 5])).any())
        self.assertFalse((count - np.array([2, 1, 2, 1, 1, 1])).any())
        os.remove(fn)

    def test_determine_usable_positions_restrictive_ratio(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        pos, count = aln.determine_usable_positions(ratio=np.finfo(float).min)
        self.assertFalse((pos - np.array([])).any())
        self.assertFalse((count - np.array([2, 1, 2, 1, 1, 1])).any())
        os.remove(fn)

    def test_comparable_positions_all_comparable(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        col1, col2, indices, count = aln.identify_comparable_sequences(pos1=0, pos2=2)
        self.assertEqual(col1.shape, (2, ))
        self.assertTrue(all([col1[i] == np.array(['M', 'M'])[i] for i in range(2)]))
        self.assertEqual(col2.shape, (2,))
        self.assertTrue(all([col2[i] == np.array(['T', 'T'])[i] for i in range(2)]))
        self.assertFalse((indices - np.array([0, 1])).any())
        self.assertEqual(count, 2)
        os.remove(fn)

    def test_comparable_positions_some_comparable1(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        col1, col2, indices, count = aln.identify_comparable_sequences(pos1=0, pos2=1)
        self.assertEqual(col1.shape, (1,))
        self.assertTrue(all([col1[i] == np.array(['M'])[i] for i in range(1)]))
        self.assertEqual(col2.shape, (1,))
        self.assertTrue(all([col2[i] == np.array(['E'])[i] for i in range(1)]))
        self.assertFalse((indices - np.array([0])).any())
        self.assertEqual(count, 1)
        os.remove(fn)

    def test_comparable_positions_some_comparable2(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        col1, col2, indices, count = aln.identify_comparable_sequences(pos1=0, pos2=5)
        self.assertEqual(col1.shape, (1,))
        self.assertTrue(all([col1[i] == np.array(['M'])[i] for i in range(1)]))
        self.assertEqual(col2.shape, (1,))
        self.assertTrue(all([col2[i] == np.array(['E'])[i] for i in range(1)]))
        self.assertFalse((indices - np.array([1])).any())
        self.assertEqual(count, 1)
        os.remove(fn)

    def test_comparable_positions_none(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        col1, col2, indices, count = aln.identify_comparable_sequences(pos1=1, pos2=3)
        self.assertEqual(col1.shape, (0, ))
        self.assertEqual(col2.shape, (0, ))
        self.assertEqual(indices.shape, (0, ))
        self.assertEqual(count, 0)
        os.remove(fn)

    def test_comparable_positions_out_of_range(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        with self.assertRaises(IndexError):
            aln.identify_comparable_sequences(pos1=1, pos2=6)
        os.remove(fn)

    def test_comparable_flipped_positions_out_of_range(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        col1, col2, indices, count = aln.identify_comparable_sequences(pos1=2, pos2=0)
        self.assertEqual(col1.shape, (2,))
        self.assertTrue(all([col1[i] == np.array(['T', 'T'])[i] for i in range(2)]))
        self.assertEqual(col2.shape, (2,))
        self.assertTrue(all([col2[i] == np.array(['M', 'M'])[i] for i in range(2)]))
        self.assertFalse((indices - np.array([0, 1])).any())
        self.assertEqual(count, 2)
        os.remove(fn)

    def test_consensus_sequence_method_failure(self):
        fn = write_out_temp_fasta(two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        with self.assertRaises(ValueError):
            aln.consensus_sequence(method='average')
        os.remove(fn)

    def test_consensus_sequence_trivial(self):
        fn = write_out_temp_fasta(single_protein_seq + single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        consensus_rec = aln.consensus_sequence(method='majority')
        self.assertEqual(consensus_rec.id, 'Consensus Sequence')
        self.assertEqual(consensus_rec.seq, 'MET')
        os.remove(fn)

    def test_consensus_sequence(self):
        fn = write_out_temp_fasta(two_protein_seqs + third_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        consensus_rec = aln.consensus_sequence(method='majority')
        self.assertEqual(consensus_rec.id, 'Consensus Sequence')
        self.assertEqual(consensus_rec.seq, 'M-TREE')
        os.remove(fn)

    def test_consensus_sequence_ties(self):
        fn = write_out_temp_fasta(two_protein_seqs + two_protein_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        consensus_rec = aln.consensus_sequence(method='majority')
        self.assertEqual(consensus_rec.id, 'Consensus Sequence')
        self.assertEqual(consensus_rec.seq, 'METREE')
        os.remove(fn)

    def test_gap_z_score_no_cutoff_failure(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        with self.assertRaises(ValueError):
            aln._gap_z_score_check(None, num_aln, alpha_size)
        os.remove(fn)

    def test_gap_z_score_no_num_aln_failure(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        with self.assertRaises(ValueError):
            aln._gap_z_score_check(2.0, None, alpha_size)
        os.remove(fn)

    def test_gap_z_score_no_gap_num(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        with self.assertRaises(ValueError):
            aln._gap_z_score_check(2.0, num_aln, None)
        os.remove(fn)

    def test_gap_z_score_single_seq(self):
        fn = write_out_temp_fasta(single_dna_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        overly_gapped_seqs = aln._gap_z_score_check(2.0, num_aln, alpha_size)
        self.assertTrue(overly_gapped_seqs.all())
        os.remove(fn)


# class TestSeqAlignment(TestBase):
#
#     @classmethod
#     def setUpClass(cls):
#         super(TestSeqAlignment, cls).setUpClass()
#         cls.query_aln_fa_small = SeqAlignment(
#             file_name=cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln'],
#             query_id=cls.small_structure_id)
#         cls.query_aln_fa_small.import_alignment()
#         cls.query_aln_fa_large = SeqAlignment(
#             file_name=cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln'],
#             query_id=cls.large_structure_id)
#         cls.query_aln_fa_large.import_alignment()
#         cls.query_aln_msf_small = deepcopy(cls.query_aln_fa_small)
#         cls.query_aln_msf_small.file_name = cls.data_set.protein_data[cls.small_structure_id]['Final_MSF_Aln']
#         cls.query_aln_msf_large = deepcopy(cls.query_aln_fa_large)
#         cls.query_aln_msf_large.file_name = cls.data_set.protein_data[cls.large_structure_id]['Final_MSF_Aln']
#         cls.save_file_small = os.path.join(cls.testing_dir, '{}_aln.pkl'.format(cls.small_structure_id))
#         cls.save_file_large = os.path.join(cls.testing_dir, '{}_aln.pkl'.format(cls.large_structure_id))
#         cls.aln_file_small = os.path.join(cls.testing_dir, 'test_{}.fa'.format(cls.small_structure_id))
#         cls.aln_file_large = os.path.join(cls.testing_dir, 'test_{}.fa'.format(cls.large_structure_id))
#         cls.save_dir_small = os.path.join(cls.testing_dir, '{}_cache'.format(cls.small_structure_id))
#         cls.save_dir_large = os.path.join(cls.testing_dir, '{}_cache'.format(cls.large_structure_id))
#         cls.single_letter_size, _, cls.single_letter_mapping, cls.single_letter_reverse = build_mapping(
#             alphabet=Gapped(cls.query_aln_fa_small.alphabet))
#         cls.pair_letter_size, _, cls.pair_letter_mapping, cls.pair_letter_reverse = build_mapping(
#             alphabet=MultiPositionAlphabet(alphabet=Gapped(cls.query_aln_fa_small.alphabet), size=2))
#         cls.single_to_pair = np.zeros((max(cls.single_letter_mapping.values()) + 1,
#                                        max(cls.single_letter_mapping.values()) + 1))
#         for char in cls.pair_letter_mapping:
#             cls.single_to_pair[cls.single_letter_mapping[char[0]],
#                                cls.single_letter_mapping[char[1]]] = cls.pair_letter_mapping[char]
#
#     def tearDown(self):
#         try:
#             os.remove(self.save_file_small)
#         except OSError:
#             pass
#         try:
#             os.remove(self.save_file_large)
#         except OSError:
#             pass
#         try:
#             rmtree(self.save_dir_small)
#         except OSError:
#             pass
#         try:
#             rmtree(self.save_dir_large)
#         except OSError:
#             pass
#
#     def evaluate_init(self, file_name, query_id, aln_type):
#         with self.assertRaises(TypeError):
#             SeqAlignment()
#         aln = SeqAlignment(file_name=file_name, query_id=query_id)
#         self.assertFalse(aln.file_name.startswith('..'))
#         self.assertEqual(aln.file_name, os.path.abspath(file_name))
#         self.assertEqual(aln.query_id, query_id)
#         self.assertIsNone(aln.alignment)
#         self.assertIsNone(aln.seq_order)
#         self.assertIsNone(aln.query_sequence)
#         self.assertIsNone(aln.seq_length)
#         self.assertIsNone(aln.size)
#         self.assertIsNone(aln.marked)
#         self.assertEqual(aln.polymer_type, aln_type)
#         self.assertTrue(isinstance(aln.alphabet, FullIUPACProtein))
#
#     def test1a_init(self):
#         self.evaluate_init(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                            query_id=self.small_structure_id, aln_type='Protein')
#
#     def test1b_init(self):
#         self.evaluate_init(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                            query_id=self.large_structure_id, aln_type='Protein')
#
#     def evaluate_import_alignment(self, file_name, query_id, save, expected_sequence, expected_save_fn, expected_len,
#                                   expected_size):
#         aln = SeqAlignment(file_name=file_name, query_id=query_id)
#         aln.import_alignment(save_file=save)
#         self.assertFalse(aln.file_name.startswith('..'))
#         self.assertEqual(aln.file_name, os.path.abspath(file_name))
#         self.assertEqual(aln.query_id, query_id)
#         self.assertIsInstance(aln.alignment, MultipleSeqAlignment)
#         self.assertEqual(len(aln.seq_order), aln.size)
#         self.assertTrue(query_id in aln.seq_order)
#         self.assertEqual(str(aln.query_sequence).replace('-', ''), expected_sequence)
#         self.assertGreaterEqual(aln.seq_length, expected_len)
#         self.assertEqual(aln.size, expected_size)
#         self.assertEqual(len(aln.marked), aln.size)
#         self.assertFalse(any(aln.marked))
#         self.assertEqual(aln.polymer_type, 'Protein')
#         self.assertTrue(isinstance(aln.alphabet, FullIUPACProtein))
#         if save is None:
#             self.assertFalse(os.path.isfile(expected_save_fn))
#         else:
#             self.assertTrue(os.path.isfile(expected_save_fn))
#
#     def test2a_import_alignment(self):
#         self.evaluate_import_alignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                        query_id=self.small_structure_id, save=self.save_file_small,
#                                        expected_sequence=str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq),
#                                        expected_save_fn=self.save_file_small,
#                                        expected_len=self.data_set.protein_data[self.small_structure_id]['Length'],
#                                        expected_size=self.data_set.protein_data[self.small_structure_id]['Final_Count'])
#
#     def test2b_import_alignment(self):
#         self.evaluate_import_alignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                        query_id=self.small_structure_id, save=None,
#                                        expected_sequence=str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq),
#                                        expected_save_fn=self.save_file_small,
#                                        expected_len=self.data_set.protein_data[self.small_structure_id]['Length'],
#                                        expected_size=self.data_set.protein_data[self.small_structure_id]['Final_Count'])
#
#     def test2c_import_alignment(self):
#         self.evaluate_import_alignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                        query_id=self.large_structure_id, save=self.save_file_large,
#                                        expected_sequence=str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq),
#                                        expected_save_fn=self.save_file_large,
#                                        expected_len=self.data_set.protein_data[self.large_structure_id]['Length'],
#                                        expected_size=self.data_set.protein_data[self.large_structure_id]['Final_Count'])
#
#     def test2d_import_alignment(self):
#         self.evaluate_import_alignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                        query_id=self.large_structure_id, save=None,
#                                        expected_sequence=str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq),
#                                        expected_save_fn=self.save_file_large,
#                                        expected_len=self.data_set.protein_data[self.large_structure_id]['Length'],
#                                        expected_size=self.data_set.protein_data[self.large_structure_id]['Final_Count'])
#
#     def evaluate_write_out_alignment(self, file_name, query_id, out_fn):
#         aln = SeqAlignment(file_name=file_name, query_id=query_id)
#         with self.assertRaises(TypeError):
#             aln.write_out_alignment(out_fn)
#         aln.import_alignment()
#         aln.write_out_alignment(out_fn)
#         self.assertTrue(os.path.isfile(out_fn))
#         aln_prime = SeqAlignment(file_name=out_fn, query_id=query_id)
#         aln_prime.import_alignment()
#         self.assertEqual(aln.query_id, aln_prime.query_id)
#         self.assertEqual(aln.seq_order, aln_prime.seq_order)
#         self.assertEqual(aln.query_sequence, aln_prime.query_sequence)
#         self.assertEqual(aln.seq_length, aln_prime.seq_length)
#         self.assertEqual(aln.size, aln_prime.size)
#         self.assertEqual(aln.marked, aln_prime.marked)
#         self.assertEqual(aln.polymer_type, aln_prime.polymer_type)
#         self.assertTrue(isinstance(aln_prime.alphabet, type(aln.alphabet)))
#         for i in range(aln.size):
#             self.assertEqual(aln.alignment[i].id, aln_prime.alignment[i].id)
#             self.assertEqual(aln.alignment[i].seq, aln_prime.alignment[i].seq)
#
#     def test3a_write_out_alignment(self):
#         self.evaluate_write_out_alignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                           query_id=self.small_structure_id, out_fn=self.aln_file_small)
#
#     def test3b_write_out_alignment(self):
#         self.evaluate_write_out_alignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                           query_id=self.large_structure_id, out_fn=self.aln_file_large)
#
#     def evaluate_generate_sub_alignment(self, file_name, query_id):
#         aln = SeqAlignment(file_name=file_name, query_id=query_id)
#         aln.import_alignment()
#         aln_halved = aln.size // 2
#         aln_seqrecords1 = aln.seq_order[:aln_halved]
#         aln_sub1 = aln.generate_sub_alignment(sequence_ids=aln_seqrecords1)
#         aln_seqrecords2 = aln.seq_order[aln_halved:]
#         aln_sub2 = aln.generate_sub_alignment(sequence_ids=aln_seqrecords2)
#         for sub_aln in [aln_sub1, aln_sub2]:
#             self.assertEqual(aln.file_name, sub_aln.file_name)
#             self.assertEqual(aln.query_id, sub_aln.query_id)
#             self.assertEqual(aln.query_sequence, sub_aln.query_sequence)
#             self.assertEqual(aln.seq_length, sub_aln.seq_length)
#             self.assertEqual(aln.polymer_type, sub_aln.polymer_type)
#             self.assertTrue(isinstance(sub_aln.alphabet, type(aln.alphabet)))
#             self.assertFalse(any(sub_aln.marked))
#         self.assertEqual(aln_sub1.seq_order, aln_seqrecords1)
#         self.assertEqual(aln_sub1.size, aln_halved)
#         self.assertEqual(len(aln_sub1.marked), aln_halved)
#         self.assertEqual(aln_sub2.seq_order, aln_seqrecords2)
#         self.assertEqual(aln_sub2.size, aln.size - aln_halved)
#         self.assertEqual(len(aln_sub2.marked), aln.size - aln_halved)
#         for i in range(aln.size):
#             if i < aln_halved:
#                 self.assertEqual(aln.alignment[i].id, aln_sub1.alignment[i].id)
#                 self.assertEqual(aln.alignment[i].seq, aln_sub1.alignment[i].seq)
#             else:
#                 self.assertEqual(aln.alignment[i].id, aln_sub2.alignment[i - aln_halved].id)
#                 self.assertEqual(aln.alignment[i].seq, aln_sub2.alignment[i - aln_halved].seq)
#
#     def test4a_generate_sub_alignment(self):
#         self.evaluate_generate_sub_alignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                              query_id=self.small_structure_id)
#
#     def test4b_generate_sub_alignment(self):
#         self.evaluate_generate_sub_alignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                              query_id=self.large_structure_id)
#
#     def evaluate__subset_columns(self, file_name, query_id, positions):
#         aln = SeqAlignment(file_name=file_name, query_id=query_id)
#         with self.assertRaises(TypeError):
#             aln._subset_columns(positions)
#         aln.import_alignment()
#         aln_prime = aln._subset_columns(positions)
#         self.assertEqual(len(aln_prime), aln.size)
#         for i in range(aln.size):
#             self.assertEqual(aln.alignment[i].id, aln_prime[i].id)
#             self.assertEqual(len(aln_prime[i].seq), len(positions))
#             self.assertEqual(''.join([str(aln.alignment[i].seq)[p] for p in positions]), str(aln_prime[i].seq), (positions, ''.join([str(aln.alignment[i].seq)[p] for p in positions]), str(aln_prime[i].seq)))
#
#     def test5a__subset_columns_one_position(self):
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                       query_id=self.small_structure_id, positions=[0])
#
#     def test5b__subset_columns_one_position(self):
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                       query_id=self.small_structure_id,
#                                       positions=[self.data_set.protein_data[self.small_structure_id]['Length'] // 2])
#
#     def test5c__subset_columns_one_position(self):
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                       query_id=self.small_structure_id,
#                                       positions=[self.data_set.protein_data[self.small_structure_id]['Length'] - 1])
#
#     def test5d__subset_columns_one_position(self):
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                       query_id=self.large_structure_id, positions=[0])
#
#     def test5e__subset_columns_one_position(self):
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                       query_id=self.large_structure_id,
#                                       positions=[self.data_set.protein_data[self.large_structure_id]['Length'] // 2])
#
#     def test5f__subset_columns_one_position(self):
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                       query_id=self.large_structure_id,
#                                       positions=[self.data_set.protein_data[self.large_structure_id]['Length'] - 1])
#
#     def test5g__subset_columns_single_range(self):
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                       query_id=self.small_structure_id, positions=list(range(5)))
#
#     def test5h__subset_columns_single_range(self):
#         positions = list(range(self.data_set.protein_data[self.small_structure_id]['Length'] // 2,
#                                self.data_set.protein_data[self.small_structure_id]['Length'] // 2 + 5))
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                       query_id=self.small_structure_id, positions=positions)
#
#     def test5i__subset_columns_single_range(self):
#         positions = list(range(self.data_set.protein_data[self.small_structure_id]['Length'] - 5,
#                                self.data_set.protein_data[self.small_structure_id]['Length']))
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                       query_id=self.small_structure_id, positions=positions)
#
#     def test5j__subset_columns_single_range(self):
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                       query_id=self.large_structure_id, positions=list(range(5)))
#
#     def test5k__subset_columns_single_range(self):
#         positions = list(range(self.data_set.protein_data[self.large_structure_id]['Length'] // 2,
#                                self.data_set.protein_data[self.large_structure_id]['Length'] // 2 + 5))
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                       query_id=self.large_structure_id, positions=positions)
#
#     def test5l__subset_columns_single_range(self):
#         positions = list(range(self.data_set.protein_data[self.large_structure_id]['Length'] - 5,
#                                self.data_set.protein_data[self.large_structure_id]['Length']))
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                       query_id=self.large_structure_id, positions=positions)
#
#     def test5m__subset_columns_range_and_position(self):
#         positions = [0] + list(range(self.data_set.protein_data[self.small_structure_id]['Length'] // 2,
#                                      self.data_set.protein_data[self.small_structure_id]['Length'] // 2 + 5))
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                       query_id=self.small_structure_id, positions=positions)
#
#     def test5n__subset_columns_range_and_position(self):
#         positions = list(range(self.data_set.protein_data[self.small_structure_id]['Length'] // 2,
#                                self.data_set.protein_data[self.small_structure_id]['Length'] // 2 + 5)) +\
#                     [self.data_set.protein_data[self.small_structure_id]['Length'] - 1]
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                       query_id=self.small_structure_id, positions=positions)
#
#     def test5o__subset_columns_range_and_position(self):
#         positions = list(range(5)) + [self.data_set.protein_data[self.small_structure_id]['Length'] // 2] +\
#                     list(range(self.data_set.protein_data[self.small_structure_id]['Length'] - 5,
#                                self.data_set.protein_data[self.small_structure_id]['Length']))
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                       query_id=self.small_structure_id, positions=positions)
#
#     def test5p__subset_columns_range_and_position(self):
#         positions = [0] + list(range(self.data_set.protein_data[self.large_structure_id]['Length'] // 2,
#                                      self.data_set.protein_data[self.large_structure_id]['Length'] // 2 + 5))
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                       query_id=self.large_structure_id, positions=positions)
#
#     def test5q__subset_columns_range_and_position(self):
#         positions = list(range(self.data_set.protein_data[self.large_structure_id]['Length'] // 2,
#                                self.data_set.protein_data[self.large_structure_id]['Length'] // 2 + 5)) +\
#                     [self.data_set.protein_data[self.large_structure_id]['Length'] - 1]
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                       query_id=self.large_structure_id, positions=positions)
#
#     def test5r__subset_columns_range_and_position(self):
#         positions = list(range(5)) + [self.data_set.protein_data[self.large_structure_id]['Length'] // 2] +\
#                     list(range(self.data_set.protein_data[self.large_structure_id]['Length'] - 5,
#                                self.data_set.protein_data[self.large_structure_id]['Length']))
#         self.evaluate__subset_columns(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                       query_id=self.large_structure_id, positions=positions)
#
#     def evaluate_remove_gaps(self, file_name, query_id):
#         aln = SeqAlignment(file_name=file_name, query_id=query_id)
#         with self.assertRaises(TypeError):
#             aln.remove_gaps()
#         aln.import_alignment()
#         aln_prime = aln.remove_gaps()
#         ungapped_pos = [i for i, char in enumerate(aln.query_sequence) if char != '-']
#         self.assertEqual(aln.query_id, aln_prime.query_id)
#         self.assertEqual(aln.seq_order, aln_prime.seq_order)
#         self.assertEqual(str(aln_prime.query_sequence), subset_string(aln.query_sequence, ungapped_pos))
#         self.assertEqual(aln_prime.seq_length, len(aln_prime.query_sequence))
#         self.assertEqual(aln.size, aln_prime.size)
#         self.assertEqual(aln.marked, aln_prime.marked)
#         self.assertEqual(aln.polymer_type, aln_prime.polymer_type)
#         self.assertTrue(isinstance(aln_prime.alphabet, type(aln.alphabet)))
#         for i in range(aln.size):
#             self.assertEqual(aln.alignment[i].id, aln_prime.alignment[i].id)
#             self.assertEqual(aln_prime.alignment[i].seq, subset_string(aln.alignment[i].seq, ungapped_pos))
#
#     def test6a_remove_gaps(self):
#         self.evaluate_remove_gaps(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                   query_id=self.small_structure_id)
#
#     def test6b_remove_gaps(self):
#         self.evaluate_remove_gaps(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                   query_id=self.large_structure_id)
#
#     def evaluate_remove_bad_sequences(self, file_name, query_id, expected_size):
#         aln = SeqAlignment(file_name=file_name, query_id=query_id)
#         with self.assertRaises(TypeError):
#             aln.remove_bad_sequences()
#         aln.import_alignment()
#         aln.size += 2
#         to_remove1 = SeqRecord(id='Test1', seq=Seq(str(aln.alignment[0].seq)[:-1] + 'U', alphabet=FullIUPACProtein()))
#         to_remove2 = SeqRecord(id='Test2', seq=Seq('U' * aln.seq_length, alphabet=FullIUPACProtein()))
#         aln.alignment.append(to_remove1)
#         aln.alignment.append(to_remove2)
#         aln.seq_order += ['Test1', 'Test2']
#         aln.marked += [False, False]
#         self.assertEqual(aln.size, expected_size + 2)
#         self.assertEqual(len(aln.seq_order), aln.size)
#         self.assertEqual(len(aln.marked), aln.size)
#         self.assertEqual(len(aln.alignment), expected_size + 2)
#         self.assertTrue('U' in aln.alignment[aln.size - 2].seq)
#         self.assertTrue('U' in aln.alignment[aln.size - 1].seq)
#         aln_prime = aln.remove_bad_sequences()
#         self.assertEqual(aln_prime.query_id, aln.query_id)
#         self.assertEqual(aln_prime.seq_order, aln.seq_order[:-2])
#         self.assertEqual(aln_prime.query_sequence, aln.query_sequence)
#         self.assertEqual(aln_prime.seq_length, aln.seq_length)
#         self.assertEqual(aln_prime.size, expected_size)
#         self.assertEqual(aln_prime.marked, aln.marked[:-2])
#         self.assertEqual(aln_prime.polymer_type, aln.polymer_type)
#         self.assertTrue(isinstance(aln_prime.alphabet, type(aln.alphabet)))
#         for i in range(aln_prime.size):
#             self.assertEqual(aln.alignment[i].id, aln_prime.alignment[i].id)
#             self.assertEqual(aln_prime.alignment[i].seq, aln.alignment[i].seq)
#
#     def test7a_remove_bad_sequences(self):
#         self.evaluate_remove_bad_sequences(
#             file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#             query_id=self.small_structure_id,
#             expected_size=self.data_set.protein_data[self.small_structure_id]['Final_Count'])
#
#     def test7b_remove_bad_sequences(self):
#         self.evaluate_remove_bad_sequences(
#             file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#             query_id=self.large_structure_id,
#             expected_size=self.data_set.protein_data[self.large_structure_id]['Final_Count'])
#
#     def evaluate_generate_positional_sub_alignment(self, file_name, query_id, positions):
#         aln = SeqAlignment(file_name=file_name, query_id=query_id)
#         aln.import_alignment()
#         aln_sub = aln.generate_positional_sub_alignment(positions=positions)
#         self.assertEqual(aln.file_name, aln_sub.file_name)
#         self.assertEqual(aln.query_id, aln_sub.query_id)
#         self.assertEqual(aln.seq_order, aln_sub.seq_order)
#         self.assertEqual(str(aln_sub.query_sequence.seq), ''.join([aln.query_sequence[i] for i in positions]))
#         self.assertEqual(aln_sub.seq_length, 2)
#         self.assertEqual(aln.size, aln_sub.size)
#         self.assertEqual(aln.marked, aln_sub.marked)
#         self.assertEqual(aln.polymer_type, aln_sub.polymer_type)
#         self.assertTrue(isinstance(aln_sub.alphabet, type(aln.alphabet)))
#         for j in range(aln.size):
#             self.assertEqual(str(aln_sub.alignment[j].seq), ''.join([aln.alignment[j].seq[i] for i in positions]))
#
#     def test8a_generate_positional_sub_alignment(self):
#         self.evaluate_generate_positional_sub_alignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                                         query_id=self.small_structure_id, positions=[0, 1])
#
#     def test8b_generate_positional_sub_alignment(self):
#         seq_len = self.data_set.protein_data[self.small_structure_id]['Length']
#         self.evaluate_generate_positional_sub_alignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                                         query_id=self.small_structure_id,
#                                                         positions=[seq_len // 2, seq_len // 2 + 1])
#
#     def test8c_generate_positional_sub_alignment(self):
#         seq_len = self.data_set.protein_data[self.small_structure_id]['Length']
#         self.evaluate_generate_positional_sub_alignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                                         query_id=self.small_structure_id,
#                                                         positions=[seq_len - 2, seq_len - 1])
#
#     def test8d_generate_positional_sub_alignment(self):
#         self.evaluate_generate_positional_sub_alignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                                         query_id=self.large_structure_id, positions=[0, 1])
#
#     def test8e_generate_positional_sub_alignment(self):
#         seq_len = self.data_set.protein_data[self.large_structure_id]['Length']
#         self.evaluate_generate_positional_sub_alignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                                         query_id=self.large_structure_id,
#                                                         positions=[seq_len // 2, seq_len // 2 + 1])
#
#     def test8f_generate_positional_sub_alignment(self):
#         seq_len = self.data_set.protein_data[self.large_structure_id]['Length']
#         self.evaluate_generate_positional_sub_alignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#                                                         query_id=self.large_structure_id,
#                                                         positions=[seq_len - 2, seq_len - 1])
#
#     def evaluate_compute_effective_alignment_size(self, file_name, query_id):
#         aln = SeqAlignment(file_name=file_name, query_id=query_id)
#         with self.assertRaises(TypeError):
#             aln.compute_effective_alignment_size()
#         aln.import_alignment()
#         calc = AlignmentDistanceCalculator()
#         distance_mat = np.array(calc.get_distance(aln.alignment))
#         identity_mat = 1 - np.array(distance_mat)
#         effective_size = 0.0
#         for i in range(aln.size):
#             n_i = 0.0
#             for j in range(aln.size):
#                 if identity_mat[i, j] >= 0.62:
#                     n_i += 1.0
#             effective_size += 1.0 / n_i
#         self.assertLess(abs(aln.compute_effective_alignment_size() - effective_size), 1.0e-12)
#
#     def test9a_compute_effective_alignment_size(self):
#         self.evaluate_compute_effective_alignment_size(
#             file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#             query_id=self.small_structure_id)
#
#     def test9b_compute_effective_alignment_size(self):
#         self.evaluate_compute_effective_alignment_size(
#             file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#             query_id=self.large_structure_id)
#
#     def evaluate_determine_usable_positions(self, file_name, query_id):
#         aln_small = SeqAlignment(file_name=file_name, query_id=query_id)
#         aln_small.import_alignment()
#         pos, evidence = aln_small.determine_usable_positions(ratio=0.5)
#         usable_pos = []
#         for i in range(aln_small.seq_length):
#             count = 0
#             for j in range(aln_small.size):
#                 if aln_small.alignment[j, i] != '-':
#                     count += 1
#             if count >= (aln_small.size / 2):
#                 usable_pos.append(i)
#             self.assertEqual(evidence[i], count)
#         self.assertEqual(list(pos), usable_pos)
#
#     def test10a_determine_usable_positions(self):
#         self.evaluate_determine_usable_positions(
#             file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#             query_id=self.small_structure_id)
#
#     def test10b_determine_usable_positions(self):
#         self.evaluate_determine_usable_positions(
#             file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#             query_id=self.large_structure_id)
#
#     def evaluate_identify_comparable_sequences(self, file_name, query_id):
#         aln = SeqAlignment(file_name=file_name, query_id=query_id)
#         aln.import_alignment()
#         for i in range(1, aln.seq_length):
#             comp_tup = aln.identify_comparable_sequences(pos1=0, pos2=i)
#             col1 = aln.alignment[:, 0]
#             col1_sub = []
#             col2 = aln.alignment[:, i]
#             col2_sub = []
#             indices = []
#             count = 0
#             for j in range(aln.size):
#                 if (col1[j] != '-') and (col2[j] != '-'):
#                     col1_sub.append(col1[j])
#                     col2_sub.append(col2[j])
#                     indices.append(j)
#                     count += 1
#             self.assertEqual(list(comp_tup[0]), col1_sub)
#             self.assertEqual(list(comp_tup[1]), col2_sub)
#             self.assertEqual(list(comp_tup[2]), indices)
#             self.assertEqual(comp_tup[3], count)
#
#     def test11a_identify_comparable_sequences(self):
#         self.evaluate_identify_comparable_sequences(
#             file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#             query_id=self.small_structure_id)
#
#     def test11b_identify_comparable_sequences(self):
#         self.evaluate_identify_comparable_sequences(
#             file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#             query_id=self.large_structure_id)
#
#     def evaluate_consensus_sequence(self, file_name, query_id):
#         aln_small = SeqAlignment(file_name=file_name, query_id=query_id)
#         aln_small.import_alignment()
#         consensus = aln_small.consensus_sequence()
#         self.assertEqual(consensus.id, 'Consensus Sequence')
#         for i in range(aln_small.seq_length):
#             best_count = -1
#             best_aa = None
#             counts = {}
#             for j in range(aln_small.size):
#                 aa = aln_small.alignment[j, i]
#                 if aa not in counts:
#                     counts[aa] = 0
#                 counts[aa] += 1
#                 if counts[aa] > best_count:
#                     best_count = counts[aa]
#                     best_aa = aa
#                 elif counts[aa] == best_count and aa < best_aa:
#                     if aa == '-':
#                         pass
#                     else:
#                         best_aa = aa
#             self.assertEqual(consensus.seq[i], best_aa)
#
#     def test12a_consensus_sequences(self):
#         self.evaluate_consensus_sequence(
#             file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#             query_id=self.small_structure_id)
#
#     def test12b_consensus_sequences(self):
#         self.evaluate_consensus_sequence(
#             file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#             query_id=self.large_structure_id)
#
#     def evaluate__alignment_to_num(self, file_name, query_id):
#         aln = SeqAlignment(file_name=file_name, query_id=query_id)
#         aln.import_alignment()
#         _, _, mapping, _ = build_mapping(alphabet=aln.alphabet)
#         aln_obj1_num = aln._alignment_to_num(mapping=mapping)
#         for i in range(aln.size):
#             for j in range(aln.seq_length):
#                 self.assertEqual(aln_obj1_num[i, j], mapping[aln.alignment[i, j]])
#
#     def test13a__alignment_to_num(self):
#         self.evaluate__alignment_to_num(
#             file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#             query_id=self.small_structure_id)
#
#     def test13b__alignment_to_num(self):
#         self.evaluate__alignment_to_num(
#             file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#             query_id=self.large_structure_id)
#
#     def evaluate__gap_z_score_check(self, file_name, query_id):
#         aln = SeqAlignment(file_name=file_name, query_id=query_id)
#         aln.import_alignment()
#         alpha_size, gap_chars, mapping, _ = build_mapping(alphabet=aln.alphabet)
#         numeric_aln = aln._alignment_to_num(mapping=mapping)
#         passing_sequences = aln._gap_z_score_check(z_score_cutoff=0.0, num_aln=numeric_aln, gap_num=alpha_size)
#         self.assertEqual(len(passing_sequences), aln.size)
#         gap_counts = []
#         for i in range(aln.size):
#             gap_count = 0
#             for j in range(aln.seq_length):
#                 if aln.alignment[i, j] in gap_chars:
#                     gap_count += 1
#             gap_counts.append(gap_count)
#         mean_count = np.mean(gap_counts)
#         for i in range(aln.size):
#             self.assertEqual(passing_sequences[i], gap_counts[i] < mean_count)
#
#     def test14a__gap_z_score_check(self):
#         self.evaluate__gap_z_score_check(
#             file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#             query_id=self.small_structure_id)
#
#     def test14b__gap_z_score_check(self):
#         self.evaluate__gap_z_score_check(
#             file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#             query_id=self.large_structure_id)
#
#     def evaluate__gap_percentile_check(self, file_name, query_id):
#         aln_small = SeqAlignment(file_name=file_name, query_id=query_id)
#         aln_small.import_alignment()
#         consensus = aln_small.consensus_sequence()
#         alpha_size, gap_chars, mapping, _ = build_mapping(alphabet=aln_small.alphabet)
#         numeric_aln = aln_small._alignment_to_num(mapping=mapping)
#         passing_sequences = aln_small._gap_percentile_check(percentile_cutoff=0.15, num_aln=numeric_aln,
#                                                             gap_num=alpha_size, mapping=mapping)
#         max_differences = np.floor(aln_small.size * 0.15)
#         for i in range(aln_small.size):
#             diff_count = 0
#             for j in range(aln_small.seq_length):
#                 if (((aln_small.alignment[i, j] in gap_chars) or (consensus.seq[j] in gap_chars)) and
#                         (aln_small.alignment[i, j] != consensus.seq[j])):
#                     diff_count += 1
#             self.assertEqual(passing_sequences[i], diff_count <= max_differences)
#
#     def test15a__gap_percentile_check(self):
#         self.evaluate__gap_percentile_check(
#             file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#             query_id=self.small_structure_id)
#
#     def test15b__gap_percentile_check(self):
#         self.evaluate__gap_percentile_check(
#             file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
#             query_id=self.large_structure_id)
#
#     def evaluate_gap_evaluation_simple(self, file_name, query_id):
#         aln_small = SeqAlignment(file_name=file_name, query_id=query_id)
#         aln_small.import_alignment()
#         kept, removed = aln_small.gap_evaluation(size_cutoff=15, z_score_cutoff=0.0, percentile_cutoff=0.15)
#         alpha_size, gap_chars, mapping, _ = build_mapping(alphabet=aln_small.alphabet)
#         self.assertEqual(len(kept) + len(removed), aln_small.size)
#         gap_counts = []
#         for i in range(aln_small.size):
#             gap_count = 0
#             for j in range(aln_small.seq_length):
#                 if aln_small.alignment[i, j] in gap_chars:
#                     gap_count += 1
#             gap_counts.append(gap_count)
#         mean_count = np.mean(gap_counts)
#         for i in range(aln_small.size):
#             if gap_counts[i] < mean_count:
#                 self.assertTrue(aln_small.seq_order[i] in kept)
#             else:
#                 self.assertTrue(aln_small.seq_order[i] in removed)
#
#     def test16a_gap_evaluation(self):
#         self.evaluate_gap_evaluation_simple(
#             file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#             query_id=self.small_structure_id)
#
#     def test16b_gap_evaluation(self):
#         self.evaluate_gap_evaluation_simple(
#             file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#             query_id=self.small_structure_id)
#
#     def evaluate_gap_evaluation(self, file_name, query_id):
#         aln = SeqAlignment(file_name=file_name, query_id=query_id)
#         aln.import_alignment()
#         aln_sub = aln.generate_sub_alignment(aln.seq_order[:10])
#         consensus = aln_sub.consensus_sequence()
#         kept, removed = aln_sub.gap_evaluation(size_cutoff=15, z_score_cutoff=0.0, percentile_cutoff=0.15)
#         alpha_size, gap_chars, _, _ = build_mapping(alphabet=aln.alphabet)
#         max_differences = np.floor(aln_sub.size * 0.15)
#         for i in range(aln_sub.size):
#             diff_count = 0
#             for j in range(aln_sub.seq_length):
#                 if (((aln_sub.alignment[i, j] in gap_chars) or (consensus.seq[j] in gap_chars)) and
#                         (aln_sub.alignment[i, j] != consensus.seq[j])):
#                     diff_count += 1
#             if diff_count <= max_differences:
#                 self.assertTrue(aln_sub.seq_order[i] in kept)
#             else:
#                 self.assertTrue(aln_sub.seq_order[i] in removed)
#
#     def test16c_gap_evaluation(self):
#         self.evaluate_gap_evaluation(
#             file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#             query_id=self.small_structure_id)
#
#     def test16d_gap_evaluation(self):
#         self.evaluate_gap_evaluation(
#             file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#             query_id=self.small_structure_id)
#
#     def evaluate_heatmap_plot(self, aln, save, plot, save_dir):
#         os.makedirs(save_dir, exist_ok=True)
#         aln.import_alignment()
#         _, _, mapping, _ = build_mapping(alphabet=aln.alphabet)
#         expected_numeric = aln._alignment_to_num(mapping)
#         name = '{} Alignment Visualization'.format(aln.query_id)
#         fig = plt.figure(figsize=(0.25 * aln.seq_length + 0.7 + 0.7, 0.25 * aln.size + 0.7 + 0.7))
#         gs = GridSpec(nrows=1, ncols=1)
#         plotting_ax = fig.add_subplot(gs[0, 0])
#         expected_path = os.path.join(save_dir, name.replace(' ', '_') + '.eps')
#         if plot:
#             ax = plotting_ax
#         else:
#             ax = None
#         print('Plotting with save: {} and ax: {}'.format(save, ax))
#         df, hm = aln.heatmap_plot(name=name, out_dir=save_dir, save=save, ax=ax)
#         for i in range(aln.size):
#             self.assertEqual(df.index[i], aln.seq_order[i])
#         for j in range(aln.seq_length):
#             self.assertTrue('{}:{}'.format(j, aln.query_sequence[j]) in df.columns)
#         self.assertFalse((df.values - expected_numeric).any())
#         if ax:
#             self.assertEqual(ax, hm)
#             ax.clear()
#         else:
#             self.assertIsNotNone(hm)
#             self.assertNotEqual(ax, hm)
#         if save:
#             self.assertTrue(os.path.isfile(expected_path))
#             os.remove(expected_path)
#         else:
#             self.assertFalse(os.path.isfile(expected_path))
#
#     def test17a_heatmap_plot(self):
#         aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                  query_id=self.small_structure_id)
#         self.evaluate_heatmap_plot(aln=aln_small, save=True, save_dir=self.save_dir_small, plot=False)
#
#     def test17b_heatmap_plot(self):
#         aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                  query_id=self.small_structure_id)
#         self.evaluate_heatmap_plot(aln=aln_small, save=True, save_dir=self.save_dir_small, plot=True)
#
#     def test17c_heatmap_plot(self):
#         aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                  query_id=self.small_structure_id)
#         self.evaluate_heatmap_plot(aln=aln_small, save=False, save_dir=self.save_dir_small, plot=False)
#
#     def test17d_heatmap_plot(self):
#         aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
#                                  query_id=self.small_structure_id)
#         self.evaluate_heatmap_plot(aln=aln_small, save=False, save_dir=self.save_dir_small, plot=True)
#
#     def evaluate_characterize_positions(self, aln, single, pair):
#         aln_sub = aln.generate_sub_alignment(sequence_ids=[aln.query_id])
#         single_pos = aln.seq_length
#         single_table, pair_table = aln_sub.characterize_positions(
#             single=single, pair=pair, single_size=self.single_letter_size, single_mapping=self.single_letter_mapping,
#             single_reverse=self.single_letter_reverse, pair_size=self.pair_letter_size,
#             pair_mapping=self.pair_letter_mapping, pair_reverse=self.pair_letter_reverse)
#         single_table2, pair_table2 = aln_sub.characterize_positions2(
#             single=single, pair=pair, single_letter_size=self.single_letter_size,
#             single_letter_mapping=self.single_letter_mapping, single_letter_reverse=self.single_letter_reverse,
#             pair_letter_size=self.pair_letter_size, pair_letter_mapping=self.pair_letter_mapping,
#             pair_letter_reverse=self.pair_letter_reverse, single_to_pair=self.single_to_pair)
#         if single:
#             self.assertEqual(single_table.get_positions(), list(range(aln.seq_length)))
#             self.assertEqual(single_table2.get_positions(), list(range(aln.seq_length)))
#             table1 = single_table.get_table()
#             position_sums = np.sum(table1, axis=1)
#             self.assertFalse((position_sums > 1).any())
#             self.assertEqual(np.sum(position_sums), single_pos)
#             character_sums = np.sum(table1, axis=0)
#             self.assertFalse((character_sums > aln.seq_length).any())
#             self.assertEqual(np.sum(character_sums), single_pos)
#             table2 = single_table2.get_table()
#             position_sums2 = np.sum(table2, axis=1)
#             self.assertFalse((position_sums2 > 1).any())
#             self.assertEqual(np.sum(position_sums2), single_pos)
#             character_sums2 = np.sum(table2, axis=0)
#             self.assertFalse((character_sums2 > single_pos).any())
#             self.assertEqual(np.sum(character_sums2), single_pos)
#         else:
#             self.assertIsNone(single_table)
#             self.assertIsNone(single_table2)
#         if pair:
#             pair_pos = np.sum(range(aln.seq_length + 1))
#             positions = []
#             for i in range(aln.seq_length):
#                 for j in range(i, aln.seq_length):
#                     position = (i, j)
#                     positions.append(position)
#             self.assertEqual(pair_table.get_positions(), positions)
#             self.assertEqual(pair_table2.get_positions(), positions)
#             table1 = pair_table.get_table()
#             position_sums = np.sum(table1, axis=1)
#             self.assertFalse((position_sums > 1).any())
#             self.assertEqual(np.sum(position_sums), pair_pos)
#             character_sums = np.sum(table1, axis=0)
#             self.assertFalse((character_sums > pair_pos).any())
#             self.assertEqual(np.sum(character_sums), pair_pos)
#             table2 = pair_table2.get_table()
#             position_sums2 = np.sum(table2, axis=1)
#             self.assertFalse((position_sums2 > 1).any())
#             self.assertEqual(np.sum(position_sums2), pair_pos)
#             character_sums2 = np.sum(table2, axis=0)
#             self.assertFalse((character_sums2 > pair_pos).any())
#             self.assertEqual(np.sum(character_sums2), pair_pos)
#         else:
#             self.assertIsNone(pair_table)
#             self.assertIsNone(pair_table2)
#
#     def test18a_characterize_positions(self):
#         self.evaluate_characterize_positions(aln=self.query_aln_fa_small, single=True, pair=False)
#
#     def test18b_characterize_positions(self):
#         self.evaluate_characterize_positions(aln=self.query_aln_fa_small, single=False, pair=True)
#
#     def test18c_characterize_positions(self):
#         self.evaluate_characterize_positions(aln=self.query_aln_fa_small, single=True, pair=True)
#
#     def test18d_characterize_positions(self):
#         self.evaluate_characterize_positions(aln=self.query_aln_fa_large, single=True, pair=False)
#
#     def test18e_characterize_positions(self):
#         self.evaluate_characterize_positions(aln=self.query_aln_fa_large, single=False, pair=True)
#
#     def test18f_characterize_positions(self):
#         self.evaluate_characterize_positions(aln=self.query_aln_fa_large, single=True, pair=True)
#
# def subset_string(in_str, positions):
#     new_str = ''
#     for i in positions:
#         new_str += in_str[i]
#     return new_str


if __name__ == '__main__':
    unittest.main()
