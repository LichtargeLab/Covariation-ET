"""
Created on Nov 9, 2018

@author: daniel
"""
import os
import unittest
import numpy as np
import pandas as pd
from shutil import rmtree
from datetime import datetime
from Bio.Alphabet import Gapped
import matplotlib.pyplot as plt
from unittest import TestCase
from test_Base import TestBase
from utils import build_mapping
from SeqAlignment import SeqAlignment
from FrequencyTable import FrequencyTable
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

    def test_gap_z_score_two_seq(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        overly_gapped_seqs = aln._gap_z_score_check(0.5, num_aln, alpha_size)
        expected_passing = np.array([False, True])
        self.assertFalse(((1 * overly_gapped_seqs) - (1 * expected_passing)).any())
        os.remove(fn)

    def test_gap_z_score_three_seq(self):
        fn = write_out_temp_fasta(two_protein_seqs + third_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        overly_gapped_seqs = aln._gap_z_score_check(1.0, num_aln, alpha_size)
        expected_passing = np.array([False, True, True])
        self.assertFalse(((1 * overly_gapped_seqs) - (1 * expected_passing)).any())
        os.remove(fn)

    def test_gap_percentile_check_no_cutoff_failure(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        with self.assertRaises(ValueError):
            aln._gap_percentile_check(None, num_aln, alpha_size, mapping)
        os.remove(fn)

    def test_gap_percentile_check_no_num_aln_failure(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        with self.assertRaises(ValueError):
            aln._gap_percentile_check(2.0, None, alpha_size, mapping)
        os.remove(fn)

    def test_gap_percentile_check_no_gap_num(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        with self.assertRaises(ValueError):
            aln._gap_percentile_check(2.0, num_aln, None, mapping)
        os.remove(fn)

    def test_gap_percentile_check_no_mapping(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        with self.assertRaises(ValueError):
            aln._gap_percentile_check(2.0, num_aln, alpha_size, None)
        os.remove(fn)

    def test_gap_percentile_check_single_seq(self):
        fn = write_out_temp_fasta(single_dna_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        overly_gapped_seqs = aln._gap_percentile_check(0.5, num_aln, alpha_size, mapping)
        self.assertTrue(overly_gapped_seqs.all())
        os.remove(fn)

    def test_gap_percentile_check_two_seq(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        overly_gapped_seqs = aln._gap_percentile_check(0.5, num_aln, alpha_size, mapping)
        expected_passing = np.array([False, True])
        self.assertFalse(((1 * overly_gapped_seqs) - (1 * expected_passing)).any())
        os.remove(fn)

    def test_gap_percentile_check_three_seq(self):
        fn = write_out_temp_fasta(two_protein_seqs + third_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(aln.alphabet)
        num_aln = aln._alignment_to_num(mapping)
        overly_gapped_seqs = aln._gap_percentile_check(0.5, num_aln, alpha_size, mapping)
        expected_passing = np.array([False, True, True])
        self.assertFalse(((1 * overly_gapped_seqs) - (1 * expected_passing)).any())
        os.remove(fn)

    def test_gap_evaluation_no_size_cutoff_failure(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        with self.assertRaises(ValueError):
            aln.gap_evaluation(None, z_score_cutoff=2.0, percentile_cutoff=0.5)
        os.remove(fn)

    def test_gap_evaluation_no_z_score_cutoff_failure(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        with self.assertRaises(ValueError):
            aln.gap_evaluation(size_cutoff=1, z_score_cutoff=None, percentile_cutoff=0.5)
        os.remove(fn)

    def test_gap_evaluation_no_percentile_cutoff_failure(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        with self.assertRaises(ValueError):
            aln.gap_evaluation(size_cutoff=2, z_score_cutoff=2.0, percentile_cutoff=None)
        os.remove(fn)

    def test_gap_evaluation_z_score_single_seq(self):
        fn = write_out_temp_fasta(single_dna_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        passing, not_passing = aln.gap_evaluation(size_cutoff=0, z_score_cutoff=2.0, percentile_cutoff=None)
        self.assertEqual(passing, [aln.query_id])
        self.assertEqual(not_passing, [])
        os.remove(fn)

    def test_gap_evaluation_z_score_two_seq(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        passing, not_passing = aln.gap_evaluation(size_cutoff=1, z_score_cutoff=0.5, percentile_cutoff=None)
        self.assertEqual(passing, [aln.seq_order[1]])
        self.assertEqual(not_passing, [aln.seq_order[0]])
        os.remove(fn)

    def test_gap_evaluation_z_score_three_seq(self):
        fn = write_out_temp_fasta(two_protein_seqs + third_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        passing, not_passing = aln.gap_evaluation(size_cutoff=2, z_score_cutoff=1.0, percentile_cutoff=None)
        self.assertEqual(passing, aln.seq_order[1:])
        self.assertEqual(not_passing, [aln.seq_order[0]])
        os.remove(fn)

    def test_gap_evaluation_percentile_check_single_seq(self):
        fn = write_out_temp_fasta(single_dna_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        passing, not_passing = aln.gap_evaluation(size_cutoff=1, z_score_cutoff=None, percentile_cutoff=0.5)
        self.assertEqual(passing, [aln.query_id])
        self.assertEqual(not_passing, [])
        os.remove(fn)

    def test_gap_evaluation_percentile_check_two_seq(self):
        fn = write_out_temp_fasta(two_dna_seqs)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        passing, not_passing = aln.gap_evaluation(size_cutoff=2, z_score_cutoff=None, percentile_cutoff=0.5)
        self.assertEqual(passing, [aln.seq_order[1]])
        self.assertEqual(not_passing, [aln.seq_order[0]])
        os.remove(fn)

    def test_gap_evaluation_percentile_check_three_seq(self):
        fn = write_out_temp_fasta(two_protein_seqs + third_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        passing, not_passing = aln.gap_evaluation(size_cutoff=3, z_score_cutoff=None, percentile_cutoff=0.5)
        self.assertEqual(passing, aln.seq_order[1:])
        self.assertEqual(not_passing, [aln.seq_order[0]])
        os.remove(fn)


class TestVisualization(TestCase):

    def test_heatmap_plot_no_name_failure(self):
        fn = write_out_temp_fasta(single_dna_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='DNA')
        aln.import_alignment()
        with self.assertRaises(ValueError):
            aln.heatmap_plot(name=None, out_dir=None, save=False, ax=None)
        os.remove(fn)

    def test_heatmap_plot_only_name(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        df, ax = aln.heatmap_plot(name='Visualization Test', out_dir=None, save=False, ax=None)
        self.assertTrue(df.equals(pd.DataFrame(np.array([[11, 4, 17]]), index=aln.seq_order,
                                               columns=['0:M', '1:E', '2:T'])))
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ['0:M', '1:E', '2:T'])
        self.assertEqual([label.get_text() for label in ax.get_yticklabels()], aln.seq_order)
        self.assertEqual(ax.title.get_text(), 'Visualization Test')
        plt.clf()
        os.remove(fn)

    def test_heatmap_plot_dir_no_save(self):
        test_dir = 'plot_test'
        os.makedirs(test_dir, exist_ok=True)
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        df, ax = aln.heatmap_plot(name='Visualization Test', out_dir=test_dir, save=False, ax=None)
        self.assertTrue(df.equals(pd.DataFrame(np.array([[11, 4, 17]]), index=aln.seq_order,
                                               columns=['0:M', '1:E', '2:T'])))
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ['0:M', '1:E', '2:T'])
        self.assertEqual([label.get_text() for label in ax.get_yticklabels()], aln.seq_order)
        self.assertEqual(ax.title.get_text(), 'Visualization Test')
        self.assertFalse(os.path.isfile(os.path.join(os.getcwd(), test_dir, 'Visualization_Test.eps')))
        plt.clf()
        os.remove(fn)
        rmtree(test_dir)

    def test_heatmap_plot_dir_save(self):
        test_dir = 'plot_test'
        os.makedirs(test_dir, exist_ok=True)
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        df, ax = aln.heatmap_plot(name='Visualization Test', out_dir=test_dir, save=True, ax=None)
        self.assertTrue(df.equals(pd.DataFrame(np.array([[11, 4, 17]]), index=aln.seq_order,
                                               columns=['0:M', '1:E', '2:T'])))
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ['0:M', '1:E', '2:T'])
        self.assertEqual([label.get_text() for label in ax.get_yticklabels()], aln.seq_order)
        self.assertEqual(ax.title.get_text(), 'Visualization Test')
        self.assertTrue(os.path.isfile(os.path.join(os.getcwd(), test_dir, 'Visualization_Test.eps')))
        plt.clf()
        os.remove(fn)
        rmtree(test_dir)

    def test_heatmap_plot_no_dir_save(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        df, ax = aln.heatmap_plot(name='Visualization Test', out_dir=None, save=True, ax=None)
        self.assertTrue(df.equals(pd.DataFrame(np.array([[11, 4, 17]]), index=aln.seq_order,
                                               columns=['0:M', '1:E', '2:T'])))
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ['0:M', '1:E', '2:T'])
        self.assertEqual([label.get_text() for label in ax.get_yticklabels()], aln.seq_order)
        self.assertEqual(ax.title.get_text(), 'Visualization Test')
        self.assertTrue(os.path.isfile(os.path.join(os.getcwd(), 'Visualization_Test.eps')))
        plt.clf()
        os.remove(fn)
        os.remove('Visualization_Test.eps')

    def test_heatmap_plot_dir_no_save_custom_ax(self):
        _, original_ax = plt.subplots(1)
        test_dir = 'plot_test'
        os.makedirs(test_dir, exist_ok=True)
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        df, ax = aln.heatmap_plot(name='Visualization Test', out_dir=test_dir, save=False, ax=original_ax)
        self.assertTrue(df.equals(pd.DataFrame(np.array([[11, 4, 17]]), index=aln.seq_order,
                                               columns=['0:M', '1:E', '2:T'])))
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ['0:M', '1:E', '2:T'])
        self.assertEqual([label.get_text() for label in ax.get_yticklabels()], aln.seq_order)
        self.assertEqual(ax.title.get_text(), 'Visualization Test')
        self.assertTrue(ax is original_ax)
        self.assertFalse(os.path.isfile(os.path.join(os.getcwd(), test_dir, 'Visualization_Test.eps')))
        plt.clf()
        os.remove(fn)
        rmtree(test_dir)

    def test_heatmap_plot_dir_save_custom_ax(self):
        _, original_ax = plt.subplots(1)
        test_dir = 'plot_test'
        os.makedirs(test_dir, exist_ok=True)
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        df, ax = aln.heatmap_plot(name='Visualization Test', out_dir=test_dir, save=True, ax=original_ax)
        self.assertTrue(df.equals(pd.DataFrame(np.array([[11, 4, 17]]), index=aln.seq_order,
                                               columns=['0:M', '1:E', '2:T'])))
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ['0:M', '1:E', '2:T'])
        self.assertEqual([label.get_text() for label in ax.get_yticklabels()], aln.seq_order)
        self.assertEqual(ax.title.get_text(), 'Visualization Test')
        self.assertTrue(ax is original_ax)
        self.assertTrue(os.path.isfile(os.path.join(os.getcwd(), test_dir, 'Visualization_Test.eps')))
        plt.clf()
        os.remove(fn)
        rmtree(test_dir)

    def test_heatmap_plot_no_dir_save_custom_ax(self):
        _, original_ax = plt.subplots(1)
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        df, ax = aln.heatmap_plot(name='Visualization Test', out_dir=None, save=True, ax=original_ax)
        self.assertTrue(df.equals(pd.DataFrame(np.array([[11, 4, 17]]), index=aln.seq_order,
                                               columns=['0:M', '1:E', '2:T'])))
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ['0:M', '1:E', '2:T'])
        self.assertEqual([label.get_text() for label in ax.get_yticklabels()], aln.seq_order)
        self.assertEqual(ax.title.get_text(), 'Visualization Test')
        self.assertTrue(ax is original_ax)
        self.assertTrue(os.path.isfile(os.path.join(os.getcwd(), 'Visualization_Test.eps')))
        plt.clf()
        os.remove(fn)
        os.remove('Visualization_Test.eps')

    def test_heatmap_plot_save_no_overwrite(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        df, ax = aln.heatmap_plot(name='Visualization Test', out_dir=None, save=True, ax=None)
        self.assertTrue(df.equals(pd.DataFrame(np.array([[11, 4, 17]]), index=aln.seq_order,
                                               columns=['0:M', '1:E', '2:T'])))
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()], ['0:M', '1:E', '2:T'])
        self.assertEqual([label.get_text() for label in ax.get_yticklabels()], aln.seq_order)
        self.assertEqual(ax.title.get_text(), 'Visualization Test')
        self.assertTrue(os.path.isfile(os.path.join(os.getcwd(), 'Visualization_Test.eps')))
        df2, ax2 = aln.heatmap_plot(name='Visualization Test', out_dir=None, save=True, ax=None)
        self.assertIsNone(df2)
        self.assertIsNone(ax2)
        plt.clf()
        os.remove(fn)
        os.remove('Visualization_Test.eps')

    def test_heatmap_plot_no_save_overwrite(self):
        fn = write_out_temp_fasta(single_protein_seq)
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


class TestCharacterization(TestCase):

    def test_single_and_pair_false(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        s_res, p_res = aln.characterize_positions(single=False, pair=False, single_size=None, single_mapping=None,
                                                  single_reverse=None, pair_size=None, pair_mapping=None,
                                                  pair_reverse=None)
        self.assertIsNone(s_res)
        self.assertIsNone(p_res)
        os.remove(fn)

    def test_single_no_inputs(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(Gapped(aln.alphabet))
        s_res, p_res = aln.characterize_positions(single=True, pair=False, single_size=None, single_mapping=None,
                                                  single_reverse=None, pair_size=None, pair_mapping=None,
                                                  pair_reverse=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, alpha_size))
        expected_count_matrix[0, mapping['M']] = 1
        expected_count_matrix[1, mapping['E']] = 1
        expected_count_matrix[2, mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsNone(p_res)
        os.remove(fn)

    def test_single_size_only(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(Gapped(aln.alphabet))
        decoy_size, _, _, _ = build_mapping(Gapped(aln.alphabet))
        s_res, p_res = aln.characterize_positions(single=True, pair=False, single_size=decoy_size, single_mapping=None,
                                                  single_reverse=None, pair_size=None, pair_mapping=None,
                                                  pair_reverse=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, alpha_size))
        expected_count_matrix[0, mapping['M']] = 1
        expected_count_matrix[1, mapping['E']] = 1
        expected_count_matrix[2, mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsNone(p_res)
        os.remove(fn)

    def test_single_mapping_only(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(Gapped(aln.alphabet))
        _, _, decoy_mapping, _ = build_mapping(Gapped(aln.alphabet))
        s_res, p_res = aln.characterize_positions(single=True, pair=False, single_size=None,
                                                  single_mapping=decoy_mapping, single_reverse=None, pair_size=None,
                                                  pair_mapping=None, pair_reverse=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, alpha_size))
        expected_count_matrix[0, mapping['M']] = 1
        expected_count_matrix[1, mapping['E']] = 1
        expected_count_matrix[2, mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsNone(p_res)
        os.remove(fn)

    def test_single_reverse_only(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(Gapped(aln.alphabet))
        _, _, _, decoy_reverse = build_mapping(Gapped(aln.alphabet))
        s_res, p_res = aln.characterize_positions(single=True, pair=False, single_size=None, single_mapping=None,
                                                  single_reverse=decoy_reverse, pair_size=None, pair_mapping=None,
                                                  pair_reverse=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, alpha_size))
        expected_count_matrix[0, mapping['M']] = 1
        expected_count_matrix[1, mapping['E']] = 1
        expected_count_matrix[2, mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsNone(p_res)
        os.remove(fn)

    def test_single_all_inputs(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, reverse = build_mapping(Gapped(aln.alphabet))
        s_res, p_res = aln.characterize_positions(single=True, pair=False, single_size=alpha_size,
                                                  single_mapping=mapping, single_reverse=reverse,
                                                  pair_size=None, pair_mapping=None, pair_reverse=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, alpha_size))
        expected_count_matrix[0, mapping['M']] = 1
        expected_count_matrix[1, mapping['E']] = 1
        expected_count_matrix[2, mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsNone(p_res)
        os.remove(fn)

    def test_pair_no_inputs(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions(single=False, pair=True, single_size=None, single_mapping=None,
                                                  single_reverse=None, pair_size=None, pair_mapping=None,
                                                  pair_reverse=None)
        self.assertIsNone(s_res)
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros((int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length),
                                          alpha_size))
        expected_count_matrix[0, mapping['MM']] = 1
        expected_count_matrix[1, mapping['ME']] = 1
        expected_count_matrix[2, mapping['MT']] = 1
        expected_count_matrix[3, mapping['EE']] = 1
        expected_count_matrix[4, mapping['ET']] = 1
        expected_count_matrix[5, mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_pair_size_only(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        decoy_size, _, _, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions(single=False, pair=True, single_size=None, single_mapping=None,
                                                  single_reverse=None, pair_size=decoy_size, pair_mapping=None,
                                                  pair_reverse=None)
        self.assertIsNone(s_res)
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros((int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length),
                                          alpha_size))
        expected_count_matrix[0, mapping['MM']] = 1
        expected_count_matrix[1, mapping['ME']] = 1
        expected_count_matrix[2, mapping['MT']] = 1
        expected_count_matrix[3, mapping['EE']] = 1
        expected_count_matrix[4, mapping['ET']] = 1
        expected_count_matrix[5, mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_pair_mapping_only(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        _, _, decoy_mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions(single=False, pair=True, single_size=None, single_mapping=None,
                                                  single_reverse=None, pair_size=decoy_mapping, pair_mapping=None,
                                                  pair_reverse=None)
        self.assertIsNone(s_res)
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros(
            (int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length), alpha_size))
        expected_count_matrix[0, mapping['MM']] = 1
        expected_count_matrix[1, mapping['ME']] = 1
        expected_count_matrix[2, mapping['MT']] = 1
        expected_count_matrix[3, mapping['EE']] = 1
        expected_count_matrix[4, mapping['ET']] = 1
        expected_count_matrix[5, mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_pair_reverse_only(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        _, _, _, decoy_reverse = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions(single=False, pair=True, single_size=None, single_mapping=None,
                                                  single_reverse=None, pair_size=None, pair_mapping=None,
                                                  pair_reverse=decoy_reverse)
        self.assertIsNone(s_res)
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros(
            (int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length), alpha_size))
        expected_count_matrix[0, mapping['MM']] = 1
        expected_count_matrix[1, mapping['ME']] = 1
        expected_count_matrix[2, mapping['MT']] = 1
        expected_count_matrix[3, mapping['EE']] = 1
        expected_count_matrix[4, mapping['ET']] = 1
        expected_count_matrix[5, mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_pair_all_inputs(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, reverse = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions(single=False, pair=True, single_size=None, single_mapping=None,
                                                  single_reverse=None, pair_size=alpha_size, pair_mapping=mapping,
                                                  pair_reverse=reverse)
        self.assertIsNone(s_res)
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros(
            (int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length), alpha_size))
        expected_count_matrix[0, mapping['MM']] = 1
        expected_count_matrix[1, mapping['ME']] = 1
        expected_count_matrix[2, mapping['MT']] = 1
        expected_count_matrix[3, mapping['EE']] = 1
        expected_count_matrix[4, mapping['ET']] = 1
        expected_count_matrix[5, mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_single_and_pair_no_inputs(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        single_size, _, single_mapping, _ = build_mapping(Gapped(aln.alphabet))
        pair_size, _, pair_mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions(single=True, pair=True, single_size=None, single_mapping=None,
                                                  single_reverse=None, pair_size=None, pair_mapping=None,
                                                  pair_reverse=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, single_size))
        expected_count_matrix[0, single_mapping['M']] = 1
        expected_count_matrix[1, single_mapping['E']] = 1
        expected_count_matrix[2, single_mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros((int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length),
                                          pair_size))
        expected_count_matrix[0, pair_mapping['MM']] = 1
        expected_count_matrix[1, pair_mapping['ME']] = 1
        expected_count_matrix[2, pair_mapping['MT']] = 1
        expected_count_matrix[3, pair_mapping['EE']] = 1
        expected_count_matrix[4, pair_mapping['ET']] = 1
        expected_count_matrix[5, pair_mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_single_and_pair_size_only(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        single_size, _, single_mapping, _ = build_mapping(Gapped(aln.alphabet))
        pair_size, _, pair_mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        single_decoy_size, _, _, _ = build_mapping(Gapped(aln.alphabet))
        pair_decoy_size, _, _, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions(single=True, pair=True, single_size=single_decoy_size,
                                                  single_mapping=None, single_reverse=None,
                                                  pair_size=pair_decoy_size, pair_mapping=None, pair_reverse=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, single_size))
        expected_count_matrix[0, single_mapping['M']] = 1
        expected_count_matrix[1, single_mapping['E']] = 1
        expected_count_matrix[2, single_mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros((int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length),
                                          pair_size))
        expected_count_matrix[0, pair_mapping['MM']] = 1
        expected_count_matrix[1, pair_mapping['ME']] = 1
        expected_count_matrix[2, pair_mapping['MT']] = 1
        expected_count_matrix[3, pair_mapping['EE']] = 1
        expected_count_matrix[4, pair_mapping['ET']] = 1
        expected_count_matrix[5, pair_mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_single_and_pair_mapping_only(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        single_size, _, single_mapping, _ = build_mapping(Gapped(aln.alphabet))
        pair_size, _, pair_mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        _, _, single_decoy_mapping, _ = build_mapping(Gapped(aln.alphabet))
        _, _, pair_decoy_mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions(single=True, pair=True, single_size=None,
                                                  single_mapping=single_decoy_mapping, single_reverse=None,
                                                  pair_size=None, pair_mapping=pair_decoy_mapping, pair_reverse=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, single_size))
        expected_count_matrix[0, single_mapping['M']] = 1
        expected_count_matrix[1, single_mapping['E']] = 1
        expected_count_matrix[2, single_mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros((int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length),
                                          pair_size))
        expected_count_matrix[0, pair_mapping['MM']] = 1
        expected_count_matrix[1, pair_mapping['ME']] = 1
        expected_count_matrix[2, pair_mapping['MT']] = 1
        expected_count_matrix[3, pair_mapping['EE']] = 1
        expected_count_matrix[4, pair_mapping['ET']] = 1
        expected_count_matrix[5, pair_mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_single_and_pair_reverse_only(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        single_size, _, single_mapping, _ = build_mapping(Gapped(aln.alphabet))
        pair_size, _, pair_mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        _, _, _, single_decoy_reverse = build_mapping(Gapped(aln.alphabet))
        _, _, _, pair_decoy_reverse = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions(single=True, pair=True, single_size=None, single_mapping=None,
                                                  single_reverse=single_decoy_reverse, pair_size=None,
                                                  pair_mapping=None, pair_reverse=pair_decoy_reverse)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, single_size))
        expected_count_matrix[0, single_mapping['M']] = 1
        expected_count_matrix[1, single_mapping['E']] = 1
        expected_count_matrix[2, single_mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros((int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length),
                                          pair_size))
        expected_count_matrix[0, pair_mapping['MM']] = 1
        expected_count_matrix[1, pair_mapping['ME']] = 1
        expected_count_matrix[2, pair_mapping['MT']] = 1
        expected_count_matrix[3, pair_mapping['EE']] = 1
        expected_count_matrix[4, pair_mapping['ET']] = 1
        expected_count_matrix[5, pair_mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_single_and_pair_all_inputs(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        single_size, _, single_mapping, single_reverse = build_mapping(Gapped(aln.alphabet))
        pair_size, _, pair_mapping, pair_reverse = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet),
                                                                                       size=2))
        s_res, p_res = aln.characterize_positions(single=True, pair=True, single_size=single_size,
                                                  single_mapping=single_mapping, single_reverse=single_reverse,
                                                  pair_size=pair_size, pair_mapping=pair_mapping,
                                                  pair_reverse=pair_reverse)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, single_size))
        expected_count_matrix[0, single_mapping['M']] = 1
        expected_count_matrix[1, single_mapping['E']] = 1
        expected_count_matrix[2, single_mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros((int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length),
                                          pair_size))
        expected_count_matrix[0, pair_mapping['MM']] = 1
        expected_count_matrix[1, pair_mapping['ME']] = 1
        expected_count_matrix[2, pair_mapping['MT']] = 1
        expected_count_matrix[3, pair_mapping['EE']] = 1
        expected_count_matrix[4, pair_mapping['ET']] = 1
        expected_count_matrix[5, pair_mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    ####################################################################################################################

    def test_single_and_pair_false2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        s_res, p_res = aln.characterize_positions2(single=False, pair=False, single_size=None, single_mapping=None,
                                                   single_reverse=None, pair_size=None, pair_mapping=None,
                                                   pair_reverse=None, single_to_pair=None)
        self.assertIsNone(s_res)
        self.assertIsNone(p_res)
        os.remove(fn)

    def test_single_no_inputs2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(Gapped(aln.alphabet))
        s_res, p_res = aln.characterize_positions2(single=True, pair=False, single_size=None, single_mapping=None,
                                                   single_reverse=None, pair_size=None, pair_mapping=None,
                                                   pair_reverse=None, single_to_pair=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, alpha_size))
        expected_count_matrix[0, mapping['M']] = 1
        expected_count_matrix[1, mapping['E']] = 1
        expected_count_matrix[2, mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsNone(p_res)
        os.remove(fn)

    def test_single_size_only2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(Gapped(aln.alphabet))
        decoy_size, _, _, _ = build_mapping(Gapped(aln.alphabet))
        s_res, p_res = aln.characterize_positions2(single=True, pair=False, single_size=decoy_size, single_mapping=None,
                                                   single_reverse=None, pair_size=None, pair_mapping=None,
                                                   pair_reverse=None, single_to_pair=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, alpha_size))
        expected_count_matrix[0, mapping['M']] = 1
        expected_count_matrix[1, mapping['E']] = 1
        expected_count_matrix[2, mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsNone(p_res)
        os.remove(fn)

    def test_single_mapping_only2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(Gapped(aln.alphabet))
        _, _, decoy_mapping, _ = build_mapping(Gapped(aln.alphabet))
        s_res, p_res = aln.characterize_positions2(single=True, pair=False, single_size=None,
                                                   single_mapping=decoy_mapping, single_reverse=None, pair_size=None,
                                                   pair_mapping=None, pair_reverse=None, single_to_pair=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, alpha_size))
        expected_count_matrix[0, mapping['M']] = 1
        expected_count_matrix[1, mapping['E']] = 1
        expected_count_matrix[2, mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsNone(p_res)
        os.remove(fn)

    def test_single_reverse_only2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(Gapped(aln.alphabet))
        _, _, _, decoy_reverse = build_mapping(Gapped(aln.alphabet))
        s_res, p_res = aln.characterize_positions2(single=True, pair=False, single_size=None, single_mapping=None,
                                                   single_reverse=decoy_reverse, pair_size=None, pair_mapping=None,
                                                   pair_reverse=None, single_to_pair=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, alpha_size))
        expected_count_matrix[0, mapping['M']] = 1
        expected_count_matrix[1, mapping['E']] = 1
        expected_count_matrix[2, mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsNone(p_res)
        os.remove(fn)

    def test_single_all_inputs2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, reverse = build_mapping(Gapped(aln.alphabet))
        s_res, p_res = aln.characterize_positions2(single=True, pair=False, single_size=alpha_size,
                                                   single_mapping=mapping, single_reverse=reverse,
                                                   pair_size=None, pair_mapping=None, pair_reverse=None,
                                                   single_to_pair=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, alpha_size))
        expected_count_matrix[0, mapping['M']] = 1
        expected_count_matrix[1, mapping['E']] = 1
        expected_count_matrix[2, mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsNone(p_res)
        os.remove(fn)

    def test_pair_no_inputs2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions2(single=False, pair=True, single_size=None, single_mapping=None,
                                                   single_reverse=None, pair_size=None, pair_mapping=None,
                                                   pair_reverse=None, single_to_pair=None)
        self.assertIsNone(s_res)
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros((int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length),
                                          alpha_size))
        expected_count_matrix[0, mapping['MM']] = 1
        expected_count_matrix[1, mapping['ME']] = 1
        expected_count_matrix[2, mapping['MT']] = 1
        expected_count_matrix[3, mapping['EE']] = 1
        expected_count_matrix[4, mapping['ET']] = 1
        expected_count_matrix[5, mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_pair_size_only2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        decoy_size, _, _, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions2(single=False, pair=True, single_size=None, single_mapping=None,
                                                   single_reverse=None, pair_size=decoy_size, pair_mapping=None,
                                                   pair_reverse=None, single_to_pair=None)
        self.assertIsNone(s_res)
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros((int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length),
                                          alpha_size))
        expected_count_matrix[0, mapping['MM']] = 1
        expected_count_matrix[1, mapping['ME']] = 1
        expected_count_matrix[2, mapping['MT']] = 1
        expected_count_matrix[3, mapping['EE']] = 1
        expected_count_matrix[4, mapping['ET']] = 1
        expected_count_matrix[5, mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_pair_mapping_only2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        _, _, decoy_mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions2(single=False, pair=True, single_size=None, single_mapping=None,
                                                   single_reverse=None, pair_size=decoy_mapping, pair_mapping=None,
                                                   pair_reverse=None, single_to_pair=None)
        self.assertIsNone(s_res)
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros(
            (int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length), alpha_size))
        expected_count_matrix[0, mapping['MM']] = 1
        expected_count_matrix[1, mapping['ME']] = 1
        expected_count_matrix[2, mapping['MT']] = 1
        expected_count_matrix[3, mapping['EE']] = 1
        expected_count_matrix[4, mapping['ET']] = 1
        expected_count_matrix[5, mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_pair_reverse_only2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        _, _, _, decoy_reverse = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions2(single=False, pair=True, single_size=None, single_mapping=None,
                                                   single_reverse=None, pair_size=None, pair_mapping=None,
                                                   pair_reverse=decoy_reverse, single_to_pair=None)
        self.assertIsNone(s_res)
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros(
            (int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length), alpha_size))
        expected_count_matrix[0, mapping['MM']] = 1
        expected_count_matrix[1, mapping['ME']] = 1
        expected_count_matrix[2, mapping['MT']] = 1
        expected_count_matrix[3, mapping['EE']] = 1
        expected_count_matrix[4, mapping['ET']] = 1
        expected_count_matrix[5, mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_pair_single_to_pair_only2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        alpha_size, _, mapping, reverse = build_mapping(Gapped(aln.alphabet))
        decoy_size, _, pair_mapping, decoy_reverse = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_to_p = np.zeros((alpha_size + 1, alpha_size + 1))
        for char in pair_mapping:
            s_to_p[mapping[char[0]], mapping[char[1]]] = pair_mapping[char]
        s_res, p_res = aln.characterize_positions2(single=False, pair=True, single_size=None, single_mapping=None,
                                                   single_reverse=None, pair_size=None, pair_mapping=None,
                                                   pair_reverse=None, single_to_pair=s_to_p)
        self.assertIsNone(s_res)
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros(
            (int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length), decoy_size))
        expected_count_matrix[0, pair_mapping['MM']] = 1
        expected_count_matrix[1, pair_mapping['ME']] = 1
        expected_count_matrix[2, pair_mapping['MT']] = 1
        expected_count_matrix[3, pair_mapping['EE']] = 1
        expected_count_matrix[4, pair_mapping['ET']] = 1
        expected_count_matrix[5, pair_mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_pair_all_inputs2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        s_size, _, s_mapping, s_reverse = build_mapping(Gapped(aln.alphabet))
        alpha_size, _, mapping, reverse = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_to_p = np.zeros((alpha_size + 1, alpha_size + 1))
        for char in mapping:
            s_to_p[s_mapping[char[0]], s_mapping[char[1]]] = mapping[char]
        s_res, p_res = aln.characterize_positions2(single=False, pair=True, single_size=None, single_mapping=None,
                                                   single_reverse=None, pair_size=alpha_size, pair_mapping=mapping,
                                                   pair_reverse=reverse, single_to_pair=s_to_p)
        self.assertIsNone(s_res)
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros(
            (int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length), alpha_size))
        expected_count_matrix[0, mapping['MM']] = 1
        expected_count_matrix[1, mapping['ME']] = 1
        expected_count_matrix[2, mapping['MT']] = 1
        expected_count_matrix[3, mapping['EE']] = 1
        expected_count_matrix[4, mapping['ET']] = 1
        expected_count_matrix[5, mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_single_and_pair_no_inputs2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        single_size, _, single_mapping, _ = build_mapping(Gapped(aln.alphabet))
        pair_size, _, pair_mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions2(single=True, pair=True, single_size=None, single_mapping=None,
                                                   single_reverse=None, pair_size=None, pair_mapping=None,
                                                   pair_reverse=None, single_to_pair=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, single_size))
        expected_count_matrix[0, single_mapping['M']] = 1
        expected_count_matrix[1, single_mapping['E']] = 1
        expected_count_matrix[2, single_mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros((int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length),
                                          pair_size))
        expected_count_matrix[0, pair_mapping['MM']] = 1
        expected_count_matrix[1, pair_mapping['ME']] = 1
        expected_count_matrix[2, pair_mapping['MT']] = 1
        expected_count_matrix[3, pair_mapping['EE']] = 1
        expected_count_matrix[4, pair_mapping['ET']] = 1
        expected_count_matrix[5, pair_mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_single_and_pair_size_only2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        single_size, _, single_mapping, _ = build_mapping(Gapped(aln.alphabet))
        pair_size, _, pair_mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        single_decoy_size, _, _, _ = build_mapping(Gapped(aln.alphabet))
        pair_decoy_size, _, _, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions2(single=True, pair=True, single_size=single_decoy_size,
                                                   single_mapping=None, single_reverse=None,
                                                   pair_size=pair_decoy_size, pair_mapping=None, pair_reverse=None,
                                                   single_to_pair=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, single_size))
        expected_count_matrix[0, single_mapping['M']] = 1
        expected_count_matrix[1, single_mapping['E']] = 1
        expected_count_matrix[2, single_mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros((int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length),
                                          pair_size))
        expected_count_matrix[0, pair_mapping['MM']] = 1
        expected_count_matrix[1, pair_mapping['ME']] = 1
        expected_count_matrix[2, pair_mapping['MT']] = 1
        expected_count_matrix[3, pair_mapping['EE']] = 1
        expected_count_matrix[4, pair_mapping['ET']] = 1
        expected_count_matrix[5, pair_mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_single_and_pair_mapping_only2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        single_size, _, single_mapping, _ = build_mapping(Gapped(aln.alphabet))
        pair_size, _, pair_mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        _, _, single_decoy_mapping, _ = build_mapping(Gapped(aln.alphabet))
        _, _, pair_decoy_mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions2(single=True, pair=True, single_size=None,
                                                   single_mapping=single_decoy_mapping, single_reverse=None,
                                                   pair_size=None, pair_mapping=pair_decoy_mapping, pair_reverse=None,
                                                   single_to_pair=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, single_size))
        expected_count_matrix[0, single_mapping['M']] = 1
        expected_count_matrix[1, single_mapping['E']] = 1
        expected_count_matrix[2, single_mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros((int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length),
                                          pair_size))
        expected_count_matrix[0, pair_mapping['MM']] = 1
        expected_count_matrix[1, pair_mapping['ME']] = 1
        expected_count_matrix[2, pair_mapping['MT']] = 1
        expected_count_matrix[3, pair_mapping['EE']] = 1
        expected_count_matrix[4, pair_mapping['ET']] = 1
        expected_count_matrix[5, pair_mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_single_and_pair_reverse_only2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        single_size, _, single_mapping, _ = build_mapping(Gapped(aln.alphabet))
        pair_size, _, pair_mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        _, _, _, single_decoy_reverse = build_mapping(Gapped(aln.alphabet))
        _, _, _, pair_decoy_reverse = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_res, p_res = aln.characterize_positions2(single=True, pair=True, single_size=None, single_mapping=None,
                                                   single_reverse=single_decoy_reverse, pair_size=None,
                                                   pair_mapping=None, pair_reverse=pair_decoy_reverse,
                                                   single_to_pair=None)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, single_size))
        expected_count_matrix[0, single_mapping['M']] = 1
        expected_count_matrix[1, single_mapping['E']] = 1
        expected_count_matrix[2, single_mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros((int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length),
                                          pair_size))
        expected_count_matrix[0, pair_mapping['MM']] = 1
        expected_count_matrix[1, pair_mapping['ME']] = 1
        expected_count_matrix[2, pair_mapping['MT']] = 1
        expected_count_matrix[3, pair_mapping['EE']] = 1
        expected_count_matrix[4, pair_mapping['ET']] = 1
        expected_count_matrix[5, pair_mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_single_and_pair_single_to_pair_only2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        single_size, _, single_mapping, _ = build_mapping(Gapped(aln.alphabet))
        pair_size, _, pair_mapping, _ = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        _, _, _, single_decoy_reverse = build_mapping(Gapped(aln.alphabet))
        _, _, _, pair_decoy_reverse = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet), size=2))
        s_to_p = np.zeros((single_size + 1, single_size + 1))
        for char in pair_mapping:
            s_to_p[single_mapping[char[0]], single_mapping[char[1]]] = pair_mapping[char]
        s_res, p_res = aln.characterize_positions2(single=True, pair=True, single_size=None, single_mapping=None,
                                                   single_reverse=None, pair_size=None, pair_mapping=None,
                                                   pair_reverse=None, single_to_pair=s_to_p)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, single_size))
        expected_count_matrix[0, single_mapping['M']] = 1
        expected_count_matrix[1, single_mapping['E']] = 1
        expected_count_matrix[2, single_mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros((int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length),
                                          pair_size))
        expected_count_matrix[0, pair_mapping['MM']] = 1
        expected_count_matrix[1, pair_mapping['ME']] = 1
        expected_count_matrix[2, pair_mapping['MT']] = 1
        expected_count_matrix[3, pair_mapping['EE']] = 1
        expected_count_matrix[4, pair_mapping['ET']] = 1
        expected_count_matrix[5, pair_mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)

    def test_single_and_pair_all_inputs2(self):
        fn = write_out_temp_fasta(single_protein_seq)
        aln = SeqAlignment(file_name=fn, query_id='test', polymer_type='Protein')
        aln.import_alignment()
        single_size, _, single_mapping, single_reverse = build_mapping(Gapped(aln.alphabet))
        pair_size, _, pair_mapping, pair_reverse = build_mapping(MultiPositionAlphabet(Gapped(aln.alphabet),
                                                                                       size=2))
        s_to_p = np.zeros((single_size + 1, single_size + 1))
        for char in pair_mapping:
            s_to_p[single_mapping[char[0]], single_mapping[char[1]]] = pair_mapping[char]
        s_res, p_res = aln.characterize_positions2(single=True, pair=True, single_size=single_size,
                                                   single_mapping=single_mapping, single_reverse=single_reverse,
                                                   pair_size=pair_size, pair_mapping=pair_mapping,
                                                   pair_reverse=pair_reverse, single_to_pair=s_to_p)
        self.assertIsInstance(s_res, FrequencyTable)
        self.assertFalse((s_res.get_positions() - np.array([0, 1, 2])).any())
        self.assertEqual(s_res.get_chars(0), np.array(['M']))
        self.assertEqual(s_res.get_chars(1), np.array(['E']))
        self.assertEqual(s_res.get_chars(2), np.array(['T']))
        expected_count_matrix = np.zeros((aln.seq_length, single_size))
        expected_count_matrix[0, single_mapping['M']] = 1
        expected_count_matrix[1, single_mapping['E']] = 1
        expected_count_matrix[2, single_mapping['T']] = 1
        self.assertFalse((s_res.get_count_matrix() - expected_count_matrix).any())
        self.assertIsInstance(p_res, FrequencyTable)
        self.assertFalse((p_res.get_positions() - np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])).any())
        self.assertEqual(p_res.get_chars((0, 0)), np.array(['MM']))
        self.assertEqual(p_res.get_chars((0, 1)), np.array(['ME']))
        self.assertEqual(p_res.get_chars((0, 2)), np.array(['MT']))
        self.assertEqual(p_res.get_chars((1, 1)), np.array(['EE']))
        self.assertEqual(p_res.get_chars((1, 2)), np.array(['ET']))
        self.assertEqual(p_res.get_chars((2, 2)), np.array(['TT']))
        expected_count_matrix = np.zeros((int(((aln.seq_length ** 2 - aln.seq_length) / 2) + aln.seq_length),
                                          pair_size))
        expected_count_matrix[0, pair_mapping['MM']] = 1
        expected_count_matrix[1, pair_mapping['ME']] = 1
        expected_count_matrix[2, pair_mapping['MT']] = 1
        expected_count_matrix[3, pair_mapping['EE']] = 1
        expected_count_matrix[4, pair_mapping['ET']] = 1
        expected_count_matrix[5, pair_mapping['TT']] = 1
        self.assertFalse((p_res.get_count_matrix() - expected_count_matrix).any())
        os.remove(fn)


if __name__ == '__main__':
    unittest.main()
