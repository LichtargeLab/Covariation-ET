"""
Created on July 10, 2019

@author: Daniel Konecki
"""
import os
import unittest
import numpy as np
import pandas as pd
from unittest import TestCase
from itertools import combinations
from scipy.sparse import csc_matrix
from test_Base import (protein_seq1, protein_seq2, protein_seq3, dna_seq1, dna_seq2, dna_seq3,  dna_alpha,
                       dna_alpha_size, dna_map, dna_rev, protein_alpha_size, protein_map, protein_rev,
                       pro_quad_alpha_size, pro_quad_map, pro_quad_rev, dna_pair_alpha_size, dna_pair_map, dna_pair_rev,
                       dna_single_to_pair, pro_pair_alpha_size, pro_pair_map, pro_pair_rev, pro_single_to_pair,
                       pro_pair_mismatch, pro_single_to_pair_map, pro_pair_to_quad, pro_quad_mismatch, protein_num_aln,
                       write_out_temp_fn)
from SeqAlignment import SeqAlignment
from FrequencyTable import FrequencyTable
from MatchMismatchTable import MatchMismatchTable

dna_aln_str = f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}'
dna_one_seq_aln_str = f'>seq1\n{str(dna_seq1.seq)}'
protein_aln_str = f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}'
protein_one_seq_aln_str = f'>seq1\n{str(protein_seq1.seq)}'


class TestFrequencyTableInit(TestCase):

    def evaluate_init(self, f_table, alpha_size, mapping, rev, seq_len, pos_size):
        self.assertEqual(f_table.mapping, mapping)
        self.assertTrue((f_table.reverse_mapping == rev).all())
        self.assertEqual(f_table.sequence_length, seq_len)
        if pos_size == 1:
            expected_num_pos = seq_len
        else:
            expected_num_pos = len(list(combinations(range(seq_len), pos_size))) + seq_len
        self.assertEqual(f_table.num_pos, expected_num_pos)
        self.assertEqual(f_table.get_depth(), 0)
        self.assertEqual(f_table.get_table(), {'values': [], 'i': [], 'j': [],
                                               'shape': (expected_num_pos, alpha_size)})

    def test__init_single_pos_dna_alphabet(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        self.evaluate_init(freq_table, dna_alpha_size, dna_map, dna_rev, 18, 1)

    def test__init_single_pos_protein_alphabet(self):
        freq_table = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
        self.evaluate_init(freq_table, protein_alpha_size, protein_map, protein_rev, 6, 1)

    def test__init_pair_pos_dna_alphabet(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        self.evaluate_init(freq_table, dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)

    def test__init_pair_pos_protein_alphabet(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        self.evaluate_init(freq_table, pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)

    def test__init_failure_mismatched_mappings(self):
        dna_rev_false = {(dna_alpha_size - (i + 1)): letter for i, letter in enumerate(dna_alpha.letters)}
        with self.assertRaises(ValueError):
            FrequencyTable(dna_alpha_size, dna_map, dna_rev_false, 18, 1)

    def test__init_failure_small_mapping(self):
        with self.assertRaises(ValueError):
            FrequencyTable(protein_alpha_size, dna_map, protein_rev, 6, 1)

    def test__init_failure_small_reverse_mapping(self):
        with self.assertRaises(ValueError):
            FrequencyTable(protein_alpha_size, protein_map, dna_rev, 6, 1)

    def test__init_failure_position_size_alphabet_mismatch(self):
        with self.assertRaises(ValueError):
            FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 2)

    def test__init_failure_large_position_size(self):
        with self.assertRaises(ValueError):
            FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 3)


class TestFrequencyTableConvertPos(TestCase):

    def evaluate_convert_pos(self, alpha_size, mapping, rev, seq_len, pos_size, curr_pos, expected_pos):
        freq_table = FrequencyTable(alpha_size, mapping, rev, seq_len, pos_size)
        pos = freq_table._convert_pos(curr_pos)
        self.assertEqual(pos, expected_pos)

    def test__convert_pos_single_pos_first(self):
        self.evaluate_convert_pos(alpha_size=dna_alpha_size, mapping=dna_map, rev=dna_rev, seq_len=18, pos_size=1,
                                  curr_pos=0, expected_pos=0)

    def test__convert_pos_single_pos_middle(self):
        self.evaluate_convert_pos(alpha_size=dna_alpha_size, mapping=dna_map, rev=dna_rev, seq_len=18, pos_size=1,
                                  curr_pos=9, expected_pos=9)

    def test__convert_pos_single_pos_last(self):
        self.evaluate_convert_pos(alpha_size=dna_alpha_size, mapping=dna_map, rev=dna_rev, seq_len=18, pos_size=1,
                                  curr_pos=17, expected_pos=17)

    def test__convert_pos_pair_pos_first(self):
        self.evaluate_convert_pos(alpha_size=dna_pair_alpha_size, mapping=dna_pair_map, rev=dna_pair_rev, seq_len=18,
                                  pos_size=2, curr_pos=(0, 0), expected_pos=0)

    def test__convert_pos_pair_pos_middle(self):
        self.evaluate_convert_pos(alpha_size=dna_pair_alpha_size, mapping=dna_pair_map, rev=dna_pair_rev, seq_len=18,
                                  pos_size=2, curr_pos=(1, 2), expected_pos=18 + 2 - 1)

    def test__convert_pos_pair_pos_last(self):
        self.evaluate_convert_pos(alpha_size=dna_pair_alpha_size, mapping=dna_pair_map, rev=dna_pair_rev, seq_len=18,
                                  pos_size=2, curr_pos=(17, 17),
                                  expected_pos=len(list(combinations(range(18), 2))) + 18 - 1)

    def test__convert_pos_failure_tuple_single_pos(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        with self.assertRaises(TypeError):
            freq_table._convert_pos((0, 0))

    def test__convert_pos_failure_int_pair_pos(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        with self.assertRaises(TypeError):
            freq_table._convert_pos(0)

    def test__convert_pos_failure_out_of_bounds_low_single_pos(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        with self.assertRaises(ValueError):
            freq_table._convert_pos(-1)

    def test__convert_pos_failure_out_of_bounds_high_single_pos(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        with self.assertRaises(ValueError):
            freq_table._convert_pos(18)

    def test__convert_pos_failure_out_of_bounds_low_pair_pos(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        with self.assertRaises(ValueError):
            freq_table._convert_pos((0, -1))

    def test__convert_pos_failure_out_of_bounds_high_pair_pos(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        with self.assertRaises(ValueError):
            freq_table._convert_pos((0, 18))


class TestFrequencyTableIncrementCount(TestCase):

    def evaluate_increment_count(self, freq_table, first_table, second_table, expected_values, expected_positions,
                                 expected_chars, expected_shape, expected_depth):
        self.assertNotEqual(first_table, second_table)
        self.assertEqual(second_table['values'], expected_values)
        self.assertEqual(second_table['i'], expected_positions)
        self.assertEqual(second_table['j'], expected_chars)
        self.assertEqual(second_table['shape'], expected_shape)
        self.assertEqual(freq_table.get_depth(), expected_depth)

    def test__increment_count_single_pos_default(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=0, char='A')
        t2 = freq_table.get_table()
        self.evaluate_increment_count(freq_table=freq_table, first_table=t1, second_table=t2, expected_values=[1],
                                      expected_positions=[0], expected_chars=[0], expected_depth=0,
                                      expected_shape=(freq_table.num_pos, dna_alpha_size))

    def test__increment_count_single_pos_one(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=0, char='A', amount=1)
        t2 = freq_table.get_table()
        self.evaluate_increment_count(freq_table=freq_table, first_table=t1, second_table=t2, expected_values=[1],
                                      expected_positions=[0], expected_chars=[0], expected_depth=0,
                                      expected_shape=(freq_table.num_pos, dna_alpha_size))

    def test__increment_count_single_pos_two(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=0, char='A', amount=2)
        t2 = freq_table.get_table()
        self.evaluate_increment_count(freq_table=freq_table, first_table=t1, second_table=t2, expected_values=[2],
                                      expected_positions=[0], expected_chars=[0], expected_depth=0,
                                      expected_shape=(freq_table.num_pos, dna_alpha_size))

    def test__increment_count_pair_pos_default(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=(0, 1), char='AA')
        t2 = freq_table.get_table()
        self.evaluate_increment_count(freq_table=freq_table, first_table=t1, second_table=t2, expected_values=[1],
                                      expected_positions=[1], expected_chars=[0], expected_depth=0,
                                      expected_shape=(freq_table.num_pos, dna_pair_alpha_size))

    def test__increment_count_pair_pos_one(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=(0, 1), char='AA', amount=1)
        t2 = freq_table.get_table()
        self.evaluate_increment_count(freq_table=freq_table, first_table=t1, second_table=t2, expected_values=[1],
                                      expected_positions=[1], expected_chars=[0], expected_depth=0,
                                      expected_shape=(freq_table.num_pos, dna_pair_alpha_size))

    def test__increment_count_pair_pos_two(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=(0, 1), char='AA', amount=2)
        t2 = freq_table.get_table()
        self.evaluate_increment_count(freq_table=freq_table, first_table=t1, second_table=t2, expected_values=[2],
                                      expected_positions=[1], expected_chars=[0], expected_depth=0,
                                      expected_shape=(freq_table.num_pos, dna_pair_alpha_size))

    def test__increment_count_single_pos_multiple_increments(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=0, char='A', amount=1)
        t2 = freq_table.get_table()
        self.evaluate_increment_count(freq_table=freq_table, first_table=t1, second_table=t2, expected_values=[1],
                                      expected_positions=[0], expected_chars=[0], expected_depth=0,
                                      expected_shape=(freq_table.num_pos, dna_alpha_size))
        freq_table._increment_count(pos=0, char='A', amount=1)
        t3 = freq_table.get_table()
        self.evaluate_increment_count(freq_table=freq_table, first_table=t2, second_table=t3, expected_values=[1, 1],
                                      expected_positions=[0, 0], expected_chars=[0, 0], expected_depth=0,
                                      expected_shape=(freq_table.num_pos, dna_alpha_size))

    def test__increment_count_single_pos_multiple_different_increments(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=0, char='A', amount=1)
        t2 = freq_table.get_table()
        self.evaluate_increment_count(freq_table=freq_table, first_table=t1, second_table=t2, expected_values=[1],
                                      expected_positions=[0], expected_chars=[0], expected_depth=0,
                                      expected_shape=(freq_table.num_pos, dna_alpha_size))
        freq_table._increment_count(pos=5, char='C', amount=1)
        t3 = freq_table.get_table()
        self.evaluate_increment_count(freq_table=freq_table, first_table=t2, second_table=t3, expected_values=[1, 1],
                                      expected_positions=[0, 5], expected_chars=[0, 2], expected_depth=0,
                                      expected_shape=(freq_table.num_pos, dna_alpha_size))

    def test__increment_count_pair_pos_multiple_increments(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=(0, 1), char='AA', amount=1)
        t2 = freq_table.get_table()
        self.evaluate_increment_count(freq_table=freq_table, first_table=t1, second_table=t2, expected_values=[1],
                                      expected_positions=[1], expected_chars=[0], expected_depth=0,
                                      expected_shape=(freq_table.num_pos, dna_pair_alpha_size))
        freq_table._increment_count(pos=(0, 1), char='AA', amount=1)
        t3 = freq_table.get_table()
        self.evaluate_increment_count(freq_table=freq_table, first_table=t2, second_table=t3, expected_values=[1, 1],
                                      expected_positions=[1, 1], expected_chars=[0, 0], expected_depth=0,
                                      expected_shape=(freq_table.num_pos, dna_pair_alpha_size))

    def test__increment_count_pair_pos_multiple_different_increments(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=(0, 1), char='AA', amount=1)
        t2 = freq_table.get_table()
        self.evaluate_increment_count(freq_table=freq_table, first_table=t1, second_table=t2, expected_values=[1],
                                      expected_positions=[1], expected_chars=[0], expected_depth=0,
                                      expected_shape=(freq_table.num_pos, dna_pair_alpha_size))
        freq_table._increment_count(pos=(0, 4), char='AC', amount=1)
        t3 = freq_table.get_table()
        self.evaluate_increment_count(freq_table=freq_table, first_table=t2, second_table=t3, expected_values=[1, 1],
                                      expected_positions=[1, 4], expected_chars=[0, 2], expected_depth=0,
                                      expected_shape=(freq_table.num_pos, dna_pair_alpha_size))

    def test__increment_count_single_pos_failure_negative(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        with self.assertRaises(ValueError):
            freq_table._increment_count(pos=0, char='A', amount=-1)

    def test__increment_count_pair_pos_failure_negative(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        with self.assertRaises(ValueError):
            freq_table._increment_count(pos=(0, 0), char='AA', amount=-1)

    def test__increment_count_single_pos_failure_low_pos(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        with self.assertRaises(ValueError):
            freq_table._increment_count(pos=-1, char='A')

    def test__increment_count_pair_pos_failure_low_pos(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        with self.assertRaises(ValueError):
            freq_table._increment_count(pos=(0, -1), char='AA')

    def test__increment_count_single_pos_failure_high_pos(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        with self.assertRaises(ValueError):
            freq_table._increment_count(pos=19, char='A')

    def test__increment_count_pair_pos_failure_high_pos(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        with self.assertRaises(ValueError):
            freq_table._increment_count(pos=(0, 19), char='AA')

    def test__increment_count_single_pos_failure_bad_character(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        with self.assertRaises(KeyError):
            freq_table._increment_count(pos=0, char='J')

    def test__increment_count_pair_pos_failure_bad_character(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        with self.assertRaises(KeyError):
            freq_table._increment_count(pos=(0, 0), char='JJ')


class TestFrequencyTableFinalizeTable(TestCase):

    def evaluate_finalize_table(self, table1, table2, expected_sum, expected_value):
        self.assertNotEqual(type(table1), type(table2))
        self.assertEqual(type(table2), csc_matrix)
        self.assertEqual(np.sum(table2), expected_sum)
        self.assertEqual(table2[0, 0], expected_value)

    def test_finalize_table_no_increments(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table.finalize_table()
        t2 = freq_table.get_table()
        self.evaluate_finalize_table(table1=t1, table2=t2, expected_sum=0, expected_value=0)

    def test_finalize_table_one_increments(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=0, char='A', amount=1)
        freq_table.finalize_table()
        t2 = freq_table.get_table()
        self.evaluate_finalize_table(table1=t1, table2=t2, expected_sum=1, expected_value=1)

    def test_finalize_table_two_increments(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=0, char='A', amount=1)
        freq_table._increment_count(pos=0, char='A', amount=1)
        freq_table.finalize_table()
        t2 = freq_table.get_table()
        self.evaluate_finalize_table(table1=t1, table2=t2, expected_sum=2, expected_value=2)


class TestFrequencyTableSetDepth(TestCase):

    def test_set_depth_zero(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.set_depth(0)
        self.assertEqual(freq_table.get_depth(), 0)

    def test_set_depth_non_zero(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.set_depth(5)
        self.assertEqual(freq_table.get_depth(), 5)

    def test_set_depth_multiple_sets(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.set_depth(1)
        self.assertEqual(freq_table.get_depth(), 1)
        freq_table.set_depth(5)
        self.assertEqual(freq_table.get_depth(), 5)
        freq_table.set_depth(0)
        self.assertEqual(freq_table.get_depth(), 0)

    def test_set_depth_failure_negative(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        with self.assertRaises(ValueError):
            freq_table.set_depth(-1)

    def test_set_depth_failure_none(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        with self.assertRaises(ValueError):
            freq_table.set_depth(None)


class TestFrequencyTableCharacterizeAlignment(TestCase):

    def evaluate_characterize_alignment(self, freq_table, table1, table2, expected_table, expected_depth):
        self.assertNotEqual(table1, table2)
        self.assertTrue(isinstance(table2, csc_matrix))
        self.assertFalse((table2 - expected_table).any())
        self.assertEqual(freq_table.get_depth(), expected_depth)

    def test_characterize_alignment_dna_single_pos(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=dna_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table.characterize_alignment(num_aln=num_aln)
        t2 = freq_table.get_table()
        expected_table = np.zeros(shape=(18, dna_alpha_size))
        idx1 = [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16,
                17, 17]
        idx2 = [0, 1, 3, 3, 4, 0, 4, 3, 4, 0, 1, 1, 2, 1, 0, 4, 3, 4, 0, 4, 3, 4, 0, 4, 3, 4, 3, 4, 0, 4, 3, 4]
        values = [3, 3, 3, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        expected_table[idx1, idx2] = values
        self.evaluate_characterize_alignment(freq_table=freq_table, table1=t1, table2=t2, expected_table=expected_table,
                                             expected_depth=3)
        os.remove(aln_fn)

    def test_characterize_alignment_protein_single_pos(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=protein_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=protein_map)
        freq_table = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
        t1 = freq_table.get_table()
        freq_table.characterize_alignment(num_aln=num_aln)
        t2 = freq_table.get_table()
        expected_table = np.zeros((6, protein_alpha_size))
        idx1 = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        idx2 = [11, 4, 23, 5, 17, 15, 23, 4, 23, 4, 23]
        values = [3, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1]
        expected_table[idx1, idx2] = values
        self.evaluate_characterize_alignment(freq_table=freq_table, table1=t1, table2=t2, expected_table=expected_table,
                                             expected_depth=3)
        os.remove(aln_fn)

    def test_characterize_alignment_dna_pair_pos(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=dna_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        t1 = freq_table.get_table()
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=dna_single_to_pair)
        t2 = freq_table.get_table()
        expected_table = np.zeros(shape=(freq_table.num_pos, dna_pair_alpha_size))
        idx1 = [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16,
                17, 17, 18, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31,
                31, 32, 32, 33, 33, 34, 34, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 42, 42, 43, 43, 44, 44, 45,
                45, 46, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 57,
                57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 68, 69, 69,
                69, 70, 70, 71, 71, 72, 72, 73, 73, 74, 74, 75, 75, 76, 76, 77, 77, 78, 78, 79, 79, 80, 80, 81, 81, 81,
                82, 82, 82, 83, 83, 84, 84, 85, 85, 86, 86, 87, 87, 88, 88, 89, 89, 90, 90, 91, 91, 92, 92, 93, 93, 94,
                94, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102,
                103, 103, 103, 104, 104, 104, 105, 105, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110,
                110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 117, 117, 118, 118,
                119, 119, 120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 126, 126, 127, 127, 128, 128, 129,
                129, 130, 130, 131, 131, 132, 132, 133, 133, 134, 134, 135, 135, 136, 136, 137, 137, 138, 138, 139, 139,
                140, 140, 141, 141, 142, 142, 143, 143, 144, 144, 145, 145, 146, 146, 147, 147, 148, 148, 149, 149, 150,
                150, 151, 151, 152, 152, 153, 153, 154, 154, 155, 155, 156, 156, 157, 157, 158, 158, 159, 159, 160, 160,
                161, 161, 162, 162, 163, 163, 164, 164, 165, 165, 166, 166, 167, 167, 168, 168, 169, 169, 170, 170]
        idx2 = [0, 1, 3, 3, 4, 0, 4, 3, 4, 0, 1, 1, 2, 1, 0, 4, 3, 4, 0, 4, 3, 4, 0, 4, 3, 4, 3, 4, 0, 4, 3, 4, 6, 8, 8,
                9, 5, 9, 8, 9, 5, 6, 6, 7, 6, 5, 9, 8, 9, 5, 9, 8, 9, 5, 9, 8, 9, 8, 9, 5, 9, 8, 9, 18, 18, 19, 15, 19,
                18, 19, 15, 16, 16, 17, 16, 15, 19, 18, 19, 15, 19, 18, 19, 15, 19, 18, 19, 18, 19, 15, 19, 18, 19, 18,
                24, 15, 24, 18, 24, 15, 20, 21, 17, 21, 22, 16, 21, 19, 20, 19, 23, 19, 20, 19, 23, 19, 20, 19, 23, 19,
                23, 19, 20, 19, 23, 0, 24, 3, 24, 0, 20, 21, 2, 21, 22, 1, 21, 4, 20, 4, 23, 4, 20, 4, 23, 4, 20, 4, 23,
                4, 23, 4, 20, 4, 23, 18, 24, 15, 20, 21, 17, 21, 22, 16, 21, 19, 20, 19, 23, 19, 20, 19, 23, 19, 20, 19,
                23, 19, 23, 19, 20, 19, 23, 0, 6, 2, 6, 1, 6, 0, 4, 5, 3, 4, 8, 0, 4, 5, 3, 4, 8, 0, 4, 5, 3, 4, 8, 3,
                4, 8, 0, 4, 5, 3, 4, 8, 6, 12, 6, 11, 5, 10, 14, 8, 13, 14, 5, 10, 14, 8, 13, 14, 5, 10, 14, 8, 13, 14,
                8, 13, 14, 5, 10, 14, 8, 13, 14, 6, 5, 9, 8, 9, 5, 9, 8, 9, 5, 9, 8, 9, 8, 9, 5, 9, 8, 9, 0, 24, 3, 24,
                0, 24, 3, 24, 0, 24, 3, 24, 3, 24, 0, 24, 3, 24, 18, 24, 15, 24, 18, 24, 15, 24, 18, 24, 18, 24, 15, 24,
                18, 24, 0, 24, 3, 24, 0, 24, 3, 24, 3, 24, 0, 24, 3, 24, 18, 24, 15, 24, 18, 24, 18, 24, 15, 24, 18,
                24, 0, 24, 3, 24, 3, 24, 0, 24, 3, 24, 18, 24, 18, 24, 15, 24, 18, 24, 18, 24, 15, 24, 18, 24, 0, 24, 3,
                24, 18, 24]
        values = [3, 3, 3, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3, 3,
                  1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3, 1, 2, 1, 2,
                  1, 2, 2, 1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1,
                  1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1,
                  2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1,
                  2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
                  2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
                  2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
                  2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        expected_table[idx1, idx2] = values
        self.evaluate_characterize_alignment(freq_table=freq_table, table1=t1, table2=t2, expected_table=expected_table,
                                             expected_depth=3)
        os.remove(aln_fn)

    def test_characterize_alignment_protein_pair_pos(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=protein_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=protein_map)
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        t1 = freq_table.get_table()
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=pro_single_to_pair)
        t2 = freq_table.get_table()
        expected_table = np.zeros((freq_table.num_pos, pro_pair_alpha_size))
        idx1 = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 12, 13, 13, 13, 14,
                14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20]
        idx2 = [275, 268, 287, 269, 281, 279, 287, 268, 287, 268, 287, 100, 575, 113, 557, 569, 119, 567, 119, 556, 119,
                556, 125, 425, 135, 423, 431, 124, 412, 431, 124, 412, 431, 375, 575, 364, 575, 364, 575,
                100, 575, 100, 575, 100, 575]
        values = [3, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        expected_table[idx1, idx2] = values
        self.evaluate_characterize_alignment(freq_table=freq_table, table1=t1, table2=t2, expected_table=expected_table,
                                             expected_depth=3)
        os.remove(aln_fn)

    def test_characterize_alignment_dna_single_pos_single_to_pair(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=dna_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=dna_single_to_pair)
        t2 = freq_table.get_table()
        expected_table = np.zeros(shape=(18, dna_alpha_size))
        idx1 = [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17]
        idx2 = [0, 1, 3, 3, 4, 0, 4, 3, 4, 0, 1, 1, 2, 1, 0, 4, 3, 4, 0, 4, 3, 4, 0, 4, 3, 4, 3, 4, 0, 4, 3, 4]
        values = [3, 3, 3, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        expected_table[idx1, idx2] = values
        self.evaluate_characterize_alignment(freq_table=freq_table, table1=t1, table2=t2, expected_table=expected_table,
                                             expected_depth=3)
        os.remove(aln_fn)

    def test_characterize_alignment_protein_single_pos_single_to_pair(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=protein_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=protein_map)
        freq_table = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
        t1 = freq_table.get_table()
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=pro_single_to_pair)
        t2 = freq_table.get_table()
        expected_table = np.zeros((6, protein_alpha_size))
        idx1 = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        idx2 = [11, 4, 23, 5, 17, 15, 23, 4, 23, 4, 23]
        values = [3, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1]
        expected_table[idx1, idx2] = values
        self.evaluate_characterize_alignment(freq_table=freq_table, table1=t1, table2=t2, expected_table=expected_table,
                                             expected_depth=3)
        os.remove(aln_fn)

    def test_characterize_alignment_failure_dna_pair_pos_no_single_to_pair(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=dna_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=None)
        os.remove(aln_fn)

    def test_characterize_alignment_failure_protein_pair_pos_no_single_to_pair(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=protein_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=protein_map)
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=None)
        os.remove(aln_fn)

    def test_characterize_alignment_failure_dna_single_pos_no_aln(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment(num_aln=None)

    def test_characterize_alignment_failure_protein_single_pos_no_aln(self):
        freq_table = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment(num_aln=None)

    def test_characterize_alignment_failure_dna_pair_pos_no_aln(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment(num_aln=None)

    def test_characterize_alignment_failure_protein_pair_pos_no_aln(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment(num_aln=None)


class TestFrequencyTableCharacterizeAlignmentMM(TestCase):

    def evaluate_characterize_alignment_mm(self, freq_table, freq_table2, table1, expected_table1, expected_table2,
                                           expected_depth):
        table2 = freq_table.get_table()
        self.assertNotEqual(table1, table2)
        self.assertTrue(isinstance(table2, csc_matrix))
        self.assertFalse((table2 - expected_table1).any())
        table3 = freq_table2.get_table()
        self.assertTrue(isinstance(table3, csc_matrix))
        self.assertFalse((table3 - expected_table2).any())
        self.assertEqual(freq_table.get_depth(), expected_depth)
        self.assertEqual(freq_table2.get_depth(), expected_depth)

    def test_characterize_alignment_protein_single_pos(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        t1 = freq_table.get_table()
        freq_table2 = freq_table.characterize_alignment_mm(num_aln=protein_num_aln, comparison=pro_single_to_pair,
                                                           mismatch_mask=pro_pair_mismatch, single_to_pair=None)
        expected_table = np.zeros((6, pro_pair_alpha_size), dtype=np.int32)
        idx1 = [0, 1, 2, 3, 4, 5]
        idx2 = [275, 575, 425, 375, 100, 100]
        values = [3, 1, 1, 1, 1, 1]
        expected_table[idx1, idx2] = values
        expected_table2 = np.zeros((6, pro_pair_alpha_size), dtype=np.int32)
        idx1 = [1, 2, 3, 4, 5]
        idx2 = [119, 413, 567, 556, 556]
        values = [2, 2, 2, 2, 2]
        expected_table2[idx1, idx2] = values
        self.evaluate_characterize_alignment_mm(freq_table=freq_table, freq_table2=freq_table2, table1=t1,
                                                expected_table1=expected_table, expected_table2=expected_table2,
                                                expected_depth=3)

    def test_characterize_alignment_protein_single_pos_partial1(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        t1 = freq_table.get_table()
        freq_table2 = freq_table.characterize_alignment_mm(num_aln=protein_num_aln, comparison=pro_single_to_pair,
                                                           mismatch_mask=pro_pair_mismatch, single_to_pair=None,
                                                           indexes1=np.array([0], dtype=np.int32),
                                                           indexes2=np.array([1, 2], dtype=np.int32))
        expected_table = np.zeros((6, pro_pair_alpha_size), dtype=np.int32)
        idx1 = [0, 2]
        idx2 = [275, 425]
        values = [2, 1]
        expected_table[idx1, idx2] = values
        expected_table2 = np.zeros((6, pro_pair_alpha_size), dtype=np.int32)
        idx1 = [1, 2, 3, 4, 5]
        idx2 = [119, 413, 567, 556, 556]
        values = [2, 1, 2, 2, 2]
        expected_table2[idx1, idx2] = values
        self.evaluate_characterize_alignment_mm(freq_table=freq_table, freq_table2=freq_table2, table1=t1,
                                                expected_table1=expected_table, expected_table2=expected_table2,
                                                expected_depth=3)

    def test_characterize_alignment_protein_single_pos_composite1(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        t1 = freq_table.get_table()
        freq_table2 = freq_table.characterize_alignment_mm(num_aln=protein_num_aln[[0], :],
                                                           comparison=pro_single_to_pair,
                                                           mismatch_mask=pro_pair_mismatch, single_to_pair=None)
        freq_table3 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        freq_table4 = freq_table3.characterize_alignment_mm(num_aln=protein_num_aln[[1, 2], :],
                                                            comparison=pro_single_to_pair,
                                                            mismatch_mask=pro_pair_mismatch, single_to_pair=None)
        freq_table5 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        freq_table6 = freq_table5.characterize_alignment_mm(num_aln=protein_num_aln,
                                                            comparison=pro_single_to_pair,
                                                            mismatch_mask=pro_pair_mismatch, single_to_pair=None,
                                                            indexes1=np.array([0], dtype=np.int32),
                                                            indexes2=np.array([1, 2], dtype=np.int32))
        freq_table_match = freq_table + freq_table3 + freq_table5
        freq_table_match.set_depth(depth=3)
        freq_table_mismatch = freq_table2 + freq_table4 + freq_table6
        freq_table_mismatch.set_depth(depth=3)

        expected_table = np.zeros((6, pro_pair_alpha_size), dtype=np.int32)
        idx1 = [0, 1, 2, 3, 4, 5]
        idx2 = [275, 575, 425, 375, 100, 100]
        values = [3, 1, 1, 1, 1, 1]
        expected_table[idx1, idx2] = values
        expected_table2 = np.zeros((6, pro_pair_alpha_size), dtype=np.int32)
        idx1 = [1, 2, 3, 4, 5]
        idx2 = [119, 413, 567, 556, 556]
        values = [2, 2, 2, 2, 2]
        expected_table2[idx1, idx2] = values
        self.evaluate_characterize_alignment_mm(freq_table=freq_table_match, freq_table2=freq_table_mismatch, table1=t1,
                                                expected_table1=expected_table, expected_table2=expected_table2,
                                                expected_depth=3)

    def test_characterize_alignment_protein_single_pos_partial2(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        t1 = freq_table.get_table()
        freq_table2 = freq_table.characterize_alignment_mm(num_aln=protein_num_aln, comparison=pro_single_to_pair,
                                                           mismatch_mask=pro_pair_mismatch, single_to_pair=None,
                                                           indexes1=np.array([0, 1], dtype=np.int32),
                                                           indexes2=np.array([2], dtype=np.int32))
        expected_table = np.zeros((6, pro_pair_alpha_size), dtype=np.int32)
        idx1 = [0, 1, 3, 4, 5]
        idx2 = [275, 575, 375, 100, 100]
        values = [2, 1, 1, 1, 1]
        expected_table[idx1, idx2] = values
        expected_table2 = np.zeros((6, pro_pair_alpha_size), dtype=np.int32)
        idx1 = [1, 2, 3, 4, 5]
        idx2 = [119, 413, 567, 556, 556]
        values = [1, 2, 1, 1, 1]
        expected_table2[idx1, idx2] = values
        self.evaluate_characterize_alignment_mm(freq_table=freq_table, freq_table2=freq_table2, table1=t1,
                                                expected_table1=expected_table, expected_table2=expected_table2,
                                                expected_depth=3)

    def test_characterize_alignment_protein_single_pos_composite2(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        t1 = freq_table.get_table()
        freq_table2 = freq_table.characterize_alignment_mm(num_aln=protein_num_aln[[0, 1], :],
                                                           comparison=pro_single_to_pair,
                                                           mismatch_mask=pro_pair_mismatch, single_to_pair=None)
        freq_table3 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        freq_table4 = freq_table3.characterize_alignment_mm(num_aln=protein_num_aln[[2], :],
                                                            comparison=pro_single_to_pair,
                                                            mismatch_mask=pro_pair_mismatch, single_to_pair=None)
        freq_table5 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        freq_table6 = freq_table5.characterize_alignment_mm(num_aln=protein_num_aln,
                                                            comparison=pro_single_to_pair,
                                                            mismatch_mask=pro_pair_mismatch, single_to_pair=None,
                                                            indexes1=np.array([0, 1], dtype=np.int32),
                                                            indexes2=np.array([2], dtype=np.int32))
        freq_table_match = freq_table + freq_table3 + freq_table5
        freq_table_match.set_depth(depth=3)
        freq_table_mismatch = freq_table2 + freq_table4 + freq_table6
        freq_table_mismatch.set_depth(depth=3)

        expected_table = np.zeros((6, pro_pair_alpha_size), dtype=np.int32)
        idx1 = [0, 1, 2, 3, 4, 5]
        idx2 = [275, 575, 425, 375, 100, 100]
        values = [3, 1, 1, 1, 1, 1]
        expected_table[idx1, idx2] = values
        expected_table2 = np.zeros((6, pro_pair_alpha_size), dtype=np.int32)
        idx1 = [1, 2, 3, 4, 5]
        idx2 = [119, 413, 567, 556, 556]
        values = [2, 2, 2, 2, 2]
        expected_table2[idx1, idx2] = values
        self.evaluate_characterize_alignment_mm(freq_table=freq_table_match, freq_table2=freq_table_mismatch, table1=t1,
                                                expected_table1=expected_table, expected_table2=expected_table2,
                                                expected_depth=3)

    def test_characterize_alignment_protein_pair_pos(self):
        freq_table = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
        t1 = freq_table.get_table()
        freq_table2 = freq_table.characterize_alignment_mm(num_aln=protein_num_aln, single_to_pair=pro_single_to_pair,
                                                           comparison=pro_pair_to_quad, mismatch_mask=pro_quad_mismatch)
        expected_table = np.zeros((freq_table.num_pos, pro_quad_alpha_size))
        idx1 = [0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19,
                19, 20, 20]
        idx2 = [158675, 165599, 162137, 160983, 154636, 154636, 58175, 331775, 65645, 327159, 69111, 320812, 69100,
                320812, 69100, 244925, 245225, 248391, 248380, 248380, 216375, 331575, 210028, 331564, 210028, 331564,
                57700, 331300, 57700, 331300, 57700, 331300]
        values = [3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        expected_table[idx1, idx2] = values
        expected_table2 = np.zeros((freq_table.num_pos, pro_quad_alpha_size))
        idx1 = [1, 2, 3, 4, 5, 7, 7, 12, 12, 13, 13, 14, 14]
        idx2 = [154655, 162125, 165591, 165580, 165580, 65657, 328301, 243783, 248679, 237436, 248668, 237436, 248668]
        values = [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        expected_table2[idx1, idx2] = values
        self.evaluate_characterize_alignment_mm(freq_table=freq_table, freq_table2=freq_table2, table1=t1,
                                                expected_table1=expected_table, expected_table2=expected_table2,
                                                expected_depth=3)

    def test_characterize_alignment_protein_pair_pos_partial1(self):
        freq_table = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
        t1 = freq_table.get_table()
        freq_table2 = freq_table.characterize_alignment_mm(num_aln=protein_num_aln, single_to_pair=pro_single_to_pair,
                                                           comparison=pro_pair_to_quad, mismatch_mask=pro_quad_mismatch,
                                                           indexes1=np.array([0], dtype=np.int32),
                                                           indexes2=np.array([1, 2], dtype=np.int32))
        expected_table = np.zeros((freq_table.num_pos, pro_quad_alpha_size))
        idx1 = [0, 2, 6, 7, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        idx2 = [158675, 162137, 58175, 65645, 69111, 69100, 69100, 244925, 245225, 248391, 248380, 248380, 331575,
                331564, 331564, 331300, 331300, 331300]
        values = [2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        expected_table[idx1, idx2] = values
        expected_table2 = np.zeros((freq_table.num_pos, pro_quad_alpha_size))
        idx1 = [1, 2, 3, 4, 5, 7, 12, 13, 14]
        idx2 = [154655, 162125, 165591, 165580, 165580, 65657, 248679, 248668, 248668]
        values = [2, 1, 2, 2, 2, 1, 1, 1, 1]
        expected_table2[idx1, idx2] = values
        self.evaluate_characterize_alignment_mm(freq_table=freq_table, freq_table2=freq_table2, table1=t1,
                                                expected_table1=expected_table, expected_table2=expected_table2,
                                                expected_depth=3)

    def test_characterize_alignment_protein_pair_pos_composite1(self):
        freq_table = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
        t1 = freq_table.get_table()
        freq_table2 = freq_table.characterize_alignment_mm(num_aln=protein_num_aln[[0], :],
                                                           single_to_pair=pro_single_to_pair,
                                                           comparison=pro_pair_to_quad, mismatch_mask=pro_quad_mismatch)
        freq_table3 = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
        freq_table4 = freq_table3.characterize_alignment_mm(num_aln=protein_num_aln[[1, 2], :],
                                                            single_to_pair=pro_single_to_pair,
                                                            comparison=pro_pair_to_quad,
                                                            mismatch_mask=pro_quad_mismatch)
        freq_table5 = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
        freq_table6 = freq_table5.characterize_alignment_mm(num_aln=protein_num_aln, single_to_pair=pro_single_to_pair,
                                                            comparison=pro_pair_to_quad,
                                                            mismatch_mask=pro_quad_mismatch,
                                                            indexes1=np.array([0], dtype=np.int32),
                                                            indexes2=np.array([1, 2], dtype=np.int32))
        match_freq_table = freq_table + freq_table3 + freq_table5
        match_freq_table.set_depth(depth=3)
        mismatch_freq_table = freq_table2 + freq_table4 + freq_table6
        mismatch_freq_table.set_depth(depth=3)
        expected_table = np.zeros((freq_table.num_pos, pro_quad_alpha_size))
        idx1 = [0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19,
                19, 20, 20]
        idx2 = [158675, 165599, 162137, 160983, 154636, 154636, 58175, 331775, 65645, 327159, 69111, 320812, 69100,
                320812, 69100, 244925, 245225, 248391, 248380, 248380, 216375, 331575, 210028, 331564, 210028, 331564,
                57700, 331300, 57700, 331300, 57700, 331300]
        values = [3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        expected_table[idx1, idx2] = values
        expected_table2 = np.zeros((freq_table.num_pos, pro_quad_alpha_size))
        idx1 = [1, 2, 3, 4, 5, 7, 7, 12, 12, 13, 13, 14, 14]
        idx2 = [154655, 162125, 165591, 165580, 165580, 65657, 328301, 243783, 248679, 237436, 248668, 237436, 248668]
        values = [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        expected_table2[idx1, idx2] = values
        self.evaluate_characterize_alignment_mm(freq_table=match_freq_table, freq_table2=mismatch_freq_table, table1=t1,
                                                expected_table1=expected_table, expected_table2=expected_table2,
                                                expected_depth=3)

    def test_characterize_alignment_protein_pair_pos_partial2(self):
        freq_table = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
        t1 = freq_table.get_table()
        freq_table2 = freq_table.characterize_alignment_mm(num_aln=protein_num_aln, single_to_pair=pro_single_to_pair,
                                                           comparison=pro_pair_to_quad, mismatch_mask=pro_quad_mismatch,
                                                           indexes1=np.array([0, 1], dtype=np.int32),
                                                           indexes2=np.array([2], dtype=np.int32))
        expected_table = np.zeros((freq_table.num_pos, pro_quad_alpha_size))
        idx1 = [0, 1, 3, 4, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20,
                20]
        idx2 = [158675, 165599, 160983, 154636, 154636, 58175, 331775, 65645, 327159, 69111, 320812, 69100, 320812,
                69100, 244925, 248391, 248380, 248380, 216375, 331575, 210028, 331564, 210028, 331564, 57700, 331300,
                57700, 331300, 57700, 331300]
        values = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        expected_table[idx1, idx2] = values
        expected_table2 = np.zeros((freq_table.num_pos, pro_quad_alpha_size))
        idx1 = [1, 2, 3, 4, 5, 7, 12, 13, 14]
        idx2 = [154655, 162125, 165591, 165580, 165580, 328301, 243783, 237436, 237436]
        values = [1, 2, 1, 1, 1, 1, 1, 1, 1]
        expected_table2[idx1, idx2] = values
        self.evaluate_characterize_alignment_mm(freq_table=freq_table, freq_table2=freq_table2, table1=t1,
                                                expected_table1=expected_table, expected_table2=expected_table2,
                                                expected_depth=3)

    def test_characterize_alignment_protein_pair_pos_composite2(self):
        freq_table = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
        t1 = freq_table.get_table()
        freq_table2 = freq_table.characterize_alignment_mm(num_aln=protein_num_aln[[0, 1], :],
                                                           single_to_pair=pro_single_to_pair,
                                                           comparison=pro_pair_to_quad, mismatch_mask=pro_quad_mismatch)
        freq_table3 = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
        freq_table4 = freq_table3.characterize_alignment_mm(num_aln=protein_num_aln[[2], :],
                                                            single_to_pair=pro_single_to_pair,
                                                            comparison=pro_pair_to_quad,
                                                            mismatch_mask=pro_quad_mismatch)
        freq_table5 = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
        freq_table6 = freq_table5.characterize_alignment_mm(num_aln=protein_num_aln, single_to_pair=pro_single_to_pair,
                                                            comparison=pro_pair_to_quad,
                                                            mismatch_mask=pro_quad_mismatch,
                                                            indexes1=np.array([0, 1], dtype=np.int32),
                                                            indexes2=np.array([2], dtype=np.int32))
        match_freq_table = freq_table + freq_table3 + freq_table5
        match_freq_table.set_depth(depth=3)
        mismatch_freq_table = freq_table2 + freq_table4 + freq_table6
        mismatch_freq_table.set_depth(depth=3)
        expected_table = np.zeros((freq_table.num_pos, pro_quad_alpha_size))
        idx1 = [0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19,
                19, 20, 20]
        idx2 = [158675, 165599, 162137, 160983, 154636, 154636, 58175, 331775, 65645, 327159, 69111, 320812, 69100,
                320812, 69100, 244925, 245225, 248391, 248380, 248380, 216375, 331575, 210028, 331564, 210028, 331564,
                57700, 331300, 57700, 331300, 57700, 331300]
        values = [3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        expected_table[idx1, idx2] = values
        expected_table2 = np.zeros((freq_table.num_pos, pro_quad_alpha_size))
        idx1 = [1, 2, 3, 4, 5, 7, 7, 12, 12, 13, 13, 14, 14]
        idx2 = [154655, 162125, 165591, 165580, 165580, 65657, 328301, 243783, 248679, 237436, 248668, 237436, 248668]
        values = [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        expected_table2[idx1, idx2] = values
        self.evaluate_characterize_alignment_mm(freq_table=match_freq_table, freq_table2=mismatch_freq_table, table1=t1,
                                                expected_table1=expected_table, expected_table2=expected_table2,
                                                expected_depth=3)

    def test_characterize_alignment_protein_single_pos_single_to_pair(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        t1 = freq_table.get_table()
        freq_table2 = freq_table.characterize_alignment_mm(num_aln=protein_num_aln, comparison=pro_single_to_pair,
                                                           mismatch_mask=pro_pair_mismatch,
                                                           single_to_pair=pro_single_to_pair)
        expected_table = np.zeros((6, pro_pair_alpha_size), dtype=np.int32)
        idx1 = [0, 1, 2, 3, 4, 5]
        idx2 = [275, 575, 425, 375, 100, 100]
        values = [3, 1, 1, 1, 1, 1]
        expected_table[idx1, idx2] = values
        expected_table2 = np.zeros((6, pro_pair_alpha_size), dtype=np.int32)
        idx1 = [1, 2, 3, 4, 5]
        idx2 = [119, 413, 567, 556, 556]
        values = [2, 2, 2, 2, 2]
        expected_table2[idx1, idx2] = values
        self.evaluate_characterize_alignment_mm(freq_table=freq_table, freq_table2=freq_table2, table1=t1,
                                                expected_table1=expected_table, expected_table2=expected_table2,
                                                expected_depth=3)

    def test_characterize_alignment_failure_protein_pair_pos_no_single_to_pair(self):
        freq_table = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment_mm(num_aln=protein_num_aln, single_to_pair=None,
                                                 comparison=pro_pair_to_quad, mismatch_mask=pro_quad_mismatch)

    def test_characterize_alignment_failure_protein_single_pos_no_comparison(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment_mm(num_aln=protein_num_aln, single_to_pair=None,
                                                 comparison=None, mismatch_mask=pro_pair_mismatch)

    def test_characterize_alignment_failure_protein_pair_pos_no_comparison(self):
        freq_table = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment_mm(num_aln=protein_num_aln, single_to_pair=pro_single_to_pair,
                                                 comparison=None, mismatch_mask=pro_quad_mismatch)

    def test_characterize_alignment_failure_protein_single_pos_no_mismatch_mask(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment_mm(num_aln=protein_num_aln, single_to_pair=None,
                                                 comparison=pro_single_to_pair, mismatch_mask=None)

    def test_characterize_alignment_failure_protein_pair_pos_no_mismatch_mask(self):
        freq_table = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment_mm(num_aln=protein_num_aln, single_to_pair=pro_single_to_pair,
                                                 comparison=pro_pair_to_quad, mismatch_mask=None)

    def test_characterize_alignment_failure_protein_single_pos_no_aln(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment_mm(num_aln=None, single_to_pair=None,
                                                 comparison=pro_single_to_pair, mismatch_mask=pro_pair_mismatch)

    def test_characterize_alignment_failure_protein_pair_pos_no_aln(self):
        freq_table = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment_mm(num_aln=None, single_to_pair=pro_single_to_pair,
                                                 comparison=pro_pair_to_quad, mismatch_mask=pro_quad_mismatch)

    def test_characterize_alignment_protein_single_pos_no_indexes1(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment_mm(num_aln=protein_num_aln, comparison=pro_single_to_pair,
                                                 mismatch_mask=pro_pair_mismatch, single_to_pair=None,
                                                 indexes1=None, indexes2=np.array([1, 2], dtype=np.int32))

    def test_characterize_alignment_protein_single_pos_no_indexes2(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment_mm(num_aln=protein_num_aln, comparison=pro_single_to_pair,
                                                 mismatch_mask=pro_pair_mismatch, single_to_pair=None,
                                                 indexes1=np.array([0], dtype=np.int32), indexes2=None)

    def test_characterize_alignment_protein_single_pos_bad_index_order(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment_mm(num_aln=protein_num_aln, comparison=pro_single_to_pair,
                                                 mismatch_mask=pro_pair_mismatch, single_to_pair=None,
                                                 indexes1=np.array([1, 2], dtype=np.int32),
                                                 indexes2=np.array([0], dtype=np.int32))


class TestFrequencyTableCharacterizeSequence(TestCase):

    def evaluate_characterize_sequence(self, seqs, seq_aln, seq_type, alpha_size, alpha_map, alpha_rev, seq_len,
                                       pos_size, single_map, expected_depth=1, single_to_pair=None):
        freq_table = FrequencyTable(alpha_size, alpha_map, alpha_rev, seq_len, pos_size)
        t1 = freq_table.get_table()
        for seq in seqs:
            freq_table.characterize_sequence(seq=seq)
            t2 = freq_table.get_table()
            self.assertNotEqual(t1, t2)
            self.assertTrue(isinstance(t2, dict))
            t1 = t2
        freq_table.finalize_table()
        t3 = freq_table.get_table()
        self.assertTrue(isinstance(t3, csc_matrix))
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=seq_aln)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type=seq_type)
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=single_map)
        freq_table2 = FrequencyTable(alpha_size, alpha_map, alpha_rev, seq_len, pos_size)
        freq_table2.characterize_alignment(num_aln=num_aln, single_to_pair=single_to_pair)
        expected_table = freq_table2.get_table()
        self.assertFalse((t3 - expected_table).toarray().any())
        self.assertEqual(freq_table.get_depth(), expected_depth)

    def test_characterize_sequence_dna_single_pos(self):
        self.evaluate_characterize_sequence(seqs=[dna_seq1], seq_aln=dna_one_seq_aln_str, seq_type='DNA',
                                            alpha_size=dna_alpha_size, alpha_map=dna_map, alpha_rev=dna_rev, seq_len=18,
                                            pos_size=1, single_map=dna_map)

    def test_characterize_sequence_protein_single_pos(self):
        self.evaluate_characterize_sequence(seqs=[protein_seq1], seq_aln=protein_one_seq_aln_str, seq_type='Protein',
                                            alpha_size=protein_alpha_size, alpha_map=protein_map, alpha_rev=protein_rev,
                                            seq_len=6, pos_size=1, single_map=protein_map)

    def test_characterize_sequence_dna_single_pair(self):
        self.evaluate_characterize_sequence(seqs=[dna_seq1], seq_aln=dna_one_seq_aln_str, seq_type='DNA',
                                            alpha_size=dna_pair_alpha_size, alpha_map=dna_pair_map,
                                            alpha_rev=dna_pair_rev, seq_len=18, pos_size=2, single_map=dna_map,
                                            single_to_pair=dna_single_to_pair)

    def test_characterize_sequence_protein_single_pair(self):
        self.evaluate_characterize_sequence(seqs=[protein_seq1], seq_aln=protein_one_seq_aln_str, seq_type='Protein',
                                            alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                            alpha_rev=pro_pair_rev, seq_len=6, pos_size=2, single_map=protein_map,
                                            single_to_pair=pro_single_to_pair)

    def test_characterize_multi_sequence_dna_single_pos(self):
        self.evaluate_characterize_sequence(seqs=[dna_seq1, dna_seq2, dna_seq3], seq_aln=dna_aln_str,
                                            seq_type='DNA', alpha_size=dna_alpha_size, alpha_map=dna_map,
                                            alpha_rev=dna_rev, seq_len=18, pos_size=1, single_map=dna_map,
                                            expected_depth=3)

    def test_characterize_multi_sequence_protein_single_pos(self):
        self.evaluate_characterize_sequence(seqs=[protein_seq1, protein_seq2, protein_seq3], seq_aln=protein_aln_str,
                                            seq_type='Protein', alpha_size=protein_alpha_size, alpha_map=protein_map,
                                            alpha_rev=protein_rev, seq_len=6, pos_size=1, single_map=protein_map,
                                            expected_depth=3)

    def test_characterize_multi_sequence_dna_pair_pos(self):
        self.evaluate_characterize_sequence(seqs=[dna_seq1, dna_seq2, dna_seq3], seq_aln=dna_aln_str, seq_type='DNA',
                                            alpha_size=dna_pair_alpha_size, alpha_map=dna_pair_map,
                                            alpha_rev=dna_pair_rev, seq_len=18, pos_size=2, single_map=dna_map,
                                            single_to_pair=dna_single_to_pair, expected_depth=3)

    def test_characterize_multi_sequence_protein_pair_pos(self):
        self.evaluate_characterize_sequence(seqs=[protein_seq1, protein_seq2, protein_seq3], seq_aln=protein_aln_str,
                                            seq_type='Protein', alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                            alpha_rev=pro_pair_rev, seq_len=6, pos_size=2, single_map=protein_map,
                                            single_to_pair=pro_single_to_pair, expected_depth=3)


    # def test_characterize_sequence_fail_dna_single_pos_wrong_mapping(self):
    #     # No error is raised here because the pair DNA alphabet is a subset of the Protein pair alphabet
    #     freq_table = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 18, 1)
    #     with self.assertRaises(KeyError):
    #         freq_table.characterize_sequence(seq=dna_seq1)

    def test_characterize_sequence_fail_protein_single_pos_wrong_mapping(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 6, 1)
        with self.assertRaises(KeyError):
            freq_table.characterize_sequence(seq=protein_seq1)

    # def test_characterize_sequence_fail_dna_pair_pos_wrong_mapping(self):
    #     # No error is raised here because the pair DNA alphabet is a subset of the Protein pair alphabet
    #     freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 18, 2)
    #     with self.assertRaises(KeyError):
    #         freq_table.characterize_sequence(seq=dna_seq1)

    def test_characterize_sequence_fail_protein_pair_pos_wrong_mapping(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 6, 2)
        with self.assertRaises(KeyError):
            freq_table.characterize_sequence(seq=protein_seq1)

    def test_characterize_sequence_fail_no_sequence(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        with self.assertRaises(ValueError):
            freq_table.characterize_sequence(seq=None)


class TestFrequencyTableGetters(TestCase):

    def evaluate_getters_initial(self, freq_table, expected_dict, expected_depth, expected_positions, test_char):
        table = freq_table.get_table()
        self.assertTrue(isinstance(table, dict))
        self.assertEqual(table, expected_dict)
        self.assertEqual(freq_table.get_depth(), expected_depth)
        positions = freq_table.get_positions()
        self.assertEqual(set(positions), set(expected_positions))
        for p in positions:
            with self.assertRaises(AttributeError):
                freq_table.get_chars(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_count(pos=p, char=test_char)
            with self.assertRaises(AttributeError):
                freq_table.get_count_array(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_frequency(pos=p, char=test_char)
            with self.assertRaises(AttributeError):
                freq_table.get_frequency_array(pos=p)
        with self.assertRaises(AttributeError):
            freq_table.get_count_matrix()
        with self.assertRaises(AttributeError):
            freq_table.get_frequency_matrix()

    def evaluate_getters_finalized_0_depth(self, freq_table, positions, expected_table, test_char):
        table = freq_table.get_table()
        self.assertFalse((table - expected_table).any())
        for p in positions:
            with self.assertRaises(AttributeError):
                freq_table.get_chars(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_count(pos=p, char=test_char)
            with self.assertRaises(AttributeError):
                freq_table.get_count_array(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_frequency(pos=p, char=test_char)
            with self.assertRaises(AttributeError):
                freq_table.get_frequency_array(pos=p)
        with self.assertRaises(AttributeError):
            freq_table.get_count_matrix()
        with self.assertRaises(AttributeError):
            freq_table.get_frequency_matrix()

    # def evaluate_getters_finalized(self, freq_table, positions, aln_str, query, query_type, alpha_size, alpha_map,
    #                                alpha_rev, seq_len, pos_size, expected_chars, expected_counts):
    #     table2 = freq_table.get_table()
    #     self.assertTrue(isinstance(table2, csc_matrix))
    #     aln_fn = write_out_temp_fn(suffix='fasta', out_str=aln_str)
    #     aln = SeqAlignment(aln_fn, query, polymer_type=query_type)
    #     aln.import_alignment()
    #     num_aln = aln._alignment_to_num(mapping=alpha_map)
    #     os.remove(aln_fn)
    #     freq_table2 = FrequencyTable(alpha_size, alpha_map, alpha_rev,seq_len, pos_size)
    #     freq_table2.characterize_alignment(num_aln=num_aln)
    #     expected_table = freq_table2.get_table()
    #     self.assertFalse((table2 - expected_table).toarray().any())
    #     for p in positions:
    #         chars = []
    #         for seq in aln:
    #             if pos_size == 1:
    #                 chars.append(seq[p])
    #             elif pos_size == 2:
    #                 chars.append(seq[p[0]] + seq[p[1]])
    #             else:
    #                 raise ValueError('Bad pos_size encountered.')
    #         self.assertEqual(set(freq_table.get_chars(pos=p)), set(chars))
    #         self.assertFalse((freq_table.get_count_array(pos=p) - np.array([1])).any())
    #         self.assertFalse((freq_table.get_frequency_array(pos=p) - np.array([1.0])).any())
    #         for c in dna_rev:
    #             if dna_seq1[p] == c:
    #                 self.assertEqual(freq_table.get_count(pos=p, char=c), 1)
    #                 self.assertEqual(freq_table.get_frequency(pos=p, char=c), 1.0)
    #             else:
    #                 self.assertEqual(freq_table.get_count(pos=p, char=c), 0)
    #                 self.assertEqual(freq_table.get_frequency(pos=p, char=c), 0.0)
    #     self.assertFalse((freq_table.get_count_matrix() - expected_table).any())
    #     self.assertFalse((freq_table.get_frequency_matrix() - expected_table).any())

    def test_get_initial(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        expected_dict = {'values': [], 'i': [], 'j': [], 'shape': (freq_table.num_pos, dna_alpha_size)}
        expected_positions = list(range(18))
        self.evaluate_getters_initial(freq_table=freq_table, expected_dict=expected_dict, expected_depth=0,
                                      expected_positions=expected_positions, test_char='A')
        freq_table.finalize_table()
        expected_table = np.zeros((freq_table.num_pos, dna_alpha_size))
        self.evaluate_getters_finalized_0_depth(freq_table=freq_table, positions=expected_positions,
                                                expected_table=expected_table, test_char='A')

    def test_get_single_update_single_pos(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table._increment_count(pos=0, char='A', amount=1)
        expected_dict = {'values': [1], 'i': [0], 'j': [0], 'shape': (freq_table.num_pos, dna_alpha_size)}
        expected_positions = list(range(18))
        self.evaluate_getters_initial(freq_table=freq_table, expected_dict=expected_dict, expected_depth=0,
                                      expected_positions=expected_positions, test_char='A')
        freq_table.finalize_table()
        expected_table = np.zeros((freq_table.num_pos, dna_alpha_size))
        expected_table[0, 0] = 1
        self.evaluate_getters_finalized_0_depth(freq_table=freq_table, positions=expected_positions,
                                                expected_table=expected_table, test_char='A')

    def test_get_single_update_pair_pos(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table._increment_count(pos=(0, 0), char='AA', amount=1)
        expected_dict = {'values': [1], 'i': [0], 'j': [0], 'shape': (freq_table.num_pos, dna_pair_alpha_size)}
        expected_positions = list(combinations(range(18), 2)) + [(x, x) for x in range(18)]
        self.evaluate_getters_initial(freq_table=freq_table, expected_dict=expected_dict, expected_depth=0,
                                      expected_positions=expected_positions, test_char='A')
        freq_table.finalize_table()
        expected_table = np.zeros((freq_table.num_pos, dna_pair_alpha_size))
        expected_table[0, 0] = 1
        self.evaluate_getters_finalized_0_depth(freq_table=freq_table, positions=expected_positions,
                                                expected_table=expected_table, test_char='A')

    def test_get_single_seq_single_pos(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        expected_positions = list(range(18))
        expected_dict = {'values': [1] * 18, 'i': list(range(18)), 'j': [0, 1, 3, 3, 0, 3, 0, 2, 1] + [4] * 9,
                         'shape': (freq_table.num_pos, dna_alpha_size)}
        self.evaluate_getters_initial(freq_table=freq_table, expected_dict=expected_dict, expected_depth=1,
                                      expected_positions=expected_positions, test_char='A')
        freq_table.finalize_table()
        table2 = freq_table.get_table()
        self.assertTrue(isinstance(table2, csc_matrix))
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=dna_one_seq_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table2 = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table2.characterize_alignment(num_aln=num_aln)
        expected_table = freq_table2.get_table()
        self.assertFalse((table2 - expected_table).toarray().any())
        for p in expected_positions:
            self.assertEqual(freq_table.get_chars(pos=p), [dna_seq1[p]])
            self.assertFalse((freq_table.get_count_array(pos=p) - np.array([1])).any())
            self.assertFalse((freq_table.get_frequency_array(pos=p) - np.array([1.0])).any())
            for c in dna_rev:
                if dna_seq1[p] == c:
                    self.assertEqual(freq_table.get_count(pos=p, char=c), 1)
                    self.assertEqual(freq_table.get_frequency(pos=p, char=c), 1.0)
                else:
                    self.assertEqual(freq_table.get_count(pos=p, char=c), 0)
                    self.assertEqual(freq_table.get_frequency(pos=p, char=c), 0.0)
        self.assertFalse((freq_table.get_count_matrix() - expected_table).any())
        self.assertFalse((freq_table.get_frequency_matrix() - expected_table).any())

    def test_get_single_seq_pair_pos(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table.characterize_sequence(seq=dna_seq1)
        expected_positions = [(x, y) for x in range(18) for y in range(x, 18)]
        expected_dict = {'values': [1] * freq_table.num_pos, 'i': list(range(len(expected_positions))),
                         'j': ([0, 1, 3, 3, 0, 3, 0, 2, 1] + [4] * 9 + [6, 8, 8, 5, 8, 5, 7, 6] + [9] * 9 +
                               [18, 18, 15, 18, 15, 17, 16] + [19] * 9 + [18, 15, 18, 15, 17, 16] + [19] * 9 +
                               [0, 3, 0, 2, 1] + [4] * 9 + [18, 15, 17, 16] + [19] * 9 + [0, 2, 1] + [4] * 9 +
                               [12, 11] + [14] * 9 + [6] + [9] * 9 + [24] * np.sum(list(range(10)))),
                         'shape': (freq_table.num_pos, dna_pair_alpha_size)}
        self.evaluate_getters_initial(freq_table=freq_table, expected_dict=expected_dict, expected_depth=1,
                                      expected_positions=expected_positions, test_char='AA')
        freq_table.finalize_table()
        table2 = freq_table.get_table()
        self.assertTrue(isinstance(table2, csc_matrix))
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=dna_one_seq_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table2 = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table2.characterize_alignment(num_aln=num_aln, single_to_pair=dna_single_to_pair)
        expected_table = freq_table2.get_table()
        self.assertFalse((table2 - expected_table).toarray().any())
        for p in expected_positions:
            curr_char = dna_seq1[p[0]] + dna_seq1[p[1]]
            self.assertEqual(freq_table.get_chars(pos=p), [curr_char])
            self.assertFalse((freq_table.get_count_array(pos=p) - np.array([1])).any())
            self.assertFalse((freq_table.get_frequency_array(pos=p) - np.array([1.0])).any())
            for c in dna_pair_rev:
                if curr_char == c:
                    self.assertEqual(freq_table.get_count(pos=p, char=c), 1)
                    self.assertEqual(freq_table.get_frequency(pos=p, char=c), 1.0)
                else:
                    self.assertEqual(freq_table.get_count(pos=p, char=c), 0)
                    self.assertEqual(freq_table.get_frequency(pos=p, char=c), 0.0)
        self.assertFalse((freq_table.get_count_matrix() - expected_table).any())
        self.assertFalse((freq_table.get_frequency_matrix() - expected_table).any())

    def test_get_aln_update_single_pos(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=dna_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table.characterize_alignment(num_aln=num_aln)
        table = freq_table.get_table()
        self.assertTrue(isinstance(table, csc_matrix))
        self.assertEqual(freq_table.get_depth(), 3)
        # This table has already been explicitly tested in the characterize_alignment test class.
        positions = freq_table.get_positions()
        expected_positions = list(range(18))
        self.assertEqual(positions, expected_positions)
        for p in positions:
            self.assertEqual(set(freq_table.get_chars(pos=p)), {dna_seq1[p], dna_seq2[p], dna_seq3[p]})
            curr_array = table[p, :].toarray()
            curr_array = curr_array[np.nonzero(curr_array)]
            self.assertFalse((freq_table.get_count_array(pos=p) - curr_array).any())
            self.assertFalse((freq_table.get_frequency_array(pos=p) - (curr_array / 3.0)).any())
            for c in dna_rev:
                self.assertEqual(freq_table.get_count(pos=p, char=c), table[p, dna_map[c]])
                self.assertEqual(freq_table.get_frequency(pos=p, char=c), table[p, dna_map[c]] / 3.0)
        self.assertFalse((freq_table.get_count_matrix() - table).any())
        self.assertFalse((freq_table.get_frequency_matrix() - (table / 3.0)).any())

    def test_get_aln_update_pair_pos(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=dna_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=dna_single_to_pair)
        table = freq_table.get_table()
        self.assertTrue(isinstance(table, csc_matrix))
        self.assertEqual(freq_table.get_depth(), 3)
        # This table has already been explicitly tested in the characterize_alignment test class.
        positions = freq_table.get_positions()
        expected_positions = list(combinations(range(18), 2)) + [(x, x) for x in range(18)]
        self.assertEqual(set(positions), set(expected_positions))
        for p in positions:
            self.assertEqual(set(freq_table.get_chars(pos=p)), {dna_seq1[p[0]] + dna_seq1[p[1]],
                                                                dna_seq2[p[0]] + dna_seq2[p[1]],
                                                                dna_seq3[p[0]] + dna_seq3[p[1]]})
            curr_array = table[freq_table._convert_pos(p), :].toarray()
            curr_array = curr_array[np.nonzero(curr_array)]
            self.assertFalse((freq_table.get_count_array(pos=p) - curr_array).any())
            self.assertFalse((freq_table.get_frequency_array(pos=p) - (curr_array / 3.0)).any())
            for c in dna_pair_rev:
                self.assertEqual(freq_table.get_count(pos=p, char=c),
                                 table[freq_table._convert_pos(p), dna_pair_map[c]])
                self.assertEqual(freq_table.get_frequency(pos=p, char=c),
                                 table[freq_table._convert_pos(p), dna_pair_map[c]] / 3.0)
        self.assertFalse((freq_table.get_count_matrix() - table).any())
        self.assertFalse((freq_table.get_frequency_matrix() - (table / 3.0)).any())


class TestFrequencyTableCSV(TestCase):

    def evaluate_to_csv(self, freq_table, csv_path, alpha_map):
        freq_table.to_csv(file_path=csv_path)
        self.assertTrue(os.path.isfile(csv_path))
        df = pd.read_csv(csv_path, sep='\t', header=0, index_col=None)
        self.assertEqual(set(df.columns), {'Position', 'Variability', 'Characters', 'Counts', 'Frequencies'})
        expected_positions = freq_table.get_positions()
        table = freq_table.get_table()
        for i in df.index:
            if freq_table.position_size == 1:
                expected_position = expected_positions[i]
            elif freq_table.position_size == 2:
                expected_position = str((expected_positions[i][0], expected_positions[i][1]))
            else:
                raise ValueError('Bas position_size')
            self.assertEqual(df.loc[i, 'Position'], expected_position)
            chars = df.loc[i, 'Characters'].split(',')
            self.assertEqual(df.loc[i, 'Variability'], len(chars))
            expected_chars = freq_table.get_chars(pos=expected_positions[i])
            self.assertEqual(set(chars), set(expected_chars))
            try:
                counts = df.loc[i, 'Counts'].split(',')
            except AttributeError:
                counts = [str(df.loc[i, 'Counts'])]
            expected_counts = [str(table[i, alpha_map[c]]) for c in chars]
            self.assertEqual(counts, expected_counts)
            try:
                freqs = df.loc[i, 'Frequencies'].split(',')
            except AttributeError:
                freqs = [str(df.loc[i, 'Frequencies'])]
            expected_freqs = [str(int(x) / float(freq_table.get_depth())) for x in counts]
            self.assertEqual(freqs, expected_freqs)
        os.remove(csv_path)

    def test_to_csv_single_pos_single_update(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        self.evaluate_to_csv(freq_table=freq_table, csv_path=csv_path, alpha_map=dna_map)

    def test_to_csv_single_pos_multi_update(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=dna_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table.characterize_alignment(num_aln=num_aln)
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        self.evaluate_to_csv(freq_table=freq_table, csv_path=csv_path, alpha_map=dna_map)

    def test_to_csv_pair_pos_single_update(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        self.evaluate_to_csv(freq_table=freq_table, csv_path=csv_path, alpha_map=dna_pair_map)

    def test_to_csv_pair_pos_multi_update(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=dna_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=dna_single_to_pair)
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        self.evaluate_to_csv(freq_table=freq_table, csv_path=csv_path, alpha_map=dna_pair_map)

    def test_to_csv_failure_no_updates(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        with self.assertRaises(AttributeError):
            freq_table.to_csv(file_path=os.path.join(os.getcwd(), 'Test_Freq_Table.csv'))

    def test_to_csv_failure_not_finalized(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        with self.assertRaises(AttributeError):
            freq_table.to_csv(file_path=os.path.join(os.getcwd(), 'Test_Freq_Table.csv'))

    def test_to_csv_failure_no_path(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=dna_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table.characterize_alignment(num_aln=num_aln)
        with self.assertRaises(TypeError):
            freq_table.to_csv(file_path=None)

    def evaluate_load_csv(self, freq_table, alpha_size, alpha_map, alpha_rev, seq_len, pos_size):
        table = freq_table.get_table()
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        freq_table.to_csv(file_path=csv_path)
        freq_table2 = FrequencyTable(alpha_size, alpha_map, alpha_rev, seq_len, pos_size)
        freq_table2.load_csv(file_path=csv_path)
        table2 = freq_table2.get_table()
        self.assertIsInstance(table, csc_matrix)
        self.assertIsInstance(table2, csc_matrix)
        self.assertFalse((table - table2).toarray().any())
        self.assertEqual(freq_table.get_depth(), freq_table2.get_depth())
        os.remove(csv_path)

    def test_load_csv_single_pos_single_update(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        self.evaluate_load_csv(freq_table=freq_table, alpha_size=dna_alpha_size, alpha_map=dna_map, alpha_rev=dna_rev,
                               seq_len=18, pos_size=1)

    def test_load_csv_single_pos_multi_update(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=dna_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table.characterize_alignment(num_aln=num_aln)
        self.evaluate_load_csv(freq_table=freq_table, alpha_size=dna_alpha_size, alpha_map=dna_map, alpha_rev=dna_rev,
                               seq_len=18, pos_size=1)

    def test_load_csv_pair_pos_single_update(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        freq_table.to_csv(file_path=csv_path)
        self.evaluate_load_csv(freq_table=freq_table, alpha_size=dna_pair_alpha_size, alpha_map=dna_pair_map,
                               alpha_rev=dna_pair_rev, seq_len=18, pos_size=2)

    def test_load_csv_pair_pos_multi_update(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=dna_aln_str)
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=dna_single_to_pair)
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        freq_table.to_csv(file_path=csv_path)

        self.evaluate_load_csv(freq_table=freq_table, alpha_size=dna_pair_alpha_size, alpha_map=dna_pair_map,
                               alpha_rev=dna_pair_rev, seq_len=18, pos_size=2)

    def test_load_csv_failure_mismatch_single_pair(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        freq_table.to_csv(file_path=csv_path)
        freq_table2 = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        with self.assertRaises(KeyError):
            freq_table2.load_csv(file_path=csv_path)
        os.remove(csv_path)

    def test_load_csv_failure_mismatch_pair_single(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        freq_table.to_csv(file_path=csv_path)
        freq_table2 = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        with self.assertRaises(TypeError):
            freq_table2.load_csv(file_path=csv_path)
        os.remove(csv_path)

    def test_load_csv_failure_updated(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        with self.assertRaises(AttributeError):
            freq_table.to_csv(file_path=os.path.join(os.getcwd(), 'Test_Freq_Table.csv'))

    def test_load_csv_failure_finalized(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.finalize_table()
        with self.assertRaises(AttributeError):
            freq_table.load_csv(file_path=os.path.join(os.getcwd(), 'Test_Freq_Table.csv'))

    def test_load_csv_failure_no_path(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        with self.assertRaises(TypeError):
            freq_table.load_csv(file_path=None)

    def test_load_csv_failure_match_mismatch_freq_table(self):
        protein_mm_table = MatchMismatchTable(seq_len=6, num_aln=protein_num_aln,
                                              single_alphabet_size=protein_alpha_size, single_mapping=protein_map,
                                              single_reverse_mapping=protein_rev,
                                              larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                              larger_reverse_mapping=pro_pair_rev,
                                              single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        protein_mm_table.identify_matches_mismatches()
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        for s1 in range(3):
            for s2 in range(s1 + 1, 3):
                for pos1 in range(6):
                    single_stat, single_char = protein_mm_table.get_status_and_character(pos=pos1, seq_ind1=s1,
                                                                                         seq_ind2=s2)
                    if single_stat == 'match':
                        freq_table._increment_count(pos=pos1, char=single_char)
        freq_table.set_depth(3)
        freq_table.finalize_table()
        table = freq_table.get_table()
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        freq_table.to_csv(file_path=csv_path)
        freq_table2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        freq_table2.load_csv(csv_path, 3)
        table2 = freq_table2.get_table()
        self.assertIsInstance(table, csc_matrix)
        self.assertIsInstance(table2, csc_matrix)
        self.assertFalse((table - table2).toarray().any())
        self.assertEqual(freq_table.get_depth(), freq_table2.get_depth())
        os.remove(csv_path)

    def test_load_csv_failure_match_mismatch_freq_table_no_depth(self):
        protein_mm_table = MatchMismatchTable(seq_len=6, num_aln=protein_num_aln,
                                              single_alphabet_size=protein_alpha_size, single_mapping=protein_map,
                                              single_reverse_mapping=protein_rev,
                                              larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                              larger_reverse_mapping=pro_pair_rev,
                                              single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        protein_mm_table.identify_matches_mismatches()
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        for s1 in range(3):
            for s2 in range(s1 + 1, 3):
                for pos1 in range(6):
                    single_stat, single_char = protein_mm_table.get_status_and_character(pos=pos1, seq_ind1=s1,
                                                                                         seq_ind2=s2)
                    if single_stat == 'match':
                        freq_table._increment_count(pos=pos1, char=single_char)
        freq_table.set_depth(3)
        freq_table.finalize_table()
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        freq_table.to_csv(file_path=csv_path)
        freq_table2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        with self.assertRaises(RuntimeError):
            freq_table2.load_csv(csv_path)
        os.remove(csv_path)


class TestFrequencyTableAdd(TestCase):

    def test_add_single_pos(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        table = freq_table.get_table()
        freq_table2 = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table2.characterize_sequence(seq=dna_seq1)
        freq_table2.finalize_table()
        table2 = freq_table2.get_table()
        freq_table3 = freq_table + freq_table2
        table3 = freq_table3.get_table()
        self.assertEqual(freq_table3.get_depth(), 2)
        self.assertIsInstance(table3, csc_matrix)
        expected_table = table + table2
        self.assertFalse((table3 - expected_table).toarray().any())

    def test_add_pair_pos(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        table = freq_table.get_table()
        freq_table2 = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table2.characterize_sequence(seq=dna_seq1)
        freq_table2.finalize_table()
        table2 = freq_table2.get_table()
        freq_table3 = freq_table + freq_table2
        table3 = freq_table3.get_table()
        self.assertEqual(freq_table3.get_depth(), 2)
        self.assertIsInstance(table3, csc_matrix)
        expected_table = table + table2
        self.assertFalse((table3 - expected_table).toarray().any())

    def test_add_failure_mismatch_num_pos(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        freq_table2 = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 9, 1)
        freq_table2.characterize_sequence(seq=dna_seq1)
        freq_table2.finalize_table()
        with self.assertRaises(ValueError):
            freq_table3 = freq_table + freq_table2

    def test_add_failure_mismatch_mapping(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        freq_table2 = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table2.characterize_sequence(seq=dna_seq1)
        freq_table2.mapping = dna_pair_map
        freq_table2.finalize_table()
        with self.assertRaises(ValueError):
            freq_table3 = freq_table + freq_table2

    def test_add_failure_mismatch_reverse_mapping(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        freq_table2 = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table2.characterize_sequence(seq=dna_seq1)
        freq_table2.mapping = dna_pair_rev
        freq_table2.finalize_table()
        with self.assertRaises(ValueError):
            freq_table3 = freq_table + freq_table2

    def test_add_failure_mismatch_single_pair(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        freq_table2 = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table2.characterize_sequence(seq=dna_seq1)
        freq_table2.finalize_table()
        with self.assertRaises(ValueError):
            freq_table3 = freq_table + freq_table2

    def test_add_failure_mismatch_pair_single(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        freq_table2 = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table2.characterize_sequence(seq=dna_seq1)
        freq_table2.finalize_table()
        with self.assertRaises(ValueError):
            freq_table3 = freq_table + freq_table2

    def test_add_failure_not_finalized(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        freq_table2 = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table2.characterize_sequence(seq=dna_seq1)
        with self.assertRaises(AttributeError):
            freq_table3 = freq_table + freq_table2

    def test_add_failure_not_updated(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        freq_table2 = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table2.finalize_table()
        with self.assertRaises(AttributeError):
            freq_table3 = freq_table + freq_table2


if __name__ == '__main__':
    unittest.main()
