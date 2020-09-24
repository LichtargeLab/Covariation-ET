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
from scipy.sparse import lil_matrix, csc_matrix
from Bio.Alphabet import Gapped
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from test_Base import TestBase
from test_seqAlignment import generate_temp_fn, write_out_temp_fasta
from utils import build_mapping
from SeqAlignment import SeqAlignment
from FrequencyTable import FrequencyTable
from EvolutionaryTraceAlphabet import FullIUPACDNA, FullIUPACProtein, MultiPositionAlphabet

dna_alpha = Gapped(FullIUPACDNA())
dna_alpha_size, _, dna_map, dna_rev = build_mapping(dna_alpha)
protein_alpha = Gapped(FullIUPACProtein())
protein_alpha_size, _, protein_map, protein_rev = build_mapping(protein_alpha)
pair_dna_alpha = MultiPositionAlphabet(dna_alpha, size=2)
dna_pair_alpha_size, _, dna_pair_map, dna_pair_rev = build_mapping(pair_dna_alpha)
dna_single_to_pair = np.zeros((max(dna_map.values()) + 1, max(dna_map.values()) + 1))
for char in dna_pair_map:
    dna_single_to_pair[dna_map[char[0]], dna_map[char[1]]] = dna_pair_map[char]
pair_protein_alpha = MultiPositionAlphabet(protein_alpha, size=2)
pro_pair_alpha_size, _, pro_pair_map, pro_pair_rev = build_mapping(pair_protein_alpha)
pro_single_to_pair = np.zeros((max(protein_map.values()) + 1, max(protein_map.values()) + 1))
for char in pro_pair_map:
    pro_single_to_pair[protein_map[char[0]], protein_map[char[1]]] = pro_pair_map[char]
protein_seq1 = SeqRecord(id='seq1', seq=Seq('MET---', alphabet=FullIUPACProtein()))
protein_seq2 = SeqRecord(id='seq2', seq=Seq('M-TREE', alphabet=FullIUPACProtein()))
protein_seq3 = SeqRecord(id='seq3', seq=Seq('M-FREE', alphabet=FullIUPACProtein()))
protein_msa = MultipleSeqAlignment(records=[protein_seq1, protein_seq2, protein_seq3], alphabet=FullIUPACProtein())
dna_seq1 = SeqRecord(id='seq1', seq=Seq('ATGGAGACT---------', alphabet=FullIUPACDNA()))
dna_seq2 = SeqRecord(id='seq2', seq=Seq('ATG---ACTAGAGAGGAG', alphabet=FullIUPACDNA()))
dna_seq3 = SeqRecord(id='seq3', seq=Seq('ATG---TTTAGAGAGGAG', alphabet=FullIUPACDNA()))
dna_msa = MultipleSeqAlignment(records=[dna_seq1, dna_seq2, dna_seq3], alphabet=FullIUPACDNA())


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

    def test__convert_pos_single_pos_first(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        pos = freq_table._convert_pos(0)
        self.assertEqual(pos, 0)

    def test__convert_pos_single_pos_middle(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        pos = freq_table._convert_pos(9)
        self.assertEqual(pos, 9)

    def test__convert_pos_single_pos_last(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        pos = freq_table._convert_pos(17)
        self.assertEqual(pos, 17)

    def test__convert_pos_pair_pos_first(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        pos = freq_table._convert_pos((0, 0))
        self.assertEqual(pos, 0)

    def test__convert_pos_pair_pos_middle(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        pos = freq_table._convert_pos((1, 2))
        self.assertEqual(pos, 18 + 2 - 1)

    def test__convert_pos_pair_pos_last(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        pos = freq_table._convert_pos((17, 17))
        self.assertEqual(pos, len(list(combinations(range(18), 2))) + 18 - 1)

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

    def test__increment_count_single_pos_default(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=0, char='A')
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertEqual(t2['values'], [1])
        self.assertEqual(t2['i'], [0])
        self.assertEqual(t2['j'], [0])
        self.assertEqual(t2['shape'], (freq_table.num_pos, dna_alpha_size))
        self.assertEqual(freq_table.get_depth(), 0)

    def test__increment_count_single_pos_one(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=0, char='A', amount=1)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertEqual(t2['values'], [1])
        self.assertEqual(t2['i'], [0])
        self.assertEqual(t2['j'], [0])
        self.assertEqual(t2['shape'], (freq_table.num_pos, dna_alpha_size))
        self.assertEqual(freq_table.get_depth(), 0)

    def test__increment_count_single_pos_two(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=0, char='A', amount=2)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertEqual(t2['values'], [2])
        self.assertEqual(t2['i'], [0])
        self.assertEqual(t2['j'], [0])
        self.assertEqual(t2['shape'], (freq_table.num_pos, dna_alpha_size))
        self.assertEqual(freq_table.get_depth(), 0)

    def test__increment_count_pair_pos_default(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=(0, 1), char='AA')
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertEqual(t2['values'], [1])
        self.assertEqual(t2['i'], [1])
        self.assertEqual(t2['j'], [0])
        self.assertEqual(t2['shape'], (freq_table.num_pos, dna_pair_alpha_size))
        self.assertEqual(freq_table.get_depth(), 0)

    def test__increment_count_pair_pos_one(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=(0, 1), char='AA', amount=1)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertEqual(t2['values'], [1])
        self.assertEqual(t2['i'], [1])
        self.assertEqual(t2['j'], [0])
        self.assertEqual(t2['shape'], (freq_table.num_pos, dna_pair_alpha_size))
        self.assertEqual(freq_table.get_depth(), 0)

    def test__increment_count_pair_pos_two(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=(0, 1), char='AA', amount=2)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertEqual(t2['values'], [2])
        self.assertEqual(t2['i'], [1])
        self.assertEqual(t2['j'], [0])
        self.assertEqual(t2['shape'], (freq_table.num_pos, dna_pair_alpha_size))
        self.assertEqual(freq_table.get_depth(), 0)

    def test__increment_count_single_pos_multiple_increments(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=0, char='A', amount=1)
        t2 = freq_table.get_table()
        freq_table._increment_count(pos=0, char='A', amount=1)
        t3 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertNotEqual(t2, t3)
        self.assertEqual(t2['values'], [1])
        self.assertEqual(t2['i'], [0])
        self.assertEqual(t2['j'], [0])
        self.assertEqual(t2['shape'], (freq_table.num_pos, dna_alpha_size))
        self.assertEqual(t3['values'], [1, 1])
        self.assertEqual(t3['i'], [0, 0])
        self.assertEqual(t3['j'], [0, 0])
        self.assertEqual(t3['shape'], (freq_table.num_pos, dna_alpha_size))
        self.assertEqual(freq_table.get_depth(), 0)

    def test__increment_count_single_pos_multiple_different_increments(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=0, char='A', amount=1)
        t2 = freq_table.get_table()
        freq_table._increment_count(pos=5, char='C', amount=1)
        t3 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertNotEqual(t2, t3)
        self.assertEqual(t2['values'], [1])
        self.assertEqual(t2['i'], [0])
        self.assertEqual(t2['j'], [0])
        self.assertEqual(t2['shape'], (freq_table.num_pos, dna_alpha_size))
        self.assertEqual(t3['values'], [1, 1])
        self.assertEqual(t3['i'], [0, 5])
        self.assertEqual(t3['j'], [0, 2])
        self.assertEqual(t3['shape'], (freq_table.num_pos, dna_alpha_size))
        self.assertEqual(freq_table.get_depth(), 0)

    def test__increment_count_pair_pos_multiple_increments(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=(0, 1), char='AA', amount=1)
        t2 = freq_table.get_table()
        freq_table._increment_count(pos=(0, 1), char='AA', amount=1)
        t3 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertNotEqual(t2, t3)
        self.assertEqual(t2['values'], [1])
        self.assertEqual(t2['i'], [1])
        self.assertEqual(t2['j'], [0])
        self.assertEqual(t2['shape'], (freq_table.num_pos, dna_pair_alpha_size))
        self.assertEqual(t3['values'], [1, 1])
        self.assertEqual(t3['i'], [1, 1])
        self.assertEqual(t3['j'], [0, 0])
        self.assertEqual(t3['shape'], (freq_table.num_pos, dna_pair_alpha_size))
        self.assertEqual(freq_table.get_depth(), 0)

    def test__increment_count_pair_pos_multiple_different_increments(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=(0, 1), char='AA', amount=1)
        t2 = freq_table.get_table()
        freq_table._increment_count(pos=(0, 4), char='AC', amount=1)
        t3 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertNotEqual(t2, t3)
        self.assertEqual(t2['values'], [1])
        self.assertEqual(t2['i'], [1])
        self.assertEqual(t2['j'], [0])
        self.assertEqual(t2['shape'], (freq_table.num_pos, dna_pair_alpha_size))
        self.assertEqual(t3['values'], [1, 1])
        self.assertEqual(t3['i'], [1, 4])
        self.assertEqual(t3['j'], [0, 2])
        self.assertEqual(t3['shape'], (freq_table.num_pos, dna_pair_alpha_size))
        self.assertEqual(freq_table.get_depth(), 0)

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

    def test_finalize_table_no_increments(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table.finalize_table()
        t2 = freq_table.get_table()
        self.assertNotEqual(type(t1), type(t2))
        self.assertEqual(type(t2), csc_matrix)
        self.assertEqual(np.sum(t2), 0)

    def test_finalize_table_one_increments(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=0, char='A', amount=1)
        freq_table.finalize_table()
        t2 = freq_table.get_table()
        self.assertNotEqual(type(t1), type(t2))
        self.assertEqual(type(t2), csc_matrix)
        self.assertEqual(np.sum(t2), 1)
        self.assertEqual(t2[0, 0], 1)

    def test_finalize_table_two_increments(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table._increment_count(pos=0, char='A', amount=1)
        freq_table._increment_count(pos=0, char='A', amount=1)
        freq_table.finalize_table()
        t2 = freq_table.get_table()
        self.assertNotEqual(type(t1), type(t2))
        self.assertEqual(type(t2), csc_matrix)
        self.assertEqual(np.sum(t2), 2)
        self.assertEqual(t2[0, 0], 2)


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

    def test_characterize_alignment_dna_single_pos(self):
        aln_fn = write_out_temp_fasta(out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table.characterize_alignment(num_aln=num_aln)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertTrue(isinstance(t2, csc_matrix))
        expected_table = np.zeros(shape=(18, dna_alpha_size))
        expected_table[0, 0] = 3
        expected_table[1, 1] = 3
        expected_table[2, 3] = 3
        expected_table[3, [3, 4]] = [1, 2]
        expected_table[4, [0, 4]] = [1, 2]
        expected_table[5, [3, 4]] = [1, 2]
        expected_table[6, [0, 1]] = [2, 1]
        expected_table[7, [1, 2]] = [1, 2]
        expected_table[8, 1] = 3
        expected_table[9, [0, 4]] = [2, 1]
        expected_table[10, [3, 4]] = [2, 1]
        expected_table[11, [0, 4]]= [2, 1]
        expected_table[12, [3, 4]] = [2, 1]
        expected_table[13, [0, 4]] = [2, 1]
        expected_table[14, [3, 4]] = [2, 1]
        expected_table[15, [3, 4]] = [2, 1]
        expected_table[16, [0, 4]] = [2, 1]
        expected_table[17, [3, 4]] = [2, 1]
        self.assertFalse((t2 - expected_table).any())
        self.assertEqual(freq_table.get_depth(), 3)
        os.remove(aln_fn)

    def test_characterize_alignment_protein_single_pos(self):
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=protein_map)
        freq_table = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
        t1 = freq_table.get_table()
        freq_table.characterize_alignment(num_aln=num_aln)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertTrue(isinstance(t2, csc_matrix))
        expected_table = np.zeros((6, protein_alpha_size))
        expected_table[0, 11] = 3
        expected_table[1, [4, 23]] = [1, 2]
        expected_table[2, [5, 17]] = [1, 2]
        expected_table[3, [15, 23]] = [2, 1]
        expected_table[4, [4, 23]] = [2, 1]
        expected_table[5, [4, 23]] = [2, 1]
        self.assertFalse((t2 - expected_table).any())
        self.assertEqual(freq_table.get_depth(), 3)
        os.remove(aln_fn)

    def test_characterize_alignment_dna_pair_pos(self):
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        t1 = freq_table.get_table()
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=dna_single_to_pair)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertTrue(isinstance(t2, csc_matrix))
        expected_table = np.zeros(shape=(freq_table.num_pos, dna_pair_alpha_size))
        expected_table[0, 0] = 3
        expected_table[1, 1] = 3
        expected_table[2, 3] = 3
        expected_table[3, [3, 4]] = [1, 2]
        expected_table[4, [0, 4]] = [1, 2]
        expected_table[5, [3, 4]] = [1, 2]
        expected_table[6, [0, 1]] = [2, 1]
        expected_table[7, [1, 2]] = [1, 2]
        expected_table[8, 1] = 3
        expected_table[9, [0, 4]] = [2, 1]
        expected_table[10, [3, 4]] = [2, 1]
        expected_table[11, [0, 4]] = [2, 1]
        expected_table[12, [3, 4]] = [2, 1]
        expected_table[13, [0, 4]] = [2, 1]
        expected_table[14, [3, 4]] = [2, 1]
        expected_table[15, [3, 4]] = [2, 1]
        expected_table[16, [0, 4]] = [2, 1]
        expected_table[17, [3, 4]] = [2, 1]
        expected_table[18, 6] = 3
        expected_table[19, 8] = 3
        expected_table[20, [8, 9]] = [1, 2]
        expected_table[21, [5, 9]] = [1, 2]
        expected_table[22, [8, 9]] = [1, 2]
        expected_table[23, [5, 6]] = [2, 1]
        expected_table[24, [6, 7]] = [1, 2]
        expected_table[25, 6] = 3
        expected_table[26, [5, 9]] = [2, 1]
        expected_table[27, [8, 9]] = [2, 1]
        expected_table[28, [5, 9]] = [2, 1]
        expected_table[29, [8, 9]] = [2, 1]
        expected_table[30, [5, 9]] = [2, 1]
        expected_table[31, [8, 9]] = [2, 1]
        expected_table[32, [8, 9]] = [2, 1]
        expected_table[33, [5, 9]] = [2, 1]
        expected_table[34, [8, 9]] = [2, 1]
        expected_table[35, 18] = 3
        expected_table[36, [18, 19]] = [1, 2]
        expected_table[37, [15, 19]] = [1, 2]
        expected_table[38, [18, 19]] = [1, 2]
        expected_table[39, [15, 16]] = [2, 1]
        expected_table[40, [16, 17]] = [1, 2]
        expected_table[41, 16] = 3
        expected_table[42, [15, 19]] = [2, 1]
        expected_table[43, [18, 19]] = [2, 1]
        expected_table[44, [15, 19]] = [2, 1]
        expected_table[45, [18, 19]] = [2, 1]
        expected_table[46, [15, 19]] = [2, 1]
        expected_table[47, [18, 19]] = [2, 1]
        expected_table[48, [18, 19]] = [2, 1]
        expected_table[49, [15, 19]] = [2, 1]
        expected_table[50, [18, 19]] = [2, 1]
        expected_table[51, [18, 24]] = [1, 2]
        expected_table[52, [15, 24]] = [1, 2]
        expected_table[53, [18, 24]] = [1, 2]
        expected_table[54, [15, 20, 21]] = [1, 1, 1]
        expected_table[55, [17, 21, 22]] = [1, 1, 1]
        expected_table[56, [16, 21]] = [1, 2]
        expected_table[57, [19, 20]] = [1, 2]
        expected_table[58, [19, 23]] = [1, 2]
        expected_table[59, [19, 20]] = [1, 2]
        expected_table[60, [19, 23]] = [1, 2]
        expected_table[61, [19, 20]] = [1, 2]
        expected_table[62, [19, 23]] = [1, 2]
        expected_table[63, [19, 23]] = [1, 2]
        expected_table[64, [19, 20]] = [1, 2]
        expected_table[65, [19, 23]] = [1, 2]
        expected_table[66, [0, 24]] = [1, 2]
        expected_table[67, [3, 24]] = [1, 2]
        expected_table[68, [0, 20, 21]] = [1, 1, 1]
        expected_table[69, [2, 21, 22]] = [1, 1, 1]
        expected_table[70, [1, 21]] = [1, 2]
        expected_table[71, [4, 20]] = [1, 2]
        expected_table[72, [4, 23]] = [1, 2]
        expected_table[73, [4, 20]] = [1, 2]
        expected_table[74, [4, 23]] = [1, 2]
        expected_table[75, [4, 20]] = [1, 2]
        expected_table[76, [4, 23]] = [1, 2]
        expected_table[77, [4, 23]] = [1, 2]
        expected_table[78, [4, 20]] = [1, 2]
        expected_table[79, [4, 23]] = [1, 2]
        expected_table[80, [18, 24]] = [1, 2]
        expected_table[81, [15, 20, 21]] = [1, 1, 1]
        expected_table[82, [17, 21, 22]] = [1, 1, 1]
        expected_table[83, [16, 21]] = [1, 2]
        expected_table[84, [19, 20]] = [1, 2]
        expected_table[85, [19, 23]] = [1, 2]
        expected_table[86, [19, 20]] = [1, 2]
        expected_table[87, [19, 23]] = [1, 2]
        expected_table[88, [19, 20]] = [1, 2]
        expected_table[89, [19, 23]] = [1, 2]
        expected_table[90, [19, 23]] = [1, 2]
        expected_table[91, [19, 20]] = [1, 2]
        expected_table[92, [19, 23]] = [1, 2]
        expected_table[93, [0, 6]] = [2, 1]
        expected_table[94, [2, 6]] = [2, 1]
        expected_table[95, [1, 6]] = [2, 1]
        expected_table[96, [0, 4, 5]] = [1, 1, 1]
        expected_table[97, [3, 4, 8]] = [1, 1, 1]
        expected_table[98, [0, 4, 5]] = [1, 1, 1]
        expected_table[99, [3, 4, 8]] = [1, 1, 1]
        expected_table[100, [0, 4, 5]] = [1, 1, 1]
        expected_table[101, [3, 4, 8]] = [1, 1, 1]
        expected_table[102, [3, 4, 8]] = [1, 1, 1]
        expected_table[103, [0, 4, 5]] = [1, 1, 1]
        expected_table[104, [3, 4, 8]] = [1, 1, 1]
        expected_table[105, [6, 12]] = [1, 2]
        expected_table[106, [6, 11]] = [1, 2]
        expected_table[107, [5, 10, 14]] = [1, 1, 1]
        expected_table[108, [8, 13, 14]] = [1, 1, 1]
        expected_table[109, [5, 10, 14]] = [1, 1, 1]
        expected_table[110, [8, 13, 14]] = [1, 1, 1]
        expected_table[111, [5, 10, 14]] = [1, 1, 1]
        expected_table[112, [8, 13, 14]] = [1, 1, 1]
        expected_table[113, [8, 13, 14]] = [1, 1, 1]
        expected_table[114, [5, 10, 14]] = [1, 1, 1]
        expected_table[115, [8, 13, 14]] = [1, 1, 1]
        expected_table[116, 6] = 3
        expected_table[117, [5, 9]] = [2, 1]
        expected_table[118, [8, 9]] = [2, 1]
        expected_table[119, [5, 9]] = [2, 1]
        expected_table[120, [8, 9]] = [2, 1]
        expected_table[121, [5, 9]] = [2, 1]
        expected_table[122, [8, 9]] = [2, 1]
        expected_table[123, [8, 9]] = [2, 1]
        expected_table[124, [5, 9]] = [2, 1]
        expected_table[125, [8, 9]] = [2, 1]
        expected_table[126, [0, 24]] = [2, 1]
        expected_table[127, [3, 24]] = [2, 1]
        expected_table[128, [0, 24]] = [2, 1]
        expected_table[129, [3, 24]] = [2, 1]
        expected_table[130, [0, 24]] = [2, 1]
        expected_table[131, [3, 24]] = [2, 1]
        expected_table[132, [3, 24]] = [2, 1]
        expected_table[133, [0, 24]] = [2, 1]
        expected_table[134, [3, 24]] = [2, 1]
        expected_table[135, [18, 24]] = [2, 1]
        expected_table[136, [15, 24]] = [2, 1]
        expected_table[137, [18, 24]] = [2, 1]
        expected_table[138, [15, 24]] = [2, 1]
        expected_table[139, [18, 24]] = [2, 1]
        expected_table[140, [18, 24]] = [2, 1]
        expected_table[141, [15, 24]] = [2, 1]
        expected_table[142, [18, 24]] = [2, 1]
        expected_table[143, [0, 24]] = [2, 1]
        expected_table[144, [3, 24]] = [2, 1]
        expected_table[145, [0, 24]] = [2, 1]
        expected_table[146, [3, 24]] = [2, 1]
        expected_table[147, [3, 24]] = [2, 1]
        expected_table[148, [0, 24]] = [2, 1]
        expected_table[149, [3, 24]] = [2, 1]
        expected_table[150, [18, 24]] = [2, 1]
        expected_table[151, [15, 24]] = [2, 1]
        expected_table[152, [18, 24]] = [2, 1]
        expected_table[153, [18, 24]] = [2, 1]
        expected_table[154, [15, 24]] = [2, 1]
        expected_table[155, [18, 24]] = [2, 1]
        expected_table[156, [0, 24]] = [2, 1]
        expected_table[157, [3, 24]] = [2, 1]
        expected_table[158, [3, 24]] = [2, 1]
        expected_table[159, [0, 24]] = [2, 1]
        expected_table[160, [3, 24]] = [2, 1]
        expected_table[161, [18, 24]] = [2, 1]
        expected_table[162, [18, 24]] = [2, 1]
        expected_table[163, [15, 24]] = [2, 1]
        expected_table[164, [18, 24]] = [2, 1]
        expected_table[165, [18, 24]] = [2, 1]
        expected_table[166, [15, 24]] = [2, 1]
        expected_table[167, [18, 24]] = [2, 1]
        expected_table[168, [0, 24]] = [2, 1]
        expected_table[169, [3, 24]] = [2, 1]
        expected_table[170, [18, 24]] = [2, 1]
        self.assertFalse((t2 - expected_table).any())
        self.assertEqual(freq_table.get_depth(), 3)
        os.remove(aln_fn)

    def test_characterize_alignment_protein_pair_pos(self):
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=protein_map)
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        t1 = freq_table.get_table()
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=pro_single_to_pair)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertTrue(isinstance(t2, csc_matrix))
        expected_table = np.zeros((freq_table.num_pos, pro_pair_alpha_size))
        expected_table[0, 275] = 3
        expected_table[1, [268, 287]] = [1, 2]
        expected_table[2, [269, 281]] = [1, 2]
        expected_table[3, [279, 287]] = [2, 1]
        expected_table[4, [268, 287]] = [2, 1]
        expected_table[5, [268, 287]] = [2, 1]
        expected_table[6, [100, 575]] = [1, 2]
        expected_table[7, [113, 557, 569]] = [1, 1, 1]
        expected_table[8, [119, 567]] = [1, 2]
        expected_table[9, [119, 556]] = [1, 2]
        expected_table[10, [119, 556]] = [1, 2]
        expected_table[11, [125, 425]] = [1, 2]
        expected_table[12, [135, 423, 431]] = [1, 1, 1]
        expected_table[13, [124, 412, 431]] = [1, 1, 1]
        expected_table[14, [124, 412, 431]] = [1, 1, 1]
        expected_table[15, [375, 575]] = [2, 1]
        expected_table[16, [364, 575]] = [2, 1]
        expected_table[17, [364, 575]] = [2, 1]
        expected_table[18, [100, 575]] = [2, 1]
        expected_table[19, [100, 575]] = [2, 1]
        expected_table[20, [100, 575]] = [2, 1]
        self.assertFalse((t2 - expected_table).any())
        self.assertEqual(freq_table.get_depth(), 3)
        os.remove(aln_fn)

    def test_characterize_alignment_dna_single_pos_single_to_pair(self):
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=dna_single_to_pair)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertTrue(isinstance(t2, csc_matrix))
        expected_table = np.zeros(shape=(18, dna_alpha_size))
        expected_table[0, 0] = 3
        expected_table[1, 1] = 3
        expected_table[2, 3] = 3
        expected_table[3, [3, 4]] = [1, 2]
        expected_table[4, [0, 4]] = [1, 2]
        expected_table[5, [3, 4]] = [1, 2]
        expected_table[6, [0, 1]] = [2, 1]
        expected_table[7, [1, 2]] = [1, 2]
        expected_table[8, 1] = 3
        expected_table[9, [0, 4]] = [2, 1]
        expected_table[10, [3, 4]] = [2, 1]
        expected_table[11, [0, 4]] = [2, 1]
        expected_table[12, [3, 4]] = [2, 1]
        expected_table[13, [0, 4]] = [2, 1]
        expected_table[14, [3, 4]] = [2, 1]
        expected_table[15, [3, 4]] = [2, 1]
        expected_table[16, [0, 4]] = [2, 1]
        expected_table[17, [3, 4]] = [2, 1]
        self.assertFalse((t2 - expected_table).any())
        self.assertEqual(freq_table.get_depth(), 3)
        os.remove(aln_fn)

    def test_characterize_alignment_protein_single_pos_single_to_pair(self):
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=protein_map)
        freq_table = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
        t1 = freq_table.get_table()
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=pro_single_to_pair)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertTrue(isinstance(t2, csc_matrix))
        expected_table = np.zeros((6, protein_alpha_size))
        expected_table[0, 11] = 3
        expected_table[1, [4, 23]] = [1, 2]
        expected_table[2, [5, 17]] = [1, 2]
        expected_table[3, [15, 23]] = [2, 1]
        expected_table[4, [4, 23]] = [2, 1]
        expected_table[5, [4, 23]] = [2, 1]
        self.assertFalse((t2 - expected_table).any())
        self.assertEqual(freq_table.get_depth(), 3)
        os.remove(aln_fn)

    def test_characterize_alignment_failure_dna_pair_pos_no_single_to_pair(self):
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        with self.assertRaises(ValueError):
            freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=None)
        os.remove(aln_fn)

    def test_characterize_alignment_failure_protein_pair_pos_no_single_to_pair(self):
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
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
# class TestFrequencyTable(TestBase):
#
#     @classmethod
#     def setUpClass(cls):
#         super(TestFrequencyTable, cls).setUpClass()
#         cls.query_aln_fa_small = SeqAlignment(
#             file_name=cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln'],
#             query_id=cls.small_structure_id)
#         cls.query_aln_fa_small.import_alignment()
#         cls.query_aln_fa_large = SeqAlignment(
#             file_name=cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln'],
#             query_id=cls.large_structure_id)
#         cls.query_aln_fa_large.import_alignment()
#         cls.query_aln_fa_small = cls.query_aln_fa_small.remove_gaps()
#         cls.query_aln_fa_large = cls.query_aln_fa_large.remove_gaps()
#         cls.single_alphabet = Gapped(FullIUPACProtein())
#         cls.single_size, _, cls.single_mapping, cls.single_reverse = build_mapping(alphabet=cls.single_alphabet)
#         cls.pair_alphabet = MultiPositionAlphabet(alphabet=cls.single_alphabet, size=2)
#         cls.pair_size, _, cls.pair_mapping, cls.pair_reverse = build_mapping(alphabet=cls.pair_alphabet)
#         # cls.single_to_pair = {}
#         # for char in cls.pair_mapping:
#         #     key = (cls.single_mapping[char[0]], cls.single_mapping[char[1]])
#         #     cls.single_to_pair[key] = cls.pair_mapping[char]
#         cls.single_to_pair = np.zeros((max(cls.single_mapping.values()) + 1, max(cls.single_mapping.values()) + 1))
#         for char in cls.pair_mapping:
#             cls.single_to_pair[cls.single_mapping[char[0]], cls.single_mapping[char[1]]] = cls.pair_mapping[char]
#
#     def evaluate_init(self, alpha_size, mapping, reverse, seq_len, pos_size):
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
#                                     seq_len=seq_len, pos_size=pos_size)
#         self.assertEqual(freq_table.mapping, mapping)
#         self.assertTrue((freq_table.reverse_mapping == reverse).all())
#         self.assertEqual(freq_table.position_size, pos_size)
#         self.assertEqual(freq_table.sequence_length, seq_len)
#         if pos_size == 1:
#             expected_num_pos = seq_len
#         elif pos_size == 2:
#             expected_num_pos = int(((seq_len**2) - seq_len) / 2.0) + seq_len
#         else:
#             raise ValueError('Only 1 and 2 are supported for pos_size.')
#         self.assertEqual(freq_table.num_pos, expected_num_pos)
#         self.assertEqual(freq_table.get_depth(), 0)
#         expected_dict = {'values': [], 'i': [], 'j': [], 'shape': (expected_num_pos, alpha_size)}
#         self.assertEqual(freq_table.get_table(), expected_dict)
#
#     def test1a_init(self):
#         self.evaluate_init(alpha_size=self.single_size, mapping=self.single_mapping, reverse=self.single_reverse,
#                            seq_len=self.query_aln_fa_small.seq_length, pos_size=1)
#
#     def test1b_init(self):
#         self.evaluate_init(alpha_size=self.single_size, mapping=self.single_mapping, reverse=self.single_reverse,
#                            seq_len=self.query_aln_fa_large.seq_length, pos_size=1)
#
#     def test1c_init(self):
#         self.evaluate_init(alpha_size=self.pair_size, mapping=self.pair_mapping, reverse=self.pair_reverse,
#                            seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
#
#     def test1d_init(self):
#         with self.assertRaises(ValueError):
#             FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping, reverse_mapping=self.pair_reverse,
#                            seq_len=self.query_aln_fa_small.seq_length, pos_size=3)
#
#     def test1e_init(self):
#         with self.assertRaises(ValueError):
#             FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
#                            reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
#
#     # __convert_pos is implicitly tested in almost all other methods.
#
#     @staticmethod
#     def convert_pos(pos, seq_len):
#         if isinstance(pos, int):
#             return pos
#         else:
#             return np.sum([seq_len - x for x in range(pos[0])]) + (pos[1] - pos[0])
#
#     def evaluate__increment_count(self, alpha_size, mapping, reverse, seq_len, pos_size, updates):
#
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
#                                     seq_len=seq_len, pos_size=pos_size)
#         # Testing inserting wrong type of position and character
#         if pos_size == 1:
#             expected_num_pos = seq_len
#             with self.assertRaises(KeyError):
#                 freq_table._increment_count(pos=1, char='AAAA')
#             with self.assertRaises(TypeError):
#                 freq_table._increment_count(pos=(1, 2), char='AA')
#         if pos_size == 2:
#             expected_num_pos = int(((seq_len ** 2) - seq_len) / 2.0) + seq_len
#             with self.assertRaises(KeyError):
#                 freq_table._increment_count(pos=(1, 2), char='AAAA')
#             with self.assertRaises(TypeError):
#                 freq_table._increment_count(pos=1, char='A')
#             with self.assertRaises(TypeError):
#                 freq_table._increment_count(pos=(1, 2, 3), char='AAA')
#         expected_table = lil_matrix((expected_num_pos, alpha_size))
#         # Test inserting single correct position and character
#         freq_table._increment_count(pos=updates[0][0], char=updates[0][1])
#         expected_table[self.convert_pos(updates[0][0], seq_len), mapping[updates[0][1]]] += 1
#         tuple1 = freq_table.get_table()
#         table1 = csc_matrix((tuple1['values'], (tuple1['i'], tuple1['j'])), shape=tuple1['shape'])
#         diff1 = table1 - expected_table
#         self.assertFalse(diff1.toarray().any())
#         self.assertEqual(freq_table.get_depth(), 0)
#         # Test re-inserting single correct position and character
#         freq_table._increment_count(pos=updates[1][0], char=updates[1][1])
#         expected_table[self.convert_pos(updates[1][0], seq_len), mapping[updates[1][1]]] += 1
#         tuple2 = freq_table.get_table()
#         table2 = csc_matrix((tuple2['values'], (tuple2['i'], tuple2['j'])), shape=tuple2['shape'])
#         diff2 = table2 - expected_table
#         self.assertFalse(diff2.toarray().any())
#         self.assertEqual(freq_table.get_depth(), 0)
#         # Test inserting another correct position and character
#         freq_table._increment_count(pos=updates[2][0], char=updates[2][1])
#         expected_table[self.convert_pos(updates[2][0], seq_len), mapping[updates[2][1]]] += 1
#         tuple3 = freq_table.get_table()
#         table3 = csc_matrix((tuple3['values'], (tuple3['i'], tuple3['j'])), shape=tuple3['shape'])
#         diff3 = table3 - expected_table
#         self.assertFalse(diff3.toarray().any())
#         self.assertEqual(freq_table.get_depth(), 0)
#
#     def test2a__increment_count(self):
#         self.evaluate__increment_count(alpha_size=self.single_size, mapping=self.single_mapping,
#                                        reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                        pos_size=1, updates=[(1, 'A'), (1, 'A'), (2, 'G')])
#
#     def test2b__increment_count(self):
#         self.evaluate__increment_count(alpha_size=self.pair_size, mapping=self.pair_mapping, reverse=self.pair_reverse,
#                                        seq_len=self.query_aln_fa_small.seq_length, pos_size=2,
#                                        updates=[((1, 2), 'AA'), ((1, 2), 'AA'), ((2, 3), 'GG')])
#
#     def evaluate_characterize_sequence(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
#                                     seq_len=seq_len, pos_size=pos_size)
#         freq_table.characterize_sequence(seq=sequence)
#         freq_table.finalize_table()
#         for i in range(seq_len):
#             if pos_size == 1:
#                 expected_char = sequence[i]
#                 self.assertEqual(freq_table.get_count(pos=i, char=expected_char), 1)
#                 continue
#             for j in range(i, seq_len):
#                 expected_char = sequence[i] + sequence[j]
#                 self.assertEqual(freq_table.get_count(pos=(i, j), char=expected_char), 1)
#         table = freq_table.get_table()
#         column_sums = np.sum(table, axis=1)
#         self.assertFalse((column_sums - 1).any())
#
#     def test3a_characterize_sequence(self):
#         self.evaluate_characterize_sequence(alpha_size=self.single_size, mapping=self.single_mapping,
#                                             reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                             pos_size=1, sequence=self.query_aln_fa_small.query_sequence)
#
#     def test3b_characterize_sequence(self):
#         self.evaluate_characterize_sequence(alpha_size=self.pair_size, mapping=self.pair_mapping,
#                                             reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                             pos_size=2, sequence=self.query_aln_fa_small.query_sequence)
#
#     def evalaute_characterize_alignment(self, alpha_size, mapping, reverse, seq_len, pos_size, num_aln, single_to_pair,
#                                         sequence):
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse, seq_len=seq_len,
#                                     pos_size=pos_size)
#         freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=single_to_pair)
#         for i in range(seq_len):
#             if pos_size == 1:
#                 expected_char = sequence[i]
#                 self.assertEqual(freq_table.get_count(pos=i, char=expected_char), 1)
#                 continue
#             for j in range(i, seq_len):
#                 expected_char = sequence[i] + sequence[j]
#                 self.assertEqual(freq_table.get_count(pos=(i, j), char=expected_char), 1)
#         table = freq_table.get_table()
#         column_sums = np.sum(table, axis=1)
#         self.assertFalse((column_sums - 1).any())
#
#     def test4a_characterize_alignment(self):
#         sub_aln = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[self.small_structure_id])
#         num_aln = sub_aln._alignment_to_num(mapping=self.single_mapping)
#         self.evalaute_characterize_alignment(alpha_size=self.single_size,mapping=self.single_mapping,
#                                              reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                              pos_size=1, num_aln=num_aln, single_to_pair=None,
#                                              sequence=self.query_aln_fa_small.query_sequence)
#
#     def test4b_characterize_alignment(self):
#         sub_aln = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[self.small_structure_id])
#         num_aln = sub_aln._alignment_to_num(mapping=self.single_mapping)
#         self.evalaute_characterize_alignment(alpha_size=self.pair_size, mapping=self.pair_mapping,
#                                              reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                              pos_size=2, num_aln=num_aln, single_to_pair=self.single_to_pair,
#                                              sequence=self.query_aln_fa_small.query_sequence)
#
#     def evaluate_finalize_table(self, alpha_size, mapping, reverse, seq_len, pos_size):
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
#                                     seq_len=seq_len, pos_size=pos_size)
#         self.assertTrue(isinstance(freq_table.get_table(), dict))
#         freq_table.finalize_table()
#         self.assertTrue(isinstance(freq_table.get_table(), csc_matrix))
#
#     def test5a_finalize_table(self):
#         self.evaluate_finalize_table(alpha_size=self.single_size, mapping=self.single_mapping,
#                                      reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                      pos_size=1)
#
#     def test5b_finalize_table(self):
#         self.evaluate_finalize_table(alpha_size=self.pair_size, mapping=self.pair_mapping,
#                                      reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                      pos_size=2)
#
#     def test6_get_table(self):
#         freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
#                                     reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
#         expected_dict = {'values': [], 'i': [], 'j': [], 'shape': (freq_table.num_pos, self.single_size)}
#         table1 = freq_table.get_table()
#         self.assertEqual(table1, expected_dict)
#         table2 = freq_table.get_table()
#         self.assertEqual(table2, expected_dict)
#         self.assertIsNot(table1, table2)
#         freq_table.finalize_table()
#         expected_table = csc_matrix((freq_table.num_pos, self.single_size))
#         table3 = freq_table.get_table()
#         diff = table3 - expected_table
#         self.assertFalse(diff.count_nonzero() > 0)
#
#     def test7_set_depth(self):
#         freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
#                                     reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
#         depth1 = freq_table.get_depth()
#         self.assertEqual(depth1, 0)
#         freq_table.set_depth(1)
#         depth2 = freq_table.get_depth()
#         self.assertNotEqual(depth1, depth2)
#         self.assertEqual(depth2, 1)
#
#     def test8_get_depth(self):
#         freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
#                                     reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
#         depth1 = freq_table.get_depth()
#         depth2 = freq_table.get_depth()
#         self.assertEqual(depth1, 0)
#         self.assertEqual(depth1, depth2)
#
#     def evaluate_get_positions(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
#                                     seq_len=seq_len, pos_size=pos_size)
#         freq_table.characterize_sequence(sequence)
#         freq_table.finalize_table()
#         positions = freq_table.get_positions()
#         count = 0
#         for i in range(seq_len):
#             if pos_size == 1:
#                 self.assertEqual(i, positions[count])
#                 count += 1
#                 continue
#             for j in range(i, seq_len):
#                 self.assertEqual((i, j), positions[count])
#                 count += 1
#
#     def test9a_get_positions(self):
#         self.evaluate_get_positions(alpha_size=self.single_size, mapping=self.single_mapping,
#                                     reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                     pos_size=1, sequence=self.query_aln_fa_small.query_sequence)
#
#     def test9b_get_positions(self):
#         self.evaluate_get_positions(alpha_size=self.pair_size, mapping=self.pair_mapping,
#                                     reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                     pos_size=2, sequence=self.query_aln_fa_small.query_sequence)
#
#     def evaluate_get_chars(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse, seq_len=seq_len,
#                                     pos_size=pos_size)
#         freq_table.characterize_sequence(seq=sequence)
#         freq_table.finalize_table()
#         for pos in freq_table.get_positions():
#             if pos_size == 1:
#                 self.assertEqual(freq_table.get_chars(pos=pos), [sequence[pos]])
#             elif pos_size == 2:
#                 self.assertEqual(freq_table.get_chars(pos=pos), [sequence[pos[0]] + sequence[pos[1]]])
#             else:
#                 raise ValueError('1 and 2 are the only supported values for pos_size.')
#
#     def test10a_get_chars(self):
#         self.evaluate_get_chars(alpha_size=self.single_size, mapping=self.single_mapping,
#                                 reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                 pos_size=1, sequence=self.query_aln_fa_small.query_sequence)
#
#     def test10b_get_chars(self):
#         self.evaluate_get_chars(alpha_size=self.pair_size, mapping=self.pair_mapping,
#                                 reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                 pos_size=2, sequence=self.query_aln_fa_small.query_sequence)
#
#     def evaluate_get_count(self, alphabet, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
#                                     seq_len=seq_len, pos_size=pos_size)
#         freq_table.characterize_sequence(seq=sequence)
#         freq_table.finalize_table()
#         pos_counts = freq_table.get_table().sum(axis=1).A1
#         for pos in freq_table.get_positions():
#             if pos_size == 1:
#                 curr_char = sequence[pos]
#             elif pos_size == 2:
#                 curr_char = sequence[pos[0]] + sequence[pos[1]]
#             else:
#                 raise ValueError('1 or 2 are the only options for pos_size.')
#             self.assertEqual(freq_table.get_count(pos=pos, char=curr_char), 1)
#             curr_pos = int(self.convert_pos(pos, seq_len))
#             self.assertEqual(pos_counts[curr_pos], 1)
#
#     def test11a_get_count(self):
#         self.evaluate_get_count(alphabet=self.single_alphabet, alpha_size=self.single_size, mapping=self.single_mapping,
#                                 reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length, pos_size=1,
#                                 sequence=self.query_aln_fa_small.query_sequence)
#
#     def test11b_get_count(self):
#         self.evaluate_get_count(alphabet=self.pair_alphabet, alpha_size=self.pair_size, mapping=self.pair_mapping,
#                                 reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length, pos_size=2,
#                                 sequence=self.query_aln_fa_small.query_sequence)
#
#     def evaluate_get_count_array(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
#                                     seq_len=seq_len, pos_size=pos_size)
#         freq_table.characterize_sequence(seq=sequence)
#         freq_table.finalize_table()
#         for pos in freq_table.get_positions():
#             self.assertEqual(freq_table.get_count_array(pos=pos), np.array([1]))
#
#     def test12a_get_count_array(self):
#         self.evaluate_get_count_array(alpha_size=self.single_size, mapping=self.single_mapping,
#                                       reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                       pos_size=1, sequence=self.query_aln_fa_small.query_sequence)
#
#     def test12b_get_count_array(self):
#         self.evaluate_get_count_array(alpha_size=self.pair_size, mapping=self.pair_mapping,
#                                       reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                       pos_size=2, sequence=self.query_aln_fa_small.query_sequence)
#
#     def evaluate_get_count_matrix(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse, seq_len=seq_len,
#                                     pos_size=pos_size)
#         freq_table.characterize_sequence(seq=sequence)
#         freq_table.finalize_table()
#         if pos_size == 1:
#             expected_num_pos = seq_len
#         elif pos_size == 2:
#             expected_num_pos = int(((seq_len**2) - seq_len) / 2.0) + seq_len
#         else:
#             raise ValueError('Only 1 and 2 are supported for pos_size.')
#         expected_mat = np.zeros((expected_num_pos, alpha_size))
#         count = 0
#         for i in range(seq_len):
#             if pos_size == 1:
#                 expected_char = sequence[i]
#                 expected_mat[count, mapping[expected_char]] += 1
#                 count += 1
#                 continue
#             for j in range(i, seq_len):
#                 expected_char = sequence[i] + sequence[j]
#                 expected_mat[count, mapping[expected_char]] += 1
#                 count += 1
#         mat = freq_table.get_count_matrix()
#         diff = mat - expected_mat
#         self.assertTrue(not diff.any())
#
#     def test13a_get_count_matrix(self):
#         self.evaluate_get_count_matrix(alpha_size=self.single_size, mapping=self.single_mapping,
#                                        reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                        pos_size=1, sequence=self.query_aln_fa_small.query_sequence)
#
#     def test13b_get_count_matrix(self):
#         self.evaluate_get_count_array(alpha_size=self.pair_size, mapping=self.pair_mapping,
#                                       reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                       pos_size=2, sequence=self.query_aln_fa_small.query_sequence)
#
#     def evaluate_get_frequency(self, alphabet, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse, seq_len=seq_len,
#                                     pos_size=pos_size)
#         freq_table.characterize_sequence(seq=sequence)
#         freq_table.finalize_table()
#         pos_counts = freq_table.get_table().sum(axis=1).A1
#         for pos in freq_table.get_positions():
#             if pos_size == 1:
#                 curr_char = sequence[pos]
#             elif pos_size == 2:
#                 curr_char = sequence[pos[0]] + sequence[pos[1]]
#             else:
#                 raise ValueError('1 and 2 are the only accepted values for pos_size.')
#             self.assertEqual(freq_table.get_frequency(pos=pos, char=curr_char), 1.0)
#             curr_pos = int(self.convert_pos(pos, seq_len))
#             self.assertEqual(pos_counts[curr_pos], 1)
#
#     def test14a_get_frequency(self):
#         self.evaluate_get_frequency(alphabet=self.single_alphabet, alpha_size=self.single_size,
#                                     mapping=self.single_mapping, reverse=self.single_reverse,
#                                     seq_len=self.query_aln_fa_small.seq_length, pos_size=1,
#                                     sequence=self.query_aln_fa_small.query_sequence)
#
#     def test14b_get_frequency(self):
#         self.evaluate_get_frequency(alphabet=self.pair_alphabet, alpha_size=self.pair_size, mapping=self.pair_mapping,
#                                     reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length, pos_size=2,
#                                     sequence=self.query_aln_fa_small.query_sequence)
#
#     def evaluate_get_frequency_array(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
#                                     seq_len=seq_len, pos_size=pos_size)
#         freq_table.characterize_sequence(seq=sequence)
#         freq_table.finalize_table()
#         for pos in freq_table.get_positions():
#             self.assertEqual(freq_table.get_frequency_array(pos=pos), np.array([1.0]))
#
#     def test15a_get_frequency_array(self):
#         self.evaluate_get_frequency_array(alpha_size=self.single_size, mapping=self.single_mapping,
#                                           reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                           pos_size=1, sequence=self.query_aln_fa_small.query_sequence)
#
#     def test15b_get_frequency_array(self):
#         self.evaluate_get_frequency_array(alpha_size=self.pair_size, mapping=self.pair_mapping,
#                                           reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                           pos_size=2, sequence=self.query_aln_fa_small.query_sequence)
#
#     def evaluate_get_frequency_matrix(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
#                                     seq_len=seq_len, pos_size=pos_size)
#         freq_table.characterize_sequence(seq=sequence)
#         freq_table.finalize_table()
#         if pos_size == 1:
#             expected_num_pos = seq_len
#         elif pos_size == 2:
#             expected_num_pos = int(((seq_len**2) - seq_len) / 2.0) + seq_len
#         else:
#             raise ValueError('Only 1 and 2 are supported for pos_size.')
#         expected_mat = np.zeros((expected_num_pos, alpha_size))
#         count = 0
#         for i in range(seq_len):
#             if pos_size == 1:
#                 expected_char = sequence[i]
#                 expected_mat[count, mapping[expected_char]] += 1.0
#                 count += 1
#                 continue
#             for j in range(i, seq_len):
#                 expected_char = sequence[i] + sequence[j]
#                 expected_mat[count, mapping[expected_char]] += 1.0
#                 count += 1
#         mat = freq_table.get_frequency_matrix()
#         diff = mat - expected_mat
#         self.assertTrue(not diff.any())
#
#     def test16a_get_frequency_matrix(self):
#         self.evaluate_get_frequency_matrix(alpha_size=self.single_size, mapping=self.single_mapping,
#                                            reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                            pos_size=1, sequence=self.query_aln_fa_small.query_sequence)
#
#     def test16b_get_frequency_matrix(self):
#         self.evaluate_get_frequency_matrix(alpha_size=self.pair_size, mapping=self.pair_mapping,
#                                            reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
#                                            pos_size=2, sequence=self.query_aln_fa_small.query_sequence)
#
#     def evaluate_to_csv(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence, fn):
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse, seq_len=seq_len,
#                                     pos_size=pos_size)
#         freq_table.characterize_sequence(seq=sequence)
#         freq_table.finalize_table()
#         fn = os.path.join(self.testing_dir, fn)
#         freq_table.to_csv(file_path=fn)
#         loaded_freq_table = pd.read_csv(fn, sep='\t', header=0, index_col=None, keep_default_na=False)
#         loaded_freq_table.set_index('Position', inplace=True)
#         for i in range(seq_len):
#             if pos_size == 1:
#                 expected_char = sequence[i]
#                 self.assertEqual(loaded_freq_table.loc[i, 'Variability'], 1)
#                 self.assertEqual(loaded_freq_table.loc[i, 'Characters'], expected_char)
#                 self.assertEqual(loaded_freq_table.loc[i, 'Counts'], 1)
#                 self.assertEqual(loaded_freq_table.loc[i, 'Frequencies'], 1.0)
#                 continue
#             for j in range(i, seq_len):
#                 expected_char = sequence[i] + sequence[j]
#                 self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Variability'], 1)
#                 self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Characters'], expected_char)
#                 self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Counts'], 1)
#                 self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Frequencies'], 1.0)
#         os.remove(fn)
#
#     def test17a_to_csv(self):
#         self.evaluate_to_csv(alpha_size=self.single_size, mapping=self.single_mapping, reverse=self.single_reverse,
#                              seq_len=self.query_aln_fa_small.seq_length, pos_size=1,
#                              sequence=self.query_aln_fa_small.query_sequence, fn='small_query_seq_freq_table.tsv')
#
#     def test17b_to_csv(self):
#         self.evaluate_to_csv(alpha_size=self.pair_size, mapping=self.pair_mapping, reverse=self.pair_reverse,
#                              seq_len=self.query_aln_fa_small.seq_length, pos_size=2,
#                              sequence=self.query_aln_fa_small.query_sequence, fn='small_query_seq_freq_table.tsv')
#
#     def evaluate_load_csv(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence, fn):
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
#                                     seq_len=seq_len, pos_size=pos_size)
#         freq_table.characterize_sequence(seq=sequence)
#         freq_table.finalize_table()
#         fn = os.path.join(self.testing_dir, fn)
#         freq_table.to_csv(file_path=fn)
#         loaded_freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
#                                            seq_len=seq_len, pos_size=pos_size)
#         loaded_freq_table.load_csv(fn)
#         diff = freq_table.get_table() - loaded_freq_table.get_table()
#         self.assertFalse(diff.count_nonzero() > 0)
#         self.assertEqual(freq_table.get_depth(), loaded_freq_table.get_depth())
#         os.remove(fn)
#
#     def test18a_load_csv(self):
#         self.evaluate_load_csv(alpha_size=self.single_size, mapping=self.single_mapping, reverse=self.single_reverse,
#                                seq_len=self.query_aln_fa_small.seq_length, pos_size=1,
#                                sequence=self.query_aln_fa_small.query_sequence, fn='small_query_seq_freq_table.tsv')
#
#     def test18b_load_csv(self):
#         self.evaluate_load_csv(alpha_size=self.pair_size, mapping=self.pair_mapping, reverse=self.pair_reverse,
#                                seq_len=self.query_aln_fa_small.seq_length, pos_size=2,
#                                sequence=self.query_aln_fa_small.query_sequence, fn='small_query_seq_freq_table.tsv')
#
#     def evaluate_add(self, aln, alpha_size, mapping, reverse, pos_size):
#         query_seq_index = aln.seq_order.index(aln.query_id)
#         second_index = 0 if query_seq_index != 0 else aln.size - 1
#         freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
#                                     seq_len=aln.seq_length, pos_size=pos_size)
#         freq_table.characterize_sequence(seq=aln.query_sequence)
#         freq_table.characterize_sequence(seq=aln.alignment[second_index].seq)
#         freq_table.finalize_table()
#         freq_table1 = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
#                                      seq_len=aln.seq_length, pos_size=pos_size)
#         freq_table1.characterize_sequence(seq=aln.query_sequence)
#         freq_table2 = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
#                                      seq_len=aln.seq_length, pos_size=pos_size)
#         freq_table2.characterize_sequence(seq=aln.alignment[second_index].seq)
#         freq_table1.finalize_table()
#         freq_table2.finalize_table()
#         freq_table_sum1 = freq_table1 + freq_table2
#         self.assertEqual(freq_table.mapping, freq_table_sum1.mapping)
#         self.assertTrue((freq_table.reverse_mapping == freq_table_sum1.reverse_mapping).all())
#         self.assertEqual(freq_table.num_pos, freq_table_sum1.num_pos)
#         self.assertEqual(freq_table.position_size, freq_table_sum1.position_size)
#         diff = freq_table.get_table() - freq_table_sum1.get_table()
#         self.assertFalse(diff.toarray().any())
#         self.assertEqual(freq_table.get_positions(), freq_table_sum1.get_positions())
#         self.assertEqual(freq_table.get_depth(), freq_table_sum1.get_depth())
#
#     def test19a_add(self):
#         self.evaluate_add(aln=self.query_aln_fa_small, alpha_size=self.single_size, mapping=self.single_mapping,
#                           reverse=self.single_reverse, pos_size=1)
#
#     def test19b_add(self):
#         self.evaluate_add(aln=self.query_aln_fa_small, alpha_size=self.pair_size, mapping=self.pair_mapping,
#                           reverse=self.pair_reverse, pos_size=2)
#
#     def test19c_add(self):
#         self.evaluate_add(aln=self.query_aln_fa_large, alpha_size=self.single_size, mapping=self.single_mapping,
#                           reverse=self.single_reverse, pos_size=1)
#
#     def test19d_add(self):
#         self.evaluate_add(aln=self.query_aln_fa_large, alpha_size=self.pair_size, mapping=self.pair_mapping,
#                           reverse=self.pair_reverse, pos_size=2)


if __name__ == '__main__':
    unittest.main()
