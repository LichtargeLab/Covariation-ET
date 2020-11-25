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
from Bio.Alphabet import Gapped
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from test_seqAlignment import generate_temp_fn, write_out_temp_fasta
from utils import build_mapping
from SeqAlignment import SeqAlignment
from FrequencyTable import FrequencyTable
from MatchMismatchTable import MatchMismatchTable
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
pro_single_to_pair = np.zeros((max(protein_map.values()) + 1, max(protein_map.values()) + 1), dtype=np.int)
pro_single_to_pair_map = {}
for char in pro_pair_map:
    pro_single_to_pair[protein_map[char[0]], protein_map[char[1]]] = pro_pair_map[char]
    pro_single_to_pair_map[(protein_map[char[0]], protein_map[char[1]])] = pro_pair_map[char]
protein_seq1 = SeqRecord(id='seq1', seq=Seq('MET---', alphabet=FullIUPACProtein()))
protein_seq2 = SeqRecord(id='seq2', seq=Seq('M-TREE', alphabet=FullIUPACProtein()))
protein_seq3 = SeqRecord(id='seq3', seq=Seq('M-FREE', alphabet=FullIUPACProtein()))
protein_msa = MultipleSeqAlignment(records=[protein_seq1, protein_seq2, protein_seq3], alphabet=FullIUPACProtein())
dna_seq1 = SeqRecord(id='seq1', seq=Seq('ATGGAGACT---------', alphabet=FullIUPACDNA()))
dna_seq2 = SeqRecord(id='seq2', seq=Seq('ATG---ACTAGAGAGGAG', alphabet=FullIUPACDNA()))
dna_seq3 = SeqRecord(id='seq3', seq=Seq('ATG---TTTAGAGAGGAG', alphabet=FullIUPACDNA()))
dna_msa = MultipleSeqAlignment(records=[dna_seq1, dna_seq2, dna_seq3], alphabet=FullIUPACDNA())

aln_fn = write_out_temp_fasta(
                out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
aln.import_alignment()
os.remove(aln_fn)
num_aln = aln._alignment_to_num(mapping=protein_map)


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


class TestFrequencyTableCharacterizeSequence(TestCase):

    def test_characterize_sequence_dna_single_pos(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table.characterize_sequence(seq=dna_seq1)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertTrue(isinstance(t2, dict))
        freq_table.finalize_table()
        t3 = freq_table.get_table()
        self.assertTrue(isinstance(t3, csc_matrix))
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(dna_seq1.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=dna_map)
        freq_table2 = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table2.characterize_alignment(num_aln=num_aln)
        expected_table = freq_table2.get_table()
        self.assertFalse((t3 - expected_table).toarray().any())
        self.assertEqual(freq_table.get_depth(), 1)

    def test_characterize_sequence_protein_single_pos(self):
        freq_table = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
        t1 = freq_table.get_table()
        freq_table.characterize_sequence(seq=protein_seq1)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertTrue(isinstance(t2, dict))
        freq_table.finalize_table()
        t3 = freq_table.get_table()
        self.assertTrue(isinstance(t3, csc_matrix))
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        freq_table2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
        freq_table2.characterize_alignment(num_aln=num_aln)
        expected_table = freq_table2.get_table()
        self.assertFalse((t3 - expected_table).toarray().any())
        self.assertEqual(freq_table.get_depth(), 1)

    def test_characterize_sequence_dna_single_pair(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        t1 = freq_table.get_table()
        freq_table.characterize_sequence(seq=dna_seq1)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertTrue(isinstance(t2, dict))
        freq_table.finalize_table()
        t3 = freq_table.get_table()
        self.assertTrue(isinstance(t3, csc_matrix))
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(dna_seq1.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table2 = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table2.characterize_alignment(num_aln=num_aln, single_to_pair=dna_single_to_pair)
        expected_table = freq_table2.get_table()
        self.assertFalse((t3 - expected_table).toarray().any())
        self.assertEqual(freq_table.get_depth(), 1)

    def test_characterize_sequence_protein_single_pair(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        t1 = freq_table.get_table()
        freq_table.characterize_sequence(seq=protein_seq1)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        self.assertTrue(isinstance(t2, dict))
        freq_table.finalize_table()
        t3 = freq_table.get_table()
        self.assertTrue(isinstance(t3, csc_matrix))
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=protein_map)
        os.remove(aln_fn)
        freq_table2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        freq_table2.characterize_alignment(num_aln=num_aln, single_to_pair=pro_single_to_pair)
        expected_table = freq_table2.get_table()
        self.assertFalse((t3 - expected_table).toarray().any())
        self.assertEqual(freq_table.get_depth(), 1)

    def test_characterize_multi_sequence_dna_single_pos(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        t1 = freq_table.get_table()
        freq_table.characterize_sequence(seq=dna_seq1)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        freq_table.characterize_sequence(seq=dna_seq2)
        t3 = freq_table.get_table()
        self.assertNotEqual(t2, t3)
        freq_table.characterize_sequence(seq=dna_seq3)
        t4 = freq_table.get_table()
        self.assertNotEqual(t3, t4)
        self.assertTrue(isinstance(t4, dict))
        freq_table.finalize_table()
        t5 = freq_table.get_table()
        self.assertTrue(isinstance(t5, csc_matrix))
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=dna_map)
        freq_table2 = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table2.characterize_alignment(num_aln=num_aln)
        expected_table = freq_table2.get_table()
        self.assertFalse((t5 - expected_table).toarray().any())
        self.assertEqual(freq_table.get_depth(), 3)

    def test_characterize_multi_sequence_protein_single_pos(self):
        freq_table = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
        t1 = freq_table.get_table()
        freq_table.characterize_sequence(seq=protein_seq1)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        freq_table.characterize_sequence(seq=protein_seq2)
        t3 = freq_table.get_table()
        self.assertNotEqual(t2, t3)
        freq_table.characterize_sequence(seq=protein_seq3)
        t4 = freq_table.get_table()
        self.assertNotEqual(t3, t4)
        self.assertTrue(isinstance(t4, dict))
        freq_table.finalize_table()
        t5 = freq_table.get_table()
        self.assertTrue(isinstance(t5, csc_matrix))
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        freq_table2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
        freq_table2.characterize_alignment(num_aln=num_aln)
        expected_table = freq_table2.get_table()
        self.assertFalse((t5 - expected_table).toarray().any())
        self.assertEqual(freq_table.get_depth(), 3)

    def test_characterize_multi_sequence_dna_pair_pos(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        t1 = freq_table.get_table()
        freq_table.characterize_sequence(seq=dna_seq1)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        freq_table.characterize_sequence(seq=dna_seq2)
        t3 = freq_table.get_table()
        self.assertNotEqual(t2, t3)
        freq_table.characterize_sequence(seq=dna_seq3)
        t4 = freq_table.get_table()
        self.assertNotEqual(t3, t4)
        self.assertTrue(isinstance(t4, dict))
        freq_table.finalize_table()
        t5 = freq_table.get_table()
        self.assertTrue(isinstance(t5, csc_matrix))
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table2 = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table2.characterize_alignment(num_aln=num_aln, single_to_pair=dna_single_to_pair)
        expected_table = freq_table2.get_table()
        self.assertFalse((t5 - expected_table).toarray().any())
        self.assertEqual(freq_table.get_depth(), 3)

    def test_characterize_multi_sequence_protein_pair_pos(self):
        freq_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        t1 = freq_table.get_table()
        freq_table.characterize_sequence(seq=protein_seq1)
        t2 = freq_table.get_table()
        self.assertNotEqual(t1, t2)
        freq_table.characterize_sequence(seq=protein_seq2)
        t3 = freq_table.get_table()
        self.assertNotEqual(t2, t3)
        freq_table.characterize_sequence(seq=protein_seq3)
        t4 = freq_table.get_table()
        self.assertNotEqual(t3, t4)
        self.assertTrue(isinstance(t4, dict))
        freq_table.finalize_table()
        t5 = freq_table.get_table()
        self.assertTrue(isinstance(t5, csc_matrix))
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=protein_map)
        os.remove(aln_fn)
        freq_table2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        freq_table2.characterize_alignment(num_aln=num_aln, single_to_pair=pro_single_to_pair)
        expected_table = freq_table2.get_table()
        self.assertFalse((t5 - expected_table).toarray().any())
        self.assertEqual(freq_table.get_depth(), 3)


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

    def test_get_initial(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        table = freq_table.get_table()
        self.assertTrue(isinstance(table, dict))
        self.assertEqual(table, {'values': [], 'i': [], 'j': [], 'shape': (freq_table.num_pos, dna_alpha_size)})
        self.assertEqual(freq_table.get_depth(), 0)
        positions = freq_table.get_positions()
        expected_positions = list(range(18))
        self.assertEqual(positions, expected_positions)
        for p in positions:
            with self.assertRaises(AttributeError):
                freq_table.get_chars(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_count(pos=p, char='A')
            with self.assertRaises(AttributeError):
                freq_table.get_count_array(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_frequency(pos=p, char='A')
            with self.assertRaises(AttributeError):
                freq_table.get_frequency_array(pos=p)
        with self.assertRaises(AttributeError):
            freq_table.get_count_matrix()
        with self.assertRaises(AttributeError):
            freq_table.get_frequency_matrix()
        freq_table.finalize_table()
        for p in positions:
            with self.assertRaises(AttributeError):
                self.assertEqual(freq_table.get_chars(pos=p), [])
            with self.assertRaises(AttributeError):
                freq_table.get_count(pos=p, char='A')
            with self.assertRaises(AttributeError):
                freq_table.get_count_array(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_frequency(pos=p, char='A')
            with self.assertRaises(AttributeError):
                freq_table.get_frequency_array(pos=p)
        with self.assertRaises(AttributeError):
            freq_table.get_count_matrix()
        with self.assertRaises(AttributeError):
            freq_table.get_frequency_matrix()

    def test_get_single_update_single_pos(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table._increment_count(pos=0, char='A', amount=1)
        table = freq_table.get_table()
        self.assertTrue(isinstance(table, dict))
        self.assertEqual(table, {'values': [1], 'i': [0], 'j': [0], 'shape': (freq_table.num_pos, dna_alpha_size)})
        self.assertEqual(freq_table.get_depth(), 0)
        positions = freq_table.get_positions()
        expected_positions = list(range(18))
        self.assertEqual(positions, expected_positions)
        for p in positions:
            with self.assertRaises(AttributeError):
                freq_table.get_chars(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_count(pos=p, char='A')
            with self.assertRaises(AttributeError):
                freq_table.get_count_array(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_frequency(pos=p, char='A')
            with self.assertRaises(AttributeError):
                freq_table.get_frequency_array(pos=p)
        with self.assertRaises(AttributeError):
            freq_table.get_count_matrix()
        with self.assertRaises(AttributeError):
            freq_table.get_frequency_matrix()
        freq_table.finalize_table()
        table2 = freq_table.get_table()
        self.assertTrue(isinstance(table2, csc_matrix))
        expected_table = np.zeros((freq_table.num_pos, dna_alpha_size))
        expected_table[0, 0] = 1
        self.assertFalse((table2 - expected_table).any())
        for p in positions:
            with self.assertRaises(AttributeError):
                self.assertEqual(freq_table.get_chars(pos=p))
            with self.assertRaises(AttributeError):
                freq_table.get_count(pos=p, char='A')
            with self.assertRaises(AttributeError):
                freq_table.get_count_array(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_frequency(pos=p, char='A')
            with self.assertRaises(AttributeError):
                freq_table.get_frequency_array(pos=p)
        with self.assertRaises(AttributeError):
            freq_table.get_count_matrix()
        with self.assertRaises(AttributeError):
            freq_table.get_frequency_matrix()

    def test_get_single_update_pair_pos(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table._increment_count(pos=(0, 0), char='AA', amount=1)
        table = freq_table.get_table()
        self.assertTrue(isinstance(table, dict))
        self.assertEqual(table, {'values': [1], 'i': [0], 'j': [0], 'shape': (freq_table.num_pos, dna_pair_alpha_size)})
        self.assertEqual(freq_table.get_depth(), 0)
        positions = freq_table.get_positions()
        expected_positions = list(combinations(range(18), 2)) + [(x, x) for x in range(18)]
        self.assertEqual(set(positions), set(expected_positions))
        for p in positions:
            with self.assertRaises(AttributeError):
                freq_table.get_chars(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_count(pos=p, char='A')
            with self.assertRaises(AttributeError):
                freq_table.get_count_array(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_frequency(pos=p, char='A')
            with self.assertRaises(AttributeError):
                freq_table.get_frequency_array(pos=p)
        with self.assertRaises(AttributeError):
            freq_table.get_count_matrix()
        with self.assertRaises(AttributeError):
            freq_table.get_frequency_matrix()
        freq_table.finalize_table()
        table2 = freq_table.get_table()
        self.assertTrue(isinstance(table2, csc_matrix))
        expected_table = np.zeros((freq_table.num_pos, dna_pair_alpha_size))
        expected_table[0, 0] = 1
        self.assertFalse((table2 - expected_table).any())
        for p in positions:
            with self.assertRaises(AttributeError):
                freq_table.get_chars(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_count(pos=p, char='A')
            with self.assertRaises(AttributeError):
                freq_table.get_count_array(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_frequency(pos=p, char='A')
            with self.assertRaises(AttributeError):
                freq_table.get_frequency_array(pos=p)
        with self.assertRaises(AttributeError):
            freq_table.get_count_matrix()
        with self.assertRaises(AttributeError):
            freq_table.get_frequency_matrix()

    def test_get_single_seq_single_pos(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        table = freq_table.get_table()
        self.assertTrue(isinstance(table, dict))
        self.assertEqual(freq_table.get_depth(), 1)
        positions = freq_table.get_positions()
        expected_positions = list(range(18))
        self.assertEqual(positions, expected_positions)
        for p in positions:
            with self.assertRaises(AttributeError):
                freq_table.get_chars(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_count(pos=p, char='A')
            with self.assertRaises(AttributeError):
                freq_table.get_count_array(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_frequency(pos=p, char='A')
            with self.assertRaises(AttributeError):
                freq_table.get_frequency_array(pos=p)
        with self.assertRaises(AttributeError):
            freq_table.get_count_matrix()
        with self.assertRaises(AttributeError):
            freq_table.get_frequency_matrix()
        freq_table.finalize_table()
        table2 = freq_table.get_table()
        self.assertTrue(isinstance(table2, csc_matrix))
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(dna_seq1.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table2 = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table2.characterize_alignment(num_aln=num_aln)
        expected_table = freq_table2.get_table()
        self.assertFalse((table2 - expected_table).toarray().any())
        for p in positions:
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
        table = freq_table.get_table()
        self.assertTrue(isinstance(table, dict))
        self.assertEqual(freq_table.get_depth(), 1)
        positions = freq_table.get_positions()
        expected_positions = list(combinations(range(18), 2)) + [(x, x) for x in range(18)]
        self.assertEqual(set(positions), set(expected_positions))
        for p in positions:
            with self.assertRaises(AttributeError):
                freq_table.get_chars(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_count(pos=p, char='AA')
            with self.assertRaises(AttributeError):
                freq_table.get_count_array(pos=p)
            with self.assertRaises(AttributeError):
                freq_table.get_frequency(pos=p, char='AA')
            with self.assertRaises(AttributeError):
                freq_table.get_frequency_array(pos=p)
        with self.assertRaises(AttributeError):
            freq_table.get_count_matrix()
        with self.assertRaises(AttributeError):
            freq_table.get_frequency_matrix()
        freq_table.finalize_table()
        table2 = freq_table.get_table()
        self.assertTrue(isinstance(table2, csc_matrix))
        aln_fn = write_out_temp_fasta(out_str=f'>seq1\n{str(dna_seq1.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table2 = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table2.characterize_alignment(num_aln=num_aln, single_to_pair=dna_single_to_pair)
        expected_table = freq_table2.get_table()
        self.assertFalse((table2 - expected_table).toarray().any())
        for p in positions:
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
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
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
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
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

    def test_to_csv_single_pos_single_update(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        freq_table.to_csv(file_path=csv_path)
        self.assertTrue(os.path.isfile(csv_path))
        df = pd.read_csv(csv_path, sep='\t', header=0, index_col=None)
        self.assertEqual(set(df.columns), {'Position', 'Variability', 'Characters', 'Counts', 'Frequencies'})
        expected_positions = list(range(18))
        for i in df.index:
            self.assertEqual(df.loc[i, 'Position'], expected_positions[i])
            self.assertEqual(df.loc[i, 'Variability'], 1)
            self.assertEqual(df.loc[i, 'Characters'], dna_seq1[i])
            self.assertEqual(df.loc[i, 'Counts'], 1)
            self.assertEqual(df.loc[i, 'Frequencies'], 1.0)
        os.remove(csv_path)

    def test_to_csv_single_pos_multi_update(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table.characterize_alignment(num_aln=num_aln)
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        freq_table.to_csv(file_path=csv_path)
        self.assertTrue(os.path.isfile(csv_path))
        df = pd.read_csv(csv_path, sep='\t', header=0, index_col=None)
        self.assertEqual(set(df.columns), {'Position', 'Variability', 'Characters', 'Counts', 'Frequencies'})
        expected_positions = list(range(18))
        table = freq_table.get_table()
        for i in df.index:
            self.assertEqual(df.loc[i, 'Position'], expected_positions[i])
            chars = df.loc[i, 'Characters'].split(',')
            self.assertEqual(df.loc[i, 'Variability'], len(chars))
            expected_chars = {dna_seq1[i], dna_seq2[i], dna_seq3[i]}
            self.assertEqual(set(chars), expected_chars)
            counts = df.loc[i, 'Counts'].split(',')
            expected_counts = [str(table[i, dna_map[c]]) for c in chars]
            self.assertEqual(counts, expected_counts)
            freqs = df.loc[i, 'Frequencies'].split(',')
            expected_freqs = [str(int(x) / float(freq_table.get_depth())) for x in counts]
            self.assertEqual(freqs, expected_freqs)
        os.remove(csv_path)

    def test_to_csv_pair_pos_single_update(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        freq_table.to_csv(file_path=csv_path)
        self.assertTrue(os.path.isfile(csv_path))
        df = pd.read_csv(csv_path, sep='\t', header=0, index_col=None)
        self.assertEqual(set(df.columns), {'Position', 'Variability', 'Characters', 'Counts', 'Frequencies'})
        ind = 0
        for i in range(18):
            for j in range(i, 18):
                self.assertEqual(df.loc[ind, 'Position'], str((i, j)))
                self.assertEqual(df.loc[ind, 'Variability'], 1)
                self.assertEqual(df.loc[ind, 'Characters'], dna_seq1[i] + dna_seq1[j])
                self.assertEqual(df.loc[ind, 'Counts'], 1)
                self.assertEqual(df.loc[ind, 'Frequencies'], 1.0)
                ind += 1
        os.remove(csv_path)

    def test_to_csv_pair_pos_multi_update(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=dna_single_to_pair)
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        freq_table.to_csv(file_path=csv_path)
        self.assertTrue(os.path.isfile(csv_path))
        df = pd.read_csv(csv_path, sep='\t', header=0, index_col=None)
        self.assertEqual(set(df.columns), {'Position', 'Variability', 'Characters', 'Counts', 'Frequencies'})
        table = freq_table.get_table()
        ind = 0
        for i in range(18):
            for j in range(i, 18):
                self.assertEqual(df.loc[ind, 'Position'], str((i, j)))
                chars = df.loc[ind, 'Characters'].split(',')
                self.assertEqual(df.loc[ind, 'Variability'], len(chars))
                expected_chars = {dna_seq1[i] + dna_seq1[j], dna_seq2[i] + dna_seq2[j], dna_seq3[i] + dna_seq3[j]}
                self.assertEqual(set(chars), expected_chars)
                counts = df.loc[ind, 'Counts'].split(',')
                expected_counts = [str(table[ind, dna_pair_map[c]]) for c in chars]
                self.assertEqual(counts, expected_counts)
                freqs = df.loc[ind, 'Frequencies'].split(',')
                expected_freqs = [str(int(x) / float(freq_table.get_depth())) for x in counts]
                self.assertEqual(freqs, expected_freqs)
                ind += 1
        os.remove(csv_path)

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
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table.characterize_alignment(num_aln=num_aln)
        with self.assertRaises(TypeError):
            freq_table.to_csv(file_path=None)

    def test_load_csv_single_pos_single_update(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        table = freq_table.get_table()
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        freq_table.to_csv(file_path=csv_path)
        freq_table2 = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table2.load_csv(file_path=csv_path)
        table2 = freq_table2.get_table()
        self.assertIsInstance(table, csc_matrix)
        self.assertIsInstance(table2, csc_matrix)
        self.assertFalse((table - table2).toarray().any())
        self.assertEqual(freq_table.get_depth(), freq_table2.get_depth())
        os.remove(csv_path)

    def test_load_csv_single_pos_multi_update(self):
        freq_table = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table.characterize_alignment(num_aln=num_aln)
        table = freq_table.get_table()
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        freq_table.to_csv(file_path=csv_path)
        freq_table2 = FrequencyTable(dna_alpha_size, dna_map, dna_rev, 18, 1)
        freq_table2.load_csv(file_path=csv_path)
        table2 = freq_table2.get_table()
        self.assertIsInstance(table, csc_matrix)
        self.assertIsInstance(table2, csc_matrix)
        self.assertFalse((table - table2).toarray().any())
        self.assertEqual(freq_table.get_depth(), freq_table2.get_depth())
        os.remove(csv_path)

    def test_load_csv_pair_pos_single_update(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table.characterize_sequence(seq=dna_seq1)
        freq_table.finalize_table()
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        freq_table.to_csv(file_path=csv_path)
        table = freq_table.get_table()
        freq_table2 = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table2.load_csv(file_path=csv_path)
        table2 = freq_table2.get_table()
        self.assertIsInstance(table, csc_matrix)
        self.assertIsInstance(table2, csc_matrix)
        self.assertFalse((table - table2).toarray().any())
        self.assertEqual(freq_table.get_depth(), freq_table2.get_depth())
        os.remove(csv_path)

    def test_load_csv_pair_pos_multi_update(self):
        freq_table = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        os.remove(aln_fn)
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=dna_single_to_pair)
        csv_path = os.path.join(os.getcwd(), 'Test_Freq_Table.csv')
        freq_table.to_csv(file_path=csv_path)
        table = freq_table.get_table()
        freq_table2 = FrequencyTable(dna_pair_alpha_size, dna_pair_map, dna_pair_rev, 18, 2)
        freq_table2.load_csv(file_path=csv_path)
        table2 = freq_table2.get_table()
        self.assertIsInstance(table, csc_matrix)
        self.assertIsInstance(table2, csc_matrix)
        self.assertFalse((table - table2).toarray().any())
        self.assertEqual(freq_table.get_depth(), freq_table2.get_depth())
        os.remove(csv_path)

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
        protein_mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                              single_mapping=protein_map, single_reverse_mapping=protein_rev,
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
        protein_mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                              single_mapping=protein_map, single_reverse_mapping=protein_rev,
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
