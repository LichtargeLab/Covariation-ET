"""
Created on July 10, 2019

@author: Daniel Konecki
"""
import os
import unittest
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csc_matrix
from Bio.Alphabet import Gapped
from test_Base import TestBase
from utils import build_mapping
from SeqAlignment import SeqAlignment
from FrequencyTable import FrequencyTable
from EvolutionaryTraceAlphabet import FullIUPACProtein, MultiPositionAlphabet


class TestFrequencyTable(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestFrequencyTable, cls).setUpClass()
        cls.query_aln_fa_small = SeqAlignment(
            file_name=cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln'],
            query_id=cls.small_structure_id)
        cls.query_aln_fa_small.import_alignment()
        cls.query_aln_fa_large = SeqAlignment(
            file_name=cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln'],
            query_id=cls.large_structure_id)
        cls.query_aln_fa_large.import_alignment()
        cls.query_aln_fa_small = cls.query_aln_fa_small.remove_gaps()
        cls.query_aln_fa_large = cls.query_aln_fa_large.remove_gaps()
        cls.single_alphabet = Gapped(FullIUPACProtein())
        cls.single_size, _, cls.single_mapping, cls.single_reverse = build_mapping(alphabet=cls.single_alphabet)
        cls.pair_alphabet = MultiPositionAlphabet(alphabet=cls.single_alphabet, size=2)
        cls.pair_size, _, cls.pair_mapping, cls.pair_reverse = build_mapping(alphabet=cls.pair_alphabet)
        cls.single_to_pair = {}
        for char in cls.pair_mapping:
            key = (cls.single_mapping[char[0]], cls.single_mapping[char[1]])
            cls.single_to_pair[key] = cls.pair_mapping[char]

    def evaluate_init(self, alpha_size, mapping, reverse, seq_len, pos_size):
        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
                                    seq_len=seq_len, pos_size=pos_size)
        self.assertEqual(freq_table.mapping, mapping)
        self.assertEqual(freq_table.reverse_mapping, reverse)
        self.assertEqual(freq_table.position_size, pos_size)
        self.assertEqual(freq_table.sequence_length, seq_len)
        if pos_size == 1:
            expected_num_pos = seq_len
        elif pos_size == 2:
            expected_num_pos = int(((seq_len**2) - seq_len) / 2.0) + seq_len
        else:
            raise ValueError('Only 1 and 2 are supported for pos_size.')
        self.assertEqual(freq_table.num_pos, expected_num_pos)
        self.assertEqual(freq_table.get_depth(), 0)
        expected_dict = {'values': [], 'i': [], 'j': [], 'shape': (expected_num_pos, alpha_size)}
        self.assertEqual(freq_table.get_table(), expected_dict)

    def test1a_init(self):
        self.evaluate_init(alpha_size=self.single_size, mapping=self.single_mapping, reverse=self.single_reverse,
                           seq_len=self.query_aln_fa_small.seq_length, pos_size=1)

    def test1b_init(self):
        self.evaluate_init(alpha_size=self.single_size, mapping=self.single_mapping, reverse=self.single_reverse,
                           seq_len=self.query_aln_fa_large.seq_length, pos_size=1)

    def test1c_init(self):
        self.evaluate_init(alpha_size=self.pair_size, mapping=self.pair_mapping, reverse=self.pair_reverse,
                           seq_len=self.query_aln_fa_small.seq_length, pos_size=2)

    def test1d_init(self):
        with self.assertRaises(ValueError):
            FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping, reverse_mapping=self.pair_reverse,
                           seq_len=self.query_aln_fa_small.seq_length, pos_size=1)

    def test1e_init(self):
        with self.assertRaises(ValueError):
            FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                           reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)

    # __convert_pos is implicitly tested in almost all other methods.

    @staticmethod
    def convert_pos(pos, seq_len):
        if isinstance(pos, int):
            return pos
        else:
            return np.sum([seq_len - x for x in range(pos[0])]) + (pos[1] - pos[0])

    def evaluate__increment_count(self, alpha_size, mapping, reverse, seq_len, pos_size, updates):

        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
                                    seq_len=seq_len, pos_size=pos_size)
        # Testing inserting wrong type of position and character
        if pos_size == 1:
            expected_num_pos = seq_len
            with self.assertRaises(KeyError):
                freq_table._increment_count(pos=1, char='AAAA')
            with self.assertRaises(TypeError):
                freq_table._increment_count(pos=(1, 2), char='AA')
        if pos_size == 2:
            expected_num_pos = int(((seq_len ** 2) - seq_len) / 2.0) + seq_len
            with self.assertRaises(KeyError):
                freq_table._increment_count(pos=(1, 2), char='AAAA')
            with self.assertRaises(TypeError):
                freq_table._increment_count(pos=1, char='A')
            with self.assertRaises(TypeError):
                freq_table._increment_count(pos=(1, 2, 3), char='AAA')
        expected_table = lil_matrix((expected_num_pos, alpha_size))
        # Test inserting single correct position and character
        freq_table._increment_count(pos=updates[0][0], char=updates[0][1])
        expected_table[self.convert_pos(updates[0][0], seq_len), mapping[updates[0][1]]] += 1
        tuple1 = freq_table.get_table()
        table1 = csc_matrix((tuple1['values'], (tuple1['i'], tuple1['j'])), shape=tuple1['shape'])
        diff1 = table1 - expected_table
        self.assertFalse(diff1.toarray().any())
        self.assertEqual(freq_table.get_depth(), 0)
        # Test re-inserting single correct position and character
        freq_table._increment_count(pos=updates[1][0], char=updates[1][1])
        expected_table[self.convert_pos(updates[1][0], seq_len), mapping[updates[1][1]]] += 1
        tuple2 = freq_table.get_table()
        table2 = csc_matrix((tuple2['values'], (tuple2['i'], tuple2['j'])), shape=tuple2['shape'])
        diff2 = table2 - expected_table
        self.assertFalse(diff2.toarray().any())
        self.assertEqual(freq_table.get_depth(), 0)
        # Test inserting another correct position and character
        freq_table._increment_count(pos=updates[2][0], char=updates[2][1])
        expected_table[self.convert_pos(updates[2][0], seq_len), mapping[updates[2][1]]] += 1
        tuple3 = freq_table.get_table()
        table3 = csc_matrix((tuple3['values'], (tuple3['i'], tuple3['j'])), shape=tuple3['shape'])
        diff3 = table3 - expected_table
        self.assertFalse(diff3.toarray().any())
        self.assertEqual(freq_table.get_depth(), 0)

    def test2a__increment_count(self):
        self.evaluate__increment_count(alpha_size=self.single_size, mapping=self.single_mapping,
                                       reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                       pos_size=1, updates=[(1, 'A'), (1, 'A'), (2, 'G')])

    def test2b__increment_count(self):
        self.evaluate__increment_count(alpha_size=self.pair_size, mapping=self.pair_mapping, reverse=self.pair_reverse,
                                       seq_len=self.query_aln_fa_small.seq_length, pos_size=2,
                                       updates=[((1, 2), 'AA'), ((1, 2), 'AA'), ((2, 3), 'GG')])

    def evaluate_characterize_sequence(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
                                    seq_len=seq_len, pos_size=pos_size)
        freq_table.characterize_sequence(seq=sequence)
        freq_table.finalize_table()
        for i in range(seq_len):
            if pos_size == 1:
                expected_char = sequence[i]
                self.assertEqual(freq_table.get_count(pos=i, char=expected_char), 1)
                continue
            for j in range(i, seq_len):
                expected_char = sequence[i] + sequence[j]
                self.assertEqual(freq_table.get_count(pos=(i, j), char=expected_char), 1)
        table = freq_table.get_table()
        column_sums = np.sum(table, axis=1)
        self.assertFalse((column_sums - 1).any())

    def test3a_characterize_sequence(self):
        self.evaluate_characterize_sequence(alpha_size=self.single_size, mapping=self.single_mapping,
                                            reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                            pos_size=1, sequence=self.query_aln_fa_small.query_sequence)

    def test3b_characterize_sequence(self):
        self.evaluate_characterize_sequence(alpha_size=self.pair_size, mapping=self.pair_mapping,
                                            reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                            pos_size=2, sequence=self.query_aln_fa_small.query_sequence)

    def evalaute_characterize_alignment(self, alpha_size, mapping, reverse, seq_len, pos_size, num_aln, single_to_pair,
                                        sequence):
        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse, seq_len=seq_len,
                                    pos_size=pos_size)
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=single_to_pair)
        for i in range(seq_len):
            if pos_size == 1:
                expected_char = sequence[i]
                self.assertEqual(freq_table.get_count(pos=i, char=expected_char), 1)
                continue
            for j in range(i, seq_len):
                expected_char = sequence[i] + sequence[j]
                self.assertEqual(freq_table.get_count(pos=(i, j), char=expected_char), 1)
        table = freq_table.get_table()
        column_sums = np.sum(table, axis=1)
        self.assertFalse((column_sums - 1).any())

    def test4a_characterize_alignment(self):
        sub_aln = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[self.small_structure_id])
        num_aln = sub_aln._alignment_to_num(mapping=self.single_mapping)
        self.evalaute_characterize_alignment(alpha_size=self.single_size,mapping=self.single_mapping,
                                             reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                             pos_size=1, num_aln=num_aln, single_to_pair=None,
                                             sequence=self.query_aln_fa_small.query_sequence)

    def test4b_characterize_alignment(self):
        sub_aln = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[self.small_structure_id])
        num_aln = sub_aln._alignment_to_num(mapping=self.single_mapping)
        self.evalaute_characterize_alignment(alpha_size=self.pair_size, mapping=self.pair_mapping,
                                             reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                             pos_size=2, num_aln=num_aln, single_to_pair=self.single_to_pair,
                                             sequence=self.query_aln_fa_small.query_sequence)

    def evaluate_finalize_table(self, alpha_size, mapping, reverse, seq_len, pos_size):
        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
                                    seq_len=seq_len, pos_size=pos_size)
        self.assertTrue(isinstance(freq_table.get_table(), dict))
        freq_table.finalize_table()
        self.assertTrue(isinstance(freq_table.get_table(), csc_matrix))

    def test5a_finalize_table(self):
        self.evaluate_finalize_table(alpha_size=self.single_size, mapping=self.single_mapping,
                                     reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                     pos_size=1)

    def test5b_finalize_table(self):
        self.evaluate_finalize_table(alpha_size=self.pair_size, mapping=self.pair_mapping,
                                     reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                     pos_size=2)

    def test6_get_table(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        expected_dict = {'values': [], 'i': [], 'j': [], 'shape': (freq_table.num_pos, self.single_size)}
        table1 = freq_table.get_table()
        self.assertEqual(table1, expected_dict)
        table2 = freq_table.get_table()
        self.assertEqual(table2, expected_dict)
        self.assertIsNot(table1, table2)
        freq_table.finalize_table()
        expected_table = lil_matrix((freq_table.num_pos, self.single_size))
        table3 = freq_table.get_table()
        diff = table3 - expected_table
        self.assertFalse(diff.toarray().any())

    def test7_set_depth(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        depth1 = freq_table.get_depth()
        self.assertEqual(depth1, 0)
        freq_table.set_depth(1)
        depth2 = freq_table.get_depth()
        self.assertNotEqual(depth1, depth2)
        self.assertEqual(depth2, 1)

    def test8_get_depth(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        depth1 = freq_table.get_depth()
        depth2 = freq_table.get_depth()
        self.assertEqual(depth1, 0)
        self.assertEqual(depth1, depth2)

    def evalaute_get_positions(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
                                    seq_len=seq_len, pos_size=pos_size)
        freq_table.characterize_sequence(sequence)
        freq_table.finalize_table()
        positions = freq_table.get_positions()
        count = 0
        for i in range(seq_len):
            if pos_size == 1:
                self.assertEqual(i, positions[count])
                count += 1
                continue
            for j in range(i, seq_len):
                self.assertEqual((i, j), positions[count])
                count += 1

    def test9a_get_positions(self):
        self.evalaute_get_positions(alpha_size=self.single_size, mapping=self.single_mapping,
                                    reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=1, sequence=self.query_aln_fa_small.query_sequence)

    def test9b_get_positions(self):
        self.evalaute_get_positions(alpha_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2, sequence=self.query_aln_fa_small.query_sequence)

    def evaluate_get_chars(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse, seq_len=seq_len,
                                    pos_size=pos_size)
        freq_table.characterize_sequence(seq=sequence)
        freq_table.finalize_table()
        for pos in freq_table.get_positions():
            if pos_size == 1:
                self.assertEqual(freq_table.get_chars(pos=pos), [sequence[pos]])
            elif pos_size == 2:
                self.assertEqual(freq_table.get_chars(pos=pos), [sequence[pos[0]] + sequence[pos[1]]])
            else:
                raise ValueError('1 and 2 are the only supported values for pos_size.')

    def test10a_get_chars(self):
        self.evaluate_get_chars(alpha_size=self.single_size, mapping=self.single_mapping,
                                reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                pos_size=1, sequence=self.query_aln_fa_small.query_sequence)

    def test10b_get_chars(self):
        self.evaluate_get_chars(alpha_size=self.pair_size, mapping=self.pair_mapping,
                                reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                pos_size=2, sequence=self.query_aln_fa_small.query_sequence)

    def evaluate_get_count(self, alphabet, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping,reverse_mapping=reverse,
                                    seq_len=seq_len, pos_size=pos_size)
        freq_table.characterize_sequence(seq=sequence)
        freq_table.finalize_table()
        for pos in freq_table.get_positions():
            if pos_size == 1:
                curr_char = sequence[pos]
            elif pos_size == 2:
                curr_char = sequence[pos[0]] + sequence[pos[1]]
            else:
                raise ValueError('1 or 2 are the only options for pos_size.')
            for char in alphabet.letters:
                if char == curr_char:
                    self.assertEqual(freq_table.get_count(pos=pos, char=char), 1)
                else:
                    self.assertEqual(freq_table.get_count(pos=pos, char=char), 0)

    def test11a_get_count(self):
        self.evaluate_get_count(alphabet=self.single_alphabet, alpha_size=self.single_size, mapping=self.single_mapping,
                                reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length, pos_size=1,
                                sequence=self.query_aln_fa_small.query_sequence)

    def test11b_get_count(self):
        self.evaluate_get_count(alphabet=self.pair_alphabet, alpha_size=self.pair_size, mapping=self.pair_mapping,
                                reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length, pos_size=2,
                                sequence=self.query_aln_fa_small.query_sequence)

    def evaluate_get_count_array(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
                                    seq_len=seq_len, pos_size=pos_size)
        freq_table.characterize_sequence(seq=sequence)
        freq_table.finalize_table()
        for pos in freq_table.get_positions():
            self.assertEqual(freq_table.get_count_array(pos=pos), np.array([1]))

    def test12a_get_count_array(self):
        self.evaluate_get_count_array(alpha_size=self.single_size, mapping=self.single_mapping,
                                      reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                      pos_size=1, sequence=self.query_aln_fa_small.query_sequence)

    def test12b_get_count_array(self):
        self.evaluate_get_count_array(alpha_size=self.pair_size, mapping=self.pair_mapping,
                                      reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                      pos_size=2, sequence=self.query_aln_fa_small.query_sequence)

    def evaluate_get_count_matrix(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse, seq_len=seq_len,
                                    pos_size=pos_size)
        freq_table.characterize_sequence(seq=sequence)
        freq_table.finalize_table()
        if pos_size == 1:
            expected_num_pos = seq_len
        elif pos_size == 2:
            expected_num_pos = int(((seq_len**2) - seq_len) / 2.0) + seq_len
        else:
            raise ValueError('Only 1 and 2 are supported for pos_size.')
        expected_mat = np.zeros((expected_num_pos, alpha_size))
        count = 0
        for i in range(seq_len):
            if pos_size == 1:
                expected_char = sequence[i]
                expected_mat[count, mapping[expected_char]] += 1
                count += 1
                continue
            for j in range(i, seq_len):
                expected_char = sequence[i] + sequence[j]
                expected_mat[count, mapping[expected_char]] += 1
                count += 1
        mat = freq_table.get_count_matrix()
        diff = mat - expected_mat
        self.assertTrue(not diff.any())

    def test13a_get_count_matrix(self):
        self.evaluate_get_count_matrix(alpha_size=self.single_size, mapping=self.single_mapping,
                                      reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                      pos_size=1, sequence=self.query_aln_fa_small.query_sequence)

    def test13b_get_count_matrix(self):
        self.evaluate_get_count_array(alpha_size=self.pair_size, mapping=self.pair_mapping,
                                      reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                      pos_size=2, sequence=self.query_aln_fa_small.query_sequence)

    def evaluate_get_frequency(self, alphabet, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse, seq_len=seq_len,
                                    pos_size=pos_size)
        freq_table.characterize_sequence(seq=sequence)
        freq_table.finalize_table()
        for pos in freq_table.get_positions():
            if pos_size == 1:
                curr_char = sequence[pos]
            elif pos_size == 2:
                curr_char = sequence[pos[0]] + sequence[pos[1]]
            else:
                raise ValueError('1 and 2 are the only accepted values for pos_size.')
            for char in alphabet.letters:
                if char == curr_char:
                    self.assertEqual(freq_table.get_frequency(pos=pos, char=char), 1.0)
                else:
                    self.assertEqual(freq_table.get_frequency(pos=pos, char=char), 0.0)

    def test14a_get_frequency(self):
        self.evaluate_get_frequency(alphabet=self.single_alphabet, alpha_size=self.single_size,
                                    mapping=self.single_mapping, reverse=self.single_reverse,
                                    seq_len=self.query_aln_fa_small.seq_length, pos_size=1,
                                    sequence=self.query_aln_fa_small.query_sequence)

    def test14b_get_frequency(self):
        self.evaluate_get_frequency(alphabet=self.pair_alphabet, alpha_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length, pos_size=2,
                                    sequence=self.query_aln_fa_small.query_sequence)

    def evaluate_get_frequency_array(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
                                    seq_len=seq_len, pos_size=pos_size)
        freq_table.characterize_sequence(seq=sequence)
        freq_table.finalize_table()
        for pos in freq_table.get_positions():
            self.assertEqual(freq_table.get_frequency_array(pos=pos), np.array([1.0]))

    def test15a_get_frequency_array(self):
        self.evaluate_get_frequency_array(alpha_size=self.single_size, mapping=self.single_mapping,
                                          reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                          pos_size=1, sequence=self.query_aln_fa_small.query_sequence)

    def test15b_get_frequency_array(self):
        self.evaluate_get_frequency_array(alpha_size=self.pair_size, mapping=self.pair_mapping,
                                          reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                          pos_size=2, sequence=self.query_aln_fa_small.query_sequence)

    def evaluate_get_frequency_matrix(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence):
        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
                                    seq_len=seq_len, pos_size=pos_size)
        freq_table.characterize_sequence(seq=sequence)
        freq_table.finalize_table()
        if pos_size == 1:
            expected_num_pos = seq_len
        elif pos_size == 2:
            expected_num_pos = int(((seq_len**2) - seq_len) / 2.0) + seq_len
        else:
            raise ValueError('Only 1 and 2 are supported for pos_size.')
        expected_mat = np.zeros((expected_num_pos, alpha_size))
        count = 0
        for i in range(seq_len):
            if pos_size == 1:
                expected_char = sequence[i]
                expected_mat[count, mapping[expected_char]] += 1.0
                count += 1
                continue
            for j in range(i, seq_len):
                expected_char = sequence[i] + sequence[j]
                expected_mat[count, mapping[expected_char]] += 1.0
                count += 1
        mat = freq_table.get_frequency_matrix()
        diff = mat - expected_mat
        self.assertTrue(not diff.any())

    def test16a_get_frequency_matrix(self):
        self.evaluate_get_frequency_matrix(alpha_size=self.single_size, mapping=self.single_mapping,
                                           reverse=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                           pos_size=1, sequence=self.query_aln_fa_small.query_sequence)

    def test16b_get_frequency_matrix(self):
        self.evaluate_get_frequency_matrix(alpha_size=self.pair_size, mapping=self.pair_mapping,
                                           reverse=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                           pos_size=2, sequence=self.query_aln_fa_small.query_sequence)

    def evaluate_to_csv(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence, fn):
        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse, seq_len=seq_len,
                                    pos_size=pos_size)
        freq_table.characterize_sequence(seq=sequence)
        freq_table.finalize_table()
        fn = os.path.join(self.testing_dir, fn)
        freq_table.to_csv(file_path=fn)
        loaded_freq_table = pd.read_csv(fn, sep='\t', header=0, index_col=None, keep_default_na=False)
        loaded_freq_table.set_index('Position', inplace=True)
        for i in range(seq_len):
            if pos_size == 1:
                expected_char = sequence[i]
                self.assertEqual(loaded_freq_table.loc[i, 'Variability'], 1)
                self.assertEqual(loaded_freq_table.loc[i, 'Characters'], expected_char)
                self.assertEqual(loaded_freq_table.loc[i, 'Counts'], 1)
                self.assertEqual(loaded_freq_table.loc[i, 'Frequencies'], 1.0)
                continue
            for j in range(i, seq_len):
                expected_char = sequence[i] + sequence[j]
                self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Variability'], 1)
                self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Characters'], expected_char)
                self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Counts'], 1)
                self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Frequencies'], 1.0)
        os.remove(fn)

    def test17a_to_csv(self):
        self.evaluate_to_csv(alpha_size=self.single_size, mapping=self.single_mapping, reverse=self.single_reverse,
                             seq_len=self.query_aln_fa_small.seq_length, pos_size=1,
                             sequence=self.query_aln_fa_small.query_sequence, fn='small_query_seq_freq_table.tsv')

    def test17b_to_csv(self):
        self.evaluate_to_csv(alpha_size=self.single_size, mapping=self.single_mapping, reverse=self.single_reverse,
                             seq_len=self.query_aln_fa_large.seq_length, pos_size=1,
                             sequence=self.query_aln_fa_large.query_sequence, fn='large_query_seq_freq_table.tsv')

    def test17c_to_csv(self):
        self.evaluate_to_csv(alpha_size=self.pair_size, mapping=self.pair_mapping, reverse=self.pair_reverse,
                             seq_len=self.query_aln_fa_small.seq_length, pos_size=2,
                             sequence=self.query_aln_fa_small.query_sequence, fn='small_query_seq_freq_table.tsv')

    def test17d_to_csv(self):
        self.evaluate_to_csv(alpha_size=self.pair_size, mapping=self.pair_mapping, reverse=self.pair_reverse,
                             seq_len=self.query_aln_fa_large.seq_length, pos_size=2,
                             sequence=self.query_aln_fa_large.query_sequence, fn='large_query_seq_freq_table.tsv')

    def evaluate_load_csv(self, alpha_size, mapping, reverse, seq_len, pos_size, sequence, fn):
        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
                                    seq_len=seq_len, pos_size=pos_size)
        freq_table.characterize_sequence(seq=sequence)
        freq_table.finalize_table()
        fn = os.path.join(self.testing_dir, fn)
        freq_table.to_csv(file_path=fn)
        loaded_freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
                                           seq_len=seq_len, pos_size=pos_size)
        loaded_freq_table.load_csv(fn)
        diff = freq_table.get_table() - loaded_freq_table.get_table()
        self.assertFalse(diff.toarray().any())
        self.assertEqual(freq_table.get_depth(), loaded_freq_table.get_depth())
        os.remove(fn)

    def test18a_load_csv(self):
        self.evaluate_load_csv(alpha_size=self.single_size, mapping=self.single_mapping, reverse=self.single_reverse,
                               seq_len=self.query_aln_fa_small.seq_length, pos_size=1,
                               sequence=self.query_aln_fa_small.query_sequence, fn='small_query_seq_freq_table.tsv')

    def test18b_load_csv(self):
        self.evaluate_load_csv(alpha_size=self.single_size, mapping=self.single_mapping, reverse=self.single_reverse,
                               seq_len=self.query_aln_fa_large.seq_length, pos_size=1,
                               sequence=self.query_aln_fa_large.query_sequence, fn='large_query_seq_freq_table.tsv')

    def test18c_load_csv(self):
        self.evaluate_load_csv(alpha_size=self.pair_size, mapping=self.pair_mapping, reverse=self.pair_reverse,
                               seq_len=self.query_aln_fa_small.seq_length, pos_size=2,
                               sequence=self.query_aln_fa_small.query_sequence, fn='small_query_seq_freq_table.tsv')

    def test18d_load_csv(self):
        self.evaluate_load_csv(alpha_size=self.pair_size, mapping=self.pair_mapping, reverse=self.pair_reverse,
                               seq_len=self.query_aln_fa_large.seq_length, pos_size=2,
                               sequence=self.query_aln_fa_large.query_sequence, fn='large_query_seq_freq_table.tsv')

    def evaluate_add(self, aln, alpha_size, mapping, reverse, pos_size):
        query_seq_index = aln.seq_order.index(aln.query_id)
        second_index = 0 if query_seq_index != 0 else aln.size - 1
        freq_table = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
                                    seq_len=aln.seq_length, pos_size=pos_size)
        freq_table.characterize_sequence(seq=aln.query_sequence)
        freq_table.characterize_sequence(seq=aln.alignment[second_index].seq)
        freq_table.finalize_table()
        freq_table1 = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
                                     seq_len=aln.seq_length, pos_size=pos_size)
        freq_table1.characterize_sequence(seq=aln.query_sequence)
        freq_table2 = FrequencyTable(alphabet_size=alpha_size, mapping=mapping, reverse_mapping=reverse,
                                     seq_len=aln.seq_length, pos_size=pos_size)
        freq_table2.characterize_sequence(seq=aln.alignment[second_index].seq)
        freq_table1.finalize_table()
        freq_table2.finalize_table()
        freq_table_sum1 = freq_table1 + freq_table2
        self.assertEqual(freq_table.mapping, freq_table_sum1.mapping)
        self.assertEqual(freq_table.reverse_mapping, freq_table_sum1.reverse_mapping)
        self.assertEqual(freq_table.num_pos, freq_table_sum1.num_pos)
        self.assertEqual(freq_table.position_size, freq_table_sum1.position_size)
        diff = freq_table.get_table() - freq_table_sum1.get_table()
        self.assertFalse(diff.toarray().any())
        self.assertEqual(freq_table.get_positions(), freq_table_sum1.get_positions())
        self.assertEqual(freq_table.get_depth(), freq_table_sum1.get_depth())

    def test19a_add(self):
        self.evaluate_add(aln=self.query_aln_fa_small, alpha_size=self.single_size, mapping=self.single_mapping,
                          reverse=self.single_reverse, pos_size=1)

    def test19b_add(self):
        self.evaluate_add(aln=self.query_aln_fa_small, alpha_size=self.pair_size, mapping=self.pair_mapping,
                          reverse=self.pair_reverse, pos_size=2)

    def test19c_add(self):
        self.evaluate_add(aln=self.query_aln_fa_large, alpha_size=self.single_size, mapping=self.single_mapping,
                          reverse=self.single_reverse, pos_size=1)

    def test19d_add(self):
        self.evaluate_add(aln=self.query_aln_fa_large, alpha_size=self.pair_size, mapping=self.pair_mapping,
                          reverse=self.pair_reverse, pos_size=2)


if __name__ == '__main__':
    unittest.main()
