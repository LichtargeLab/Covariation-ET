"""
Created on July 10, 2019

@author: Daniel Konecki
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csc_matrix
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import Alphabet, Gapped
from Bio.Align import MultipleSeqAlignment
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

    def test1a_init(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        self.assertEqual(freq_table.mapping, self.single_mapping)
        self.assertEqual(freq_table.reverse_mapping, self.single_reverse)
        self.assertEqual(freq_table.position_size, 1)
        self.assertEqual(freq_table.sequence_length, self.query_aln_fa_small.seq_length)
        self.assertEqual(freq_table.num_pos, self.query_aln_fa_small.seq_length)
        self.assertEqual(freq_table.get_depth(), 0)
        expected_table = lil_matrix((self.query_aln_fa_small.seq_length, self.single_size))
        diff = freq_table.get_table() - expected_table
        self.assertFalse(diff.toarray().any())

    def test1b_init(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_large.seq_length)
        self.assertEqual(freq_table.mapping, self.single_mapping)
        self.assertEqual(freq_table.reverse_mapping, self.single_reverse)
        self.assertEqual(freq_table.position_size, 1)
        self.assertEqual(freq_table.sequence_length, self.query_aln_fa_large.seq_length)
        self.assertEqual(freq_table.num_pos, self.query_aln_fa_large.seq_length)
        self.assertEqual(freq_table.get_depth(), 0)
        expected_table = lil_matrix((self.query_aln_fa_large.seq_length, self.single_size))
        diff = freq_table.get_table() - expected_table
        self.assertFalse(diff.toarray().any())

    def test1c_init(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        self.assertEqual(freq_table.mapping, self.pair_mapping)
        self.assertEqual(freq_table.reverse_mapping, self.pair_reverse)
        self.assertEqual(freq_table.position_size, 2)
        self.assertEqual(freq_table.sequence_length, self.query_aln_fa_small.seq_length)
        expected_num_pos = int(np.sum(range(self.query_aln_fa_small.seq_length + 1)))
        self.assertEqual(freq_table.num_pos, expected_num_pos)
        self.assertEqual(freq_table.get_depth(), 0)
        expected_table = lil_matrix((expected_num_pos, self.pair_size))
        diff = freq_table.get_table() - expected_table
        self.assertFalse(diff.toarray().any())

    def test1d_init(self):
        with self.assertRaises(ValueError):
            FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping, reverse_mapping=self.pair_reverse,
                           seq_len=self.query_aln_fa_small.seq_length, pos_size=1)

    def test1e_init(self):
        with self.assertRaises(ValueError):
            FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                           reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)

    # __convert_pos is implicitly tested in almost all other methods.

    def test4a__increment_count(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        expected_table = lil_matrix((self.query_aln_fa_small.seq_length, self.single_size))
        # Testing inserting wrong type of position and character
        with self.assertRaises(TypeError):
            freq_table._increment_count(pos=(1, 2), char='AA')
        with self.assertRaises(KeyError):
            freq_table._increment_count(pos=1, char='AA')
        # Test inserting single correct position and character
        freq_table._increment_count(pos=1, char='A')
        expected_table[1, self.single_mapping['A']] += 1
        diff1 = freq_table.get_table() - expected_table
        self.assertFalse(diff1.toarray().any())
        self.assertEqual(freq_table.get_depth(), 0)
        # Test re-inserting single correct position and character
        freq_table._increment_count(pos=1, char='A')
        expected_table[1, self.single_mapping['A']] += 1
        diff2 = freq_table.get_table() - expected_table
        self.assertFalse(diff2.toarray().any())
        self.assertEqual(freq_table.get_depth(), 0)
        # Test inserting another correct position and character
        freq_table._increment_count(pos=2, char='G')
        expected_table[2, self.single_mapping['G']] += 1
        diff3 = freq_table.get_table() - expected_table
        self.assertFalse(diff3.toarray().any())
        self.assertEqual(freq_table.get_depth(), 0)

    def test4b__increment_count(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        expected_table = lil_matrix((freq_table.num_pos, self.pair_size))
        # Testing inserting wrong type of position and character
        with self.assertRaises(TypeError):
            freq_table._increment_count(pos=1, char='A')
        # Testing inserting wrong size of position and character
        with self.assertRaises(KeyError):
            freq_table._increment_count(pos=(1, 2, 3), char='AAA')
        # Test inserting single correct position and character
        freq_table._increment_count(pos=(1, 2), char='AA')
        expected_table[self.query_aln_fa_small.seq_length + 1, self.pair_mapping['AA']] += 1
        diff1 = freq_table.get_table() - expected_table
        self.assertFalse(diff1.toarray().any())
        self.assertEqual(freq_table.get_depth(), 0)
        # Test re-inserting single correct position and character
        freq_table._increment_count(pos=(1, 2), char='AA')
        expected_table[self.query_aln_fa_small.seq_length + 1, self.pair_mapping['AA']] += 1
        diff2 = freq_table.get_table() - expected_table
        self.assertFalse(diff2.toarray().any())
        self.assertEqual(freq_table.get_depth(), 0)
        # Test inserting another correct position and character
        freq_table._increment_count(pos=(2, 3), char='GG')
        expected_table[self.query_aln_fa_small.seq_length + (self.query_aln_fa_small.seq_length - 1) + 1,
                       self.pair_mapping['GG']] += 1
        diff3 = freq_table.get_table() - expected_table
        self.assertFalse(diff3.toarray().any())
        self.assertEqual(freq_table.get_depth(), 0)

    def test5a_characterize_sequence(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        table = freq_table.get_table()
        for i in range(self.query_aln_fa_small.seq_length):
            expected_char = self.query_aln_fa_small.query_sequence[i]
            self.assertEqual(table[i, self.single_mapping[expected_char]], 1)
        column_sums = np.sum(table, axis=1)
        self.assertFalse((column_sums - 1).any())

    def test5b_characterize_sequence(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        table = freq_table.get_table()
        count = 0
        for i in range(self.query_aln_fa_small.seq_length):
            expected_char_i = self.query_aln_fa_small.query_sequence[i]
            for j in range(i, self.query_aln_fa_small.seq_length):
                expected_char_j = self.query_aln_fa_small.query_sequence[j]
                expected_char = expected_char_i + expected_char_j
                self.assertEqual(table[count, self.pair_mapping[expected_char]], 1)
                count += 1
        column_sums = np.sum(table, axis=1)
        self.assertFalse((column_sums - 1).any())

    def test6a_characterize_alignment(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        sub_aln = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[self.small_structure_id])
        num_aln = sub_aln._alignment_to_num(mapping=self.single_mapping)
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=self.single_to_pair)
        table = freq_table.get_table()
        for i in range(self.query_aln_fa_small.seq_length):
            expected_char = self.query_aln_fa_small.query_sequence[i]
            self.assertEqual(table[i, self.single_mapping[expected_char]], 1)
        column_sums = np.sum(table, axis=1)
        self.assertFalse((column_sums - 1).any())

    def test6b_characterize_alignment(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        sub_aln = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[self.small_structure_id])
        num_aln = sub_aln._alignment_to_num(mapping=self.single_mapping)
        freq_table.characterize_alignment(num_aln=num_aln, single_to_pair=self.single_to_pair)
        table = freq_table.get_table()
        count = 0
        for i in range(self.query_aln_fa_small.seq_length):
            expected_char_i = self.query_aln_fa_small.query_sequence[i]
            for j in range(i, self.query_aln_fa_small.seq_length):
                expected_char_j = self.query_aln_fa_small.query_sequence[j]
                expected_char = expected_char_i + expected_char_j
                self.assertEqual(table[count, self.pair_mapping[expected_char]], 1)
                count += 1
        column_sums = np.sum(table, axis=1)
        self.assertFalse((column_sums - 1).any())

    def test7a_finalize_table(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        self.assertTrue(isinstance(freq_table.get_table(), lil_matrix))
        freq_table.finalize_table()
        self.assertTrue(isinstance(freq_table.get_table(), csc_matrix))

    def test7b_finalize_table(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        self.assertTrue(isinstance(freq_table.get_table(), lil_matrix))
        freq_table.finalize_table()
        self.assertTrue(isinstance(freq_table.get_table(), csc_matrix))

    def test8_get_table(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        expected_table = lil_matrix((freq_table.num_pos, self.single_size))
        table1 = freq_table.get_table()
        diff1 = table1 - expected_table
        self.assertFalse(diff1.toarray().any())
        table2 = freq_table.get_table()
        diff2 = table1 - table2
        self.assertFalse(diff2.toarray().any())
        self.assertIsNot(table1, table2)

    def test9_get_depth(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        depth1 = freq_table.get_depth()
        depth2 = freq_table.get_depth()
        self.assertEqual(depth1, 0)
        self.assertEqual(depth1, depth2)

    def test10a_get_positions(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        self.assertEqual(freq_table.get_positions(), list(range(self.query_aln_fa_small.seq_length)))

    def test10b_get_positions(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        positions = freq_table.get_positions()
        count = 0
        for i in range(self.query_aln_fa_small.seq_length):
            for j in range(i, self.query_aln_fa_small.seq_length):
                expected_position = (i, j)
                self.assertEqual(positions[count], expected_position)
                count += 1

    def test11a_get_chars(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            self.assertEqual(freq_table.get_chars(pos=pos), [self.query_aln_fa_small.query_sequence[pos]])

    def test11b_get_chars(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            self.assertEqual(freq_table.get_chars(pos=pos), [self.query_aln_fa_small.query_sequence[pos[0]] +
                                                             self.query_aln_fa_small.query_sequence[pos[1]]])

    def test12a_get_count(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            curr_char = self.query_aln_fa_small.query_sequence[pos]
            for char in self.single_alphabet.letters:
                if char == curr_char:
                    self.assertEqual(freq_table.get_count(pos=pos, char=char), 1)
                else:
                    self.assertEqual(freq_table.get_count(pos=pos, char=char), 0)

    def test12b_get_count(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            curr_char = self.query_aln_fa_small.query_sequence[pos[0]] + self.query_aln_fa_small.query_sequence[pos[1]]
            for char in self.pair_alphabet.letters:
                if char == curr_char:
                    self.assertEqual(freq_table.get_count(pos=pos, char=char), 1)
                else:
                    self.assertEqual(freq_table.get_count(pos=pos, char=char), 0)

    def test13a_get_count_array(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            self.assertEqual(freq_table.get_count_array(pos=pos), np.array([1]))

    def test13b_get_count_array(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            self.assertEqual(freq_table.get_count_array(pos=pos), np.array([1]))

    def test14a_get_count_matrix(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        expected_mat = np.zeros((self.query_aln_fa_small.seq_length, self.single_size))
        for i in range(self.query_aln_fa_small.seq_length):
            expected_char = self.query_aln_fa_small.query_sequence[i]
            expected_mat[i, self.single_mapping[expected_char]] += 1
        mat = freq_table.get_count_matrix()
        diff = mat - expected_mat
        self.assertTrue(not diff.any())

    def test14b_get_count_matrix(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        expected_mat = np.zeros((np.sum(range(self.query_aln_fa_small.seq_length + 1)), self.pair_size))
        count = 0
        for i in range(self.query_aln_fa_small.seq_length):
            for j in range(i, self.query_aln_fa_small.seq_length):
                expected_char = self.query_aln_fa_small.query_sequence[i] + self.query_aln_fa_small.query_sequence[j]
                expected_mat[count, self.pair_mapping[expected_char]] += 1
                count += 1
        mat = freq_table.get_count_matrix()
        diff = mat - expected_mat
        self.assertTrue(not diff.any())

    def test15a_get_frequency(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            curr_char = self.query_aln_fa_small.query_sequence[pos]
            for char in self.single_alphabet.letters:
                if char == curr_char:
                    self.assertEqual(freq_table.get_frequency(pos=pos, char=char), 1.0)
                else:
                    self.assertEqual(freq_table.get_frequency(pos=pos, char=char), 0.0)

    def test15b_get_frequency(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            curr_char = self.query_aln_fa_small.query_sequence[pos[0]] + self.query_aln_fa_small.query_sequence[pos[1]]
            for char in self.pair_alphabet.letters:
                if char == curr_char:
                    self.assertEqual(freq_table.get_frequency(pos=pos, char=char), 1.0)
                else:
                    self.assertEqual(freq_table.get_frequency(pos=pos, char=char), 0.0)

    def test16a_get_frequency_array(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            self.assertEqual(freq_table.get_frequency_array(pos=pos), np.array([1.0]))

    def test16b_get_frequency_array(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            self.assertEqual(freq_table.get_frequency_array(pos=pos), np.array([1.0]))

    def test17a_get_frequency_matrix(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        expected_mat = np.zeros((self.query_aln_fa_small.seq_length, self.single_size))
        for i in range(self.query_aln_fa_small.seq_length):
            expected_char = self.query_aln_fa_small.query_sequence[i]
            expected_mat[i, self.single_mapping[expected_char]] += 1.0
        mat = freq_table.get_frequency_matrix()
        diff = mat - expected_mat
        self.assertTrue(not diff.any())

    def test17b_get_frequency_matrix(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        expected_mat = np.zeros((np.sum(range(self.query_aln_fa_small.seq_length + 1)), self.pair_size))
        count = 0
        for i in range(self.query_aln_fa_small.seq_length):
            for j in range(i, self.query_aln_fa_small.seq_length):
                expected_char = self.query_aln_fa_small.query_sequence[i] + self.query_aln_fa_small.query_sequence[j]
                expected_mat[count, self.pair_mapping[expected_char]] += 1.0
                count += 1
        mat = freq_table.get_frequency_matrix()
        diff = mat - expected_mat
        self.assertTrue(not diff.any())

    def test18a_to_csv(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        fn = os.path.join(self.testing_dir, 'small_query_seq_freq_table.tsv')
        freq_table.to_csv(file_path=fn)
        loaded_freq_table = pd.read_csv(fn, sep='\t', header=0, index_col=None, keep_default_na=False)
        loaded_freq_table.set_index('Position', inplace=True)
        for i in range(self.query_aln_fa_small.seq_length):
            expected_char = self.query_aln_fa_small.query_sequence[i]
            self.assertEqual(loaded_freq_table.loc[i, 'Variability'], 1)
            self.assertEqual(loaded_freq_table.loc[i, 'Characters'], expected_char)
            self.assertEqual(loaded_freq_table.loc[i, 'Counts'], 1)
            self.assertEqual(loaded_freq_table.loc[i, 'Frequencies'], 1.0)
        os.remove(fn)

    def test18b_to_csv(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_large.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
        fn = os.path.join(self.testing_dir, 'large_query_seq_freq_table.tsv')
        freq_table.to_csv(file_path=fn)
        loaded_freq_table = pd.read_csv(fn, sep='\t', header=0, index_col=None, keep_default_na=False)
        loaded_freq_table.set_index('Position', inplace=True)
        for i in range(self.query_aln_fa_large.seq_length):
            expected_char = self.query_aln_fa_large.query_sequence[i]
            self.assertEqual(loaded_freq_table.loc[i, 'Variability'], 1)
            self.assertEqual(loaded_freq_table.loc[i, 'Characters'], expected_char)
            self.assertEqual(loaded_freq_table.loc[i, 'Counts'], 1)
            self.assertEqual(loaded_freq_table.loc[i, 'Frequencies'], 1.0)
        os.remove(fn)

    def test18c_to_csv(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        fn = os.path.join(self.testing_dir, 'small_query_seq_freq_table.tsv')
        freq_table.to_csv(file_path=fn)
        loaded_freq_table = pd.read_csv(fn, sep='\t', header=0, index_col=None, keep_default_na=False)
        loaded_freq_table.set_index('Position', inplace=True)
        for i in range(self.query_aln_fa_small.seq_length):
            for j in range(i, self.query_aln_fa_small.seq_length):
                expected_char = self.query_aln_fa_small.query_sequence[i] + self.query_aln_fa_small.query_sequence[j]
                self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Variability'], 1)
                self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Characters'], expected_char)
                self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Counts'], 1)
                self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Frequencies'], 1.0)
        os.remove(fn)

    def test18d_to_csv(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_large.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
        fn = os.path.join(self.testing_dir, 'large_query_seq_freq_table.tsv')
        freq_table.to_csv(file_path=fn)
        loaded_freq_table = pd.read_csv(fn, sep='\t', header=0, index_col=None, keep_default_na=False)
        loaded_freq_table.set_index('Position', inplace=True)
        for i in range(self.query_aln_fa_large.seq_length):
            for j in range(i, self.query_aln_fa_large.seq_length):
                expected_char = self.query_aln_fa_large.query_sequence[i] + self.query_aln_fa_large.query_sequence[j]
                self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Variability'], 1)
                self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Characters'], expected_char)
                self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Counts'], 1)
                self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Frequencies'], 1.0)
        os.remove(fn)

    def test19a_load_csv(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        fn = os.path.join(self.testing_dir, 'small_query_seq_freq_table.tsv')
        freq_table.to_csv(file_path=fn)
        loaded_freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                           reverse_mapping=self.single_reverse,
                                           seq_len=self.query_aln_fa_small.seq_length)
        loaded_freq_table.load_csv(fn)
        diff = freq_table.get_table() - loaded_freq_table.get_table()
        self.assertFalse(diff.toarray().any())
        self.assertEqual(freq_table.get_depth(), loaded_freq_table.get_depth())
        os.remove(fn)

    def test19b_load_csv(self):
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_large.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
        fn = os.path.join(self.testing_dir, 'large_query_seq_freq_table.tsv')
        freq_table.to_csv(file_path=fn)
        loaded_freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                           reverse_mapping=self.single_reverse,
                                           seq_len=self.query_aln_fa_large.seq_length)
        loaded_freq_table.load_csv(file_path=fn)
        diff = freq_table.get_table() - loaded_freq_table.get_table()
        self.assertFalse(diff.toarray().any())
        self.assertEqual(freq_table.get_depth(), loaded_freq_table.get_depth())
        os.remove(fn)

    def test19c_load_csv(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        fn = os.path.join(self.testing_dir, 'small_query_seq_freq_table.tsv')
        freq_table.to_csv(file_path=fn)
        loaded_freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                           reverse_mapping=self.pair_reverse,
                                           seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
        loaded_freq_table.load_csv(file_path=fn)
        diff = freq_table.get_table() - loaded_freq_table.get_table()
        self.assertFalse(diff.toarray().any())
        self.assertEqual(freq_table.get_depth(), loaded_freq_table.get_depth())
        os.remove(fn)

    def test19d_load_csv(self):
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_large.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
        fn = os.path.join(self.testing_dir, 'large_query_seq_freq_table.tsv')
        freq_table.to_csv(file_path=fn)
        loaded_freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                           reverse_mapping=self.pair_reverse,
                                           seq_len=self.query_aln_fa_large.seq_length, pos_size=2)
        loaded_freq_table.load_csv(file_path=fn)
        diff = freq_table.get_table() - loaded_freq_table.get_table()
        self.assertFalse(diff.toarray().any())
        self.assertEqual(freq_table.get_depth(), loaded_freq_table.get_depth())
        os.remove(fn)

    def test20a_add(self):
        query_seq_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
        second_index = 0 if query_seq_index != 0 else self.query_aln_fa_small.size - 1
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
        freq_table1 = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                     reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        freq_table1.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        freq_table2 = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                     reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_small.seq_length)
        freq_table2.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
        freq_table_sum1 = freq_table1 + freq_table2
        self.assertEqual(freq_table.mapping, freq_table_sum1.mapping)
        self.assertEqual(freq_table.reverse_mapping, freq_table_sum1.reverse_mapping)
        self.assertEqual(freq_table.num_pos, freq_table_sum1.num_pos)
        self.assertEqual(freq_table.position_size, freq_table_sum1.position_size)
        diff = freq_table.get_table() - freq_table_sum1.get_table()
        self.assertFalse(diff.toarray().any())
        self.assertEqual(freq_table.get_positions(), freq_table_sum1.get_positions())
        self.assertEqual(freq_table.get_depth(), freq_table_sum1.get_depth())

    def test20b_add(self):
        query_seq_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
        second_index = 0 if query_seq_index != 0 else self.query_aln_fa_small.size - 1
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
        freq_table1 = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                     reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                     pos_size=2)
        freq_table1.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        freq_table2 = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                     reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_small.seq_length,
                                     pos_size=2)
        freq_table2.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
        freq_table_sum1 = freq_table1 + freq_table2
        self.assertEqual(freq_table.mapping, freq_table_sum1.mapping)
        self.assertEqual(freq_table.reverse_mapping, freq_table_sum1.reverse_mapping)
        self.assertEqual(freq_table.num_pos, freq_table_sum1.num_pos)
        self.assertEqual(freq_table.position_size, freq_table_sum1.position_size)
        diff = freq_table.get_table() - freq_table_sum1.get_table()
        self.assertFalse(diff.toarray().any())
        self.assertEqual(freq_table.get_positions(), freq_table_sum1.get_positions())
        self.assertEqual(freq_table.get_depth(), freq_table_sum1.get_depth())

    def test20c_add(self):
        query_seq_index = self.query_aln_fa_large.seq_order.index(self.query_aln_fa_large.query_id)
        second_index = 0 if query_seq_index != 0 else self.query_aln_fa_large.size - 1
        freq_table = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                    reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_large.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
        freq_table.characterize_sequence(seq=self.query_aln_fa_large.alignment[second_index].seq)
        freq_table1 = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                     reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_large.seq_length)
        freq_table1.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
        freq_table2 = FrequencyTable(alphabet_size=self.single_size, mapping=self.single_mapping,
                                     reverse_mapping=self.single_reverse, seq_len=self.query_aln_fa_large.seq_length)
        freq_table2.characterize_sequence(seq=self.query_aln_fa_large.alignment[second_index].seq)
        freq_table_sum1 = freq_table1 + freq_table2
        self.assertEqual(freq_table.mapping, freq_table_sum1.mapping)
        self.assertEqual(freq_table.reverse_mapping, freq_table_sum1.reverse_mapping)
        self.assertEqual(freq_table.num_pos, freq_table_sum1.num_pos)
        self.assertEqual(freq_table.position_size, freq_table_sum1.position_size)
        diff = freq_table.get_table() - freq_table_sum1.get_table()
        self.assertFalse(diff.toarray().any())
        self.assertEqual(freq_table.get_positions(), freq_table_sum1.get_positions())
        self.assertEqual(freq_table.get_depth(), freq_table_sum1.get_depth())

    def test20d_add(self):
        query_seq_index = self.query_aln_fa_large.seq_order.index(self.query_aln_fa_large.query_id)
        second_index = 0 if query_seq_index != 0 else self.query_aln_fa_large.size - 1
        freq_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                    reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_large.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
        freq_table.characterize_sequence(seq=self.query_aln_fa_large.alignment[second_index].seq)
        freq_table1 = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                     reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_large.seq_length,
                                     pos_size=2)
        freq_table1.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
        freq_table2 = FrequencyTable(alphabet_size=self.pair_size, mapping=self.pair_mapping,
                                     reverse_mapping=self.pair_reverse, seq_len=self.query_aln_fa_large.seq_length,
                                     pos_size=2)
        freq_table2.characterize_sequence(seq=self.query_aln_fa_large.alignment[second_index].seq)
        freq_table_sum1 = freq_table1 + freq_table2
        self.assertEqual(freq_table.mapping, freq_table_sum1.mapping)
        self.assertEqual(freq_table.reverse_mapping, freq_table_sum1.reverse_mapping)
        self.assertEqual(freq_table.num_pos, freq_table_sum1.num_pos)
        self.assertEqual(freq_table.position_size, freq_table_sum1.position_size)
        diff = freq_table.get_table() - freq_table_sum1.get_table()
        self.assertFalse(diff.toarray().any())
        self.assertEqual(freq_table.get_positions(), freq_table_sum1.get_positions())
        self.assertEqual(freq_table.get_depth(), freq_table_sum1.get_depth())