"""
Created on July 10, 2019

@author: Daniel Konecki
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
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

    # def test1a_init(self):
    #     freq_table = FrequencyTable(alphabet=Gapped(FullIUPACProtein()), seq_len=self.query_aln_fa_small.seq_length)
    #     self.assertTrue(isinstance(freq_table.alphabet, Gapped))
    #     for char in FullIUPACProtein().letters + '-':
    #         self.assertTrue(char in freq_table.alphabet.letters)
    #     self.assertEqual(freq_table.position_size, 1)
    #     self.assertEqual(freq_table.sequence_length, self.query_aln_fa_small.seq_length)
    #     self.assertTrue(freq_table.get_table() == {})
    #     self.assertEqual(freq_table.get_depth(), 0)

    def test1a_init(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        self.assertEqual(freq_table.mapping, mapping)
        self.assertEqual(freq_table.position_size, 1)
        self.assertEqual(freq_table.sequence_length, self.query_aln_fa_small.seq_length)
        self.assertEqual(freq_table.num_pos, self.query_aln_fa_small.seq_length)
        self.assertEqual(freq_table.get_depth(), 0)
        expected_table = lil_matrix((self.query_aln_fa_small.seq_length, a_size))
        diff = freq_table.get_table() - expected_table
        self.assertFalse(diff.todense().any())

    # def test1b_init(self):
    #     freq_table = FrequencyTable(alphabet=FullIUPACProtein(), seq_len=self.query_aln_fa_small.seq_length)
    #     self.assertTrue(isinstance(freq_table.alphabet, Alphabet))
    #     for char in FullIUPACProtein().letters:
    #         self.assertTrue(char in freq_table.alphabet.letters)
    #     self.assertFalse('-' in freq_table.alphabet.letters)
    #     self.assertEqual(freq_table.position_size, 1)
    #     self.assertEqual(freq_table.sequence_length, self.query_aln_fa_small.seq_length)
    #     self.assertTrue(freq_table.get_table() == {})
    #     self.assertEqual(freq_table.get_depth(), 0)

    def test1b_init(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_large.seq_length)
        self.assertEqual(freq_table.mapping, mapping)
        self.assertEqual(freq_table.position_size, 1)
        self.assertEqual(freq_table.sequence_length, self.query_aln_fa_large.seq_length)
        self.assertEqual(freq_table.num_pos, self.query_aln_fa_large.seq_length)
        self.assertEqual(freq_table.get_depth(), 0)
        expected_table = lil_matrix((self.query_aln_fa_large.seq_length, a_size))
        diff = freq_table.get_table() - expected_table
        self.assertFalse(diff.todense().any())

    # def test1c_init(self):
    #     freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2),
    #                                 seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     self.assertTrue(isinstance(freq_table.alphabet, Alphabet))
    #     for char1 in FullIUPACProtein().letters:
    #         for char2 in FullIUPACProtein().letters:
    #             self.assertTrue('{}{}'.format(char1, char2) in freq_table.alphabet.letters)
    #     self.assertFalse('--' in freq_table.alphabet.letters)
    #     self.assertEqual(freq_table.position_size, 2)
    #     self.assertEqual(freq_table.sequence_length, self.query_aln_fa_small.seq_length)
    #     self.assertTrue(freq_table.get_table() == {})
    #     self.assertEqual(freq_table.get_depth(), 0)

    def test1c_init(self):
        alpha = MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2)
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        self.assertEqual(freq_table.mapping, mapping)
        self.assertEqual(freq_table.position_size, 2)
        self.assertEqual(freq_table.sequence_length, self.query_aln_fa_small.seq_length)
        expected_num_pos = int(np.sum(range(self.query_aln_fa_small.seq_length + 1)))
        self.assertEqual(freq_table.num_pos, expected_num_pos)
        self.assertEqual(freq_table.get_depth(), 0)
        expected_table = lil_matrix((expected_num_pos, a_size))
        diff = freq_table.get_table() - expected_table
        self.assertFalse(diff.todense().any())

    # def test1d_init(self):
    #     with self.assertRaises(ValueError):
    #         freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2),
    #                                     seq_len=self.query_aln_fa_small.seq_length, pos_size=1)

    def test1d_init(self):
        with self.assertRaises(ValueError):
            alpha = MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2)
            a_size, _, mapping = build_mapping(alphabet=alpha)
            freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping,
                                        seq_len=self.query_aln_fa_small.seq_length, pos_size=1)

    # def test1e_init(self):
    #     with self.assertRaises(ValueError):
    #         freq_table = FrequencyTable(alphabet=Gapped(FullIUPACProtein()), size=2),
    #                                     seq_len=self.query_aln_fa_small.seq_length, pos_size=1)

    def test1e_init(self):
        with self.assertRaises(ValueError):
            alpha = Gapped(FullIUPACProtein())
            a_size, _, mapping = build_mapping(alphabet=alpha)
            freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
                                        pos_size=2)

    # def test2a__add_position(self):
    #     freq_table = FrequencyTable(alphabet=FullIUPACProtein(), seq_len=self.query_aln_fa_small.seq_length)
    #     # Testing inserting wrong type of position
    #     with self.assertRaises(TypeError):
    #         freq_table._add_position(pos=(1, 2))
    #     # Test inserting single correct position
    #     freq_table._add_position(1)
    #     self.assertEqual(freq_table.get_table(), {1: {}})
    #     # Test re-inserting single correct position
    #     freq_table._add_position(1)
    #     self.assertEqual(freq_table.get_table(), {1: {}})
    #     # Test inserting another correct position
    #     freq_table._add_position(2)
    #     self.assertEqual(freq_table.get_table(), {1: {}, 2: {}})

    # def test2b__add_position(self):
    #     freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2),
    #                                 seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     # Testing inserting wrong type of position
    #     with self.assertRaises(TypeError):
    #         freq_table._add_position(pos=1)
    #     # Testing inserting wrong size of position
    #     with self.assertRaises(TypeError):
    #         freq_table._add_position(pos=(1, 2, 3))
    #     # Test inserting single correct position
    #     freq_table._add_position((1, 2))
    #     self.assertEqual(freq_table.get_table(), {(1, 2): {}})
    #     # Test re-inserting single correct position
    #     freq_table._add_position((1, 2))
    #     self.assertEqual(freq_table.get_table(), {(1, 2): {}})
    #     # Test inserting another correct position
    #     freq_table._add_position((2, 3))
    #     self.assertEqual(freq_table.get_table(), {(1, 2): {}, (2, 3): {}})

    # def test3a__add_pos_char(self):
    #     freq_table = FrequencyTable(alphabet=FullIUPACProtein(), seq_len=self.query_aln_fa_small.seq_length)
    #     # Testing inserting wrong type of position and character
    #     with self.assertRaises(ValueError):
    #         freq_table._add_pos_char(pos=(1, 2), char='AA')
    #     # Test inserting single correct position and character
    #     freq_table._add_pos_char(pos=1, char='A')
    #     self.assertEqual(freq_table.get_table(), {1: {'A': {'count': 0}}})
    #     # Test re-inserting single correct position and character
    #     freq_table._add_pos_char(pos=1, char='A')
    #     self.assertEqual(freq_table.get_table(), {1: {'A': {'count': 0}}})
    #     # Test inserting another correct position and character
    #     freq_table._add_pos_char(pos=2, char='G')
    #     self.assertEqual(freq_table.get_table(), {1: {'A': {'count': 0}}, 2: {'G': {'count': 0}}})

    # def test3b__add_pos_char(self):
    #     freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2),
    #                                 seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     # Testing inserting wrong type of position and character
    #     with self.assertRaises(ValueError):
    #         freq_table._add_pos_char(pos=1, char='A')
    #     # Testing inserting wrong size of position and character
    #     with self.assertRaises(ValueError):
    #         freq_table._add_pos_char(pos=(1, 2, 3), char='AAA')
    #     # Test inserting single correct position and character
    #     freq_table._add_pos_char(pos=(1, 2), char='AA')
    #     self.assertEqual(freq_table.get_table(), {(1, 2): {'AA': {'count': 0}}})
    #     # Test re-inserting single correct position and character
    #     freq_table._add_pos_char(pos=(1, 2), char='AA')
    #     self.assertEqual(freq_table.get_table(), {(1, 2): {'AA': {'count': 0}}})
    #     # Test inserting another correct position and character
    #     freq_table._add_pos_char(pos=(2, 3), char='GG')
    #     self.assertEqual(freq_table.get_table(), {(1, 2): {'AA': {'count': 0}}, (2, 3): {'GG': {'count': 0}}})

    # def test4a__increment_count(self):
    #     freq_table = FrequencyTable(alphabet=FullIUPACProtein(), seq_len=self.query_aln_fa_small.seq_length)
    #     # Testing inserting wrong type of position and character
    #     with self.assertRaises(ValueError):
    #         freq_table._increment_count(pos=(1, 2), char='AA')
    #     # Test inserting single correct position and character
    #     freq_table._increment_count(pos=1, char='A')
    #     self.assertEqual(freq_table.get_table(), {1: {'A': {'count': 1}}})
    #     self.assertEqual(freq_table.get_depth(), 0)
    #     # Test re-inserting single correct position and character
    #     freq_table._increment_count(pos=1, char='A')
    #     self.assertEqual(freq_table.get_table(), {1: {'A': {'count': 2}}})
    #     self.assertEqual(freq_table.get_depth(), 0)
    #     # Test inserting another correct position and character
    #     freq_table._increment_count(pos=2, char='G')
    #     self.assertEqual(freq_table.get_table(), {1: {'A': {'count': 2}}, 2: {'G': {'count': 1}}})
    #     self.assertEqual(freq_table.get_depth(), 0)

    def test4a__increment_count(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        expected_table = lil_matrix((self.query_aln_fa_small.seq_length, a_size))
        # Testing inserting wrong type of position and character
        with self.assertRaises(KeyError):
            freq_table._increment_count(pos=(1, 2), char='AA')
        # Test inserting single correct position and character
        freq_table._increment_count(pos=1, char='A')
        expected_table[1, mapping['A']] += 1
        diff1 = freq_table.get_table() - expected_table
        self.assertFalse(diff1.todense().any())
        self.assertEqual(freq_table.get_depth(), 0)
        # Test re-inserting single correct position and character
        freq_table._increment_count(pos=1, char='A')
        expected_table[1, mapping['A']] += 1
        diff2 = freq_table.get_table() - expected_table
        self.assertFalse(diff2.todense().any())
        self.assertEqual(freq_table.get_depth(), 0)
        # Test inserting another correct position and character
        freq_table._increment_count(pos=2, char='G')
        expected_table[2, mapping['G']] += 1
        diff3 = freq_table.get_table() - expected_table
        self.assertFalse(diff3.todense().any())
        self.assertEqual(freq_table.get_depth(), 0)

    # def test4b__increment_count(self):
    #     freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2),
    #                                 seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     # Testing inserting wrong type of position and character
    #     with self.assertRaises(ValueError):
    #         freq_table._increment_count(pos=1, char='A')
    #     # Testing inserting wrong size of position and character
    #     with self.assertRaises(ValueError):
    #         freq_table._increment_count(pos=(1, 2, 3), char='AAA')
    #     # Test inserting single correct position and character
    #     freq_table._increment_count(pos=(1, 2), char='AA')
    #     self.assertEqual(freq_table.get_table(), {(1, 2): {'AA': {'count': 1}}})
    #     self.assertEqual(freq_table.get_depth(), 0)
    #     # Test re-inserting single correct position and character
    #     freq_table._increment_count(pos=(1, 2), char='AA')
    #     self.assertEqual(freq_table.get_table(), {(1, 2): {'AA': {'count': 2}}})
    #     self.assertEqual(freq_table.get_depth(), 0)
    #     # Test inserting another correct position and character
    #     freq_table._increment_count(pos=(2, 3), char='GG')
    #     self.assertEqual(freq_table.get_table(), {(1, 2): {'AA': {'count': 2}}, (2, 3): {'GG': {'count': 1}}})
    #     self.assertEqual(freq_table.get_depth(), 0)

    def test4b__increment_count(self):
        alpha = MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2)
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping,
                                    seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
        expected_table = lil_matrix((freq_table.num_pos, a_size))
        # Testing inserting wrong type of position and character
        with self.assertRaises(TypeError):
            freq_table._increment_count(pos=1, char='A')
        # Testing inserting wrong size of position and character
        with self.assertRaises(KeyError):
            freq_table._increment_count(pos=(1, 2, 3), char='AAA')
        # Test inserting single correct position and character
        freq_table._increment_count(pos=(1, 2), char='AA')
        expected_table[self.query_aln_fa_small.seq_length + 2, mapping['AA']] += 1
        diff1 = freq_table.get_table() - expected_table
        self.assertFalse(diff1.todense().any())
        self.assertEqual(freq_table.get_depth(), 0)
        # Test re-inserting single correct position and character
        freq_table._increment_count(pos=(1, 2), char='AA')
        expected_table[self.query_aln_fa_small.seq_length + 2, mapping['AA']] += 1
        diff2 = freq_table.get_table() - expected_table
        self.assertFalse(diff2.todense().any())
        self.assertEqual(freq_table.get_depth(), 0)
        # Test inserting another correct position and character
        freq_table._increment_count(pos=(2, 3), char='GG')
        expected_table[self.query_aln_fa_small.seq_length + (self.query_aln_fa_small.seq_length - 1) + 3,
                       mapping['GG']] += 1
        diff3 = freq_table.get_table() - expected_table
        self.assertFalse(diff3.todense().any())
        self.assertEqual(freq_table.get_depth(), 0)

    # def test5a_charatcerize_sequence(self):
    #     freq_table = FrequencyTable(alphabet=Gapped(FullIUPACProtein()), seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     table = freq_table.get_table()
    #     for i in range(self.query_aln_fa_small.seq_length):
    #         self.assertTrue(i in table)
    #         self.assertEqual(len(table[i]), 1)
    #         expected_char = self.query_aln_fa_small.query_sequence[i]
    #         self.assertTrue(expected_char in table[i])
    #         self.assertTrue('count' in table[i][expected_char])
    #         self.assertEqual(table[i][expected_char]['count'], 1)
    #         self.assertFalse('frequency' in table[i][expected_char])
    #     self.assertEqual(len(table), self.query_aln_fa_small.seq_length)

    def test5a_characterize_sequence(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        table = freq_table.get_table()
        for i in range(self.query_aln_fa_small.seq_length):
            expected_char = self.query_aln_fa_small.query_sequence[i]
            self.assertEqual(table[i, mapping[expected_char]], 1)
        column_sums = np.sum(table, axis=1)
        self.assertFalse((column_sums - 1).any())

    # def test5b_characterize_sequence(self):
    #     freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2),
    #                                 seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     table = freq_table.get_table()
    #     expected_size = 0
    #     for i in range(self.query_aln_fa_small.seq_length):
    #         # for j in range(i + 1, self.query_aln_fa_small.seq_length):
    #         for j in range(i, self.query_aln_fa_small.seq_length):
    #             self.assertTrue((i, j) in table)
    #             self.assertEqual(len(table[(i, j)]), 1)
    #             expected_char = self.query_aln_fa_small.query_sequence[i] + self.query_aln_fa_small.query_sequence[j]
    #             self.assertTrue(expected_char in table[(i, j)])
    #             self.assertTrue('count' in table[(i, j)][expected_char])
    #             self.assertEqual(table[(i, j)][expected_char]['count'], 1)
    #             self.assertFalse('frequency' in table[(i, j)][expected_char])
    #             expected_size += 1
    #     self.assertEqual(len(table), expected_size)

    def test5b_characterize_sequence(self):
        alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        table = freq_table.get_table()
        count = 0
        for i in range(self.query_aln_fa_small.seq_length):
            expected_char_i = self.query_aln_fa_small.query_sequence[i]
            # for j in range(i + 1, self.query_aln_fa_small.seq_length):
            for j in range(i, self.query_aln_fa_small.seq_length):
                expected_char_j = self.query_aln_fa_small.query_sequence[j]
                expected_char = expected_char_i + expected_char_j
                self.assertEqual(table[count, mapping[expected_char]], 1)
                count += 1
        column_sums = np.sum(table, axis=1)
        self.assertFalse((column_sums - 1).any())

    # def test6_get_table(self):
    #     freq_table = FrequencyTable(alphabet=FullIUPACProtein(), seq_len=self.query_aln_fa_small.seq_length)
    #     table1 = freq_table.get_table()
    #     table2 = freq_table.get_table()
    #     self.assertEqual(table1, table2)
    #     self.assertIsNot(table1, table2)

    def test6_get_table(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        expected_table = lil_matrix((freq_table.num_pos, a_size))
        table1 = freq_table.get_table()
        diff1 = table1 - expected_table
        self.assertFalse(diff1.todense().any())
        table2 = freq_table.get_table()
        diff2 = table1 - table2
        self.assertFalse(diff2.todense().any())
        self.assertIsNot(table1, table2)

    # def test7_get_depth(self):
    #     freq_table = FrequencyTable(alphabet=FullIUPACProtein(), seq_len=self.query_aln_fa_small.seq_length)
    #     depth1 = freq_table.get_depth()
    #     depth2 = freq_table.get_depth()
    #     self.assertEqual(depth1, depth2)

    def test7_get_depth(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        depth1 = freq_table.get_depth()
        depth2 = freq_table.get_depth()
        self.assertEqual(depth1, 0)
        self.assertEqual(depth1, depth2)

    # def test8a_get_positions(self):
    #     freq_table = FrequencyTable(alphabet=Gapped(FullIUPACProtein()), seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     self.assertEqual(freq_table.get_positions(), list(range(self.query_aln_fa_small.seq_length)))

    def test8a_get_positions(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        self.assertEqual(freq_table.get_positions(), list(range(self.query_aln_fa_small.seq_length)))

    # def test8b_get_positions(self):
    #     freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2),
    #                                 seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     positions = freq_table.get_positions()
    #     count = 0
    #     for i in range(self.query_aln_fa_small.seq_length):
    #         # for j in range(i + 1, self.query_aln_fa_small.seq_length):
    #         for j in range(i, self.query_aln_fa_small.seq_length):
    #             expected_position = (i, j)
    #             self.assertEqual(positions[count], expected_position)
    #             count += 1

    def test8b_get_positions(self):
        alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        positions = freq_table.get_positions()
        count = 0
        for i in range(self.query_aln_fa_small.seq_length):
            for j in range(i, self.query_aln_fa_small.seq_length):
                expected_position = (i, j)
                self.assertEqual(positions[count], expected_position)
                count += 1

    # def test9a_get_chars(self):
    #     freq_table = FrequencyTable(alphabet=Gapped(FullIUPACProtein()), seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     for pos in freq_table.get_positions():
    #         self.assertEqual(freq_table.get_chars(pos=pos), [self.query_aln_fa_small.query_sequence[pos]])

    def test9a_get_chars(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            self.assertEqual(freq_table.get_chars(pos=pos), [self.query_aln_fa_small.query_sequence[pos]])

    # def test9b_get_chars(self):
    #     freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2),
    #                                 seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     for pos in freq_table.get_positions():
    #         self.assertEqual(freq_table.get_chars(pos=pos), [self.query_aln_fa_small.query_sequence[pos[0]] +
    #                                                          self.query_aln_fa_small.query_sequence[pos[1]]])

    def test9b_get_chars(self):
        alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            self.assertEqual(freq_table.get_chars(pos=pos), [self.query_aln_fa_small.query_sequence[pos[0]] +
                                                             self.query_aln_fa_small.query_sequence[pos[1]]])

    # def test10a_get_count(self):
    #     freq_table = FrequencyTable(alphabet=Gapped(FullIUPACProtein()), seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     for pos in freq_table.get_positions():
    #         curr_char = self.query_aln_fa_small.query_sequence[pos]
    #         for char in freq_table.alphabet.letters:
    #             if char == curr_char:
    #                 self.assertEqual(freq_table.get_count(pos=pos, char=char), 1)
    #             else:
    #                 self.assertEqual(freq_table.get_count(pos=pos, char=char), 0)

    def test10a_get_count(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            curr_char = self.query_aln_fa_small.query_sequence[pos]
            for char in alpha.letters:
                if char == curr_char:
                    self.assertEqual(freq_table.get_count(pos=pos, char=char), 1)
                else:
                    self.assertEqual(freq_table.get_count(pos=pos, char=char), 0)

    # def test10b_get_count(self):
    #     freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2),
    #                                 seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     for pos in freq_table.get_positions():
    #         curr_char = self.query_aln_fa_small.query_sequence[pos[0]] + self.query_aln_fa_small.query_sequence[pos[1]]
    #         for char in freq_table.alphabet.letters:
    #             if char == curr_char:
    #                 self.assertEqual(freq_table.get_count(pos=pos, char=char), 1)
    #             else:
    #                 self.assertEqual(freq_table.get_count(pos=pos, char=char), 0)

    def test10b_get_count(self):
        alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            curr_char = self.query_aln_fa_small.query_sequence[pos[0]] + self.query_aln_fa_small.query_sequence[pos[1]]
            for char in alpha.letters:
                if char == curr_char:
                    self.assertEqual(freq_table.get_count(pos=pos, char=char), 1)
                else:
                    self.assertEqual(freq_table.get_count(pos=pos, char=char), 0)

    # def test11a_get_count_array(self):
    #     freq_table = FrequencyTable(alphabet=Gapped(FullIUPACProtein()), seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     for pos in freq_table.get_positions():
    #         self.assertEqual(freq_table.get_count_array(pos=pos), np.array([1]))

    def test11a_get_count_array(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            self.assertEqual(freq_table.get_count_array(pos=pos), np.array([1]))

    # def test11b_get_count_array(self):
    #     freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2),
    #                                 seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     for pos in freq_table.get_positions():
    #         self.assertEqual(freq_table.get_count_array(pos=pos), np.array([1]))

    def test11b_get_count_array(self):
        alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
        a_size, _, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            self.assertEqual(freq_table.get_count_array(pos=pos), np.array([1]))

    # def test12a_get_count_matrix(self):
    #     alpha = Gapped(FullIUPACProtein())
    #     alpha_size, gap_chars, mapping = build_mapping(alphabet=alpha)
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     expected_mat = np.zeros((self.query_aln_fa_small.seq_length, alpha_size))
    #     for i in range(self.query_aln_fa_small.seq_length):
    #         expected_char = self.query_aln_fa_small.query_sequence[i]
    #         expected_mat[i, mapping[expected_char]] += 1
    #     mat = freq_table.get_count_matrix()
    #     diff = mat - expected_mat
    #     self.assertTrue(not diff.any())

    def test12a_get_count_matrix(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        expected_mat = np.zeros((self.query_aln_fa_small.seq_length, a_size))
        for i in range(self.query_aln_fa_small.seq_length):
            expected_char = self.query_aln_fa_small.query_sequence[i]
            expected_mat[i, mapping[expected_char]] += 1
        mat = freq_table.get_count_matrix()
        diff = mat - expected_mat
        self.assertTrue(not diff.any())

    # def test12b_get_count_matrix(self):
    #     alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
    #     alpha_size, gap_chars, mapping = build_mapping(alphabet=alpha)
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     expected_mat = np.zeros((np.sum(range(self.query_aln_fa_small.seq_length + 1)), alpha_size))
    #     count = 0
    #     for i in range(self.query_aln_fa_small.seq_length):
    #         # for j in range(i + 1, self.query_aln_fa_small.seq_length):
    #         for j in range(i, self.query_aln_fa_small.seq_length):
    #             expected_char = self.query_aln_fa_small.query_sequence[i] + self.query_aln_fa_small.query_sequence[j]
    #             expected_mat[count, mapping[expected_char]] += 1
    #             count += 1
    #     mat = freq_table.get_count_matrix()
    #     diff = mat - expected_mat
    #     self.assertTrue(not diff.any())

    def test12b_get_count_matrix(self):
        alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        expected_mat = np.zeros((np.sum(range(self.query_aln_fa_small.seq_length + 1)), a_size))
        count = 0
        for i in range(self.query_aln_fa_small.seq_length):
            for j in range(i, self.query_aln_fa_small.seq_length):
                expected_char = self.query_aln_fa_small.query_sequence[i] + self.query_aln_fa_small.query_sequence[j]
                expected_mat[count, mapping[expected_char]] += 1
                count += 1
        mat = freq_table.get_count_matrix()
        diff = mat - expected_mat
        self.assertTrue(not diff.any())

    # def test13a_compute_frequencies(self):
    #     freq_table = FrequencyTable(alphabet=Gapped(FullIUPACProtein()), seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.compute_frequencies()
    #     table = freq_table.get_table()
    #     for i in range(self.query_aln_fa_small.seq_length):
    #         self.assertTrue(i in table)
    #         self.assertEqual(len(table[i]), 1)
    #         expected_char = self.query_aln_fa_small.query_sequence[i]
    #         self.assertTrue(expected_char in table[i])
    #         self.assertTrue('count' in table[i][expected_char])
    #         self.assertEqual(table[i][expected_char]['count'], 1)
    #         self.assertTrue('frequency' in table[i][expected_char])
    #         self.assertEqual(table[i][expected_char]['frequency'], 1.0)
    #     self.assertEqual(len(table), self.query_aln_fa_small.seq_length)
    #
    # def test13b_compute_frequencies(self):
    #     freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2),
    #                                 seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.compute_frequencies()
    #     table = freq_table.get_table()
    #     expected_size = 0
    #     for i in range(self.query_aln_fa_small.seq_length):
    #         # for j in range(i + 1, self.query_aln_fa_small.seq_length):
    #         for j in range(i, self.query_aln_fa_small.seq_length):
    #             self.assertTrue((i, j) in table)
    #             self.assertEqual(len(table[(i, j)]), 1)
    #             expected_char = self.query_aln_fa_small.query_sequence[i] + self.query_aln_fa_small.query_sequence[j]
    #             self.assertTrue(expected_char in table[(i, j)])
    #             self.assertTrue('count' in table[(i, j)][expected_char])
    #             self.assertEqual(table[(i, j)][expected_char]['count'], 1)
    #             self.assertTrue('frequency' in table[(i, j)][expected_char])
    #             self.assertEqual(table[(i, j)][expected_char]['frequency'], 1.0)
    #             expected_size += 1
    #     self.assertEqual(len(table), expected_size)

    # def test14a_get_frequency(self):
    #     freq_table = FrequencyTable(alphabet=Gapped(FullIUPACProtein()), seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.compute_frequencies()
    #     for pos in freq_table.get_positions():
    #         curr_char = self.query_aln_fa_small.query_sequence[pos]
    #         for char in freq_table.alphabet.letters:
    #             if char == curr_char:
    #                 self.assertEqual(freq_table.get_frequency(pos=pos, char=char), 1.0)
    #             else:
    #                 self.assertEqual(freq_table.get_frequency(pos=pos, char=char), 0.0)

    def test14a_get_frequency(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            curr_char = self.query_aln_fa_small.query_sequence[pos]
            for char in alpha.letters:
                if char == curr_char:
                    self.assertEqual(freq_table.get_frequency(pos=pos, char=char), 1.0)
                else:
                    self.assertEqual(freq_table.get_frequency(pos=pos, char=char), 0.0)

    # def test14b_get_frequency(self):
    #     freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2),
    #                                 seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.compute_frequencies()
    #     for pos in freq_table.get_positions():
    #         curr_char = self.query_aln_fa_small.query_sequence[pos[0]] + self.query_aln_fa_small.query_sequence[pos[1]]
    #         for char in freq_table.alphabet.letters:
    #             if char == curr_char:
    #                 self.assertEqual(freq_table.get_frequency(pos=pos, char=char), 1.0)
    #             else:
    #                 self.assertEqual(freq_table.get_frequency(pos=pos, char=char), 0.0)

    def test14b_get_frequency(self):
        alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            curr_char = self.query_aln_fa_small.query_sequence[pos[0]] + self.query_aln_fa_small.query_sequence[pos[1]]
            for char in alpha.letters:
                if char == curr_char:
                    self.assertEqual(freq_table.get_frequency(pos=pos, char=char), 1.0)
                else:
                    self.assertEqual(freq_table.get_frequency(pos=pos, char=char), 0.0)

    # def test15a_get_frequency_array(self):
    #     freq_table = FrequencyTable(alphabet=Gapped(FullIUPACProtein()), seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.compute_frequencies()
    #     for pos in freq_table.get_positions():
    #         self.assertEqual(freq_table.get_frequency_array(pos=pos), np.array([1.0]))

    def test15a_get_frequency_array(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            self.assertEqual(freq_table.get_frequency_array(pos=pos), np.array([1.0]))

    # def test15b_get_frequency_array(self):
    #     freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2),
    #                                 seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.compute_frequencies()
    #     for pos in freq_table.get_positions():
    #         self.assertEqual(freq_table.get_frequency_array(pos=pos), np.array([1.0]))

    def test15b_get_frequency_array(self):
        alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        for pos in freq_table.get_positions():
            self.assertEqual(freq_table.get_frequency_array(pos=pos), np.array([1.0]))

    # def test16a_get_frequency_matrix(self):
    #     alpha = Gapped(FullIUPACProtein())
    #     alpha_size, gap_chars, mapping = build_mapping(alphabet=alpha)
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.compute_frequencies()
    #     expected_mat = np.zeros((self.query_aln_fa_small.seq_length, alpha_size))
    #     for i in range(self.query_aln_fa_small.seq_length):
    #         expected_char = self.query_aln_fa_small.query_sequence[i]
    #         expected_mat[i, mapping[expected_char]] += 1.0
    #     mat = freq_table.get_frequency_matrix()
    #     diff = mat - expected_mat
    #     self.assertTrue(not diff.any())

    def test16a_get_frequency_matrix(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        expected_mat = np.zeros((self.query_aln_fa_small.seq_length, a_size))
        for i in range(self.query_aln_fa_small.seq_length):
            expected_char = self.query_aln_fa_small.query_sequence[i]
            expected_mat[i, mapping[expected_char]] += 1.0
        mat = freq_table.get_frequency_matrix()
        diff = mat - expected_mat
        self.assertTrue(not diff.any())

    # def test16b_get_frequency_matrix(self):
    #     alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
    #     alpha_size, gap_chars, mapping = build_mapping(alphabet=alpha)
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.compute_frequencies()
    #     expected_mat = np.zeros((np.sum(range(self.query_aln_fa_small.seq_length + 1)), alpha_size))
    #     count = 0
    #     for i in range(self.query_aln_fa_small.seq_length):
    #         # for j in range(i + 1, self.query_aln_fa_small.seq_length):
    #         for j in range(i, self.query_aln_fa_small.seq_length):
    #             expected_char = self.query_aln_fa_small.query_sequence[i] + self.query_aln_fa_small.query_sequence[j]
    #             expected_mat[count, mapping[expected_char]] += 1.0
    #             count += 1
    #     mat = freq_table.get_frequency_matrix()
    #     diff = mat - expected_mat
    #     self.assertTrue(not diff.any())

    def test16b_get_frequency_matrix(self):
        alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        expected_mat = np.zeros((np.sum(range(self.query_aln_fa_small.seq_length + 1)), a_size))
        count = 0
        for i in range(self.query_aln_fa_small.seq_length):
            for j in range(i, self.query_aln_fa_small.seq_length):
                expected_char = self.query_aln_fa_small.query_sequence[i] + self.query_aln_fa_small.query_sequence[j]
                expected_mat[count, mapping[expected_char]] += 1.0
                count += 1
        mat = freq_table.get_frequency_matrix()
        diff = mat - expected_mat
        self.assertTrue(not diff.any())

    # def test17a_to_csv(self):
    #     alpha = Gapped(FullIUPACProtein())
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.compute_frequencies()
    #     fn = os.path.join(self.testing_dir, 'small_query_seq_freq_table.tsv')
    #     freq_table.to_csv(file_path=fn)
    #     loaded_freq_table = pd.read_csv(fn, sep='\t', header=0, index_col=None, keep_default_na=False)
    #     loaded_freq_table.set_index('Position', inplace=True)
    #     for i in range(self.query_aln_fa_small.seq_length):
    #         expected_char = self.query_aln_fa_small.query_sequence[i]
    #         self.assertEqual(loaded_freq_table.loc[i, 'Variability'], 1)
    #         self.assertEqual(loaded_freq_table.loc[i, 'Characters'], expected_char)
    #         self.assertEqual(loaded_freq_table.loc[i, 'Counts'], 1)
    #         self.assertEqual(loaded_freq_table.loc[i, 'Frequencies'], 1.0)
    #     os.remove(fn)

    def test17a_to_csv(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
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

    # def test17b_to_csv(self):
    #     alpha = Gapped(FullIUPACProtein())
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_large.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
    #     freq_table.compute_frequencies()
    #     fn = os.path.join(self.testing_dir, 'large_query_seq_freq_table.tsv')
    #     freq_table.to_csv(file_path=fn)
    #     loaded_freq_table = pd.read_csv(fn, sep='\t', header=0, index_col=None, keep_default_na=False)
    #     loaded_freq_table.set_index('Position', inplace=True)
    #     for i in range(self.query_aln_fa_large.seq_length):
    #         expected_char = self.query_aln_fa_large.query_sequence[i]
    #         self.assertEqual(loaded_freq_table.loc[i, 'Variability'], 1)
    #         self.assertEqual(loaded_freq_table.loc[i, 'Characters'], expected_char)
    #         self.assertEqual(loaded_freq_table.loc[i, 'Counts'], 1)
    #         self.assertEqual(loaded_freq_table.loc[i, 'Frequencies'], 1.0)
    #     os.remove(fn)

    def test17b_to_csv(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_large.seq_length)
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

    # def test17c_to_csv(self):
    #     alpha = MultiPositionAlphabet(Gapped(FullIUPACProtein()), size=2)
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.compute_frequencies()
    #     fn = os.path.join(self.testing_dir, 'small_query_seq_freq_table.tsv')
    #     freq_table.to_csv(file_path=fn)
    #     loaded_freq_table = pd.read_csv(fn, sep='\t', header=0, index_col=None, keep_default_na=False)
    #     loaded_freq_table.set_index('Position', inplace=True)
    #     for i in range(self.query_aln_fa_small.seq_length):
    #         for j in range(i, self.query_aln_fa_small.seq_length):
    #             expected_char = self.query_aln_fa_small.query_sequence[i] + self.query_aln_fa_small.query_sequence[j]
    #             self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Variability'], 1)
    #             self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Characters'], expected_char)
    #             self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Counts'], 1)
    #             self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Frequencies'], 1.0)
    #     os.remove(fn)

    def test17c_to_csv(self):
        alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
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

    # def test17d_to_csv(self):
    #     alpha = MultiPositionAlphabet(Gapped(FullIUPACProtein()), size=2)
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_large.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
    #     freq_table.compute_frequencies()
    #     fn = os.path.join(self.testing_dir, 'large_query_seq_freq_table.tsv')
    #     freq_table.to_csv(file_path=fn)
    #     loaded_freq_table = pd.read_csv(fn, sep='\t', header=0, index_col=None, keep_default_na=False)
    #     loaded_freq_table.set_index('Position', inplace=True)
    #     for i in range(self.query_aln_fa_large.seq_length):
    #         for j in range(i, self.query_aln_fa_large.seq_length):
    #             expected_char = self.query_aln_fa_large.query_sequence[i] + self.query_aln_fa_large.query_sequence[j]
    #             self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Variability'], 1)
    #             self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Characters'], expected_char)
    #             self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Counts'], 1)
    #             self.assertEqual(loaded_freq_table.loc[str((i, j)), 'Frequencies'], 1.0)
    #     os.remove(fn)

    def test17d_to_csv(self):
        alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_large.seq_length,
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

    # def test18a_load_csv(self):
    #     alpha = Gapped(FullIUPACProtein())
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.compute_frequencies()
    #     fn = os.path.join(self.testing_dir, 'small_query_seq_freq_table.tsv')
    #     freq_table.to_csv(file_path=fn)
    #     loaded_freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     loaded_freq_table.load_csv(fn)
    #     self.assertEqual(freq_table.get_table(), loaded_freq_table.get_table())
    #     self.assertEqual(freq_table.get_depth(), loaded_freq_table.get_depth())
    #     os.remove(fn)

    def test18a_load_csv(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        fn = os.path.join(self.testing_dir, 'small_query_seq_freq_table.tsv')
        freq_table.to_csv(file_path=fn)
        loaded_freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping,
                                           seq_len=self.query_aln_fa_small.seq_length)
        loaded_freq_table.load_csv(fn)
        self.assertEqual(freq_table.get_table(), loaded_freq_table.get_table())
        self.assertEqual(freq_table.get_depth(), loaded_freq_table.get_depth())
        os.remove(fn)

    # def test18b_load_csv(self):
    #     alpha = Gapped(FullIUPACProtein())
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_large.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
    #     freq_table.compute_frequencies()
    #     fn = os.path.join(self.testing_dir, 'large_query_seq_freq_table.tsv')
    #     freq_table.to_csv(file_path=fn)
    #     loaded_freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_large.seq_length)
    #     loaded_freq_table.load_csv(file_path=fn)
    #     self.assertEqual(freq_table.get_table(), loaded_freq_table.get_table())
    #     self.assertEqual(freq_table.get_depth(), loaded_freq_table.get_depth())
    #     os.remove(fn)

    def test18b_load_csv(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_large.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
        fn = os.path.join(self.testing_dir, 'large_query_seq_freq_table.tsv')
        freq_table.to_csv(file_path=fn)
        loaded_freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping,
                                           seq_len=self.query_aln_fa_large.seq_length)
        loaded_freq_table.load_csv(file_path=fn)
        self.assertEqual(freq_table.get_table(), loaded_freq_table.get_table())
        self.assertEqual(freq_table.get_depth(), loaded_freq_table.get_depth())
        os.remove(fn)

    # def test18c_load_csv(self):
    #     alpha = MultiPositionAlphabet(Gapped(FullIUPACProtein()), size=2)
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.compute_frequencies()
    #     fn = os.path.join(self.testing_dir, 'small_query_seq_freq_table.tsv')
    #     freq_table.to_csv(file_path=fn)
    #     loaded_freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     loaded_freq_table.load_csv(file_path=fn)
    #     self.assertEqual(freq_table.get_table(), loaded_freq_table.get_table())
    #     self.assertEqual(freq_table.get_depth(), loaded_freq_table.get_depth())
    #     os.remove(fn)

    def test18c_load_csv(self):
        alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        fn = os.path.join(self.testing_dir, 'small_query_seq_freq_table.tsv')
        freq_table.to_csv(file_path=fn)
        loaded_freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping,
                                           seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
        loaded_freq_table.load_csv(file_path=fn)
        self.assertEqual(freq_table.get_table(), loaded_freq_table.get_table())
        self.assertEqual(freq_table.get_depth(), loaded_freq_table.get_depth())
        os.remove(fn)

    # def test18d_load_csv(self):
    #     alpha = MultiPositionAlphabet(Gapped(FullIUPACProtein()), size=2)
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_large.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
    #     freq_table.compute_frequencies()
    #     fn = os.path.join(self.testing_dir, 'large_query_seq_freq_table.tsv')
    #     freq_table.to_csv(file_path=fn)
    #     loaded_freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_large.seq_length, pos_size=2)
    #     loaded_freq_table.load_csv(file_path=fn)
    #     self.assertEqual(freq_table.get_table(), loaded_freq_table.get_table())
    #     self.assertEqual(freq_table.get_depth(), loaded_freq_table.get_depth())
    #     os.remove(fn)

    def test18d_load_csv(self):
        alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_large.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
        fn = os.path.join(self.testing_dir, 'large_query_seq_freq_table.tsv')
        freq_table.to_csv(file_path=fn)
        loaded_freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping,
                                           seq_len=self.query_aln_fa_large.seq_length, pos_size=2)
        loaded_freq_table.load_csv(file_path=fn)
        self.assertEqual(freq_table.get_table(), loaded_freq_table.get_table())
        self.assertEqual(freq_table.get_depth(), loaded_freq_table.get_depth())
        os.remove(fn)

    # def test19a_add(self):
    #     alpha = Gapped(FullIUPACProtein())
    #     query_seq_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
    #     second_index = 0 if query_seq_index != 0 else self.query_aln_fa_small.size - 1
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table1 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table1.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table2 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table2.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table_sum1 = freq_table1 + freq_table2
    #     self.assertTrue(isinstance(freq_table.alphabet, type(freq_table_sum1.alphabet)))
    #     self.assertEqual(len(freq_table.alphabet.letters), len(freq_table_sum1.alphabet.letters))
    #     for char in freq_table.alphabet.letters:
    #         self.assertTrue(char in freq_table_sum1.alphabet.letters)
    #     self.assertEqual(freq_table.position_size, freq_table_sum1.position_size)
    #     self.assertEqual(freq_table.get_table(), freq_table_sum1.get_table())
    #     self.assertEqual(freq_table.get_positions(), freq_table_sum1.get_positions())
    #     for i in freq_table.get_positions():
    #         self.assertEqual(freq_table.get_chars(pos=i), freq_table_sum1.get_chars(pos=i))
    #         for c in freq_table.get_chars(pos=i):
    #             self.assertEqual(freq_table.get_count(pos=i, char=c), freq_table_sum1.get_count(pos=i, char=c))

    def test19a_add(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        query_seq_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
        second_index = 0 if query_seq_index != 0 else self.query_aln_fa_small.size - 1
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
        freq_table1 = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        freq_table1.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        freq_table2 = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length)
        freq_table2.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
        freq_table_sum1 = freq_table1 + freq_table2
        self.assertEqual(freq_table.mapping, freq_table_sum1.mapping)
        self.assertEqual(freq_table.reverse_mapping, freq_table_sum1.reverse_mapping)
        self.assertEqual(freq_table.num_pos, freq_table_sum1.num_pos)
        self.assertEqual(freq_table.position_size, freq_table_sum1.position_size)
        self.assertEqual(freq_table.get_table(), freq_table_sum1.get_table())
        self.assertEqual(freq_table.get_positions(), freq_table_sum1.get_positions())
        self.assertEqual(freq_table.get_depth(), freq_table_sum1.get_depth())

    # def test19b_add(self):
    #     alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
    #     query_seq_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
    #     second_index = 0 if query_seq_index != 0 else self.query_aln_fa_small.size - 1
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table1 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table1.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table2 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table2.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table_sum1 = freq_table1 + freq_table2
    #     self.assertTrue(isinstance(freq_table.alphabet, type(freq_table_sum1.alphabet)))
    #     self.assertEqual(len(freq_table.alphabet.letters), len(freq_table_sum1.alphabet.letters))
    #     for char in freq_table.alphabet.letters:
    #         self.assertTrue(char in freq_table_sum1.alphabet.letters)
    #     self.assertEqual(freq_table.position_size, freq_table_sum1.position_size)
    #     self.assertEqual(freq_table.get_table(), freq_table_sum1.get_table())
    #     self.assertEqual(freq_table.get_positions(), freq_table_sum1.get_positions())
    #     for i in freq_table.get_positions():
    #         self.assertEqual(freq_table.get_chars(pos=i), freq_table_sum1.get_chars(pos=i))
    #         for c in freq_table.get_chars(pos=i):
    #             self.assertEqual(freq_table.get_count(pos=i, char=c), freq_table_sum1.get_count(pos=i, char=c))

    def test19b_add(self):
        alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        query_seq_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
        second_index = 0 if query_seq_index != 0 else self.query_aln_fa_small.size - 1
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        freq_table.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
        freq_table1 = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
                                     pos_size=2)
        freq_table1.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
        freq_table2 = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_small.seq_length,
                                     pos_size=2)
        freq_table2.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
        freq_table_sum1 = freq_table1 + freq_table2
        self.assertEqual(freq_table.mapping, freq_table_sum1.mapping)
        self.assertEqual(freq_table.reverse_mapping, freq_table_sum1.reverse_mapping)
        self.assertEqual(freq_table.num_pos, freq_table_sum1.num_pos)
        self.assertEqual(freq_table.position_size, freq_table_sum1.position_size)
        self.assertEqual(freq_table.get_table(), freq_table_sum1.get_table())
        self.assertEqual(freq_table.get_positions(), freq_table_sum1.get_positions())
        self.assertEqual(freq_table.get_depth(), freq_table_sum1.get_depth())

    # def test19c_add(self):
    #     # Ensure that if frequencies have been computed for the first table, the merged table has no frequencies.
    #     alpha = Gapped(FullIUPACProtein())
    #     query_seq_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
    #     second_index = 0 if query_seq_index != 0 else self.query_aln_fa_small.size - 1
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table1 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table1.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table1.compute_frequencies()
    #     freq_table2 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table2.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table_sum1 = freq_table1 + freq_table2
    #     # Since frequencies were not compute for the original frequency table checking that the underlying dictionaries
    #     # are equal tests what we are hoping to test.
    #     self.assertEqual(freq_table.get_table(), freq_table_sum1.get_table())

    # def test19d_add(self):
    #     # Ensure that if frequencies have been computed for the first table, the merged table has no frequencies.
    #     alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
    #     query_seq_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
    #     second_index = 0 if query_seq_index != 0 else self.query_aln_fa_small.size - 1
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table1 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table1.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table1.compute_frequencies()
    #     freq_table2 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table2.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table_sum1 = freq_table1 + freq_table2
    #     # Since frequencies were not compute for the original frequency table checking that the underlying dictionaries
    #     # are equal tests what we are hoping to test.
    #     self.assertEqual(freq_table.get_table(), freq_table_sum1.get_table())

    # def test19e_add(self):
    #     # Ensure that if frequencies have been computed for the second table, the merged table has no frequencies.
    #     alpha = Gapped(FullIUPACProtein())
    #     query_seq_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
    #     second_index = 0 if query_seq_index != 0 else self.query_aln_fa_small.size - 1
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table1 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table1.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table2 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table2.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table2.compute_frequencies()
    #     freq_table_sum1 = freq_table1 + freq_table2
    #     # Since frequencies were not compute for the original frequency table checking that the underlying dictionaries
    #     # are equal tests what we are hoping to test.
    #     self.assertEqual(freq_table.get_table(), freq_table_sum1.get_table())

    # def test19f_add(self):
    #     # Ensure that if frequencies have been computed for the first table, the merged table has no frequencies.
    #     alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
    #     query_seq_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
    #     second_index = 0 if query_seq_index != 0 else self.query_aln_fa_small.size - 1
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table1 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table1.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table2 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table2.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table2.compute_frequencies()
    #     freq_table_sum1 = freq_table1 + freq_table2
    #     # Since frequencies were not compute for the original frequency table checking that the underlying dictionaries
    #     # are equal tests what we are hoping to test.
    #     self.assertEqual(freq_table.get_table(), freq_table_sum1.get_table())

    # def test19g_add(self):
    #     # Ensure that if frequencies have been computed for both tables, the merged table has no frequencies.
    #     alpha = Gapped(FullIUPACProtein())
    #     query_seq_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
    #     second_index = 0 if query_seq_index != 0 else self.query_aln_fa_small.size - 1
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table1 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table1.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table1.compute_frequencies()
    #     freq_table2 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table2.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table2.compute_frequencies()
    #     freq_table_sum1 = freq_table1 + freq_table2
    #     # Since frequencies were not compute for the original frequency table checking that the underlying dictionaries
    #     # are equal tests what we are hoping to test.
    #     self.assertEqual(freq_table.get_table(), freq_table_sum1.get_table())

    # def test19h_add(self):
    #     # Ensure that if frequencies have been computed for both tables, the merged table has no frequencies.
    #     alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
    #     query_seq_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
    #     second_index = 0 if query_seq_index != 0 else self.query_aln_fa_small.size - 1
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table1 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table1.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table1.compute_frequencies()
    #     freq_table2 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table2.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table2.compute_frequencies()
    #     freq_table_sum1 = freq_table1 + freq_table2
    #     # Since frequencies were not compute for the original frequency table checking that the underlying dictionaries
    #     # are equal tests what we are hoping to test.
    #     self.assertEqual(freq_table.get_table(), freq_table_sum1.get_table())

    # def test19i_add(self):
    #     # Test for correct depth after combining two frequency tables.
    #     alpha = Gapped(FullIUPACProtein())
    #     query_seq_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
    #     second_index = 0 if query_seq_index != 0 else self.query_aln_fa_small.size - 1
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table1 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table1.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table2 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length)
    #     freq_table2.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table_sum1 = freq_table1 + freq_table2
    #     self.assertEqual(freq_table_sum1.get_depth(), 2)

    # def test19j_add(self):
    #     # Test for correct depth after combining two frequency tables.
    #     alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
    #     query_seq_index = self.query_aln_fa_small.seq_order.index(self.query_aln_fa_small.query_id)
    #     second_index = 0 if query_seq_index != 0 else self.query_aln_fa_small.size - 1
    #     freq_table = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table1 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table1.characterize_sequence(seq=self.query_aln_fa_small.query_sequence)
    #     freq_table2 = FrequencyTable(alphabet=alpha, seq_len=self.query_aln_fa_small.seq_length, pos_size=2)
    #     freq_table2.characterize_sequence(seq=self.query_aln_fa_small.alignment[second_index].seq)
    #     freq_table_sum1 = freq_table1 + freq_table2
    #     self.assertEqual(freq_table_sum1.get_depth(), 2)

    def test19c_add(self):
        alpha = Gapped(FullIUPACProtein())
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        query_seq_index = self.query_aln_fa_large.seq_order.index(self.query_aln_fa_large.query_id)
        second_index = 0 if query_seq_index != 0 else self.query_aln_fa_large.size - 1
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_large.seq_length)
        freq_table.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
        freq_table.characterize_sequence(seq=self.query_aln_fa_large.alignment[second_index].seq)
        freq_table1 = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_large.seq_length)
        freq_table1.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
        freq_table2 = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_large.seq_length)
        freq_table2.characterize_sequence(seq=self.query_aln_fa_large.alignment[second_index].seq)
        freq_table_sum1 = freq_table1 + freq_table2
        self.assertEqual(freq_table.mapping, freq_table_sum1.mapping)
        self.assertEqual(freq_table.reverse_mapping, freq_table_sum1.reverse_mapping)
        self.assertEqual(freq_table.num_pos, freq_table_sum1.num_pos)
        self.assertEqual(freq_table.position_size, freq_table_sum1.position_size)
        self.assertEqual(freq_table.get_table(), freq_table_sum1.get_table())
        self.assertEqual(freq_table.get_positions(), freq_table_sum1.get_positions())
        self.assertEqual(freq_table.get_depth(), freq_table_sum1.get_depth())

    def test19d_add(self):
        alpha = MultiPositionAlphabet(alphabet=Gapped(FullIUPACProtein()), size=2)
        a_size, gap_chars, mapping = build_mapping(alphabet=alpha)
        query_seq_index = self.query_aln_fa_large.seq_order.index(self.query_aln_fa_large.query_id)
        second_index = 0 if query_seq_index != 0 else self.query_aln_fa_large.size - 1
        freq_table = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_large.seq_length,
                                    pos_size=2)
        freq_table.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
        freq_table.characterize_sequence(seq=self.query_aln_fa_large.alignment[second_index].seq)
        freq_table1 = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_large.seq_length,
                                     pos_size=2)
        freq_table1.characterize_sequence(seq=self.query_aln_fa_large.query_sequence)
        freq_table2 = FrequencyTable(alphabet_size=a_size, mapping=mapping, seq_len=self.query_aln_fa_large.seq_length,
                                     pos_size=2)
        freq_table2.characterize_sequence(seq=self.query_aln_fa_large.alignment[second_index].seq)
        freq_table_sum1 = freq_table1 + freq_table2
        self.assertEqual(freq_table.mapping, freq_table_sum1.mapping)
        self.assertEqual(freq_table.reverse_mapping, freq_table_sum1.reverse_mapping)
        self.assertEqual(freq_table.num_pos, freq_table_sum1.num_pos)
        self.assertEqual(freq_table.position_size, freq_table_sum1.position_size)
        self.assertEqual(freq_table.get_table(), freq_table_sum1.get_table())
        self.assertEqual(freq_table.get_positions(), freq_table_sum1.get_positions())
        self.assertEqual(freq_table.get_depth(), freq_table_sum1.get_depth())