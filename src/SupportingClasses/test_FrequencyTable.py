"""
Created on July 10, 2019

@author: Daniel Konecki
"""
import numpy as np
from test_Base import TestBase
from Bio.Alphabet import Alphabet, Gapped
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

    def test1a_init(self):
        freq_table = FrequencyTable(alphabet=Gapped(FullIUPACProtein()))
        self.assertTrue(isinstance(freq_table.alphabet, Gapped))
        for char in FullIUPACProtein().letters + '-':
            self.assertTrue(char in freq_table.alphabet.letters)
        self.assertEqual(freq_table.position_size, 1)
        self.assertTrue(freq_table.get_table() == {})

    def test1b_init(self):
        freq_table = FrequencyTable(alphabet=FullIUPACProtein())
        self.assertTrue(isinstance(freq_table.alphabet, Alphabet))
        for char in FullIUPACProtein().letters:
            self.assertTrue(char in freq_table.alphabet.letters)
        self.assertFalse('-' in freq_table.alphabet.letters)
        self.assertEqual(freq_table.position_size, 1)
        self.assertTrue(freq_table.get_table() == {})

    def test1c_init(self):
        with self.assertRaises(ValueError):
            freq_table = FrequencyTable(alphabet=Gapped(FullIUPACProtein()), pos_size=2)

    def test2a__add_position(self):
        freq_table = FrequencyTable(alphabet=FullIUPACProtein())
        # Testing inserting wrong type of position
        with self.assertRaises(TypeError):
            freq_table._add_position(pos=(1, 2))
        # Test inserting single correct position
        freq_table._add_position(1)
        self.assertEqual(freq_table.get_table(), {1: {}})
        # Test re-inserting single correct position
        freq_table._add_position(1)
        self.assertEqual(freq_table.get_table(), {1: {}})
        # Test inserting another correct position
        freq_table._add_position(2)
        self.assertEqual(freq_table.get_table(), {1: {}, 2: {}})

    def test2b__add_position(self):
        freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2), pos_size=2)
        # Testing inserting wrong type of position
        with self.assertRaises(TypeError):
            freq_table._add_position(pos=1)
        # Testing inserting wrong size of position
        with self.assertRaises(TypeError):
            freq_table._add_position(pos=(1, 2, 3))
        # Test inserting single correct position
        freq_table._add_position((1, 2))
        self.assertEqual(freq_table.get_table(), {(1, 2): {}})
        # Test re-inserting single correct position
        freq_table._add_position((1, 2))
        self.assertEqual(freq_table.get_table(), {(1, 2): {}})
        # Test inserting another correct position
        freq_table._add_position((2, 3))
        self.assertEqual(freq_table.get_table(), {(1, 2): {}, (2, 3): {}})

    def test3a__add_pos_char(self):
        freq_table = FrequencyTable(alphabet=FullIUPACProtein())
        # Testing inserting wrong type of position and character
        with self.assertRaises(ValueError):
            freq_table._add_pos_char(pos=(1, 2), char='AA')
        # Test inserting single correct position and character
        freq_table._add_pos_char(pos=1, char='A')
        self.assertEqual(freq_table.get_table(), {1: {'A': 0}})
        # Test re-inserting single correct position and character
        freq_table._add_pos_char(pos=1, char='A')
        self.assertEqual(freq_table.get_table(), {1: {'A': 0}})
        # Test inserting another correct position and character
        freq_table._add_pos_char(pos=2, char='G')
        self.assertEqual(freq_table.get_table(), {1: {'A': 0}, 2: {'G': 0}})

    def test3b__add_pos_char(self):
        freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2), pos_size=2)
        # Testing inserting wrong type of position and character
        with self.assertRaises(ValueError):
            freq_table._add_pos_char(pos=1, char='A')
        # Testing inserting wrong size of position and character
        with self.assertRaises(ValueError):
            freq_table._add_pos_char(pos=(1, 2, 3), char='AAA')
        # Test inserting single correct position and character
        freq_table._add_pos_char(pos=(1, 2), char='AA')
        self.assertEqual(freq_table.get_table(), {(1, 2): {'AA': 0}})
        # Test re-inserting single correct position and character
        freq_table._add_pos_char(pos=(1, 2), char='AA')
        self.assertEqual(freq_table.get_table(), {(1, 2): {'AA': 0}})
        # Test inserting another correct position and character
        freq_table._add_pos_char(pos=(2, 3), char='GG')
        self.assertEqual(freq_table.get_table(), {(1, 2): {'AA': 0}, (2, 3): {'GG': 0}})

    def test4a_increment_count(self):
        freq_table = FrequencyTable(alphabet=FullIUPACProtein())
        # Testing inserting wrong type of position and character
        with self.assertRaises(ValueError):
            freq_table.increment_count(pos=(1, 2), char='AA')
        # Test inserting single correct position and character
        freq_table.increment_count(pos=1, char='A')
        self.assertEqual(freq_table.get_table(), {1: {'A': 1}})
        # Test re-inserting single correct position and character
        freq_table.increment_count(pos=1, char='A')
        self.assertEqual(freq_table.get_table(), {1: {'A': 2}})
        # Test inserting another correct position and character
        freq_table.increment_count(pos=2, char='G')
        self.assertEqual(freq_table.get_table(), {1: {'A': 2}, 2: {'G': 1}})

    def test4b_increment_count(self):
        freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2), pos_size=2)
        # Testing inserting wrong type of position and character
        with self.assertRaises(ValueError):
            freq_table.increment_count(pos=1, char='A')
        # Testing inserting wrong size of position and character
        with self.assertRaises(ValueError):
            freq_table.increment_count(pos=(1, 2, 3), char='AAA')
        # Test inserting single correct position and character
        freq_table.increment_count(pos=(1, 2), char='AA')
        self.assertEqual(freq_table.get_table(), {(1, 2): {'AA': 1}})
        # Test re-inserting single correct position and character
        freq_table.increment_count(pos=(1, 2), char='AA')
        self.assertEqual(freq_table.get_table(), {(1, 2): {'AA': 2}})
        # Test inserting another correct position and character
        freq_table.increment_count(pos=(2, 3), char='GG')
        self.assertEqual(freq_table.get_table(), {(1, 2): {'AA': 2}, (2, 3): {'GG': 1}})

    def test5_get_table(self):
        freq_table = FrequencyTable(alphabet=FullIUPACProtein())
        table1 = freq_table.get_table()
        table2 = freq_table.get_table()
        self.assertEqual(table1, table2)
        self.assertIsNot(table1, table2)

    def test6a_get_positions(self):
        freq_table = FrequencyTable(alphabet=FullIUPACProtein())
        freq_table.increment_count(pos=1, char='A')
        freq_table.increment_count(pos=1, char='A')
        freq_table.increment_count(pos=2, char='G')
        self.assertEqual(freq_table.get_positions(), [1, 2])

    def test6b_get_positions(self):
        freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2), pos_size=2)
        freq_table.increment_count(pos=(1, 2), char='AA')
        freq_table.increment_count(pos=(1, 2), char='AA')
        freq_table.increment_count(pos=(2, 3), char='GG')
        self.assertEqual(freq_table.get_positions(), [(1, 2), (2, 3)])

    def test7a_get_chars(self):
        freq_table = FrequencyTable(alphabet=FullIUPACProtein())
        freq_table.increment_count(pos=1, char='A')
        freq_table.increment_count(pos=1, char='A')
        freq_table.increment_count(pos=2, char='G')
        self.assertEqual(freq_table.get_chars(pos=1), ['A'])
        self.assertEqual(freq_table.get_chars(pos=2), ['G'])

    def test7b_get_chars(self):
        freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2), pos_size=2)
        freq_table.increment_count(pos=(1, 2), char='AA')
        freq_table.increment_count(pos=(1, 2), char='AA')
        freq_table.increment_count(pos=(2, 3), char='GG')
        self.assertEqual(freq_table.get_chars(pos=(1, 2)), ['AA'])
        self.assertEqual(freq_table.get_chars(pos=(2, 3)), ['GG'])

    def test8a_get_count(self):
        freq_table = FrequencyTable(alphabet=FullIUPACProtein())
        self.assertEqual(freq_table.get_count(pos=1, char='A'), 0)
        self.assertEqual(freq_table.get_count(pos=2, char='G'), 0)
        freq_table.increment_count(pos=1, char='A')
        self.assertEqual(freq_table.get_count(pos=1, char='A'), 1)
        self.assertEqual(freq_table.get_count(pos=2, char='G'), 0)
        freq_table.increment_count(pos=1, char='A')
        self.assertEqual(freq_table.get_count(pos=1, char='A'), 2)
        self.assertEqual(freq_table.get_count(pos=2, char='G'), 0)
        freq_table.increment_count(pos=2, char='G')
        self.assertEqual(freq_table.get_count(pos=1, char='A'), 2)
        self.assertEqual(freq_table.get_count(pos=2, char='G'), 1)

    def test8b_get_count(self):
        freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2), pos_size=2)
        self.assertEqual(freq_table.get_count(pos=(1, 2), char='AA'), 0)
        self.assertEqual(freq_table.get_count(pos=(2, 3), char='GG'), 0)
        freq_table.increment_count(pos=(1, 2), char='AA')
        self.assertEqual(freq_table.get_count(pos=(1, 2), char='AA'), 1)
        self.assertEqual(freq_table.get_count(pos=(2, 3), char='GG'), 0)
        freq_table.increment_count(pos=(1, 2), char='AA')
        self.assertEqual(freq_table.get_count(pos=(1, 2), char='AA'), 2)
        self.assertEqual(freq_table.get_count(pos=(2, 3), char='GG'), 0)
        freq_table.increment_count(pos=(2, 3), char='GG')
        self.assertEqual(freq_table.get_count(pos=(1, 2), char='AA'), 2)
        self.assertEqual(freq_table.get_count(pos=(2, 3), char='GG'), 1)

    def test9a_get_count_array(self):
        freq_table = FrequencyTable(alphabet=FullIUPACProtein())
        self.assertIsNone(freq_table.get_count_array(pos=1))
        self.assertIsNone(freq_table.get_count_array(pos=2))
        freq_table.increment_count(pos=1, char='A')
        self.assertEqual(freq_table.get_count_array(pos=1), np.array([1]))
        self.assertIsNone(freq_table.get_count_array(pos=2))
        freq_table.increment_count(pos=1, char='A')
        self.assertEqual(freq_table.get_count_array(pos=1), np.array([2]))
        self.assertIsNone(freq_table.get_count_array(pos=2))
        freq_table.increment_count(pos=2, char='G')
        self.assertEqual(freq_table.get_count_array(pos=1), np.array([2]))
        self.assertEqual(freq_table.get_count_array(pos=2), np.array([1]))

    def test9b_get_count(self):
        freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2), pos_size=2)
        self.assertIsNone(freq_table.get_count_array(pos=(1, 2)))
        self.assertIsNone(freq_table.get_count_array(pos=(2, 3)))
        freq_table.increment_count(pos=(1, 2), char='AA')
        self.assertEqual(freq_table.get_count_array(pos=(1, 2)), np.array([1]))
        self.assertIsNone(freq_table.get_count_array(pos=(2, 3)))
        freq_table.increment_count(pos=(1, 2), char='AA')
        self.assertEqual(freq_table.get_count_array(pos=(1, 2)), np.array([2]))
        self.assertIsNone(freq_table.get_count_array(pos=(2, 3)))
        freq_table.increment_count(pos=(2, 3), char='GG')
        self.assertEqual(freq_table.get_count_array(pos=(1, 2)), np.array([2]))
        self.assertEqual(freq_table.get_count_array(pos=(2, 3)), np.array([1]))

    def test10a_get_count_matrix(self):
        freq_table = FrequencyTable(alphabet=FullIUPACProtein())
        alpha_size, gap_chars, mapping = build_mapping(alphabet=freq_table.alphabet)
        self.assertIsNone(freq_table.get_count_matrix())
        freq_table.increment_count(pos=1, char='A')
        expected_table = np.zeros((1, alpha_size))
        expected_table[0, mapping['A']] = 1
        diff = freq_table.get_count_matrix() - expected_table
        self.assertTrue(not diff.any())
        freq_table.increment_count(pos=1, char='A')
        expected_table[0, mapping['A']] = 2
        diff2 = freq_table.get_count_matrix() - expected_table
        self.assertTrue(not diff2.any())
        freq_table.increment_count(pos=2, char='G')
        expected_table2 = np.zeros((2, alpha_size))
        expected_table2[0, mapping['A']] = 2
        expected_table2[1, mapping['G']] = 1
        diff3 = freq_table.get_count_matrix() - expected_table2
        self.assertTrue(not diff3.any())

    def test10b_get_count_matrix(self):
        freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2), pos_size=2)
        alpha_size, gap_chars, mapping = build_mapping(alphabet=freq_table.alphabet)
        self.assertIsNone(freq_table.get_count_matrix())
        freq_table.increment_count(pos=(1, 2), char='AA')
        expected_table = np.zeros((1, alpha_size))
        expected_table[0, mapping['AA']] = 1
        diff = freq_table.get_count_matrix() - expected_table
        self.assertTrue(not diff.any())
        freq_table.increment_count(pos=(1, 2), char='AA')
        expected_table[0, mapping['AA']] = 2
        diff2 = freq_table.get_count_matrix() - expected_table
        self.assertTrue(not diff2.any())
        freq_table.increment_count(pos=(2, 3), char='GG')
        expected_table2 = np.zeros((2, alpha_size))
        expected_table2[0, mapping['AA']] = 2
        expected_table2[1, mapping['GG']] = 1
        diff3 = freq_table.get_count_matrix() - expected_table2
        self.assertTrue(not diff3.any())

    def test11a_add(self):
        freq_table = FrequencyTable(alphabet=FullIUPACProtein())
        freq_table.increment_count(pos=1, char='A')
        freq_table.increment_count(pos=1, char='A')
        freq_table1 = FrequencyTable(alphabet=FullIUPACProtein())
        freq_table1.increment_count(pos=1, char='A')
        freq_table2 = FrequencyTable(alphabet=FullIUPACProtein())
        freq_table2.increment_count(pos=1, char='A')
        freq_table_sum1 = freq_table1 + freq_table2
        self.assertTrue(isinstance(freq_table.alphabet, type(freq_table_sum1.alphabet)))
        self.assertEqual(len(freq_table.alphabet.letters), len(freq_table_sum1.alphabet.letters))
        for char in freq_table.alphabet.letters:
            self.assertTrue(char in freq_table_sum1.alphabet.letters)
        self.assertEqual(freq_table.position_size, freq_table_sum1.position_size)
        self.assertEqual(freq_table.get_table(), freq_table_sum1.get_table())
        self.assertEqual(freq_table.get_positions(), freq_table_sum1.get_positions())
        for i in freq_table.get_positions():
            self.assertEqual(freq_table.get_chars(pos=i), freq_table_sum1.get_chars(pos=i))
            for c in freq_table.get_chars(pos=i):
                self.assertEqual(freq_table.get_count(pos=i, char=c), freq_table_sum1.get_count(pos=i, char=c))
        freq_table.increment_count(pos=2, char='G')
        freq_table3 = FrequencyTable(alphabet=FullIUPACProtein())
        freq_table3.increment_count(pos=2, char='G')
        freq_table_sum2 = freq_table_sum1 + freq_table3
        self.assertTrue(isinstance(freq_table.alphabet, type(freq_table_sum2.alphabet)))
        self.assertEqual(len(freq_table.alphabet.letters), len(freq_table_sum2.alphabet.letters))
        for char in freq_table.alphabet.letters:
            self.assertTrue(char in freq_table_sum2.alphabet.letters)
        self.assertEqual(freq_table.position_size, freq_table_sum2.position_size)
        self.assertEqual(freq_table.get_table(), freq_table_sum2.get_table())
        self.assertEqual(freq_table.get_positions(), freq_table_sum2.get_positions())
        for i in freq_table.get_positions():
            self.assertEqual(freq_table.get_chars(pos=i), freq_table_sum2.get_chars(pos=i))
            for c in freq_table.get_chars(pos=i):
                self.assertEqual(freq_table.get_count(pos=i, char=c), freq_table_sum2.get_count(pos=i, char=c))

    def test11b_add(self):
        freq_table = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2), pos_size=2)
        freq_table.increment_count(pos=(1, 2), char='AA')
        freq_table.increment_count(pos=(1, 2), char='AA')
        freq_table1 = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2), pos_size=2)
        freq_table1.increment_count(pos=(1, 2), char='AA')
        freq_table2 = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2), pos_size=2)
        freq_table2.increment_count(pos=(1, 2), char='AA')
        freq_table_sum1 = freq_table1 + freq_table2
        self.assertTrue(isinstance(freq_table.alphabet, type(freq_table_sum1.alphabet)))
        self.assertEqual(len(freq_table.alphabet.letters), len(freq_table_sum1.alphabet.letters))
        for char in freq_table.alphabet.letters:
            self.assertTrue(char in freq_table_sum1.alphabet.letters)
        self.assertEqual(freq_table.position_size, freq_table_sum1.position_size)
        self.assertEqual(freq_table.get_table(), freq_table_sum1.get_table())
        self.assertEqual(freq_table.get_positions(), freq_table_sum1.get_positions())
        for i in freq_table.get_positions():
            self.assertEqual(freq_table.get_chars(pos=i), freq_table_sum1.get_chars(pos=i))
            for c in freq_table.get_chars(pos=i):
                self.assertEqual(freq_table.get_count(pos=i, char=c), freq_table_sum1.get_count(pos=i, char=c))
        freq_table.increment_count(pos=(2, 3), char='GG')
        freq_table3 = FrequencyTable(alphabet=MultiPositionAlphabet(alphabet=FullIUPACProtein(), size=2), pos_size=2)
        freq_table3.increment_count(pos=(2, 3), char='GG')
        freq_table_sum2 = freq_table_sum1 + freq_table3
        self.assertTrue(isinstance(freq_table.alphabet, type(freq_table_sum2.alphabet)))
        self.assertEqual(len(freq_table.alphabet.letters), len(freq_table_sum2.alphabet.letters))
        for char in freq_table.alphabet.letters:
            self.assertTrue(char in freq_table_sum2.alphabet.letters)
        self.assertEqual(freq_table.position_size, freq_table_sum2.position_size)
        self.assertEqual(freq_table.get_table(), freq_table_sum2.get_table())
        self.assertEqual(freq_table.get_positions(), freq_table_sum2.get_positions())
        for i in freq_table.get_positions():
            self.assertEqual(freq_table.get_chars(pos=i), freq_table_sum2.get_chars(pos=i))
            for c in freq_table.get_chars(pos=i):
                self.assertEqual(freq_table.get_count(pos=i, char=c), freq_table_sum2.get_count(pos=i, char=c))
