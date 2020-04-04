"""
Created on Mar 4, 2020

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
from MatchMismatchTable import MatchMismatchTable
from EvolutionaryTraceAlphabet import FullIUPACProtein, MultiPositionAlphabet

class TestMatchMismatchTable(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestMatchMismatchTable, cls).setUpClass()
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
        cls.quad_alphabet = MultiPositionAlphabet(alphabet=cls.single_alphabet, size=4)
        cls.quad_size, _, cls.quad_mapping, cls.quad_reverse = build_mapping(alphabet=cls.quad_alphabet)
        cls.single_to_quad = {}
        for char in cls.quad_mapping:
            key = (cls.single_mapping[char[0]], cls.single_mapping[char[1]], cls.single_mapping[char[2]],
                   cls.single_mapping[char[3]])
            cls.single_to_quad[key] = cls.quad_mapping[char]

    def evaluate_init(self, seq_len, num_aln, large_alpha_size, large_alpha_map, large_alpha_reverse, single_to_large,
                      pos_size):
        mm_table = MatchMismatchTable(seq_len=seq_len, num_aln=num_aln, single_alphabet_size=self.single_size,
                                      single_mapping=self.single_mapping, single_reverse_mapping=self.single_reverse,
                                      larger_alphabet_size=large_alpha_size, larger_alphabet_mapping=large_alpha_map,
                                      larger_alphabet_reverse_mapping=large_alpha_reverse,
                                      single_to_larger_mapping=single_to_large, pos_size=pos_size)
        self.assertEqual(mm_table.seq_len, seq_len)
        self.assertEqual(mm_table.pos_size, pos_size)
        self.assertFalse(((mm_table.num_aln - num_aln) != 0.0).any())
        self.assertEqual(mm_table.get_depth(), num_aln.shape[0])
        self.assertEqual(mm_table.single_alphabet_size, self.single_size)
        self.assertEqual(mm_table.single_mapping, self.single_mapping)
        self.assertTrue((mm_table.single_reverse_mapping == self.single_reverse).all())
        self.assertEqual(mm_table.larger_alphabet_size, large_alpha_size)
        self.assertEqual(mm_table.larger_mapping, large_alpha_map)
        self.assertTrue((mm_table.larger_reverse_mapping == large_alpha_reverse).all())
        self.assertEqual(mm_table.single_to_larger_mapping, single_to_large)
        self.assertIsNone(mm_table.match_mismatch_tables)

    def test1a_init(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_init(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln, large_alpha_size=self.pair_size,
                           large_alpha_map=self.pair_mapping, large_alpha_reverse=self.pair_reverse,
                           single_to_large=self.single_to_pair, pos_size=1)

    def test1b_init(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_init(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln, large_alpha_size=self.quad_size,
                           large_alpha_map=self.quad_mapping, large_alpha_reverse=self.quad_reverse,
                           single_to_large=self.single_to_quad, pos_size=2)

    def test1c_init(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_init(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln, large_alpha_size=self.pair_size,
                           large_alpha_map=self.pair_mapping, large_alpha_reverse=self.pair_reverse,
                           single_to_large=self.single_to_pair, pos_size=1)

    def test1d_init(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_init(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln, large_alpha_size=self.quad_size,
                           large_alpha_map=self.quad_mapping, large_alpha_reverse=self.quad_reverse,
                           single_to_large=self.single_to_quad, pos_size=2)

    def evaluate_identify_matches_mismatches(self, seq_len, num_aln, large_alpha_size, large_alpha_map,
                                             large_alpha_reverse, single_to_large, pos_size):
        mm_table = MatchMismatchTable(seq_len=seq_len, num_aln=num_aln, single_alphabet_size=self.single_size,
                                      single_mapping=self.single_mapping, single_reverse_mapping=self.single_reverse,
                                      larger_alphabet_size=large_alpha_size, larger_alphabet_mapping=large_alpha_map,
                                      larger_alphabet_reverse_mapping=large_alpha_reverse,
                                      single_to_larger_mapping=single_to_large, pos_size=pos_size)
        mm_table.identify_matches_mismatches()
        self.assertEqual(list(sorted(mm_table.match_mismatch_tables.keys())), list(range(seq_len)))
        for pos in mm_table.match_mismatch_tables:
            for i in range(num_aln.shape[0]):
                for j in range(i + 1, num_aln.shape[0]):
                    if num_aln[i, pos] == num_aln[j, pos]:
                        self.assertEqual(mm_table.match_mismatch_tables[pos][i, j], 1)
                    else:
                        self.assertEqual(mm_table.match_mismatch_tables[pos][i, j], -1)
            self.assertFalse(np.tril(mm_table.match_mismatch_tables[pos]).any())

    def test2a_identify_matches_mismatches(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_identify_matches_mismatches(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                                  large_alpha_size=self.pair_size, large_alpha_map=self.pair_mapping,
                                                  large_alpha_reverse=self.pair_reverse,
                                                  single_to_large=self.single_to_pair, pos_size=1)

    def test2b_identify_matches_mismatches(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_identify_matches_mismatches(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                                  large_alpha_size=self.quad_size, large_alpha_map=self.quad_mapping,
                                                  large_alpha_reverse=self.quad_reverse,
                                                  single_to_large=self.single_to_quad, pos_size=2)

    def test2c_identify_matches_mismatches(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_identify_matches_mismatches(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                                  large_alpha_size=self.pair_size, large_alpha_map=self.pair_mapping,
                                                  large_alpha_reverse=self.pair_reverse,
                                                  single_to_large=self.single_to_pair, pos_size=1)

    def test2d_identify_matches_mismatches(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_identify_matches_mismatches(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                                  large_alpha_size=self.quad_size, large_alpha_map=self.quad_mapping,
                                                  large_alpha_reverse=self.quad_reverse,
                                                  single_to_large=self.single_to_quad, pos_size=2)

    def evaluate_get_status_and_character(self, seq_len, num_aln, large_alpha_size, large_alpha_map,
                                          large_alpha_reverse, single_to_large, pos_size):
        mm_table = MatchMismatchTable(seq_len=seq_len, num_aln=num_aln, single_alphabet_size=self.single_size,
                                      single_mapping=self.single_mapping, single_reverse_mapping=self.single_reverse,
                                      larger_alphabet_size=large_alpha_size, larger_alphabet_mapping=large_alpha_map,
                                      larger_alphabet_reverse_mapping=large_alpha_reverse,
                                      single_to_larger_mapping=single_to_large, pos_size=pos_size)
        mm_table.identify_matches_mismatches()
        for s1 in range(num_aln.shape[0] - 1):
            s2 = np.random.choice(list(range(s1 + 1, num_aln.shape[0])), 1)[0]
            for i in range(seq_len):
                if pos_size == 1:
                    if s1 > 1:
                        with self.assertRaises(ValueError):
                            mm_table.get_status_and_character(pos=i, seq_ind1=s1, seq_ind2=s1 - 1)
                    with self.assertRaises(ValueError):
                        mm_table.get_status_and_character(pos=i, seq_ind1=s1, seq_ind2=s1)
                    expected_char1 = self.single_reverse[num_aln[s1, i]]
                    expected_char2 = self.single_reverse[num_aln[s2, i]]
                    expected_char = expected_char1 + expected_char2
                    expected_status = expected_char1 == expected_char2
                    status, char = mm_table.get_status_and_character(pos=i, seq_ind1=s1, seq_ind2=s2)
                    self.assertEqual(char, expected_char)
                    if expected_status:
                        self.assertEqual(status, 'match')
                    else:
                        self.assertEqual(status, 'mismatch')
                    continue
                for j in range(i, seq_len):
                    if s1 > 1:
                        with self.assertRaises(ValueError):
                            mm_table.get_status_and_character(pos=(i, j), seq_ind1=s1, seq_ind2=s1 - 1)
                    with self.assertRaises(ValueError):
                        mm_table.get_status_and_character(pos=(i, j), seq_ind1=s1, seq_ind2=s1)
                    expected_char1 = self.single_reverse[num_aln[s1, i]]
                    expected_char2 = self.single_reverse[num_aln[s1, j]]
                    expected_pair1 = expected_char1 + expected_char2
                    expected_char3 = self.single_reverse[num_aln[s2, i]]
                    expected_char4 = self.single_reverse[num_aln[s2, j]]
                    expected_pair2 = expected_char3 + expected_char4
                    expected_quad = expected_pair1 + expected_pair2
                    expected_status = ((expected_pair1 == expected_pair2) or
                                       ((expected_char1 != expected_char3) and (expected_char2 != expected_char4)))
                    status, char = mm_table.get_status_and_character(pos=(i, j), seq_ind1=s1, seq_ind2=s2)
                    self.assertEqual(char, expected_quad)
                    if expected_status:
                        self.assertEqual(status, 'match')
                    else:
                        self.assertEqual(status, 'mismatch')

    def test3a_get_status_and_character(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_get_status_and_character(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                               large_alpha_size=self.pair_size, large_alpha_map=self.pair_mapping,
                                               large_alpha_reverse=self.pair_reverse,
                                               single_to_large=self.single_to_pair, pos_size=1)

    def test3b_get_status_and_character(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_get_status_and_character(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                               large_alpha_size=self.quad_size, large_alpha_map=self.quad_mapping,
                                               large_alpha_reverse=self.quad_reverse,
                                               single_to_large=self.single_to_quad, pos_size=2)

    def test3c_get_status_and_character(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_get_status_and_character(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                               large_alpha_size=self.pair_size, large_alpha_map=self.pair_mapping,
                                               large_alpha_reverse=self.pair_reverse,
                                               single_to_large=self.single_to_pair, pos_size=1)

    def test3d_get_status_and_character(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_get_status_and_character(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                               large_alpha_size=self.quad_size, large_alpha_map=self.quad_mapping,
                                               large_alpha_reverse=self.quad_reverse,
                                               single_to_large=self.single_to_quad, pos_size=2)

    def evaluate_get_depth(self, aln, match_mismatch_table):
        self.assertEqual(aln.size, match_mismatch_table.get_depth())

    def test4a_get_depth(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.pair_size,
                                      larger_alphabet_mapping=self.pair_mapping,
                                      larger_alphabet_reverse_mapping=self.pair_reverse,
                                      single_to_larger_mapping=self.single_to_pair, pos_size=1)
        self.evaluate_get_depth(aln=self.query_aln_fa_small, match_mismatch_table=mm_table)

    def test4b_get_depth(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.quad_size,
                                      larger_alphabet_mapping=self.quad_mapping,
                                      larger_alphabet_reverse_mapping=self.quad_reverse,
                                      single_to_larger_mapping=self.single_to_quad, pos_size=2)
        self.evaluate_get_depth(aln=self.query_aln_fa_small, match_mismatch_table=mm_table)

    def test4c_get_depth(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.pair_size,
                                      larger_alphabet_mapping=self.pair_mapping,
                                      larger_alphabet_reverse_mapping=self.pair_reverse,
                                      single_to_larger_mapping=self.single_to_pair, pos_size=1)
        self.evaluate_get_depth(aln=self.query_aln_fa_large, match_mismatch_table=mm_table)

    def test4d_get_depth(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.quad_size,
                                      larger_alphabet_mapping=self.quad_mapping,
                                      larger_alphabet_reverse_mapping=self.quad_reverse,
                                      single_to_larger_mapping=self.single_to_quad, pos_size=2)
        self.evaluate_get_depth(aln=self.query_aln_fa_large, match_mismatch_table=mm_table)

    def evaluate_get_characters_and_statuses(self, mm_table, pos, indices1, indices2, test_chars, test_statuses):
        self.assertEqual(len(indices1), len(indices2))
        self.assertEqual(len(test_chars), len(test_statuses))
        counter = 0
        for i in range(len(indices1)):
            s1 = indices1[i]
            s2 = indices2[i]
            if s1 > s2:
                s1, s2 = s2, s1
            status, char = mm_table.get_status_and_character(pos=pos, seq_ind1=s1, seq_ind2=s2)
            self.assertEqual(test_chars[counter], char)
            self.assertEqual(test_statuses[counter], status)
            counter += 1
        self.assertEqual(counter, len(test_chars))

    def test5a__get_characters_and_statuses_single_pos(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.pair_size,
                                      larger_alphabet_mapping=self.pair_mapping,
                                      larger_alphabet_reverse_mapping=self.pair_reverse,
                                      single_to_larger_mapping=self.single_to_pair, pos_size=1)
        mm_table.identify_matches_mismatches()
        pos = self.query_aln_fa_small.seq_length // 2
        indices1, indices2 = np.triu_indices(n=self.query_aln_fa_small.size // 3, k=1)
        test_chars1, test_chars2, test_statuses = mm_table._get_characters_and_statuses_single_pos(
            pos=pos, indices1=indices1, indices2=indices2)
        combined_char = np.hstack([test_chars1, test_chars2])
        test_chars = np.array([self.pair_reverse[self.single_to_pair[char_tup]]
                               for char_tup in map(tuple, combined_char)])
        status_con = np.array(['mismatch', 'match'])
        test_statuses = status_con[(test_statuses == 1) * 1]
        self.evaluate_get_characters_and_statuses(mm_table=mm_table, pos=pos, indices1=indices1, indices2=indices2,
                                                  test_chars=test_chars, test_statuses=test_statuses)

    def test5b__get_characters_and_statuses_single_pos(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.pair_size,
                                      larger_alphabet_mapping=self.pair_mapping,
                                      larger_alphabet_reverse_mapping=self.pair_reverse,
                                      single_to_larger_mapping=self.single_to_pair, pos_size=1)
        mm_table.identify_matches_mismatches()
        pos = self.query_aln_fa_large.seq_length // 2
        indices1, indices2 = np.triu_indices(n=self.query_aln_fa_large.size // 3, k=1)
        test_chars1, test_chars2, test_statuses = mm_table._get_characters_and_statuses_single_pos(
            pos=pos, indices1=indices1, indices2=indices2)
        combined_char = np.hstack([test_chars1, test_chars2])
        test_chars = np.array([self.pair_reverse[self.single_to_pair[char_tup]]
                               for char_tup in map(tuple, combined_char)])
        status_con = np.array(['mismatch', 'match'])
        test_statuses = status_con[(test_statuses == 1) * 1]
        self.evaluate_get_characters_and_statuses(mm_table=mm_table, pos=pos, indices1=indices1, indices2=indices2,
                                                  test_chars=test_chars, test_statuses=test_statuses)

    def test5c__get_characters_and_statuses_multi_pos(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.quad_size,
                                      larger_alphabet_mapping=self.quad_mapping,
                                      larger_alphabet_reverse_mapping=self.quad_reverse,
                                      single_to_larger_mapping=self.single_to_quad, pos_size=2)
        mm_table.identify_matches_mismatches()
        pos = (self.query_aln_fa_small.seq_length // 3, (2 * self.query_aln_fa_small.seq_length) // 3)
        indices1, indices2 = np.triu_indices(n=self.query_aln_fa_small.size // 3, k=1)
        test_chars1, test_chars2, test_statuses = mm_table._get_characters_and_statuses_multi_pos(
            pos=pos, indices1=indices1, indices2=indices2)
        combined_char = np.hstack([test_chars1, test_chars2])
        test_chars = np.array([self.quad_reverse[self.single_to_quad[char_tup]]
                               for char_tup in map(tuple, combined_char)])
        status_con = np.array(['mismatch', 'match'])
        test_statuses = status_con[(np.abs(test_statuses) == 2) * 1]
        self.evaluate_get_characters_and_statuses(mm_table=mm_table, pos=pos, indices1=indices1, indices2=indices2,
                                                  test_chars=test_chars, test_statuses=test_statuses)

    def test5d__get_characters_and_statuses_multi_pos(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.quad_size,
                                      larger_alphabet_mapping=self.quad_mapping,
                                      larger_alphabet_reverse_mapping=self.quad_reverse,
                                      single_to_larger_mapping=self.single_to_quad, pos_size=2)
        mm_table.identify_matches_mismatches()
        pos = (self.query_aln_fa_large.seq_length // 3, (2 * self.query_aln_fa_large.seq_length) // 3)
        indices1, indices2 = np.triu_indices(n=self.query_aln_fa_large.size // 3, k=1)
        test_chars1, test_chars2, test_statuses = mm_table._get_characters_and_statuses_multi_pos(
            pos=pos, indices1=indices1, indices2=indices2)
        combined_char = np.hstack([test_chars1, test_chars2])
        test_chars = np.array([self.quad_reverse[self.single_to_quad[char_tup]]
                               for char_tup in map(tuple, combined_char)])
        status_con = np.array(['mismatch', 'match'])
        test_statuses = status_con[(np.abs(test_statuses) == 2) * 1]
        self.evaluate_get_characters_and_statuses(mm_table=mm_table, pos=pos, indices1=indices1, indices2=indices2,
                                                  test_chars=test_chars, test_statuses=test_statuses)

    def test6e_get_upper_triangle(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.pair_size,
                                      larger_alphabet_mapping=self.pair_mapping,
                                      larger_alphabet_reverse_mapping=self.pair_reverse,
                                      single_to_larger_mapping=self.single_to_pair, pos_size=1)
        mm_table.identify_matches_mismatches()
        pos = self.query_aln_fa_small.seq_length // 2
        indices = list(range(self.query_aln_fa_small.size // 3))
        indices1, indices2 = np.triu_indices(n=self.query_aln_fa_small.size // 3, k=1)
        test_chars, test_statuses = mm_table.get_upper_triangle(pos=pos, indices=indices)
        self.evaluate_get_characters_and_statuses(mm_table=mm_table, pos=pos, indices1=indices1, indices2=indices2,
                                                  test_chars=test_chars, test_statuses=test_statuses)

    def test6f_get_upper_triangle(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.pair_size,
                                      larger_alphabet_mapping=self.pair_mapping,
                                      larger_alphabet_reverse_mapping=self.pair_reverse,
                                      single_to_larger_mapping=self.single_to_pair, pos_size=1)
        mm_table.identify_matches_mismatches()
        pos = self.query_aln_fa_large.seq_length // 2
        indices = list(range(self.query_aln_fa_large.size // 3))
        indices1, indices2 = np.triu_indices(n=self.query_aln_fa_large.size // 3, k=1)
        test_chars, test_statuses = mm_table.get_upper_triangle(pos=pos, indices=indices)
        self.evaluate_get_characters_and_statuses(mm_table=mm_table, pos=pos, indices1=indices1, indices2=indices2,
                                                  test_chars=test_chars, test_statuses=test_statuses)

    def test6g_get_upper_triangle(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.quad_size,
                                      larger_alphabet_mapping=self.quad_mapping,
                                      larger_alphabet_reverse_mapping=self.quad_reverse,
                                      single_to_larger_mapping=self.single_to_quad, pos_size=2)
        mm_table.identify_matches_mismatches()
        pos = (self.query_aln_fa_small.seq_length // 3, (2 * self.query_aln_fa_small.seq_length) // 3)
        indices = list(range(self.query_aln_fa_small.size // 3))
        indices1, indices2 = np.triu_indices(n=self.query_aln_fa_small.size // 3, k=1)
        test_chars, test_statuses = mm_table.get_upper_triangle(pos=pos, indices=indices)
        self.evaluate_get_characters_and_statuses(mm_table=mm_table, pos=pos, indices1=indices1, indices2=indices2,
                                                  test_chars=test_chars, test_statuses=test_statuses)

    def test6h_get_upper_triangle(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.quad_size,
                                      larger_alphabet_mapping=self.quad_mapping,
                                      larger_alphabet_reverse_mapping=self.quad_reverse,
                                      single_to_larger_mapping=self.single_to_quad, pos_size=2)
        mm_table.identify_matches_mismatches()
        pos = (self.query_aln_fa_large.seq_length // 3, (2 * self.query_aln_fa_large.seq_length) // 3)
        indices = list(range(self.query_aln_fa_large.size // 3))
        indices1, indices2 = np.triu_indices(n=self.query_aln_fa_large.size // 3, k=1)
        test_chars, test_statuses = mm_table.get_upper_triangle(pos=pos, indices=indices)
        self.evaluate_get_characters_and_statuses(mm_table=mm_table, pos=pos, indices1=indices1, indices2=indices2,
                                                  test_chars=test_chars, test_statuses=test_statuses)

    def test7i_get_upper_rectangle(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.pair_size,
                                      larger_alphabet_mapping=self.pair_mapping,
                                      larger_alphabet_reverse_mapping=self.pair_reverse,
                                      single_to_larger_mapping=self.single_to_pair, pos_size=1)
        mm_table.identify_matches_mismatches()
        pos = self.query_aln_fa_small.seq_length // 2
        indices1 = np.array(range(self.query_aln_fa_small.size // 3))
        indices2 = np.array(range(((2 * self.query_aln_fa_small.size) // 3) + 1, self.query_aln_fa_small.size))
        final_indices1 = []
        final_indices2 = []
        for i in indices1:
            for j in indices2:
                if i > j:
                    final_indices1.append(j)
                    final_indices2.append(i)
                else:
                    final_indices1.append(i)
                    final_indices2.append(j)
        test_chars, test_statuses = mm_table.get_upper_rectangle(pos=pos, indices1=indices1, indices2=indices2)
        self.evaluate_get_characters_and_statuses(mm_table=mm_table, pos=pos, indices1=final_indices1,
                                                  indices2=final_indices2, test_chars=test_chars,
                                                  test_statuses=test_statuses)

    def test7j_get_upper_rectangle(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.pair_size,
                                      larger_alphabet_mapping=self.pair_mapping,
                                      larger_alphabet_reverse_mapping=self.pair_reverse,
                                      single_to_larger_mapping=self.single_to_pair, pos_size=1)
        mm_table.identify_matches_mismatches()
        pos = self.query_aln_fa_large.seq_length // 2
        indices1 = np.array(range(self.query_aln_fa_large.size // 3))
        indices2 = np.array(range(((2 * self.query_aln_fa_large.size) // 3) + 1, self.query_aln_fa_large.size))
        final_indices1 = []
        final_indices2 = []
        for i in indices1:
            for j in indices2:
                if i > j:
                    final_indices1.append(j)
                    final_indices2.append(i)
                else:
                    final_indices1.append(i)
                    final_indices2.append(j)
        test_chars, test_statuses = mm_table.get_upper_rectangle(pos=pos, indices1=indices1, indices2=indices2)
        self.evaluate_get_characters_and_statuses(mm_table=mm_table, pos=pos, indices1=final_indices1,
                                                  indices2=final_indices2, test_chars=test_chars,
                                                  test_statuses=test_statuses)

    def test7k_get_upper_rectangle(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.quad_size,
                                      larger_alphabet_mapping=self.quad_mapping,
                                      larger_alphabet_reverse_mapping=self.quad_reverse,
                                      single_to_larger_mapping=self.single_to_quad, pos_size=2)
        mm_table.identify_matches_mismatches()
        pos = (self.query_aln_fa_small.seq_length // 3, (2 * self.query_aln_fa_small.seq_length) // 3)
        indices1 = np.array(range(self.query_aln_fa_small.size // 3))
        indices2 = np.array(range(((2 * self.query_aln_fa_small.size) // 3) + 1, self.query_aln_fa_small.size))
        final_indices1 = []
        final_indices2 = []
        for i in indices1:
            for j in indices2:
                if i > j:
                    final_indices1.append(j)
                    final_indices2.append(i)
                else:
                    final_indices1.append(i)
                    final_indices2.append(j)
        test_chars, test_statuses = mm_table.get_upper_rectangle(pos=pos, indices1=indices1, indices2=indices2)
        self.evaluate_get_characters_and_statuses(mm_table=mm_table, pos=pos, indices1=final_indices1,
                                                  indices2=final_indices2, test_chars=test_chars,
                                                  test_statuses=test_statuses)

    def test7l_get_upper_rectangle(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.quad_size,
                                      larger_alphabet_mapping=self.quad_mapping,
                                      larger_alphabet_reverse_mapping=self.quad_reverse,
                                      single_to_larger_mapping=self.single_to_quad, pos_size=2)
        mm_table.identify_matches_mismatches()
        pos = (self.query_aln_fa_large.seq_length // 3, (2 * self.query_aln_fa_large.seq_length) // 3)
        indices1 = np.array(range(self.query_aln_fa_large.size // 3))
        indices2 = np.array(range(((2 * self.query_aln_fa_large.size) // 3) + 1, self.query_aln_fa_large.size))
        final_indices1 = []
        final_indices2 = []
        for i in indices1:
            for j in indices2:
                if i > j:
                    final_indices1.append(j)
                    final_indices2.append(i)
                else:
                    final_indices1.append(i)
                    final_indices2.append(j)
        test_chars, test_statuses = mm_table.get_upper_rectangle(pos=pos, indices1=indices1, indices2=indices2)
        self.evaluate_get_characters_and_statuses(mm_table=mm_table, pos=pos, indices1=final_indices1,
                                                  indices2=final_indices2, test_chars=test_chars,
                                                  test_statuses=test_statuses)


if __name__ == '__main__':
    unittest.main()