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
        self.assertEqual(mm_table.single_reverse_mapping, self.single_reverse)
        self.assertEqual(mm_table.larger_alphabet_size, large_alpha_size)
        self.assertEqual(mm_table.larger_mapping, large_alpha_map)
        self.assertEqual(mm_table.larger_reverse_mapping, large_alpha_reverse)
        self.assertEqual(mm_table.single_to_larger_mapping, single_to_large)
        self.assertIsNone(mm_table.match_mismatch_tables)
        self.assertIsNone(mm_table.match_freq_table)
        self.assertIsNone(mm_table.mismatch_freq_table)

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
                for j in range(num_aln.shape[0]):
                    if i < j:
                        if num_aln[i, pos] == num_aln[j, pos]:
                            self.assertEqual(mm_table.match_mismatch_tables[pos][i, j], 1)
                        else:
                            self.assertEqual(mm_table.match_mismatch_tables[pos][i, j], -1)
                    else:
                        self.assertEqual(mm_table.match_mismatch_tables[pos][i, j], 0)

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

    # def evaluate_single_match_mismatch_freq_tables(self, seq_len, num_aln, match_freq_table, mismatch_freq_table):
    #     for pos in range(seq_len):
    #         match_counts = {}
    #         mismatch_counts = {}
    #         for i in range(num_aln.shape[0]):
    #             for j in range(i+1, num_aln.shape[0]):
    #                 int_rep = self.single_to_pair[(num_aln[i, pos], num_aln[j, pos])]
    #                 char = self.pair_reverse[int_rep]
    #                 if num_aln[i, pos] == num_aln[j, pos]:
    #                     if char not in match_counts:
    #                         match_counts[char] = 0
    #                     match_counts[char] += 1
    #                 else:
    #                     if char not in mismatch_counts:
    #                         mismatch_counts[char] = 0
    #                     mismatch_counts[char] += 1
    #         for char in match_counts:
    #             self.assertEqual(match_freq_table.get_count(pos, char), match_counts[char])
    #         for char in mismatch_counts:
    #             self.assertEqual(mismatch_freq_table.get_count(pos, char), mismatch_counts[char])
    #
    # def evaluate__characterize_single_mm(self, seq_len, num_aln):
    #     mm_table = MatchMismatchTable(seq_len=seq_len, num_aln=num_aln, single_alphabet_size=self.single_size,
    #                                   single_mapping=self.single_mapping, single_reverse_mapping=self.single_reverse,
    #                                   larger_alphabet_size=self.pair_size, larger_alphabet_mapping=self.pair_mapping,
    #                                   larger_alphabet_reverse_mapping=self.pair_reverse,
    #                                   single_to_larger_mapping=self.single_to_pair, pos_size=1)
    #     mm_table.identify_matches_mismatches()
    #     match_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.single_mapping,
    #                                  reverse_mapping=self.pair_reverse, seq_len=seq_len, pos_size=1)
    #     match_table.mapping = self.pair_mapping
    #     mismatch_table = FrequencyTable(alphabet_size=self.pair_size, mapping=self.single_mapping,
    #                                     reverse_mapping=self.pair_reverse, seq_len=seq_len, pos_size=1)
    #     mismatch_table.mapping = self.pair_mapping
    #     mm_table._characterize_single_mm(match_table, mismatch_table)
    #     self.evaluate_single_match_mismatch_freq_tables(seq_len=seq_len, num_aln=num_aln, match_freq_table=match_table,
    #                                                     mismatch_freq_table=mismatch_table)
    #
    # def test3a__characterize_single_mm(self):
    #     num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
    #     self.evaluate__characterize_single_mm(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln)
    #
    # def test3b__characterize_single_mm(self):
    #     num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
    #     self.evaluate__characterize_single_mm(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln)
    #
    # def evaluate_pair_match_mismatch_freq_tables(self, seq_len, num_aln, match_freq_table, mismatch_freq_table):
    #     for pos1 in range(seq_len):
    #         for pos2 in range(pos1 + 1, seq_len):
    #             match_counts = {}
    #             mismatch_counts = {}
    #             # print(seq_len)
    #             # print(num_aln.shape)
    #             # print('Pos1: ', pos1)
    #             # print(num_aln[:, pos1])
    #             # print('Pos2: ', pos2)
    #             # print(num_aln[:, pos2])
    #             # print('EXPECTED MATCH/MISMATCH: ', ((num_aln.shape[0]**2) - num_aln.shape[0])/2.0)
    #             for i in range(num_aln.shape[0]):
    #                 for j in range(i+1, num_aln.shape[0]):
    #                     int_rep = self.single_to_quad[(num_aln[i, pos1], num_aln[j, pos1], num_aln[i, pos2],
    #                                                    num_aln[j, pos2])]
    #                     char = self.quad_reverse[int_rep]
    #                     if (((num_aln[i, pos1] == num_aln[j, pos1]) and (num_aln[i, pos2] == num_aln[j, pos2])) or
    #                             ((num_aln[i, pos1] != num_aln[j, pos1]) and (num_aln[i, pos2] != num_aln[j, pos2]))):
    #                         if char not in match_counts:
    #                             match_counts[char] = 0
    #                             # print('CHAR: ', char)
    #                             # print('Invariant Match: ', ((num_aln[i, pos1] == num_aln[j, pos1]) and (num_aln[i, pos2] == num_aln[j, pos2])))
    #                             # print('Concerted Change: ', ((num_aln[i, pos1] != num_aln[j, pos1]) and (num_aln[i, pos2] != num_aln[j, pos2])))
    #                         match_counts[char] += 1
    #                     else:
    #                         if char not in mismatch_counts:
    #                             mismatch_counts[char] = 0
    #                         mismatch_counts[char] += 1
    #             # match_freq = match_freq_table.get_count_array((pos1, pos2))
    #             # match_chars = match_freq_table.get_chars((pos1, pos2))
    #             # observed_match_counts = {match_chars[i]: match_freq[i] for i in range(len(match_chars))}
    #             # unique_matches = set(match_counts.keys()) | set(observed_match_counts.keys())
    #             # total_expected = 0
    #             # total_observed = 0
    #             # print('UNIQUE MATCHES:\tEXPECTED MATCHES:\tOBSERVED MATCHES:')
    #             # for um in unique_matches:
    #             #     expected_count = match_counts[um] if um in match_counts else 0
    #             #     total_expected += expected_count
    #             #     observed_count = observed_match_counts[um] if um in observed_match_counts else 0
    #             #     total_observed += observed_count
    #             #     print('{}\t\t{}\t\t{}'.format(um, expected_count, observed_count))
    #             # print('{}\t\t{}\t\t{}'.format('Total', total_expected, total_observed))
    #             # mismatch_freq = mismatch_freq_table.get_count_array((pos1, pos2))
    #             # mismatch_chars = mismatch_freq_table.get_chars((pos1, pos2))
    #             # observed_mismatch_counts = {mismatch_chars[i]: mismatch_freq[i] for i in range(len(mismatch_chars))}
    #             # unique_mismatches = set(mismatch_counts.keys()) | set(observed_mismatch_counts.keys())
    #             # print('UNIQUE MISMATCHES:\tEXPECTED MISMATCHES:\tOBSERVED MISMATCHES:')
    #             # total_expected = 0
    #             # total_observed = 0
    #             # for umm in unique_mismatches:
    #             #     expected_count = mismatch_counts[umm] if umm in mismatch_counts else 0
    #             #     total_expected += expected_count
    #             #     observed_count = observed_mismatch_counts[umm] if umm in observed_mismatch_counts else 0
    #             #     total_observed += observed_count
    #             #     print('{}\t\t{}\t\t{}'.format(umm, expected_count, observed_count))
    #             # print('{}\t\t{}\t\t{}'.format('Total', total_expected, total_observed))
    #             for char in match_counts:
    #                 self.assertEqual(int(match_freq_table.get_count((pos1, pos2), char)), match_counts[char])
    #             for char in mismatch_counts:
    #                 self.assertEqual(int(mismatch_freq_table.get_count((pos1, pos2), char)), mismatch_counts[char])
    #
    # def evaluate__characterize_pair_mm(self, seq_len, num_aln):
    #     mm_table = MatchMismatchTable(seq_len=seq_len, num_aln=num_aln, single_alphabet_size=self.single_size,
    #                                   single_mapping=self.single_mapping, single_reverse_mapping=self.single_reverse,
    #                                   larger_alphabet_size=self.quad_size, larger_alphabet_mapping=self.quad_mapping,
    #                                   larger_alphabet_reverse_mapping=self.quad_reverse,
    #                                   single_to_larger_mapping=self.single_to_quad, pos_size=2)
    #     mm_table.identify_matches_mismatches()
    #     match_table = FrequencyTable(alphabet_size=self.quad_size,
    #                                  mapping={'{0}{0}'.format(next(iter(self.single_mapping))): 0},
    #                                  reverse_mapping=self.quad_reverse, seq_len=seq_len, pos_size=2)
    #     match_table.mapping = self.quad_mapping
    #     mismatch_table = FrequencyTable(alphabet_size=self.quad_size,
    #                                     mapping={'{0}{0}'.format(next(iter(self.single_mapping))): 0},
    #                                     reverse_mapping=self.quad_reverse, seq_len=seq_len, pos_size=2)
    #     mismatch_table.mapping = self.quad_mapping
    #     mm_table._characterize_pair_mm(match_table, mismatch_table)
    #     self.evaluate_pair_match_mismatch_freq_tables(seq_len=seq_len, num_aln=num_aln, match_freq_table=match_table,
    #                                                   mismatch_freq_table=mismatch_table)
    #
    # def test3c__characterize_pair_mm(self):
    #     num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
    #     self.evaluate__characterize_pair_mm(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln)
    #
    # def test3d__characterize_pair_mm(self):
    #     num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
    #     self.evaluate__characterize_pair_mm(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln)
    #
    # def evaluate_characterize_matches_mismatches(self, match_mismatch_table):
    #     match_mismatch_table.identify_matches_mismatches()
    #     match_mismatch_table.characterize_matches_mismatches()
    #     expected_freq_table_depth = ((match_mismatch_table.seq_len**2) - match_mismatch_table.seq_len) / 2.0
    #     self.assertEqual(match_mismatch_table.match_freq_table.get_depth(), expected_freq_table_depth)
    #     self.assertEqual(match_mismatch_table.mismatch_freq_table.get_depth(), expected_freq_table_depth)
    #     if match_mismatch_table.pos_size == 1:
    #         self.evaluate_single_match_mismatch_freq_tables(seq_len=match_mismatch_table.seq_len,
    #                                                         num_aln=match_mismatch_table.num_aln,
    #                                                         match_freq_table=match_mismatch_table.match_freq_table,
    #                                                         mismatch_freq_table=match_mismatch_table.mismatch_freq_table)
    #     elif match_mismatch_table.pos_size == 2:
    #         self.evaluate_pair_match_mismatch_freq_tables(seq_len=match_mismatch_table.seq_len,
    #                                                       num_aln=match_mismatch_table.num_aln,
    #                                                       match_freq_table=match_mismatch_table.match_freq_table,
    #                                                       mismatch_freq_table=match_mismatch_table.mismatch_freq_table)
    #     else:
    #         raise ValueError('characterize_matches_mismatches() only implemented for pos_size 1 or 2')
    #
    # def test4a_characterize_matches_mismatches(self):
    #     num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
    #     mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
    #                                   single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
    #                                   single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.pair_size,
    #                                   larger_alphabet_mapping=self.pair_mapping,
    #                                   larger_alphabet_reverse_mapping=self.pair_reverse,
    #                                   single_to_larger_mapping=self.single_to_pair, pos_size=1)
    #     self.evaluate_characterize_matches_mismatches(match_mismatch_table=mm_table)
    #
    # def test4b_characterize_matches_mismatches(self):
    #     num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
    #     print('Seq length: ', self.query_aln_fa_small.seq_length)
    #     mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
    #                                   single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
    #                                   single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.quad_size,
    #                                   larger_alphabet_mapping=self.quad_mapping,
    #                                   larger_alphabet_reverse_mapping=self.quad_reverse,
    #                                   single_to_larger_mapping=self.single_to_quad, pos_size=2)
    #     self.evaluate_characterize_matches_mismatches(match_mismatch_table=mm_table)
    #
    # def test4c_characterize_matches_mismatches(self):
    #     num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
    #     mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
    #                                   single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
    #                                   single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.pair_size,
    #                                   larger_alphabet_mapping=self.pair_mapping,
    #                                   larger_alphabet_reverse_mapping=self.pair_reverse,
    #                                   single_to_larger_mapping=self.single_to_pair, pos_size=1)
    #     self.evaluate_characterize_matches_mismatches(match_mismatch_table=mm_table)
    #
    # def test4d_characterize_matches_mismatches(self):
    #     num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
    #     print('Seq length: ', self.query_aln_fa_large.seq_length)
    #     mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
    #                                   single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
    #                                   single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.quad_size,
    #                                   larger_alphabet_mapping=self.quad_mapping,
    #                                   larger_alphabet_reverse_mapping=self.quad_reverse,
    #                                   single_to_larger_mapping=self.single_to_quad, pos_size=2)
    #     self.evaluate_characterize_matches_mismatches(match_mismatch_table=mm_table)

    def evaluate_get_status_and_character(self, seq_len, num_aln, large_alpha_size, large_alpha_map,
                                          large_alpha_reverse, single_to_large, pos_size):
        mm_table = MatchMismatchTable(seq_len=seq_len, num_aln=num_aln, single_alphabet_size=self.single_size,
                                      single_mapping=self.single_mapping, single_reverse_mapping=self.single_reverse,
                                      larger_alphabet_size=large_alpha_size, larger_alphabet_mapping=large_alpha_map,
                                      larger_alphabet_reverse_mapping=large_alpha_reverse,
                                      single_to_larger_mapping=single_to_large, pos_size=pos_size)
        mm_table.identify_matches_mismatches()
        for s1 in range(num_aln.shape[0]):
            for s2 in range(s1 + 1, num_aln.shape[0]):
                for i in range(seq_len):
                    if pos_size == 1:
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

    def test5a_get_status_and_character(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_get_status_and_character(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                               large_alpha_size=self.pair_size, large_alpha_map=self.pair_mapping,
                                               large_alpha_reverse=self.pair_reverse,
                                               single_to_large=self.single_to_pair, pos_size=1)

    def test5b_get_status_and_character(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_get_status_and_character(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                               large_alpha_size=self.quad_size, large_alpha_map=self.quad_mapping,
                                               large_alpha_reverse=self.quad_reverse,
                                               single_to_large=self.single_to_quad, pos_size=2)

    def test5c_get_status_and_character(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_get_status_and_character(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                               large_alpha_size=self.pair_size, large_alpha_map=self.pair_mapping,
                                               large_alpha_reverse=self.pair_reverse,
                                               single_to_large=self.single_to_pair, pos_size=1)

    def test5d_get_status_and_character(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        self.evaluate_get_status_and_character(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                               large_alpha_size=self.quad_size, large_alpha_map=self.quad_mapping,
                                               large_alpha_reverse=self.quad_reverse,
                                               single_to_large=self.single_to_quad, pos_size=2)

    def evaluate_get_depth(self, aln, match_mismatch_table):
        self.assertEqual(aln.size, match_mismatch_table.get_depth())

    def test6a_get_depth(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.pair_size,
                                      larger_alphabet_mapping=self.pair_mapping,
                                      larger_alphabet_reverse_mapping=self.pair_reverse,
                                      single_to_larger_mapping=self.single_to_pair, pos_size=1)
        self.evaluate_get_depth(aln=self.query_aln_fa_small, match_mismatch_table=mm_table)

    def test6b_get_depth(self):
        num_aln = self.query_aln_fa_small._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_small.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.quad_size,
                                      larger_alphabet_mapping=self.quad_mapping,
                                      larger_alphabet_reverse_mapping=self.quad_reverse,
                                      single_to_larger_mapping=self.single_to_quad, pos_size=2)
        self.evaluate_get_depth(aln=self.query_aln_fa_small, match_mismatch_table=mm_table)

    def test6c_get_depth(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.pair_size,
                                      larger_alphabet_mapping=self.pair_mapping,
                                      larger_alphabet_reverse_mapping=self.pair_reverse,
                                      single_to_larger_mapping=self.single_to_pair, pos_size=1)
        self.evaluate_get_depth(aln=self.query_aln_fa_large, match_mismatch_table=mm_table)


    def test6d_get_depth(self):
        num_aln = self.query_aln_fa_large._alignment_to_num(mapping=self.single_mapping)
        mm_table = MatchMismatchTable(seq_len=self.query_aln_fa_large.seq_length, num_aln=num_aln,
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=self.quad_size,
                                      larger_alphabet_mapping=self.quad_mapping,
                                      larger_alphabet_reverse_mapping=self.quad_reverse,
                                      single_to_larger_mapping=self.single_to_quad, pos_size=2)
        self.evaluate_get_depth(aln=self.query_aln_fa_large, match_mismatch_table=mm_table)


if __name__ == '__main__':
    unittest.main()