"""
Created on Mar 4, 2020

@author: Daniel Konecki
"""
import os
import sys
import unittest
import numpy as np
from unittest import TestCase

#
from dotenv import find_dotenv, load_dotenv
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)
source_code_path = os.path.join(os.environ.get('PROJECT_PATH'), 'src')
# Add the project path to the python path so the required clases can be imported
if source_code_path not in sys.path:
    sys.path.append(os.path.join(os.environ.get('PROJECT_PATH'), 'src'))
#

from SupportingClasses.SeqAlignment import SeqAlignment
from SupportingClasses.MatchMismatchTable import MatchMismatchTable
from Testing.test_Base import (dna_alpha, dna_alpha_size, dna_map, dna_rev, protein_alpha, protein_alpha_size,
                               protein_map, protein_rev, pair_dna_alpha, dna_pair_alpha_size, dna_pair_map,
                               dna_pair_rev, quad_dna_alpha, dna_quad_alpha_size, dna_quad_map, dna_quad_rev,
                               pair_protein_alpha, pro_pair_alpha_size, pro_pair_map, pro_pair_rev, quad_protein_alpha,
                               pro_quad_alpha_size, pro_quad_map, pro_quad_rev, dna_single_to_pair_map,
                               dna_single_to_quad_map, pro_single_to_pair_map, pro_single_to_quad_map, protein_seq1,
                               protein_seq2, protein_seq3, protein_msa, dna_seq1, dna_seq2, dna_seq3, dna_msa,
                               generate_temp_fn, write_out_temp_fn)


class TestMatchMismatchTableInit(TestCase):

    def evaluate_init(self, aln, alpha_size, alpha_map, alpha_rev, large_alpha_size, large_alpha_map, large_alpha_rev,
                      small_to_large, seq_len, pos_size):
        num_aln = aln._alignment_to_num(mapping=alpha_map)
        mm_table = MatchMismatchTable(seq_len=seq_len, num_aln=num_aln, single_alphabet_size=alpha_size,
                                      single_mapping=alpha_map, single_reverse_mapping=alpha_rev,
                                      larger_alphabet_size=large_alpha_size, larger_mapping=large_alpha_map,
                                      larger_reverse_mapping=large_alpha_rev,
                                      single_to_larger_mapping=small_to_large, pos_size=pos_size)
        self.assertEqual(mm_table.seq_len, seq_len)
        self.assertEqual(mm_table.pos_size, pos_size)
        self.assertFalse((mm_table.num_aln - num_aln).any())
        self.assertEqual(mm_table.single_alphabet_size, alpha_size)
        self.assertEqual(mm_table.single_mapping, alpha_map)
        self.assertEqual(mm_table.single_reverse_mapping.tolist(), alpha_rev.tolist())
        self.assertEqual(mm_table.larger_alphabet_size, large_alpha_size)
        self.assertEqual(mm_table.larger_mapping, large_alpha_map)
        self.assertEqual(mm_table.larger_reverse_mapping.tolist(), large_alpha_rev.tolist())
        self.assertEqual(mm_table.single_to_larger_mapping, small_to_large)
        self.assertIsNone(mm_table.match_mismatch_tables)

    def test_init_single_pos(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        self.evaluate_init(aln=aln, alpha_size=dna_alpha_size, alpha_map=dna_map, alpha_rev=dna_rev,
                           large_alpha_size=dna_pair_alpha_size, large_alpha_map=dna_pair_map,
                           large_alpha_rev=dna_pair_rev, small_to_large=dna_single_to_pair_map, seq_len=18, pos_size=1)
        os.remove(aln_fn)

    def test_init_pair_pos(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        self.evaluate_init(aln=aln, alpha_size=dna_alpha_size, alpha_map=dna_map, alpha_rev=dna_rev,
                           large_alpha_size=dna_quad_alpha_size, large_alpha_map=dna_quad_map,
                           large_alpha_rev=dna_quad_rev, small_to_large=dna_single_to_quad_map, seq_len=18, pos_size=2)
        os.remove(aln_fn)

    def test_init_failure_pair_to_quad(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        dna_pair_to_quad = {}
        with self.assertRaises(ValueError):
            mm_table = MatchMismatchTable(seq_len=18, num_aln=num_aln, single_alphabet_size=dna_pair_alpha_size,
                                          single_mapping=dna_pair_map, single_reverse_mapping=dna_pair_rev,
                                          larger_alphabet_size=dna_quad_alpha_size, larger_mapping=dna_quad_map,
                                          larger_reverse_mapping=dna_quad_rev,
                                          single_to_larger_mapping=dna_pair_to_quad, pos_size=2)
        os.remove(aln_fn)

    def test_init_failure_pair_to_single(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(dna_seq1.seq)}\n>seq2\n{str(dna_seq2.seq)}\n>seq3\n{str(dna_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='DNA')
        aln.import_alignment()
        num_aln = aln._alignment_to_num(mapping=dna_map)
        dna_pair_to_single = {}
        with self.assertRaises(ValueError):
            mm_table = MatchMismatchTable(seq_len=18, num_aln=num_aln, single_alphabet_size=dna_pair_alpha_size,
                                          single_mapping=dna_pair_map, single_reverse_mapping=dna_pair_rev,
                                          larger_alphabet_size=dna_alpha_size, larger_mapping=dna_map,
                                          larger_reverse_mapping=dna_rev,
                                          single_to_larger_mapping=dna_pair_to_single, pos_size=2)
        os.remove(aln_fn)

    def test_init_failure_no_num_aln(self):
        with self.assertRaises(AttributeError):
            mm_table = MatchMismatchTable(seq_len=18, num_aln=None, single_alphabet_size=dna_alpha_size,
                                          single_mapping=dna_map, single_reverse_mapping=dna_rev,
                                          larger_alphabet_size=dna_pair_alpha_size, larger_mapping=dna_pair_map,
                                          larger_reverse_mapping=dna_pair_rev,
                                          single_to_larger_mapping=dna_single_to_pair_map, pos_size=1)


class TestMatchMismatchTableIdentifyMatchesMismatches(TestCase):

    def evaluate_identify_matches_mismatches(self, aln, seq_len, pos_size, depth, alpha_size, alpha_map, alpha_rev,
                                             pair_alpha_size, pair_map, pair_rev, small_to_large):
        num_aln = aln._alignment_to_num(mapping=alpha_map)
        mm_table = MatchMismatchTable(seq_len=seq_len, num_aln=num_aln, single_alphabet_size=alpha_size,
                                      single_mapping=alpha_map, single_reverse_mapping=alpha_rev,
                                      larger_alphabet_size=pair_alpha_size, larger_mapping=pair_map,
                                      larger_reverse_mapping=pair_rev, single_to_larger_mapping=small_to_large,
                                      pos_size=pos_size)
        mm_table.identify_matches_mismatches()
        for i in range(seq_len):
            self.assertTrue(i in mm_table.match_mismatch_tables)
            expected_table = np.zeros((depth, depth))
            for j in range(depth):
                for k in range(j + 1, depth):
                    if num_aln[j, i] == num_aln[k, i]:
                        expected_table[j, k] = 1
                    else:
                        expected_table[j, k] = -1
            self.assertFalse((mm_table.match_mismatch_tables[i] - expected_table).any())

    def test_identify_matches_mismatches_single_to_pair(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        self.evaluate_identify_matches_mismatches(aln=aln, seq_len=6, pos_size=1, depth=3,
                                                  alpha_size=protein_alpha_size, alpha_map=protein_map,
                                                  alpha_rev=protein_rev, pair_alpha_size=pro_pair_alpha_size,
                                                  pair_map=pro_pair_map, pair_rev=pro_pair_rev,
                                                  small_to_large=pro_single_to_pair_map)

    def test_identify_matches_mismatches_pair_to_quad(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        self.evaluate_identify_matches_mismatches(aln=aln, seq_len=6, pos_size=2, depth=3,
                                                  alpha_size=protein_alpha_size, alpha_map=protein_map,
                                                  alpha_rev=protein_rev, pair_alpha_size=pro_quad_alpha_size,
                                                  pair_map=pro_quad_map, pair_rev=pro_quad_rev,
                                                  small_to_large=pro_single_to_quad_map)


class TestMatchMismatchTable(TestCase):

    def test_get_status_and_character_single_pos(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        mm_table.identify_matches_mismatches()
        for i in range(6):
            for j in range(3):
                for k in range(j + 1, 3):
                    status, curr_char = mm_table.get_status_and_character(pos=i, seq_ind1=j, seq_ind2=k)
                    if num_aln[j, i] == num_aln[k, i]:
                        self.assertEqual(status, 'match')
                    else:
                        self.assertEqual(status, 'mismatch')
                    self.assertEqual(curr_char, protein_msa[j, i] + protein_msa[k, i])

    def test_get_status_and_character_pair_pos(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse_mapping=pro_quad_rev,
                                      single_to_larger_mapping=pro_single_to_quad_map, pos_size=2)
        mm_table.identify_matches_mismatches()
        for i in range(6):
            for j in range(i + 1, 6):
                for k in range(3):
                    for l in range(k + 1, 3):
                        status, curr_char = mm_table.get_status_and_character(pos=(i, j), seq_ind1=k, seq_ind2=l)
                        pair_k = protein_msa[k, i] + protein_msa[k, j]
                        pair_l = protein_msa[l, i] + protein_msa[l, j]
                        if (pair_k == pair_l) or ((protein_msa[k, i] != protein_msa[l, i]) and
                                                  (protein_msa[k, j] != protein_msa[l, j])):
                            self.assertEqual(status, 'match')
                        else:
                            self.assertEqual(status, 'mismatch')
                        self.assertEqual(curr_char, pair_k + pair_l)

    def test_get_status_and_character_failure_not_initialized(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        with self.assertRaises(AttributeError):
            mm_table.get_status_and_character(pos=0, seq_ind1=0, seq_ind2=1)

    def test_get_status_and_character_failure_disordered_seq_ind(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        mm_table.identify_matches_mismatches()
        with self.assertRaises(ValueError):
            mm_table.get_status_and_character(pos=0, seq_ind1=1, seq_ind2=0)

    def test_get_status_and_character_failure_pos_size_mismatch_bigger(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        mm_table.identify_matches_mismatches()
        with self.assertRaises(ValueError):
            mm_table.get_status_and_character(pos=(0, 1), seq_ind1=0, seq_ind2=1)

    def test_get_status_and_character_failure_pos_size_mismatch_smaller(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse_mapping=pro_quad_rev,
                                      single_to_larger_mapping=pro_single_to_quad_map, pos_size=2)
        mm_table.identify_matches_mismatches()
        with self.assertRaises(ValueError):
            mm_table.get_status_and_character(pos=0, seq_ind1=0, seq_ind2=1)

    def test_get_status_and_character_failure_bad_pos(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        mm_table.identify_matches_mismatches()
        with self.assertRaises(ValueError):
            mm_table.get_status_and_character(pos=[1, 2], seq_ind1=0, seq_ind2=1)


class TestMatchMismatchTableGetDepth(TestCase):

    def test_get_depth_single(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=np.array([num_aln[0, :].tolist()]),
                                      single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        self.assertEqual(mm_table.get_depth(), 1)

    def test_get_depth_double(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln[:2, :], single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        self.assertEqual(mm_table.get_depth(), 2)

    def test_get_depth_triple(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        self.assertEqual(mm_table.get_depth(), 3)


class TestMatchMismatchTableGetSubTable(TestCase):

    def test__get_characters_and_statuses_single_pos_correctly_ordered(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        mm_table.identify_matches_mismatches()
        ind1 = [0, 0, 1]
        ind2 = [1, 2, 2]
        for p in range(6):
            chars1, chars2, statuses = mm_table._get_characters_and_statuses_single_pos(pos=p, indices1=ind1,
                                                                                        indices2=ind2)
            expected_chars1 = np.array([[protein_map[protein_msa[x, p]]] for x in ind1])
            expected_chars2 = np.array([[protein_map[protein_msa[x, p]]] for x in ind2])
            expected_statuses = np.zeros(3)
            for i in range(3):
                expected_statuses[i] = 1 if expected_chars1[i] == expected_chars2[i] else -1
            self.assertFalse((chars1 - expected_chars1).any())
            self.assertFalse((chars2 - expected_chars2).any())
            self.assertFalse((statuses - expected_statuses).any())

    def test__get_characters_and_statuses_single_pos_incorrectly_ordered(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        mm_table.identify_matches_mismatches()
        ind1 = [1, 2, 2]
        ind2 = [0, 0, 1]
        for p in range(6):
            chars1, chars2, statuses = mm_table._get_characters_and_statuses_single_pos(pos=p, indices1=ind1,
                                                                                        indices2=ind2)
            expected_chars1 = np.array([[protein_map[protein_msa[x, p]]] for x in ind1])
            expected_chars2 = np.array([[protein_map[protein_msa[x, p]]] for x in ind2])
            expected_statuses = np.zeros(3)
            self.assertFalse((chars1 - expected_chars1).any())
            self.assertFalse((chars2 - expected_chars2).any())
            self.assertFalse((statuses - expected_statuses).any())

    def test__get_characters_and_statuses_single_pos_failure_uninitialized(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        with self.assertRaises(AttributeError):
            mm_table._get_characters_and_statuses_single_pos(pos=0, indices1=[0, 1, 2], indices2=[0, 1, 2])

    def test__get_characters_and_statuses_multi_pos_pair_correctly_ordered(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse_mapping=pro_quad_rev,
                                      single_to_larger_mapping=pro_single_to_quad_map, pos_size=2)
        mm_table.identify_matches_mismatches()
        ind1 = [0, 0, 1]
        ind2 = [1, 2, 2]
        for p1 in range(6):
            for p2 in range(p1 + 1, 6):
                chars1, chars2, statuses = mm_table._get_characters_and_statuses_multi_pos(pos=(p1, p2), indices1=ind1,
                                                                                           indices2=ind2)
                expected_chars1 = np.array([[protein_map[protein_msa[x, p1]],
                                             protein_map[protein_msa[x, p2]]] for x in ind1])
                expected_chars2 = np.array([[protein_map[protein_msa[x, p1]],
                                             protein_map[protein_msa[x, p2]]] for x in ind2])
                expected_statuses = np.zeros(3)
                for i in range(3):
                    for j in range(2):
                        expected_statuses[i] += 1 if expected_chars1[i, j] == expected_chars2[i, j] else -1
                self.assertFalse((chars1 - expected_chars1).any())
                self.assertFalse((chars2 - expected_chars2).any())
                self.assertFalse((statuses - expected_statuses).any())

    def test__get_characters_and_statuses_multi_pos_pair_incorrectly_ordered(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse_mapping=pro_quad_rev,
                                      single_to_larger_mapping=pro_single_to_quad_map, pos_size=2)
        mm_table.identify_matches_mismatches()
        ind1 = [1, 2, 2]
        ind2 = [0, 0, 1]
        for p1 in range(6):
            for p2 in range(p1 + 1, 6):
                chars1, chars2, statuses = mm_table._get_characters_and_statuses_multi_pos(pos=(p1, p2), indices1=ind1,
                                                                                           indices2=ind2)
                expected_chars1 = np.array([[protein_map[protein_msa[x, p1]],
                                             protein_map[protein_msa[x, p2]]] for x in ind1])
                expected_chars2 = np.array([[protein_map[protein_msa[x, p1]],
                                             protein_map[protein_msa[x, p2]]] for x in ind2])
                expected_statuses = np.zeros(3)
                self.assertFalse((chars1 - expected_chars1).any())
                self.assertFalse((chars2 - expected_chars2).any())
                self.assertFalse((statuses - expected_statuses).any())

    def test__get_characters_and_statuses_multi_pos_pair_failure_uninitiated(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse_mapping=pro_quad_rev,
                                      single_to_larger_mapping=pro_single_to_quad_map, pos_size=2)
        with self.assertRaises(AttributeError):
            mm_table._get_characters_and_statuses_single_pos(pos=(0, 1), indices1=[0, 1, 2], indices2=[0, 1, 2])

    def test_get_upper_triangle_single_correctly_ordered(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        mm_table.identify_matches_mismatches()
        ind = [0, 1, 2]
        ind1 = [0, 0, 1]
        ind2 = [1, 2, 2]
        for p in range(6):
            curr_chars, statuses = mm_table.get_upper_triangle(pos=p, indices=ind)
            expected_chars = np.array([protein_msa[ind1[x], p] + protein_msa[ind2[x], p] for x in range(3)])
            self.assertEqual(curr_chars.tolist(), expected_chars.tolist())
            expected_statuses = ['match' if expected_chars[i][0] == expected_chars[i][1] else 'mismatch'
                                 for i in range(3)]
            self.assertEqual(statuses.tolist(), expected_statuses)

    def test_get_upper_triangle_single_incorrectly_ordered(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        mm_table.identify_matches_mismatches()
        ind = [1, 2, 0]
        ind1 = [0, 0, 1]
        ind2 = [1, 2, 2]
        for p in range(6):
            curr_chars, statuses = mm_table.get_upper_triangle(pos=p, indices=ind)
            expected_chars = np.array([protein_msa[ind1[x], p] + protein_msa[ind2[x], p] for x in range(3)])
            self.assertEqual(curr_chars.tolist(), expected_chars.tolist())
            expected_statuses = ['match' if expected_chars[i][0] == expected_chars[i][1] else 'mismatch'
                                 for i in range(3)]
            self.assertEqual(statuses.tolist(), expected_statuses)

    def test_get_upper_triangle_single_failure_uninitiated(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        with self.assertRaises(AttributeError):
            mm_table.get_upper_triangle(pos=0, indices=[1, 2])

    def test_get_upper_triangle_pair_correctly_ordered(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse_mapping=pro_quad_rev,
                                      single_to_larger_mapping=pro_single_to_quad_map, pos_size=2)
        mm_table.identify_matches_mismatches()
        ind = [0, 1, 2]
        ind1 = [0, 0, 1]
        ind2 = [1, 2, 2]
        for p1 in range(6):
            for p2 in range(p1 + 1, 6):
                curr_chars, statuses = mm_table.get_upper_triangle(pos=(p1, p2), indices=ind)
                expected_chars = np.array([(protein_msa[ind1[x], p1] + protein_msa[ind1[x], p2] +
                                            protein_msa[ind2[x], p1] + protein_msa[ind2[x], p2]) for x in range(3)])
                self.assertEqual(curr_chars.tolist(), expected_chars.tolist())
                expected_statuses = ['match' if ((expected_chars[i][:2] == expected_chars[i][2:]) or
                                                 ((expected_chars[i][0] != expected_chars[i][2]) and
                                                  (expected_chars[i][1] != expected_chars[i][3]))) else 'mismatch'
                                     for i in range(3)]
                self.assertEqual(statuses.tolist(), expected_statuses)

    def test_get_upper_triangle_pair_incorrectly_ordered(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse_mapping=pro_quad_rev,
                                      single_to_larger_mapping=pro_single_to_quad_map, pos_size=2)
        mm_table.identify_matches_mismatches()
        ind = [1, 2, 0]
        ind1 = [0, 0, 1]
        ind2 = [1, 2, 2]
        for p1 in range(6):
            for p2 in range(p1 + 1, 6):
                curr_chars, statuses = mm_table.get_upper_triangle(pos=(p1, p2), indices=ind)
                expected_chars = np.array([(protein_msa[ind1[x], p1] + protein_msa[ind1[x], p2] +
                                            protein_msa[ind2[x], p1] + protein_msa[ind2[x], p2]) for x in range(3)])
                self.assertEqual(curr_chars.tolist(), expected_chars.tolist())
                expected_statuses = ['match' if ((expected_chars[i][:2] == expected_chars[i][2:]) or
                                                 ((expected_chars[i][0] != expected_chars[i][2]) and
                                                  (expected_chars[i][1] != expected_chars[i][3]))) else 'mismatch'
                                     for i in range(3)]
                self.assertEqual(statuses.tolist(), expected_statuses)

    def test_get_upper_triangle_pair_failure_uninitiated(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse_mapping=pro_quad_rev,
                                      single_to_larger_mapping=pro_single_to_quad_map, pos_size=2)
        with self.assertRaises(AttributeError):
            mm_table.get_upper_triangle(pos=(0, 1), indices=[1, 2])

    # def test_get_upper_rectangle_single(self):
    # def test_get_upper_rectangle_pair(self):

    def test_get_upper_rectangle_single_correctly_ordered(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        mm_table.identify_matches_mismatches()
        full_ind1 = [0, 0]
        ind1 = [0]
        ind2 = [1, 2]
        for p in range(6):
            curr_chars, statuses = mm_table.get_upper_rectangle(pos=p, indices1=ind1, indices2=ind2)
            expected_chars = np.array([protein_msa[full_ind1[x], p] + protein_msa[ind2[x], p] for x in range(2)])
            self.assertEqual(curr_chars.tolist(), expected_chars.tolist())
            expected_statuses = ['match' if expected_chars[i][0] == expected_chars[i][1] else 'mismatch'
                                 for i in range(2)]
            self.assertEqual(statuses.tolist(), expected_statuses)

    def test_get_upper_rectangle_single_incorrectly_ordered(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        mm_table.identify_matches_mismatches()
        full_ind1 = [0, 0]
        full_ind2 = [1, 2]
        ind1 = [1, 2]
        ind2 = [0]
        for p in range(6):
            curr_chars, statuses = mm_table.get_upper_rectangle(pos=p, indices1=ind1, indices2=ind2)
            expected_chars = np.array([protein_msa[full_ind1[x], p] + protein_msa[full_ind2[x], p] for x in range(2)])
            self.assertEqual(curr_chars.tolist(), expected_chars.tolist())
            expected_statuses = ['match' if expected_chars[i][0] == expected_chars[i][1] else 'mismatch'
                                 for i in range(2)]
            self.assertEqual(statuses.tolist(), expected_statuses)

    def test_get_upper_rectangle_single_failure_uninitiated(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
        with self.assertRaises(AttributeError):
            mm_table.get_upper_rectangle(pos=0, indices1=[0], indices2=[1, 2])

    def test_get_upper_rectangle_pair_correctly_ordered(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse_mapping=pro_quad_rev,
                                      single_to_larger_mapping=pro_single_to_quad_map, pos_size=2)
        mm_table.identify_matches_mismatches()
        full_ind1 = [0, 0]
        ind1 = [0]
        ind2 = [1, 2]
        for p1 in range(6):
            for p2 in range(p1 + 1, 6):
                curr_chars, statuses = mm_table.get_upper_rectangle(pos=(p1, p2), indices1=ind1, indices2=ind2)
                expected_chars = np.array([(protein_msa[full_ind1[x], p1] + protein_msa[full_ind1[x], p2] +
                                            protein_msa[ind2[x], p1] + protein_msa[ind2[x], p2]) for x in range(2)])
                self.assertEqual(curr_chars.tolist(), expected_chars.tolist())
                expected_statuses = ['match' if ((expected_chars[i][:2] == expected_chars[i][2:]) or
                                                 ((expected_chars[i][0] != expected_chars[i][2]) and
                                                  (expected_chars[i][1] != expected_chars[i][3]))) else 'mismatch'
                                     for i in range(2)]
                self.assertEqual(statuses.tolist(), expected_statuses)

    def test_get_upper_rectangle_pair_incorrectly_ordered(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse_mapping=pro_quad_rev,
                                      single_to_larger_mapping=pro_single_to_quad_map, pos_size=2)
        mm_table.identify_matches_mismatches()
        full_ind1 = [0, 0]
        full_ind2 = [1, 2]
        ind1 = [1, 2]
        ind2 = [0]
        for p1 in range(6):
            for p2 in range(p1 + 1, 6):
                curr_chars, statuses = mm_table.get_upper_rectangle(pos=(p1, p2), indices1=ind1, indices2=ind2)
                expected_chars = np.array([(protein_msa[full_ind1[x], p1] + protein_msa[full_ind1[x], p2] +
                                            protein_msa[full_ind2[x], p1] + protein_msa[full_ind2[x], p2])
                                           for x in range(2)])
                self.assertEqual(curr_chars.tolist(), expected_chars.tolist())
                expected_statuses = ['match' if ((expected_chars[i][:2] == expected_chars[i][2:]) or
                                                 ((expected_chars[i][0] != expected_chars[i][2]) and
                                                  (expected_chars[i][1] != expected_chars[i][3]))) else 'mismatch'
                                     for i in range(2)]
                self.assertEqual(statuses.tolist(), expected_statuses)

    def test_get_upper_rectangle_pair_failure_uninitiated(self):
        aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
        aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
        aln.import_alignment()
        os.remove(aln_fn)
        num_aln = aln._alignment_to_num(mapping=protein_map)
        mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse_mapping=pro_quad_rev,
                                      single_to_larger_mapping=pro_single_to_quad_map, pos_size=2)
        with self.assertRaises(AttributeError):
            mm_table.get_upper_rectangle(pos=(0, 1), indices1=[0], indices2=[1, 2])


if __name__ == '__main__':
    unittest.main()