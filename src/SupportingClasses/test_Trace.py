"""
Created on July 11, 2019

@author: Daniel Konecki
"""
import os
import unittest
from unittest import TestCase
import numpy as np
from copy import deepcopy
from shutil import rmtree
from multiprocessing import Lock, Manager, Queue
from Bio.Alphabet import Gapped
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from test_Base import TestBase
from utils import build_mapping, gap_characters
from ETMIPWrapper import ETMIPWrapper
from SeqAlignment import SeqAlignment
from FrequencyTable import FrequencyTable
from PhylogeneticTree import PhylogeneticTree
from PositionalScorer import PositionalScorer
from MatchMismatchTable import MatchMismatchTable
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from test_seqAlignment import generate_temp_fn, write_out_temp_fasta
from EvolutionaryTraceAlphabet import FullIUPACProtein, MultiPositionAlphabet
from Trace import (Trace, init_characterization_pool, init_characterization_mm_pool, characterization,
                   characterization_mm, init_trace_groups, trace_groups, init_trace_ranks, trace_ranks,
                   check_freq_table, save_freq_table, load_freq_table,
                   check_numpy_array, save_numpy_array, load_numpy_array)

protein_alpha = Gapped(FullIUPACProtein())
protein_alpha_size, _, protein_map, protein_rev = build_mapping(protein_alpha)
pair_protein_alpha = MultiPositionAlphabet(protein_alpha, size=2)
pro_pair_alpha_size, _, pro_pair_map, pro_pair_rev = build_mapping(pair_protein_alpha)
quad_protein_alpha = MultiPositionAlphabet(protein_alpha, size=4)
pro_quad_alpha_size, _, pro_quad_map, pro_quad_rev = build_mapping(quad_protein_alpha)
pro_single_to_pair = np.zeros((max(protein_map.values()) + 1, max(protein_map.values()) + 1), dtype=np.int)
pro_single_to_pair_map = {}
for char in pro_pair_map:
    pro_single_to_pair[protein_map[char[0]], protein_map[char[1]]] = pro_pair_map[char]
    pro_single_to_pair_map[(protein_map[char[0]], protein_map[char[1]])] = pro_pair_map[char]
pro_single_to_quad = {}
for char in pro_quad_map:
    key = (protein_map[char[0]], protein_map[char[1]], protein_map[char[2]], protein_map[char[3]])
    pro_single_to_quad[key] = pro_quad_map[char]
protein_seq1 = SeqRecord(id='seq1', seq=Seq('MET---', alphabet=FullIUPACProtein()))
protein_seq2 = SeqRecord(id='seq2', seq=Seq('M-TREE', alphabet=FullIUPACProtein()))
protein_seq3 = SeqRecord(id='seq3', seq=Seq('M-FREE', alphabet=FullIUPACProtein()))
protein_msa = MultipleSeqAlignment(records=[protein_seq1, protein_seq2, protein_seq3], alphabet=FullIUPACProtein())

aln_fn = write_out_temp_fasta(
                out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
aln.import_alignment()
os.remove(aln_fn)
num_aln = aln._alignment_to_num(mapping=protein_map)

adc = AlignmentDistanceCalculator(model='identity')
dm = adc.get_distance(msa=protein_msa, processes=2)

phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
phylo_tree.construct_tree(dm=dm)
rank_dict = phylo_tree.assign_group_rank(ranks=None)

pro_single_ft = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
pro_single_ft.characterize_alignment(num_aln=num_aln)
pro_single_ft_i2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
pro_single_ft_i2.characterize_alignment(num_aln=num_aln[[1, 2], :])
pro_single_ft_s1 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
pro_single_ft_s1.characterize_alignment(num_aln=np.array([num_aln[0, :]]))
pro_single_ft_s2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
pro_single_ft_s2.characterize_alignment(num_aln=np.array([num_aln[1, :]]))
pro_single_ft_s3 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
pro_single_ft_s3.characterize_alignment(num_aln=np.array([num_aln[2, :]]))

pro_pair_ft = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
pro_pair_ft.characterize_alignment(num_aln=num_aln, single_to_pair=pro_single_to_pair)
pro_pair_ft_i2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
pro_pair_ft_i2.characterize_alignment(num_aln=num_aln[[1, 2], :], single_to_pair=pro_single_to_pair)
pro_pair_ft_s1 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
pro_pair_ft_s1.characterize_alignment(num_aln=np.array([num_aln[0, :]]), single_to_pair=pro_single_to_pair)
pro_pair_ft_s2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
pro_pair_ft_s2.characterize_alignment(num_aln=np.array([num_aln[1, :]]), single_to_pair=pro_single_to_pair)
pro_pair_ft_s3 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
pro_pair_ft_s3.characterize_alignment(num_aln=np.array([num_aln[2, :]]), single_to_pair=pro_single_to_pair)

protein_mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
protein_mm_table.identify_matches_mismatches()

protein_mm_table_large = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                                            single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                            larger_alphabet_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                            larger_reverse_mapping=pro_quad_rev,
                                            single_to_larger_mapping=pro_single_to_quad, pos_size=2)
protein_mm_table_large.identify_matches_mismatches()

expected_single_tables = {'Inner1': {'match': FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1),
                                     'mismatch': FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)},
                          'Inner2': {'match': FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1),
                                     'mismatch': FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)}}
expected_pair_tables = {'Inner1': {'match': FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2),
                                   'mismatch': FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)},
                        'Inner2': {'match': FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2),
                                   'mismatch': FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)}}
for s1 in range(3):
    for s2 in range(s1 + 1, 3):
        for pos1 in range(6):
            single_stat, single_char = protein_mm_table.get_status_and_character(pos=pos1, seq_ind1=s1, seq_ind2=s2)
            if single_stat == 'match':
                expected_single_tables['Inner1']['match']._increment_count(pos=pos1, char=single_char)
                if s1 > 0 and s2 > 0:
                    expected_single_tables['Inner2']['match']._increment_count(pos=pos1, char=single_char)
            else:
                expected_single_tables['Inner1']['mismatch']._increment_count(pos=pos1, char=single_char)
                if s1 > 0 and s2 > 0:
                    expected_single_tables['Inner2']['mismatch']._increment_count(pos=pos1, char=single_char)
            for pos2 in range(pos1, 6):
                pair_stat, pair_char = protein_mm_table_large.get_status_and_character(pos=(pos1, pos2), seq_ind1=s1,
                                                                                       seq_ind2=s2)
                if pair_stat == 'match':
                    expected_pair_tables['Inner1']['match']._increment_count(pos=(pos1, pos2), char=pair_char)
                    if s1 > 0 and s2 > 0:
                        expected_pair_tables['Inner2']['match']._increment_count(pos=(pos1, pos2), char=pair_char)
                else:
                    expected_pair_tables['Inner1']['mismatch']._increment_count(pos=(pos1, pos2), char=pair_char)
                    if s1 > 0 and s2 > 0:
                        expected_pair_tables['Inner2']['mismatch']._increment_count(pos=(pos1, pos2), char=pair_char)
for node in expected_single_tables:
    if node == 'Inner1':
        expected_depth = 3
    else:
        expected_depth = 1
    for stat in expected_single_tables[node]:
        expected_single_tables[node][stat].set_depth(expected_depth)
        expected_pair_tables[node][stat].set_depth(expected_depth)
        expected_single_tables[node][stat].finalize_table()
        expected_pair_tables[node][stat].finalize_table()


class TestTraceCheckFreqTable(TestCase):

    def test_check_freq_table_not_low_memory_no_other_variables(self):
        check, fn = check_freq_table(low_memory=False, node_name=None, table_type=None, out_dir=None)
        self.assertFalse(check)
        self.assertIsNone(fn)

    def test_check_freq_table_not_low_memory_node_name_only(self):
        check, fn = check_freq_table(low_memory=False, node_name='Inner1', table_type=None, out_dir=None)
        self.assertFalse(check)
        self.assertIsNone(fn)

    def test_check_freq_table_not_low_memory_table_type_only(self):
        check, fn = check_freq_table(low_memory=False, node_name=None, table_type='single', out_dir=None)
        self.assertFalse(check)
        self.assertIsNone(fn)

    def test_check_freq_table_not_low_memory_out_dir_only(self):
        check, fn = check_freq_table(low_memory=False, node_name=None, table_type=None, out_dir=os.getcwd())
        self.assertFalse(check)
        self.assertIsNone(fn)

    def test_check_freq_table_not_low_memory_all_variables(self):
        check, fn = check_freq_table(low_memory=False, node_name='Inner1', table_type='single', out_dir=os.getcwd())
        self.assertFalse(check)
        self.assertIsNone(fn)

    def test_check_freq_table_low_memory_no_other_variables(self):
        with self.assertRaises(ValueError):
            check_freq_table(low_memory=True, node_name=None, table_type=None, out_dir=None)

    def test_check_freq_table_low_memory_node_name_only(self):
        with self.assertRaises(ValueError):
            check_freq_table(low_memory=True, node_name='Inner1', table_type=None, out_dir=None)

    def test_check_freq_table_low_memory_table_type_only(self):
        with self.assertRaises(ValueError):
            check_freq_table(low_memory=True, node_name=None, table_type='single', out_dir=None)

    def test_check_freq_table_low_memory_out_dir_only(self):
        with self.assertRaises(ValueError):
            check_freq_table(low_memory=True, node_name=None, table_type=None, out_dir=os.getcwd())

    def test_check_freq_table_low_memory_all_variables(self):
        check, fn = check_freq_table(low_memory=True, node_name='Inner1', table_type='single', out_dir=os.getcwd())
        self.assertFalse(check)
        self.assertEqual(fn, os.path.join(os.getcwd(), 'Inner1_single_freq_table.pkl'))

    def test_check_freq_table_low_memory_all_variables_file_exists(self):
        expected_fn = os.path.join(os.getcwd(), 'Inner1_single_freq_table.pkl')
        with open(expected_fn, 'a'):
            os.utime(expected_fn, None)
        check, fn = check_freq_table(low_memory=True, node_name='Inner1', table_type='single', out_dir=os.getcwd())
        self.assertTrue(check)
        self.assertEqual(fn, expected_fn)
        os.remove(expected_fn)


class TestTraceSaveFreqTable(TestCase):

    def test_save_freq_table_not_low_memory_no_other_variables(self):
        ft = save_freq_table(freq_table=pro_single_ft, low_memory=False, node_name=None, table_type=None, out_dir=None)
        self.assertIs(ft, pro_single_ft)

    def test_save_freq_table_not_low_memory_node_name_only(self):
        ft = save_freq_table(freq_table=pro_single_ft, low_memory=False, node_name='Inner1', table_type=None,
                             out_dir=None)
        self.assertIs(ft, pro_single_ft)

    def test_save_freq_table_not_low_memory_table_type_only(self):
        ft = save_freq_table(freq_table=pro_single_ft, low_memory=False, node_name=None, table_type='single',
                             out_dir=None)
        self.assertIs(ft, pro_single_ft)

    def test_save_freq_table_not_low_memory_out_dir_only(self):
        ft = save_freq_table(freq_table=pro_single_ft, low_memory=False, node_name=None, table_type=None,
                             out_dir=os.getcwd())
        self.assertIs(ft, pro_single_ft)

    def test_save_freq_table_not_low_memory_all_variables(self):
        ft = save_freq_table(freq_table=pro_single_ft, low_memory=False, node_name='Inner1', table_type='single',
                             out_dir=os.getcwd())
        self.assertIs(ft, pro_single_ft)

    def test_save_freq_table_low_memory_no_other_variables(self):
        with self.assertRaises(ValueError):
            save_freq_table(freq_table=pro_single_ft, low_memory=True, node_name=None, table_type=None, out_dir=None)

    def test_save_freq_table_low_memory_node_name_only(self):
        with self.assertRaises(ValueError):
            save_freq_table(freq_table=pro_single_ft, low_memory=True, node_name='Inner1', table_type=None,
                            out_dir=None)

    def test_save_freq_table_low_memory_table_type_only(self):
        with self.assertRaises(ValueError):
            save_freq_table(freq_table=pro_single_ft, low_memory=True, node_name=None, table_type='single',
                            out_dir=None)

    def test_save_freq_table_low_memory_out_dir_only(self):
        with self.assertRaises(ValueError):
            save_freq_table(freq_table=pro_single_ft, low_memory=True, node_name=None, table_type=None,
                            out_dir=os.getcwd())

    def test_save_freq_table_low_memory_all_variables(self):
        fn = save_freq_table(freq_table=pro_single_ft, low_memory=True, node_name='Inner1', table_type='single',
                             out_dir=os.getcwd())
        self.assertEqual(fn, os.path.join(os.getcwd(), 'Inner1_single_freq_table.pkl'))
        os.remove(fn)

    def test_save_freq_table_low_memory_all_variables_file_exists(self):
        expected_fn = os.path.join(os.getcwd(), 'Inner1_single_freq_table.pkl')
        with open(expected_fn, 'a'):
            os.utime(expected_fn, None)
        fn = save_freq_table(freq_table=pro_single_ft, low_memory=True, node_name='Inner1', table_type='single',
                             out_dir=os.getcwd())
        self.assertEqual(fn, expected_fn)
        os.remove(expected_fn)


class TestTraceLoadFreqTable(TestCase):

    def test_load_freq_table_not_low_memory(self):
        ft = load_freq_table(freq_table=pro_single_ft, low_memory=False)
        self.assertIs(ft, pro_single_ft)

    def test_load_freq_table_not_low_memory_failure_no_freq_table(self):
        with self.assertRaises(ValueError):
            load_freq_table(freq_table=None, low_memory=False)

    def test_load_freq_table_not_low_memory_failure_file_name(self):
        with self.assertRaises(ValueError):
            load_freq_table(freq_table=os.path.join(os.getcwd(), 'Inner1_single_freq_table.pkl'), low_memory=False)

    def test_load_freq_table_low_memory(self):
        fn = save_freq_table(freq_table=pro_single_ft, low_memory=True, node_name='Inner1', table_type='single',
                             out_dir=os.getcwd())
        ft = load_freq_table(freq_table=fn, low_memory=True)
        self.assertEqual(pro_single_ft.get_depth(), ft.get_depth())
        self.assertFalse((pro_single_ft.get_count_matrix() - ft.get_count_matrix()).any())
        os.remove(fn)

    def test_load_freq_table_low_memory_failure_freq_table(self):
        with self.assertRaises(ValueError):
            load_freq_table(freq_table=pro_single_ft, low_memory=True)

    def test_load_freq_table_not_low_memory_failure_no_file_name(self):
        with self.assertRaises(ValueError):
            load_freq_table(freq_table=None, low_memory=False)


class TestTraceCheckNumpyArray(TestCase):

    def test_numpy_array_not_low_memory_no_other_variables(self):
        check, fn = check_numpy_array(low_memory=False, node_name=None, pos_type=None, score_type=None, metric=None,
                                      out_dir=None)
        self.assertFalse(check)
        self.assertIsNone(fn)

    def test_check_numpy_array_not_low_memory_node_name_only(self):
        check, fn = check_numpy_array(low_memory=False, node_name='Inner1', pos_type=None, score_type=None, metric=None,
                                      out_dir=None)
        self.assertFalse(check)
        self.assertIsNone(fn)

    def test_check_numpy_array_not_low_memory_pos_type_only(self):
        check, fn = check_numpy_array(low_memory=False, node_name=None, pos_type='single', score_type=None, metric=None,
                                      out_dir=None)
        self.assertFalse(check)
        self.assertIsNone(fn)

    def test_check_numpy_array_not_low_memory_score_type_only(self):
        check, fn = check_numpy_array(low_memory=False, node_name=None, pos_type=None, score_type='group', metric=None,
                                      out_dir=None)
        self.assertFalse(check)
        self.assertIsNone(fn)

    def test_check_numpy_array_not_low_memory_metric_only(self):
        check, fn = check_numpy_array(low_memory=False, node_name=None, pos_type=None, score_type=None,
                                      metric='identity', out_dir=None)
        self.assertFalse(check)
        self.assertIsNone(fn)

    def test_check_numpy_array_not_low_memory_out_dir_only(self):
        check, fn = check_numpy_array(low_memory=False, node_name=None, pos_type=None, score_type=None, metric=None,
                                      out_dir=os.getcwd())
        self.assertFalse(check)
        self.assertIsNone(fn)

    def test_check_numpy_array_not_low_memory_all_variables(self):
        check, fn = check_numpy_array(low_memory=False, node_name='Inner1', pos_type='single', score_type='group',
                                      metric='identity', out_dir=os.getcwd())
        self.assertFalse(check)
        self.assertIsNone(fn)

    def test_check_numpy_array_low_memory_no_other_variables(self):
        with self.assertRaises(ValueError):
            check_numpy_array(low_memory=True, node_name=None, pos_type=None, score_type=None, metric=None,
                              out_dir=None)

    def test_check_numpy_array_low_memory_node_name_only(self):
        with self.assertRaises(ValueError):
            check_numpy_array(low_memory=True, node_name='Inner1', pos_type=None, score_type=None, metric=None,
                              out_dir=None)

    def test_check_numpy_array_low_memory_pos_type_only(self):
        with self.assertRaises(ValueError):
            check_numpy_array(low_memory=True, node_name=None, pos_type='single', score_type=None, metric=None,
                              out_dir=None)

    def test_check_numpy_array_low_memory_score_type_only(self):
        with self.assertRaises(ValueError):
            check_numpy_array(low_memory=True, node_name=None, pos_type=None, score_type='group', metric=None,
                              out_dir=None)

    def test_check_numpy_array_low_memory_metric_only(self):
        with self.assertRaises(ValueError):
            check_numpy_array(low_memory=True, node_name=None, pos_type=None, score_type=None, metric='identity',
                              out_dir=None)

    def test_check_numpy_array_low_memory_out_dir_only(self):
        with self.assertRaises(ValueError):
            check_numpy_array(low_memory=True, node_name=None, pos_type=None, score_type=None, metric=None,
                              out_dir=os.getcwd())

    def test_check_numpy_array_low_memory_all_variables(self):
        check, fn = check_numpy_array(low_memory=True, node_name='Inner1', pos_type='single', score_type='group',
                                      metric='identity', out_dir=os.getcwd())
        self.assertFalse(check)
        self.assertEqual(fn, os.path.join(os.getcwd(), 'Inner1_single_group_identity_score.npz'))

    def test_check_numpy_array_low_memory_all_variables_file_exists(self):
        expected_fn = os.path.join(os.getcwd(), 'Inner1_single_group_identity_score.npz')
        with open(expected_fn, 'a'):
            os.utime(expected_fn, None)
        check, fn = check_numpy_array(low_memory=True, node_name='Inner1', pos_type='single', score_type='group',
                                      metric='identity', out_dir=os.getcwd())
        self.assertTrue(check)
        self.assertEqual(fn, expected_fn)
        os.remove(expected_fn)


class TestTraceSaveNumpyArray(TestCase):

    def test_save_numpy_array_not_low_memory_no_other_variables(self):
        input_mat = np.random.rand(6, 6)
        mat = save_numpy_array(mat=input_mat, low_memory=False, node_name=None, pos_type=None, score_type=None,
                               metric=None, out_dir=None)
        self.assertIs(mat, input_mat)

    def test_save_numpy_array_not_low_memory_node_name_only(self):
        input_mat = np.random.rand(6, 6)
        mat = save_numpy_array(mat=input_mat, low_memory=False, node_name='Inner1', pos_type=None, score_type=None,
                               metric=None, out_dir=None)
        self.assertIs(mat, input_mat)

    def test_save_numpy_array_not_low_memory_pos_type_only(self):
        input_mat = np.random.rand(6, 6)
        mat = save_numpy_array(mat=input_mat, low_memory=False, node_name=None, pos_type='single',
                               score_type=None, metric=None, out_dir=None)
        self.assertIs(mat, input_mat)

    def test_save_numpy_array_not_low_memory_score_type_only(self):
        input_mat = np.random.rand(6, 6)
        mat = save_numpy_array(mat=input_mat, low_memory=False, node_name=None, pos_type=None,
                               score_type='group', metric=None, out_dir=None)
        self.assertIs(mat, input_mat)

    def test_save_numpy_array_not_low_memory_metric_only(self):
        input_mat = np.random.rand(6, 6)
        mat = save_numpy_array(mat=input_mat, low_memory=False, node_name=None, pos_type=None,
                               score_type=None, metric='identity', out_dir=None)
        self.assertIs(mat, input_mat)

    def test_save_numpy_array_not_low_memory_out_dir_only(self):
        input_mat = np.random.rand(6, 6)
        mat = save_numpy_array(mat=input_mat, low_memory=False, node_name=None, pos_type=None,
                               score_type=None, metric=None, out_dir=os.getcwd())
        self.assertIs(mat, input_mat)

    def test_save_numpy_array_not_low_memory_all_variables(self):
        input_mat = np.random.rand(6, 6)
        mat = save_numpy_array(mat=input_mat, low_memory=False, node_name='Inner1', pos_type='single',
                               score_type='group', metric='identity', out_dir=os.getcwd())
        self.assertIs(mat, input_mat)

    def test_save_numpy_array_low_memory_no_other_variables(self):
        input_mat = np.random.rand(6, 6)
        with self.assertRaises(ValueError):
            save_numpy_array(mat=input_mat, low_memory=True, node_name=None, pos_type=None, score_type=None,
                             metric=None, out_dir=None)

    def test_save_numpy_array_low_memory_node_name_only(self):
        input_mat = np.random.rand(6, 6)
        with self.assertRaises(ValueError):
            save_numpy_array(mat=input_mat, low_memory=True, node_name='Inner1', pos_type=None, score_type=None,
                             metric=None, out_dir=None)

    def test_save_numpy_array_low_memory_pos_type_only(self):
        input_mat = np.random.rand(6, 6)
        with self.assertRaises(ValueError):
            save_numpy_array(mat=input_mat, low_memory=True, node_name=None, pos_type='single', score_type=None,
                             metric=None, out_dir=None)

    def test_save_numpy_array_low_memory_score_type_only(self):
        input_mat = np.random.rand(6, 6)
        with self.assertRaises(ValueError):
            save_numpy_array(mat=input_mat, low_memory=True, node_name=None, pos_type=None, score_type='group',
                             metric=None, out_dir=None)

    def test_save_numpy_array_low_memory_metric_only(self):
        input_mat = np.random.rand(6, 6)
        with self.assertRaises(ValueError):
            save_numpy_array(mat=input_mat, low_memory=True, node_name=None, pos_type=None, score_type=None,
                             metric='identity', out_dir=None)

    def test_save_numpy_array_low_memory_out_dir_only(self):
        input_mat = np.random.rand(6, 6)
        with self.assertRaises(ValueError):
            save_numpy_array(mat=input_mat, low_memory=True, node_name=None, pos_type=None, score_type=None,
                             metric='identity', out_dir=os.getcwd())

    def test_save_numpy_array_low_memory_all_variables(self):
        input_mat = np.random.rand(6, 6)
        fn = save_numpy_array(mat=input_mat, low_memory=True, node_name='Inner1', pos_type='single', score_type='group',
                              metric='identity', out_dir=os.getcwd())
        self.assertEqual(fn, os.path.join(os.getcwd(), 'Inner1_single_group_identity_score.npz'))
        os.remove(fn)

    def test_save_numpy_array_low_memory_all_variables_file_exists(self):
        input_mat = np.random.rand(6, 6)
        expected_fn = os.path.join(os.getcwd(), 'Inner1_single_group_identity_score.npz')
        with open(expected_fn, 'a'):
            os.utime(expected_fn, None)
        fn = save_numpy_array(mat=input_mat, low_memory=True, node_name='Inner1', pos_type='single',
                              score_type='group', metric='identity', out_dir=os.getcwd())
        self.assertEqual(fn, expected_fn)
        os.remove(expected_fn)


class TestTraceLoadNumpyArray(TestCase):

    def test_load_numpy_array_not_low_memory(self):
        input_arr = np.random.rand(6, 6)
        arr = load_numpy_array(mat=input_arr, low_memory=False)
        self.assertIs(arr, input_arr)

    def test_load_numpy_array_not_low_memory_failure_no_mat(self):
        with self.assertRaises(ValueError):
            load_numpy_array(mat=None, low_memory=False)

    def test_load_numpy_array_not_low_memory_failure_file_name(self):
        with self.assertRaises(ValueError):
            load_numpy_array(mat=os.path.join(os.getcwd(), 'Inner1_single_group_identity_score.npz'), low_memory=False)

    def test_load_numpy_array_low_memory(self):
        input_mat = np.random.rand(6, 6)
        fn = save_numpy_array(mat=input_mat, low_memory=True, node_name='Inner1', pos_type='single',
                              score_type='group', metric='identity', out_dir=os.getcwd())
        mat = load_numpy_array(mat=fn, low_memory=True)
        self.assertFalse((mat - input_mat).any())
        os.remove(fn)

    def test_load_numpy_array_low_memory_failure_mat(self):
        input_mat = np.random.rand(6, 6)
        with self.assertRaises(ValueError):
            load_numpy_array(mat=input_mat, low_memory=True)

    def test_load_numpy_array_not_low_memory_failure_no_file_name(self):
        with self.assertRaises(ValueError):
            load_numpy_array(mat=None, low_memory=False)


class TestTraceCharacterizationPool(TestCase):

    def test_characterization_pool_single_low_memory_single_process(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_pool(alpha_size=protein_alpha_size, alpha_mapping=protein_map, alpha_reverse=protein_rev,
                                   single_to_pair=None, alignment=aln, pos_size=1, components=components,
                                   sharable_dict=freq_tables, sharable_lock=tables_lock, unique_dir=u_dir,
                                   low_memory=True, write_out_sub_aln=False, write_out_freq_table=False, processes=1)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], str)
            loaded1 = load_freq_table(freq_tables[node_name]['freq_table'], True)
            expected1 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]))
            self.assertEqual(loaded1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded1.get_count_matrix() - expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], str)
            loaded2 = load_freq_table(freq_tables[node_name]['freq_table'], True)
            expected2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected2.characterize_alignment(num_aln=num_aln[inds, :])
            self.assertEqual(loaded2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded2.get_count_matrix() - expected2.get_count_matrix()).any())
        rmtree(u_dir)

    def test_characterization_pool_single_low_memory_multi_process(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_pool(alpha_size=protein_alpha_size, alpha_mapping=protein_map, alpha_reverse=protein_rev,
                                   single_to_pair=None, alignment=aln, pos_size=1, components=components,
                                   sharable_dict=freq_tables, sharable_lock=tables_lock, unique_dir=u_dir,
                                   low_memory=True, write_out_sub_aln=False, write_out_freq_table=False, processes=2)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], str)
            loaded1 = load_freq_table(freq_tables[node_name]['freq_table'], True)
            expected1 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]))
            self.assertEqual(loaded1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded1.get_count_matrix() - expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], str)
            loaded2 = load_freq_table(freq_tables[node_name]['freq_table'], True)
            expected2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected2.characterize_alignment(num_aln=num_aln[inds, :])
            self.assertEqual(loaded2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded2.get_count_matrix() - expected2.get_count_matrix()).any())
        rmtree(u_dir)

    def test_characterization_pool_single_not_low_memory_single_process(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=protein_alpha_size, alpha_mapping=protein_map,
                                   alpha_reverse=protein_rev, single_to_pair=None, alignment=aln, pos_size=1,
                                   components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                   unique_dir=None, low_memory=False, write_out_sub_aln=False,
                                   write_out_freq_table=False, processes=1)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], FrequencyTable)
            loaded1 = freq_tables[node_name]['freq_table']
            expected1 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]))
            self.assertEqual(loaded1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded1.get_count_matrix() - expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], FrequencyTable)
            loaded2 = freq_tables[node_name]['freq_table']
            expected2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected2.characterize_alignment(num_aln=num_aln[inds, :])
            self.assertEqual(loaded2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded2.get_count_matrix() - expected2.get_count_matrix()).any())

    def test_characterization_pool_single_not_low_memory_multi_process(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=protein_alpha_size, alpha_mapping=protein_map,
                                   alpha_reverse=protein_rev, single_to_pair=None, alignment=aln, pos_size=1,
                                   components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                   unique_dir=None, low_memory=False, write_out_sub_aln=False,
                                   write_out_freq_table=False, processes=2)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], FrequencyTable)
            loaded1 = freq_tables[node_name]['freq_table']
            expected1 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]))
            self.assertEqual(loaded1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded1.get_count_matrix() - expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], FrequencyTable)
            loaded2 = freq_tables[node_name]['freq_table']
            expected2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected2.characterize_alignment(num_aln=num_aln[inds, :])
            self.assertEqual(loaded2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded2.get_count_matrix() - expected2.get_count_matrix()).any())

    def test_characterization_pool_single_write_out(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=protein_alpha_size, alpha_mapping=protein_map,
                                   alpha_reverse=protein_rev, single_to_pair=None, alignment=aln, pos_size=1,
                                   components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                   unique_dir=u_dir, low_memory=False, write_out_sub_aln=True,
                                   write_out_freq_table=True, processes=1)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], FrequencyTable)
            loaded1 = freq_tables[node_name]['freq_table']
            expected1 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]))
            self.assertEqual(loaded1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded1.get_count_matrix() - expected1.get_count_matrix()).any())

            expected_path1a = os.path.join(u_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path1a))
            expected_path1b = os.path.join(u_dir, f'{node_name}_single_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path1b))
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], FrequencyTable)
            loaded2 = freq_tables[node_name]['freq_table']
            expected2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected2.characterize_alignment(num_aln=num_aln[inds, :])
            self.assertEqual(loaded2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded2.get_count_matrix() - expected2.get_count_matrix()).any())
            expected_path2a = os.path.join(u_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path2a))
            expected_path2b = os.path.join(u_dir, f'{node_name}_single_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path2b))
        rmtree(u_dir)

    def test_characterization_pool_pair_low_memory_single_process(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=u_dir, low_memory=True,
                                   write_out_sub_aln=False, write_out_freq_table=False, processes=1)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], str)
            loaded1 = load_freq_table(freq_tables[node_name]['freq_table'], True)
            expected1 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]), single_to_pair=pro_single_to_pair)
            self.assertEqual(loaded1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded1.get_count_matrix() - expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], str)
            loaded2 = load_freq_table(freq_tables[node_name]['freq_table'], True)
            expected2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected2.characterize_alignment(num_aln=num_aln[inds, :], single_to_pair=pro_single_to_pair)
            self.assertEqual(loaded2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded2.get_count_matrix() - expected2.get_count_matrix()).any())
        rmtree(u_dir)

    def test_characterization_pool_pair_low_memory_multi_process(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=u_dir, low_memory=True,
                                   write_out_sub_aln=False, write_out_freq_table=False, processes=2)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], str)
            loaded1 = load_freq_table(freq_tables[node_name]['freq_table'], True)
            expected1 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]), single_to_pair=pro_single_to_pair)
            self.assertEqual(loaded1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded1.get_count_matrix() - expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], str)
            loaded2 = load_freq_table(freq_tables[node_name]['freq_table'], True)
            expected2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected2.characterize_alignment(num_aln=num_aln[inds, :], single_to_pair=pro_single_to_pair)
            self.assertEqual(loaded2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded2.get_count_matrix() - expected2.get_count_matrix()).any())
        rmtree(u_dir)

    def test_characterization_pool_pair_not_low_memory_single_process(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=None, low_memory=False,
                                   write_out_sub_aln=False, write_out_freq_table=False, processes=1)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], FrequencyTable)
            loaded1 = freq_tables[node_name]['freq_table']
            expected1 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]), single_to_pair=pro_single_to_pair)
            self.assertEqual(loaded1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded1.get_count_matrix() - expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], FrequencyTable)
            loaded2 = freq_tables[node_name]['freq_table']
            expected2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected2.characterize_alignment(num_aln=num_aln[inds, :], single_to_pair=pro_single_to_pair)
            self.assertEqual(loaded2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded2.get_count_matrix() - expected2.get_count_matrix()).any())

    def test_characterization_pool_pair_not_low_memory_multi_process(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=None, low_memory=False,
                                   write_out_sub_aln=False, write_out_freq_table=False, processes=2)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], FrequencyTable)
            loaded1 = freq_tables[node_name]['freq_table']
            expected1 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]), single_to_pair=pro_single_to_pair)
            self.assertEqual(loaded1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded1.get_count_matrix() - expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], FrequencyTable)
            loaded2 = freq_tables[node_name]['freq_table']
            expected2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected2.characterize_alignment(num_aln=num_aln[inds, :], single_to_pair=pro_single_to_pair)
            self.assertEqual(loaded2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded2.get_count_matrix() - expected2.get_count_matrix()).any())

    def test_characterization_pool_pair_write_out(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], FrequencyTable)
            loaded1 = freq_tables[node_name]['freq_table']
            expected1 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]), single_to_pair=pro_single_to_pair)
            self.assertEqual(loaded1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded1.get_count_matrix() - expected1.get_count_matrix()).any())
            expected_path1a = os.path.join(u_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path1a))
            expected_path1b = os.path.join(u_dir, f'{node_name}_pair_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path1b))
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['freq_table'], FrequencyTable)
            loaded2 = freq_tables[node_name]['freq_table']
            expected2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected2.characterize_alignment(num_aln=num_aln[inds, :], single_to_pair=pro_single_to_pair)
            self.assertEqual(loaded2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded2.get_count_matrix() - expected2.get_count_matrix()).any())
            expected_path2a = os.path.join(u_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path2a))
            expected_path2b = os.path.join(u_dir, f'{node_name}_pair_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path2b))
        rmtree(u_dir)

    def test_characterization_pool_failure_no_node_name(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(KeyError):
            characterization(node_name=None, node_type='component')
        rmtree(u_dir)

    def test_characterization_pool_failure_missing_node_name(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(KeyError):
            characterization(node_name='seq4', node_type='component')
        rmtree(u_dir)

    def test_characterization_pool_failure_no_node_type(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(ValueError):
            characterization(node_name='seq1', node_type=None)
        rmtree(u_dir)

    def test_characterization_pool_failure_unexpected_node_type(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(ValueError):
            characterization(node_name='seq1', node_type='root')
        rmtree(u_dir)

    # Missing (i.e. None) alphabet size, mapping, reverse mapping, and single to pair mapping will all succeed because
    # the characterize_positions and characterize_positions2 methods in SeqAlignment which are called by this method
    # will generate those if missing from the provided alignment.

    def test_characterization_pool_failure_no_alignment(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=None,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(AttributeError):
            characterization(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_pool_failure_no_pos_size(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        with self.assertRaises(ValueError):
            init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                       alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                       pos_size=None, components=components, sharable_dict=freq_tables,
                                       sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                       write_out_sub_aln=True, write_out_freq_table=True, processes=1)
            characterization(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_pool_failure_bad_pos_size(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        with self.assertRaises(ValueError):
            init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                       alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                       pos_size=3, components=components, sharable_dict=freq_tables,
                                       sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                       write_out_sub_aln=True, write_out_freq_table=True, processes=1)
            characterization(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_pool_failure_no_components(self):
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()

        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=None, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(TypeError):
            characterization(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_pool_failure_no_shareable_dict(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=components, sharable_dict=None,
                                   sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(TypeError):
            characterization(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_pool_failure_no_shareable_lock(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=None, unique_dir=u_dir, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(AttributeError):
            characterization(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_pool_failure_bad_low_memory(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=None, low_memory='low',
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(ValueError):
            characterization(node_name='seq1', node_type='component')

    def test_characterization_pool_failure_low_memory_no_unique_dir(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=None, low_memory=True,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(ValueError):
            characterization(node_name='seq1', node_type='component')

    def test_characterization_pool_failure_no_processes(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=None, low_memory=False,
                                   write_out_sub_aln=False, write_out_freq_table=False, processes=None)
        with self.assertRaises(TypeError):
            characterization(node_name='seq1', node_type='component')

    def test_characterization_pool_failure_write_out_no_u_dir(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=None, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(TypeError):
            characterization(node_name='seq1', node_type='component')


class TestTraceCharacterizationMMPool(TestCase):

    def test_characterization_mm_pool_single_low_memory(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(larger_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse=pro_pair_rev, match_mismatch_table=protein_mm_table,
                                      alignment=aln, position_size=1, components=components, sharable_dict=freq_tables,
                                      sharable_lock=tables_lock, unique_dir=u_dir, low_memory=True,
                                      write_out_sub_aln=False, write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization_mm(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('match' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['match'], str)
            loaded1a = load_freq_table(freq_tables[node_name]['match'], True)
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['mismatch'], str)
            loaded1b = load_freq_table(freq_tables[node_name]['mismatch'], True)
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization_mm(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('match' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['match'], str)
            loaded2a = load_freq_table(freq_tables[node_name]['match'], True)
            self.assertEqual(loaded2a.get_depth(), expected_single_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_single_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['mismatch'], str)
            loaded2b = load_freq_table(freq_tables[node_name]['mismatch'], True)
            self.assertEqual(loaded2b.get_depth(), expected_single_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_single_tables[node_name]['mismatch'].get_count_matrix()).any())
        rmtree(u_dir)

    def test_characterization_mm_pool_single_not_low_memory(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_mm_pool(larger_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse=pro_pair_rev, match_mismatch_table=protein_mm_table,
                                      alignment=aln, position_size=1, components=components, sharable_dict=freq_tables,
                                      sharable_lock=tables_lock, unique_dir=None, low_memory=False,
                                      write_out_sub_aln=False, write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization_mm(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('match' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['match'], FrequencyTable)
            loaded1a = freq_tables[node_name]['match']
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['mismatch'], FrequencyTable)
            loaded1b = freq_tables[node_name]['mismatch']
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization_mm(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('match' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['match'], FrequencyTable)
            loaded2a = freq_tables[node_name]['match']
            self.assertEqual(loaded2a.get_depth(), expected_single_tables[node_name]['match'].get_depth())
            for pos in range(6):
                self.assertFalse((loaded2a.get_count_matrix() -
                                  expected_single_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['mismatch'], FrequencyTable)
            loaded2b = freq_tables[node_name]['mismatch']
            self.assertEqual(loaded2b.get_depth(), expected_single_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_single_tables[node_name]['mismatch'].get_count_matrix()).any())

    def test_characterization_mm_pool_single_write_out(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(larger_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse=pro_pair_rev, match_mismatch_table=protein_mm_table,
                                      alignment=aln, position_size=1, components=components, sharable_dict=freq_tables,
                                      sharable_lock=tables_lock, unique_dir=u_dir, low_memory=True,
                                      write_out_sub_aln=True, write_out_freq_table=True)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization_mm(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('match' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['match'], str)
            loaded1a = load_freq_table(freq_tables[node_name]['match'], True)
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['mismatch'], str)
            loaded1b = load_freq_table(freq_tables[node_name]['mismatch'], True)
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
            expected_path1a = os.path.join(u_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path1a))
            expected_path1b = os.path.join(u_dir, f'{node_name}_single_match_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path1b))
            expected_path1c = os.path.join(u_dir, f'{node_name}_single_mismatch_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path1c))
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization_mm(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('match' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['match'], str)
            loaded2a = load_freq_table(freq_tables[node_name]['match'], True)
            self.assertEqual(loaded2a.get_depth(), expected_single_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_single_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['mismatch'], str)
            loaded2b = load_freq_table(freq_tables[node_name]['mismatch'], True)
            self.assertEqual(loaded2b.get_depth(), expected_single_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_single_tables[node_name]['mismatch'].get_count_matrix()).any())
            expected_path2a = os.path.join(u_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path2a))
            expected_path2b = os.path.join(u_dir, f'{node_name}_single_match_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path2b))
            expected_path2c = os.path.join(u_dir, f'{node_name}_single_mismatch_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path2c))
        rmtree(u_dir)

    def test_characterization_mm_pool_pair_low_memory(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                      alignment=aln, position_size=2, components=components, sharable_dict=freq_tables,
                                      sharable_lock=tables_lock, unique_dir=u_dir, low_memory=True,
                                      write_out_sub_aln=False, write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization_mm(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('match' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['match'], str)
            loaded1a = load_freq_table(freq_tables[node_name]['match'], True)
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['mismatch'], str)
            loaded1b = load_freq_table(freq_tables[node_name]['mismatch'], True)
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization_mm(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('match' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['match'], str)
            loaded2a = load_freq_table(freq_tables[node_name]['match'], True)
            self.assertEqual(loaded2a.get_depth(), expected_pair_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_pair_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['mismatch'], str)
            loaded2b = load_freq_table(freq_tables[node_name]['mismatch'], True)
            self.assertEqual(loaded2b.get_depth(), expected_pair_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_pair_tables[node_name]['mismatch'].get_count_matrix()).any())
        rmtree(u_dir)

    def test_characterization_mm_pool_pair_not_low_memory(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                      alignment=aln, position_size=2, components=components, sharable_dict=freq_tables,
                                      sharable_lock=tables_lock, unique_dir=None, low_memory=False,
                                      write_out_sub_aln=False, write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization_mm(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('match' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['match'], FrequencyTable)
            loaded1a = freq_tables[node_name]['match']
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['mismatch'], FrequencyTable)
            loaded1b = freq_tables[node_name]['mismatch']
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization_mm(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('match' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['match'], FrequencyTable)
            loaded2a = freq_tables[node_name]['match']
            self.assertEqual(loaded2a.get_depth(), expected_pair_tables[node_name]['match'].get_depth())
            for pos in range(6):
                self.assertFalse((loaded2a.get_count_matrix() -
                                  expected_pair_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['mismatch'], FrequencyTable)
            loaded2b = freq_tables[node_name]['mismatch']
            self.assertEqual(loaded2b.get_depth(), expected_pair_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_pair_tables[node_name]['mismatch'].get_count_matrix()).any())

    def test_characterization_mm_pool_pair_write_out(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                      alignment=aln, position_size=2, components=components, sharable_dict=freq_tables,
                                      sharable_lock=tables_lock, unique_dir=u_dir, low_memory=True,
                                      write_out_sub_aln=True, write_out_freq_table=True)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization_mm(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('match' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['match'], str)
            loaded1a = load_freq_table(freq_tables[node_name]['match'], True)
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['mismatch'], str)
            loaded1b = load_freq_table(freq_tables[node_name]['mismatch'], True)
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
            expected_path1a = os.path.join(u_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path1a))
            expected_path1b = os.path.join(u_dir, f'{node_name}_pair_match_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path1b))
            expected_path1c = os.path.join(u_dir, f'{node_name}_pair_mismatch_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path1c))
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization_mm(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('match' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['match'], str)
            loaded2a = load_freq_table(freq_tables[node_name]['match'], True)
            self.assertEqual(loaded2a.get_depth(), expected_pair_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_pair_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in freq_tables[node_name])
            self.assertIsInstance(freq_tables[node_name]['mismatch'], str)
            loaded2b = load_freq_table(freq_tables[node_name]['mismatch'], True)
            self.assertEqual(loaded2b.get_depth(), expected_pair_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_pair_tables[node_name]['mismatch'].get_count_matrix()).any())
            expected_path2a = os.path.join(u_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path2a))
            expected_path2b = os.path.join(u_dir, f'{node_name}_pair_match_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path2b))
            expected_path2c = os.path.join(u_dir, f'{node_name}_pair_mismatch_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path2c))
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_node_name(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                      alignment=aln, position_size=2, components=components, sharable_dict=freq_tables,
                                      sharable_lock=tables_lock, unique_dir=u_dir, low_memory=True,
                                      write_out_sub_aln=True, write_out_freq_table=True)
        with self.assertRaises(ValueError):
            characterization_mm(node_name=None, node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_missing_node_name(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                      alignment=aln, position_size=2, components=components, sharable_dict=freq_tables,
                                      sharable_lock=tables_lock, unique_dir=u_dir, low_memory=True,
                                      write_out_sub_aln=True, write_out_freq_table=True)
        with self.assertRaises(KeyError):
            characterization_mm(node_name='Seq4', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_node_type(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                      alignment=aln, position_size=2, components=components, sharable_dict=freq_tables,
                                      sharable_lock=tables_lock, unique_dir=u_dir, low_memory=True,
                                      write_out_sub_aln=True, write_out_freq_table=True)
        with self.assertRaises(ValueError):
            characterization_mm(node_name='seq1', node_type=None)
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_unexpected_node_type(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                      alignment=aln, position_size=2, components=components, sharable_dict=freq_tables,
                                      sharable_lock=tables_lock, unique_dir=u_dir, low_memory=True,
                                      write_out_sub_aln=True, write_out_freq_table=True)
        with self.assertRaises(ValueError):
            characterization_mm(node_name='seq1', node_type='intermediate')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_larger_size(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(larger_size=None, larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                      match_mismatch_table=protein_mm_table_large, alignment=aln, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(TypeError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_larger_mapping(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=None, larger_reverse=pro_quad_rev,
                                      match_mismatch_table=protein_mm_table_large, alignment=aln, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(AttributeError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_larger_reverse(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map, larger_reverse=None,
                                      match_mismatch_table=protein_mm_table_large, alignment=aln, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(TypeError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_match_mismatch_table(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        freq_tables['seq2'] = {'match': FrequencyTable(alphabet_size=pro_quad_alpha_size, mapping=pro_quad_map,
                                                       reverse_mapping=pro_quad_rev, seq_len=6, pos_size=2),
                               'mismatch': FrequencyTable(alphabet_size=pro_quad_alpha_size, mapping=pro_quad_map,
                                                          reverse_mapping=pro_quad_rev, seq_len=6, pos_size=2)}
        freq_tables['seq3'] = {'match': FrequencyTable(alphabet_size=pro_quad_alpha_size, mapping=pro_quad_map,
                                                       reverse_mapping=pro_quad_rev, seq_len=6, pos_size=2),
                               'mismatch': FrequencyTable(alphabet_size=pro_quad_alpha_size, mapping=pro_quad_map,
                                                          reverse_mapping=pro_quad_rev, seq_len=6, pos_size=2)}
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, match_mismatch_table=None, alignment=aln,
                                      position_size=2, components=components, sharable_dict=freq_tables,
                                      sharable_lock=tables_lock, unique_dir=u_dir, low_memory=True,
                                      write_out_sub_aln=True, write_out_freq_table=True)
        with self.assertRaises(AttributeError):
            characterization_mm(node_name='Inner2', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_alignment(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                      alignment=None, position_size=2, components=components, sharable_dict=freq_tables,
                                      sharable_lock=tables_lock, unique_dir=u_dir, low_memory=True,
                                      write_out_sub_aln=True, write_out_freq_table=True)
        with self.assertRaises(AttributeError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_pos_size(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        with self.assertRaises(ValueError):
            init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                          larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                          alignment=aln, position_size=None, components=components,
                                          sharable_dict=freq_tables, sharable_lock=tables_lock, unique_dir=u_dir,
                                          low_memory=True, write_out_sub_aln=True, write_out_freq_table=True)
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_bad_pos_size(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        with self.assertRaises(ValueError):
            init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                          larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                          alignment=aln, position_size=3, components=components,
                                          sharable_dict=freq_tables, sharable_lock=tables_lock, unique_dir=u_dir,
                                          low_memory=True, write_out_sub_aln=True, write_out_freq_table=True)
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_components(self):
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                      alignment=aln, position_size=2, components=None,
                                      sharable_dict=freq_tables, sharable_lock=tables_lock, unique_dir=u_dir,
                                      low_memory=True, write_out_sub_aln=True, write_out_freq_table=True)
        with self.assertRaises(TypeError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_shareable_dict(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        tables_lock = Lock()
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                      alignment=aln, position_size=2, components=components,
                                      sharable_dict=None, sharable_lock=tables_lock, unique_dir=u_dir,
                                      low_memory=True, write_out_sub_aln=True, write_out_freq_table=True)
        with self.assertRaises(TypeError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_shareable_lock(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                      alignment=aln, position_size=2, components=components,
                                      sharable_dict=freq_tables, sharable_lock=None, unique_dir=u_dir,
                                      low_memory=True, write_out_sub_aln=True, write_out_freq_table=True)
        with self.assertRaises(AttributeError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_bad_low_memory(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                      alignment=aln, position_size=2, components=components,
                                      sharable_dict=freq_tables, sharable_lock=tables_lock, unique_dir=None,
                                      low_memory='low', write_out_sub_aln=True, write_out_freq_table=True)
        with self.assertRaises(ValueError):
            characterization_mm(node_name='seq1', node_type='component')

    def test_characterization_mm_pool_failure_low_memory_no_unique_dir(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                      alignment=aln, position_size=2, components=components,
                                      sharable_dict=freq_tables, sharable_lock=tables_lock, unique_dir=None,
                                      low_memory=True, write_out_sub_aln=True, write_out_freq_table=True)
        with self.assertRaises(ValueError):
            characterization_mm(node_name='seq1', node_type='component')

    def test_characterization_mm_pool_failure_write_out_no_u_dir(self):
        components = {'Inner1': rank_dict[1][1], 'Inner2': rank_dict[2][1],
                      'seq1': rank_dict[3][3], 'seq2': rank_dict[3][2], 'seq3': rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_mm_pool(larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, match_mismatch_table=protein_mm_table_large,
                                      alignment=aln, position_size=2, components=components,
                                      sharable_dict=freq_tables, sharable_lock=tables_lock, unique_dir=None,
                                      low_memory=False, write_out_sub_aln=True, write_out_freq_table=True)
        with self.assertRaises(TypeError):
            characterization_mm(node_name='seq1', node_type='component')


class TestTraceTraceGroups(TestCase):

    def setUp(self):
        self.single_u_dict = {'Inner1': {'freq_table': pro_single_ft},
                              'Inner2': {'freq_table': pro_single_ft_i2},
                              'seq1': {'freq_table': pro_single_ft_s1},
                              'seq2': {'freq_table': pro_single_ft_s2},
                              'seq3': {'freq_table': pro_single_ft_s3}}
        self.pair_u_dict = {'Inner1': {'freq_table': pro_pair_ft},
                            'Inner2': {'freq_table': pro_pair_ft_i2},
                            'seq1': {'freq_table': pro_pair_ft_s1},
                            'seq2': {'freq_table': pro_pair_ft_s2},
                            'seq3': {'freq_table': pro_pair_ft_s3}}
        single_mm_decoy = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
        single_mm_decoy.set_depth(1)
        self.single_mm_dict = {'Inner1': {'match': deepcopy(single_mm_decoy),
                                          'mismatch': deepcopy(single_mm_decoy)},
                               'Inner2': {'match': deepcopy(single_mm_decoy),
                                          'mismatch': deepcopy(single_mm_decoy)},
                               'seq1': {'match': deepcopy(single_mm_decoy),
                                        'mismatch': deepcopy(single_mm_decoy)},
                               'seq2': {'match': deepcopy(single_mm_decoy),
                                        'mismatch': deepcopy(single_mm_decoy)},
                               'seq3': {'match': deepcopy(single_mm_decoy),
                                        'mismatch': deepcopy(single_mm_decoy)}}
        for pos in range(6):
            for s1 in range(3):
                for s2 in range(s1 + 1, 3):
                    stat, curr_char = protein_mm_table.get_status_and_character(pos=pos, seq_ind1=s1, seq_ind2=s2)
                    self.single_mm_dict['Inner1'][stat]._increment_count(pos=pos, char=curr_char)
                    if s1 >= 1 and s2 >= 2:
                        self.single_mm_dict['Inner2'][stat]._increment_count(pos=pos, char=curr_char)
        for node in self.single_mm_dict:
            for status in self.single_mm_dict[node]:
                if node == 'Inner1':
                    self.single_mm_dict[node][status].set_depth(3)
                self.single_mm_dict[node][status].finalize_table()
        pair_mm_decoy = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
        pair_mm_decoy.set_depth(1)
        self.pair_mm_dict = {'Inner1': {'match': deepcopy(pair_mm_decoy),
                                        'mismatch': deepcopy(pair_mm_decoy)},
                             'Inner2': {'match': deepcopy(pair_mm_decoy),
                                        'mismatch': deepcopy(pair_mm_decoy)},
                             'seq1': {'match': deepcopy(pair_mm_decoy),
                                      'mismatch': deepcopy(pair_mm_decoy)},
                             'seq2': {'match': deepcopy(pair_mm_decoy),
                                      'mismatch': deepcopy(pair_mm_decoy)},
                             'seq3': {'match': deepcopy(pair_mm_decoy),
                                      'mismatch': deepcopy(pair_mm_decoy)}}
        for p1 in range(6):
            for p2 in range(p1, 6):
                for s1 in range(3):
                    for s2 in range(s1 + 1, 3):
                        stat, curr_char = protein_mm_table_large.get_status_and_character(pos=(p1, p2), seq_ind1=s1,
                                                                                          seq_ind2=s2)
                        self.pair_mm_dict['Inner1'][stat]._increment_count(pos=(p1, p2), char=curr_char)
                        if s1 >= 1 and s2 >= 2:
                            self.pair_mm_dict['Inner2'][stat]._increment_count(pos=(p1, p2), char=curr_char)
        for node in self.pair_mm_dict:
            for status in self.pair_mm_dict[node]:
                if node == 'Inner1':
                    self.pair_mm_dict[node][status].set_depth(3)
                self.pair_mm_dict[node][status].finalize_table()

    def evaluate_trace_groups_standard(self, scorer, u_dict, mem, u_dir, mm=False):
        init_trace_groups(scorer=scorer, match_mismatch=mm, u_dict=u_dict,
                          low_memory=(True if mem == 'low' else False), unique_dir=u_dir)
        for node in u_dict:
            curr_node_name, curr_scores = trace_groups(node_name=node)
            self.assertEqual(curr_node_name, node)
            if mem == 'low':
                self.assertIsInstance(curr_scores, str)
                self.assertTrue(os.path.isfile(curr_scores))
                curr_scores = load_numpy_array(curr_scores, True)
            else:
                self.assertIsInstance(curr_scores, np.ndarray)
            if mm:
                curr_tables = {'match': load_freq_table(u_dict[node]['match'], mem == 'low'),
                               'mismatch': load_freq_table(u_dict[node]['mismatch'], mem == 'low')}
                self.assertFalse((curr_scores - scorer.score_group(curr_tables)).any())
            else:
                self.assertFalse((curr_scores - scorer.score_group(load_freq_table(u_dict[node]['freq_table'],
                                                                                   mem == 'low'))).any())

    def test_trace_groups_identity_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.single_u_dict, mem='high', u_dir=None)

    def test_trace_groups_identity_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_u_dict, mem='high', u_dir=None)

    def test_trace_groups_plain_entropy_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='plain_entropy')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.single_u_dict, mem='high', u_dir=None)

    def test_trace_groups_plain_entropy_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_u_dict, mem='high', u_dir=None)

    def test_trace_groups_mi_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='mutual_information')
        init_trace_groups(scorer=scorer, match_mismatch=False, u_dict=self.single_u_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_u_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_mi_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='mutual_information')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_u_dict, mem='high', u_dir=None)

    def test_trace_groups_nmi_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='normalized_mutual_information')
        init_trace_groups(scorer=scorer, match_mismatch=False, u_dict=self.single_u_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_u_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_nmi_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='normalized_mutual_information')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_u_dict, mem='high', u_dir=None)

    def test_trace_groups_mip_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='average_product_corrected_mutual_information')
        init_trace_groups(scorer=scorer, match_mismatch=False, u_dict=self.single_u_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_u_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_mip_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='average_product_corrected_mutual_information')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_u_dict, mem='high', u_dir=None)

    def test_trace_groups_fmip_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2,
                                  metric='filtered_average_product_corrected_mutual_information')
        init_trace_groups(scorer=scorer, match_mismatch=False, u_dict=self.single_u_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_u_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_fmip_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2,
                                  metric='filtered_average_product_corrected_mutual_information')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_u_dict, mem='high', u_dir=None)

    def test_trace_groups_match_count_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_count')
        init_trace_groups(scorer=scorer, match_mismatch=True, u_dict=self.single_mm_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_mm_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_match_count_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_count')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_mm_dict, mem='high', u_dir=None, mm=True)

    def test_trace_groups_mismatch_count_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_count')
        init_trace_groups(scorer=scorer, match_mismatch=True, u_dict=self.single_mm_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_mm_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_mismatch_count_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_count')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_mm_dict, mem='high', u_dir=None, mm=True)

    def test_trace_groups_mm_count_ratio_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_ratio')
        init_trace_groups(scorer=scorer, match_mismatch=True, u_dict=self.single_mm_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_mm_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_mm_count_ratio_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_ratio')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_mm_dict, mem='high', u_dir=None, mm=True)

    def test_trace_groups_mm_count_angle_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_angle')
        init_trace_groups(scorer=scorer, match_mismatch=True, u_dict=self.single_mm_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_mm_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_mm_count_angle_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_angle')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_mm_dict, mem='high', u_dir=None, mm=True)

    def test_trace_groups_match_entropy_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_entropy')
        init_trace_groups(scorer=scorer, match_mismatch=True, u_dict=self.single_mm_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_mm_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_match_entropy_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_entropy')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_mm_dict, mem='high', u_dir=None, mm=True)

    def test_trace_groups_mismatch_entropy_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_entropy')
        init_trace_groups(scorer=scorer, match_mismatch=True, u_dict=self.single_mm_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_mm_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_mismatch_entropy_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_entropy')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_mm_dict, mem='high', u_dir=None, mm=True)

    def test_trace_groups_mm_entropy_ratio_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_ratio')
        init_trace_groups(scorer=scorer, match_mismatch=True, u_dict=self.single_mm_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_mm_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_mm_entropy_ratio_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_ratio')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_mm_dict, mem='high', u_dir=None, mm=True)

    def test_trace_groups_mm_entropy_angle_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_angle')
        init_trace_groups(scorer=scorer, match_mismatch=True, u_dict=self.single_mm_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_mm_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_mm_entropy_angle_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_angle')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_mm_dict, mem='high', u_dir=None, mm=True)

    def test_trace_groups_match_diversity_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity')
        init_trace_groups(scorer=scorer, match_mismatch=True, u_dict=self.single_mm_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_mm_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_match_diversity_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_mm_dict, mem='high', u_dir=None, mm=True)

    def test_trace_groups_mismatch_diversity_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_diversity')
        init_trace_groups(scorer=scorer, match_mismatch=True, u_dict=self.single_mm_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_mm_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_mismatch_diversity_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_diversity')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_mm_dict, mem='high', u_dir=None, mm=True)

    def test_trace_groups_mm_diversity_ratio_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_ratio')
        init_trace_groups(scorer=scorer, match_mismatch=True, u_dict=self.single_mm_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_mm_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_mm_diversity_ratio_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_ratio')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_mm_dict, mem='high', u_dir=None, mm=True)

    def test_trace_groups_mm_diversity_angle_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_angle')
        init_trace_groups(scorer=scorer, match_mismatch=True, u_dict=self.single_mm_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_mm_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_mm_diversity_angle_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_angle')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_mm_dict, mem='high', u_dir=None, mm=True)

    def test_trace_groups_match_diversity_mismatch_entropy_ratio_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_ratio')
        init_trace_groups(scorer=scorer, match_mismatch=True, u_dict=self.single_mm_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_mm_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_match_diversity_mismatch_entropy_ratio_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_ratio')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_mm_dict, mem='high', u_dir=None, mm=True)

    def test_trace_groups_match_diversity_mismatch_entropy_angle_single(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_angle')
        init_trace_groups(scorer=scorer, match_mismatch=True, u_dict=self.single_mm_dict, low_memory=False,
                          unique_dir=None)
        for node in self.single_mm_dict:
            with self.assertRaises(ValueError):
                trace_groups(node)

    def test_trace_groups_match_diversity_mismatch_entropy_angle_pair(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_angle')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=self.pair_mm_dict, mem='high', u_dir=None, mm=True)

    def test_trace_groups_int_single_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(test_dir)
        new_u_dict = {}
        for node in self.single_u_dict:
            new_u_dict[node] = {'freq_table': save_freq_table(self.single_u_dict[node]['freq_table'], True, node,
                                                              'single', test_dir)}
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=new_u_dict, mem='low', u_dir=test_dir)
        rmtree(test_dir)

    def test_trace_groups_int_pair_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(test_dir)
        new_u_dict = {}
        for node in self.pair_u_dict:
            new_u_dict[node] = {'freq_table': save_freq_table(self.pair_u_dict[node]['freq_table'], True, node,
                                                              'pair', test_dir)}
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=new_u_dict, mem='low', u_dir=test_dir)
        rmtree(test_dir)

    def test_trace_groups_real_single_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(test_dir)
        new_u_dict = {}
        for node in self.single_u_dict:
            new_u_dict[node] = {'freq_table': save_freq_table(self.single_u_dict[node]['freq_table'], True, node,
                                                              'single', test_dir)}
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='plain_entropy')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=new_u_dict, mem='low', u_dir=test_dir)
        rmtree(test_dir)

    def test_trace_groups_real_pair_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(test_dir)
        new_u_dict = {}
        for node in self.pair_u_dict:
            new_u_dict[node] = {'freq_table': save_freq_table(self.pair_u_dict[node]['freq_table'], True, node,
                                                              'pair', test_dir)}
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=new_u_dict, mem='low', u_dir=test_dir)
        rmtree(test_dir)

    def test_trace_groups_match_mismatch_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(test_dir)
        new_u_dict = {}
        for node in self.pair_mm_dict:
            new_u_dict[node] = {'match': save_freq_table(self.pair_mm_dict[node]['match'], True, node, 'pair_match',
                                                         test_dir),
                                'mismatch': save_freq_table(self.pair_mm_dict[node]['mismatch'], True, node,
                                                            'pair_mismatch', test_dir)}
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_count')
        self.evaluate_trace_groups_standard(scorer=scorer, u_dict=new_u_dict, mem='low', u_dir=test_dir, mm=True)
        rmtree(test_dir)

    def test_trace_groups_failure_no_node_name(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        init_trace_groups(scorer=scorer, match_mismatch=False, u_dict=self.single_u_dict, low_memory=False,
                          unique_dir=None)
        with self.assertRaises(KeyError):
            trace_groups(None)

    def test_trace_groups_failure_no_scorer(self):
        init_trace_groups(scorer=None, match_mismatch=False, u_dict=self.single_u_dict, low_memory=False,
                          unique_dir=None)
        with self.assertRaises(AttributeError):
            trace_groups('Inner1')

    def test_trace_groups_failure_bad_match_mismatch(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        init_trace_groups(scorer=scorer, match_mismatch='no', u_dict=self.single_u_dict, low_memory=False,
                          unique_dir=None)
        with self.assertRaises(KeyError):
            trace_groups('Inner1')

    def test_trace_groups_failure_no_u_dict(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        init_trace_groups(scorer=scorer, match_mismatch=False, u_dict=None, low_memory=False, unique_dir=None)
        with self.assertRaises(TypeError):
            trace_groups('Inner1')

    def test_trace_groups_failure_bad_low_mem(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        init_trace_groups(scorer=scorer, match_mismatch=False, u_dict=self.single_u_dict, low_memory='High',
                          unique_dir=None)
        with self.assertRaises(ValueError):
            trace_groups('Inner1')

    def test_trace_groups_failure_low_mem_no_u_dir(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        init_trace_groups(scorer=scorer, match_mismatch=False, u_dict=self.single_u_dict, low_memory=True,
                          unique_dir=None)
        with self.assertRaises(ValueError):
            trace_groups('Inner1')


class TestTraceTraceRanks(TestCase):

    def evaluate_trace_ranks(self, scorer, u_dict, mode, mem=None):
        for r in [3, 2, 1]:
            cumulative_ranks = np.zeros(scorer.dimensions)
            for g in range(1, r + 1):
                if mem == 'low':
                    cumulative_ranks += load_numpy_array(u_dict[rank_dict[r][g]['node'].name]['group_scores'], True)
                else:
                    cumulative_ranks += u_dict[rank_dict[r][g]['node'].name]['group_scores']
            if mode == 'int':
                expected_ranks = cumulative_ranks > 0
            elif mode == 'real':
                expected_ranks = cumulative_ranks / float(r)
            else:
                raise ValueError('Bad mode in trace_ranks evaluation!')
            if scorer.position_size == 2:
                expected_ranks = np.triu(expected_ranks, k=1)
            curr_rank, curr_ranks = trace_ranks(r)
            self.assertEqual(curr_rank, r)
            if mem == 'low':
                self.assertIsInstance(curr_ranks, str)
                self.assertTrue(os.path.isfile(curr_ranks))
                curr_ranks = load_numpy_array(curr_ranks, True)
            else:
                self.assertIsInstance(curr_ranks, np.ndarray)
            self.assertFalse((curr_ranks - expected_ranks).any())

    def test_trace_ranks_pos_size_1_integer_rank(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        unique_dict = {'Inner1': {'group_scores': scorer.score_group(pro_single_ft)},
                       'Inner2': {'group_scores': scorer.score_group(pro_single_ft_i2)},
                       'seq1': {'group_scores': scorer.score_group(pro_single_ft_s1)},
                       'seq2': {'group_scores': scorer.score_group(pro_single_ft_s2)},
                       'seq3': {'group_scores': scorer.score_group(pro_single_ft_s3)}}
        init_trace_ranks(scorer=scorer, a_dict=rank_dict, u_dict=unique_dict, low_memory=False,
                         unique_dir=None)
        self.evaluate_trace_ranks(scorer=scorer, u_dict=unique_dict, mode='int')

    def test_trace_ranks_pos_size_2_integer_rank(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        unique_dict = {'Inner1': {'group_scores': scorer.score_group(pro_pair_ft)},
                       'Inner2': {'group_scores': scorer.score_group(pro_pair_ft_i2)},
                       'seq1': {'group_scores': scorer.score_group(pro_pair_ft_s1)},
                       'seq2': {'group_scores': scorer.score_group(pro_pair_ft_s2)},
                       'seq3': {'group_scores': scorer.score_group(pro_pair_ft_s3)}}
        init_trace_ranks(scorer=scorer, a_dict=rank_dict, u_dict=unique_dict, low_memory=False,
                         unique_dir=None)
        self.evaluate_trace_ranks(scorer=scorer, u_dict=unique_dict, mode='int')

    def test_trace_ranks_low_mem_pos_size_1_integer_rank(self):
        u_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(u_dir)
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        unique_dict = {'Inner1': {'group_scores': save_numpy_array(scorer.score_group(pro_single_ft), out_dir=u_dir,
                                                                   node_name='Inner1', pos_type='single',
                                                                   score_type='group', metric=scorer.metric,
                                                                   low_memory=True)},
                       'Inner2': {'group_scores': save_numpy_array(scorer.score_group(pro_single_ft_i2), out_dir=u_dir,
                                                                   node_name='Inner2', pos_type='single',
                                                                   score_type='group', metric=scorer.metric,
                                                                   low_memory=True)},
                       'seq1': {'group_scores': save_numpy_array(scorer.score_group(pro_single_ft_s1), out_dir=u_dir,
                                                                 node_name='seq1', pos_type='single',
                                                                 score_type='group', metric=scorer.metric,
                                                                 low_memory=True)},
                       'seq2': {'group_scores': save_numpy_array(scorer.score_group(pro_single_ft_s2), out_dir=u_dir,
                                                                 node_name='seq2', pos_type='single',
                                                                 score_type='group', metric=scorer.metric,
                                                                 low_memory=True)},
                       'seq3': {'group_scores': save_numpy_array(scorer.score_group(pro_single_ft_s3), out_dir=u_dir,
                                                                 node_name='seq3', pos_type='single',
                                                                 score_type='group', metric=scorer.metric,
                                                                 low_memory=True)}}
        init_trace_ranks(scorer=scorer, a_dict=rank_dict, u_dict=unique_dict, low_memory=True,
                         unique_dir=u_dir)
        self.evaluate_trace_ranks(scorer=scorer, u_dict=unique_dict, mode='int', mem='low')
        rmtree(u_dir)

    def test_trace_ranks_low_mem_pos_size_2_integer_rank(self):
        u_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(u_dir)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        unique_dict = {'Inner1': {'group_scores': save_numpy_array(scorer.score_group(pro_pair_ft), out_dir=u_dir,
                                                                   node_name='Inner1', pos_type='pair',
                                                                   score_type='group', metric=scorer.metric,
                                                                   low_memory=True)},
                       'Inner2': {'group_scores': save_numpy_array(scorer.score_group(pro_pair_ft_i2), out_dir=u_dir,
                                                                   node_name='Inner2', pos_type='pair',
                                                                   score_type='group', metric=scorer.metric,
                                                                   low_memory=True)},
                       'seq1': {'group_scores': save_numpy_array(scorer.score_group(pro_pair_ft_s1), out_dir=u_dir,
                                                                 node_name='seq1', pos_type='pair',
                                                                 score_type='group', metric=scorer.metric,
                                                                 low_memory=True)},
                       'seq2': {'group_scores': save_numpy_array(scorer.score_group(pro_pair_ft_s2), out_dir=u_dir,
                                                                 node_name='seq2', pos_type='pair',
                                                                 score_type='group', metric=scorer.metric,
                                                                 low_memory=True)},
                       'seq3': {'group_scores': save_numpy_array(scorer.score_group(pro_pair_ft_s3), out_dir=u_dir,
                                                                 node_name='seq3', pos_type='pair',
                                                                 score_type='group', metric=scorer.metric,
                                                                 low_memory=True)}}
        init_trace_ranks(scorer=scorer, a_dict=rank_dict, u_dict=unique_dict, low_memory=True,
                         unique_dir=u_dir)
        self.evaluate_trace_ranks(scorer=scorer, u_dict=unique_dict, mode='int', mem='low')
        rmtree(u_dir)

    def test_trace_ranks_pos_size_1_real_rank(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='plain_entropy')
        unique_dict = {'Inner1': {'group_scores': scorer.score_group(pro_single_ft)},
                       'Inner2': {'group_scores': scorer.score_group(pro_single_ft_i2)},
                       'seq1': {'group_scores': scorer.score_group(pro_single_ft_s1)},
                       'seq2': {'group_scores': scorer.score_group(pro_single_ft_s2)},
                       'seq3': {'group_scores': scorer.score_group(pro_single_ft_s3)}}
        init_trace_ranks(scorer=scorer, a_dict=rank_dict, u_dict=unique_dict, low_memory=False,
                         unique_dir=None)
        self.evaluate_trace_ranks(scorer=scorer, u_dict=unique_dict, mode='real')

    def test_trace_ranks_pos_size_2_real_rank(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        unique_dict = {'Inner1': {'group_scores': scorer.score_group(pro_pair_ft)},
                       'Inner2': {'group_scores': scorer.score_group(pro_pair_ft_i2)},
                       'seq1': {'group_scores': scorer.score_group(pro_pair_ft_s1)},
                       'seq2': {'group_scores': scorer.score_group(pro_pair_ft_s2)},
                       'seq3': {'group_scores': scorer.score_group(pro_pair_ft_s3)}}
        init_trace_ranks(scorer=scorer, a_dict=rank_dict, u_dict=unique_dict, low_memory=False,
                         unique_dir=None)
        self.evaluate_trace_ranks(scorer=scorer, u_dict=unique_dict, mode='real')

    def test_trace_ranks_low_mem_pos_size_1_real_rank(self):
        u_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(u_dir)
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='plain_entropy')
        unique_dict = {'Inner1': {'group_scores': save_numpy_array(scorer.score_group(pro_single_ft), out_dir=u_dir,
                                                                   node_name='Inner1', pos_type='single',
                                                                   score_type='group', metric=scorer.metric,
                                                                   low_memory=True)},
                       'Inner2': {'group_scores': save_numpy_array(scorer.score_group(pro_single_ft_i2), out_dir=u_dir,
                                                                   node_name='Inner2', pos_type='single',
                                                                   score_type='group', metric=scorer.metric,
                                                                   low_memory=True)},
                       'seq1': {'group_scores': save_numpy_array(scorer.score_group(pro_single_ft_s1), out_dir=u_dir,
                                                                 node_name='seq1', pos_type='single',
                                                                 score_type='group', metric=scorer.metric,
                                                                 low_memory=True)},
                       'seq2': {'group_scores': save_numpy_array(scorer.score_group(pro_single_ft_s2), out_dir=u_dir,
                                                                 node_name='seq2', pos_type='single',
                                                                 score_type='group', metric=scorer.metric,
                                                                 low_memory=True)},
                       'seq3': {'group_scores': save_numpy_array(scorer.score_group(pro_single_ft_s3), out_dir=u_dir,
                                                                 node_name='seq3', pos_type='single',
                                                                 score_type='group', metric=scorer.metric,
                                                                 low_memory=True)}}
        init_trace_ranks(scorer=scorer, a_dict=rank_dict, u_dict=unique_dict, low_memory=True,
                         unique_dir=u_dir)
        self.evaluate_trace_ranks(scorer=scorer, u_dict=unique_dict, mode='real', mem='low')
        rmtree(u_dir)

    def test_trace_ranks_low_mem_pos_size_2_real_rank(self):
        u_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(u_dir)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        unique_dict = {'Inner1': {'group_scores': save_numpy_array(scorer.score_group(pro_pair_ft), out_dir=u_dir,
                                                                   node_name='Inner1', pos_type='pair',
                                                                   score_type='group', metric=scorer.metric,
                                                                   low_memory=True)},
                       'Inner2': {'group_scores': save_numpy_array(scorer.score_group(pro_pair_ft_i2), out_dir=u_dir,
                                                                   node_name='Inner2', pos_type='pair',
                                                                   score_type='group', metric=scorer.metric,
                                                                   low_memory=True)},
                       'seq1': {'group_scores': save_numpy_array(scorer.score_group(pro_pair_ft_s1), out_dir=u_dir,
                                                                 node_name='seq1', pos_type='pair',
                                                                 score_type='group', metric=scorer.metric,
                                                                 low_memory=True)},
                       'seq2': {'group_scores': save_numpy_array(scorer.score_group(pro_pair_ft_s2), out_dir=u_dir,
                                                                 node_name='seq2', pos_type='pair',
                                                                 score_type='group', metric=scorer.metric,
                                                                 low_memory=True)},
                       'seq3': {'group_scores': save_numpy_array(scorer.score_group(pro_pair_ft_s3), out_dir=u_dir,
                                                                 node_name='seq3', pos_type='pair',
                                                                 score_type='group', metric=scorer.metric,
                                                                 low_memory=True)}}
        init_trace_ranks(scorer=scorer, a_dict=rank_dict, u_dict=unique_dict, low_memory=True,
                         unique_dir=u_dir)
        self.evaluate_trace_ranks(scorer=scorer, u_dict=unique_dict, mode='real', mem='low')
        rmtree(u_dir)

    def test_trace_ranks_failure_no_rank(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        unique_dict = {'Inner1': {'group_scores': scorer.score_group(pro_single_ft)}}
        init_trace_ranks(scorer=scorer, a_dict=rank_dict, u_dict=unique_dict, low_memory=False,
                         unique_dir=None)
        with self.assertRaises(KeyError):
            trace_ranks(None)

    def test_trace_ranks_failure_no_scorer(self):
        unique_dict = {'Inner1': {'group_scores': np.random.rand(6)}}
        init_trace_ranks(scorer=None, a_dict=rank_dict, u_dict=unique_dict, low_memory=False,
                         unique_dir=None)
        with self.assertRaises(AttributeError):
            trace_ranks(1)

    def test_trace_ranks_failure_no_a_dict(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        unique_dict = {'Inner1': {'group_scores': scorer.score_group(pro_single_ft)}}
        init_trace_ranks(scorer=scorer, a_dict=None, u_dict=unique_dict, low_memory=False,
                         unique_dir=None)
        with self.assertRaises(TypeError):
            trace_ranks(1)

    def test_trace_ranks_failure_no_u_dict(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        init_trace_ranks(scorer=scorer, a_dict=rank_dict, u_dict=None, low_memory=False,
                         unique_dir=None)
        with self.assertRaises(TypeError):
            trace_ranks(1)

    def test_trace_ranks_failure_bad_low_memory(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        unique_dict = {'Inner1': {'group_scores': scorer.score_group(pro_single_ft)}}
        init_trace_ranks(scorer=scorer, a_dict=rank_dict, u_dict=unique_dict, low_memory='low_mem',
                         unique_dir=None)
        with self.assertRaises(ValueError):
            trace_ranks(1)

    def test_trace_ranks_failure_low_memory_no_unique_dir(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        unique_dict = {'Inner1': None}
        init_trace_ranks(scorer=scorer, a_dict=rank_dict, u_dict=unique_dict, low_memory=True, unique_dir=None)
        with self.assertRaises(ValueError):
            trace_ranks(1)


class TestTraceInit(TestCase):

    def test_init_out_dir(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        self.assertIs(trace.aln, aln)
        self.assertIs(trace.phylo_tree, phylo_tree)
        self.assertIs(trace.assignments, rank_dict)
        self.assertFalse(trace.match_mismatch)
        self.assertFalse(trace.low_memory)
        self.assertEqual(trace.pos_size, 1)
        self.assertEqual(trace.out_dir, expected_dir)
        self.assertIsNone(trace.unique_nodes)
        self.assertIsNone(trace.rank_scores)
        self.assertIsNone(trace.final_scores)
        self.assertIsNone(trace.final_ranks)
        self.assertIsNone(trace.final_coverage)
        rmtree(expected_dir)

    def test_init_out_dir_does_not_exist(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        self.assertFalse(os.path.isdir(expected_dir))
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        self.assertIs(trace.aln, aln)
        self.assertIs(trace.phylo_tree, phylo_tree)
        self.assertIs(trace.assignments, rank_dict)
        self.assertFalse(trace.match_mismatch)
        self.assertFalse(trace.low_memory)
        self.assertEqual(trace.pos_size, 1)
        self.assertEqual(trace.out_dir, expected_dir)
        self.assertIsNone(trace.unique_nodes)
        self.assertIsNone(trace.rank_scores)
        self.assertIsNone(trace.final_scores)
        self.assertIsNone(trace.final_ranks)
        self.assertIsNone(trace.final_coverage)
        self.assertTrue(os.path.isdir(expected_dir))
        rmtree(expected_dir)

    def test_init_no_out_dir(self):
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=False, output_dir=None, low_memory=False)
        self.assertIs(trace.aln, aln)
        self.assertIs(trace.phylo_tree, phylo_tree)
        self.assertIs(trace.assignments, rank_dict)
        self.assertFalse(trace.match_mismatch)
        self.assertFalse(trace.low_memory)
        self.assertEqual(trace.pos_size, 1)
        self.assertEqual(trace.out_dir, os.getcwd())
        self.assertIsNone(trace.unique_nodes)
        self.assertIsNone(trace.rank_scores)
        self.assertIsNone(trace.final_scores)
        self.assertIsNone(trace.final_ranks)
        self.assertIsNone(trace.final_coverage)

    def test_init_position_size_2(self):
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=2,
                      match_mismatch=False, output_dir=None, low_memory=False)
        self.assertIs(trace.aln, aln)
        self.assertIs(trace.phylo_tree, phylo_tree)
        self.assertIs(trace.assignments, rank_dict)
        self.assertFalse(trace.match_mismatch)
        self.assertEqual(trace.pos_size, 2)
        self.assertFalse(trace.low_memory)
        self.assertEqual(trace.out_dir, os.getcwd())
        self.assertIsNone(trace.unique_nodes)
        self.assertIsNone(trace.rank_scores)
        self.assertIsNone(trace.final_scores)
        self.assertIsNone(trace.final_ranks)
        self.assertIsNone(trace.final_coverage)

    def test_init_failure_position_size_low(self):
        with self.assertRaises(ValueError):
            Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=0,
                  match_mismatch=False, output_dir=None, low_memory=False)

    def test_init_failure_position_size_high(self):
        with self.assertRaises(ValueError):
            Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=3,
                  match_mismatch=False, output_dir=None, low_memory=False)

    def test_init_failure_bad_alignment(self):
        with self.assertRaises(ValueError):
            Trace(alignment=None, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                  match_mismatch=False, output_dir=None, low_memory=False)

    def test_init_failure_bad_phylo_tree(self):
        with self.assertRaises(ValueError):
            Trace(alignment=aln, phylo_tree=None, group_assignments=rank_dict, pos_size=1,
                  match_mismatch=False, output_dir=None, low_memory=False)

    def test_init_failure_bad_group_assignments(self):
        with self.assertRaises(ValueError):
            Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=None, pos_size=1,
                  match_mismatch=False, output_dir=None, low_memory=False)

    def test_init_failure_bad_match_mismatch(self):
        with self.assertRaises(ValueError):
            Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                  match_mismatch=None, output_dir=None, low_memory=False)

    def test_init_failure_bad_low_memory(self):
        with self.assertRaises(ValueError):
            Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                  match_mismatch=False, output_dir=None, low_memory=None)

    def test_init_failure_bad_output_dir(self):
        with self.assertRaises(TypeError):
            Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                  match_mismatch=False, output_dir=100, low_memory=False)


class TestTraceCharacterizeRankGroupStandard(TestCase):

    def test_characterize_rank_groups_standard_single(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups_standard(unique_dir=None, alpha_size=protein_alpha_size,
                                                alpha_mapping=protein_map, alpha_reverse=protein_rev,
                                                single_to_pair=None, processes=1, write_out_sub_aln=False,
                                                write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            expected1 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]))
            self.assertEqual(trace.unique_nodes[node_name]['freq_table'].get_depth(), expected1.get_depth())
            self.assertFalse((trace.unique_nodes[node_name]['freq_table'].get_count_matrix() -
                              expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            expected2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected2.characterize_alignment(num_aln=num_aln[inds, :])
            self.assertEqual(trace.unique_nodes[node_name]['freq_table'].get_depth(), expected2.get_depth())
            self.assertFalse((trace.unique_nodes[node_name]['freq_table'].get_count_matrix() -
                              expected2.get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_single_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups_standard(unique_dir=None, alpha_size=protein_alpha_size,
                                                alpha_mapping=protein_map, alpha_reverse=protein_rev,
                                                single_to_pair=None, processes=2, write_out_sub_aln=False,
                                                write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            expected1 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]))
            self.assertEqual(trace.unique_nodes[node_name]['freq_table'].get_depth(), expected1.get_depth())
            self.assertFalse((trace.unique_nodes[node_name]['freq_table'].get_count_matrix() -
                              expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            expected2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected2.characterize_alignment(num_aln=num_aln[inds, :])
            self.assertEqual(trace.unique_nodes[node_name]['freq_table'].get_depth(), expected2.get_depth())
            self.assertFalse((trace.unique_nodes[node_name]['freq_table'].get_count_matrix() -
                              expected2.get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_single_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=False, output_dir=expected_dir, low_memory=True)
        trace.characterize_rank_groups_standard(unique_dir=expected_dir, alpha_size=protein_alpha_size,
                                                alpha_mapping=protein_map, alpha_reverse=protein_rev,
                                                single_to_pair=None, processes=1, write_out_sub_aln=False,
                                                write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], str)
            loaded1 = load_freq_table(trace.unique_nodes[node_name]['freq_table'], True)
            expected1 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]))
            self.assertEqual(loaded1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded1.get_count_matrix() - expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], str)
            loaded2 = load_freq_table(trace.unique_nodes[node_name]['freq_table'], True)
            expected2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected2.characterize_alignment(num_aln=num_aln[inds, :])
            self.assertEqual(loaded2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded2.get_count_matrix() - expected2.get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_single_write_out(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups_standard(unique_dir=expected_dir, alpha_size=protein_alpha_size,
                                                alpha_mapping=protein_map, alpha_reverse=protein_rev,
                                                single_to_pair=None, processes=1, write_out_sub_aln=True,
                                                write_out_freq_table=True)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            expected1 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]))
            self.assertEqual(trace.unique_nodes[node_name]['freq_table'].get_depth(), expected1.get_depth())
            self.assertFalse((trace.unique_nodes[node_name]['freq_table'].get_count_matrix() -
                              expected1.get_count_matrix()).any())
            expected_path1a = os.path.join(expected_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path1a))
            expected_path1b = os.path.join(expected_dir, f'{node_name}_single_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path1b))
            loaded_1 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            loaded_1.load_csv(expected_path1b)
            self.assertEqual(loaded_1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded_1.get_count_matrix() - expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            expected2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            expected2.characterize_alignment(num_aln=num_aln[inds, :])
            self.assertEqual(trace.unique_nodes[node_name]['freq_table'].get_depth(), expected2.get_depth())
            self.assertFalse((trace.unique_nodes[node_name]['freq_table'].get_count_matrix() -
                              expected2.get_count_matrix()).any())
            expected_path2a = os.path.join(expected_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path2a))
            expected_path2b = os.path.join(expected_dir, f'{node_name}_single_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path2b))
            loaded_2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
            loaded_2.load_csv(expected_path2b)
            self.assertEqual(loaded_2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded_2.get_count_matrix() - expected2.get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_pair(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=2,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups_standard(unique_dir=None, alpha_size=pro_pair_alpha_size,
                                                alpha_mapping=pro_pair_map, alpha_reverse=pro_pair_rev,
                                                single_to_pair=pro_single_to_pair, processes=1,
                                                write_out_sub_aln=False, write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            expected1 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]), single_to_pair=pro_single_to_pair)
            self.assertEqual(trace.unique_nodes[node_name]['freq_table'].get_depth(), expected1.get_depth())
            self.assertFalse((trace.unique_nodes[node_name]['freq_table'].get_count_matrix() -
                              expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            expected2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected2.characterize_alignment(num_aln=num_aln[inds, :], single_to_pair=pro_single_to_pair)
            self.assertEqual(trace.unique_nodes[node_name]['freq_table'].get_depth(), expected2.get_depth())
            self.assertFalse((trace.unique_nodes[node_name]['freq_table'].get_count_matrix() -
                              expected2.get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_pair_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=2,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups_standard(unique_dir=None, alpha_size=pro_pair_alpha_size,
                                                alpha_mapping=pro_pair_map, alpha_reverse=pro_pair_rev,
                                                single_to_pair=pro_single_to_pair, processes=2,
                                                write_out_sub_aln=False, write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            expected1 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]), single_to_pair=pro_single_to_pair)
            self.assertEqual(trace.unique_nodes[node_name]['freq_table'].get_depth(), expected1.get_depth())
            self.assertFalse((trace.unique_nodes[node_name]['freq_table'].get_count_matrix() -
                              expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            expected2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected2.characterize_alignment(num_aln=num_aln[inds, :], single_to_pair=pro_single_to_pair)
            self.assertEqual(trace.unique_nodes[node_name]['freq_table'].get_depth(), expected2.get_depth())
            self.assertFalse((trace.unique_nodes[node_name]['freq_table'].get_count_matrix() -
                              expected2.get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_pair_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=2,
                      match_mismatch=False, output_dir=expected_dir, low_memory=True)
        trace.characterize_rank_groups_standard(unique_dir=expected_dir, alpha_size=pro_pair_alpha_size,
                                                alpha_mapping=pro_pair_map, alpha_reverse=pro_pair_rev,
                                                single_to_pair=pro_single_to_pair, processes=1,
                                                write_out_sub_aln=False, write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], str)
            loaded1 = load_freq_table(trace.unique_nodes[node_name]['freq_table'], True)
            expected1 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]), single_to_pair=pro_single_to_pair)
            self.assertEqual(loaded1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded1.get_count_matrix() - expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], str)
            loaded2 = load_freq_table(trace.unique_nodes[node_name]['freq_table'], True)
            expected2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected2.characterize_alignment(num_aln=num_aln[inds, :], single_to_pair=pro_single_to_pair)
            self.assertEqual(loaded2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded2.get_count_matrix() - expected2.get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_pair_write_out(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=2,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups_standard(unique_dir=expected_dir, alpha_size=pro_pair_alpha_size,
                                                alpha_mapping=pro_pair_map, alpha_reverse=pro_pair_rev,
                                                single_to_pair=pro_single_to_pair, processes=1,
                                                write_out_sub_aln=True, write_out_freq_table=True)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            expected1 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected1.characterize_alignment(num_aln=np.array([num_aln[ind, :]]), single_to_pair=pro_single_to_pair)
            self.assertEqual(trace.unique_nodes[node_name]['freq_table'].get_depth(), expected1.get_depth())
            self.assertFalse((trace.unique_nodes[node_name]['freq_table'].get_count_matrix() -
                              expected1.get_count_matrix()).any())
            expected_path1a = os.path.join(expected_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path1a))
            expected_path1b = os.path.join(expected_dir, f'{node_name}_pair_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path1b))
            loaded_1 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            loaded_1.load_csv(expected_path1b)
            self.assertEqual(loaded_1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded_1.get_count_matrix() - expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            expected2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            expected2.characterize_alignment(num_aln=num_aln[inds, :], single_to_pair=pro_single_to_pair)
            self.assertEqual(trace.unique_nodes[node_name]['freq_table'].get_depth(), expected2.get_depth())
            self.assertFalse((trace.unique_nodes[node_name]['freq_table'].get_count_matrix() -
                              expected2.get_count_matrix()).any())
            expected_path2a = os.path.join(expected_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path2a))
            expected_path2b = os.path.join(expected_dir, f'{node_name}_pair_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path2b))
            loaded_2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
            loaded_2.load_csv(expected_path2b)
            self.assertEqual(loaded_2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded_2.get_count_matrix() - expected2.get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_failure_low_mem_no_unique_dir(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=False, output_dir=expected_dir, low_memory=True)
        with self.assertRaises(ValueError):
            trace.characterize_rank_groups_standard(unique_dir=None, alpha_size=protein_alpha_size,
                                                    alpha_mapping=protein_map, alpha_reverse=protein_rev,
                                                    single_to_pair=None, processes=1, write_out_sub_aln=False,
                                                    write_out_freq_table=False)
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_failure_write_out_no_unique_dir(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        with self.assertRaises(ValueError):
            trace.characterize_rank_groups_standard(unique_dir=None, alpha_size=protein_alpha_size,
                                                    alpha_mapping=protein_map, alpha_reverse=protein_rev,
                                                    single_to_pair=None, processes=1, write_out_sub_aln=True,
                                                    write_out_freq_table=True)
        rmtree(expected_dir)

    # Missing (i.e. None) alphabet size, mapping, reverse mapping, and single to pair mapping will all succeed because
    # the characterize_positions and characterize_positions2 methods in SeqAlignment which are called by this method
    # will generate those if missing from the provided alignment.

    def test_characterize_rank_groups_standard_failure_no_processes(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        with self.assertRaises(ValueError):
            trace.characterize_rank_groups_standard(unique_dir=None, alpha_size=protein_alpha_size,
                                                    alpha_mapping=protein_map, alpha_reverse=protein_rev,
                                                    single_to_pair=None, processes=None, write_out_sub_aln=True,
                                                    write_out_freq_table=True)
        rmtree(expected_dir)


class TestTraceCharacterizeRankGroupMatchMismatch(TestCase):

    def test_characterize_rank_groups_match_mismatch_single(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups_match_mismatch(unique_dir=None, single_size=protein_alpha_size,
                                                      single_mapping=protein_map, single_reverse=protein_rev,
                                                      processes=1, write_out_sub_aln=False, write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded1a = trace.unique_nodes[node_name]['match']
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded1b = trace.unique_nodes[node_name]['mismatch']
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded2a = trace.unique_nodes[node_name]['match']
            self.assertEqual(loaded2a.get_depth(), expected_single_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_single_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded2b = trace.unique_nodes[node_name]['mismatch']
            self.assertEqual(loaded2b.get_depth(), expected_single_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_single_tables[node_name]['mismatch'].get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_single_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups_match_mismatch(unique_dir=None, single_size=protein_alpha_size,
                                                      single_mapping=protein_map, single_reverse=protein_rev,
                                                      processes=2, write_out_sub_aln=False,
                                                      write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded1a = trace.unique_nodes[node_name]['match']
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded1b = trace.unique_nodes[node_name]['mismatch']
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded2a = trace.unique_nodes[node_name]['match']
            self.assertEqual(loaded2a.get_depth(), expected_single_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_single_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded2b = trace.unique_nodes[node_name]['mismatch']
            self.assertEqual(loaded2b.get_depth(), expected_single_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_single_tables[node_name]['mismatch'].get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_single_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=True, output_dir=expected_dir, low_memory=True)
        trace.characterize_rank_groups_match_mismatch(unique_dir=expected_dir, single_size=protein_alpha_size,
                                                      single_mapping=protein_map, single_reverse=protein_rev,
                                                      processes=1, write_out_sub_aln=False,
                                                      write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], str)
            loaded1a = load_freq_table(trace.unique_nodes[node_name]['match'], True)
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], str)
            loaded1b = load_freq_table(trace.unique_nodes[node_name]['mismatch'], True)
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], str)
            loaded2a = load_freq_table(trace.unique_nodes[node_name]['match'], True)
            self.assertEqual(loaded2a.get_depth(), expected_single_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_single_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], str)
            loaded2b = load_freq_table(trace.unique_nodes[node_name]['mismatch'], True)
            self.assertEqual(loaded2b.get_depth(), expected_single_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_single_tables[node_name]['mismatch'].get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_single_write_out(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups_match_mismatch(unique_dir=expected_dir, single_size=protein_alpha_size,
                                                      single_mapping=protein_map, single_reverse=protein_rev,
                                                      processes=1, write_out_sub_aln=True, write_out_freq_table=True)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded1a = trace.unique_nodes[node_name]['match']
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded1b = trace.unique_nodes[node_name]['mismatch']
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
            expected_path1a = os.path.join(expected_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path1a))
            expected_path1b = os.path.join(expected_dir, f'{node_name}_single_match_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path1b))
            loaded_1a = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
            loaded_1a.load_csv(expected_path1b)
            self.assertEqual(loaded_1a.get_depth(), 1)
            self.assertFalse(loaded_1a.get_count_matrix().any())
            expected_path1c = os.path.join(expected_dir, f'{node_name}_single_mismatch_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path1c))
            loaded_1b = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
            loaded_1b.load_csv(expected_path1c)
            self.assertEqual(loaded_1b.get_depth(), 1)
            self.assertFalse(loaded_1b.get_count_matrix().any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded2a = trace.unique_nodes[node_name]['match']
            self.assertEqual(loaded2a.get_depth(), expected_single_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_single_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded2b = trace.unique_nodes[node_name]['mismatch']
            self.assertEqual(loaded2b.get_depth(), expected_single_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_single_tables[node_name]['mismatch'].get_count_matrix()).any())
            expected_path2a = os.path.join(expected_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path2a))
            expected_path2b = os.path.join(expected_dir, f'{node_name}_single_match_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path2b))
            loaded_2a = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
            loaded_2a.load_csv(expected_path2b, expected_single_tables[node_name]['match'].get_depth())
            self.assertEqual(loaded_2a.get_depth(), expected_single_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded_2a.get_count_matrix() -
                              expected_single_tables[node_name]['match'].get_count_matrix()).any())
            expected_path2c = os.path.join(expected_dir, f'{node_name}_single_mismatch_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path2c))
            loaded_2b = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 1)
            loaded_2b.load_csv(expected_path2c, expected_single_tables[node_name]['match'].get_depth())
            self.assertEqual(loaded_2b.get_depth(), expected_single_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded_2b.get_count_matrix() -
                              expected_single_tables[node_name]['mismatch'].get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_pair(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups_match_mismatch(unique_dir=None, single_size=protein_alpha_size,
                                                      single_mapping=protein_map, single_reverse=protein_rev,
                                                      processes=1, write_out_sub_aln=False, write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded1a = trace.unique_nodes[node_name]['match']
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded1b = trace.unique_nodes[node_name]['mismatch']
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded2a = trace.unique_nodes[node_name]['match']
            self.assertEqual(loaded2a.get_depth(), expected_pair_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_pair_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded2b = trace.unique_nodes[node_name]['mismatch']
            self.assertEqual(loaded2b.get_depth(), expected_pair_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_pair_tables[node_name]['mismatch'].get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_pair_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups_match_mismatch(unique_dir=None, single_size=protein_alpha_size,
                                                      single_mapping=protein_map, single_reverse=protein_rev,
                                                      processes=2, write_out_sub_aln=False,
                                                      write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded1a = trace.unique_nodes[node_name]['match']
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded1b = trace.unique_nodes[node_name]['mismatch']
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded2a = trace.unique_nodes[node_name]['match']
            self.assertEqual(loaded2a.get_depth(), expected_pair_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_pair_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded2b = trace.unique_nodes[node_name]['mismatch']
            self.assertEqual(loaded2b.get_depth(), expected_pair_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_pair_tables[node_name]['mismatch'].get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_pair_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=True)
        trace.characterize_rank_groups_match_mismatch(unique_dir=expected_dir, single_size=protein_alpha_size,
                                                      single_mapping=protein_map, single_reverse=protein_rev,
                                                      processes=1, write_out_sub_aln=False,
                                                      write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], str)
            loaded1a = load_freq_table(trace.unique_nodes[node_name]['match'], True)
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], str)
            loaded1b = load_freq_table(trace.unique_nodes[node_name]['mismatch'], True)
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], str)
            loaded2a = load_freq_table(trace.unique_nodes[node_name]['match'], True)
            self.assertEqual(loaded2a.get_depth(), expected_pair_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_pair_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], str)
            loaded2b = load_freq_table(trace.unique_nodes[node_name]['mismatch'], True)
            self.assertEqual(loaded2b.get_depth(), expected_pair_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_pair_tables[node_name]['mismatch'].get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_pair_write_out(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups_match_mismatch(unique_dir=expected_dir, single_size=protein_alpha_size,
                                                      single_mapping=protein_map, single_reverse=protein_rev,
                                                      processes=1, write_out_sub_aln=True, write_out_freq_table=True)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded1a = trace.unique_nodes[node_name]['match']
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded1b = trace.unique_nodes[node_name]['mismatch']
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
            expected_path1a = os.path.join(expected_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path1a))
            expected_path1b = os.path.join(expected_dir, f'{node_name}_pair_match_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path1b))
            loaded_1a = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
            loaded_1a.load_csv(expected_path1b)
            self.assertEqual(loaded_1a.get_depth(), 1)
            self.assertFalse(loaded_1a.get_count_matrix().any())
            expected_path1c = os.path.join(expected_dir, f'{node_name}_pair_mismatch_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path1c))
            loaded_1b = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
            loaded_1b.load_csv(expected_path1c)
            self.assertEqual(loaded_1b.get_depth(), 1)
            self.assertFalse(loaded_1b.get_count_matrix().any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded2a = trace.unique_nodes[node_name]['match']
            self.assertEqual(loaded2a.get_depth(), expected_pair_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_pair_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded2b = trace.unique_nodes[node_name]['mismatch']
            self.assertEqual(loaded2b.get_depth(), expected_pair_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_pair_tables[node_name]['mismatch'].get_count_matrix()).any())
            expected_path2a = os.path.join(expected_dir, f'{node_name}.fa')
            self.assertTrue(os.path.isfile(expected_path2a))
            expected_path2b = os.path.join(expected_dir, f'{node_name}_pair_match_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path2b))
            loaded_2a = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
            loaded_2a.load_csv(expected_path2b, expected_pair_tables[node_name]['match'].get_depth())
            self.assertEqual(loaded_2a.get_depth(), expected_pair_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded_2a.get_count_matrix() -
                              expected_pair_tables[node_name]['match'].get_count_matrix()).any())
            expected_path2c = os.path.join(expected_dir, f'{node_name}_pair_mismatch_freq_table.tsv')
            self.assertTrue(os.path.isfile(expected_path2c))
            loaded_2b = FrequencyTable(pro_quad_alpha_size, pro_quad_map, pro_quad_rev, 6, 2)
            loaded_2b.load_csv(expected_path2c, expected_pair_tables[node_name]['match'].get_depth())
            self.assertEqual(loaded_2b.get_depth(), expected_pair_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded_2b.get_count_matrix() -
                              expected_pair_tables[node_name]['mismatch'].get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_failure_low_mem_no_unique_dir(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=True, output_dir=expected_dir, low_memory=True)
        with self.assertRaises(ValueError):
            trace.characterize_rank_groups_match_mismatch(unique_dir=None, single_size=protein_alpha_size,
                                                          single_mapping=protein_map, single_reverse=protein_rev,
                                                          processes=1, write_out_sub_aln=False,
                                                          write_out_freq_table=False)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_failure_no_single_size(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        with self.assertRaises(TypeError):
            trace.characterize_rank_groups_match_mismatch(unique_dir=None, single_size=None,
                                                          single_mapping=protein_map, single_reverse=protein_rev,
                                                          processes=1, write_out_sub_aln=False,
                                                          write_out_freq_table=False)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_failure_no_single_mapping(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        with self.assertRaises(TypeError):
            trace.characterize_rank_groups_match_mismatch(unique_dir=None, single_size=protein_alpha_size,
                                                          single_mapping=None, single_reverse=protein_rev,
                                                          processes=1, write_out_sub_aln=False,
                                                          write_out_freq_table=False)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_failure_no_single_reverse(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        with self.assertRaises(TypeError):
            trace.characterize_rank_groups_match_mismatch(unique_dir=None, single_size=protein_alpha_size,
                                                          single_mapping=protein_map, single_reverse=None,
                                                          processes=1, write_out_sub_aln=False,
                                                          write_out_freq_table=False)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_success_no_processes(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups_match_mismatch(unique_dir=None, single_size=protein_alpha_size,
                                                      single_mapping=protein_map, single_reverse=protein_rev,
                                                      processes=None, write_out_sub_aln=False,
                                                      write_out_freq_table=False)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded1a = trace.unique_nodes[node_name]['match']
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded1b = trace.unique_nodes[node_name]['mismatch']
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded2a = trace.unique_nodes[node_name]['match']
            self.assertEqual(loaded2a.get_depth(), expected_single_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_single_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded2b = trace.unique_nodes[node_name]['mismatch']
            self.assertEqual(loaded2b.get_depth(), expected_single_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_single_tables[node_name]['mismatch'].get_count_matrix()).any())
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_failure_write_out_no_unique_dir(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=1,
                      match_mismatch=True, output_dir=expected_dir, low_memory=True)
        with self.assertRaises(ValueError):
            trace.characterize_rank_groups_match_mismatch(unique_dir=None, single_size=protein_alpha_size,
                                                          single_mapping=protein_map, single_reverse=protein_rev,
                                                          processes=1, write_out_sub_aln=True,
                                                          write_out_freq_table=True)
        rmtree(expected_dir)

# class TestTraceCharacterizeRankGroups(TestCase):
# class TestTraceTrace(TestCase):

# class TestTrace(TestBase):
#
#     @classmethod
#     def setUpClass(cls):
#         super(TestTrace, cls).setUpClass()
#         cls.query_aln_fa_small = SeqAlignment(
#             file_name=cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln'],
#             query_id=cls.small_structure_id)
#         cls.query_aln_fa_small.import_alignment()
#         cls.phylo_tree_small = PhylogeneticTree()
#         calc = AlignmentDistanceCalculator()
#         cls.phylo_tree_small.construct_tree(dm=calc.get_distance(cls.query_aln_fa_small.alignment))
#         cls.assignments_small = cls.phylo_tree_small.assign_group_rank()
#         cls.assignments_custom_small = cls.phylo_tree_small.assign_group_rank(ranks=[1, 2, 3, 5, 7, 10, 25])
#         cls.query_aln_fa_large = SeqAlignment(
#             file_name=cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln'],
#             query_id=cls.large_structure_id)
#         cls.query_aln_fa_large.import_alignment()
#         cls.phylo_tree_large = PhylogeneticTree()
#         calc = AlignmentDistanceCalculator()
#         cls.phylo_tree_large.construct_tree(dm=calc.get_distance(cls.query_aln_fa_large.alignment))
#         cls.assignments_large = cls.phylo_tree_large.assign_group_rank()
#         cls.assignments_custom_large = cls.phylo_tree_large.assign_group_rank(ranks=[1, 2, 3, 5, 7, 10, 25])
#         cls.query_aln_msf_small = deepcopy(cls.query_aln_fa_small)
#         cls.query_aln_msf_small.file_name = cls.data_set.protein_data[cls.small_structure_id]['Final_MSF_Aln']
#         cls.out_small_dir = os.path.join(cls.testing_dir, cls.small_structure_id)
#         if not os.path.isdir(cls.out_small_dir):
#             os.makedirs(cls.out_small_dir)
#         cls.query_aln_msf_large = deepcopy(cls.query_aln_fa_large)
#         cls.query_aln_msf_large.file_name = cls.data_set.protein_data[cls.large_structure_id]['Final_MSF_Aln']
#         cls.out_large_dir = os.path.join(cls.testing_dir, cls.large_structure_id)
#         if not os.path.isdir(cls.out_large_dir):
#             os.makedirs(cls.out_large_dir)
#         cls.query_aln_fa_small = cls.query_aln_fa_small.remove_gaps()
#         cls.query_aln_fa_large = cls.query_aln_fa_large.remove_gaps()
#         cls.single_alphabet = Gapped(FullIUPACProtein())
#         cls.single_size, _, cls.single_mapping, cls.single_reverse = build_mapping(alphabet=cls.single_alphabet)
#         cls.pair_alphabet = MultiPositionAlphabet(alphabet=cls.single_alphabet, size=2)
#         cls.pair_size, _, cls.pair_mapping, cls.pair_reverse = build_mapping(alphabet=cls.pair_alphabet)
#         cls.single_to_pair_dict = {}
#         for char in cls.pair_mapping:
#             key = (cls.single_mapping[char[0]], cls.single_mapping[char[1]])
#             cls.single_to_pair_dict[key] = cls.pair_mapping[char]
#         cls.single_to_pair_arr = np.zeros((max(cls.single_mapping.values()) + 1, max(cls.single_mapping.values()) + 1))
#         for char in cls.pair_mapping:
#             cls.single_to_pair_arr[cls.single_mapping[char[0]], cls.single_mapping[char[1]]] = cls.pair_mapping[char]
#         cls.quad_alphabet = MultiPositionAlphabet(alphabet=cls.single_alphabet, size=4)
#         cls.quad_size, _, cls.quad_mapping, cls.quad_reverse = build_mapping(alphabet=cls.quad_alphabet)
#         cls.single_to_quad_dict = {}
#         for char in cls.quad_mapping:
#             key = (cls.single_mapping[char[0]], cls.single_mapping[char[1]], cls.single_mapping[char[2]],
#                    cls.single_mapping[char[3]])
#             cls.single_to_quad_dict[key] = cls.quad_mapping[char]
#
#     # @classmethod
#     # def tearDownClass(cls):
#     #     rmtree(cls.out_small_dir)
#     #     rmtree(cls.out_large_dir)
#
#     def evalaute_init(self, aln, tree, assignments, pos_specific, pair_specific, out_dir, low_mem):
#         trace_small = Trace(alignment=aln, phylo_tree=tree, group_assignments=assignments,
#                             position_specific=pos_specific, pair_specific=pair_specific, output_dir=out_dir,
#                             low_memory=low_mem)
#         self.assertEqual(trace_small.aln.file_name, aln.file_name)
#         self.assertEqual(trace_small.aln.query_id, aln.query_id)
#         for s in range(trace_small.aln.size):
#             self.assertEqual(str(trace_small.aln.alignment[s].seq), str(aln.alignment[s].seq))
#         self.assertEqual(trace_small.aln.seq_order, aln.seq_order)
#         self.assertEqual(str(trace_small.aln.query_sequence), str(aln.query_sequence))
#         self.assertEqual(trace_small.aln.seq_length, aln.seq_length)
#         self.assertEqual(trace_small.aln.size, aln.size)
#         self.assertEqual(trace_small.aln.marked, aln.marked)
#         self.assertEqual(trace_small.aln.polymer_type, aln.polymer_type)
#         self.assertTrue(isinstance(trace_small.aln.alphabet, type(aln.alphabet)))
#         self.assertEqual(len(trace_small.aln.alphabet.letters), len(aln.alphabet.letters))
#         for char in trace_small.aln.alphabet.letters:
#             self.assertTrue(char in aln.alphabet.letters)
#         self.assertEqual(trace_small.phylo_tree, tree)
#         self.assertEqual(trace_small.assignments, assignments)
#         self.assertIsNone(trace_small.unique_nodes)
#         self.assertIsNone(trace_small.rank_scores)
#         self.assertEqual(trace_small.pos_specific, pos_specific)
#         self.assertEqual(trace_small.pair_specific, pair_specific)
#         self.assertEqual(trace_small.out_dir, out_dir)
#         self.assertEqual(trace_small.low_memory, low_mem)
#
#     def test1a_init(self):
#         self.evalaute_init(aln=self.query_aln_fa_small, tree=self.phylo_tree_small, assignments=self.assignments_small,
#                            pos_specific=True, pair_specific=True, out_dir=self.out_small_dir, low_mem=False)
#
#     def test1b_init(self):
#         self.evalaute_init(aln=self.query_aln_fa_large, tree=self.phylo_tree_large, assignments=self.assignments_large,
#                            pos_specific=True, pair_specific=True, out_dir=self.out_large_dir, low_mem=True)
#
#     def evaluate_characterize_rank_groups_pooling_functions(self, single, pair, aln, assign, out_dir, low_mem,
#                                                             write_sub_aln, write_freq_table):
#         unique_dir = os.path.join(out_dir, 'unique_node_data')
#         if not os.path.isdir(unique_dir):
#             os.makedirs(unique_dir)
#         # Build a minimal set of nodes to characterize (the query sequence, its neighbor node, and their parent node)
#         visited = {}
#         to_characterize = []
#         found_query = False
#         for r in sorted(assign.keys(), reverse=True):
#             for g in assign[r]:
#                 node = assign[r][g]['node']
#                 if assign[r][g]['descendants'] and (aln.query_id in [d.name for d in assign[r][g]['descendants']]):
#                     found_query = True
#                     descendants_to_find = set([d.name for d in assign[r][g]['descendants']])
#                     searching = len(descendants_to_find)
#                     for r2 in range(r + 1, max(assign.keys()) + 1):
#                         for g2 in assign[r2]:
#                             if assign[r2][g2]['node'].name in descendants_to_find:
#                                 to_characterize.append((assign[r2][g2]['node'].name, 'component'))
#                                 visited[assign[r2][g2]['node'].name] = {'terminals': assign[r2][g2]['terminals'],
#                                                                         'descendants': assign[r2][g2]['descendants']}
#                                 searching -= 1
#                         if searching == 0:
#                             break
#                 if found_query:
#                     to_characterize.append((node.name, 'inner'))
#                     visited[node.name] = {'terminals': assign[r][g]['terminals'],
#                                           'descendants': assign[r][g]['descendants']}
#                     break
#             if found_query:
#                 break
#         # Perform characterization
#         pool_manager = Manager()
#         lock = Lock()
#         frequency_tables = pool_manager.dict()
#         init_characterization_pool(self.single_size, self.single_mapping, self.single_reverse, self.pair_size,
#                                    self.pair_mapping, self.pair_reverse, self.single_to_pair_arr, aln, single, pair,
#                                    visited, frequency_tables, lock, unique_dir, low_mem, write_sub_aln,
#                                    write_freq_table, 1)
#         for to_char in to_characterize:
#             ret_name = characterization(*to_char)
#             self.assertEqual(ret_name, to_char[0])
#             # Evaluate the characterized positions
#             sub_aln = aln.generate_sub_alignment(sequence_ids=visited[ret_name]['terminals'])
#             if sub_aln.size >= 5:
#                 expected_single, expected_pair = sub_aln.characterize_positions2(
#                     single=single, pair=pair, single_letter_size=self.single_size,
#                     single_letter_mapping=self.single_mapping,
#                     single_letter_reverse=self.single_reverse, pair_letter_size=self.pair_size,
#                     pair_letter_mapping=self.pair_mapping, pair_letter_reverse=self.pair_reverse,
#                     single_to_pair=self.single_to_pair_arr)
#             else:
#                 expected_single, expected_pair = sub_aln.characterize_positions(
#                     single=single, pair=pair, single_size=self.single_size, single_mapping=self.single_mapping,
#                     single_reverse=self.single_reverse, pair_size=self.pair_size, pair_mapping=self.pair_mapping,
#                     pair_reverse=self.pair_reverse)
#             if write_sub_aln:
#                 self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}.fa'.format(ret_name))))
#             self.assertTrue(ret_name in frequency_tables)
#             self.assertTrue('single' in frequency_tables[ret_name])
#             if single:
#                 expected_single_array = expected_single.get_table().toarray()
#                 single_table = frequency_tables[ret_name]['single']
#                 if low_mem:
#                     single_table = load_freq_table(freq_table=single_table, low_memory=low_mem)
#                 single_array = single_table.get_table().toarray()
#                 single_diff = single_array - expected_single_array
#                 self.assertFalse(single_diff.any())
#                 if write_freq_table:
#                     self.assertTrue(os.path.isfile(os.path.join(unique_dir,
#                                                                 '{}_position_freq_table.tsv'.format(ret_name))))
#             else:
#                 self.assertIsNone(frequency_tables[ret_name]['single'])
#             self.assertTrue('pair' in frequency_tables[ret_name])
#             if pair:
#                 expected_pair_array = expected_pair.get_table().toarray()
#                 pair_table = frequency_tables[ret_name]['pair']
#                 if low_mem:
#                     pair_table = load_freq_table(freq_table=pair_table, low_memory=low_mem)
#                 pair_array = pair_table.get_table().toarray()
#                 pair_diff = pair_array - expected_pair_array
#                 self.assertFalse(pair_diff.any())
#                 if write_freq_table:
#                     self.assertTrue(os.path.isfile(os.path.join(unique_dir,
#                                                                 '{}_pair_freq_table.tsv'.format(ret_name))))
#             else:
#                 self.assertIsNone(frequency_tables[ret_name]['pair'])
#         rmtree(unique_dir)
#
#     def test2a_characterize_rank_groups_initialize_characterization_pool(self):
#         # Test pool initialization function and mappable function (minimal example) for characterization, small aln
#         self.evaluate_characterize_rank_groups_pooling_functions(
#             single=True, pair=True, aln=self.query_aln_fa_small, assign=self.assignments_small,
#             out_dir=self.out_small_dir, low_mem=False, write_sub_aln=True, write_freq_table=True)
#
#     def test2b_characterize_rank_groups_initialize_characterization_pool(self):
#         # Test pool initialization function and mappable function (minimal example) for characterization, large aln
#         self.evaluate_characterize_rank_groups_pooling_functions(
#             single=True, pair=True, aln=self.query_aln_fa_large, assign=self.assignments_large,
#             out_dir=self.out_large_dir, low_mem=True, write_sub_aln=False, write_freq_table=False)
#
#     def evaluate_characterize_rank_groups_mm_pooling_functions(self, single, pair, aln, assign, out_dir, low_mem,
#                                                                write_sub_aln, write_freq_table):
#         unique_dir = os.path.join(out_dir, 'unique_node_data')
#         if not os.path.isdir(unique_dir):
#             os.makedirs(unique_dir)
#         single_alphabet = Gapped(aln.alphabet)
#         single_size, _, single_mapping, single_reverse = build_mapping(alphabet=single_alphabet)
#         if single and not pair:
#             larger_alphabet = MultiPositionAlphabet(alphabet=Gapped(aln.alphabet), size=2)
#             larger_size, _, larger_mapping, larger_reverse = build_mapping(alphabet=larger_alphabet)
#             single_to_larger = self.single_to_pair_dict
#             position_size = 1
#             position_type = 'position'
#             table_type = 'single'
#         elif pair and not single:
#             larger_alphabet = MultiPositionAlphabet(alphabet=Gapped(aln.alphabet), size=4)
#             larger_size, _, larger_mapping, larger_reverse = build_mapping(alphabet=larger_alphabet)
#             single_to_larger = self.single_to_quad_dict
#             position_size = 2
#             position_type = 'pair'
#             table_type = 'pair'
#         else:
#             raise ValueError('Either single or pair permitted, not both or neither.')
#         mm_table = MatchMismatchTable(seq_len=aln.seq_length,num_aln=aln._alignment_to_num(self.single_mapping),
#                                       single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
#                                       single_reverse_mapping=self.single_reverse, larger_alphabet_size=larger_size,
#                                       larger_alphabet_mapping=larger_mapping,
#                                       larger_alphabet_reverse_mapping=larger_reverse,
#                                       single_to_larger_mapping=single_to_larger, pos_size=position_size)
#         mm_table.identify_matches_mismatches()
#         # Build a minimal set of nodes to characterize (the query sequence, its neighbor node, and their parent node)
#         visited = {}
#         to_characterize = []
#         found_query = False
#         for r in sorted(assign.keys(), reverse=True):
#             for g in assign[r]:
#                 node = assign[r][g]['node']
#                 if assign[r][g]['descendants'] and (aln.query_id in [d.name for d in assign[r][g]['descendants']]):
#                     found_query = True
#                     descendants_to_find = set([d.name for d in assign[r][g]['descendants']])
#                     searching = len(descendants_to_find)
#                     for r2 in range(r + 1, max(assign.keys()) + 1):
#                         for g2 in assign[r2]:
#                             if assign[r2][g2]['node'].name in descendants_to_find:
#                                 to_characterize.append((assign[r2][g2]['node'].name, 'component'))
#                                 visited[assign[r2][g2]['node'].name] = {'terminals': assign[r2][g2]['terminals'],
#                                                                         'descendants': assign[r2][g2]['descendants']}
#                                 searching -= 1
#                         if searching == 0:
#                             break
#                 if found_query:
#                     to_characterize.append((node.name, 'inner'))
#                     visited[node.name] = {'terminals': assign[r][g]['terminals'],
#                                           'descendants': assign[r][g]['descendants']}
#                     break
#             if found_query:
#                 break
#         pool_manager = Manager()
#         lock = Lock()
#         frequency_tables = pool_manager.dict()
#         init_characterization_mm_pool(single_size, single_mapping, single_reverse, larger_size, larger_mapping,
#                                       larger_reverse, single_to_larger, mm_table, aln, position_size,
#                                       position_type, table_type, visited, frequency_tables, lock,
#                                       unique_dir, low_mem, write_sub_aln, write_freq_table)
#         for to_char in to_characterize:
#             ret_name = characterization_mm(*to_char)
#             self.assertEqual(ret_name, to_char[0])
#         frequency_tables = dict(frequency_tables)
#         for node_name in visited:
#             sub_aln = aln.generate_sub_alignment(sequence_ids=visited[node_name]['terminals'])
#             if write_sub_aln:
#                 self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}.fa'.format(node_name))))
#             sub_aln_ind = [aln.seq_order.index(s) for s in sub_aln.seq_order]
#             possible_matches_mismatches = ((sub_aln.size ** 2) - sub_aln.size) / 2.0
#             expected_tables = {'match': FrequencyTable(alphabet_size=larger_size, mapping=larger_mapping,
#                                                        reverse_mapping=larger_reverse,
#                                                        seq_len=sub_aln.seq_length, pos_size=position_size)}
#             expected_tables['match'].set_depth(possible_matches_mismatches)
#             expected_tables['mismatch'] = deepcopy(expected_tables['match'])
#             for p in expected_tables['match'].get_positions():
#                 char_dict = {'match': {}, 'mismatch': {}}
#                 for i in range(sub_aln.size):
#                     s1 = sub_aln_ind[i]
#                     for j in range(i + 1, sub_aln.size):
#                         s2 = sub_aln_ind[j]
#                         status, char = mm_table.get_status_and_character(pos=p, seq_ind1=s1, seq_ind2=s2)
#                         if char not in char_dict[status]:
#                             char_dict[status][char] = 0
#                         char_dict[status][char] += 1
#                 for status in char_dict:
#                     for char in char_dict[status]:
#                         expected_tables[status]._increment_count(pos=p, char=char,
#                                                                  amount=char_dict[status][char])
#             expected_tables['match'].finalize_table()
#             expected_tables['mismatch'].finalize_table()
#             for m in ['match', 'mismatch']:
#                 m_table = frequency_tables[node_name][m]
#                 m_table = load_freq_table(freq_table=m_table, low_memory=low_mem)
#                 m_table_mat = m_table.get_table()
#                 expected_m_table_mat = expected_tables[m].get_table()
#                 sparse_diff = m_table_mat - expected_m_table_mat
#                 nonzero_check = sparse_diff.count_nonzero() > 0
#                 if nonzero_check:
#                     print(m_table.get_table().toarray())
#                     print(expected_tables[m].get_table().toarray())
#                     print(sparse_diff)
#                     indices = np.nonzero(sparse_diff)
#                     print(m_table.get_table().toarray()[indices])
#                     print(expected_tables[m].get_table().toarray()[indices])
#                     print(sparse_diff[indices])
#                     print(node_name)
#                     print(sub_aln.alignment)
#                 self.assertFalse(nonzero_check)
#                 if write_freq_table:
#                     expected_table_path = os.path.join(unique_dir, '{}_{}_{}_freq_table.tsv'.format(
#                         node_name, position_type, m))
#                     self.assertTrue(os.path.isfile(expected_table_path), 'Not found: {}'.format(expected_table_path))
#         rmtree(unique_dir)
#
#     def test2c_characterize_rank_groups_mm_initialize_characterization_pool(self):
#         # Test pool initialization function and mappable function (minimal example) for characterization, small aln
#         self.evaluate_characterize_rank_groups_mm_pooling_functions(
#             single=True, pair=False, aln=self.query_aln_fa_small, assign=self.assignments_small,
#             out_dir=self.out_small_dir, low_mem=False, write_sub_aln=True, write_freq_table=True)
#
#     def test2d_characterize_rank_groups_mm_initialize_characterization_pool(self):
#         # Test pool initialization function and mappable function (minimal example) for characterization, small aln
#         self.evaluate_characterize_rank_groups_mm_pooling_functions(
#             single=False, pair=True, aln=self.query_aln_fa_small, assign=self.assignments_small,
#             out_dir=self.out_small_dir, low_mem=False, write_sub_aln=False, write_freq_table=True)
#
#     def test2e_characterize_rank_groups_mm_initialize_characterization_pool(self):
#         # Test pool initialization function and mappable function (minimal example) for characterization, small aln
#         self.evaluate_characterize_rank_groups_mm_pooling_functions(
#             single=True, pair=False, aln=self.query_aln_fa_large, assign=self.assignments_large,
#             out_dir=self.out_large_dir, low_mem=True, write_sub_aln=False, write_freq_table=False)
#
#     def test2f_characterize_rank_groups_mm_initialize_characterization_pool(self):
#         # Test pool initialization function and mappable function (minimal example) for characterization, small aln
#         self.evaluate_characterize_rank_groups_mm_pooling_functions(
#             single=False, pair=True, aln=self.query_aln_fa_large, assign=self.assignments_large,
#             out_dir=self.out_large_dir, low_mem=True, write_sub_aln=False, write_freq_table=False)
#
#     def evaluate_characterize_rank_groups(self, aln, phylo_tree, assign, single, pair, processors, low_mem, write_aln,
#                                           write_freq_table):
#         trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=assign, position_specific=single,
#                       pair_specific=pair, output_dir=os.path.join(self.testing_dir, aln.query_id), low_memory=low_mem)
#         trace.characterize_rank_groups(processes=processors, write_out_sub_aln=write_aln,
#                                        write_out_freq_table=write_freq_table)
#         visited = set()
#         unique_dir = os.path.join(trace.out_dir, 'unique_node_data')
#         if not os.path.isdir(unique_dir):
#             os.makedirs(unique_dir)
#         for rank in trace.assignments:
#             for group in trace.assignments[rank]:
#                 node_name = trace.assignments[rank][group]['node'].name
#                 self.assertTrue(node_name in trace.unique_nodes)
#                 self.assertTrue('single' in trace.unique_nodes[node_name])
#                 self.assertTrue('pair' in trace.unique_nodes[node_name])
#                 if node_name not in visited:
#                     sub_aln = aln.generate_sub_alignment(
#                         sequence_ids=trace.assignments[rank][group]['terminals'])
#                     if write_aln:
#                         self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}.fa'.format(node_name))))
#                     if sub_aln.size < 5:
#                         expected_single_table, expected_pair_table = sub_aln.characterize_positions(
#                             single=single, pair=pair, single_size=self.single_size, single_mapping=self.single_mapping,
#                             single_reverse=self.single_reverse, pair_size=self.pair_size, pair_mapping=self.pair_mapping,
#                             pair_reverse=self.pair_reverse)
#                     else:
#                         expected_single_table, expected_pair_table = sub_aln.characterize_positions2(
#                             single=single, pair=pair, single_letter_size=self.single_size,
#                             single_letter_mapping=self.single_mapping, single_letter_reverse=self.single_reverse,
#                             pair_letter_size=self.pair_size, pair_letter_mapping=self.pair_mapping,
#                             pair_letter_reverse=self.pair_reverse, single_to_pair=self.single_to_pair_arr)
#                     if single:
#                         single_table = trace.unique_nodes[node_name]['single']
#                         if low_mem:
#                             single_table = load_freq_table(freq_table=single_table, low_memory=low_mem)
#                         diff = (single_table.get_table() - expected_single_table.get_table()).toarray()
#                         if diff.any():
#                             print(single_table.get_table().toarray())
#                             print(expected_single_table.get_table().toarray())
#                             print(diff)
#                             indices = np.nonzero(diff)
#                             print(single_table.get_table().toarray()[indices])
#                             print(expected_single_table.get_table().toarray()[indices])
#                             print(diff[indices])
#                             print(node_name)
#                             print(sub_aln.alignment)
#                         self.assertFalse(diff.any())
#                         if write_freq_table:
#                             self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}_position_freq_table.tsv'.format(
#                                 node_name))))
#                     else:
#                         self.assertIsNone(trace.unique_nodes[node_name]['single'])
#                     if pair:
#                         pair_table = trace.unique_nodes[node_name]['pair']
#                         if low_mem:
#                             pair_table = load_freq_table(freq_table=pair_table, low_memory=low_mem)
#                         diff = pair_table.get_table() - expected_pair_table.get_table()
#                         self.assertFalse(diff.toarray().any())
#                         if write_freq_table:
#                             self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}_pair_freq_table.tsv'.format(
#                                 node_name))))
#                     else:
#                         self.assertIsNone(trace.unique_nodes[node_name]['pair'])
#                     visited.add(node_name)
#         rmtree(unique_dir)
#
#     def test3a_characterize_rank_groups(self):
#         # Test characterizing both single and pair positions, small alignment, multi-processed
#         self.evaluate_characterize_rank_groups(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
#                                                assign=self.assignments_custom_small, single=True, pair=True,
#                                                processors=self.max_threads, low_mem=True, write_aln=True,
#                                                write_freq_table=True)
#
#     def test3b_characterize_rank_groups(self):
#         # Test characterizing both single and pair positions, large alignment, single processed
#         self.evaluate_characterize_rank_groups(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
#                                                assign=self.assignments_custom_large, single=True, pair=True,
#                                                processors=1, low_mem=True, write_aln=False, write_freq_table=False)
#
#     def evaluate_characterize_rank_groups_match_mismatch(self, aln, phylo_tree, assign, single, pair, processors,
#                                                          low_mem, write_aln, write_freq_table):
#         if single:
#             pos_size = 1
#             pos_type = 'position'
#             larger_size = self.pair_size
#             larger_mapping = self.pair_mapping
#             larger_reverse = self.pair_reverse
#             single_to_larger = self.single_to_pair_dict
#         else:
#             pos_size = 2
#             pos_type = 'pair'
#             larger_size = self.quad_size
#             larger_mapping = self.quad_mapping
#             larger_reverse = self.quad_reverse
#             single_to_larger = self.single_to_quad_dict
#         mm_table = MatchMismatchTable(seq_len=aln.seq_length, num_aln=aln._alignment_to_num(self.single_mapping),
#                                       single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
#                                       single_reverse_mapping=self.single_reverse, larger_alphabet_size=larger_size,
#                                       larger_alphabet_mapping=larger_mapping,
#                                       larger_alphabet_reverse_mapping=larger_reverse,
#                                       single_to_larger_mapping=single_to_larger, pos_size=pos_size)
#         mm_table.identify_matches_mismatches()
#         trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=assign, position_specific=single,
#                       pair_specific=pair, match_mismatch=True, output_dir=os.path.join(self.testing_dir, aln.query_id),
#                       low_memory=low_mem)
#         trace.characterize_rank_groups(processes=processors, write_out_sub_aln=write_aln,
#                                        write_out_freq_table=write_freq_table)
#         visited = set()
#         unique_dir = os.path.join(trace.out_dir, 'unique_node_data')
#         if not os.path.isdir(unique_dir):
#             os.makedirs(unique_dir)
#         for rank in trace.assignments:
#             for group in trace.assignments[rank]:
#                 node_name = trace.assignments[rank][group]['node'].name
#                 self.assertTrue(node_name in trace.unique_nodes)
#                 self.assertTrue('match' in trace.unique_nodes[node_name])
#                 self.assertTrue('mismatch' in trace.unique_nodes[node_name])
#                 if node_name not in visited:
#                     sub_aln = aln.generate_sub_alignment(
#                         sequence_ids=trace.assignments[rank][group]['terminals'])
#                     if write_aln:
#                         self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}.fa'.format(node_name))))
#                     sub_aln_ind = [aln.seq_order.index(s) for s in sub_aln.seq_order]
#                     possible_matches_mismatches = ((sub_aln.size**2) - sub_aln.size) / 2.0
#                     expected_tables = {'match': FrequencyTable(alphabet_size=larger_size, mapping=larger_mapping,
#                                                                reverse_mapping=larger_reverse,
#                                                                seq_len=sub_aln.seq_length, pos_size=pos_size)}
#                     expected_tables['match'].set_depth(possible_matches_mismatches)
#                     expected_tables['mismatch'] = deepcopy(expected_tables['match'])
#                     for p in expected_tables['match'].get_positions():
#                         char_dict = {'match': {}, 'mismatch': {}}
#                         for i in range(sub_aln.size):
#                             s1 = sub_aln_ind[i]
#                             for j in range(i + 1, sub_aln.size):
#                                 s2 = sub_aln_ind[j]
#                                 status, char = mm_table.get_status_and_character(pos=p, seq_ind1=s1, seq_ind2=s2)
#                                 if char not in char_dict[status]:
#                                     char_dict[status][char] = 0
#                                 char_dict[status][char] += 1
#                         for status in char_dict:
#                             for char in char_dict[status]:
#                                 expected_tables[status]._increment_count(pos=p, char=char,
#                                                                          amount=char_dict[status][char])
#                     expected_tables['match'].finalize_table()
#                     expected_tables['mismatch'].finalize_table()
#                     for m in ['match', 'mismatch']:
#                         m_table = trace.unique_nodes[node_name][m]
#                         m_table = load_freq_table(freq_table=m_table, low_memory=low_mem)
#                         m_table_mat = m_table.get_table()
#                         expected_m_table_mat = expected_tables[m].get_table()
#                         sparse_diff = m_table_mat - expected_m_table_mat
#                         nonzero_check = sparse_diff.count_nonzero() > 0
#                         if nonzero_check:
#                             print(m_table.get_table().toarray())
#                             print(expected_tables[m].get_table().toarray())
#                             print(sparse_diff)
#                             indices = np.nonzero(sparse_diff)
#                             print(m_table.get_table().toarray()[indices])
#                             print(expected_tables[m].get_table().toarray()[indices])
#                             print(sparse_diff[indices])
#                             print(node_name)
#                             print(sub_aln.alignment)
#                         self.assertFalse(nonzero_check)
#                         if write_freq_table:
#                             expected_table_path = os.path.join(unique_dir, '{}_{}_{}_freq_table.tsv'.format(
#                                 node_name, pos_type, m))
#                             self.assertTrue(os.path.isfile(expected_table_path), 'Not found: {}'.format(expected_table_path))
#                     visited.add(node_name)
#         rmtree(unique_dir)
#
#     def test3c_characterize_rank_groups(self):
#         # Test characterizing both single and pair positions, small alignment, single processed
#         self.evaluate_characterize_rank_groups_match_mismatch(
#             aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
#             single=True, pair=False, processors=1, low_mem=False, write_aln=False, write_freq_table=False)
#
#     def test3d_characterize_rank_groups(self):
#         # Test characterizing both single and pair positions, large alignment, single processed
#         self.evaluate_characterize_rank_groups_match_mismatch(
#             aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
#             single=False, pair=True, processors=1, low_mem=False, write_aln=False, write_freq_table=False)
#
#     def test3e_characterize_rank_groups(self):
#         # Test characterizing both single and pair positions, small alignment, single processed
#         self.evaluate_characterize_rank_groups_match_mismatch(
#             aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
#             single=True, pair=False, processors=self.max_threads, low_mem=True, write_aln=True, write_freq_table=True)
#
#     def test3f_characterize_rank_groups(self):
#         # Test characterizing both single and pair positions, large alignment, single processed
#         self.evaluate_characterize_rank_groups_match_mismatch(
#             aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
#             single=False, pair=True, processors=self.max_threads, low_mem=True, write_aln=True, write_freq_table=True)
#
#     def evaluate_trace_pool_functions(self, aln, phylo_tree, assign, single, pair, metric, low_memory,
#                                       out_dir, write_out_aln, write_out_freq_table):
#         unique_dir = os.path.join(out_dir, 'unique_node_data')
#         trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=assign, position_specific=single,
#                       pair_specific=pair, match_mismatch=(('match' in metric) and ('mismatch' in metric)),
#                       output_dir=os.path.join(self.testing_dir, aln.query_id), low_memory=low_memory)
#         trace.characterize_rank_groups(processes=self.max_threads, write_out_sub_aln=write_out_aln,
#                                        write_out_freq_table=write_out_freq_table)
#         if single:
#             pos_size = 1
#         elif pair:
#             pos_size = 2
#         else:
#             raise ValueError('Cannot evaluate if both single and pair are False.')
#         scorer = PositionalScorer(seq_length=aln.seq_length, pos_size=pos_size, metric=metric)
#         init_trace_groups(scorer=scorer, pos_specific=single, pair_specific=pair,
#                           match_mismatch=(('match' in metric) and ('mismatch' in metric)),
#                           u_dict=trace.unique_nodes, low_memory=low_memory, unique_dir=unique_dir)
#         group_dict = {}
#         for node_name in trace.unique_nodes:
#             ret_node_name, ret_components = trace_groups(node_name=node_name)
#             group_dict[ret_node_name] = ret_components
#         self.assertEqual(len(group_dict.keys()), len(trace.unique_nodes.keys()))
#         for node_name in group_dict:
#             self.assertTrue('single_scores' in group_dict[node_name])
#             if single:
#                 if trace.match_mismatch:
#                     single_freq_table = {'match': load_freq_table(freq_table=trace.unique_nodes[node_name]['match'],
#                                                                   low_memory=low_memory),
#                                          'mismatch': load_freq_table(freq_table=trace.unique_nodes[node_name]['mismatch'],
#                                                                      low_memory=low_memory)}
#                 else:
#                     single_freq_table = load_freq_table(freq_table=trace.unique_nodes[node_name]['single'],
#                                                         low_memory=low_memory)
#                 expected_scores = scorer.score_group(freq_table=single_freq_table)
#                 single_scores = load_numpy_array(mat=group_dict[node_name]['single_scores'], low_memory=low_memory)
#                 diff = single_scores - expected_scores
#                 if diff.any():
#                     print(single_scores)
#                     print(expected_scores)
#                     print(diff)
#                     indices = np.nonzero(diff)
#                     print(single_scores[indices])
#                     print(expected_scores[indices])
#                     print(diff[indices])
#                 self.assertTrue(not diff.any())
#             else:
#                 self.assertIsNone(group_dict[node_name]['single_scores'])
#             self.assertTrue('pair_scores' in group_dict[node_name])
#             if pair:
#                 if trace.match_mismatch:
#                     pair_freq_table = {'match': load_freq_table(freq_table=trace.unique_nodes[node_name]['match'],
#                                                                 low_memory=low_memory),
#                                        'mismatch': load_freq_table(freq_table=trace.unique_nodes[node_name]['mismatch'],
#                                                                    low_memory=low_memory)}
#                 else:
#                     pair_freq_table = load_freq_table(freq_table=trace.unique_nodes[node_name]['pair'],
#                                                       low_memory=low_memory)
#                 expected_scores = scorer.score_group(freq_table=pair_freq_table)
#                 pair_scores = load_numpy_array(mat=group_dict[node_name]['pair_scores'], low_memory=low_memory)
#                 diff = pair_scores - expected_scores
#                 if diff.any():
#                     print(pair_scores)
#                     print(expected_scores)
#                     print(diff)
#                     indices = np.nonzero(diff)
#                     print(pair_scores[indices])
#                     print(expected_scores[indices])
#                     print(diff[indices])
#                 self.assertTrue(not diff.any())
#             else:
#                 self.assertIsNone(group_dict[node_name]['pair_scores'])
#             trace.unique_nodes[node_name].update(group_dict[node_name])
#         rank_dict = {}
#         init_trace_ranks(scorer=scorer, pos_specific=single, pair_specific=pair, a_dict=assign,
#                          u_dict=trace.unique_nodes, low_memory=low_memory, unique_dir=unique_dir)
#         for rank in assign:
#             ret_rank, ret_components = trace_ranks(rank=rank)
#             rank_dict[ret_rank] = ret_components
#         for rank in assign.keys():
#             group_scores = []
#             for group in sorted(assign[rank].keys(), reverse=True):
#                 node_name = assign[rank][group]['node'].name
#                 if single:
#                     single_scores = load_numpy_array(mat=trace.unique_nodes[node_name]['single_scores'],
#                                                      low_memory=low_memory)
#                     group_scores.append(single_scores)
#                 elif pair:
#                     pair_scores = load_numpy_array(mat=trace.unique_nodes[node_name]['pair_scores'],
#                                                    low_memory=low_memory)
#                     group_scores.append(pair_scores)
#                 else:
#                     raise ValueError('Cannot evaluate if both single and pair are False.')
#             group_scores = np.stack(group_scores, axis=0)
#             expected_rank_scores = np.sum(group_scores, axis=0)
#             if scorer.metric_type == 'integer':
#                 expected_rank_scores = (expected_rank_scores > 0) * 1.0
#             else:
#                 expected_rank_scores = (1.0 / rank) * expected_rank_scores
#             self.assertTrue(rank in rank_dict)
#             self.assertTrue('single_ranks' in rank_dict[rank])
#             if single:
#                 rank_scores = load_numpy_array(mat=rank_dict[rank]['single_ranks'], low_memory=low_memory)
#                 diff = np.abs(rank_scores - expected_rank_scores)
#                 not_passing = diff > 1E-15
#                 if not_passing.any():
#                     print(rank)
#                     print(rank_scores)
#                     print(expected_rank_scores)
#                     print(diff)
#                     indices = np.nonzero(not_passing)
#                     print(rank_scores[indices])
#                     print(expected_rank_scores[indices])
#                     print(diff[indices])
#                 self.assertTrue(not not_passing.any())
#             else:
#                 self.assertIsNone(rank_dict[rank]['single_ranks'])
#             self.assertTrue(rank in rank_dict)
#             self.assertTrue('pair_ranks' in rank_dict[rank])
#             if pair:
#                 expected_rank_scores = np.triu(expected_rank_scores, k=1)
#                 rank_scores = load_numpy_array(mat=rank_dict[rank]['pair_ranks'], low_memory=low_memory)
#                 diff = np.abs(rank_scores - expected_rank_scores)
#                 not_passing = diff > 1E-12
#                 not_passing[expected_rank_scores > 1E14] = diff[expected_rank_scores > 1E14] > 5
#                 if not_passing.any():
#                     print(rank_scores)
#                     print(expected_rank_scores)
#                     print(diff)
#                     indices = np.nonzero(not_passing)
#                     print(rank_scores[indices])
#                     print(expected_rank_scores[indices])
#                     print(diff[indices])
#                 self.assertTrue(not not_passing.any())
#             else:
#                 self.assertIsNone(rank_dict[rank]['pair_ranks'])
#
#     def test4a_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the small alignment, single positions,
#         # and the identity metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
#             single=True, pair=False, metric='identity', low_memory=True, out_dir=self.out_small_dir,
#             write_out_aln=False, write_out_freq_table=False)
#
#     def test4b_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the small alignment, paired positions,
#         # and the identity metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
#             single=False, pair=True, metric='identity', low_memory=True, out_dir=self.out_small_dir,
#             write_out_aln=False, write_out_freq_table=False)
#
#     def test4c_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, single positions,
#         # and identity metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
#             single=True, pair=False, metric='identity', low_memory=True, out_dir=self.out_large_dir,
#             write_out_aln=False, write_out_freq_table=False)
#
#     def test4d_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
#         # and identity metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
#             single=False, pair=True, metric='identity', low_memory=True, out_dir=self.out_large_dir,
#             write_out_aln=False, write_out_freq_table=False)
#
#     def test4e_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the small alignment, single positions,
#         # and the plain entropy metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
#             single=True, pair=False, metric='plain_entropy', low_memory=True, out_dir=self.out_small_dir,
#             write_out_aln=False, write_out_freq_table=False)
#
#     def test4f_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the small alignment, paired positions,
#         # and the plain entropy metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
#             single=False, pair=True, metric='plain_entropy', low_memory=True, out_dir=self.out_small_dir,
#             write_out_aln=False, write_out_freq_table=False)
#
#     def test4g_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, single positions,
#         # and plain entropy metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
#             single=True, pair=False, metric='plain_entropy', low_memory=True, out_dir=self.out_large_dir,
#             write_out_aln=False, write_out_freq_table=False)
#
#     def test4h_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
#         # and plain entropy metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
#             single=False, pair=True, metric='plain_entropy', low_memory=True, out_dir=self.out_large_dir,
#             write_out_aln=False, write_out_freq_table=False)
#
#     def test4i_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the small alignment, paired positions,
#         # and the mutual information metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
#             single=False, pair=True, metric='mutual_information', low_memory=True, out_dir=self.out_small_dir,
#             write_out_aln=False, write_out_freq_table=False)
#
#     def test4j_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
#         # and mutual information metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
#             single=False, pair=True, metric='mutual_information', low_memory=True, out_dir=self.out_large_dir,
#             write_out_aln=False, write_out_freq_table=False)
#
#     def test4k_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the small alignment, paired positions,
#         # and the normalized mutual information metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
#             single=False, pair=True, metric='normalized_mutual_information', low_memory=True,
#             out_dir=self.out_small_dir, write_out_aln=False, write_out_freq_table=False)
#
#     def test4l_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
#         # and normalized mutual information metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
#             single=False, pair=True, metric='normalized_mutual_information', low_memory=True,
#             out_dir=self.out_large_dir, write_out_aln=False, write_out_freq_table=False)
#
#     def test4m_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the small alignment, paired positions,
#         # and the average product corrected mutual information (MIp) metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
#             single=False, pair=True, metric='average_product_corrected_mutual_information', low_memory=True,
#             out_dir=self.out_small_dir, write_out_aln=False, write_out_freq_table=False)
#
#     def test4n_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
#         # and average product corrected mutual information (MIp) metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
#             single=False, pair=True, metric='average_product_corrected_mutual_information', low_memory=True,
#             out_dir=self.out_large_dir, write_out_aln=False, write_out_freq_table=False)
#
#     def test4o_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
#         # and average product corrected mutual information (MIp) metric (all ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
#             single=False, pair=True, metric='filtered_average_product_corrected_mutual_information', low_memory=True,
#             out_dir=self.out_small_dir, write_out_aln=False, write_out_freq_table=False)
#
#     def test4p_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
#         # and average product corrected mutual information (MIp) metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
#             single=False, pair=True, metric='filtered_average_product_corrected_mutual_information', low_memory=True,
#             out_dir=self.out_large_dir, write_out_aln=False, write_out_freq_table=False)
#
#     def test4q_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
#         # and average product corrected mutual information (MIp) metric (all ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
#             single=False, pair=True, metric='match_mismatch_entropy_ratio', low_memory=True,
#             out_dir=self.out_small_dir, write_out_aln=False, write_out_freq_table=False)
#
#     def test4r_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
#         # and average product corrected mutual information (MIp) metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
#             single=False, pair=True, metric='match_mismatch_entropy_ratio', low_memory=True,
#             out_dir=self.out_large_dir, write_out_aln=False, write_out_freq_table=False)
#
#     def test4t_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
#         # and average product corrected mutual information (MIp) metric (all ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
#             single=False, pair=True, metric='match_mismatch_entropy_angle', low_memory=True,
#             out_dir=self.out_small_dir, write_out_aln=False, write_out_freq_table=False)
#
#     def test4u_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
#         # and average product corrected mutual information (MIp) metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
#             single=False, pair=True, metric='match_mismatch_entropy_angle', low_memory=True,
#             out_dir=self.out_large_dir, write_out_aln=False, write_out_freq_table=False)
#
#     def test4v_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
#         # and average product corrected mutual information (MIp) metric (all ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
#             single=False, pair=True, metric='match_diversity_mismatch_entropy_ratio', low_memory=True,
#             out_dir=self.out_small_dir, write_out_aln=False, write_out_freq_table=False)
#
#     def test4w_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
#         # and average product corrected mutual information (MIp) metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
#             single=False, pair=True, metric='match_diversity_mismatch_entropy_ratio', low_memory=True,
#             out_dir=self.out_large_dir, write_out_aln=False, write_out_freq_table=False)
#
#     def test4x_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
#         # and average product corrected mutual information (MIp) metric (all ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
#             single=False, pair=True, metric='match_diversity_mismatch_entropy_angle', low_memory=True,
#             out_dir=self.out_small_dir, write_out_aln=False, write_out_freq_table=False)
#
#     def test4y_trace_pool_functions(self):
#         # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
#         # and average product corrected mutual information (MIp) metric (custom ranks)
#         self.evaluate_trace_pool_functions(
#             aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
#             single=False, pair=True, metric='match_diversity_mismatch_entropy_angle', low_memory=True,
#             out_dir=self.out_large_dir, write_out_aln=False, write_out_freq_table=False)
#
#     def evaluate_trace(self, aln, phylo_tree, assignments, single, pair, metric, num_proc, out_dir,
#                        match_mismatch=False, gap_correction=None, low_mem=True):
#         if single:
#             pos_size = 1
#             expected_ranks = np.ones(aln.seq_length)
#         elif pair:
#             pos_size = 2
#             expected_ranks = np.ones((aln.seq_length, aln.seq_length))
#         else:
#             pos_size = None
#             expected_ranks = None
#         trace_obj = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=assignments,
#                           position_specific=single, pair_specific=pair, match_mismatch=match_mismatch,
#                           low_memory=low_mem, output_dir=out_dir)
#         trace_obj.characterize_rank_groups(processes=num_proc, write_out_sub_aln=False, write_out_freq_table=False)
#         scorer = PositionalScorer(seq_length=aln.seq_length, pos_size=pos_size, metric=metric)
#         rank_array, score_array, coverage_array = trace_obj.trace(scorer=scorer, processes=num_proc,
#                                                                   gap_correction=gap_correction)
#         # Simple check against ranking and coverage is to find the min and max values
#         if pos_size == 2:
#             self.assertFalse(np.tril(rank_array).any())
#             self.assertFalse(np.tril(score_array).any())
#             self.assertFalse(np.tril(coverage_array).any())
#         if scorer.rank_type == 'min':
#             if pos_size == 1:
#                 min_score = np.min(score_array)
#                 min_rank = np.min(rank_array)
#                 max_coverage = np.max(coverage_array)
#                 max_score = np.max(score_array)
#                 max_rank = np.max(rank_array)
#                 min_coverage = np.min(coverage_array)
#             else:
#                 min_score = np.min(score_array[np.triu_indices(aln.seq_length, k=1)])
#                 min_rank = np.min(rank_array[np.triu_indices(aln.seq_length, k=1)])
#                 max_coverage = np.max(coverage_array[np.triu_indices(aln.seq_length, k=1)])
#                 max_score = np.max(score_array[np.triu_indices(aln.seq_length, k=1)])
#                 max_rank = np.max(rank_array[np.triu_indices(aln.seq_length, k=1)])
#                 min_coverage = np.min(coverage_array[np.triu_indices(aln.seq_length, k=1)])
#             min_mask = (score_array == min_score) * 1
#             rank_mask = (rank_array == min_rank) * 1
#             diff_min_ranks = min_mask - rank_mask
#             if diff_min_ranks.any():
#                 print(min_score)
#                 print(min_rank)
#                 print(np.sum(min_mask))
#                 print(np.sum(rank_mask))
#             self.assertFalse(diff_min_ranks.any())
#             cov_mask = coverage_array == min_coverage
#             diff_min_cov = min_mask - cov_mask
#             if diff_min_cov.any():
#                 print(min_score)
#                 print(min_coverage)
#                 print(np.sum(min_mask))
#                 print(np.sum(cov_mask))
#             self.assertFalse(diff_min_cov.any())
#             max_mask = score_array == max_score
#             rank_mask2 = rank_array == max_rank
#             diff_max_ranks = max_mask ^ rank_mask2
#             if diff_max_ranks.any():
#                 print(max_score)
#                 print(max_rank)
#                 print(np.sum(max_mask))
#                 print(np.sum(rank_mask2))
#             self.assertFalse(diff_max_ranks.any())
#             cov_mask2 = coverage_array == max_coverage
#             diff_max_cov = max_mask ^ cov_mask2
#             if diff_max_cov.any():
#                 print(max_score)
#                 print(max_coverage)
#                 print(np.sum(max_mask))
#                 print(np.sum(cov_mask2))
#             self.assertFalse(diff_min_cov.any())
#         else:
#             if pos_size == 1:
#                 max_score = np.max(score_array)
#                 max_rank = np.max(rank_array)
#                 max_coverage = np.max(coverage_array)
#                 min_score = np.min(score_array)
#                 min_rank = np.min(rank_array)
#                 min_coverage = np.min(coverage_array)
#                 min_mask = score_array == min_score
#                 rank_mask = rank_array == max_rank
#                 cov_mask = coverage_array == max_coverage
#                 max_mask = score_array == max_score
#                 rank_mask2 = rank_array == min_rank
#                 cov_mask2 = coverage_array == min_coverage
#             else:
#                 max_score = np.max(score_array[np.triu_indices(aln.seq_length, k=1)])
#                 max_rank = np.max(rank_array[np.triu_indices(aln.seq_length, k=1)])
#                 max_coverage = np.max(coverage_array[np.triu_indices(aln.seq_length, k=1)])
#                 min_score = np.min(score_array[np.triu_indices(aln.seq_length, k=1)])
#                 min_rank = np.min(rank_array[np.triu_indices(aln.seq_length, k=1)])
#                 min_coverage = np.min(coverage_array[np.triu_indices(aln.seq_length, k=1)])
#                 min_mask = np.triu(score_array == min_score, k=1)
#                 rank_mask = np.triu(rank_array == max_rank, k=1)
#                 cov_mask = np.triu(coverage_array == max_coverage, k=1)
#                 max_mask = np.triu(score_array == max_score, k=1)
#                 rank_mask2 = np.triu(rank_array == min_rank, k=1)
#                 cov_mask2 = np.triu(coverage_array == min_coverage, k=1)
#             diff_min_ranks = min_mask ^ rank_mask
#             if diff_min_ranks.any():
#                 print(min_score)
#                 print(min_rank)
#                 print(np.sum(min_mask))
#                 print(np.sum(rank_mask))
#             self.assertFalse(diff_min_ranks.any())
#             diff_min_cov = min_mask ^ cov_mask
#             if diff_min_cov.any():
#                 print(min_score)
#                 print(max_coverage)
#                 print(np.sum(min_mask))
#                 print(np.sum(cov_mask))
#             self.assertFalse(diff_min_cov.any())
#             diff_max_ranks = max_mask ^ rank_mask2
#             if diff_max_ranks.any():
#                 print(max_score)
#                 print(max_rank)
#                 print(np.sum(max_mask))
#                 print(np.sum(rank_mask2))
#             self.assertFalse(diff_max_ranks.any())
#             diff_max_cov = max_mask ^ cov_mask2
#             if diff_max_cov.any():
#                 print(max_score)
#                 print(max_coverage)
#                 print(np.sum(max_mask))
#                 print(np.sum(cov_mask2))
#             self.assertFalse(diff_min_cov.any())
#         unique_scores = {}
#         for rank in sorted(assignments.keys(), reverse=True):
#             group_scores = []
#             for group in assignments[rank].keys():
#                 node_name = assignments[rank][group]['node'].name
#                 if node_name not in unique_scores:
#                     if match_mismatch:
#                         mm_dict = {'match': load_freq_table(trace_obj.unique_nodes[node_name]['match'],
#                                                             low_memory=low_mem),
#                                    'mismatch': load_freq_table(trace_obj.unique_nodes[node_name]['mismatch'],
#                                                                low_memory=low_mem)}
#                         group_score = scorer.score_group(mm_dict)
#                     elif pair:
#                         group_score = scorer.score_group(load_freq_table(trace_obj.unique_nodes[node_name]['pair'],
#                                                                          low_memory=low_mem))
#                     elif single:
#                         group_score = scorer.score_group(load_freq_table(trace_obj.unique_nodes[node_name]['single'],
#                                                                          low_memory=low_mem))
#                     else:
#                         raise ValueError('Either pair or single must be true for this test.')
#                     unique_scores[node_name] = group_score
#                 else:
#                     group_score = unique_scores[node_name]
#                 group_scores.append(group_score)
#             group_scores = np.stack(group_scores, axis=0)
#             if metric == 'identity':
#                 weight = 1.0
#             else:
#                 weight = 1.0 / rank
#             rank_scores = weight * np.sum(group_scores, axis=0)
#             if metric == 'identity':
#                 rank_scores = rank_scores > 0 * 1
#             if single:
#                 curr_rank = load_numpy_array(trace_obj.rank_scores[rank]['single_ranks'], low_memory=low_mem)
#             elif pair:
#                 curr_rank = load_numpy_array(trace_obj.rank_scores[rank]['pair_ranks'], low_memory=low_mem)
#             else:
#                 raise ValueError('Either pair or single must be true for this test.')
#             diff_rank = curr_rank - rank_scores
#             not_passing_rank = diff_rank > 1E-12
#             if not_passing_rank.any():
#                 print(curr_rank)
#                 print(rank_scores)
#                 print(diff_rank)
#                 indices_rank = np.nonzero(not_passing_rank)
#                 print(curr_rank[indices_rank])
#                 print(rank_scores[indices_rank])
#                 print(diff_rank[indices_rank])
#             self.assertFalse(not_passing_rank.any())
#             expected_ranks += rank_scores
#         diff_ranks = score_array - expected_ranks
#         not_passing = diff_ranks > 1E-12
#         not_passing[expected_ranks > 1E-14] = diff_ranks[expected_ranks > 1E-14] > 10
#         if not_passing.any():
#             print(score_array)
#             print(expected_ranks)
#             print(diff_ranks)
#             indices = np.nonzero(not_passing)
#             print(score_array[indices])
#             print(expected_ranks[indices])
#             print(diff_ranks[indices])
#         self.assertFalse(not_passing.any())
#
#     def test5a_trace(self):
#         # Perform identity trace on single positions only for the small alignment (custom ranks)
#         # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
#         self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
#                             assignments=self.assignments_custom_small, single=True, pair=False, metric='identity',
#                             low_mem=True, num_proc=self.max_threads, out_dir=self.out_small_dir)
#
#     def test5b_trace(self):
#         # Perform identity trace on single positions only for the large alignment (custom ranks)
#         # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
#         self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
#                             assignments=self.assignments_custom_large, single=True, pair=False, metric='identity',
#                             low_mem=True, num_proc=self.max_threads, out_dir=self.out_large_dir)
#
#     def evaluate_integer_et_comparison(self, p_id, msf_aln, fa_aln, low_mem, out_dir):
#         if os.path.isdir(out_dir):
#             rmtree(out_dir)
#         os.mkdir(out_dir)
#         et_mip_obj = ETMIPWrapper(query=p_id, aln_file=fa_aln.file_name, out_dir=out_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         trace_small = Trace(alignment=fa_aln, phylo_tree=et_mip_obj.tree,
#                             group_assignments=et_mip_obj.rank_group_assignments, position_specific=True,
#                             pair_specific=False, output_dir=out_dir, low_memory=low_mem)
#         trace_small.characterize_rank_groups(processes=self.max_threads, write_out_sub_aln=False,
#                                              write_out_freq_table=False)
#         scorer = PositionalScorer(seq_length=fa_aln.seq_length, pos_size=1, metric='identity')
#         rank_ids, score_ids, coverage_ids = trace_small.trace(scorer=scorer, gap_correction=None,
#                                                               processes=self.max_threads)
#         diff_ranks = score_ids - et_mip_obj.scores
#         if diff_ranks.any():
#             print(score_ids)
#             print(et_mip_obj.scores)
#             print(diff_ranks)
#             indices = np.nonzero(diff_ranks)
#             print(score_ids[indices])
#             print(et_mip_obj.scores[indices])
#             print(diff_ranks[indices])
#         self.assertFalse(diff_ranks.any())
#         diff_coverage = coverage_ids - et_mip_obj.coverages
#         not_passing = np.abs(diff_coverage) > 1e-2
#         if not_passing.any():
#             print(coverage_ids)
#             print(et_mip_obj.coverage)
#             print(diff_coverage)
#             indices = np.nonzero(diff_coverage)
#             print(coverage_ids[indices])
#             print(et_mip_obj.coverage[indices])
#             print(diff_coverage[indices])
#         self.assertFalse(not_passing.any())
#         rounded_coverages = np.round(coverage_ids, decimals=3)
#         diff_coverages2 = rounded_coverages - et_mip_obj.coverages
#         not_passing2 = diff_coverages2 > 1E-15
#         if not_passing2.any():
#             print(rounded_coverages)
#             print(et_mip_obj.coverage)
#             print(diff_coverages2)
#             indices = np.nonzero(not_passing2)
#             print(rounded_coverages[indices])
#             print(et_mip_obj.coverage[indices])
#             print(diff_coverages2[indices])
#         self.assertFalse(not_passing2.any())
#
#     def test5c_trace(self):
#         # Compare the results of identity trace over single positions between this implementation and the WETC
#         # implementation for the small alignment.
#         self.evaluate_integer_et_comparison(p_id=self.small_structure_id, msf_aln=self.query_aln_msf_small,
#                                             fa_aln=self.query_aln_fa_small, low_mem=False, out_dir=self.out_small_dir)
#
#     def test5d_trace(self):
#         # Compare the results of identity trace over single positions between this implementation and the WETC
#         # implementation for the large alignment.
#         self.evaluate_integer_et_comparison(p_id=self.large_structure_id, msf_aln=self.query_aln_msf_large,
#                                             fa_aln=self.query_aln_fa_large, low_mem=True, out_dir=self.out_large_dir)
#
#     def test5e_trace(self):
#         # Perform identity trace on pairs of positions only for the small alignment (custom ranks)
#         # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
#         self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
#                             assignments=self.assignments_custom_small, single=False, pair=True, metric='identity',
#                             low_mem=True, num_proc=self.max_threads, out_dir=self.out_small_dir)
#
#     def test5f_trace(self):
#         # Perform identity trace on pairs of positions only for the large alignment (custom ranks)
#         # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
#         self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
#                             assignments=self.assignments_custom_large, single=False, pair=True, metric='identity',
#                             low_mem=True, num_proc=self.max_threads, out_dir=self.out_large_dir)
#
#     def test5g_trace(self):
#         # Perform plain entropy trace on single positions only for the small alignment (custom ranks)
#         # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
#         self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
#                             assignments=self.assignments_custom_small, single=True, pair=False, metric='plain_entropy',
#                             low_mem=True, num_proc=self.max_threads, out_dir=self.out_small_dir)
#
#     def test5h_trace(self):
#         # Perform plain entropy trace on single positions only for the large alignment (custom ranks)
#         # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
#         self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
#                             assignments=self.assignments_custom_large, single=True, pair=False, metric='plain_entropy',
#                             low_mem=True, num_proc=self.max_threads, out_dir=self.out_large_dir)
#
#     def evaluate_real_value_et_comparison(self, p_id, msf_aln, fa_aln, low_mem, out_dir):
#         if os.path.isdir(out_dir):
#             rmtree(out_dir)
#         os.mkdir(out_dir)
#         et_mip_obj = ETMIPWrapper(query=p_id, aln_file=fa_aln.file_name, out_dir=out_dir)
#         et_mip_obj.calculate_scores(method='rvET', delete_files=False)
#         trace_small = Trace(alignment=fa_aln, phylo_tree=et_mip_obj.tree,
#                             group_assignments=et_mip_obj.rank_group_assignments, position_specific=True,
#                             pair_specific=False, output_dir=out_dir, low_memory=low_mem)
#         trace_small.characterize_rank_groups(processes=self.max_threads, write_out_sub_aln=False,
#                                              write_out_freq_table=False)
#         scorer = PositionalScorer(seq_length=fa_aln.seq_length, pos_size=1, metric='plain_entropy')
#         rank_entropies, score_entropies, coverage_entropies = trace_small.trace(scorer=scorer, gap_correction=0.6,
#                                                                                 processes=self.max_threads)
#         diff_ranks = score_entropies - et_mip_obj.scores
#         not_passing = np.abs(diff_ranks) > 1e-2
#         if not_passing.any():
#             print(score_entropies)
#             print(et_mip_obj.scores)
#             print(diff_ranks)
#             indices = np.nonzero(diff_ranks)
#             print(score_entropies[indices])
#             print(et_mip_obj.scores[indices])
#             print(diff_ranks[indices])
#         self.assertFalse(not_passing.any())
#         rounded_entropies = np.round(score_entropies, decimals=2)
#         diff_ranks2 = rounded_entropies - et_mip_obj.scores
#         if diff_ranks2.any():
#             print(rounded_entropies)
#             print(et_mip_obj.scores)
#             print(diff_ranks2)
#             indices = np.nonzero(diff_ranks2)
#             print(rounded_entropies[indices])
#             print(et_mip_obj.scores[indices])
#             print(diff_ranks2[indices])
#         self.assertFalse(diff_ranks2.any())
#         diff_coverage = coverage_entropies - et_mip_obj.coverages
#         not_passing = np.abs(diff_coverage) > 1e-2
#         if not_passing.any():
#             print(coverage_entropies)
#             print(et_mip_obj.coverage)
#             print(diff_coverage)
#             indices = np.nonzero(diff_coverage)
#             print(coverage_entropies[indices])
#             print(et_mip_obj.coverage[indices])
#             print(diff_coverage[indices])
#         self.assertFalse(not_passing.any())
#         rounded_coverages = np.round(coverage_entropies, decimals=3)
#         diff_coverages2 = rounded_coverages - et_mip_obj.coverages
#         not_passing2 = diff_coverages2 > 1E-15
#         if not_passing2.any():
#             print(rounded_coverages)
#             print(et_mip_obj.coverage)
#             print(diff_coverages2)
#             indices = np.nonzero(not_passing2)
#             print(rounded_coverages[indices])
#             print(et_mip_obj.coverage[indices])
#             print(diff_coverages2[indices])
#         self.assertFalse(not_passing2.any())
#
#     def test5i_trace(self):
#         # Compare the results of plain entropy trace over single positions between this implementation and the WETC
#         # implementation for the small alignment.
#         self.evaluate_real_value_et_comparison(p_id=self.small_structure_id, msf_aln=self.query_aln_msf_small,
#                                                fa_aln=self.query_aln_fa_small, low_mem=False,
#                                                out_dir=self.out_small_dir)
#
#     def test5j_trace(self):
#         # Compare the results of identity trace over single positions between this implementation and the WETC
#         # implementation for the large alignment.
#         self.evaluate_real_value_et_comparison(p_id=self.large_structure_id, msf_aln=self.query_aln_msf_large,
#                                                fa_aln=self.query_aln_fa_large, low_mem=True,
#                                                out_dir=self.out_large_dir)
#
#     def test5k_trace(self):
#         # Perform plain entropy trace on pairs of positions only for the small alignment (custom ranks)
#         # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
#         self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
#                             assignments=self.assignments_custom_small, single=False, pair=True, metric='plain_entropy',
#                             low_mem=True, num_proc=self.max_threads, out_dir=self.out_small_dir)
#
#     def test5l_trace(self):
#         # Perform plain entropy trace on pairs of positions only for the large alignment (custom ranks)
#         # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
#         self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
#                             assignments=self.assignments_custom_large, single=False, pair=True, metric='plain_entropy',
#                             low_mem=True, num_proc=self.max_threads, out_dir=self.out_large_dir)
#
#     def test5m_trace(self):
#         # Perform mutual information trace on pairs of positions only for the small alignment (custom ranks)
#         # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
#         self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
#                             assignments=self.assignments_custom_small, single=False, pair=True,
#                             metric='mutual_information', low_mem=True, num_proc=self.max_threads,
#                             out_dir=self.out_small_dir)
#
#     def test5n_trace(self):
#         # Perform mutual information trace on pairs of positions only for the large alignment (custom ranks)
#         # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
#         self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
#                             assignments=self.assignments_custom_large, single=False, pair=True,
#                             metric='mutual_information', low_mem=True, num_proc=self.max_threads,
#                             out_dir=self.out_large_dir)
#
#     def test5o_trace(self):
#         # Perform normalize mutual information trace on pairs of positions only for the small alignment (custom ranks)
#         # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
#         self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
#                             assignments=self.assignments_custom_small, single=False, pair=True,
#                             metric='normalized_mutual_information', low_mem=True, num_proc=self.max_threads,
#                             out_dir=self.out_small_dir)
#
#     def test5p_trace(self):
#         # Perform normalize mutual information trace on pairs of positions only for the large alignment (custom ranks)
#         # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
#         self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
#                             assignments=self.assignments_custom_large, single=False, pair=True,
#                             metric='normalized_mutual_information', low_mem=True, num_proc=self.max_threads,
#                             out_dir=self.out_large_dir)
#
#     def test5q_trace(self):
#         # Perform average product corrected mutual information (MIp) trace on pairs of positions only for the small
#         # alignment (custom ranks). Assume scoring happens correctly since it has been tested above and ensure that
#         # expected ranks are achieved.
#         self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
#                             assignments=self.assignments_custom_small, single=False, pair=True,
#                             metric='average_product_corrected_mutual_information',
#                             low_mem=True, num_proc=self.max_threads, out_dir=self.out_small_dir)
#
#     def test5r_trace(self):
#         # Perform average product corrected mutual information (MIp) trace on pairs of positions only for the large
#         # alignment (custom ranks). Assume scoring happens correctly since it has been tested above and ensure that
#         # expected ranks are achieved.
#         self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
#                             assignments=self.assignments_custom_large, single=False, pair=True,
#                             metric='average_product_corrected_mutual_information',
#                             low_mem=True, num_proc=self.max_threads, out_dir=self.out_large_dir)
#
#     def test5s_trace(self):
#         # Perform average product corrected mutual information (MIp) trace on pairs of positions only for the small
#         # alignment (custom ranks). Assume scoring happens correctly since it has been tested above and ensure that
#         # expected ranks are achieved.
#         self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
#                             assignments=self.assignments_custom_small, single=False, pair=True,
#                             metric='filtered_average_product_corrected_mutual_information',
#                             low_mem=True, num_proc=self.max_threads, out_dir=self.out_small_dir)
#
#     def test5t_trace(self):
#         # Perform average product corrected mutual information (MIp) trace on pairs of positions only for the large
#         # alignment (custom ranks). Assume scoring happens correctly since it has been tested above and ensure that
#         # expected ranks are achieved.
#         self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
#                             assignments=self.assignments_custom_large, single=False, pair=True,
#                             metric='filtered_average_product_corrected_mutual_information',
#                             low_mem=True, num_proc=self.max_threads, out_dir=self.out_large_dir)
#
#     def evaluate_mip_et_comparison(self, p_id, fa_aln, low_mem, out_dir):
#         rmtree(out_dir, ignore_errors=True)
#         os.makedirs(out_dir)
#         filtered_fa_fn = os.path.join(out_dir, '{}_filtered_aln.fa'.format(p_id))
#         if os.path.isfile(filtered_fa_fn):
#             char_filtered_fa_aln = SeqAlignment(file_name=filtered_fa_fn, query_id=p_id)
#             char_filtered_fa_aln.import_alignment()
#         else:
#             curr_fa_aln = SeqAlignment(file_name=fa_aln.file_name, query_id=p_id)
#             curr_fa_aln.import_alignment()
#             curr_fa_aln.alphabet = Gapped(IUPACProtein())
#             char_filtered_fa_aln = curr_fa_aln.remove_bad_sequences()
#             char_filtered_fa_aln.write_out_alignment(file_name=filtered_fa_fn)
#             char_filtered_fa_aln.file_name = filtered_fa_fn
#         et_mip_obj = ETMIPWrapper(query=p_id, aln_file=filtered_fa_fn, out_dir=out_dir)
#         et_mip_obj.convert_alignment()
#         et_mip_obj.calculate_scores(method='ET-MIp', delete_files=False)
#         gap_filtered_aln = char_filtered_fa_aln.remove_gaps()
#         gap_filtered_aln.alphabet = FullIUPACProtein()
#         trace_mip = Trace(alignment=gap_filtered_aln, phylo_tree=et_mip_obj.tree,
#                           group_assignments=et_mip_obj.rank_group_assignments, position_specific=False,
#                           pair_specific=True, output_dir=out_dir, low_memory=low_mem)
#         trace_mip.characterize_rank_groups(processes=self.max_threads, write_out_sub_aln=False,
#                                            write_out_freq_table=False)
#         scorer_mip = PositionalScorer(seq_length=gap_filtered_aln.seq_length, pos_size=2,
#                                       metric='filtered_average_product_corrected_mutual_information')
#         rank_mips, score_mips, coverage_mips = trace_mip.trace(scorer=scorer_mip, gap_correction=None,
#                                                                processes=self.max_threads)
#         diff_ranks = score_mips - et_mip_obj.scores
#         not_passing = np.abs(diff_ranks) > 1e-3
#         if not_passing.any():
#             print(score_mips)
#             print(et_mip_obj.scores)
#             print(diff_ranks)
#             indices = np.nonzero(not_passing)
#             print(score_mips[indices])
#             print(et_mip_obj.scores[indices])
#             print(diff_ranks[indices])
#             print(score_mips[indices][0])
#             print(et_mip_obj.scores[indices][0])
#             print(diff_ranks[indices][0])
#         self.assertFalse(not_passing.any())
#         rounded_scores = np.round(score_mips, decimals=3)
#         diff_ranks2 = rounded_scores - et_mip_obj.scores
#         not_passing_rounded = np.abs(diff_ranks2) > 1e-15
#         if not_passing_rounded.any():
#             print(rounded_scores)
#             print(et_mip_obj.scores)
#             print(diff_ranks2)
#             indices = np.nonzero(not_passing_rounded)
#             print(rounded_scores[indices])
#             print(et_mip_obj.scores[indices])
#             print(diff_ranks2[indices])
#         self.assertFalse(not_passing_rounded.any())
#         diff_coverages = coverage_mips - et_mip_obj.coverages
#         not_passing = np.abs(diff_coverages) > 1E-3
#         if not_passing.any():
#             print(coverage_mips)
#             print(et_mip_obj.coverages)
#             print(diff_coverages)
#             indices = np.nonzero(not_passing)
#             for i in range(len(indices[0])):
#                 print(indices[0][i], indices[1][i], et_mip_obj.coverages[indices[0][i], indices[1][i]],
#                       coverage_mips[indices[0][i], indices[1][i]], diff_coverages[indices[0][i], indices[1][i]],
#                       1e-2, np.abs(diff_coverages[indices[0][i], indices[1][i]]) > 1e-2)
#             print(score_mips[indices])
#             print(rank_mips[indices])
#             print(np.sum(not_passing))
#             print(np.nonzero(not_passing))
#             self.assertLessEqual(np.sum(not_passing), np.ceil(0.01 * np.sum(range(fa_aln.seq_length - 1))))
#         else:
#             self.assertFalse(not_passing.any())
#         rmtree(out_dir)
#
#     def test5u_trace(self):
#         # Compare the results of average product corrected mutual information over pairs of positions between this
#         # implementation and the WETC implementation for the small alignment.
#         self.evaluate_mip_et_comparison(p_id=self.small_structure_id, fa_aln=self.query_aln_fa_small, low_mem=False,
#                                         out_dir=self.out_small_dir)
#
#     def test5v_trace(self):
#         # Compare the results of average product corrected mutual information over pairs of positions between this
#         # implementation and the WETC implementation for the large alignment.
#         self.evaluate_mip_et_comparison(p_id=self.large_structure_id, fa_aln=self.query_aln_fa_large, low_mem=True,
#                                         out_dir=self.out_large_dir)
#
#     def test5w_trace(self):
#         # Test the small alignment for the computation of angles between the match and mismatch entropy but only
#         # considering a subset of the rank/groups.
#         self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
#                             assignments=self.assignments_custom_small, single=False, pair=True,
#                             metric='match_mismatch_entropy_ratio', num_proc=self.max_threads,
#                             out_dir=self.out_small_dir, match_mismatch=True, gap_correction=None, low_mem=True)
#
#     def test5x_trace(self):
#         # Test the large alignment for the computation of angles between the match and mismatch entropy but only
#         # considering a subset of the rank/groups.
#         self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
#                             assignments=self.assignments_custom_large, single=False, pair=True,
#                             metric='match_mismatch_entropy_ratio', num_proc=self.max_threads,
#                             out_dir=self.out_large_dir, match_mismatch=True, gap_correction=None, low_mem=True)
#
#     def test5y_trace(self):
#         # Test the small alignment for the computation of angles between the match and mismatch entropy but only
#         # considering a subset of the rank/groups.
#         self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
#                             assignments=self.assignments_custom_small, single=False, pair=True,
#                             metric='match_mismatch_entropy_angle', num_proc=self.max_threads,
#                             out_dir=self.out_small_dir, match_mismatch=True, gap_correction=None, low_mem=True)
#
#     def test5z_trace(self):
#         # Test the large alignment for the computation of angles between the match and mismatch entropy but only
#         # considering a subset of the rank/groups.
#         self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
#                             assignments=self.assignments_custom_large, single=False, pair=True,
#                             metric='match_mismatch_entropy_angle', num_proc=self.max_threads,
#                             out_dir=self.out_large_dir, match_mismatch=True, gap_correction=None, low_mem=True)
#
#     def test5aa_trace(self):
#         # Test the small alignment for the computation of angles between the match and mismatch entropy but only
#         # considering a subset of the rank/groups.
#         self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
#                             assignments=self.assignments_custom_small, single=False, pair=True,
#                             metric='match_diversity_mismatch_entropy_ratio', num_proc=self.max_threads,
#                             out_dir=self.out_small_dir, match_mismatch=True, gap_correction=None, low_mem=True)
#
#     def test5ab_trace(self):
#         # Test the large alignment for the computation of angles between the match and mismatch entropy but only
#         # considering a subset of the rank/groups.
#         self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
#                             assignments=self.assignments_custom_large, single=False, pair=True,
#                             metric='match_diversity_mismatch_entropy_ratio', num_proc=self.max_threads,
#                             out_dir=self.out_large_dir, match_mismatch=True, gap_correction=None, low_mem=True)
#
#     def test5ac_trace(self):
#         # Test the small alignment for the computation of angles between the match and mismatch entropy but only
#         # considering a subset of the rank/groups.
#         self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
#                             assignments=self.assignments_custom_small, single=False, pair=True,
#                             metric='match_diversity_mismatch_entropy_angle', num_proc=self.max_threads,
#                             out_dir=self.out_small_dir, match_mismatch=True, gap_correction=None, low_mem=True)
#
#     def test5ad_trace(self):
#         # Test the large alignment for the computation of angles between the match and mismatch entropy but only
#         # considering a subset of the rank/groups.
#         self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
#                             assignments=self.assignments_custom_large, single=False, pair=True,
#                             metric='match_diversity_mismatch_entropy_angle', num_proc=self.max_threads,
#                             out_dir=self.out_large_dir, match_mismatch=True, gap_correction=None, low_mem=True)


if __name__ == '__main__':
    unittest.main()
