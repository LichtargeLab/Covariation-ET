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
from test_Base import (protein_alpha, protein_alpha_size, _, protein_map, protein_rev, pair_protein_alpha,
                       pro_pair_alpha_size, pro_pair_map, pro_pair_rev, quad_protein_alpha, pro_quad_alpha_size,
                       pro_quad_map, pro_quad_rev, pro_single_to_pair, pro_single_to_pair_map, pro_single_to_quad_map,
                       protein_seq1, protein_seq2, protein_seq3, protein_msa, protein_aln, protein_num_aln,
                       protein_phylo_tree, protein_rank_dict, pro_single_ft, pro_single_ft_i2, pro_single_ft_s1,
                       pro_single_ft_s2, pro_single_ft_s3, pro_pair_ft, pro_pair_ft_i2, pro_pair_ft_s1, pro_pair_ft_s2,
                       pro_pair_ft_s3, protein_mm_table, protein_mm_table_large, pro_single_to_pair, pro_pair_to_quad,
                       pro_pair_mismatch, pro_quad_mismatch)
from utils import compute_rank_and_coverage
from FrequencyTable import FrequencyTable
from PositionalScorer import PositionalScorer
from Trace import (Trace, init_characterization_pool, init_characterization_mm_pool, characterization,
                   characterization_mm, init_trace_groups, trace_groups, init_trace_ranks, trace_ranks,
                   check_freq_table, save_freq_table, load_freq_table,
                   check_numpy_array, save_numpy_array, load_numpy_array)


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
                # if s1 > 0 and s2 > 0:
                if s1 < 1 and s2 < 2:
                    expected_single_tables['Inner2']['match']._increment_count(pos=pos1, char=single_char)
            else:
                expected_single_tables['Inner1']['mismatch']._increment_count(pos=pos1, char=single_char)
                # if s1 > 0 and s2 > 0:
                if s1 < 1 and s2 < 2:
                    expected_single_tables['Inner2']['mismatch']._increment_count(pos=pos1, char=single_char)
            for pos2 in range(pos1, 6):
                pair_stat, pair_char = protein_mm_table_large.get_status_and_character(pos=(pos1, pos2), seq_ind1=s1,
                                                                                       seq_ind2=s2)
                if pair_stat == 'match':
                    expected_pair_tables['Inner1']['match']._increment_count(pos=(pos1, pos2), char=pair_char)
                    # if s1 > 0 and s2 > 0:
                    if s1 < 1 and s2 < 2:
                        expected_pair_tables['Inner2']['match']._increment_count(pos=(pos1, pos2), char=pair_char)
                else:
                    expected_pair_tables['Inner1']['mismatch']._increment_count(pos=(pos1, pos2), char=pair_char)
                    # if s1 > 0 and s2 > 0:
                    if s1 < 1 and s2 < 2:
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

    def evaluate_characterization_pool(self, alpha_size, alpha_map, alpha_rev, single_to_pair, pos_size, low_mem,
                                       write_aln, write_ft, processes):
        pos_type = 'single' if pos_size == 1 else 'pair'
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_pool(alpha_size=alpha_size, alpha_mapping=alpha_map, alpha_reverse=alpha_rev,
                                   single_to_pair=single_to_pair, alignment=protein_aln, pos_size=pos_size,
                                   components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                   unique_dir=u_dir, low_memory=low_mem, write_out_sub_aln=write_aln,
                                   write_out_freq_table=write_ft, processes=processes)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            if low_mem:
                self.assertIsInstance(freq_tables[node_name]['freq_table'], str)
            else:
                self.assertIsInstance(freq_tables[node_name]['freq_table'], FrequencyTable)
            loaded1 = load_freq_table(freq_tables[node_name]['freq_table'], low_mem)
            expected1 = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
            expected1.characterize_alignment(num_aln=np.array([protein_num_aln[ind, :]]), single_to_pair=single_to_pair)
            self.assertEqual(loaded1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded1.get_count_matrix() - expected1.get_count_matrix()).any())
            if write_aln:
                expected_path1a = os.path.join(u_dir, f'{node_name}.fa')
                self.assertTrue(os.path.isfile(expected_path1a))
            if write_ft:
                expected_path1b = os.path.join(u_dir, f'{node_name}_{pos_type}_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path1b))
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('freq_table' in freq_tables[node_name])
            if low_mem:
                self.assertIsInstance(freq_tables[node_name]['freq_table'], str)
            else:
                self.assertIsInstance(freq_tables[node_name]['freq_table'], FrequencyTable)
            loaded2 = load_freq_table(freq_tables[node_name]['freq_table'], low_mem)
            expected2 = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
            expected2.characterize_alignment(num_aln=protein_num_aln[inds, :], single_to_pair=single_to_pair)
            self.assertEqual(loaded2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded2.get_count_matrix() - expected2.get_count_matrix()).any())
            if write_aln:
                expected_path2a = os.path.join(u_dir, f'{node_name}.fa')
                self.assertTrue(os.path.isfile(expected_path2a))
            if write_ft:
                expected_path2b = os.path.join(u_dir, f'{node_name}_{pos_type}_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path2b))
        rmtree(u_dir)

    def test_characterization_pool_single_low_memory_single_process(self):
        self.evaluate_characterization_pool(alpha_size=protein_alpha_size, alpha_map=protein_map, alpha_rev=protein_rev,
                                            single_to_pair=None, pos_size=1, low_mem=True, write_aln=False,
                                            write_ft=False, processes=1)

    def test_characterization_pool_single_low_memory_multi_process(self):
        self.evaluate_characterization_pool(alpha_size=protein_alpha_size, alpha_map=protein_map, alpha_rev=protein_rev,
                                            single_to_pair=None, pos_size=1, low_mem=True, write_aln=False,
                                            write_ft=False, processes=2)

    def test_characterization_pool_single_not_low_memory_single_process(self):
        self.evaluate_characterization_pool(alpha_size=protein_alpha_size, alpha_map=protein_map, alpha_rev=protein_rev,
                                            single_to_pair=None, pos_size=1, low_mem=False, write_aln=False,
                                            write_ft=False, processes=1)

    def test_characterization_pool_single_not_low_memory_multi_process(self):
        self.evaluate_characterization_pool(alpha_size=protein_alpha_size, alpha_map=protein_map, alpha_rev=protein_rev,
                                            single_to_pair=None, pos_size=1, low_mem=False, write_aln=False,
                                            write_ft=False, processes=2)

    def test_characterization_pool_single_write_out(self):
        self.evaluate_characterization_pool(alpha_size=protein_alpha_size, alpha_map=protein_map, alpha_rev=protein_rev,
                                            single_to_pair=None, pos_size=1, low_mem=False, write_aln=True,
                                            write_ft=True, processes=1)

    def test_characterization_pool_pair_low_memory_single_process(self):
        self.evaluate_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                            alpha_rev=pro_pair_rev, single_to_pair=pro_single_to_pair, pos_size=2,
                                            low_mem=True, write_aln=False, write_ft=False, processes=1)

    def test_characterization_pool_pair_low_memory_multi_process(self):
        self.evaluate_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                            alpha_rev=pro_pair_rev, single_to_pair=pro_single_to_pair, pos_size=2,
                                            low_mem=True, write_aln=False, write_ft=False, processes=2)

    def test_characterization_pool_pair_not_low_memory_single_process(self):
        self.evaluate_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                            alpha_rev=pro_pair_rev, single_to_pair=pro_single_to_pair, pos_size=2,
                                            low_mem=False, write_aln=False, write_ft=False, processes=1)

    def test_characterization_pool_pair_not_low_memory_multi_process(self):
        self.evaluate_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                            alpha_rev=pro_pair_rev, single_to_pair=pro_single_to_pair, pos_size=2,
                                            low_mem=False, write_aln=False, write_ft=False, processes=2)

    def test_characterization_pool_pair_write_out(self):
        self.evaluate_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                            alpha_rev=pro_pair_rev, single_to_pair=pro_single_to_pair, pos_size=2,
                                            low_mem=False, write_aln=True, write_ft=True, processes=1)

    def test_characterization_pool_failure_no_node_name(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=protein_aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(KeyError):
            characterization(node_name=None, node_type='component')
        rmtree(u_dir)

    def test_characterization_pool_failure_missing_node_name(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=protein_aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(KeyError):
            characterization(node_name='seq4', node_type='component')
        rmtree(u_dir)

    def test_characterization_pool_failure_no_node_type(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=protein_aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(ValueError):
            characterization(node_name='seq1', node_type=None)
        rmtree(u_dir)

    def test_characterization_pool_failure_unexpected_node_type(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=protein_aln,
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
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
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
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        with self.assertRaises(ValueError):
            init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                       alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=protein_aln,
                                       pos_size=None, components=components, sharable_dict=freq_tables,
                                       sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                       write_out_sub_aln=True, write_out_freq_table=True, processes=1)
            characterization(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_pool_failure_bad_pos_size(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        with self.assertRaises(ValueError):
            init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                       alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=protein_aln,
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
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=protein_aln,
                                   pos_size=2, components=None, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(TypeError):
            characterization(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_pool_failure_no_shareable_dict(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=protein_aln,
                                   pos_size=2, components=components, sharable_dict=None,
                                   sharable_lock=tables_lock, unique_dir=u_dir, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(TypeError):
            characterization(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_pool_failure_no_shareable_lock(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=protein_aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=None, unique_dir=u_dir, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(AttributeError):
            characterization(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_pool_failure_bad_low_memory(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=protein_aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=None, low_memory='low',
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(ValueError):
            characterization(node_name='seq1', node_type='component')

    def test_characterization_pool_failure_low_memory_no_unique_dir(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=protein_aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=None, low_memory=True,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(ValueError):
            characterization(node_name='seq1', node_type='component')

    def test_characterization_pool_failure_no_processes(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=protein_aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=None, low_memory=False,
                                   write_out_sub_aln=False, write_out_freq_table=False, processes=None)
        with self.assertRaises(TypeError):
            characterization(node_name='seq1', node_type='component')

    def test_characterization_pool_failure_write_out_no_u_dir(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=protein_aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=None, low_memory=False,
                                   write_out_sub_aln=True, write_out_freq_table=True, processes=1)
        with self.assertRaises(TypeError):
            characterization(node_name='seq1', node_type='component')

    def test_characterization_pool_failure_max_iters_reached(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_pool(alpha_size=pro_pair_alpha_size, alpha_mapping=pro_pair_map,
                                   alpha_reverse=pro_pair_rev, single_to_pair=pro_single_to_pair, alignment=protein_aln,
                                   pos_size=2, components=components, sharable_dict=freq_tables,
                                   sharable_lock=tables_lock, unique_dir=None, low_memory=False,
                                   write_out_sub_aln=False, write_out_freq_table=False, processes=1,
                                   maximum_iterations=10)
        with self.assertRaises(TimeoutError):
            characterization(node_name='Inner1', node_type='inner')


class TestTraceCharacterizationMMPool(TestCase):

    def evaluate_characterization_mm_pool(self, l_size, l_map, l_rev, s_to_p, comp, mis_mask, pos_size, low_mem,
                                          write_aln, write_ft):
        if pos_size == 1:
            expected_tables = expected_single_tables
            pos_type = 'single'
        else:
            expected_tables = expected_pair_tables
            pos_type = 'pair'
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=l_size, larger_mapping=l_map,
                                      larger_reverse=l_rev, single_to_pair=s_to_p, comparison_mapping=comp,
                                      mismatch_mask=mis_mask, alignment=protein_aln, position_size=pos_size,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=u_dir, low_memory=low_mem, write_out_sub_aln=write_aln,
                                      write_out_freq_table=write_ft)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            curr_node = characterization_mm(node_name=node_name, node_type='component')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('match' in freq_tables[node_name])
            if low_mem:
                self.assertIsInstance(freq_tables[node_name]['match'], str)
            else:
                self.assertIsInstance(freq_tables[node_name]['match'], FrequencyTable)
            loaded1a = load_freq_table(freq_tables[node_name]['match'], low_mem)
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in freq_tables[node_name])
            if low_mem:
                self.assertIsInstance(freq_tables[node_name]['mismatch'], str)
            else:
                self.assertIsInstance(freq_tables[node_name]['mismatch'], FrequencyTable)
            loaded1b = load_freq_table(freq_tables[node_name]['mismatch'], low_mem)
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
            if write_aln:
                expected_path1a = os.path.join(u_dir, f'{node_name}.fa')
                self.assertTrue(os.path.isfile(expected_path1a))
            if write_ft:
                expected_path1b = os.path.join(u_dir, f'{node_name}_{pos_type}_match_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path1b))
                expected_path1c = os.path.join(u_dir, f'{node_name}_{pos_type}_mismatch_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path1c))
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            curr_node = characterization_mm(node_name=node_name, node_type='inner')
            self.assertEqual(curr_node, node_name)
            self.assertTrue(node_name in freq_tables)
            self.assertTrue('match' in freq_tables[node_name])
            if low_mem:
                self.assertIsInstance(freq_tables[node_name]['match'], str)
            else:
                self.assertIsInstance(freq_tables[node_name]['match'], FrequencyTable)
            loaded2a = load_freq_table(freq_tables[node_name]['match'], low_mem)
            self.assertEqual(loaded2a.get_depth(), expected_tables[node_name]['match'].get_depth())

            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in freq_tables[node_name])
            if low_mem:
                self.assertIsInstance(freq_tables[node_name]['mismatch'], str)
            else:
                self.assertIsInstance(freq_tables[node_name]['mismatch'], FrequencyTable)
            loaded2b = load_freq_table(freq_tables[node_name]['mismatch'], low_mem)
            self.assertEqual(loaded2b.get_depth(), expected_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_tables[node_name]['mismatch'].get_count_matrix()).any())
            if write_aln:
                expected_path2a = os.path.join(u_dir, f'{node_name}.fa')
                self.assertTrue(os.path.isfile(expected_path2a))
            if write_ft:
                expected_path2b = os.path.join(u_dir, f'{node_name}_{pos_type}_match_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path2b))
                expected_path2c = os.path.join(u_dir, f'{node_name}_{pos_type}_mismatch_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path2c))
        rmtree(u_dir)

    def test_characterization_mm_pool_single_low_memory(self):
        self.evaluate_characterization_mm_pool(l_size=pro_pair_alpha_size, l_map=pro_pair_map, l_rev=pro_pair_rev,
                                               s_to_p=None, comp=pro_single_to_pair, mis_mask=pro_pair_mismatch,
                                               pos_size=1, low_mem=True, write_aln=False, write_ft=False)

    def test_characterization_mm_pool_single_not_low_memory(self):
        self.evaluate_characterization_mm_pool(l_size=pro_pair_alpha_size, l_map=pro_pair_map, l_rev=pro_pair_rev,
                                               s_to_p=None, comp=pro_single_to_pair, mis_mask=pro_pair_mismatch,
                                               pos_size=1, low_mem=False, write_aln=False, write_ft=False)

    def test_characterization_mm_pool_single_write_out(self):
        self.evaluate_characterization_mm_pool(l_size=pro_pair_alpha_size, l_map=pro_pair_map, l_rev=pro_pair_rev,
                                               s_to_p=None, comp=pro_single_to_pair, mis_mask=pro_pair_mismatch,
                                               pos_size=1, low_mem=True, write_aln=True, write_ft=True)

    def test_characterization_mm_pool_pair_low_memory(self):
        self.evaluate_characterization_mm_pool(l_size=pro_quad_alpha_size, l_map=pro_quad_map, l_rev=pro_quad_rev,
                                               s_to_p=pro_single_to_pair, comp=pro_pair_to_quad,
                                               mis_mask=pro_quad_mismatch, pos_size=2, low_mem=True, write_aln=False,
                                               write_ft=False)

    def test_characterization_mm_pool_pair_not_low_memory(self):
        self.evaluate_characterization_mm_pool(l_size=pro_quad_alpha_size, l_map=pro_quad_map, l_rev=pro_quad_rev,
                                               s_to_p=pro_single_to_pair, comp=pro_pair_to_quad,
                                               mis_mask=pro_quad_mismatch, pos_size=2, low_mem=False,
                                               write_aln=False, write_ft=False)

    def test_characterization_mm_pool_pair_write_out(self):
        self.evaluate_characterization_mm_pool(l_size=pro_quad_alpha_size, l_map=pro_quad_map, l_rev=pro_quad_rev,
                                               s_to_p=pro_single_to_pair, comp=pro_pair_to_quad,
                                               mis_mask=pro_quad_mismatch, pos_size=2, low_mem=True,
                                               write_aln=True, write_ft=True)

    def test_characterization_mm_pool_failure_no_node_name(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                      larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                      single_to_pair=pro_single_to_pair, comparison_mapping=pro_pair_to_quad,
                                      mismatch_mask=pro_quad_mismatch, alignment=protein_aln, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(ValueError):
            characterization_mm(node_name=None, node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_missing_node_name(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                      larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                      single_to_pair=pro_single_to_pair, comparison_mapping=pro_pair_to_quad,
                                      mismatch_mask=pro_quad_mismatch, alignment=protein_aln, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(KeyError):
            characterization_mm(node_name='Seq4', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_node_type(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                      larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                      single_to_pair=pro_single_to_pair, comparison_mapping=pro_pair_to_quad,
                                      mismatch_mask=pro_quad_mismatch, alignment=protein_aln, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(ValueError):
            characterization_mm(node_name='seq1', node_type=None)
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_unexpected_node_type(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                      larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                      single_to_pair=pro_single_to_pair, comparison_mapping=pro_pair_to_quad,
                                      mismatch_mask=pro_quad_mismatch, alignment=protein_aln, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(ValueError):
            characterization_mm(node_name='seq1', node_type='intermediate')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_single_mapping(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(single_mapping=None, larger_size=pro_quad_alpha_size, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, single_to_pair=pro_single_to_pair,
                                      comparison_mapping=pro_pair_to_quad, mismatch_mask=pro_quad_mismatch,
                                      alignment=protein_aln, position_size=2, components=components,
                                      sharable_dict=freq_tables, sharable_lock=tables_lock, unique_dir=u_dir,
                                      low_memory=True, write_out_sub_aln=True, write_out_freq_table=True)
        with self.assertRaises(TypeError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_larger_size(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=None, larger_mapping=pro_quad_map,
                                      larger_reverse=pro_quad_rev, single_to_pair=pro_single_to_pair,
                                      comparison_mapping=pro_pair_to_quad, mismatch_mask=pro_quad_mismatch,
                                      alignment=protein_aln, position_size=2, components=components,
                                      sharable_dict=freq_tables, sharable_lock=tables_lock, unique_dir=u_dir,
                                      low_memory=True, write_out_sub_aln=True, write_out_freq_table=True)
        with self.assertRaises(TypeError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_larger_mapping(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size, larger_mapping=None,
                                      larger_reverse=pro_quad_rev, single_to_pair=pro_single_to_pair,
                                      comparison_mapping=pro_pair_to_quad, mismatch_mask=pro_quad_mismatch,
                                      alignment=protein_aln, position_size=2, components=components,
                                      sharable_dict=freq_tables, sharable_lock=tables_lock, unique_dir=u_dir,
                                      low_memory=True, write_out_sub_aln=True, write_out_freq_table=True)
        with self.assertRaises(AttributeError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_larger_reverse(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                      larger_mapping=pro_quad_map, larger_reverse=None,
                                      single_to_pair=pro_single_to_pair, comparison_mapping=pro_pair_to_quad,
                                      mismatch_mask=pro_quad_mismatch, alignment=protein_aln, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(TypeError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_single_to_pair(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                      larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                      single_to_pair=None, comparison_mapping=pro_pair_to_quad,
                                      mismatch_mask=pro_quad_mismatch, alignment=protein_aln, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(ValueError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_comparison_mapping(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                      larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                      single_to_pair=pro_single_to_pair, comparison_mapping=None,
                                      mismatch_mask=pro_quad_mismatch, alignment=protein_aln, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(ValueError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_mismatch_mask(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                      larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                      single_to_pair=pro_single_to_pair, comparison_mapping=pro_pair_to_quad,
                                      mismatch_mask=None, alignment=protein_aln, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(ValueError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_alignment(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                      larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                      single_to_pair=pro_single_to_pair, comparison_mapping=pro_pair_to_quad,
                                      mismatch_mask=pro_quad_mismatch, alignment=None, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(AttributeError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_pos_size(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        with self.assertRaises(ValueError):
            init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                          larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                          single_to_pair=pro_single_to_pair, comparison_mapping=pro_pair_to_quad,
                                          mismatch_mask=pro_quad_mismatch, alignment=protein_aln, position_size=None,
                                          components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                          unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                          write_out_freq_table=True)
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_bad_pos_size(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        with self.assertRaises(ValueError):
            init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                          larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                          single_to_pair=pro_single_to_pair, comparison_mapping=pro_pair_to_quad,
                                          mismatch_mask=pro_quad_mismatch, alignment=protein_aln, position_size=3,
                                          components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                          unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                          write_out_freq_table=True)
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_components(self):
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                      larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                      single_to_pair=pro_single_to_pair, comparison_mapping=pro_pair_to_quad,
                                      mismatch_mask=pro_quad_mismatch, alignment=protein_aln, position_size=2,
                                      components=None, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(TypeError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_shareable_dict(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        tables_lock = Lock()
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                      larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                      single_to_pair=pro_single_to_pair, comparison_mapping=pro_pair_to_quad,
                                      mismatch_mask=pro_quad_mismatch, alignment=protein_aln, position_size=2,
                                      components=components, sharable_dict=None, sharable_lock=tables_lock,
                                      unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(TypeError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_no_shareable_lock(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        u_dir = os.path.join(os.getcwd(), 'char_test')
        os.mkdir(u_dir)
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                      larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                      single_to_pair=pro_single_to_pair, comparison_mapping=pro_pair_to_quad,
                                      mismatch_mask=pro_quad_mismatch, alignment=protein_aln, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=None,
                                      unique_dir=u_dir, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(AttributeError):
            characterization_mm(node_name='seq1', node_type='component')
        rmtree(u_dir)

    def test_characterization_mm_pool_failure_bad_low_memory(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                      larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                      single_to_pair=pro_single_to_pair, comparison_mapping=pro_pair_to_quad,
                                      mismatch_mask=pro_quad_mismatch, alignment=protein_aln, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=None, low_memory='low', write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(ValueError):
            characterization_mm(node_name='seq1', node_type='component')

    def test_characterization_mm_pool_failure_low_memory_no_unique_dir(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                      larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                      single_to_pair=pro_single_to_pair, comparison_mapping=pro_pair_to_quad,
                                      mismatch_mask=pro_quad_mismatch, alignment=protein_aln, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=None, low_memory=True, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(ValueError):
            characterization_mm(node_name='seq1', node_type='component')

    def test_characterization_mm_pool_failure_write_out_no_u_dir(self):
        components = {'Inner1': protein_rank_dict[1][1], 'Inner2': protein_rank_dict[2][1],
                      'seq1': protein_rank_dict[3][3], 'seq2': protein_rank_dict[3][2], 'seq3': protein_rank_dict[3][1]}
        pool_manager = Manager()
        freq_tables = pool_manager.dict()
        tables_lock = Lock()
        init_characterization_mm_pool(single_mapping=protein_map, larger_size=pro_quad_alpha_size,
                                      larger_mapping=pro_quad_map, larger_reverse=pro_quad_rev,
                                      single_to_pair=pro_single_to_pair, comparison_mapping=pro_pair_to_quad,
                                      mismatch_mask=pro_quad_mismatch, alignment=protein_aln, position_size=2,
                                      components=components, sharable_dict=freq_tables, sharable_lock=tables_lock,
                                      unique_dir=None, low_memory=False, write_out_sub_aln=True,
                                      write_out_freq_table=True)
        with self.assertRaises(TypeError):
            characterization_mm(node_name='seq1', node_type='component')

    # Need to add a test for maximum_iterations


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
                    cumulative_ranks += load_numpy_array(u_dict[protein_rank_dict[r][g]['node'].name]['group_scores'], True)
                else:
                    cumulative_ranks += u_dict[protein_rank_dict[r][g]['node'].name]['group_scores']
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
        init_trace_ranks(scorer=scorer, a_dict=protein_rank_dict, u_dict=unique_dict, low_memory=False,
                         unique_dir=None)
        self.evaluate_trace_ranks(scorer=scorer, u_dict=unique_dict, mode='int')

    def test_trace_ranks_pos_size_2_integer_rank(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        unique_dict = {'Inner1': {'group_scores': scorer.score_group(pro_pair_ft)},
                       'Inner2': {'group_scores': scorer.score_group(pro_pair_ft_i2)},
                       'seq1': {'group_scores': scorer.score_group(pro_pair_ft_s1)},
                       'seq2': {'group_scores': scorer.score_group(pro_pair_ft_s2)},
                       'seq3': {'group_scores': scorer.score_group(pro_pair_ft_s3)}}
        init_trace_ranks(scorer=scorer, a_dict=protein_rank_dict, u_dict=unique_dict, low_memory=False,
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
        init_trace_ranks(scorer=scorer, a_dict=protein_rank_dict, u_dict=unique_dict, low_memory=True,
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
        init_trace_ranks(scorer=scorer, a_dict=protein_rank_dict, u_dict=unique_dict, low_memory=True,
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
        init_trace_ranks(scorer=scorer, a_dict=protein_rank_dict, u_dict=unique_dict, low_memory=False,
                         unique_dir=None)
        self.evaluate_trace_ranks(scorer=scorer, u_dict=unique_dict, mode='real')

    def test_trace_ranks_pos_size_2_real_rank(self):
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        unique_dict = {'Inner1': {'group_scores': scorer.score_group(pro_pair_ft)},
                       'Inner2': {'group_scores': scorer.score_group(pro_pair_ft_i2)},
                       'seq1': {'group_scores': scorer.score_group(pro_pair_ft_s1)},
                       'seq2': {'group_scores': scorer.score_group(pro_pair_ft_s2)},
                       'seq3': {'group_scores': scorer.score_group(pro_pair_ft_s3)}}
        init_trace_ranks(scorer=scorer, a_dict=protein_rank_dict, u_dict=unique_dict, low_memory=False,
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
        init_trace_ranks(scorer=scorer, a_dict=protein_rank_dict, u_dict=unique_dict, low_memory=True,
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
        init_trace_ranks(scorer=scorer, a_dict=protein_rank_dict, u_dict=unique_dict, low_memory=True,
                         unique_dir=u_dir)
        self.evaluate_trace_ranks(scorer=scorer, u_dict=unique_dict, mode='real', mem='low')
        rmtree(u_dir)

    def test_trace_ranks_failure_no_rank(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        unique_dict = {'Inner1': {'group_scores': scorer.score_group(pro_single_ft)}}
        init_trace_ranks(scorer=scorer, a_dict=protein_rank_dict, u_dict=unique_dict, low_memory=False,
                         unique_dir=None)
        with self.assertRaises(KeyError):
            trace_ranks(None)

    def test_trace_ranks_failure_no_scorer(self):
        unique_dict = {'Inner1': {'group_scores': np.random.rand(6)}}
        init_trace_ranks(scorer=None, a_dict=protein_rank_dict, u_dict=unique_dict, low_memory=False,
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
        init_trace_ranks(scorer=scorer, a_dict=protein_rank_dict, u_dict=None, low_memory=False,
                         unique_dir=None)
        with self.assertRaises(TypeError):
            trace_ranks(1)

    def test_trace_ranks_failure_bad_low_memory(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        unique_dict = {'Inner1': {'group_scores': scorer.score_group(pro_single_ft)}}
        init_trace_ranks(scorer=scorer, a_dict=protein_rank_dict, u_dict=unique_dict, low_memory='low_mem',
                         unique_dir=None)
        with self.assertRaises(ValueError):
            trace_ranks(1)

    def test_trace_ranks_failure_low_memory_no_unique_dir(self):
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        unique_dict = {'Inner1': None}
        init_trace_ranks(scorer=scorer, a_dict=protein_rank_dict, u_dict=unique_dict, low_memory=True, unique_dir=None)
        with self.assertRaises(ValueError):
            trace_ranks(1)


class TestTraceInit(TestCase):

    def evaluate_init(self, aln, phylo_tree, rank_dict, pos_size, mm_bool, out_dir, low_mem):
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=rank_dict, pos_size=pos_size,
                      match_mismatch=mm_bool, output_dir=out_dir, low_memory=low_mem)
        self.assertIs(trace.aln, aln)
        self.assertIs(trace.phylo_tree, phylo_tree)
        self.assertIs(trace.assignments, rank_dict)
        self.assertEqual(trace.match_mismatch, mm_bool)
        self.assertEqual(trace.low_memory, low_mem)
        self.assertEqual(trace.pos_size, pos_size)
        if out_dir is None:
            self.assertEqual(trace.out_dir, os.getcwd())
        else:
            self.assertEqual(trace.out_dir, out_dir)
        self.assertIsNone(trace.unique_nodes)
        self.assertIsNone(trace.rank_scores)
        self.assertIsNone(trace.final_scores)
        self.assertIsNone(trace.final_ranks)
        self.assertIsNone(trace.final_coverage)

    def test_init_out_dir(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_init(aln=protein_aln, phylo_tree=protein_phylo_tree, rank_dict=protein_rank_dict, pos_size=1,
                           mm_bool=False, out_dir=expected_dir, low_mem=False)
        rmtree(expected_dir)

    def test_init_out_dir_does_not_exist(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        self.assertFalse(os.path.isdir(expected_dir))
        self.evaluate_init(aln=protein_aln, phylo_tree=protein_phylo_tree, rank_dict=protein_rank_dict, pos_size=1,
                           mm_bool=False, out_dir=expected_dir, low_mem=False)
        rmtree(expected_dir)

    def test_init_no_out_dir(self):
        self.evaluate_init(aln=protein_aln, phylo_tree=protein_phylo_tree, rank_dict=protein_rank_dict, pos_size=1,
                           mm_bool=False, out_dir=None, low_mem=False)

    def test_init_position_size_2(self):
        self.evaluate_init(aln=protein_aln, phylo_tree=protein_phylo_tree, rank_dict=protein_rank_dict, pos_size=2,
                           mm_bool=False, out_dir=None, low_mem=False)

    def test_init_failure_position_size_low(self):
        with self.assertRaises(ValueError):
            Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=0,
                  match_mismatch=False, output_dir=None, low_memory=False)

    def test_init_failure_position_size_high(self):
        with self.assertRaises(ValueError):
            Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=3,
                  match_mismatch=False, output_dir=None, low_memory=False)

    def test_init_failure_bad_alignment(self):
        with self.assertRaises(ValueError):
            Trace(alignment=None, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=1,
                  match_mismatch=False, output_dir=None, low_memory=False)

    def test_init_failure_bad_phylo_tree(self):
        with self.assertRaises(ValueError):
            Trace(alignment=protein_aln, phylo_tree=None, group_assignments=protein_rank_dict, pos_size=1,
                  match_mismatch=False, output_dir=None, low_memory=False)

    def test_init_failure_bad_group_assignments(self):
        with self.assertRaises(ValueError):
            Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=None, pos_size=1,
                  match_mismatch=False, output_dir=None, low_memory=False)

    def test_init_failure_bad_match_mismatch(self):
        with self.assertRaises(ValueError):
            Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=1,
                  match_mismatch=None, output_dir=None, low_memory=False)

    def test_init_failure_bad_low_memory(self):
        with self.assertRaises(ValueError):
            Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=1,
                  match_mismatch=False, output_dir=None, low_memory=None)

    def test_init_failure_bad_output_dir(self):
        with self.assertRaises(TypeError):
            Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=1,
                  match_mismatch=False, output_dir=100, low_memory=False)


class TestTraceCharacterizeRankGroupStandard(TestCase):

    def evaluate_characterize_rank_groups(self, pos_size, out_dir, low_mem, u_dir, alpha_size, alpha_map,
                                          alpha_rev, s_to_p, processes, write_aln, write_ft):
        pos_type = 'single' if pos_size == 1 else 'pair'
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=pos_size, match_mismatch=False, output_dir=out_dir, low_memory=low_mem)
        trace.characterize_rank_groups_standard(unique_dir=u_dir, alpha_size=alpha_size, alpha_mapping=alpha_map,
                                                alpha_reverse=alpha_rev, single_to_pair=s_to_p, processes=processes,
                                                write_out_sub_aln=write_aln, write_out_freq_table=write_ft)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            if low_mem:
                self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], str)
            else:
                self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            loaded1 = load_freq_table(trace.unique_nodes[node_name]['freq_table'], low_mem)
            expected1 = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
            expected1.characterize_alignment(num_aln=np.array([protein_num_aln[ind, :]]), single_to_pair=s_to_p)
            self.assertEqual(loaded1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded1.get_count_matrix() - expected1.get_count_matrix()).any())
            if write_aln:
                expected_path1a = os.path.join(out_dir, f'{node_name}.fa')
                self.assertTrue(os.path.isfile(expected_path1a))
            if write_ft:
                expected_path1b = os.path.join(out_dir, f'{node_name}_{pos_type}_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path1b))
                loaded_1 = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
                loaded_1.load_csv(expected_path1b)
                self.assertEqual(loaded_1.get_depth(), expected1.get_depth())
                self.assertFalse((loaded_1.get_count_matrix() - expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            if low_mem:
                self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], str)
            else:
                self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            loaded2 = load_freq_table(trace.unique_nodes[node_name]['freq_table'], low_mem)
            expected2 = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
            expected2.characterize_alignment(num_aln=protein_num_aln[inds, :], single_to_pair=s_to_p)
            self.assertEqual(loaded2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded2.get_count_matrix() - expected2.get_count_matrix()).any())
            if write_aln:
                expected_path2a = os.path.join(out_dir, f'{node_name}.fa')
                self.assertTrue(os.path.isfile(expected_path2a))
            if write_ft:
                expected_path2b = os.path.join(out_dir, f'{node_name}_{pos_type}_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path2b))
                loaded_2 = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
                loaded_2.load_csv(expected_path2b)
                self.assertEqual(loaded_2.get_depth(), expected2.get_depth())
                self.assertFalse((loaded_2.get_count_matrix() - expected2.get_count_matrix()).any())

    def test_characterize_rank_groups_standard_single(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups(pos_size=1, out_dir=expected_dir, low_mem=False, u_dir=None,
                                               alpha_size=protein_alpha_size, alpha_map=protein_map,
                                               alpha_rev=protein_rev, s_to_p=None, processes=1, write_aln=False,
                                               write_ft=False)
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_single_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups(pos_size=1, out_dir=expected_dir, low_mem=False, u_dir=None,
                                               alpha_size=protein_alpha_size, alpha_map=protein_map,
                                               alpha_rev=protein_rev, s_to_p=None, processes=2, write_aln=False,
                                               write_ft=False)
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_single_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups(pos_size=1, out_dir=expected_dir, low_mem=True, u_dir=expected_dir,
                                               alpha_size=protein_alpha_size, alpha_map=protein_map,
                                               alpha_rev=protein_rev, s_to_p=None, processes=1, write_aln=False,
                                               write_ft=False)
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_single_write_out(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups(pos_size=1, out_dir=expected_dir, low_mem=False, u_dir=expected_dir,
                                               alpha_size=protein_alpha_size, alpha_map=protein_map,
                                               alpha_rev=protein_rev, s_to_p=None, processes=1, write_aln=True,
                                               write_ft=True)
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_pair(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups(pos_size=2, out_dir=expected_dir, low_mem=False, u_dir=None,
                                               alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                               alpha_rev=pro_pair_rev, s_to_p=pro_single_to_pair, processes=1,
                                               write_aln=False, write_ft=False)
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_pair_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups(pos_size=2, out_dir=expected_dir, low_mem=False, u_dir=None,
                                               alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                               alpha_rev=pro_pair_rev, s_to_p=pro_single_to_pair, processes=2,
                                               write_aln=False, write_ft=False)
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_pair_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups(pos_size=2, out_dir=expected_dir, low_mem=True, u_dir=expected_dir,
                                               alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                               alpha_rev=pro_pair_rev, s_to_p=pro_single_to_pair, processes=1,
                                               write_aln=False, write_ft=False)
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_pair_write_out(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups(pos_size=2, out_dir=expected_dir, low_mem=False, u_dir=expected_dir,
                                               alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                               alpha_rev=pro_pair_rev, s_to_p=pro_single_to_pair, processes=1,
                                               write_aln=True, write_ft=True)
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_failure_low_mem_no_unique_dir(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=1,
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
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=1,
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
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=1,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        with self.assertRaises(ValueError):
            trace.characterize_rank_groups_standard(unique_dir=None, alpha_size=protein_alpha_size,
                                                    alpha_mapping=protein_map, alpha_reverse=protein_rev,
                                                    single_to_pair=None, processes=None, write_out_sub_aln=True,
                                                    write_out_freq_table=True)
        rmtree(expected_dir)

    def test_characterize_rank_groups_standard_failure_max_iters_reached(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=1,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        with self.assertRaises(ValueError):
            trace.characterize_rank_groups_standard(unique_dir=None, alpha_size=protein_alpha_size,
                                                    alpha_mapping=protein_map, alpha_reverse=protein_rev,
                                                    single_to_pair=None, processes=None, write_out_sub_aln=False,
                                                    write_out_freq_table=False, maximum_iterations=10)
        rmtree(expected_dir)


class TestTraceCharacterizeRankGroupMatchMismatch(TestCase):

    def evaluate_characterize_rank_group_mm(self, pos_size, out_dir, low_mem, u_dir, processes, write_aln, write_ft,
                                            alpha_size, alpha_map, alpha_rev):
        if pos_size == 1:
            pos_type = 'single'
            expected_tables = expected_single_tables
        else:
            pos_type = 'pair'
            expected_tables = expected_pair_tables

        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=pos_size, match_mismatch=True, output_dir=out_dir, low_memory=low_mem)
        trace.characterize_rank_groups_match_mismatch(unique_dir=u_dir, single_size=protein_alpha_size,
                                                      single_mapping=protein_map, processes=processes,
                                                      write_out_sub_aln=write_aln, write_out_freq_table=write_ft)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            if low_mem:
                self.assertIsInstance(trace.unique_nodes[node_name]['match'], str)
            else:
                self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded1a = load_freq_table(trace.unique_nodes[node_name]['match'], low_mem)
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            if low_mem:
                self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], str)
            else:
                self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded1b = load_freq_table(trace.unique_nodes[node_name]['mismatch'], low_mem)
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
            if write_aln:
                expected_path1a = os.path.join(out_dir, f'{node_name}.fa')
                self.assertTrue(os.path.isfile(expected_path1a))
            if write_ft:
                expected_path1b = os.path.join(out_dir, f'{node_name}_{pos_type}_match_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path1b))
                loaded_1a = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
                loaded_1a.load_csv(expected_path1b)
                self.assertEqual(loaded_1a.get_depth(), 1)
                self.assertFalse(loaded_1a.get_count_matrix().any())
                expected_path1c = os.path.join(out_dir, f'{node_name}_{pos_type}_mismatch_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path1c))
                loaded_1b = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
                loaded_1b.load_csv(expected_path1c)
                self.assertEqual(loaded_1b.get_depth(), 1)
                self.assertFalse(loaded_1b.get_count_matrix().any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            if low_mem:
                self.assertIsInstance(trace.unique_nodes[node_name]['match'], str)
            else:
                self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded2a = load_freq_table(trace.unique_nodes[node_name]['match'], low_mem)
            self.assertEqual(loaded2a.get_depth(), expected_tables[node_name]['match'].get_depth())
            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_tables[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            if low_mem:
                self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], str)
            else:
                self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded2b = load_freq_table(trace.unique_nodes[node_name]['mismatch'], low_mem)
            self.assertEqual(loaded2b.get_depth(), expected_tables[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_tables[node_name]['mismatch'].get_count_matrix()).any())
            if write_aln:
                expected_path2a = os.path.join(out_dir, f'{node_name}.fa')
                self.assertTrue(os.path.isfile(expected_path2a))
            if write_ft:
                expected_path2b = os.path.join(out_dir, f'{node_name}_{pos_type}_match_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path2b))
                loaded_2a = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
                loaded_2a.load_csv(expected_path2b, expected_tables[node_name]['match'].get_depth())
                self.assertEqual(loaded_2a.get_depth(), expected_tables[node_name]['match'].get_depth())
                self.assertFalse((loaded_2a.get_count_matrix() -
                                  expected_tables[node_name]['match'].get_count_matrix()).any())
                expected_path2c = os.path.join(out_dir, f'{node_name}_{pos_type}_mismatch_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path2c))
                loaded_2b = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
                loaded_2b.load_csv(expected_path2c, expected_tables[node_name]['match'].get_depth())
                self.assertEqual(loaded_2b.get_depth(), expected_tables[node_name]['match'].get_depth())
                self.assertFalse((loaded_2b.get_count_matrix() -
                                  expected_tables[node_name]['mismatch'].get_count_matrix()).any())

    def test_characterize_rank_groups_match_mismatch_single(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_group_mm(pos_size=1, out_dir=expected_dir, low_mem=False, u_dir=None,
                                                 processes=1, write_aln=False, write_ft=False,
                                                 alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                                 alpha_rev=pro_pair_rev)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_single_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_group_mm(pos_size=1, out_dir=expected_dir, low_mem=False, u_dir=None,
                                                 processes=2, write_aln=False, write_ft=False,
                                                 alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                                 alpha_rev=pro_pair_rev)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_single_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_group_mm(pos_size=1, out_dir=expected_dir, low_mem=True, u_dir=expected_dir,
                                                 processes=1, write_aln=False, write_ft=False,
                                                 alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                                 alpha_rev=pro_pair_rev)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_single_write_out(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_group_mm(pos_size=1, out_dir=expected_dir, low_mem=False, u_dir=expected_dir,
                                                 processes=1, write_aln=True, write_ft=True,
                                                 alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                                 alpha_rev=pro_pair_rev)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_pair(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_group_mm(pos_size=2, out_dir=expected_dir, low_mem=False, u_dir=None,
                                                 processes=1, write_aln=False, write_ft=False,
                                                 alpha_size=pro_quad_alpha_size, alpha_map=pro_quad_map,
                                                 alpha_rev=pro_quad_rev)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_pair_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_group_mm(pos_size=2, out_dir=expected_dir, low_mem=False, u_dir=None,
                                                 processes=2, write_aln=False, write_ft=False,
                                                 alpha_size=pro_quad_alpha_size, alpha_map=pro_quad_map,
                                                 alpha_rev=pro_quad_rev)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_pair_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_group_mm(pos_size=2, out_dir=expected_dir, low_mem=True, u_dir=expected_dir,
                                                 processes=1, write_aln=False, write_ft=False,
                                                 alpha_size=pro_quad_alpha_size, alpha_map=pro_quad_map,
                                                 alpha_rev=pro_quad_rev)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_pair_write_out(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_group_mm(pos_size=2, out_dir=expected_dir, low_mem=False, u_dir=expected_dir,
                                                 processes=1, write_aln=True, write_ft=True,
                                                 alpha_size=pro_quad_alpha_size, alpha_map=pro_quad_map,
                                                 alpha_rev=pro_quad_rev)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_failure_low_mem_no_unique_dir(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=1,
                      match_mismatch=True, output_dir=expected_dir, low_memory=True)
        with self.assertRaises(ValueError):
            trace.characterize_rank_groups_match_mismatch(unique_dir=None, single_size=protein_alpha_size,
                                                          single_mapping=protein_map, processes=1,
                                                          write_out_sub_aln=False, write_out_freq_table=False)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_failure_no_single_size(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=1,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        with self.assertRaises(TypeError):
            trace.characterize_rank_groups_match_mismatch(unique_dir=None, single_mapping=protein_map, processes=1,
                                                          write_out_sub_aln=False, write_out_freq_table=False)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_failure_no_single_mapping(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=1,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        with self.assertRaises(TypeError):
            trace.characterize_rank_groups_match_mismatch(unique_dir=None, single_size=protein_alpha_size,
                                                          single_mapping=None, processes=1, write_out_sub_aln=False,
                                                          write_out_freq_table=False)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_success_no_processes(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_group_mm(pos_size=1, out_dir=expected_dir, low_mem=False, u_dir=None,
                                                 processes=None, write_aln=False, write_ft=False,
                                                 alpha_size=pro_pair_alpha_size, alpha_map=pro_pair_map,
                                                 alpha_rev=pro_pair_rev)
        rmtree(expected_dir)

    def test_characterize_rank_groups_match_mismatch_failure_write_out_no_unique_dir(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=1,
                      match_mismatch=True, output_dir=expected_dir, low_memory=True)
        with self.assertRaises(ValueError):
            trace.characterize_rank_groups_match_mismatch(unique_dir=None, single_size=protein_alpha_size,
                                                          single_mapping=protein_map, processes=1,
                                                          write_out_sub_aln=True, write_out_freq_table=True)
        rmtree(expected_dir)

    # Need to add a test for maximum_iterations


class TestTraceCharacterizeRankGroups(TestCase):

    def evaluate_characterize_rank_groups(self, pos_size, out_dir, low_mem, processes, write_aln, write_ft, alpha_size,
                                          alpha_map, alpha_rev, s_to_p):
        u_dir = os.path.join(out_dir, 'unique_node_data')
        if pos_size == 1:
            pos_type = 'single'
        else:
            pos_type = 'pair'
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=pos_size, match_mismatch=False, output_dir=out_dir, low_memory=low_mem)
        trace.characterize_rank_groups(processes=processes, write_out_sub_aln=write_aln, write_out_freq_table=write_ft)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            if low_mem:
                self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], str)
            else:
                self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            loaded1 = load_freq_table(trace.unique_nodes[node_name]['freq_table'], low_mem)
            expected1 = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
            expected1.characterize_alignment(num_aln=np.array([protein_num_aln[ind, :]]), single_to_pair=s_to_p)
            self.assertEqual(loaded1.get_depth(), expected1.get_depth())
            self.assertFalse((loaded1.get_count_matrix() - expected1.get_count_matrix()).any())
            if write_aln:
                expected_path1a = os.path.join(u_dir, f'{node_name}.fa')
                self.assertTrue(os.path.isfile(expected_path1a), expected_path1a)
            if write_ft:
                expected_path1b = os.path.join(u_dir, f'{node_name}_{pos_type}_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path1b))
                loaded_1 = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
                loaded_1.load_csv(expected_path1b)
                self.assertEqual(loaded_1.get_depth(), expected1.get_depth())
                self.assertFalse((loaded_1.get_count_matrix() - expected1.get_count_matrix()).any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('freq_table' in trace.unique_nodes[node_name])
            if low_mem:
                self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], str)
            else:
                self.assertIsInstance(trace.unique_nodes[node_name]['freq_table'], FrequencyTable)
            loaded2 = load_freq_table(trace.unique_nodes[node_name]['freq_table'], low_mem)
            expected2 = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
            expected2.characterize_alignment(num_aln=protein_num_aln[inds, :], single_to_pair=s_to_p)
            self.assertEqual(loaded2.get_depth(), expected2.get_depth())
            self.assertFalse((loaded2.get_count_matrix() - expected2.get_count_matrix()).any())
            if write_aln:
                expected_path2a = os.path.join(u_dir, f'{node_name}.fa')
                self.assertTrue(os.path.isfile(expected_path2a))
            if write_ft:
                expected_path2b = os.path.join(u_dir, f'{node_name}_{pos_type}_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path2b))
                loaded_2 = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
                loaded_2.load_csv(expected_path2b)
                self.assertEqual(loaded_2.get_depth(), expected2.get_depth())
                self.assertFalse((loaded_2.get_count_matrix() - expected2.get_count_matrix()).any())

    def test_characterize_rank_groups_single(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups(pos_size=1, out_dir=expected_dir, low_mem=False, processes=1,
                                               write_aln=False, write_ft=False, alpha_size=protein_alpha_size,
                                               alpha_map=protein_map, alpha_rev=protein_rev, s_to_p=None)
        rmtree(expected_dir)

    def test_characterize_rank_groups_single_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups(pos_size=1, out_dir=expected_dir, low_mem=False, processes=2,
                                               write_aln=False, write_ft=False, alpha_size=protein_alpha_size,
                                               alpha_map=protein_map, alpha_rev=protein_rev, s_to_p=None)
        rmtree(expected_dir)

    def test_characterize_rank_groups_single_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups(pos_size=1, out_dir=expected_dir, low_mem=True, processes=1,
                                               write_aln=False, write_ft=False, alpha_size=protein_alpha_size,
                                               alpha_map=protein_map, alpha_rev=protein_rev, s_to_p=None)
        rmtree(expected_dir)

    def test_characterize_rank_groups_single_write_out(self):
        starting_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(starting_dir)
        self.evaluate_characterize_rank_groups(pos_size=1, out_dir=starting_dir, low_mem=False, processes=1,
                                               write_aln=True, write_ft=True, alpha_size=protein_alpha_size,
                                               alpha_map=protein_map, alpha_rev=protein_rev, s_to_p=None)
        rmtree(starting_dir)

    def test_characterize_rank_groups_pair(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups(pos_size=2, out_dir=expected_dir, low_mem=False, processes=1,
                                               write_aln=False, write_ft=False, alpha_size=pro_pair_alpha_size,
                                               alpha_map=pro_pair_map, alpha_rev=pro_pair_rev,
                                               s_to_p=pro_single_to_pair)
        rmtree(expected_dir)

    def test_characterize_rank_groups_pair_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups(pos_size=2, out_dir=expected_dir, low_mem=False, processes=2,
                                               write_aln=False, write_ft=False, alpha_size=pro_pair_alpha_size,
                                               alpha_map=pro_pair_map, alpha_rev=pro_pair_rev,
                                               s_to_p=pro_single_to_pair)
        rmtree(expected_dir)

    def test_characterize_rank_groups_pair_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups(pos_size=2, out_dir=expected_dir, low_mem=True, processes=1,
                                               write_aln=False, write_ft=False, alpha_size=pro_pair_alpha_size,
                                               alpha_map=pro_pair_map, alpha_rev=pro_pair_rev,
                                               s_to_p=pro_single_to_pair)
        rmtree(expected_dir)

    def test_characterize_rank_groups_pair_write_out(self):
        starting_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(starting_dir)
        self.evaluate_characterize_rank_groups(pos_size=2, out_dir=starting_dir, low_mem=False, processes=1,
                                               write_aln=True, write_ft=True, alpha_size=pro_pair_alpha_size,
                                               alpha_map=pro_pair_map, alpha_rev=pro_pair_rev,
                                               s_to_p=pro_single_to_pair)
        rmtree(starting_dir)

    def evaluate_characterize_rank_groups_mm(self, pos_size, out_dir, low_mem, processes, write_aln, write_ft,
                                             alpha_size, alpha_map, alpha_rev):
        if pos_size == 1:
            pos_type = 'single'
            expected_table = expected_single_tables
        else:
            pos_type = 'pair'
            expected_table = expected_pair_tables
        u_dir = os.path.join(out_dir, 'unique_node_data')
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=pos_size, match_mismatch=True, output_dir=out_dir, low_memory=low_mem)
        trace.characterize_rank_groups(processes=processes, write_out_sub_aln=write_aln, write_out_freq_table=write_ft)
        for ind, node_name in enumerate(['seq1', 'seq2', 'seq3']):
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            if low_mem:
                self.assertIsInstance(trace.unique_nodes[node_name]['match'], str)
            else:
                self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded1a = load_freq_table(trace.unique_nodes[node_name]['match'], low_mem)
            self.assertEqual(loaded1a.get_depth(), 1)
            self.assertFalse(loaded1a.get_count_matrix().any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            if low_mem:
                self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], str)
            else:
                self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded1b = load_freq_table(trace.unique_nodes[node_name]['mismatch'], low_mem)
            self.assertEqual(loaded1b.get_depth(), 1)
            self.assertFalse(loaded1b.get_count_matrix().any())
            if write_aln:
                expected_path1a = os.path.join(u_dir, f'{node_name}.fa')
                self.assertTrue(os.path.isfile(expected_path1a))
            if write_ft:
                expected_path1b = os.path.join(u_dir, f'{node_name}_{pos_type}_match_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path1b))
                loaded_1a = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
                loaded_1a.load_csv(expected_path1b)
                self.assertEqual(loaded_1a.get_depth(), 1)
                self.assertFalse(loaded_1a.get_count_matrix().any())
                expected_path1c = os.path.join(u_dir, f'{node_name}_{pos_type}_mismatch_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path1c))
                loaded_1b = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
                loaded_1b.load_csv(expected_path1c)
                self.assertEqual(loaded_1b.get_depth(), 1)
                self.assertFalse(loaded_1b.get_count_matrix().any())
        for inds, node_name in [([1, 2], 'Inner2'), ([0, 1, 2], 'Inner1')]:
            self.assertTrue(node_name in trace.unique_nodes)
            self.assertTrue('match' in trace.unique_nodes[node_name])
            if low_mem:
                self.assertIsInstance(trace.unique_nodes[node_name]['match'], str)
            else:
                self.assertIsInstance(trace.unique_nodes[node_name]['match'], FrequencyTable)
            loaded2a = load_freq_table(trace.unique_nodes[node_name]['match'], low_mem)
            self.assertEqual(loaded2a.get_depth(), expected_table[node_name]['match'].get_depth())
            self.assertFalse((loaded2a.get_count_matrix() -
                              expected_table[node_name]['match'].get_count_matrix()).any())
            self.assertTrue('mismatch' in trace.unique_nodes[node_name])
            if low_mem:
                self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], str)
            else:
                self.assertIsInstance(trace.unique_nodes[node_name]['mismatch'], FrequencyTable)
            loaded2b = load_freq_table(trace.unique_nodes[node_name]['mismatch'], low_mem)
            self.assertEqual(loaded2b.get_depth(), expected_table[node_name]['mismatch'].get_depth())
            self.assertFalse((loaded2b.get_count_matrix() -
                              expected_table[node_name]['mismatch'].get_count_matrix()).any())
            if write_aln:
                expected_path2a = os.path.join(u_dir, f'{node_name}.fa')
                self.assertTrue(os.path.isfile(expected_path2a))
            if write_ft:
                expected_path2b = os.path.join(u_dir, f'{node_name}_{pos_type}_match_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path2b))
                loaded_2a = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
                loaded_2a.load_csv(expected_path2b, expected_table[node_name]['match'].get_depth())
                self.assertEqual(loaded_2a.get_depth(), expected_table[node_name]['match'].get_depth())
                self.assertFalse((loaded_2a.get_count_matrix() -
                                  expected_table[node_name]['match'].get_count_matrix()).any())
                expected_path2c = os.path.join(u_dir, f'{node_name}_{pos_type}_mismatch_freq_table.tsv')
                self.assertTrue(os.path.isfile(expected_path2c))
                loaded_2b = FrequencyTable(alpha_size, alpha_map, alpha_rev, 6, pos_size)
                loaded_2b.load_csv(expected_path2c, expected_table[node_name]['match'].get_depth())
                self.assertEqual(loaded_2b.get_depth(), expected_table[node_name]['match'].get_depth())
                self.assertFalse((loaded_2b.get_count_matrix() -
                                  expected_table[node_name]['mismatch'].get_count_matrix()).any())

    def test_characterize_rank_groups_mm_single(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups_mm(pos_size=1, out_dir=expected_dir, low_mem=False, processes=1,
                                                  write_aln=False, write_ft=False, alpha_size=None, alpha_map=None,
                                                  alpha_rev=None)
        rmtree(expected_dir)

    def test_characterize_rank_groups_mm_single_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups_mm(pos_size=1, out_dir=expected_dir, low_mem=False, processes=2,
                                                  write_aln=False, write_ft=False, alpha_size=None, alpha_map=None,
                                                  alpha_rev=None)
        rmtree(expected_dir)

    def test_characterize_rank_groups_mm_single_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups_mm(pos_size=1, out_dir=expected_dir, low_mem=True, processes=1,
                                                  write_aln=False, write_ft=False, alpha_size=None, alpha_map=None,
                                                  alpha_rev=None)
        rmtree(expected_dir)

    def test_characterize_rank_groups_mm_single_write_out(self):
        starting_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(starting_dir)
        self.evaluate_characterize_rank_groups_mm(pos_size=1, out_dir=starting_dir, low_mem=False, processes=1,
                                                  write_aln=True, write_ft=True, alpha_size=pro_pair_alpha_size,
                                                  alpha_map=pro_pair_map, alpha_rev=pro_pair_rev)
        rmtree(starting_dir)

    def test_characterize_rank_groups_mm_pair(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups_mm(pos_size=2, out_dir=expected_dir, low_mem=False, processes=1,
                                                  write_aln=False, write_ft=False, alpha_size=None, alpha_map=None,
                                                  alpha_rev=None)
        rmtree(expected_dir)

    def test_characterize_rank_groups_mm_pair_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups_mm(pos_size=2, out_dir=expected_dir, low_mem=False, processes=2,
                                                  write_aln=False, write_ft=False, alpha_size=None, alpha_map=None,
                                                  alpha_rev=None)
        rmtree(expected_dir)

    def test_characterize_rank_groups_mm_pair_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups_mm(pos_size=2, out_dir=expected_dir, low_mem=True, processes=1,
                                                  write_aln=False, write_ft=False, alpha_size=None, alpha_map=None,
                                                  alpha_rev=None)
        rmtree(expected_dir)

    def test_characterize_rank_groups_mm_pair_write_out(self):
        starting_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(starting_dir)
        self.evaluate_characterize_rank_groups_mm(pos_size=2, out_dir=starting_dir, low_mem=False, processes=1,
                                                  write_aln=True, write_ft=True, alpha_size=pro_quad_alpha_size,
                                                  alpha_map=pro_quad_map, alpha_rev=pro_quad_rev)
        rmtree(starting_dir)

    def test_characterize_rank_groups_failure_no_processes(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=1,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        with self.assertRaises(ValueError):
            trace.characterize_rank_groups(processes=None, maximum_iterations=10,
                                           write_out_sub_aln=False, write_out_freq_table=False)
        rmtree(expected_dir)

    def test_characterize_rank_groups_mm_success_no_processes(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        self.evaluate_characterize_rank_groups_mm(pos_size=1, out_dir=expected_dir, low_mem=False, processes=None,
                                                  write_aln=False, write_ft=False, alpha_size=None, alpha_map=None,
                                                  alpha_rev=None)
        rmtree(expected_dir)


class TestTraceTrace(TestCase):

    def evaluate_trace(self, scorer, trace, gap_correction=None, gap_char_num=None):
        if trace.low_memory:
            expected_class = str
        else:
            expected_class = np.ndarray
        checked = set()
        cumulative_rank_score = np.zeros(scorer.dimensions)
        for rank in sorted(protein_rank_dict.keys()):
            cumulative_group_score = np.zeros(scorer.dimensions)
            for group in sorted(protein_rank_dict[rank]):
                curr_node = protein_rank_dict[rank][group]['node'].name
                group_score = load_numpy_array(trace.unique_nodes[curr_node]['group_scores'], trace.low_memory)
                if node not in checked:
                    self.assertTrue(curr_node in trace.unique_nodes)
                    self.assertTrue('group_scores' in trace.unique_nodes[curr_node])
                    checked.add(curr_node)
                    self.assertIsInstance(trace.unique_nodes[curr_node]['group_scores'], expected_class)
                    if trace.match_mismatch:
                        freq_table = {'match': load_freq_table(trace.unique_nodes[curr_node]['match'],
                                                               trace.low_memory),
                                      'mismatch': load_freq_table(trace.unique_nodes[curr_node]['mismatch'],
                                                                  trace.low_memory)}
                    else:
                        freq_table = load_freq_table(trace.unique_nodes[curr_node]['freq_table'], trace.low_memory)

                    expected_group_score = scorer.score_group(freq_table=freq_table)
                    self.assertFalse((group_score - expected_group_score).any())
                cumulative_group_score += group_score
            self.assertTrue(rank in trace.rank_scores)
            self.assertIsInstance(trace.rank_scores[rank], expected_class)
            rank_score = load_numpy_array(trace.rank_scores[rank], trace.low_memory)
            expected_rank_score = scorer.score_rank(cumulative_group_score, rank)
            self.assertFalse((rank_score - expected_rank_score).any())
            cumulative_rank_score += expected_rank_score
        if scorer.rank_type == 'min':
            if trace.pos_size == 1:
                cumulative_rank_score += 1
            else:
                cumulative_rank_score += np.triu(np.ones(scorer.dimensions), k=1)
        if gap_correction is None:
            self.assertFalse((trace.final_scores - cumulative_rank_score).any())
        else:
            if scorer.rank_type == 'min':
                worst_rank = np.max(cumulative_rank_score)
            else:
                worst_rank = np.min(cumulative_rank_score)
            if trace.match_mismatch:
                root_ft = (load_freq_table(trace.unique_nodes['Inner1']['match'], trace.low_memory) +
                           load_freq_table(trace.unique_nodes['Inner1']['mismatch'], trace.low_memory))
            else:
                root_ft = load_freq_table(trace.unique_nodes['Inner1']['freq_table'], trace.low_memory)
            frequencies = root_ft.get_frequency_matrix()
            positions = frequencies[:, gap_char_num] > gap_correction
            if trace.pos_size == 1:
                cumulative_rank_score[positions] = worst_rank
            else:
                all_indices = np.triu_indices(scorer.dimensions[0])
                relevant_ind_1 = all_indices[0][positions]
                relevant_ind_2 = all_indices[1][positions]
                second_mask = relevant_ind_1 != relevant_ind_2
                cumulative_rank_score[relevant_ind_1[second_mask], relevant_ind_2[second_mask]] = worst_rank
            self.assertFalse((trace.final_scores - cumulative_rank_score).any())
        expected_ranks, expected_coverage = compute_rank_and_coverage(seq_length=6, scores=cumulative_rank_score,
                                                                      pos_size=trace.pos_size,
                                                                      rank_type=scorer.rank_type)
        self.assertFalse((trace.final_ranks - expected_ranks).any())
        self.assertFalse((trace.final_coverage - expected_coverage).any())

    def test_trace_single_identity(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=1,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_identity(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_single_plain_entropy(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=1,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='plain_entropy')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_plain_entropy(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_mutual_information(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='mutual_information')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_normalized_mutual_information(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='normalized_mutual_information')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_average_product_corrected_mutual_information(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='average_product_corrected_mutual_information')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_filtered_average_product_corrected_mutual_information(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2,
                                  metric='filtered_average_product_corrected_mutual_information')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_match_count(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_count')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_mismatch_count(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_count')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_match_mismatch_count_ratio(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_ratio')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_match_mismatch_count_angle(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_angle')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_match_entropy(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_entropy')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_mismatch_entropy(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_entropy')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_match_mismatch_entropy_ratio(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_ratio')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_match_mismatch_entropy_angle(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_angle')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_match_diversity(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_mismatch_diversity(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_diversity')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_match_mismatch_diversity_ratio(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_ratio')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_match_mismatch_diversity_angle(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_angle')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_match_diversity_mismatch_entropy_ratio(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
                      match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_ratio')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_match_diversity_mismatch_entropy_angle(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=2, match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_angle')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_single_identity_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=1, match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=2, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        trace.trace(scorer=scorer, gap_correction=None, processes=2)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_identity_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=2, match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=2, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        trace.trace(scorer=scorer, gap_correction=None, processes=2)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_single_plain_entropy_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=1, match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=2, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='plain_entropy')
        trace.trace(scorer=scorer, gap_correction=None, processes=2)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_plain_entropy_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=2, match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=2, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        trace.trace(scorer=scorer, gap_correction=None, processes=2)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_match_mismatch_count_ratio_multi_process(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=2, match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=2, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_ratio')
        trace.trace(scorer=scorer, gap_correction=None, processes=2)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_single_identity_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=1, match_mismatch=False, output_dir=expected_dir, low_memory=True)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_identity_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=2, match_mismatch=False, output_dir=expected_dir, low_memory=True)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_single_plain_entropy_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=1, match_mismatch=False, output_dir=expected_dir, low_memory=True)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='plain_entropy')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_plain_entropy_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=2, match_mismatch=False, output_dir=expected_dir, low_memory=True)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_pair_match_mismatch_count_ratio_low_mem(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=2, match_mismatch=True, output_dir=expected_dir, low_memory=True)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_ratio')
        trace.trace(scorer=scorer, gap_correction=None, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace)
        rmtree(expected_dir)

    def test_trace_single_identity_gap_correction(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=1, match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        trace.trace(scorer=scorer, gap_correction=0.6, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace, gap_correction=0.6, gap_char_num=protein_map['-'])
        rmtree(expected_dir)

    def test_trace_pair_identity_gap_correction(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=2, match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        trace.trace(scorer=scorer, gap_correction=0.2, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace, gap_correction=0.2, gap_char_num=pro_pair_map['--'])
        rmtree(expected_dir)

    def test_trace_single_plain_entropy_gap_correction(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=1, match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='plain_entropy')
        trace.trace(scorer=scorer, gap_correction=0.6, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace, gap_correction=0.6, gap_char_num=protein_map['-'])
        rmtree(expected_dir)

    def test_trace_pair_plain_entropy_gap_correction(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=2, match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        trace.trace(scorer=scorer, gap_correction=0.2, processes=1)
        self.evaluate_trace(scorer=scorer, trace=trace, gap_correction=0.2, gap_char_num=pro_pair_map['--'])
        rmtree(expected_dir)

    # This test is not useful at the moment because no pair of positions has gaps in multiple sequences.
    # def test_trace_pair_match_mismatch_count_ratio_gap_correction(self):
    #     expected_dir = os.path.join(os.getcwd(), 'test_case')
    #     os.makedirs(expected_dir)
    #     trace = Trace(alignment=protein_aln, protein_phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict, pos_size=2,
    #                   match_mismatch=True, output_dir=expected_dir, low_memory=False)
    #     trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
    #                                    maximum_iterations=10)
    #     scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_ratio')
    #     trace.trace(scorer=scorer, gap_correction=0.1, processes=1)
    #     self.evaluate_trace(scorer=scorer, trace=trace, gap_correction=0.1, gap_char_num=pro_quad_map['----'])
    #     rmtree(expected_dir)

    def test_trace_failure_scorer_pos_size_mismatch_larger(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=1, match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        with self.assertRaises(ValueError):
            trace.trace(scorer=scorer, gap_correction=None, processes=1)
        rmtree(expected_dir)

    def test_trace_failure_scorer_pos_size_mismatch_smaller(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=2, match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        with self.assertRaises(ValueError):
            trace.trace(scorer=scorer, gap_correction=None, processes=1)
        rmtree(expected_dir)

    def test_trace_failure_scorer_mm_disagreement_non_match_scorer(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=2, match_mismatch=True, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        with self.assertRaises(ValueError):
            trace.trace(scorer=scorer, gap_correction=0.1, processes=1)
        rmtree(expected_dir)

    def test_trace_failure_scorer_mm_disagreement_match_scorer(self):
        expected_dir = os.path.join(os.getcwd(), 'test_case')
        os.makedirs(expected_dir)
        trace = Trace(alignment=protein_aln, phylo_tree=protein_phylo_tree, group_assignments=protein_rank_dict,
                      pos_size=2, match_mismatch=False, output_dir=expected_dir, low_memory=False)
        trace.characterize_rank_groups(processes=1, write_out_sub_aln=False, write_out_freq_table=False,
                                       maximum_iterations=10)
        scorer = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_ratio')
        with self.assertRaises(ValueError):
            trace.trace(scorer=scorer, gap_correction=None, processes=1)
        rmtree(expected_dir)


if __name__ == '__main__':
    unittest.main()
