"""
Created on July 11, 2019

@author: Daniel Konecki
"""
import os
import unittest
import numpy as np
from copy import deepcopy
from shutil import rmtree
from Bio.Alphabet import Gapped
from multiprocessing import Lock, Manager, Queue
from Bio.Alphabet.IUPAC import IUPACProtein
from test_Base import TestBase
from utils import build_mapping, gap_characters
from ETMIPWrapper import ETMIPWrapper
from SeqAlignment import SeqAlignment
from FrequencyTable import FrequencyTable
from PhylogeneticTree import PhylogeneticTree
from PositionalScorer import PositionalScorer
from MatchMismatchTable import MatchMismatchTable
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from EvolutionaryTraceAlphabet import FullIUPACProtein, MultiPositionAlphabet
from Trace import (Trace, init_characterization_pool, init_characterization_mm_pool, characterization,
                   characterization_mm, init_trace_groups, trace_groups, init_trace_ranks, trace_ranks, load_freq_table,
                   load_numpy_array)


class TestTrace(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestTrace, cls).setUpClass()
        cls.query_aln_fa_small = SeqAlignment(
            file_name=cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln'],
            query_id=cls.small_structure_id)
        cls.query_aln_fa_small.import_alignment()
        cls.phylo_tree_small = PhylogeneticTree()
        calc = AlignmentDistanceCalculator()
        cls.phylo_tree_small.construct_tree(dm=calc.get_distance(cls.query_aln_fa_small.alignment))
        cls.assignments_small = cls.phylo_tree_small.assign_group_rank()
        cls.assignments_custom_small = cls.phylo_tree_small.assign_group_rank(ranks=[1, 2, 3, 5, 7, 10, 25])
        cls.query_aln_fa_large = SeqAlignment(
            file_name=cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln'],
            query_id=cls.large_structure_id)
        cls.query_aln_fa_large.import_alignment()
        cls.phylo_tree_large = PhylogeneticTree()
        calc = AlignmentDistanceCalculator()
        cls.phylo_tree_large.construct_tree(dm=calc.get_distance(cls.query_aln_fa_large.alignment))
        cls.assignments_large = cls.phylo_tree_large.assign_group_rank()
        cls.assignments_custom_large = cls.phylo_tree_large.assign_group_rank(ranks=[1, 2, 3, 5, 7, 10, 25])
        cls.query_aln_msf_small = deepcopy(cls.query_aln_fa_small)
        cls.query_aln_msf_small.file_name = cls.data_set.protein_data[cls.small_structure_id]['Final_MSF_Aln']
        cls.out_small_dir = os.path.join(cls.testing_dir, cls.small_structure_id)
        if not os.path.isdir(cls.out_small_dir):
            os.makedirs(cls.out_small_dir)
        cls.query_aln_msf_large = deepcopy(cls.query_aln_fa_large)
        cls.query_aln_msf_large.file_name = cls.data_set.protein_data[cls.large_structure_id]['Final_MSF_Aln']
        cls.out_large_dir = os.path.join(cls.testing_dir, cls.large_structure_id)
        if not os.path.isdir(cls.out_large_dir):
            os.makedirs(cls.out_large_dir)
        cls.query_aln_fa_small = cls.query_aln_fa_small.remove_gaps()
        cls.query_aln_fa_large = cls.query_aln_fa_large.remove_gaps()
        cls.single_alphabet = Gapped(FullIUPACProtein())
        cls.single_size, _, cls.single_mapping, cls.single_reverse = build_mapping(alphabet=cls.single_alphabet)
        cls.pair_alphabet = MultiPositionAlphabet(alphabet=cls.single_alphabet, size=2)
        cls.pair_size, _, cls.pair_mapping, cls.pair_reverse = build_mapping(alphabet=cls.pair_alphabet)
        cls.single_to_pair_dict = {}
        for char in cls.pair_mapping:
            key = (cls.single_mapping[char[0]], cls.single_mapping[char[1]])
            cls.single_to_pair_dict[key] = cls.pair_mapping[char]
        cls.single_to_pair_arr = np.zeros((max(cls.single_mapping.values()) + 1, max(cls.single_mapping.values()) + 1))
        for char in cls.pair_mapping:
            cls.single_to_pair_arr[cls.single_mapping[char[0]], cls.single_mapping[char[1]]] = cls.pair_mapping[char]
        cls.quad_alphabet = MultiPositionAlphabet(alphabet=cls.single_alphabet, size=4)
        cls.quad_size, _, cls.quad_mapping, cls.quad_reverse = build_mapping(alphabet=cls.quad_alphabet)
        cls.single_to_quad_dict = {}
        for char in cls.quad_mapping:
            key = (cls.single_mapping[char[0]], cls.single_mapping[char[1]], cls.single_mapping[char[2]],
                   cls.single_mapping[char[3]])
            cls.single_to_quad_dict[key] = cls.quad_mapping[char]

    # @classmethod
    # def tearDownClass(cls):
    #     rmtree(cls.out_small_dir)
    #     rmtree(cls.out_large_dir)

    def evalaute_init(self, aln, tree, assignments, pos_specific, pair_specific, out_dir, low_mem):
        trace_small = Trace(alignment=aln, phylo_tree=tree, group_assignments=assignments,
                            position_specific=pos_specific, pair_specific=pair_specific, output_dir=out_dir,
                            low_memory=low_mem)
        self.assertEqual(trace_small.aln.file_name, aln.file_name)
        self.assertEqual(trace_small.aln.query_id, aln.query_id)
        for s in range(trace_small.aln.size):
            self.assertEqual(str(trace_small.aln.alignment[s].seq), str(aln.alignment[s].seq))
        self.assertEqual(trace_small.aln.seq_order, aln.seq_order)
        self.assertEqual(str(trace_small.aln.query_sequence), str(aln.query_sequence))
        self.assertEqual(trace_small.aln.seq_length, aln.seq_length)
        self.assertEqual(trace_small.aln.size, aln.size)
        self.assertEqual(trace_small.aln.marked, aln.marked)
        self.assertEqual(trace_small.aln.polymer_type, aln.polymer_type)
        self.assertTrue(isinstance(trace_small.aln.alphabet, type(aln.alphabet)))
        self.assertEqual(len(trace_small.aln.alphabet.letters), len(aln.alphabet.letters))
        for char in trace_small.aln.alphabet.letters:
            self.assertTrue(char in aln.alphabet.letters)
        self.assertEqual(trace_small.phylo_tree, tree)
        self.assertEqual(trace_small.assignments, assignments)
        self.assertIsNone(trace_small.unique_nodes)
        self.assertIsNone(trace_small.rank_scores)
        self.assertEqual(trace_small.pos_specific, pos_specific)
        self.assertEqual(trace_small.pair_specific, pair_specific)
        self.assertEqual(trace_small.out_dir, out_dir)
        self.assertEqual(trace_small.low_memory, low_mem)

    # def test1a_init(self):
    #     self.evalaute_init(aln=self.query_aln_fa_small, tree=self.phylo_tree_small, assignments=self.assignments_small,
    #                        pos_specific=True, pair_specific=True, out_dir=self.out_small_dir, low_mem=False)
    #
    # def test1b_init(self):
    #     self.evalaute_init(aln=self.query_aln_fa_large, tree=self.phylo_tree_large, assignments=self.assignments_large,
    #                        pos_specific=True, pair_specific=True, out_dir=self.out_large_dir, low_mem=True)

    def evaluate_characterize_rank_groups_pooling_functions(self, single, pair, aln, assign, out_dir, low_mem,
                                                            write_sub_aln, write_freq_table):
        unique_dir = os.path.join(out_dir, 'unique_node_data')
        if not os.path.isdir(unique_dir):
            os.makedirs(unique_dir)
        # Build a minimal set of nodes to characterize (the query sequence, its neighbor node, and their parent node)
        visited = {}
        to_characterize = []
        found_query = False
        for r in sorted(assign.keys(), reverse=True):
            for g in assign[r]:
                node = assign[r][g]['node']
                if assign[r][g]['descendants'] and (aln.query_id in [d.name for d in assign[r][g]['descendants']]):
                    found_query = True
                    descendants_to_find = set([d.name for d in assign[r][g]['descendants']])
                    searching = len(descendants_to_find)
                    for r2 in range(r + 1, max(assign.keys()) + 1):
                        for g2 in assign[r2]:
                            if assign[r2][g2]['node'].name in descendants_to_find:
                                to_characterize.append((assign[r2][g2]['node'].name, 'component'))
                                visited[assign[r2][g2]['node'].name] = {'terminals': assign[r2][g2]['terminals'],
                                                                        'descendants': assign[r2][g2]['descendants']}
                                searching -= 1
                        if searching == 0:
                            break
                if found_query:
                    to_characterize.append((node.name, 'inner'))
                    visited[node.name] = {'terminals': assign[r][g]['terminals'],
                                          'descendants': assign[r][g]['descendants']}
                    break
            if found_query:
                break
        # Perform characterization
        pool_manager = Manager()
        lock = Lock()
        frequency_tables = pool_manager.dict()
        init_characterization_pool(self.single_size, self.single_mapping, self.single_reverse, self.pair_size,
                                   self.pair_mapping, self.pair_reverse, self.single_to_pair_arr, aln, single, pair,
                                   visited, frequency_tables, lock, unique_dir, low_mem, write_sub_aln,
                                   write_freq_table, 1)
        for to_char in to_characterize:
            ret_name = characterization(*to_char)
            self.assertEqual(ret_name, to_char[0])
            # Evaluate the characterized positions
            sub_aln = aln.generate_sub_alignment(sequence_ids=visited[ret_name]['terminals'])
            if sub_aln.size >= 5:
                expected_single, expected_pair = sub_aln.characterize_positions2(
                    single=single, pair=pair, single_letter_size=self.single_size,
                    single_letter_mapping=self.single_mapping,
                    single_letter_reverse=self.single_reverse, pair_letter_size=self.pair_size,
                    pair_letter_mapping=self.pair_mapping, pair_letter_reverse=self.pair_reverse,
                    single_to_pair=self.single_to_pair_arr)
            else:
                expected_single, expected_pair = sub_aln.characterize_positions(
                    single=single, pair=pair, single_size=self.single_size, single_mapping=self.single_mapping,
                    single_reverse=self.single_reverse, pair_size=self.pair_size, pair_mapping=self.pair_mapping,
                    pair_reverse=self.pair_reverse)
            if write_sub_aln:
                self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}.fa'.format(ret_name))))
            self.assertTrue(ret_name in frequency_tables)
            self.assertTrue('single' in frequency_tables[ret_name])
            if single:
                expected_single_array = expected_single.get_table().toarray()
                single_table = frequency_tables[ret_name]['single']
                if low_mem:
                    single_table = load_freq_table(freq_table=single_table, low_memory=low_mem)
                single_array = single_table.get_table().toarray()
                single_diff = single_array - expected_single_array
                self.assertFalse(single_diff.any())
                if write_freq_table:
                    self.assertTrue(os.path.isfile(os.path.join(unique_dir,
                                                                '{}_position_freq_table.tsv'.format(ret_name))))
            else:
                self.assertIsNone(frequency_tables[ret_name]['single'])
            self.assertTrue('pair' in frequency_tables[ret_name])
            if pair:
                expected_pair_array = expected_pair.get_table().toarray()
                pair_table = frequency_tables[ret_name]['pair']
                if low_mem:
                    pair_table = load_freq_table(freq_table=pair_table, low_memory=low_mem)
                pair_array = pair_table.get_table().toarray()
                pair_diff = pair_array - expected_pair_array
                self.assertFalse(pair_diff.any())
                if write_freq_table:
                    self.assertTrue(os.path.isfile(os.path.join(unique_dir,
                                                                '{}_pair_freq_table.tsv'.format(ret_name))))
            else:
                self.assertIsNone(frequency_tables[ret_name]['pair'])
        rmtree(unique_dir)

    # def test2a_characterize_rank_groups_initialize_characterization_pool(self):
    #     # Test pool initialization function and mappable function (minimal example) for characterization, small aln
    #     self.evaluate_characterize_rank_groups_pooling_functions(
    #         single=True, pair=True, aln=self.query_aln_fa_small, assign=self.assignments_small,
    #         out_dir=self.out_small_dir, low_mem=False, write_sub_aln=True, write_freq_table=True)
    #
    # def test2b_characterize_rank_groups_initialize_characterization_pool(self):
    #     # Test pool initialization function and mappable function (minimal example) for characterization, large aln
    #     self.evaluate_characterize_rank_groups_pooling_functions(
    #         single=True, pair=True, aln=self.query_aln_fa_large, assign=self.assignments_large,
    #         out_dir=self.out_large_dir, low_mem=True, write_sub_aln=False, write_freq_table=False)

    # def evaluate_characterize_rank_groups_mm_pooling_functions(self, single, pair, aln, assign, out_dir, low_mem,
    #                                                            write_sub_aln, write_freq_table):
    #     unique_dir = os.path.join(out_dir, 'unique_node_data')
    #     if not os.path.isdir(unique_dir):
    #         os.makedirs(unique_dir)
    #     single_alphabet = Gapped(aln.alphabet)
    #     single_size, _, single_mapping, single_reverse = build_mapping(alphabet=single_alphabet)
    #     if single and not pair:
    #         larger_alphabet = MultiPositionAlphabet(alphabet=Gapped(aln.alphabet), size=2)
    #         larger_size, _, larger_mapping, larger_reverse = build_mapping(alphabet=larger_alphabet)
    #         single_to_larger = self.single_to_pair_dict
    #         position_size = 1
    #         position_type = 'position'
    #         table_type = 'single'
    #     elif pair and not single:
    #         larger_alphabet = MultiPositionAlphabet(alphabet=Gapped(aln.alphabet), size=4)
    #         larger_size, _, larger_mapping, larger_reverse = build_mapping(alphabet=larger_alphabet)
    #         single_to_larger = self.single_to_quad_dict
    #         position_size = 2
    #         position_type = 'pair'
    #         table_type = 'pair'
    #     else:
    #         raise ValueError('Either single or pair permitted, not both or neither.')
    #     mm_table = MatchMismatchTable(seq_len=aln.seq_length,num_aln=aln._alignment_to_num(self.single_mapping),
    #                                   single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
    #                                   single_reverse_mapping=self.single_reverse, larger_alphabet_size=larger_size,
    #                                   larger_alphabet_mapping=larger_mapping,
    #                                   larger_alphabet_reverse_mapping=larger_reverse,
    #                                   single_to_larger_mapping=single_to_larger, pos_size=position_size)
    #     mm_table.identify_matches_mismatches()
    #     # Build a minimal set of nodes to characterize (the query sequence, its neighbor node, and their parent node)
    #     visited = {}
    #     to_characterize = []
    #     found_query = False
    #     for r in sorted(assign.keys(), reverse=True):
    #         for g in assign[r]:
    #             node = assign[r][g]['node']
    #             if assign[r][g]['descendants'] and (aln.query_id in [d.name for d in assign[r][g]['descendants']]):
    #                 found_query = True
    #                 descendants_to_find = set([d.name for d in assign[r][g]['descendants']])
    #                 searching = len(descendants_to_find)
    #                 for r2 in range(r + 1, max(assign.keys()) + 1):
    #                     for g2 in assign[r2]:
    #                         if assign[r2][g2]['node'].name in descendants_to_find:
    #                             to_characterize.append((assign[r2][g2]['node'].name, 'component'))
    #                             visited[assign[r2][g2]['node'].name] = {'terminals': assign[r2][g2]['terminals'],
    #                                                                     'descendants': assign[r2][g2]['descendants']}
    #                             searching -= 1
    #                     if searching == 0:
    #                         break
    #             if found_query:
    #                 to_characterize.append((node.name, 'inner'))
    #                 visited[node.name] = {'terminals': assign[r][g]['terminals'],
    #                                       'descendants': assign[r][g]['descendants']}
    #                 break
    #         if found_query:
    #             break
    #     pool_manager = Manager()
    #     lock = Lock()
    #     frequency_tables = pool_manager.dict()
    #     init_characterization_mm_pool(single_size, single_mapping, single_reverse, larger_size, larger_mapping,
    #                                   larger_reverse, single_to_larger, mm_table, aln, position_size,
    #                                   position_type, table_type, visited, frequency_tables, lock,
    #                                   unique_dir, low_mem, write_sub_aln, write_freq_table)
    #     for to_char in to_characterize:
    #         ret_name = characterization_mm(*to_char)
    #         self.assertEqual(ret_name, to_char[0])
    #     frequency_tables = dict(frequency_tables)
    #     for node_name in visited:
    #         sub_aln = aln.generate_sub_alignment(sequence_ids=visited[node_name]['terminals'])
    #         if write_sub_aln:
    #             self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}.fa'.format(node_name))))
    #         sub_aln_ind = [aln.seq_order.index(s) for s in sub_aln.seq_order]
    #         possible_matches_mismatches = ((sub_aln.size ** 2) - sub_aln.size) / 2.0
    #         expected_tables = {'match': FrequencyTable(alphabet_size=larger_size, mapping=larger_mapping,
    #                                                    reverse_mapping=larger_reverse,
    #                                                    seq_len=sub_aln.seq_length, pos_size=position_size)}
    #         expected_tables['match'].set_depth(possible_matches_mismatches)
    #         expected_tables['mismatch'] = deepcopy(expected_tables['match'])
    #         for p in expected_tables['match'].get_positions():
    #             char_dict = {'match': {}, 'mismatch': {}}
    #             for i in range(sub_aln.size):
    #                 s1 = sub_aln_ind[i]
    #                 for j in range(i + 1, sub_aln.size):
    #                     s2 = sub_aln_ind[j]
    #                     status, char = mm_table.get_status_and_character(pos=p, seq_ind1=s1, seq_ind2=s2)
    #                     if char not in char_dict[status]:
    #                         char_dict[status][char] = 0
    #                     char_dict[status][char] += 1
    #             for status in char_dict:
    #                 for char in char_dict[status]:
    #                     expected_tables[status]._increment_count(pos=p, char=char,
    #                                                              amount=char_dict[status][char])
    #         expected_tables['match'].finalize_table()
    #         expected_tables['mismatch'].finalize_table()
    #         for m in ['match', 'mismatch']:
    #             m_table = frequency_tables[node_name][m]
    #             m_table = load_freq_table(freq_table=m_table, low_memory=low_mem)
    #             m_table_mat = m_table.get_table()
    #             expected_m_table_mat = expected_tables[m].get_table()
    #             sparse_diff = m_table_mat - expected_m_table_mat
    #             nonzero_check = sparse_diff.count_nonzero() > 0
    #             if nonzero_check:
    #                 print(m_table.get_table().toarray())
    #                 print(expected_tables[m].get_table().toarray())
    #                 print(sparse_diff)
    #                 indices = np.nonzero(sparse_diff)
    #                 print(m_table.get_table().toarray()[indices])
    #                 print(expected_tables[m].get_table().toarray()[indices])
    #                 print(sparse_diff[indices])
    #                 print(node_name)
    #                 print(sub_aln.alignment)
    #             self.assertFalse(nonzero_check)
    #             if write_freq_table:
    #                 expected_table_path = os.path.join(unique_dir, '{}_{}_{}_freq_table.tsv'.format(
    #                     node_name, position_type, m))
    #                 self.assertTrue(os.path.isfile(expected_table_path), 'Not found: {}'.format(expected_table_path))
    #     rmtree(unique_dir)

    def evaluate_characterize_rank_groups_mm_pooling_functions(self, single, pair, aln, assign, out_dir, low_mem,
                                                               write_out_freq_tables):
        unique_dir = os.path.join(out_dir, 'unique_node_data')
        rmtree(unique_dir, ignore_errors=True)
        os.makedirs(unique_dir)
        single_alphabet = Gapped(aln.alphabet)
        single_size, _, single_mapping, single_reverse = build_mapping(alphabet=single_alphabet)
        if single and not pair:
            larger_alphabet = MultiPositionAlphabet(alphabet=Gapped(aln.alphabet), size=2)
            larger_size, _, larger_mapping, larger_reverse = build_mapping(alphabet=larger_alphabet)
            single_to_larger = self.single_to_pair_dict
            position_size = 1
            position_type = 'position'
            table_type = 'single'
        elif pair and not single:
            larger_alphabet = MultiPositionAlphabet(alphabet=Gapped(aln.alphabet), size=4)
            larger_size, _, larger_mapping, larger_reverse = build_mapping(alphabet=larger_alphabet)
            single_to_larger = self.single_to_quad_dict
            position_size = 2
            position_type = 'pair'
            table_type = 'pair'
        else:
            raise ValueError('Either single or pair permitted, not both or neither.')
        mm_table = MatchMismatchTable(seq_len=aln.seq_length,num_aln=aln._alignment_to_num(self.single_mapping),
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=larger_size,
                                      larger_alphabet_mapping=larger_mapping,
                                      larger_alphabet_reverse_mapping=larger_reverse,
                                      single_to_larger_mapping=single_to_larger, pos_size=position_size)
        mm_table.identify_matches_mismatches()
        top_freq_table = FrequencyTable(alphabet_size=larger_size, mapping=larger_mapping,
                                        reverse_mapping=larger_reverse, seq_len=aln.seq_length, pos_size=position_size)
        # Build a minimal set of nodes to characterize (the query sequence, its neighbor node, and their parent node)
        visited = {}
        frequency_tables = {}
        frequency_tables_lock = Lock()
        to_characterize = []
        possible_positions = top_freq_table.get_positions()
        found_query = False
        for r in sorted(assign.keys(), reverse=True):
            for g in assign[r]:
                node = assign[r][g]['node']
                if assign[r][g]['descendants'] and (aln.query_id in [d.name for d in assign[r][g]['descendants']]):
                    found_query = True
                    descendants_to_find = set([d.name for d in assign[r][g]['descendants']])
                    searching = len(descendants_to_find)
                    for r2 in range(r + 1, max(assign.keys()) + 1):
                        for g2 in assign[r2]:
                            if assign[r2][g2]['node'].name in descendants_to_find:
                                to_characterize.append((assign[r2][g2]['node'].name, 'component'))
                                visited[assign[r2][g2]['node'].name] = {'terminals': [aln.seq_order.index(t)
                                                                                      for t in assign[r2][g2]['terminals']],
                                                                        'descendants': assign[r2][g2]['descendants']}
                                sub_aln_size = len(assign[r2][g2]['terminals'])
                                possible_matches_mismatches = (1 if sub_aln_size == 1
                                                               else ((sub_aln_size ** 2) - sub_aln_size) / 2.0)
                                frequency_tables[assign[r2][g2]['node'].name] = {'depth': possible_matches_mismatches,
                                                                                 'remaining_positions': len(possible_positions)}
                                searching -= 1
                        if searching == 0:
                            break
                if found_query:
                    to_characterize.append((node.name, 'inner'))
                    visited[node.name] = {'terminals': [aln.seq_order.index(t) for t in assign[r][g]['terminals']],
                                          'descendants': assign[r][g]['descendants']}
                    sub_aln_size = len(assign[r][g]['terminals'])
                    possible_matches_mismatches = 1 if sub_aln_size == 1 else ((sub_aln_size ** 2) - sub_aln_size) / 2.0
                    frequency_tables[node.name] = {'depth': possible_matches_mismatches,
                                                   'remaining_positions': len(possible_positions)}
                    break
            if found_query:
                break
        init_characterization_mm_pool(single_size, single_mapping, single_reverse, larger_size, larger_mapping,
                                      larger_reverse, single_to_larger, mm_table, aln, position_size,
                                      position_type, table_type, visited, frequency_tables, frequency_tables_lock,
                                      unique_dir, low_mem, write_out_freq_tables)
        for to_char in to_characterize:
            sub_aln = aln.generate_sub_alignment(
                sequence_ids=[aln.seq_order[t] for t in visited[to_char[0]]['terminals']])
            expected_freq_tables = {'match': deepcopy(top_freq_table)}
            expected_freq_tables['match'].set_depth(frequency_tables[to_char[0]]['depth'])
            expected_freq_tables['mismatch'] = deepcopy(expected_freq_tables['match'])
            for i in range(len(possible_positions)):
                p = possible_positions[i]
                ret_name, ret_match, ret_mismatch = characterization_mm(node_name=to_char[0], node_type=to_char[1],
                                                                        pos=p)
                ret_tables = {'match': ret_match, 'mismatch': ret_mismatch}
                self.assertEqual(ret_name, to_char[0])
                expected_char_dict = {'match': {}, 'mismatch': {}}
                if to_char[1] == 'component':
                    for j in range(sub_aln.size):
                        s1 = visited[to_char[0]]['terminals'][j]
                        for k in range(i + 1, sub_aln.size):
                            s2 = visited[to_char[0]]['terminals'][k]
                            status, char = mm_table.get_status_and_character(pos=p, seq_ind1=s1, seq_ind2=s2)
                            if char not in expected_char_dict[status]:
                                expected_char_dict[status][char] = 0
                            expected_char_dict[status][char] += 1
                else:
                    terminal_indices = []
                    for d in visited[to_char[0]]['descendants']:
                        for prev_indices in terminal_indices:
                            for r1 in prev_indices:
                                for r2 in visited[d.name]['terminals']:
                                    first, second = (r1, r2) if r1 < r2 else (r2, r1)
                                    status, char = mm_table.get_status_and_character(p, seq_ind1=first, seq_ind2=second)
                                    if char not in expected_char_dict[status]:
                                        expected_char_dict[status][char] = 0
                                    expected_char_dict[status][char] += 1
                        terminal_indices.append(visited[d.name]['terminals'])
                for status in expected_char_dict:
                    for char in expected_char_dict[status]:
                        expected_freq_tables[status]._increment_count(pos=p, char=char,
                                                                      amount=expected_char_dict[status][char])
                if i < len(possible_positions) - 1:
                    self.assertTrue(to_char[0] in frequency_tables)
                    for curr_status in ['match', 'mismatch']:
                        self.assertIsNone(ret_tables[curr_status])
                else:
                    # self.assertEqual(frequency_tables[to_char[0]]['remaining_positions'], 0)
                    self.assertFalse(to_char[0] in frequency_tables)
                    for curr_status in ['match', 'mismatch']:
                        curr_freq_table = load_freq_table(ret_tables[curr_status], low_mem)
                        expected_freq_tables[curr_status].finalize_table()
                        self.assertEqual(curr_freq_table.get_depth(),
                                         expected_freq_tables[curr_status].get_depth())
                        sparse_diff = curr_freq_table.get_table() - expected_freq_tables[curr_status].get_table()
                        nonzero_check = sparse_diff.count_nonzero() > 0
                        if nonzero_check:
                            print(curr_freq_table.get_table())
                            print(expected_freq_tables[curr_status].get_table())
                            print(sparse_diff)
                            indices = np.nonzero(sparse_diff)
                            print(curr_freq_table.get_table()[indices])
                            print(expected_freq_tables[curr_status].get_table()[indices])
                            print(sparse_diff[indices])
                            print(to_char[0])
                            print(sub_aln.alignment)
                        self.assertFalse(nonzero_check)
                        if (to_char[1] == 'component') and write_out_freq_tables:
                            expected_table_path = os.path.join(unique_dir, '{}_{}_{}_freq_table.tsv'.format(
                                to_char[0], position_type, curr_status))
                            self.assertTrue(os.path.isfile(expected_table_path), 'Not found: {}'.format(expected_table_path))

    # def test2c_characterize_rank_groups_mm_initialize_characterization_pool(self):
    #     # Test pool initialization function and mappable function (minimal example) for characterization, small aln
    #     self.evaluate_characterize_rank_groups_mm_pooling_functions(
    #         single=True, pair=False, aln=self.query_aln_fa_small, assign=self.assignments_small,
    #         out_dir=self.out_small_dir, low_mem=False, write_out_freq_tables=True)
    #         # out_dir=self.out_small_dir, low_mem=False, write_sub_aln=True, write_freq_table=True)
    #         # out_dir=self.out_small_dir, low_mem=False)

    # def test2d_characterize_rank_groups_mm_initialize_characterization_pool(self):
    #     # Test pool initialization function and mappable function (minimal example) for characterization, small aln
    #     self.evaluate_characterize_rank_groups_mm_pooling_functions(
    #         single=False, pair=True, aln=self.query_aln_fa_small, assign=self.assignments_small,
    #         out_dir=self.out_small_dir, low_mem=False, write_out_freq_tables=True)
    #         # out_dir=self.out_small_dir, low_mem=False, write_sub_aln=False, write_freq_table=True)
    #         # out_dir=self.out_small_dir, low_mem=False)

    # def test2e_characterize_rank_groups_mm_initialize_characterization_pool(self):
    #     # Test pool initialization function and mappable function (minimal example) for characterization, small aln
    #     self.evaluate_characterize_rank_groups_mm_pooling_functions(
    #         single=True, pair=False, aln=self.query_aln_fa_large, assign=self.assignments_large,
    #         out_dir=self.out_large_dir, low_mem=True, write_out_freq_tables=False)
    #         # out_dir=self.out_large_dir, low_mem=True, write_sub_aln=False, write_freq_table=False)
    #         # out_dir=self.out_large_dir, low_mem=True)

    # def test2f_characterize_rank_groups_mm_initialize_characterization_pool(self):
    #     # Test pool initialization function and mappable function (minimal example) for characterization, small aln
    #     self.evaluate_characterize_rank_groups_mm_pooling_functions(
    #         single=False, pair=True, aln=self.query_aln_fa_large, assign=self.assignments_large,
    #         out_dir=self.out_large_dir, low_mem=True, write_out_freq_tables=False)
    #         # out_dir=self.out_large_dir, low_mem=True, write_sub_aln=False, write_freq_table=False)
    #         # out_dir=self.out_large_dir, low_mem=True)

    def evaluate_characterize_rank_groups(self, aln, phylo_tree, assign, single, pair, processors, low_mem, write_aln,
                                          write_freq_table):
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=assign, position_specific=single,
                      pair_specific=pair, output_dir=os.path.join(self.testing_dir, aln.query_id), low_memory=low_mem)
        trace.characterize_rank_groups(processes=processors, write_out_sub_aln=write_aln,
                                       write_out_freq_table=write_freq_table)
        visited = set()
        unique_dir = os.path.join(trace.out_dir, 'unique_node_data')
        if not os.path.isdir(unique_dir):
            os.makedirs(unique_dir)
        for rank in trace.assignments:
            for group in trace.assignments[rank]:
                node_name = trace.assignments[rank][group]['node'].name
                self.assertTrue(node_name in trace.unique_nodes)
                self.assertTrue('single' in trace.unique_nodes[node_name])
                self.assertTrue('pair' in trace.unique_nodes[node_name])
                if node_name not in visited:
                    sub_aln = aln.generate_sub_alignment(
                        sequence_ids=trace.assignments[rank][group]['terminals'])
                    if write_aln:
                        self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}.fa'.format(node_name))))
                    if sub_aln.size < 5:
                        expected_single_table, expected_pair_table = sub_aln.characterize_positions(
                            single=single, pair=pair, single_size=self.single_size, single_mapping=self.single_mapping,
                            single_reverse=self.single_reverse, pair_size=self.pair_size, pair_mapping=self.pair_mapping,
                            pair_reverse=self.pair_reverse)
                    else:
                        expected_single_table, expected_pair_table = sub_aln.characterize_positions2(
                            single=single, pair=pair, single_letter_size=self.single_size,
                            single_letter_mapping=self.single_mapping, single_letter_reverse=self.single_reverse,
                            pair_letter_size=self.pair_size, pair_letter_mapping=self.pair_mapping,
                            pair_letter_reverse=self.pair_reverse, single_to_pair=self.single_to_pair_arr)
                    if single:
                        single_table = trace.unique_nodes[node_name]['single']
                        if low_mem:
                            single_table = load_freq_table(freq_table=single_table, low_memory=low_mem)
                        diff = (single_table.get_table() - expected_single_table.get_table()).toarray()
                        if diff.any():
                            print(single_table.get_table().toarray())
                            print(expected_single_table.get_table().toarray())
                            print(diff)
                            indices = np.nonzero(diff)
                            print(single_table.get_table().toarray()[indices])
                            print(expected_single_table.get_table().toarray()[indices])
                            print(diff[indices])
                            print(node_name)
                            print(sub_aln.alignment)
                        self.assertFalse(diff.any())
                        if write_freq_table:
                            self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}_position_freq_table.tsv'.format(
                                node_name))))
                    else:
                        self.assertIsNone(trace.unique_nodes[node_name]['single'])
                    if pair:
                        pair_table = trace.unique_nodes[node_name]['pair']
                        if low_mem:
                            pair_table = load_freq_table(freq_table=pair_table, low_memory=low_mem)
                        diff = pair_table.get_table() - expected_pair_table.get_table()
                        self.assertFalse(diff.toarray().any())
                        if write_freq_table:
                            self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}_pair_freq_table.tsv'.format(
                                node_name))))
                    else:
                        self.assertIsNone(trace.unique_nodes[node_name]['pair'])
                    visited.add(node_name)
        rmtree(unique_dir)

    # def test3a_characterize_rank_groups(self):
    #     # Test characterizing both single and pair positions, small alignment, multi-processed
    #     self.evaluate_characterize_rank_groups(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
    #                                            assign=self.assignments_custom_small, single=True, pair=True,
    #                                            processors=self.max_threads, low_mem=True, write_aln=True,
    #                                            write_freq_table=True)
    #
    # def test3b_characterize_rank_groups(self):
    #     # Test characterizing both single and pair positions, large alignment, single processed
    #     self.evaluate_characterize_rank_groups(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
    #                                            assign=self.assignments_custom_large, single=True, pair=True,
    #                                            processors=1, low_mem=True, write_aln=False, write_freq_table=False)

    def evaluate_characterize_rank_groups_match_mismatch(self, aln, phylo_tree, assign, single, pair, processors,
                                                         low_mem, write_aln, write_freq_table):
        if single:
            pos_size = 1
            pos_type = 'position'
            larger_size = self.pair_size
            larger_mapping = self.pair_mapping
            larger_reverse = self.pair_reverse
            single_to_larger = self.single_to_pair_dict
        else:
            pos_size = 2
            pos_type = 'pair'
            larger_size = self.quad_size
            larger_mapping = self.quad_mapping
            larger_reverse = self.quad_reverse
            single_to_larger = self.single_to_quad_dict
        mm_table = MatchMismatchTable(seq_len=aln.seq_length, num_aln=aln._alignment_to_num(self.single_mapping),
                                      single_alphabet_size=self.single_size, single_mapping=self.single_mapping,
                                      single_reverse_mapping=self.single_reverse, larger_alphabet_size=larger_size,
                                      larger_alphabet_mapping=larger_mapping,
                                      larger_alphabet_reverse_mapping=larger_reverse,
                                      single_to_larger_mapping=single_to_larger, pos_size=pos_size)
        mm_table.identify_matches_mismatches()
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=assign, position_specific=single,
                      pair_specific=pair, match_mismatch=True, output_dir=os.path.join(self.testing_dir, aln.query_id),
                      low_memory=low_mem)
        unique_dir = os.path.join(trace.out_dir, 'unique_node_data')
        rmtree(unique_dir, ignore_errors=True)
        trace.characterize_rank_groups(processes=processors, write_out_sub_aln=write_aln,
                                       write_out_freq_table=write_freq_table)
        visited = set()
        for rank in sorted(list(trace.assignments.keys()), reverse=True):
            for group in trace.assignments[rank]:
                node_name = trace.assignments[rank][group]['node'].name
                self.assertTrue(node_name in trace.unique_nodes)
                self.assertTrue('match' in trace.unique_nodes[node_name])
                self.assertTrue('mismatch' in trace.unique_nodes[node_name])
                if node_name not in visited:
                    sub_aln = aln.generate_sub_alignment(
                        sequence_ids=trace.assignments[rank][group]['terminals'])
                    if write_aln:
                        self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}.fa'.format(node_name))))
                    sub_aln_ind = [aln.seq_order.index(s) for s in sub_aln.seq_order]
                    possible_matches_mismatches = ((sub_aln.size**2) - sub_aln.size) / 2.0
                    expected_tables = {'match': FrequencyTable(alphabet_size=larger_size, mapping=larger_mapping,
                                                               reverse_mapping=larger_reverse,
                                                               seq_len=sub_aln.seq_length, pos_size=pos_size)}
                    expected_tables['match'].set_depth(possible_matches_mismatches)
                    expected_tables['mismatch'] = deepcopy(expected_tables['match'])
                    for p in expected_tables['match'].get_positions():
                        char_dict = {'match': {}, 'mismatch': {}}
                        for i in range(sub_aln.size):
                            s1 = sub_aln_ind[i]
                            for j in range(i + 1, sub_aln.size):
                                s2 = sub_aln_ind[j]
                                status, char = mm_table.get_status_and_character(pos=p, seq_ind1=s1, seq_ind2=s2)
                                if char not in char_dict[status]:
                                    char_dict[status][char] = 0
                                char_dict[status][char] += 1
                        for status in char_dict:
                            for char in char_dict[status]:
                                expected_tables[status]._increment_count(pos=p, char=char,
                                                                         amount=char_dict[status][char])
                    expected_tables['match'].finalize_table()
                    expected_tables['mismatch'].finalize_table()
                    for m in ['match', 'mismatch']:
                        m_table = trace.unique_nodes[node_name][m]
                        m_table = load_freq_table(freq_table=m_table, low_memory=low_mem)
                        m_table_mat = m_table.get_table()
                        expected_m_table_mat = expected_tables[m].get_table()
                        sparse_diff = m_table_mat - expected_m_table_mat
                        nonzero_check = sparse_diff.count_nonzero() > 0
                        if nonzero_check:
                            print(m_table.get_table())
                            print(expected_tables[m].get_table())
                            print(sparse_diff)
                            indices = np.nonzero(sparse_diff)
                            print(m_table.get_table()[indices])
                            print(expected_tables[m].get_table()[indices])
                            print(sparse_diff[indices])
                            print(node_name)
                            print(sub_aln.alignment)
                        self.assertFalse(nonzero_check)
                        if write_freq_table:
                            expected_table_path = os.path.join(unique_dir, '{}_{}_{}_freq_table.tsv'.format(
                                node_name, pos_type, m))
                            self.assertTrue(os.path.isfile(expected_table_path), 'Not found: {}'.format(expected_table_path))
                    visited.add(node_name)

    # def test3c_characterize_rank_groups(self):
    #     # Test characterizing both single and pair positions, small alignment, single processed
    #     self.evaluate_characterize_rank_groups_match_mismatch(
    #         aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
    #         single=True, pair=False, processors=1, low_mem=False, write_aln=False, write_freq_table=False)

    def test3d_characterize_rank_groups(self):
        # Test characterizing both single and pair positions, large alignment, single processed
        self.evaluate_characterize_rank_groups_match_mismatch(
            aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
            single=False, pair=True, processors=1, low_mem=False, write_aln=False, write_freq_table=False)

    # def test3e_characterize_rank_groups(self):
    #     # Test characterizing both single and pair positions, small alignment, single processed
    #     self.evaluate_characterize_rank_groups_match_mismatch(
    #         aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
    #         single=True, pair=False, processors=self.max_threads, low_mem=True, write_aln=True, write_freq_table=True)
    #
    # def test3f_characterize_rank_groups(self):
    #     # Test characterizing both single and pair positions, large alignment, single processed
    #     self.evaluate_characterize_rank_groups_match_mismatch(
    #         aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
    #         single=False, pair=True, processors=self.max_threads, low_mem=True, write_aln=True, write_freq_table=True)

    def evaluate_trace_pool_functions(self, aln, phylo_tree, assign, single, pair, metric, low_memory,
                                      out_dir, write_out_aln, write_out_freq_table):
        unique_dir = os.path.join(out_dir, 'unique_node_data')
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=assign, position_specific=single,
                      pair_specific=pair, match_mismatch=(metric == 'match_mismatch_entropy_angle'),
                      output_dir=os.path.join(self.testing_dir, aln.query_id), low_memory=low_memory)
        trace.characterize_rank_groups(processes=self.max_threads, write_out_sub_aln=write_out_aln,
                                       write_out_freq_table=write_out_freq_table)
        if single:
            pos_size = 1
        elif pair:
            pos_size = 2
        else:
            raise ValueError('Cannot evaluate if both single and pair are False.')
        scorer = PositionalScorer(seq_length=aln.seq_length, pos_size=pos_size, metric=metric)
        init_trace_groups(scorer=scorer, pos_specific=single, pair_specific=pair,
                          match_mismatch=(metric == 'match_mismatch_entropy_angle'),
                          u_dict=trace.unique_nodes, low_memory=low_memory, unique_dir=unique_dir)
        group_dict = {}
        for node_name in trace.unique_nodes:
            ret_node_name, ret_components = trace_groups(node_name=node_name)
            group_dict[ret_node_name] = ret_components
        self.assertEqual(len(group_dict.keys()), len(trace.unique_nodes.keys()))
        for node_name in group_dict:
            self.assertTrue('single_scores' in group_dict[node_name])
            if single:
                if trace.match_mismatch:
                    single_freq_table = {'match': load_freq_table(freq_table=trace.unique_nodes[node_name]['match'],
                                                                  low_memory=low_memory),
                                         'mismatch': load_freq_table(freq_table=trace.unique_nodes[node_name]['mismatch'],
                                                                     low_memory=low_memory)}
                else:
                    single_freq_table = load_freq_table(freq_table=trace.unique_nodes[node_name]['single'],
                                                        low_memory=low_memory)
                expected_scores = scorer.score_group(freq_table=single_freq_table)
                single_scores = load_numpy_array(mat=group_dict[node_name]['single_scores'], low_memory=low_memory)
                diff = single_scores - expected_scores
                if diff.any():
                    print(single_scores)
                    print(expected_scores)
                    print(diff)
                    indices = np.nonzero(diff)
                    print(single_scores[indices])
                    print(expected_scores[indices])
                    print(diff[indices])
                self.assertTrue(not diff.any())
            else:
                self.assertIsNone(group_dict[node_name]['single_scores'])
            self.assertTrue('pair_scores' in group_dict[node_name])
            if pair:
                if trace.match_mismatch:
                    pair_freq_table = {'match': load_freq_table(freq_table=trace.unique_nodes[node_name]['match'],
                                                                low_memory=low_memory),
                                       'mismatch': load_freq_table(freq_table=trace.unique_nodes[node_name]['mismatch'],
                                                                   low_memory=low_memory)}
                else:
                    pair_freq_table = load_freq_table(freq_table=trace.unique_nodes[node_name]['pair'],
                                                      low_memory=low_memory)
                expected_scores = scorer.score_group(freq_table=pair_freq_table)
                pair_scores = load_numpy_array(mat=group_dict[node_name]['pair_scores'], low_memory=low_memory)
                diff = pair_scores - expected_scores
                if diff.any():
                    print(pair_scores)
                    print(expected_scores)
                    print(diff)
                    indices = np.nonzero(diff)
                    print(pair_scores[indices])
                    print(expected_scores[indices])
                    print(diff[indices])
                self.assertTrue(not diff.any())
            else:
                self.assertIsNone(group_dict[node_name]['pair_scores'])
            trace.unique_nodes[node_name].update(group_dict[node_name])
        rank_dict = {}
        init_trace_ranks(scorer=scorer, pos_specific=single, pair_specific=pair, a_dict=assign,
                         u_dict=trace.unique_nodes, low_memory=low_memory, unique_dir=unique_dir)
        for rank in assign:
            ret_rank, ret_components = trace_ranks(rank=rank)
            rank_dict[ret_rank] = ret_components
        for rank in assign.keys():
            group_scores = []
            for group in sorted(assign[rank].keys(), reverse=True):
                node_name = assign[rank][group]['node'].name
                if single:
                    single_scores = load_numpy_array(mat=trace.unique_nodes[node_name]['single_scores'],
                                                     low_memory=low_memory)
                    group_scores.append(single_scores)
                elif pair:
                    pair_scores = load_numpy_array(mat=trace.unique_nodes[node_name]['pair_scores'],
                                                   low_memory=low_memory)
                    group_scores.append(pair_scores)
                else:
                    raise ValueError('Cannot evaluate if both single and pair are False.')
            group_scores = np.stack(group_scores, axis=0)
            expected_rank_scores = np.sum(group_scores, axis=0)
            if scorer.metric_type == 'integer':
                expected_rank_scores = (expected_rank_scores > 0) * 1.0
            else:
                expected_rank_scores = (1.0 / rank) * expected_rank_scores
            self.assertTrue(rank in rank_dict)
            self.assertTrue('single_ranks' in rank_dict[rank])
            if single:
                rank_scores = load_numpy_array(mat=rank_dict[rank]['single_ranks'], low_memory=low_memory)
                diff = np.abs(rank_scores - expected_rank_scores)
                not_passing = diff > 1E-15
                if not_passing.any():
                    print(rank)
                    print(rank_scores)
                    print(expected_rank_scores)
                    print(diff)
                    indices = np.nonzero(not_passing)
                    print(rank_scores[indices])
                    print(expected_rank_scores[indices])
                    print(diff[indices])
                self.assertTrue(not not_passing.any())
            else:
                self.assertIsNone(rank_dict[rank]['single_ranks'])
            self.assertTrue(rank in rank_dict)
            self.assertTrue('pair_ranks' in rank_dict[rank])
            if pair:
                expected_rank_scores = np.triu(expected_rank_scores, k=1)
                rank_scores = load_numpy_array(mat=rank_dict[rank]['pair_ranks'], low_memory=low_memory)
                diff = np.abs(rank_scores - expected_rank_scores)
                not_passing = diff > 1E-15
                if not_passing.any():
                    print(rank_dict[rank]['pair_ranks'])
                    print(expected_rank_scores)
                    print(diff)
                    indices = np.nonzero(not_passing)
                    print(rank_scores)
                    print(expected_rank_scores[indices])
                    print(diff[indices])
                self.assertTrue(not not_passing.any())
            else:
                self.assertIsNone(rank_dict[rank]['pair_ranks'])

    # def test4a_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the small alignment, single positions,
    #     # and the identity metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
    #         single=True, pair=False, metric='identity', low_memory=True, out_dir=self.out_small_dir,
    #         write_out_aln=False, write_out_freq_table=False)
    #
    # def test4b_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the small alignment, paired positions,
    #     # and the identity metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
    #         single=False, pair=True, metric='identity', low_memory=True, out_dir=self.out_small_dir,
    #         write_out_aln=False, write_out_freq_table=False)
    #
    # def test4c_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the large alignment, single positions,
    #     # and identity metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
    #         single=True, pair=False, metric='identity', low_memory=True, out_dir=self.out_large_dir,
    #         write_out_aln=False, write_out_freq_table=False)
    #
    # def test4d_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
    #     # and identity metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
    #         single=False, pair=True, metric='identity', low_memory=True, out_dir=self.out_large_dir,
    #         write_out_aln=False, write_out_freq_table=False)
    #
    # def test4e_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the small alignment, single positions,
    #     # and the plain entropy metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
    #         single=True, pair=False, metric='plain_entropy', low_memory=True, out_dir=self.out_small_dir,
    #         write_out_aln=False, write_out_freq_table=False)
    #
    # def test4f_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the small alignment, paired positions,
    #     # and the plain entropy metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
    #         single=False, pair=True, metric='plain_entropy', low_memory=True, out_dir=self.out_small_dir,
    #         write_out_aln=False, write_out_freq_table=False)
    #
    # def test4g_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the large alignment, single positions,
    #     # and plain entropy metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
    #         single=True, pair=False, metric='plain_entropy', low_memory=True, out_dir=self.out_large_dir,
    #         write_out_aln=False, write_out_freq_table=False)
    #
    # def test4h_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
    #     # and plain entropy metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
    #         single=False, pair=True, metric='plain_entropy', low_memory=True, out_dir=self.out_large_dir,
    #         write_out_aln=False, write_out_freq_table=False)
    #
    # def test4i_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the small alignment, paired positions,
    #     # and the mutual information metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
    #         single=False, pair=True, metric='mutual_information', low_memory=True, out_dir=self.out_small_dir,
    #         write_out_aln=False, write_out_freq_table=False)
    #
    # def test4j_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
    #     # and mutual information metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
    #         single=False, pair=True, metric='mutual_information', low_memory=True, out_dir=self.out_large_dir,
    #         write_out_aln=False, write_out_freq_table=False)
    #
    # def test4k_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the small alignment, paired positions,
    #     # and the normalized mutual information metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
    #         single=False, pair=True, metric='normalized_mutual_information', low_memory=True,
    #         out_dir=self.out_small_dir, write_out_aln=False, write_out_freq_table=False)
    #
    # def test4l_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
    #     # and normalized mutual information metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
    #         single=False, pair=True, metric='normalized_mutual_information', low_memory=True,
    #         out_dir=self.out_large_dir, write_out_aln=False, write_out_freq_table=False)
    #
    # def test4m_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the small alignment, paired positions,
    #     # and the average product corrected mutual information (MIp) metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
    #         single=False, pair=True, metric='average_product_corrected_mutual_information', low_memory=True,
    #         out_dir=self.out_small_dir, write_out_aln=False, write_out_freq_table=False)
    #
    # def test4n_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
    #     # and average product corrected mutual information (MIp) metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
    #         single=False, pair=True, metric='average_product_corrected_mutual_information', low_memory=True,
    #         out_dir=self.out_large_dir, write_out_aln=False, write_out_freq_table=False)
    #
    # def test4o_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
    #     # and average product corrected mutual information (MIp) metric (all ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
    #         single=False, pair=True, metric='filtered_average_product_corrected_mutual_information', low_memory=True,
    #         out_dir=self.out_small_dir, write_out_aln=False, write_out_freq_table=False)
    #
    # def test4p_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
    #     # and average product corrected mutual information (MIp) metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
    #         single=False, pair=True, metric='filtered_average_product_corrected_mutual_information', low_memory=True,
    #         out_dir=self.out_large_dir, write_out_aln=False, write_out_freq_table=False)

    # def test4q_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
    #     # and average product corrected mutual information (MIp) metric (all ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_custom_small,
    #         single=False, pair=True, metric='match_mismatch_entropy_angle', low_memory=True,
    #         out_dir=self.out_small_dir, write_out_aln=False, write_out_freq_table=False)
    #
    # def test4r_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the large alignment, paired positions,
    #     # and average product corrected mutual information (MIp) metric (custom ranks)
    #     self.evaluate_trace_pool_functions(
    #         aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_custom_large,
    #         single=False, pair=True, metric='match_mismatch_entropy_angle', low_memory=True,
    #         out_dir=self.out_large_dir, write_out_aln=False, write_out_freq_table=False)

    def evaluate_trace(self, aln, phylo_tree, assignments, single, pair, metric, num_proc, out_dir,
                       match_mismatch=False, gap_correction=None, low_mem=True):
        if single:
            pos_size = 1
            expected_ranks = np.ones(aln.seq_length)
        elif pair:
            pos_size = 2
            expected_ranks = np.ones((aln.seq_length, aln.seq_length))
        else:
            pos_size = None
            expected_ranks = None
        trace_obj = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=assignments,
                          position_specific=single, pair_specific=pair, match_mismatch=match_mismatch,
                          low_memory=low_mem, output_dir=out_dir)
        trace_obj.characterize_rank_groups(processes=num_proc, write_out_sub_aln=False, write_out_freq_table=False)
        scorer = PositionalScorer(seq_length=aln.seq_length, pos_size=pos_size, metric=metric)
        rank_array, score_array, coverage_array = trace_obj.trace(scorer=scorer, processes=num_proc,
                                                                  gap_correction=gap_correction)
        # Simple check against ranking and coverage is to find the min and max values
        if pos_size == 2:
            self.assertFalse(np.tril(rank_array).any())
            self.assertFalse(np.tril(score_array).any())
            self.assertFalse(np.tril(coverage_array).any())
        if scorer.rank_type == 'min':
            if pos_size == 1:
                min_score = np.min(score_array)
                min_rank = np.min(rank_array)
                max_coverage = np.max(coverage_array)
                max_score = np.max(score_array)
                max_rank = np.max(rank_array)
                min_coverage = np.min(coverage_array)
            else:
                min_score = np.min(score_array[np.triu_indices(aln.seq_length, k=1)])
                min_rank = np.min(rank_array[np.triu_indices(aln.seq_length, k=1)])
                max_coverage = np.max(coverage_array[np.triu_indices(aln.seq_length, k=1)])
                max_score = np.max(score_array[np.triu_indices(aln.seq_length, k=1)])
                max_rank = np.max(rank_array[np.triu_indices(aln.seq_length, k=1)])
                min_coverage = np.min(coverage_array[np.triu_indices(aln.seq_length, k=1)])
            min_mask = (score_array == min_score) * 1
            rank_mask = (rank_array == min_rank) * 1
            diff_min_ranks = min_mask - rank_mask
            if diff_min_ranks.any():
                print(min_score)
                print(min_rank)
                print(np.sum(min_mask))
                print(np.sum(rank_mask))
            self.assertFalse(diff_min_ranks.any())
            cov_mask = coverage_array == min_coverage
            diff_min_cov = min_mask - cov_mask
            if diff_min_cov.any():
                print(min_score)
                print(min_coverage)
                print(np.sum(min_mask))
                print(np.sum(cov_mask))
            self.assertFalse(diff_min_cov.any())
            max_mask = score_array == max_score
            rank_mask2 = rank_array == max_rank
            diff_max_ranks = max_mask ^ rank_mask2
            if diff_max_ranks.any():
                print(max_score)
                print(max_rank)
                print(np.sum(max_mask))
                print(np.sum(rank_mask2))
            self.assertFalse(diff_max_ranks.any())
            cov_mask2 = coverage_array == max_coverage
            diff_max_cov = max_mask ^ cov_mask2
            if diff_max_cov.any():
                print(max_score)
                print(max_coverage)
                print(np.sum(max_mask))
                print(np.sum(cov_mask2))
            self.assertFalse(diff_min_cov.any())
        else:
            if pos_size == 1:
                max_score = np.max(score_array)
                max_rank = np.max(rank_array)
                max_coverage = np.max(coverage_array)
                min_score = np.min(score_array)
                min_rank = np.min(rank_array)
                min_coverage = np.min(coverage_array)
                min_mask = score_array == min_score
                rank_mask = rank_array == max_rank
                cov_mask = coverage_array == max_coverage
                max_mask = score_array == max_score
                rank_mask2 = rank_array == min_rank
                cov_mask2 = coverage_array == min_coverage
            else:
                max_score = np.max(score_array[np.triu_indices(aln.seq_length, k=1)])
                max_rank = np.max(rank_array[np.triu_indices(aln.seq_length, k=1)])
                max_coverage = np.max(coverage_array[np.triu_indices(aln.seq_length, k=1)])
                min_score = np.min(score_array[np.triu_indices(aln.seq_length, k=1)])
                min_rank = np.min(rank_array[np.triu_indices(aln.seq_length, k=1)])
                min_coverage = np.min(coverage_array[np.triu_indices(aln.seq_length, k=1)])
                min_mask = np.triu(score_array == min_score, k=1)
                rank_mask = np.triu(rank_array == max_rank, k=1)
                cov_mask = np.triu(coverage_array == max_coverage, k=1)
                max_mask = np.triu(score_array == max_score, k=1)
                rank_mask2 = np.triu(rank_array == min_rank, k=1)
                cov_mask2 = np.triu(coverage_array == min_coverage, k=1)
            diff_min_ranks = min_mask ^ rank_mask
            if diff_min_ranks.any():
                print(min_score)
                print(min_rank)
                print(np.sum(min_mask))
                print(np.sum(rank_mask))
            self.assertFalse(diff_min_ranks.any())
            diff_min_cov = min_mask ^ cov_mask
            if diff_min_cov.any():
                print(min_score)
                print(max_coverage)
                print(np.sum(min_mask))
                print(np.sum(cov_mask))
            self.assertFalse(diff_min_cov.any())
            diff_max_ranks = max_mask ^ rank_mask2
            if diff_max_ranks.any():
                print(max_score)
                print(max_rank)
                print(np.sum(max_mask))
                print(np.sum(rank_mask2))
            self.assertFalse(diff_max_ranks.any())
            diff_max_cov = max_mask ^ cov_mask2
            if diff_max_cov.any():
                print(max_score)
                print(max_coverage)
                print(np.sum(max_mask))
                print(np.sum(cov_mask2))
            self.assertFalse(diff_min_cov.any())
        unique_scores = {}
        for rank in sorted(assignments.keys(), reverse=True):
            group_scores = []
            for group in assignments[rank].keys():
                node_name = assignments[rank][group]['node'].name
                if node_name not in unique_scores:
                    if match_mismatch:
                        mm_dict = {'match': load_freq_table(trace_obj.unique_nodes[node_name]['match'],
                                                            low_memory=low_mem),
                                   'mismatch': load_freq_table(trace_obj.unique_nodes[node_name]['mismatch'],
                                                               low_memory=low_mem)}
                        group_score = scorer.score_group(mm_dict)
                    elif pair:
                        group_score = scorer.score_group(load_freq_table(trace_obj.unique_nodes[node_name]['pair'],
                                                                         low_memory=low_mem))
                    elif single:
                        group_score = scorer.score_group(load_freq_table(trace_obj.unique_nodes[node_name]['single'],
                                                                         low_memory=low_mem))
                    else:
                        raise ValueError('Either pair or single must be true for this test.')
                    unique_scores[node_name] = group_score
                else:
                    group_score = unique_scores[node_name]
                group_scores.append(group_score)
            group_scores = np.stack(group_scores, axis=0)
            if metric == 'identity':
                weight = 1.0
            else:
                weight = 1.0 / rank
            rank_scores = weight * np.sum(group_scores, axis=0)
            if metric == 'identity':
                rank_scores = rank_scores > 0 * 1
            if single:
                curr_rank = load_numpy_array(trace_obj.rank_scores[rank]['single_ranks'], low_memory=low_mem)
            elif pair:
                curr_rank = load_numpy_array(trace_obj.rank_scores[rank]['pair_ranks'], low_memory=low_mem)
            else:
                raise ValueError('Either pair or single must be true for this test.')
            diff_rank = curr_rank - rank_scores
            not_passing_rank = diff_rank > 1E-12
            if not_passing_rank.any():
                print(curr_rank)
                print(rank_scores)
                print(diff_rank)
                indices_rank = np.nonzero(not_passing_rank)
                print(curr_rank[indices_rank])
                print(rank_scores[indices_rank])
                print(diff_rank[indices_rank])
            self.assertFalse(not_passing_rank.any())
            expected_ranks += rank_scores
        diff_ranks = score_array - expected_ranks
        not_passing = diff_ranks > 1E-12
        if not_passing.any():
            print(score_array)
            print(expected_ranks)
            print(diff_ranks)
            indices = np.nonzero(not_passing)
            print(score_array[indices])
            print(expected_ranks[indices])
            print(diff_ranks[indices])
        self.assertFalse(not_passing.any())

    # def test5a_trace(self):
    #     # Perform identity trace on single positions only for the small alignment (custom ranks)
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    #     self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
    #                         assignments=self.assignments_custom_small, single=True, pair=False, metric='identity',
    #                         low_mem=True, num_proc=self.max_threads, out_dir=self.out_small_dir)
    #
    # def test5b_trace(self):
    #     # Perform identity trace on single positions only for the large alignment (custom ranks)
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    #     self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
    #                         assignments=self.assignments_custom_large, single=True, pair=False, metric='identity',
    #                         low_mem=True, num_proc=self.max_threads, out_dir=self.out_large_dir)

    def evaluate_integer_et_comparison(self, p_id, msf_aln, fa_aln, low_mem, out_dir):
        if os.path.isdir(out_dir):
            rmtree(out_dir)
        os.mkdir(out_dir)
        et_mip_obj = ETMIPWrapper(query=p_id, aln_file=fa_aln.file_name, out_dir=out_dir)
        et_mip_obj.calculate_scores(method='intET', delete_files=False)
        trace_small = Trace(alignment=fa_aln, phylo_tree=et_mip_obj.tree,
                            group_assignments=et_mip_obj.rank_group_assignments, position_specific=True,
                            pair_specific=False, output_dir=out_dir, low_memory=low_mem)
        trace_small.characterize_rank_groups(processes=self.max_threads, write_out_sub_aln=False,
                                             write_out_freq_table=False)
        scorer = PositionalScorer(seq_length=fa_aln.seq_length, pos_size=1, metric='identity')
        rank_ids, score_ids, coverage_ids = trace_small.trace(scorer=scorer, gap_correction=None,
                                                              processes=self.max_threads)
        diff_ranks = score_ids - et_mip_obj.scores
        if diff_ranks.any():
            print(score_ids)
            print(et_mip_obj.scores)
            print(diff_ranks)
            indices = np.nonzero(diff_ranks)
            print(score_ids[indices])
            print(et_mip_obj.scores[indices])
            print(diff_ranks[indices])
        self.assertFalse(diff_ranks.any())
        diff_coverage = coverage_ids - et_mip_obj.coverages
        not_passing = np.abs(diff_coverage) > 1e-2
        if not_passing.any():
            print(coverage_ids)
            print(et_mip_obj.coverage)
            print(diff_coverage)
            indices = np.nonzero(diff_coverage)
            print(coverage_ids[indices])
            print(et_mip_obj.coverage[indices])
            print(diff_coverage[indices])
        self.assertFalse(not_passing.any())
        rounded_coverages = np.round(coverage_ids, decimals=3)
        diff_coverages2 = rounded_coverages - et_mip_obj.coverages
        not_passing2 = diff_coverages2 > 1E-15
        if not_passing2.any():
            print(rounded_coverages)
            print(et_mip_obj.coverage)
            print(diff_coverages2)
            indices = np.nonzero(not_passing2)
            print(rounded_coverages[indices])
            print(et_mip_obj.coverage[indices])
            print(diff_coverages2[indices])
        self.assertFalse(not_passing2.any())

    # def test5c_trace(self):
    #     # Compare the results of identity trace over single positions between this implementation and the WETC
    #     # implementation for the small alignment.
    #     self.evaluate_integer_et_comparison(p_id=self.small_structure_id, msf_aln=self.query_aln_msf_small,
    #                                         fa_aln=self.query_aln_fa_small, low_mem=False, out_dir=self.out_small_dir)
    #
    # def test5d_trace(self):
    #     # Compare the results of identity trace over single positions between this implementation and the WETC
    #     # implementation for the large alignment.
    #     self.evaluate_integer_et_comparison(p_id=self.large_structure_id, msf_aln=self.query_aln_msf_large,
    #                                         fa_aln=self.query_aln_fa_large, low_mem=True, out_dir=self.out_large_dir)
    #
    # def test5e_trace(self):
    #     # Perform identity trace on pairs of positions only for the small alignment (custom ranks)
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    #     self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
    #                         assignments=self.assignments_custom_small, single=False, pair=True, metric='identity',
    #                         low_mem=True, num_proc=self.max_threads, out_dir=self.out_small_dir)
    #
    # def test5f_trace(self):
    #     # Perform identity trace on pairs of positions only for the large alignment (custom ranks)
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    #     self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
    #                         assignments=self.assignments_custom_large, single=False, pair=True, metric='identity',
    #                         low_mem=True, num_proc=self.max_threads, out_dir=self.out_large_dir)
    #
    # def test5g_trace(self):
    #     # Perform plain entropy trace on single positions only for the small alignment (custom ranks)
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    #     self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
    #                         assignments=self.assignments_custom_small, single=True, pair=False, metric='plain_entropy',
    #                         low_mem=True, num_proc=self.max_threads, out_dir=self.out_small_dir)
    #
    # def test5h_trace(self):
    #     # Perform plain entropy trace on single positions only for the large alignment (custom ranks)
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    #     self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
    #                         assignments=self.assignments_custom_large, single=True, pair=False, metric='plain_entropy',
    #                         low_mem=True, num_proc=self.max_threads, out_dir=self.out_large_dir)

    def evaluate_real_value_et_comparison(self, p_id, msf_aln, fa_aln, low_mem, out_dir):
        if os.path.isdir(out_dir):
            rmtree(out_dir)
        os.mkdir(out_dir)
        et_mip_obj = ETMIPWrapper(query=p_id, aln_file=fa_aln.file_name, out_dir=out_dir)
        et_mip_obj.calculate_scores(method='rvET', delete_files=False)
        trace_small = Trace(alignment=fa_aln, phylo_tree=et_mip_obj.tree,
                            group_assignments=et_mip_obj.rank_group_assignments, position_specific=True,
                            pair_specific=False, output_dir=out_dir, low_memory=low_mem)
        trace_small.characterize_rank_groups(processes=self.max_threads, write_out_sub_aln=False,
                                             write_out_freq_table=False)
        scorer = PositionalScorer(seq_length=fa_aln.seq_length, pos_size=1, metric='plain_entropy')
        rank_entropies, score_entropies, coverage_entropies = trace_small.trace(scorer=scorer, gap_correction=0.6,
                                                                                processes=self.max_threads)
        diff_ranks = score_entropies - et_mip_obj.scores
        not_passing = np.abs(diff_ranks) > 1e-2
        if not_passing.any():
            print(score_entropies)
            print(et_mip_obj.scores)
            print(diff_ranks)
            indices = np.nonzero(diff_ranks)
            print(score_entropies[indices])
            print(et_mip_obj.scores[indices])
            print(diff_ranks[indices])
        self.assertFalse(not_passing.any())
        rounded_entropies = np.round(score_entropies, decimals=2)
        diff_ranks2 = rounded_entropies - et_mip_obj.scores
        if diff_ranks2.any():
            print(rounded_entropies)
            print(et_mip_obj.scores)
            print(diff_ranks2)
            indices = np.nonzero(diff_ranks2)
            print(rounded_entropies[indices])
            print(et_mip_obj.scores[indices])
            print(diff_ranks2[indices])
        self.assertFalse(diff_ranks2.any())
        diff_coverage = coverage_entropies - et_mip_obj.coverages
        not_passing = np.abs(diff_coverage) > 1e-2
        if not_passing.any():
            print(coverage_entropies)
            print(et_mip_obj.coverage)
            print(diff_coverage)
            indices = np.nonzero(diff_coverage)
            print(coverage_entropies[indices])
            print(et_mip_obj.coverage[indices])
            print(diff_coverage[indices])
        self.assertFalse(not_passing.any())
        rounded_coverages = np.round(coverage_entropies, decimals=3)
        diff_coverages2 = rounded_coverages - et_mip_obj.coverages
        not_passing2 = diff_coverages2 > 1E-15
        if not_passing2.any():
            print(rounded_coverages)
            print(et_mip_obj.coverage)
            print(diff_coverages2)
            indices = np.nonzero(not_passing2)
            print(rounded_coverages[indices])
            print(et_mip_obj.coverage[indices])
            print(diff_coverages2[indices])
        self.assertFalse(not_passing2.any())

    # def test5i_trace(self):
    #     # Compare the results of plain entropy trace over single positions between this implementation and the WETC
    #     # implementation for the small alignment.
    #     self.evaluate_real_value_et_comparison(p_id=self.small_structure_id, msf_aln=self.query_aln_msf_small,
    #                                            fa_aln=self.query_aln_fa_small, low_mem=False,
    #                                            out_dir=self.out_small_dir)
    #
    # def test5j_trace(self):
    #     # Compare the results of identity trace over single positions between this implementation and the WETC
    #     # implementation for the large alignment.
    #     self.evaluate_real_value_et_comparison(p_id=self.large_structure_id, msf_aln=self.query_aln_msf_large,
    #                                            fa_aln=self.query_aln_fa_large, low_mem=True,
    #                                            out_dir=self.out_large_dir)
    #
    # def test5k_trace(self):
    #     # Perform plain entropy trace on pairs of positions only for the small alignment (custom ranks)
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    #     self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
    #                         assignments=self.assignments_custom_small, single=False, pair=True, metric='plain_entropy',
    #                         low_mem=True, num_proc=self.max_threads, out_dir=self.out_small_dir)
    #
    # def test5l_trace(self):
    #     # Perform plain entropy trace on pairs of positions only for the large alignment (custom ranks)
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    #     self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
    #                         assignments=self.assignments_custom_large, single=False, pair=True, metric='plain_entropy',
    #                         low_mem=True, num_proc=self.max_threads, out_dir=self.out_large_dir)
    #
    # def test5m_trace(self):
    #     # Perform mutual information trace on pairs of positions only for the small alignment (custom ranks)
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    #     self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
    #                         assignments=self.assignments_custom_small, single=False, pair=True,
    #                         metric='mutual_information', low_mem=True, num_proc=self.max_threads,
    #                         out_dir=self.out_small_dir)
    #
    # def test5n_trace(self):
    #     # Perform mutual information trace on pairs of positions only for the large alignment (custom ranks)
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    #     self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
    #                         assignments=self.assignments_custom_large, single=False, pair=True,
    #                         metric='mutual_information', low_mem=True, num_proc=self.max_threads,
    #                         out_dir=self.out_large_dir)
    #
    # def test5o_trace(self):
    #     # Perform normalize mutual information trace on pairs of positions only for the small alignment (custom ranks)
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    #     self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
    #                         assignments=self.assignments_custom_small, single=False, pair=True,
    #                         metric='normalized_mutual_information', low_mem=True, num_proc=self.max_threads,
    #                         out_dir=self.out_small_dir)
    #
    # def test5p_trace(self):
    #     # Perform normalize mutual information trace on pairs of positions only for the large alignment (custom ranks)
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    #     self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
    #                         assignments=self.assignments_custom_large, single=False, pair=True,
    #                         metric='normalized_mutual_information', low_mem=True, num_proc=self.max_threads,
    #                         out_dir=self.out_large_dir)
    #
    # def test5q_trace(self):
    #     # Perform average product corrected mutual information (MIp) trace on pairs of positions only for the small
    #     # alignment (custom ranks). Assume scoring happens correctly since it has been tested above and ensure that
    #     # expected ranks are achieved.
    #     self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
    #                         assignments=self.assignments_custom_small, single=False, pair=True,
    #                         metric='average_product_corrected_mutual_information',
    #                         low_mem=True, num_proc=self.max_threads, out_dir=self.out_small_dir)
    #
    # def test5r_trace(self):
    #     # Perform average product corrected mutual information (MIp) trace on pairs of positions only for the large
    #     # alignment (custom ranks). Assume scoring happens correctly since it has been tested above and ensure that
    #     # expected ranks are achieved.
    #     self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
    #                         assignments=self.assignments_custom_large, single=False, pair=True,
    #                         metric='average_product_corrected_mutual_information',
    #                         low_mem=True, num_proc=self.max_threads, out_dir=self.out_large_dir)
    #
    # def test5s_trace(self):
    #     # Perform average product corrected mutual information (MIp) trace on pairs of positions only for the small
    #     # alignment (custom ranks). Assume scoring happens correctly since it has been tested above and ensure that
    #     # expected ranks are achieved.
    #     self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
    #                         assignments=self.assignments_custom_small, single=False, pair=True,
    #                         metric='filtered_average_product_corrected_mutual_information',
    #                         low_mem=True, num_proc=self.max_threads, out_dir=self.out_small_dir)
    #
    # def test5t_trace(self):
    #     # Perform average product corrected mutual information (MIp) trace on pairs of positions only for the large
    #     # alignment (custom ranks). Assume scoring happens correctly since it has been tested above and ensure that
    #     # expected ranks are achieved.
    #     self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
    #                         assignments=self.assignments_custom_large, single=False, pair=True,
    #                         metric='filtered_average_product_corrected_mutual_information',
    #                         low_mem=True, num_proc=self.max_threads, out_dir=self.out_large_dir)

    def evaluate_mip_et_comparison(self, p_id, fa_aln, low_mem, out_dir):
        rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        filtered_fa_fn = os.path.join(out_dir, '{}_filtered_aln.fa'.format(p_id))
        if os.path.isfile(filtered_fa_fn):
            char_filtered_fa_aln = SeqAlignment(file_name=filtered_fa_fn, query_id=p_id)
            char_filtered_fa_aln.import_alignment()
        else:
            curr_fa_aln = SeqAlignment(file_name=fa_aln.file_name, query_id=p_id)
            curr_fa_aln.import_alignment()
            curr_fa_aln.alphabet = Gapped(IUPACProtein())
            char_filtered_fa_aln = curr_fa_aln.remove_bad_sequences()
            char_filtered_fa_aln.write_out_alignment(file_name=filtered_fa_fn)
            char_filtered_fa_aln.file_name = filtered_fa_fn
        et_mip_obj = ETMIPWrapper(query=p_id, aln_file=filtered_fa_fn, out_dir=out_dir)
        et_mip_obj.convert_alignment()
        et_mip_obj.calculate_scores(method='ET-MIp', delete_files=False)
        gap_filtered_aln = char_filtered_fa_aln.remove_gaps()
        gap_filtered_aln.alphabet = FullIUPACProtein()
        trace_mip = Trace(alignment=gap_filtered_aln, phylo_tree=et_mip_obj.tree,
                          group_assignments=et_mip_obj.rank_group_assignments, position_specific=False,
                          pair_specific=True, output_dir=out_dir, low_memory=low_mem)
        trace_mip.characterize_rank_groups(processes=self.max_threads, write_out_sub_aln=False,
                                           write_out_freq_table=False)
        scorer_mip = PositionalScorer(seq_length=gap_filtered_aln.seq_length, pos_size=2,
                                      metric='filtered_average_product_corrected_mutual_information')
        rank_mips, score_mips, coverage_mips = trace_mip.trace(scorer=scorer_mip, gap_correction=None,
                                                               processes=self.max_threads)
        diff_ranks = score_mips - et_mip_obj.scores
        not_passing = np.abs(diff_ranks) > 1e-3
        if not_passing.any():
            print(score_mips)
            print(et_mip_obj.scores)
            print(diff_ranks)
            indices = np.nonzero(not_passing)
            print(score_mips[indices])
            print(et_mip_obj.scores[indices])
            print(diff_ranks[indices])
            print(score_mips[indices][0])
            print(et_mip_obj.scores[indices][0])
            print(diff_ranks[indices][0])
        self.assertFalse(not_passing.any())
        rounded_scores = np.round(score_mips, decimals=3)
        diff_ranks2 = rounded_scores - et_mip_obj.scores
        not_passing_rounded = np.abs(diff_ranks2) > 1e-15
        if not_passing_rounded.any():
            print(rounded_scores)
            print(et_mip_obj.scores)
            print(diff_ranks2)
            indices = np.nonzero(not_passing_rounded)
            print(rounded_scores[indices])
            print(et_mip_obj.scores[indices])
            print(diff_ranks2[indices])
        self.assertFalse(not_passing_rounded.any())
        diff_coverages = coverage_mips - et_mip_obj.coverages
        not_passing = np.abs(diff_coverages) > 1E-3
        if not_passing.any():
            print(coverage_mips)
            print(et_mip_obj.coverages)
            print(diff_coverages)
            indices = np.nonzero(not_passing)
            for i in range(len(indices[0])):
                print(indices[0][i], indices[1][i], et_mip_obj.coverages[indices[0][i], indices[1][i]],
                      coverage_mips[indices[0][i], indices[1][i]], diff_coverages[indices[0][i], indices[1][i]],
                      1e-2, np.abs(diff_coverages[indices[0][i], indices[1][i]]) > 1e-2)
            print(score_mips[indices])
            print(rank_mips[indices])
            print(np.sum(not_passing))
            print(np.nonzero(not_passing))
            self.assertLessEqual(np.sum(not_passing), np.ceil(0.01 * np.sum(range(fa_aln.seq_length - 1))))
        else:
            self.assertFalse(not_passing.any())
        rmtree(out_dir)

    # def test5u_trace(self):
    #     # Compare the results of average product corrected mutual information over pairs of positions between this
    #     # implementation and the WETC implementation for the small alignment.
    #     self.evaluate_mip_et_comparison(p_id=self.small_structure_id, fa_aln=self.query_aln_fa_small, low_mem=False,
    #                                     out_dir=self.out_small_dir)
    #
    # def test5v_trace(self):
    #     # Compare the results of average product corrected mutual information over pairs of positions between this
    #     # implementation and the WETC implementation for the large alignment.
    #     self.evaluate_mip_et_comparison(p_id=self.large_structure_id, fa_aln=self.query_aln_fa_large, low_mem=True,
    #                                     out_dir=self.out_large_dir)

    # def test5w_trace(self):
    #     # Test the small alignment for the computation of angles between the match and mismatch entropy but only
    #     # considering a subset of the rank/groups.
    #     self.evaluate_trace(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
    #                         assignments=self.assignments_custom_small, single=False, pair=True,
    #                         metric='match_mismatch_entropy_angle', num_proc=self.max_threads,
    #                         out_dir=self.out_small_dir, match_mismatch=True, gap_correction=None, low_mem=True)

    # def test5x_trace(self):
    #     # Test the large alignment for the computation of angles between the match and mismatch entropy but only
    #     # considering a subset of the rank/groups.
    #     self.evaluate_trace(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
    #                         assignments=self.assignments_custom_large, single=False, pair=True,
    #                         metric='match_mismatch_entropy_angle', num_proc=self.max_threads,
    #                         out_dir=self.out_large_dir, match_mismatch=True, gap_correction=None, low_mem=True)


if __name__ == '__main__':
    unittest.main()
