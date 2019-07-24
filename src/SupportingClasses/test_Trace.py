"""
Created on July 11, 2019

@author: Daniel Konecki
"""
import os
import numpy as np
from copy import deepcopy
from shutil import rmtree
from multiprocessing import Lock, Manager, Queue
from test_Base import TestBase
from ETMIPWrapper import ETMIPWrapper
from SeqAlignment import SeqAlignment
from PositionalScorer import PositionalScorer
from PhylogeneticTree import PhylogeneticTree
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from Trace import (Trace, init_characterization_pool, characterization, init_trace_groups, trace_groups,
                   init_trace_ranks, trace_ranks, save_freq_table, load_freq_table)


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
        cls.query_aln_fa_large = SeqAlignment(
            file_name=cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln'],
            query_id=cls.large_structure_id)
        cls.query_aln_fa_large.import_alignment()
        cls.phylo_tree_large = PhylogeneticTree()
        calc = AlignmentDistanceCalculator()
        cls.phylo_tree_large.construct_tree(dm=calc.get_distance(cls.query_aln_fa_large.alignment))
        cls.assignments_large = cls.phylo_tree_large.assign_group_rank()
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

    # @classmethod
    # def tearDownClass(cls):
    #     rmtree(cls.out_small_dir)
    #     rmtree(cls.out_large_dir)

    def test1a_init(self):
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                            group_assignments=self.assignments_small, position_specific=True, pair_specific=True,
                            output_dir=self.out_small_dir, low_mem=False)
        self.assertEqual(trace_small.aln.file_name, self.query_aln_fa_small.file_name)
        self.assertEqual(trace_small.aln.query_id, self.query_aln_fa_small.query_id)
        for s in range(trace_small.aln.size):
            self.assertEqual(str(trace_small.aln.alignment[s].seq), str(self.query_aln_fa_small.alignment[s].seq))
        self.assertEqual(trace_small.aln.seq_order, self.query_aln_fa_small.seq_order)
        self.assertEqual(str(trace_small.aln.query_sequence), str(self.query_aln_fa_small.query_sequence))
        self.assertEqual(trace_small.aln.seq_length, self.query_aln_fa_small.seq_length)
        self.assertEqual(trace_small.aln.size, self.query_aln_fa_small.size)
        self.assertEqual(trace_small.aln.marked, self.query_aln_fa_small.marked)
        self.assertEqual(trace_small.aln.polymer_type, self.query_aln_fa_small.polymer_type)
        self.assertTrue(isinstance(trace_small.aln.alphabet, type(self.query_aln_fa_small.alphabet)))
        self.assertEqual(len(trace_small.aln.alphabet.letters), len(self.query_aln_fa_small.alphabet.letters))
        for char in trace_small.aln.alphabet.letters:
            self.assertTrue(char in self.query_aln_fa_small.alphabet.letters)
        self.assertEqual(trace_small.phylo_tree, self.phylo_tree_small)
        self.assertEqual(trace_small.assignments, self.assignments_small)
        self.assertIsNone(trace_small.unique_nodes)
        self.assertEqual(trace_small.pos_specific, True)
        self.assertEqual(trace_small.pair_specific, True)
        self.assertEqual(trace_small.out_dir, self.out_small_dir)
        self.assertFalse(trace_small.low_memory)

    def test1b_init(self):
        trace_large = Trace(alignment=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
                            group_assignments=self.assignments_large, position_specific=True, pair_specific=True,
                            output_dir=self.out_large_dir, low_mem=True)
        self.assertEqual(trace_large.aln.file_name, self.query_aln_fa_large.file_name)
        self.assertEqual(trace_large.aln.query_id, self.query_aln_fa_large.query_id)
        for s in range(trace_large.aln.size):
            self.assertEqual(str(trace_large.aln.alignment[s].seq), str(self.query_aln_fa_large.alignment[s].seq))
        self.assertEqual(trace_large.aln.seq_order, self.query_aln_fa_large.seq_order)
        self.assertEqual(str(trace_large.aln.query_sequence), str(self.query_aln_fa_large.query_sequence))
        self.assertEqual(trace_large.aln.seq_length, self.query_aln_fa_large.seq_length)
        self.assertEqual(trace_large.aln.size, self.query_aln_fa_large.size)
        self.assertEqual(trace_large.aln.marked, self.query_aln_fa_large.marked)
        self.assertEqual(trace_large.aln.polymer_type, self.query_aln_fa_large.polymer_type)
        self.assertTrue(isinstance(trace_large.aln.alphabet, type(self.query_aln_fa_large.alphabet)))
        self.assertEqual(len(trace_large.aln.alphabet.letters), len(self.query_aln_fa_large.alphabet.letters))
        for char in trace_large.aln.alphabet.letters:
            self.assertTrue(char in self.query_aln_fa_large.alphabet.letters)
        self.assertEqual(trace_large.phylo_tree, self.phylo_tree_large)
        self.assertEqual(trace_large.assignments, self.assignments_large)
        self.assertIsNone(trace_large.unique_nodes)
        self.assertEqual(trace_large.pos_specific, True)
        self.assertEqual(trace_large.pair_specific, True)
        self.assertEqual(trace_large.out_dir, self.out_large_dir)
        self.assertTrue(trace_large.low_memory)

    ####################################################################################################################

    def evaluate_characterize_rank_groups_pooling_functions(self, p_id, aln, phylo_tree, out_dir, low_mem):
        unique_dir = os.path.join(out_dir, 'unique_node_data')
        if not os.path.isdir(unique_dir):
            os.makedirs(unique_dir)
        test_queue = Queue(maxsize=3)
        # Retrieve 3 node tree consisting of query node, its sibling, and its parent.
        query_node = [x for x in phylo_tree.tree.get_terminals() if x.name == p_id][0]
        parent_node = phylo_tree.tree.get_path(query_node)[-2]
        query_neighbor_node = parent_node.clades[0] if parent_node.clades[1] == query_node else parent_node.clades[1]
        test_queue.put_nowait(query_node)
        test_queue.put_nowait(query_neighbor_node)
        test_queue.put_nowait(parent_node)
        # Test the functions
        test_manager = Manager()
        test_dict = test_manager.dict()
        init_characterization_pool(alignment=aln, pos_specific=True, pair_specific=True, queue=test_queue,
                                   sharable_dict=test_dict, unique_dir=unique_dir, low_memory=low_mem)
        characterization(processor=1)
        test_dict = dict(test_dict)
        self.assertEqual(len(test_dict), 3)
        sub_aln1 = aln.generate_sub_alignment(sequence_ids=[x.name for x in parent_node.get_terminals()])
        single_table1, pair_table1 = sub_aln1.characterize_positions()
        single_table1.compute_frequencies()
        pair_table1.compute_frequencies()
        self.assertTrue(parent_node.name in test_dict)
        self.assertTrue('single' in test_dict[parent_node.name])
        self.assertTrue('pair' in test_dict[parent_node.name])
        single_freq_table1 = test_dict[parent_node.name]['single']
        if low_mem:
            single_freq_table1 = load_freq_table(freq_table=single_freq_table1, low_memory=low_mem)
        self.assertEqual(single_freq_table1.get_table(), single_table1.get_table())
        pair_freq_table1 = test_dict[parent_node.name]['pair']
        if low_mem:
            pair_freq_table1 = load_freq_table(freq_table=pair_freq_table1, low_memory=low_mem)
        self.assertEqual(pair_freq_table1.get_table(), pair_table1.get_table())
        self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}.fa'.format(parent_node.name))))
        sub_aln2 = aln.generate_sub_alignment(sequence_ids=[x.name for x in query_node.get_terminals()])
        single_table2, pair_table2 = sub_aln2.characterize_positions()
        single_table2.compute_frequencies()
        pair_table2.compute_frequencies()
        self.assertTrue(query_node.name in test_dict)
        self.assertTrue('single' in test_dict[query_node.name])
        single_freq_table2 = test_dict[query_node.name]['single']
        if low_mem:
            single_freq_table2 = load_freq_table(freq_table=single_freq_table2, low_memory=low_mem)
        self.assertEqual(single_freq_table2.get_table(), single_table2.get_table())
        self.assertTrue('pair' in test_dict[query_node.name])
        pair_freq_table2 = test_dict[query_node.name]['pair']
        if low_mem:
            pair_freq_table2 = load_freq_table(freq_table=pair_freq_table2, low_memory=low_mem)
        self.assertEqual(pair_freq_table2.get_table(), pair_table2.get_table())
        self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}.fa'.format(query_node.name))))
        sub_aln3 = aln.generate_sub_alignment(sequence_ids=[query_neighbor_node.name])
        single_table3, pair_table3 = sub_aln3.characterize_positions()
        single_table3.compute_frequencies()
        pair_table3.compute_frequencies()
        self.assertTrue(query_neighbor_node.name in test_dict)
        self.assertTrue('single' in test_dict[query_neighbor_node.name])
        single_freq_table3 = test_dict[query_neighbor_node.name]['single']
        if low_mem:
            single_freq_table3 = load_freq_table(freq_table=single_freq_table3, low_memory=low_mem)
        self.assertEqual(single_freq_table3.get_table(), single_table3.get_table())
        self.assertTrue('pair' in test_dict[query_neighbor_node.name])
        pair_freq_table3 = test_dict[query_neighbor_node.name]['pair']
        if low_mem:
            pair_freq_table3 = load_freq_table(freq_table=pair_freq_table3, low_memory=low_mem)
        self.assertEqual(pair_freq_table3.get_table(), pair_table3.get_table())
        self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}.fa'.format(query_neighbor_node.name))))

    def test2a_characterize_rank_groups_initialize_characterization_pool(self):
        # Test pool initialization function and mappable function (minimal example) for characterization, small aln
        self.evaluate_characterize_rank_groups_pooling_functions(
            p_id=self.small_structure_id, aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
            out_dir=self.out_small_dir, low_mem=False)

    def test2b_characterize_rank_groups_initialize_characterization_pool(self):
        # Test pool initialization function and mappable function (minimal example) for characterization, large aln
        self.evaluate_characterize_rank_groups_pooling_functions(
            p_id=self.large_structure_id, aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
            out_dir=self.out_large_dir, low_mem=True)

    def evaluate_characterize_rank_groups(self, aln, phylo_tree, assign, single, pair, processors, low_mem):
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=assign, position_specific=single,
                      pair_specific=pair, output_dir=os.path.join(self.testing_dir, aln.query_id), low_mem=low_mem)
        trace.characterize_rank_groups(processes=processors)
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
                    self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}.fa'.format(node_name))))
                    expected_single_table, expected_pair_table = sub_aln.characterize_positions(single=single,
                                                                                                pair=pair)
                    if single:
                        expected_single_table.compute_frequencies()
                        single_table = trace.unique_nodes[node_name]['single']
                        if low_mem:
                            single_table = load_freq_table(freq_table=single_table, low_memory=low_mem)
                        self.assertEqual(single_table.get_table(),
                                         expected_single_table.get_table())
                    else:
                        self.assertIsNone(trace.unique_nodes[node_name]['single'])
                    if pair:
                        expected_pair_table.compute_frequencies()
                        pair_table = trace.unique_nodes[node_name]['pair']
                        if low_mem:
                            pair_table = load_freq_table(freq_table=pair_table, low_memory=low_mem)
                        self.assertEqual(pair_table.get_table(),
                                         expected_pair_table.get_table())
                    else:
                        self.assertIsNone(trace.unique_nodes[node_name]['pair'])
                    visited.add(node_name)

    def test2c_characterize_rank_groups(self):
        # Test characterizing both single and pair positions, small alignment, single processed
        self.evaluate_characterize_rank_groups(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                                               assign=self.assignments_small, single=True, pair=True, processors=1,
                                               low_mem=False)

    def test2d_characterize_rank_groups(self):
        # Test characterizing both single and pair positions, large alignment, single processed
        self.evaluate_characterize_rank_groups(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
                                               assign=self.assignments_large, single=True, pair=True, processors=1,
                                               low_mem=True)

    def test2e_characterize_rank_groups(self):
        # Test characterizing both single and pair positions, small alignment, multi-processed
        self.evaluate_characterize_rank_groups(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                                               assign=self.assignments_small, single=True, pair=True,
                                               processors=self.max_threads)

    def test2f_characterize_rank_groups(self):
        # Test characterizing both single and pair positions, large alignment, multi-processed
        self.evaluate_characterize_rank_groups(aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
                                               assign=self.assignments_large, single=True, pair=True,
                                               processors=self.max_threads)

    def test2g_characterize_rank_groups(self):
        # Test characterizing single (single processed)
        self.evaluate_characterize_rank_groups(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                                               assign=self.assignments_small, single=True, pair=False, processors=1)

    def test2h_characterize_rank_groups(self):
        # Test characterizing pair positions (single processed)
        self.evaluate_characterize_rank_groups(aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                                               assign=self.assignments_small, single=False, pair=True, processors=1)

    ####################################################################################################################

    def evaluate_trace_pool_functions_identity_metric(self, aln, phylo_tree, assign, single, pair, metric, processors):
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=assign, position_specific=single,
                      pair_specific=pair, output_dir=os.path.join(self.testing_dir, aln.query_id))
        trace.characterize_rank_groups(processes=processors)
        if single:
            pos_size = 1
        elif pair:
            pos_size = 2
        else:
            raise ValueError('Cannot evaluate if both single and pair are False.')
        scorer = PositionalScorer(seq_length=aln.seq_length, pos_size=pos_size, metric=metric)
        manager = Manager()
        group_queue = Queue(maxsize=(aln.size * 2) - 1)
        for node_name in trace.unique_nodes:
            group_queue.put_nowait(node_name)
        group_dict = manager.dict()
        init_trace_groups(group_queue=group_queue, scorer=scorer, group_dict=group_dict, pos_specific=single,
                          pair_specific=pair, u_dict=trace.unique_nodes)
        trace_groups(processor=0)
        self.assertEqual(len(group_dict.keys()), len(trace.unique_nodes.keys()))
        group_dict = dict(group_dict)
        for node_name in group_dict:
            self.assertTrue('single_scores' in group_dict[node_name])
            if single:
                expected_scores = scorer.score_group(trace.unique_nodes[node_name]['single'])
                diff = group_dict[node_name]['single_scores'] - expected_scores
                self.assertTrue(not diff.any())
            else:
                self.assertIsNone(group_dict[node_name]['single_scores'])
            self.assertTrue('pair_scores' in group_dict[node_name])
            if pair:
                expected_scores = scorer.score_group(trace.unique_nodes[node_name]['pair'])
                diff = group_dict[node_name]['pair_scores'] - expected_scores
                self.assertTrue(not diff.any())
            else:
                self.assertIsNone(group_dict[node_name]['pair_scores'])
            trace.unique_nodes[node_name].update(group_dict[node_name])
        #
        rank_queue = Queue(maxsize=aln.size)
        for rank in assign:
            rank_queue.put_nowait(rank)
        rank_dict = manager.dict()
        init_trace_ranks(rank_queue=rank_queue, scorer=scorer, rank_dict=rank_dict, pos_specific=single,
                         pair_specific=pair, a_dict=assign, u_dict=trace.unique_nodes)
        trace_ranks(processor=0)
        rank_dict = dict(rank_dict)
        for rank in assign.keys():
            group_scores = []
            for group in sorted(assign[rank].keys(), reverse=True):
                node_name = assign[rank][group]['node'].name
                if single:
                    group_scores.append(trace.unique_nodes[node_name]['single_scores'])
                elif pair:
                    group_scores.append(trace.unique_nodes[node_name]['pair_scores'])
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
                diff = np.abs(rank_dict[rank]['single_ranks'] - expected_rank_scores)
                not_passing = diff > 1E-15
                if not_passing.any():
                    print(rank_dict[rank]['single_ranks'])
                    print(expected_rank_scores)
                    print(diff)
                    indices = np.nonzero(not_passing)
                    print(rank_dict[rank]['single_ranks'][indices])
                    print(expected_rank_scores[indices])
                    print(diff[indices])
                self.assertTrue(not not_passing.any())
            else:
                self.assertIsNone(rank_dict[rank]['single_ranks'])
            self.assertTrue(rank in rank_dict)
            self.assertTrue('pair_ranks' in rank_dict[rank])
            if pair:
                diff = np.abs(rank_dict[rank]['pair_ranks'] - expected_rank_scores)
                not_passing = diff > 1E-15
                if not_passing.any():
                    print(rank_dict[rank]['pair_ranks'])
                    print(expected_rank_scores)
                    print(diff)
                    indices = np.nonzero(not_passing)
                    print(rank_dict[rank]['pairle_ranks'][indices])
                    print(expected_rank_scores[indices])
                    print(diff[indices])
                self.assertTrue(not not_passing.any())
            else:
                self.assertIsNone(rank_dict[rank]['pair_ranks'])

    def test3a_trace_pool_functions(self):
        # Test the pool functions outside of a multiprocessing environment for the small alignment, single positions,
        # and the identity metric
        self.evaluate_trace_pool_functions_identity_metric(
            aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_small, single=True,
            pair=False, metric='identity', processors=self.max_threads)

    def test3b_trace_pool_functions(self):
        # Test the pool functions outside of a multiprocessing environment for the large alignment, single positions,
        # and identity metric
        self.evaluate_trace_pool_functions_identity_metric(
            aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_large, single=True,
            pair=False, metric='identity', processors=self.max_threads)

    def test3c_trace_pool_functions(self):
        # Test the pool functions outside of a multiprocessing environment for the small alignment, single positions,
        # and plain entropy metric
        self.evaluate_trace_pool_functions_identity_metric(
            aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_small, single=True,
            pair=False, metric='plain_entropy', processors=self.max_threads)

    def test3d_trace_pool_functions(self):
        # Test the pool functions outside of a multiprocessing environment for the large alignment, single positions,
        # and plain entropy metric
        self.evaluate_trace_pool_functions_identity_metric(
            aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_large, single=True,
            pair=False, metric='plain_entropy', processors=self.max_threads)

    def test3e_trace_pool_functions(self):
        # Test the pool functions outside of a multiprocessing environment for the small alignment, pair positions,
        # and the identity metric
        self.evaluate_trace_pool_functions_identity_metric(
            aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_small, single=False,
            pair=True, metric='identity', processors=self.max_threads)


    def test3f_trace_pool_functions(self):
        # Test the pool functions outside of a multiprocessing environment for the large alignment, pair positions,
        # and identity metric
        self.evaluate_trace_pool_functions_identity_metric(
            aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_large, single=False,
            pair=True, metric='identity', processors=self.max_threads)

    def test3g_trace_pool_functions(self):
        # Test the pool functions outside of a multiprocessing environment for the small alignment, pair positions,
        # and plain entropy metric
        self.evaluate_trace_pool_functions_identity_metric(
            aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_small, single=False,
            pair=True, metric='plain_entropy', processors=self.max_threads)

    def test3h_trace_pool_functions(self):
        # Test the pool functions outside of a multiprocessing environment for the large alignment, pair positions,
        # and plain entropy metric
        self.evaluate_trace_pool_functions_identity_metric(
            aln=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large, assign=self.assignments_large, single=False,
            pair=True, metric='plain_entropy', processors=self.max_threads)



    def test5a_trace_pool_functions(self):
        # Test the pool functions outside of a multiprocessing environment for the small alignment and mutual
        # information metric
        self.evaluate_trace_pool_functions_identity_metric(
            aln=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small, assign=self.assignments_small, single=False,
            pair=True, metric='mutual_information', processors=self.max_threads)

    def test5b_trace_pool_functions(self):
        # Test the pool functions outside of a multiprocessing environment for the large alignment and mutual
        # information metric
        self.evaluate_trace_pair_position_pooling_function(aln=self.query_aln_fa_large, metric='mutual_information')

    ####################################################################################################################

    def test3c_trace(self):
        # Perform identity trace on single positions only for the small alignment
        # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(msa=self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        assignments = phylo_tree.assign_group_rank()
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=phylo_tree, group_assignments=assignments,
                            position_specific=True, pair_specific=False)
        trace_small.characterize_rank_groups(processes=self.max_threads)
        scorer = PositionalScorer(seq_length=self.query_aln_fa_small.seq_length, pos_size=1, metric='identity')
        rank_id = trace_small.trace(scorer=scorer, processes=self.max_threads)
        ranks = np.ones(self.query_aln_fa_small.seq_length)
        unique_scores = {}
        for rank in sorted(assignments.keys(), reverse=True):
            group_scores = []
            for group in sorted(assignments[rank].keys(), reverse=True):
                node_name = assignments[rank][group]['node'].name
                if node_name not in unique_scores:
                    group_score = scorer.score_group(trace_small.unique_nodes[node_name]['single'])
                    unique_scores[node_name] = group_score
                else:
                    group_score = unique_scores[node_name]
                group_scores.append(group_score)
            group_scores = np.stack(group_scores, axis=0)
            rank_scores = np.sum(group_scores, axis=0)
            for i in range(self.query_aln_fa_small.seq_length):
                if rank_scores[i] != 0:
                    ranks[i] += 1
        diff_ranks = rank_id - ranks
        self.assertTrue(not diff_ranks.any())

    def test3d_trace(self):
        # Perform identity trace on single positions only for the large alignment
        # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(msa=self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        assignments = phylo_tree.assign_group_rank()
        trace_large = Trace(alignment=self.query_aln_fa_large, phylo_tree=phylo_tree, group_assignments=assignments,
                            position_specific=True, pair_specific=False)
        trace_large.characterize_rank_groups(processes=self.max_threads)
        scorer = PositionalScorer(seq_length=self.query_aln_fa_large.seq_length, pos_size=1, metric='identity')
        rank_id = trace_large.trace(scorer=scorer, processes=self.max_threads)
        ranks = np.ones(self.query_aln_fa_large.seq_length)
        unique_scores = {}
        for rank in sorted(assignments.keys(), reverse=True):
            group_scores = []
            for group in sorted(assignments[rank].keys(), reverse=True):
                node_name = assignments[rank][group]['node'].name
                if node_name not in unique_scores:
                    group_score = scorer.score_group(trace_large.unique_nodes[node_name]['single'])
                    unique_scores[node_name] = group_score
                else:
                    group_score = unique_scores[node_name]
                group_scores.append(group_score)
            group_scores = np.stack(group_scores, axis=0)
            rank_scores = np.sum(group_scores, axis=0)
            for i in range(self.query_aln_fa_large.seq_length):
                if rank_scores[i] != 0:
                    ranks[i] += 1
        diff_ranks = rank_id - ranks
        self.assertTrue(not diff_ranks.any())

    def test3e_trace(self):
        # Test trace, metric identity, against ETC small alignment
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        et_mip_obj.import_phylogenetic_tree(out_dir=wetc_test_dir)
        et_mip_obj.import_rank_sores(out_dir=wetc_test_dir)
        et_mip_obj.import_assignments(out_dir=wetc_test_dir)
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=et_mip_obj.tree,
                            group_assignments=et_mip_obj.rank_group_assignments, position_specific=True,
                            pair_specific=False)
        trace_small.characterize_rank_groups(processes=self.max_threads)
        scorer = PositionalScorer(seq_length=self.query_aln_fa_small.seq_length, pos_size=1, metric='identity')
        rank_ids = trace_small.trace(scorer=scorer)
        # print(rank_ids)
        # print(et_mip_obj.rank_scores)
        diff_ranks = rank_ids - et_mip_obj.rank_scores
        # print(diff_ranks)
        self.assertTrue(not diff_ranks.any())

    def test3f_trace(self):
        # Test trace, metric identity, against ETC large alignment
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        et_mip_obj.import_phylogenetic_tree(out_dir=wetc_test_dir)
        et_mip_obj.import_rank_sores(out_dir=wetc_test_dir)
        et_mip_obj.import_assignments(out_dir=wetc_test_dir)
        trace_large = Trace(alignment=self.query_aln_fa_large, phylo_tree=et_mip_obj.tree,
                            group_assignments=et_mip_obj.rank_group_assignments, position_specific=True,
                            pair_specific=False)
        trace_large.characterize_rank_groups(processes=self.max_threads)
        scorer = PositionalScorer(seq_length=self.query_aln_fa_large.seq_length, pos_size=1, metric='identity')
        rank_ids = trace_large.trace(scorer=scorer)
        # print(rank_ids)
        # print(et_mip_obj.rank_scores)
        diff_ranks = rank_ids - et_mip_obj.rank_scores
        # print(diff_ranks)
        self.assertTrue(not diff_ranks.any())

     ###################################################################################################################

    def test4c_trace(self):
        # Perform plain entropy trace on single positions only for the small alignment
        # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(msa=self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        assignments = phylo_tree.assign_group_rank()
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=phylo_tree, group_assignments=assignments,
                            position_specific=True, pair_specific=False)
        trace_small.characterize_rank_groups(processes=self.max_threads)
        scorer = PositionalScorer(seq_length=self.query_aln_fa_small.seq_length, pos_size=1, metric='plain_entropy')
        rank_plain_entropy = trace_small.trace(scorer=scorer, gap_correction=None, processes=self.max_threads)
        ranks = np.ones(self.query_aln_fa_small.seq_length)
        unique_scores = {}
        for rank in sorted(assignments.keys(), reverse=True):
            group_scores = []
            for group in sorted(assignments[rank].keys(), reverse=True):
                node_name = assignments[rank][group]['node'].name
                if node_name not in unique_scores:
                    group_score = scorer.score_group(trace_small.unique_nodes[node_name]['single'])
                    unique_scores[node_name] = group_score
                else:
                    group_score = unique_scores[node_name]
                group_scores.append(group_score)
            group_scores = np.stack(group_scores, axis=0)
            weight = 1.0 / rank
            rank_scores = weight * np.sum(group_scores, axis=0)
            ranks += rank_scores
        diff_ranks = rank_plain_entropy - ranks
        not_passing = diff_ranks > 1E-14
        # if not_passing.any():
        #     print(rank_plain_entropy)
        #     print(ranks)
        #     indices = np.nonzero(not_passing)
        #     print(diff_ranks[indices])
        #     print(rank_plain_entropy[indices])
        #     print(ranks[indices])
        self.assertTrue(not diff_ranks.any())

    def test4d_trace(self):
        # Perform plain entropy trace on single positions only for the large alignment
        # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(msa=self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        assignments = phylo_tree.assign_group_rank()
        trace_large = Trace(alignment=self.query_aln_fa_large, phylo_tree=phylo_tree, group_assignments=assignments,
                            position_specific=True, pair_specific=False)
        trace_large.characterize_rank_groups(processes=self.max_threads)
        scorer = PositionalScorer(seq_length=self.query_aln_fa_large.seq_length, pos_size=1, metric='plain_entropy')
        rank_plain_entropy = trace_large.trace(scorer=scorer, gap_correction=None, processes=self.max_threads)
        ranks = np.ones(self.query_aln_fa_large.seq_length)
        unique_scores = {}
        for rank in sorted(assignments.keys(), reverse=True):
            group_scores = []
            for group in sorted(assignments[rank].keys(), reverse=True):
                node_name = assignments[rank][group]['node'].name
                if node_name not in unique_scores:
                    group_score = scorer.score_group(trace_large.unique_nodes[node_name]['single'])
                    unique_scores[node_name] = group_score
                else:
                    group_score = unique_scores[node_name]
                group_scores.append(group_score)
            group_scores = np.stack(group_scores, axis=0)
            weight = 1.0 / rank
            rank_scores = weight * np.sum(group_scores, axis=0)
            ranks += rank_scores
        diff_ranks = rank_plain_entropy - ranks
        not_passing = diff_ranks > 1E-13
        # if not_passing.any():
        #     print(rank_plain_entropy)
        #     print(ranks)
        #     indices = np.nonzero(not_passing)
        #     print(diff_ranks[indices])
        #     print(rank_plain_entropy[indices])
        #     print(ranks[indices])
        self.assertTrue(not not_passing.any())

    def test4e_trace(self):
        # Test trace, metric plain entropy, against ETC small alignment
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        # Perform the wetc traces
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        # Import the values needed to run a comparable trace with the python implementation
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        et_mip_obj.import_phylogenetic_tree(out_dir=wetc_test_dir)
        et_mip_obj.import_assignments(out_dir=wetc_test_dir)
        # Import the scores to compare to
        et_mip_obj.import_entropy_rank_sores(out_dir=wetc_test_dir)
        et_mip_obj.import_rank_sores(out_dir=wetc_test_dir, rank_type='rvET')
        # Perform the trace with the python implementation
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=et_mip_obj.tree,
                            group_assignments=et_mip_obj.rank_group_assignments, position_specific=True,
                            pair_specific=False)
        trace_small.characterize_rank_groups(processes=self.max_threads)
        scorer = PositionalScorer(seq_length=self.query_aln_fa_small.seq_length, pos_size=1, metric='plain_entropy')
        rank_plain_entropy = trace_small.trace(scorer=scorer)
        # Compare the rank scores computed by both methods
        rank_scores = []
        for rank in range(1, et_mip_obj.alignment.size):  # et_mip_obj.rank_group_assignments:
            group_scores = []
            for group in et_mip_obj.rank_group_assignments[rank]:
                node_name = et_mip_obj.rank_group_assignments[rank][group]['node'].name
                group_score = trace_small.unique_nodes[node_name]['single_scores']
                group_scores.append(group_score)
            group_scores = np.stack(group_scores, axis=0)
            weight = 1.0 / rank
            rank_score = weight * np.sum(group_scores, axis=0)
            diff_rank = et_mip_obj.entropy[rank] - rank_score
            not_passing_rank = diff_rank > 1E-6
            # if not_passing_rank.any():
            #     print('RANK {} COMPARISON FAILED'.format(rank))
            #     print(rank_scores)
            #     print(et_mip_obj.entropy[rank])
            #     print(diff_rank)
            #     indices = np.nonzero(not_passing_rank)
            #     print(rank_scores[indices])
            #     print(et_mip_obj.entropy[rank][indices])
            #     print(diff_rank[indices])
            self.assertTrue(not not_passing_rank.any())
            rank_scores.append(rank_score)
        # Compare the final ranks computed by both methods
        final_ranks = 1 + np.sum(np.stack(rank_scores, axis=0), axis=0)
        # diff_final = rank_plain_entropy - et_mip_obj.rho
        diff_final = final_ranks - et_mip_obj.rho
        not_passing_final = diff_final > 1E-6
        # if not_passing_final.any():
        #     print('FINAL COMPARISON FAILED')
        #     print(final_ranks)
        #     print(et_mip_obj.rho)
        #     print(diff_final)
        #     indices = np.nonzero(not_passing_final)
        #     print(final_ranks[indices])
        #     print(et_mip_obj.rho[indices])
        #     print(diff_final[indices])
        self.assertTrue(not not_passing_final.any())
        # Compare the rank scores after gap correction
        diff_plain_entropy = rank_plain_entropy - et_mip_obj.rank_scores
        not_passing_plain_entropy = diff_plain_entropy > 1E-13
        # if not_passing_plain_entropy.any():
        #     print('GAP CORRECTION FAILED')
        #     print(rank_plain_entropy)
        #     print(et_mip_obj.rank_scores)
        #     print(diff_plain_entropy)
        #     indices = np.nonzero(not_passing_plain_entropy)
        #     print(rank_plain_entropy[indices])
        #     print(et_mip_obj.rank_scores[indices])
        #     print(diff_plain_entropy[indices])
        self.assertTrue(not not_passing_plain_entropy.any())

    def test4f_trace(self):
        # Test trace, metric plain entropy, against ETC large alignment
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        # Perform the wetc traces
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        # Import the values needed to run a comparable trace with the python implementation
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        et_mip_obj.import_phylogenetic_tree(out_dir=wetc_test_dir)
        et_mip_obj.import_assignments(out_dir=wetc_test_dir)
        # Import the scores to compare to
        et_mip_obj.import_entropy_rank_sores(out_dir=wetc_test_dir)
        et_mip_obj.import_rank_sores(out_dir=wetc_test_dir, rank_type='rvET')
        # Perform the trace with the python implementation
        trace_large = Trace(alignment=self.query_aln_fa_large, phylo_tree=et_mip_obj.tree,
                            group_assignments=et_mip_obj.rank_group_assignments, position_specific=True,
                            pair_specific=False)
        trace_large.characterize_rank_groups(processes=self.max_threads)
        scorer = PositionalScorer(seq_length=self.query_aln_fa_large.seq_length, pos_size=1, metric='plain_entropy')
        rank_plain_entropy = trace_large.trace(scorer=scorer)
        # Compare the rank scores computed by both methods
        rank_scores = []
        for rank in range(1, et_mip_obj.alignment.size):  # et_mip_obj.rank_group_assignments:
            group_scores = []
            for group in et_mip_obj.rank_group_assignments[rank]:
                node_name = et_mip_obj.rank_group_assignments[rank][group]['node'].name
                group_score = trace_large.unique_nodes[node_name]['single_scores']
                group_scores.append(group_score)
            group_scores = np.stack(group_scores, axis=0)
            weight = 1.0 / rank
            rank_score = weight * np.sum(group_scores, axis=0)
            diff_rank = et_mip_obj.entropy[rank] - rank_score
            not_passing_rank = diff_rank > 1E-6
            # if not_passing_rank.any():
            #     print('RANK {} COMPARISON FAILED'.format(rank))
            #     print(rank_scores)
            #     print(et_mip_obj.entropy[rank])
            #     print(diff_rank)
            #     indices = np.nonzero(not_passing_rank)
            #     print(rank_scores[indices])
            #     print(et_mip_obj.entropy[rank][indices])
            #     print(diff_rank[indices])
            self.assertTrue(not not_passing_rank.any())
            rank_scores.append(rank_score)
        # Compare the final ranks computed by both methods
        final_ranks = 1 + np.sum(np.stack(rank_scores, axis=0), axis=0)
        # diff_final = rank_plain_entropy - et_mip_obj.rho
        diff_final = final_ranks - et_mip_obj.rho
        not_passing_final = diff_final > 1E-6
        # if not_passing_final.any():
        #     print('FINAL COMPARISON FAILED')
        #     print(final_ranks)
        #     print(et_mip_obj.rho)
        #     print(diff_final)
        #     indices = np.nonzero(not_passing_final)
        #     print(final_ranks[indices])
        #     print(et_mip_obj.rho[indices])
        #     print(diff_final[indices])
        self.assertTrue(not not_passing_final.any())
        # Compare the rank scores after gap correction
        diff_plain_entropy = rank_plain_entropy - et_mip_obj.rank_scores
        not_passing_plain_entropy = diff_plain_entropy > 1E-13
        # if not_passing_plain_entropy.any():
        #     print('GAP CORRECTION FAILED')
        #     print(rank_plain_entropy)
        #     print(et_mip_obj.rank_scores)
        #     print(diff_plain_entropy)
        #     indices = np.nonzero(not_passing_plain_entropy)
        #     print(rank_plain_entropy[indices])
        #     print(et_mip_obj.rank_scores[indices])
        #     print(diff_plain_entropy[indices])
        self.assertTrue(not not_passing_plain_entropy.any())

    ####################################################################################################################

    def evaluate_trace_pair_position_pooling_function(self, aln, metric):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(msa=aln.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        assignments = phylo_tree.assign_group_rank()
        trace = Trace(alignment=aln, phylo_tree=phylo_tree, group_assignments=assignments, position_specific=False,
                      pair_specific=True)
        trace.characterize_rank_groups(processes=self.max_threads)
        scorer = PositionalScorer(seq_length=aln.seq_length, pos_size=2, metric='mutual_information')
        manager = Manager()
        group_queue = Queue(maxsize=(aln.size * 2) - 1)
        for node_name in trace.unique_nodes:
            group_queue.put_nowait(node_name)
        group_dict = manager.dict()
        init_trace_groups(group_queue=group_queue, scorer=scorer, group_dict=group_dict, pos_specific=False,
                          pair_specific=True, u_dict=trace.unique_nodes)
        trace_groups(processor=0)
        self.assertEqual(len(group_dict.keys()), len(trace.unique_nodes.keys()))
        group_dict = dict(group_dict)
        for node_name in group_dict:
            self.assertTrue('single_scores' in group_dict[node_name])
            self.assertIsNone(group_dict[node_name]['single_scores'])
            self.assertTrue('pair_scores' in group_dict[node_name])
            expected_scores = scorer.score_group(trace.unique_nodes[node_name]['pair'])
            diff = group_dict[node_name]['pair_scores'] - expected_scores
            self.assertTrue(not diff.any())
            trace.unique_nodes[node_name].update(group_dict[node_name])
        #
        rank_queue = Queue(maxsize=aln.size)
        for rank in assignments:
            rank_queue.put_nowait(rank)
        rank_dict = manager.dict()
        init_trace_ranks(rank_queue=rank_queue, scorer=scorer, rank_dict=rank_dict, pos_specific=False,
                         pair_specific=True, a_dict=assignments, u_dict=trace.unique_nodes)
        trace_ranks(processor=0)
        rank_dict = dict(rank_dict)
        for rank in assignments.keys():
            print('RANK: {}'.format(rank))
            group_scores = []
            for group in sorted(assignments[rank].keys(), reverse=True):
                node_name = assignments[rank][group]['node'].name
                group_score = trace.unique_nodes[node_name]['pair_scores']
                group_scores.append(group_score)
            group_scores = np.stack(group_scores, axis=0)
            weight = 1.0 / rank
            rank_scores = weight * np.sum(group_scores, axis=0)
            diff = rank_dict[rank]['pair_ranks'] - rank_scores
            not_passing = diff > 1E-15
            if not_passing.any():
                print(rank_dict[rank]['pair_ranks'])
                print(rank_scores)
                indices = np.nonzero(not_passing)
                print(diff[indices])
                print(rank_dict[rank]['pair_ranks'][indices])
                print(rank_scores[indices])
            self.assertTrue(not not_passing.any())
            self.assertIsNone(rank_dict[rank]['single_ranks'])

    # def test5c_trace(self):
    #     # Perform mutual information trace on pairs of positions only for the small alignment
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    # def test5d_trace(self):
    #     # Perform mutual information trace on pairs of positions only for the large alignment
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    #
    # ####################################################################################################################
    #
    # def test6a_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the small alignment and normalized mutual
    #     # information metric
    # def test6b_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the large alignment and normalized mutual
    #     # information metric
    # def test6c_trace(self):
    #     # Perform normalized mutual information trace on pairs of positions only for the small alignment
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    # def test6d_trace(self):
    #     # Perform normalized mutual information trace on pairs of positions only for the large alignment
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    #
    # ####################################################################################################################
    #
    # def test7a_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the small alignment and average product
    #     # corrected mutual information metric
    # def test7b_trace_pool_functions(self):
    #     # Test the pool functions outside of a multiprocessing environment for the large alignment and average product
    #     # corrected mutual information metric
    # def test7c_trace(self):
    #     # Perform average product corrected mutual information trace on pairs of positions only for the small alignment
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    # def test7d_trace(self):
    #     # Perform average product corrected mutual information trace on pairs of positions only for the large alignment
    #     # Assume scoring happens correctly since it has been tested above and ensure that expected ranks are achieved
    # def test7e_trace(self):
    #     # Test trace, metric average product corrected mutual information, against ETC small alignment
    # def test7f_trace(self):
    #     # Test trace, metric average product corrected mutual informatoin, against ETC large alignment