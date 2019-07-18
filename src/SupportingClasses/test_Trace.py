"""
Created on July 11, 2019

@author: Daniel Konecki
"""
import os
import numpy as np
from copy import deepcopy
from multiprocessing import Lock, Manager, Queue
from test_Base import TestBase
from ETMIPWrapper import ETMIPWrapper
from SeqAlignment import SeqAlignment
from PositionalScorer import PositionalScorer
from PhylogeneticTree import PhylogeneticTree
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from Trace import (Trace, init_characterization_pool, characterization, init_trace_groups, trace_groups,
                   init_trace_ranks, trace_ranks)


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
        cls.query_aln_msf_large = deepcopy(cls.query_aln_fa_large)
        cls.query_aln_msf_large.file_name = cls.data_set.protein_data[cls.large_structure_id]['Final_MSF_Aln']

    def test1a_init(self):
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                            group_assignments=self.assignments_small, position_specific=True, pair_specific=True)
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

    def test1b_init(self):
        trace_large = Trace(alignment=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
                            group_assignments=self.assignments_large, position_specific=True, pair_specific=True)
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

    def test2a_characterize_rank_groups_initialize_characterization_pool(self):
        # Test pool initialization function and mappable function (minimal example)
        test_queue = Queue(maxsize=3)
        # Retrieve 3 node tree consisting of query node, its sibling, and its parent.
        query_node = [x for x in self.phylo_tree_small.tree.get_terminals() if x.name == self.small_structure_id][0]
        parent_node = self.phylo_tree_small.tree.get_path(query_node)[-2]
        test_queue.put_nowait(parent_node.clades[0])
        test_queue.put_nowait(parent_node.clades[1])
        test_queue.put_nowait(parent_node)
        # Test the functions
        test_manager = Manager()
        test_dict = test_manager.dict()
        init_characterization_pool(alignment=self.query_aln_fa_small, pos_specific=True, pair_specific=True,
                                   queue=test_queue, sharable_dict=test_dict)
        characterization(processor=1)
        test_dict = dict(test_dict)
        self.assertEqual(len(test_dict), 3)
        sub_aln1 = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[x.name for x in parent_node.clades])
        single_table1, pair_table1 = sub_aln1.characterize_positions()
        single_table1.compute_frequencies()
        pair_table1.compute_frequencies()
        self.assertTrue(parent_node.name in test_dict)
        self.assertTrue('single' in test_dict[parent_node.name])
        self.assertTrue('pair' in test_dict[parent_node.name])
        self.assertEqual(test_dict[parent_node.name]['single'].get_table(), single_table1.get_table())
        self.assertEqual(test_dict[parent_node.name]['pair'].get_table(), pair_table1.get_table())
        sub_aln2 = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[parent_node.clades[0].name])
        single_table2, pair_table2 = sub_aln2.characterize_positions()
        single_table2.compute_frequencies()
        pair_table2.compute_frequencies()
        self.assertTrue(parent_node.clades[0].name in test_dict)
        self.assertTrue('single' in test_dict[parent_node.clades[0].name])
        self.assertTrue('pair' in test_dict[parent_node.clades[0].name])
        self.assertEqual(test_dict[parent_node.clades[0].name]['single'].get_table(), single_table2.get_table())
        self.assertEqual(test_dict[parent_node.clades[0].name]['pair'].get_table(), pair_table2.get_table())
        sub_aln3 = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[parent_node.clades[1].name])
        single_table3, pair_table3 = sub_aln3.characterize_positions()
        single_table3.compute_frequencies()
        pair_table3.compute_frequencies()
        self.assertTrue(parent_node.clades[1].name in test_dict)
        self.assertTrue('single' in test_dict[parent_node.clades[1].name])
        self.assertTrue('pair' in test_dict[parent_node.clades[1].name])
        self.assertEqual(test_dict[parent_node.clades[1].name]['single'].get_table(), single_table3.get_table())
        self.assertEqual(test_dict[parent_node.clades[1].name]['pair'].get_table(), pair_table3.get_table())

    def test2b_characterize_rank_groups(self):
        # Test characterizing both single and pair positions (single processed)
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                            group_assignments=self.assignments_small, position_specific=True, pair_specific=True)
        trace_small.characterize_rank_groups()
        visited = set()
        for rank in trace_small.assignments:
            for group in trace_small.assignments[rank]:
                node_name = trace_small.assignments[rank][group]['node'].name
                self.assertTrue(node_name in trace_small.unique_nodes)
                self.assertTrue('single' in trace_small.unique_nodes[node_name])
                self.assertTrue('pair' in trace_small.unique_nodes[node_name])
                if node_name not in visited:
                    sub_aln = self.query_aln_fa_small.generate_sub_alignment(
                        sequence_ids=trace_small.assignments[rank][group]['terminals'])
                    single_table, pair_table = sub_aln.characterize_positions(single=True, pair=True)
                    single_table.compute_frequencies()
                    pair_table.compute_frequencies()
                    self.assertEqual(trace_small.unique_nodes[node_name]['single'].get_table(),
                                     single_table.get_table())
                    self.assertEqual(trace_small.unique_nodes[node_name]['pair'].get_table(), pair_table.get_table())
                    visited.add(node_name)

    def test2c_characterize_rank_groups(self):
        # Test characterizing both single and pair positions (multi-processed)
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                            group_assignments=self.assignments_small, position_specific=True, pair_specific=True)
        trace_small.characterize_rank_groups(processes=self.max_threads)
        visited = set()
        for rank in trace_small.assignments:
            for group in trace_small.assignments[rank]:
                node_name = trace_small.assignments[rank][group]['node'].name
                self.assertTrue('single' in trace_small.unique_nodes[node_name])
                self.assertTrue('pair' in trace_small.unique_nodes[node_name])
                if node_name not in visited:
                    sub_aln = self.query_aln_fa_small.generate_sub_alignment(
                        sequence_ids=trace_small.assignments[rank][group]['terminals'])
                    single_table, pair_table = sub_aln.characterize_positions(single=True, pair=True)
                    single_table.compute_frequencies()
                    pair_table.compute_frequencies()
                    self.assertEqual(trace_small.unique_nodes[node_name]['single'].get_table(),
                                     single_table.get_table())
                    self.assertEqual(trace_small.unique_nodes[node_name]['pair'].get_table(), pair_table.get_table())
                    visited.add(node_name)

    def test2d_characterize_rank_groups(self):
        # Test characterizing single (single processed)
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                            group_assignments=self.assignments_small, position_specific=True, pair_specific=False)
        trace_small.characterize_rank_groups()
        visited = set()
        for rank in trace_small.assignments:
            for group in trace_small.assignments[rank]:
                node_name = trace_small.assignments[rank][group]['node'].name
                self.assertTrue('single' in trace_small.unique_nodes[node_name])
                self.assertTrue('pair' in trace_small.unique_nodes[node_name])
                if node_name not in visited:
                    sub_aln = self.query_aln_fa_small.generate_sub_alignment(
                        sequence_ids=trace_small.assignments[rank][group]['terminals'])
                    single_table, pair_table = sub_aln.characterize_positions(single=True, pair=False)
                    single_table.compute_frequencies()
                    self.assertEqual(trace_small.unique_nodes[node_name]['single'].get_table(),
                                     single_table.get_table())
                    self.assertIsNone(trace_small.unique_nodes[node_name]['pair'])
                    visited.add(node_name)

    def test2e_characterize_rank_groups(self):
        # Test characterizing pair positions (single processed)
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                            group_assignments=self.assignments_small, position_specific=False, pair_specific=True)
        trace_small.characterize_rank_groups()
        visited = set()
        for rank in trace_small.assignments:
            for group in trace_small.assignments[rank]:
                node_name = trace_small.assignments[rank][group]['node'].name
                self.assertTrue('single' in trace_small.unique_nodes[node_name])
                self.assertTrue('pair' in trace_small.unique_nodes[node_name])
                if node_name not in visited:
                    sub_aln = self.query_aln_fa_small.generate_sub_alignment(
                        sequence_ids=trace_small.assignments[rank][group]['terminals'])
                    single_table, pair_table = sub_aln.characterize_positions(single=False, pair=True)
                    pair_table.compute_frequencies()
                    self.assertIsNone(trace_small.unique_nodes[node_name]['single'])
                    self.assertEqual(trace_small.unique_nodes[node_name]['pair'].get_table(), pair_table.get_table())
                    visited.add(node_name)

    def test3a_trace_pool_functions(self):
        # Test the pool functions outside of a multiprocessing environment for the small alignment and identity metric
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(msa=self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        assignments = phylo_tree.assign_group_rank()
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=phylo_tree, group_assignments=assignments,
                            position_specific=True, pair_specific=False)
        trace_small.characterize_rank_groups(processes=self.max_threads)
        scorer = PositionalScorer(seq_length=self.query_aln_fa_small.seq_length, pos_size=1, metric='identity')
        manager = Manager()
        group_queue = Queue(maxsize=(self.query_aln_fa_small.size * 2) - 1)
        for node_name in trace_small.unique_nodes:
            group_queue.put_nowait(node_name)
        group_dict = manager.dict()
        init_trace_groups(group_queue=group_queue, scorer=scorer, group_dict=group_dict, pos_specific=True,
                          pair_specific=False, u_dict=trace_small.unique_nodes)
        trace_groups(processor=0)
        self.assertEqual(len(group_dict.keys()), len(trace_small.unique_nodes.keys()))
        group_dict = dict(group_dict)
        for node_name in group_dict:
            self.assertTrue('single_scores' in group_dict[node_name])
            expected_scores = scorer.score_group(trace_small.unique_nodes[node_name]['single'])
            diff = group_dict[node_name]['single_scores'] - expected_scores
            self.assertTrue(not diff.any())
            self.assertTrue('pair_scores' in group_dict[node_name])
            self.assertIsNone(group_dict[node_name]['pair_scores'])
            trace_small.unique_nodes[node_name].update(group_dict[node_name])
        #
        rank_queue = Queue(maxsize=self.query_aln_fa_small.size)
        for rank in assignments:
            rank_queue.put_nowait(rank)
        rank_dict = manager.dict()
        init_trace_ranks(rank_queue=rank_queue, scorer=scorer, rank_dict=rank_dict, pos_specific=True,
                         pair_specific=False, a_dict=assignments, u_dict=trace_small.unique_nodes)
        trace_ranks(processor=0)
        rank_dict = dict(rank_dict)
        for rank in assignments.keys():
            group_scores = []
            for group in sorted(assignments[rank].keys(), reverse=True):
                node_name = assignments[rank][group]['node'].name
                group_scores.append(trace_small.unique_nodes[node_name]['single_scores'])
            group_scores = np.stack(group_scores, axis=0)
            rank_scores = np.sum(group_scores, axis=0)
            for i in range(self.query_aln_fa_small.seq_length):
                if rank_scores[i] == 0:
                    self.assertEqual(rank_dict[rank]['single_ranks'][i], 0)
                else:
                    self.assertEqual(rank_dict[rank]['single_ranks'][i], 1)

    def test3b_trace_pool_functions(self):
        # Test the pool functions outside of a multiprocessing environment for the arge alignment and identity metric
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(msa=self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        assignments = phylo_tree.assign_group_rank()
        trace_large = Trace(alignment=self.query_aln_fa_large, phylo_tree=phylo_tree, group_assignments=assignments,
                            position_specific=True, pair_specific=False)
        trace_large.characterize_rank_groups(processes=self.max_threads)
        scorer = PositionalScorer(seq_length=self.query_aln_fa_large.seq_length, pos_size=1, metric='identity')
        manager = Manager()
        group_queue = Queue(maxsize=(self.query_aln_fa_large.size * 2) - 1)
        for node_name in trace_large.unique_nodes:
            group_queue.put_nowait(node_name)
        group_dict = manager.dict()
        init_trace_groups(group_queue=group_queue, scorer=scorer, group_dict=group_dict, pos_specific=True,
                          pair_specific=False, u_dict=trace_large.unique_nodes)
        trace_groups(processor=0)
        self.assertEqual(len(group_dict.keys()), len(trace_large.unique_nodes.keys()))
        group_dict = dict(group_dict)
        for node_name in group_dict:
            self.assertTrue('single_scores' in group_dict[node_name])
            expected_scores = scorer.score_group(trace_large.unique_nodes[node_name]['single'])
            diff = group_dict[node_name]['single_scores'] - expected_scores
            self.assertTrue(not diff.any())
            self.assertTrue('pair_scores' in group_dict[node_name])
            self.assertIsNone(group_dict[node_name]['pair_scores'])
            trace_large.unique_nodes[node_name].update(group_dict[node_name])
        #
        rank_queue = Queue(maxsize=self.query_aln_fa_large.size)
        for rank in assignments:
            rank_queue.put_nowait(rank)
        rank_dict = manager.dict()
        init_trace_ranks(rank_queue=rank_queue, scorer=scorer, rank_dict=rank_dict, pos_specific=True,
                         pair_specific=False, a_dict=assignments, u_dict=trace_large.unique_nodes)
        trace_ranks(processor=0)
        rank_dict = dict(rank_dict)
        for rank in assignments.keys():
            group_scores = []
            for group in sorted(assignments[rank].keys(), reverse=True):
                node_name = assignments[rank][group]['node'].name
                group_scores.append(trace_large.unique_nodes[node_name]['single_scores'])
            group_scores = np.stack(group_scores, axis=0)
            rank_scores = np.sum(group_scores, axis=0)
            for i in range(self.query_aln_fa_large.seq_length):
                if rank_scores[i] == 0:
                    self.assertEqual(rank_dict[rank]['single_ranks'][i], 0)
                else:
                    self.assertEqual(rank_dict[rank]['single_ranks'][i], 1)

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

    def test4a_trace_pool_functions(self):
        # Test the pool functions outside of a multiprocessing environment for the small alignment and plain entropy
        # metric
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(msa=self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        assignments = phylo_tree.assign_group_rank()
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=phylo_tree, group_assignments=assignments,
                            position_specific=True, pair_specific=False)
        trace_small.characterize_rank_groups(processes=self.max_threads)
        scorer = PositionalScorer(seq_length=self.query_aln_fa_small.seq_length, pos_size=1, metric='plain_entropy')
        manager = Manager()
        group_queue = Queue(maxsize=(self.query_aln_fa_small.size * 2) - 1)
        for node_name in trace_small.unique_nodes:
            group_queue.put_nowait(node_name)
        group_dict = manager.dict()
        init_trace_groups(group_queue=group_queue, scorer=scorer, group_dict=group_dict, pos_specific=True,
                          pair_specific=False, u_dict=trace_small.unique_nodes)
        trace_groups(processor=0)
        self.assertEqual(len(group_dict.keys()), len(trace_small.unique_nodes.keys()))
        group_dict = dict(group_dict)
        for node_name in group_dict:
            self.assertTrue('single_scores' in group_dict[node_name])
            expected_scores = scorer.score_group(trace_small.unique_nodes[node_name]['single'])
            diff = group_dict[node_name]['single_scores'] - expected_scores
            self.assertTrue(not diff.any())
            self.assertTrue('pair_scores' in group_dict[node_name])
            self.assertIsNone(group_dict[node_name]['pair_scores'])
            trace_small.unique_nodes[node_name].update(group_dict[node_name])
        #
        rank_queue = Queue(maxsize=self.query_aln_fa_small.size)
        for rank in assignments:
            rank_queue.put_nowait(rank)
        rank_dict = manager.dict()
        init_trace_ranks(rank_queue=rank_queue, scorer=scorer, rank_dict=rank_dict, pos_specific=True,
                         pair_specific=False, a_dict=assignments, u_dict=trace_small.unique_nodes)
        trace_ranks(processor=0)
        rank_dict = dict(rank_dict)
        for rank in assignments.keys():
            print('RANK: {}'.format(rank))
            group_scores = []
            for group in sorted(assignments[rank].keys(), reverse=True):
                node_name = assignments[rank][group]['node'].name
                group_score = trace_small.unique_nodes[node_name]['single_scores']
                group_scores.append(group_score)
            group_scores = np.stack(group_scores, axis=0)
            weight = 1.0 / rank
            rank_scores = weight * np.sum(group_scores, axis=0)
            diff = rank_dict[rank]['single_ranks'] - rank_scores
            not_passing = diff > 1E-15
            # if not_passing.any():
            #     print(rank_dict[rank]['single_ranks'])
            #     print(rank_scores)
            #     indices = np.nonzero(not_passing)
            #     print(diff[indices])
            #     print(rank_dict[rank]['single_ranks'][indices])
            #     print(rank_scores[indices])
            self.assertTrue(not not_passing.any())
            self.assertIsNone(rank_dict[rank]['pair_ranks'])

    def test4b_trace_pool_functions(self):
        # Test the pool functions outside of a multiprocessing environment for the large alignment and plain entropy
        # metric
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(msa=self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        assignments = phylo_tree.assign_group_rank()
        trace_large = Trace(alignment=self.query_aln_fa_large, phylo_tree=phylo_tree, group_assignments=assignments,
                            position_specific=True, pair_specific=False)
        trace_large.characterize_rank_groups(processes=self.max_threads)
        scorer = PositionalScorer(seq_length=self.query_aln_fa_large.seq_length, pos_size=1, metric='plain_entropy')
        manager = Manager()
        group_queue = Queue(maxsize=(self.query_aln_fa_large.size * 2) - 1)
        for node_name in trace_large.unique_nodes:
            group_queue.put_nowait(node_name)
        group_dict = manager.dict()
        init_trace_groups(group_queue=group_queue, scorer=scorer, group_dict=group_dict, pos_specific=True,
                          pair_specific=False, u_dict=trace_large.unique_nodes)
        trace_groups(processor=0)
        self.assertEqual(len(group_dict.keys()), len(trace_large.unique_nodes.keys()))
        group_dict = dict(group_dict)
        for node_name in group_dict:
            self.assertTrue('single_scores' in group_dict[node_name])
            expected_scores = scorer.score_group(trace_large.unique_nodes[node_name]['single'])
            diff = group_dict[node_name]['single_scores'] - expected_scores
            self.assertTrue(not diff.any())
            self.assertTrue('pair_scores' in group_dict[node_name])
            self.assertIsNone(group_dict[node_name]['pair_scores'])
            trace_large.unique_nodes[node_name].update(group_dict[node_name])
        #
        rank_queue = Queue(maxsize=self.query_aln_fa_large.size)
        for rank in assignments:
            rank_queue.put_nowait(rank)
        rank_dict = manager.dict()
        init_trace_ranks(rank_queue=rank_queue, scorer=scorer, rank_dict=rank_dict, pos_specific=True,
                         pair_specific=False, a_dict=assignments, u_dict=trace_large.unique_nodes)
        trace_ranks(processor=0)
        rank_dict = dict(rank_dict)
        for rank in assignments.keys():
            group_scores = []
            for group in sorted(assignments[rank].keys(), reverse=True):
                node_name = assignments[rank][group]['node'].name
                group_scores.append(trace_large.unique_nodes[node_name]['single_scores'])
            group_scores = np.stack(group_scores, axis=0)
            weight = 1.0 / rank
            rank_scores = weight * np.sum(group_scores, axis=0)
            diff = rank_dict[rank]['single_ranks'] - rank_scores
            not_passing = diff > 1E-15
            # if not_passing.any():
            #     print(rank_dict[rank]['single_ranks'])
            #     print(rank_scores)
            #     indices = np.nonzero(not_passing)
            #     print(diff[indices])
            #     print(rank_dict[rank]['single_ranks'][indices])
            #     print(rank_scores[indices])
            self.assertTrue(not not_passing.any())
            self.assertIsNone(rank_dict[rank]['pair_ranks'])

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
        rank_plain_entropy = trace_small.trace(scorer=scorer, processes=self.max_threads)
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
        rank_plain_entropy = trace_large.trace(scorer=scorer, processes=self.max_threads)
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
        # Perform the trace with the python implementation
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=et_mip_obj.tree,
                            group_assignments=et_mip_obj.rank_group_assignments, position_specific=True,
                            pair_specific=False)
        trace_small.characterize_rank_groups(processes=self.max_threads)
        scorer = PositionalScorer(seq_length=self.query_aln_fa_small.seq_length, pos_size=1, metric='plain_entropy')
        rank_plain_entropy = trace_small.trace(scorer=scorer)
        # Compare the rank scores computed by both methods
        for rank in range(1, et_mip_obj.alignment.size):  # et_mip_obj.rank_group_assignments:
            group_scores = []
            for group in et_mip_obj.rank_group_assignments[rank]:
                node_name = et_mip_obj.rank_group_assignments[rank][group]['node'].name
                group_score = trace_small.unique_nodes[node_name]['single_scores']
                group_scores.append(group_score)
            group_scores = np.stack(group_scores, axis=0)
            weight = 1.0 / rank
            rank_scores = weight * np.sum(group_scores, axis=0)
            diff_rank = et_mip_obj.entropy[rank] - rank_scores
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
        # Compare the final ranks computed by both methods
        diff_final = rank_plain_entropy - et_mip_obj.rho
        not_passing_final = diff_final > 1E-6
        # if not_passing_final.any():
        #     print('FINAL COMPARISON FAILED')
        #     print(rank_plain_entropy)
        #     print(et_mip_obj.rho)
        #     print(diff_final)
        #     indices = np.nonzero(not_passing_final)
        #     print(rank_plain_entropy[indices])
        #     print(et_mip_obj.rho[indices])
        #     print(diff_final[indices])
        self.assertTrue(not not_passing_final.any())

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
        # Perform the trace with the python implementation
        trace_large = Trace(alignment=self.query_aln_fa_large, phylo_tree=et_mip_obj.tree,
                            group_assignments=et_mip_obj.rank_group_assignments, position_specific=True,
                            pair_specific=False)
        trace_large.characterize_rank_groups(processes=self.max_threads)
        scorer = PositionalScorer(seq_length=self.query_aln_fa_large.seq_length, pos_size=1, metric='plain_entropy')
        rank_plain_entropy = trace_large.trace(scorer=scorer)
        # Compare the rank scores computed by both methods
        for rank in range(1, et_mip_obj.alignment.size):  # et_mip_obj.rank_group_assignments:
            group_scores = []
            for group in et_mip_obj.rank_group_assignments[rank]:
                node_name = et_mip_obj.rank_group_assignments[rank][group]['node'].name
                group_score = trace_large.unique_nodes[node_name]['single_scores']
                group_scores.append(group_score)
            group_scores = np.stack(group_scores, axis=0)
            weight = 1.0 / rank
            rank_scores = weight * np.sum(group_scores, axis=0)
            diff_rank = et_mip_obj.entropy[rank] - rank_scores
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
        # Compare the final ranks computed by both methods
        diff_final = rank_plain_entropy - et_mip_obj.rho
        not_passing_final = diff_final > 1E-6
        # if not_passing_final.any():
        #     print('FINAL COMPARISON FAILED')
        #     print(rank_plain_entropy)
        #     print(et_mip_obj.rho)
        #     print(diff_final)
        #     indices = np.nonzero(not_passing_final)
        #     print(rank_plain_entropy[indices])
        #     print(et_mip_obj.rho[indices])
        #     print(diff_final[indices])
        self.assertTrue(not not_passing_final.any())
