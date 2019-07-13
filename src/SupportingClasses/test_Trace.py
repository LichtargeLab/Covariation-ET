"""
Created on July 11, 2019

@author: Daniel Konecki
"""
import os
from copy import deepcopy
from multiprocessing import Queue, Manager
from test_Base import TestBase
from ETMIPWrapper import ETMIPWrapper
from SeqAlignment import SeqAlignment
from PositionalScorer import PositionalScorer
from PhylogeneticTree import PhylogeneticTree
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from Trace import Trace, init_characterization_pool, characterization


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
        self.assertTrue(parent_node.name in test_dict)
        self.assertTrue('single' in test_dict[parent_node.name])
        self.assertTrue('pair' in test_dict[parent_node.name])
        self.assertEqual(test_dict[parent_node.name]['single'].get_table(), single_table1.get_table())
        self.assertEqual(test_dict[parent_node.name]['pair'].get_table(), pair_table1.get_table())
        sub_aln2 = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[parent_node.clades[0].name])
        single_table2, pair_table2 = sub_aln2.characterize_positions()
        self.assertTrue(parent_node.clades[0].name in test_dict)
        self.assertTrue('single' in test_dict[parent_node.clades[0].name])
        self.assertTrue('pair' in test_dict[parent_node.clades[0].name])
        self.assertEqual(test_dict[parent_node.clades[0].name]['single'].get_table(), single_table2.get_table())
        self.assertEqual(test_dict[parent_node.clades[0].name]['pair'].get_table(), pair_table2.get_table())
        sub_aln3 = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[parent_node.clades[1].name])
        single_table3, pair_table3 = sub_aln3.characterize_positions()
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
        for rank in trace_small.assignments:
            for group in trace_small.assignments[rank]:
                sub_aln = self.query_aln_fa_small.generate_sub_alignment(
                    sequence_ids=trace_small.assignments[rank][group]['terminals'])
                single_table, pair_table = sub_aln.characterize_positions(single=True, pair=True)
                self.assertTrue('single' in trace_small.assignments[rank][group])
                self.assertEqual(trace_small.assignments[rank][group]['single'].get_table(), single_table.get_table())
                self.assertTrue('pair' in trace_small.assignments[rank][group])
                self.assertEqual(trace_small.assignments[rank][group]['pair'].get_table(), pair_table.get_table())

    def test2c_characterize_rank_groups(self):
        # Test characterizing both single and pair positions (multi-processed)
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                            group_assignments=self.assignments_small, position_specific=True, pair_specific=True)
        trace_small.characterize_rank_groups(processes=self.max_threads)
        for rank in trace_small.assignments:
            for group in trace_small.assignments[rank]:
                sub_aln = self.query_aln_fa_small.generate_sub_alignment(
                    sequence_ids=trace_small.assignments[rank][group]['terminals'])
                single_table, pair_table = sub_aln.characterize_positions(single=True, pair=True)
                self.assertTrue('single' in trace_small.assignments[rank][group])
                self.assertEqual(trace_small.assignments[rank][group]['single'].get_table(), single_table.get_table())
                self.assertTrue('pair' in trace_small.assignments[rank][group])
                self.assertEqual(trace_small.assignments[rank][group]['pair'].get_table(), pair_table.get_table())

    def test2d_characterize_rank_groups(self):
        # Test characterizing single (single processed)
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                            group_assignments=self.assignments_small, position_specific=True, pair_specific=False)
        trace_small.characterize_rank_groups()
        for rank in trace_small.assignments:
            for group in trace_small.assignments[rank]:
                sub_aln = self.query_aln_fa_small.generate_sub_alignment(
                    sequence_ids=trace_small.assignments[rank][group]['terminals'])
                single_table, pair_table = sub_aln.characterize_positions(single=True, pair=False)
                self.assertTrue('single' in trace_small.assignments[rank][group])
                self.assertEqual(trace_small.assignments[rank][group]['single'].get_table(), single_table.get_table())
                self.assertTrue('pair' in trace_small.assignments[rank][group])
                self.assertIsNone(trace_small.assignments[rank][group]['pair'])

    def test2e_characterize_rank_groups(self):
        # Test characterizing pair positions (single processed)
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                            group_assignments=self.assignments_small, position_specific=False, pair_specific=True)
        trace_small.characterize_rank_groups()
        for rank in trace_small.assignments:
            for group in trace_small.assignments[rank]:
                sub_aln = self.query_aln_fa_small.generate_sub_alignment(
                    sequence_ids=trace_small.assignments[rank][group]['terminals'])
                single_table, pair_table = sub_aln.characterize_positions(single=True, pair=True)
                self.assertTrue('single' in trace_small.assignments[rank][group])
                self.assertIsNone(trace_small.assignments[rank][group]['single'])
                self.assertTrue('pair' in trace_small.assignments[rank][group])
                self.assertEqual(trace_small.assignments[rank][group]['pair'].get_table(), pair_table.get_table())

    def test3a_trace(self):
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
        print(rank_ids)
        print(et_mip_obj.rank_scores)
        diff_ranks = rank_ids - et_mip_obj.rank_scores
        print(diff_ranks)
        self.assertTrue(not diff_ranks.any())

    def test3b_trace(self):
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
        print(rank_ids)
        print(et_mip_obj.rank_scores)
        diff_ranks = rank_ids - et_mip_obj.rank_scores
        print(diff_ranks)
        self.assertTrue(not diff_ranks.any())
