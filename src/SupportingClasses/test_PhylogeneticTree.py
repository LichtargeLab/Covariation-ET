import os
import numpy as np
from time import time
from re import compile
from shutil import rmtree
from copy import deepcopy
from test_Base import TestBase
from ETMIPWrapper import ETMIPWrapper
from SeqAlignment import SeqAlignment
from PhylogeneticTree import PhylogeneticTree, get_path_length
from AlignmentDistanceCalculator import AlignmentDistanceCalculator


class TestPhylogeneticTree(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestPhylogeneticTree, cls).setUpClass()
        cls.query_aln_fa_small = SeqAlignment(file_name=cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln'],
                                               query_id=cls.small_structure_id)
        cls.query_aln_fa_small.import_alignment()
        cls.query_aln_fa_large = SeqAlignment(file_name=cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln'],
                                               query_id=cls.large_structure_id)
        cls.query_aln_fa_large.import_alignment()
        cls.query_aln_msf_small = deepcopy(cls.query_aln_fa_small)
        cls.query_aln_msf_small.file_name = cls.data_set.protein_data[cls.small_structure_id]['Final_MSF_Aln']
        cls.query_aln_msf_large = deepcopy(cls.query_aln_fa_large)
        cls.query_aln_msf_large.file_name = cls.data_set.protein_data[cls.large_structure_id]['Final_MSF_Aln']

    def tearDown(self):
        if os.path.exists('./identity.pkl'):
            os.remove('./identity.pkl')
        cache_dir = os.path.join(self.testing_dir, 'joblib')
        if os.path.exists(cache_dir):
            rmtree(cache_dir)
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test')
        # if os.path.exists(wetc_test_dir):
        #     rmtree(wetc_test_dir)

    def evaluate_internal_nodes(self, internal_nodes):
        non_terminal_pattern = compile(r'^Inner(\d+)$')
        nodes = set()
        for n in internal_nodes:
            match = non_terminal_pattern.match(n.name)
            self.assertIsNotNone(match)
            self.assertNotIn(match.group(1), nodes)
            nodes.add(match.group(1))

    def evaluate_leaf_nodes(self, leaf_nodes, aln):
        leaves = set()
        self.assertEqual(len(leaf_nodes), aln.size)
        for l in leaf_nodes:
            self.assertIn(l.name, aln.seq_order)
            self.assertNotIn(l.name, leaves)
            leaves.add(l)

    def get_path_length(self, node, tree):
        path = tree.get_path(node)
        dist = 0
        for node in path:
            dist += node.branch_length
        return dist

    def evaluate_top_down_traversal(self, phylo_tree):
        last_dist = 0.0
        for node in phylo_tree.traverse_top_down():
            dist = self.get_path_length(node, phylo_tree.tree)
            self.assertGreaterEqual(dist, last_dist)
            last_dist = dist

    def evaluate_bottom_up_traversal(self, phylo_tree):
        last_dist = phylo_tree.tree.root.total_branch_length()
        for node in phylo_tree.traverse_bottom_up():
            dist = self.get_path_length(node, phylo_tree.tree)
            self.assertLessEqual(dist, last_dist)
            last_dist = dist

    @staticmethod
    def build_node_groups(phylo_tree):
        node_groups = []
        curr_nodes = []
        prev_dist = 0
        for node in phylo_tree.traverse_top_down():
            path_length = get_path_length(phylo_tree.tree.get_path(node))
            if prev_dist != path_length:
                node_groups.append(curr_nodes)
                curr_nodes = []
                prev_dist = path_length
            curr_nodes.append(node)
        return node_groups

    def check_nodes(self, node1, node2):
        if node1.is_terminal():
            self.assertTrue(node2.is_terminal())
            self.assertEqual(node1.name, node2.name)
        else:
            self.assertTrue(node2.is_bifurcating())
            self.assertEqual(set([x.name for x in node1.get_terminals()]),
                             set([x.name for x in node2.get_terminals()]))

    def check_lists_of_nodes_for_equality(self, list1, list2):
        self.assertEqual(len(list1), len(list2))
        for i in range(len(list1)):
            group1 = list1[i]
            group1 = sorted(group1, cmp=compare_nodes)
            group2 = list2[i]
            group2 = sorted(group2, cmp=compare_nodes)
            self.assertEqual(len(group1), len(group2))
            for j in range(len(group1)):
                node1 = group1[j]
                node2 = group2[j]
                self.check_nodes(node1=node1, node2=node2)

    def evaluate_rank(self, rank, dict1, dict2):
        d1_keys = set(dict1.keys())
        d2_keys = set(dict2.keys())
        for k1 in dict1:
            print('Group1: {}'.format(k1))
            if k1 not in d1_keys:
                continue
            node1 = dict1[k1]['node']
            terminals1 = dict1[k1]['terminals']
            for k2 in dict2:
                print('Group2: {}'.format(k2))
                if k2 not in d2_keys:
                    continue
                node2 = dict2[k2]['node']
                terminals2 = dict2[k2]['terminals']
                if set(terminals1) == set(terminals2):
                    self.check_nodes(node1, node2)
                else:
                    continue
                d1_keys.remove(k1)
                d2_keys.remove(k2)
                break
            print(len(d1_keys))
            print(d1_keys)
            if len(d1_keys) != rank - k1:
                for k in d1_keys:
                    print(dict1[k]['node'])
                    print(dict1[k]['terminals'])
                for k in d2_keys:
                    print(dict2[k]['node'])
                    print(dict2[k]['terminals'])
            self.assertEqual(len(d1_keys), len(d2_keys))
            self.assertEqual(len(d1_keys), rank - k1)

    def test1_init(self):
        phylo_tree = PhylogeneticTree()
        self.assertEqual(phylo_tree.tree_method, 'upgma')
        self.assertEqual(phylo_tree.tree_args, {})
        self.assertIsNone(phylo_tree.tree)

    def test2a_upgma_tree_small(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.assertIsNotNone(phylo_tree.tree)
        non_terminal_nodes = phylo_tree.tree.get_nonterminals()
        self.assertEqual(len(non_terminal_nodes), self.query_aln_fa_small.size - 1)
        self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
        terminal_nodes = phylo_tree.tree.get_terminals()
        self.evaluate_leaf_nodes(terminal_nodes, self.query_aln_fa_small)

    def test2b_upgma_tree_large(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.assertIsNotNone(phylo_tree.tree)
        non_terminal_nodes = phylo_tree.tree.get_nonterminals()
        self.assertEqual(len(non_terminal_nodes), self.query_aln_fa_large.size - 1)
        self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
        terminal_nodes = phylo_tree.tree.get_terminals()
        self.evaluate_leaf_nodes(terminal_nodes, self.query_aln_fa_large)

    def test3a_agglomerative_tree_small(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.assertIsNotNone(phylo_tree.tree)
        non_terminal_nodes = phylo_tree.tree.get_nonterminals()
        self.assertEqual(len(non_terminal_nodes), self.query_aln_fa_small.size - 1)
        self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
        terminal_nodes = phylo_tree.tree.get_terminals()
        self.evaluate_leaf_nodes(terminal_nodes, self.query_aln_fa_small)

    def test3b_agglomerative_tree_large(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.assertIsNotNone(phylo_tree.tree)
        non_terminal_nodes = phylo_tree.tree.get_nonterminals()
        self.assertEqual(len(non_terminal_nodes), self.query_aln_fa_large.size - 1)
        self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
        terminal_nodes = phylo_tree.tree.get_terminals()
        self.evaluate_leaf_nodes(terminal_nodes, self.query_aln_fa_large)

    def test4a_custom_tree_small(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=self.query_aln_msf_small.seq_order)
        self.assertIsNotNone(phylo_tree.tree)
        non_terminal_nodes = phylo_tree.tree.get_nonterminals()
        self.assertEqual(len(non_terminal_nodes), self.query_aln_fa_small.size - 1)
        self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
        terminal_nodes = phylo_tree.tree.get_terminals()
        self.evaluate_leaf_nodes(leaf_nodes=terminal_nodes, aln=self.query_aln_fa_small)

    def test4b_custom_tree_large(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=self.query_aln_msf_large.seq_order)
        self.assertIsNotNone(phylo_tree.tree)
        non_terminal_nodes = phylo_tree.tree.get_nonterminals()
        self.assertEqual(len(non_terminal_nodes), self.query_aln_fa_large.size - 1)
        self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
        terminal_nodes = phylo_tree.tree.get_terminals()
        self.evaluate_leaf_nodes(leaf_nodes=terminal_nodes, aln=self.query_aln_fa_large)

    def test5a_traverse_top_down(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_top_down_traversal(phylo_tree)

    def test5b_traverse_top_down(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_top_down_traversal(phylo_tree)

    def test5c_traverse_top_down(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=self.query_aln_msf_small.seq_order)
        self.evaluate_top_down_traversal(phylo_tree)

    def test5d_traverse_top_down(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_top_down_traversal(phylo_tree)

    def test5e_traverse_top_down(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_top_down_traversal(phylo_tree)

    def test5f_traverse_top_down(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=self.query_aln_msf_large.seq_order)
        self.evaluate_top_down_traversal(phylo_tree)

    def test6a_traverse_bottom_up(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test6b_traverse_bottom_up(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test6c_traverse_bottom_up(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=self.query_aln_msf_small.seq_order)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test6d_traverse_bottom_up(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test6e_traverse_bottom_up(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test6f_traverse_bottom_up(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=self.query_aln_msf_large.seq_order)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test7a_build_ETC_tree_small(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        et_mip_obj.import_phylogenetic_tree(out_dir=wetc_test_dir)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
        etc_nodes = self.build_node_groups(et_mip_obj.tree)
        print(etc_nodes)
        phylo_nodes = self.build_node_groups(phylo_tree)
        print(phylo_nodes)
        self.check_lists_of_nodes_for_equality(etc_nodes, phylo_nodes)

    # def test7b_bulid_ETC_tree_small(self):
    #     wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
    #     if not os.path.isdir(wetc_test_dir):
    #         os.makedirs(wetc_test_dir)
    #     et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
    #     et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
    #     et_mip_obj.import_phylogenetic_tree(out_dir=wetc_test_dir)
    #     calculator = AlignmentDistanceCalculator(model='blosum62')
    #     _, dm, _, _ = calculator.get_et_distance(self.query_aln_fa_small.alignment)
    #     phylo_tree = PhylogeneticTree()
    #     phylo_tree.construct_tree(dm=dm)
    #     etc_nodes = self.build_node_groups(et_mip_obj.tree)
    #     print(etc_nodes)
    #     phylo_nodes = self.build_node_groups(phylo_tree)
    #     print(phylo_nodes)
    #     self.check_lists_of_nodes_for_equality(etc_nodes, phylo_nodes)

    def test7c_build_ETC_tree_large(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_phylogenetic_tree(out_dir=wetc_test_dir)
        calculator = AlignmentDistanceCalculator(model='blosum62')
        _, dm, _, _ = calculator.get_et_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        etc_nodes = self.build_node_groups(et_mip_obj.tree)
        phylo_nodes = self.build_node_groups(phylo_tree)
        self.check_lists_of_nodes_for_equality(etc_nodes, phylo_nodes)

    def test8_write_out_tree(self):
        nhx_fn = os.path.join(self.testing_dir, 'UPGMA_Newcki_tree.nhx')
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        self.assertFalse(os.path.isfile(nhx_fn))
        with self.assertRaises(ValueError):
            phylo_tree.write_out_tree(filename=nhx_fn)
        self.assertFalse(os.path.isfile(nhx_fn))
        phylo_tree.construct_tree(dm=dm)
        phylo_tree.write_out_tree(filename=nhx_fn)
        self.assertTrue(os.path.isfile(nhx_fn))
        loaded_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_fn})
        loaded_tree.construct_tree(dm=self.query_aln_msf_small.seq_order)
        phylo_nodes = self.build_node_groups(phylo_tree)
        loaded_nodes = self.build_node_groups(loaded_tree)
        self.check_lists_of_nodes_for_equality(list1=phylo_nodes, list2=loaded_nodes)
        os.remove(nhx_fn)

    def test9a_assign_rank_group_small(self):
        expected_terminals = set(self.query_aln_fa_small.seq_order)
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        assignment = phylo_tree.assign_group_rank()
        previous_nodes = set([])
        for rank in range(1, self.query_aln_fa_small.size + 1):
            self.assertTrue(rank in assignment)
            all_roots = set([])
            all_terminals = []
            for group in range(1, rank + 1):
                self.assertTrue(group in assignment[rank])
                self.assertTrue('node' in assignment[rank][group])
                all_roots.add(assignment[rank][group]['node'])
                self.assertTrue('terminals' in assignment[rank][group])
                all_terminals += assignment[rank][group]['terminals']
            self.assertEqual(len(all_roots), rank)
            unique_to_prev = previous_nodes - all_roots
            unique_to_curr = all_roots - previous_nodes
            if rank == 1:
                self.assertEqual(len(unique_to_prev), 0)
                self.assertEqual(len(unique_to_curr), 1)
            else:
                self.assertEqual(len(unique_to_prev), 1)
                self.assertEqual(len(unique_to_curr), 2)
            self.assertEqual(len(all_terminals), self.query_aln_fa_small.size)
            all_terminals_set = set(all_terminals)
            self.assertEqual(len(all_terminals), len(all_terminals_set))
            self.assertTrue(expected_terminals == all_terminals_set)
            previous_nodes = all_roots

    def test9b_assign_rank_group_large(self):
        expected_terminals = set(self.query_aln_fa_large.seq_order)
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        assignment = phylo_tree.assign_group_rank()
        previous_nodes = set([])
        for rank in range(1, self.query_aln_fa_large.size + 1):
            self.assertTrue(rank in assignment)
            all_roots = set([])
            all_terminals = []
            for group in range(1, rank + 1):
                self.assertTrue(group in assignment[rank])
                self.assertTrue('node' in assignment[rank][group])
                all_roots.add(assignment[rank][group]['node'])
                self.assertTrue('terminals' in assignment[rank][group])
                all_terminals += assignment[rank][group]['terminals']
            self.assertEqual(len(all_roots), rank)
            unique_to_prev = previous_nodes - all_roots
            unique_to_curr = all_roots - previous_nodes
            if rank == 1:
                self.assertEqual(len(unique_to_prev), 0)
                self.assertEqual(len(unique_to_curr), 1)
            else:
                self.assertEqual(len(unique_to_prev), 1)
                self.assertEqual(len(unique_to_curr), 2)
            self.assertEqual(len(all_terminals), self.query_aln_fa_large.size)
            all_terminals_set = set(all_terminals)
            self.assertEqual(len(all_terminals), len(all_terminals_set))
            self.assertTrue(expected_terminals == all_terminals_set)
            previous_nodes = all_roots

    def test10a_compare_assignments_small(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        start = time()
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        inter1 = time()
        print('Initializatoin took {} min'.format((inter1 - start) / 60.0))
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        inter2 = time()
        print('Score calculation took {} min'.format((inter2 - inter1) / 60.0))
        et_mip_obj.import_phylogenetic_tree(out_dir=wetc_test_dir)
        inter3 = time()
        print('Tree construction too {} min'.format((inter3 - inter2) / 60.0))
        et_mip_obj.import_assignments(out_dir=wetc_test_dir)
        end = time()
        print('Assignment import took {} min'.format((end - inter3) / 60.0))
        assignments = et_mip_obj.tree.assign_group_rank()
        print(assignments)
        print(et_mip_obj.rank_group_assignments)
        self.assertEqual(len(assignments.keys()), len(et_mip_obj.rank_group_assignments.keys()))
        for rank in assignments:
            self.assertTrue(rank in et_mip_obj.rank_group_assignments)
            self.assertEqual(len(assignments[rank].keys()), len(et_mip_obj.rank_group_assignments[rank].keys()))
            self.evaluate_rank(rank, assignments[rank], et_mip_obj.rank_group_assignments[rank])

    def test10b_compare_assignments_large(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        start = time()
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        inter1 = time()
        print('Initializatoin took {} min'.format((inter1 - start) / 60.0))
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        inter2 = time()
        print('Score calculation took {} min'.format((inter2 - inter1) / 60.0))
        et_mip_obj.import_phylogenetic_tree(out_dir=wetc_test_dir)
        assignments = et_mip_obj.tree.assign_group_rank()
        inter3 = time()
        print('Tree construction too {} min'.format((inter3 - inter2) / 60.0))
        et_mip_obj.import_assignments(out_dir=wetc_test_dir)
        end = time()
        print('Assignment import took {} min'.format((end - inter3) / 60.0))
        self.assertEqual(len(assignments.keys()), len(et_mip_obj.rank_group_assignments.keys()))
        for rank in assignments:
            self.assertTrue(rank in et_mip_obj.rank_group_assignments)
            self.assertEqual(len(assignments[rank].keys()), len(et_mip_obj.rank_group_assignments[rank].keys()))
            self.evaluate_rank(rank, assignments[rank], et_mip_obj.rank_group_assignments[rank])

def compare_nodes(node1, node2):
    if node1.is_terminal and not node2.is_terminal():
        return -1
    elif not node1.is_terminal() and node2.is_terminal():
        return 1
    else:
        if node1.name < node2.name:
            return 1
        elif node1.name > node2.name:
            return -1
        else:
            return 0
