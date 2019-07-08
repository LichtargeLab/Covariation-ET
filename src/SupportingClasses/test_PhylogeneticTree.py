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

    def test1_init(self):
        phylo_tree = PhylogeneticTree()
        self.assertEqual(phylo_tree.tree_method, 'upgma')
        self.assertEqual(phylo_tree.tree_args, {})
        self.assertIsNone(phylo_tree.distance_matrix)
        self.assertIsNone(phylo_tree.tree)
        self.assertIsNone(phylo_tree.size)

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
        for l in leaf_nodes:
            self.assertIn(l.name, aln.seq_order)
            self.assertNotIn(l.name, leaves)
            leaves.add(l.name)

    def test2a__custom_tree_small(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        self.assertEqual(phylo_tree.tree_method, 'custom')
        self.assertEqual(phylo_tree.tree_args, {'tree_path': nhx_path})
        self.assertIsNone(phylo_tree.distance_matrix)
        self.assertIsNone(phylo_tree.tree)
        self.assertIsNone(phylo_tree.size)
        phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
        self.assertEqual(phylo_tree.tree_method, 'custom')
        self.assertEqual(phylo_tree.tree_args, {'tree_path': nhx_path})
        self.assertEqual(phylo_tree.distance_matrix, et_mip_obj.distance_matrix)
        self.assertEqual(phylo_tree.size, self.query_aln_msf_small.size)
        self.assertIsNotNone(phylo_tree.tree)
        non_terminal_nodes = phylo_tree.tree.get_nonterminals()
        self.assertEqual(len(non_terminal_nodes), self.query_aln_msf_small.size - 1)
        self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
        terminal_nodes = phylo_tree.tree.get_terminals()
        self.assertEqual(len(terminal_nodes), phylo_tree.size)
        self.evaluate_leaf_nodes(leaf_nodes=terminal_nodes, aln=self.query_aln_msf_small)

    def test2b__custom_tree_large(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        self.assertEqual(phylo_tree.tree_method, 'custom')
        self.assertEqual(phylo_tree.tree_args, {'tree_path': nhx_path})
        self.assertIsNone(phylo_tree.distance_matrix)
        self.assertIsNone(phylo_tree.tree)
        self.assertIsNone(phylo_tree.size)
        phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
        self.assertEqual(phylo_tree.tree_method, 'custom')
        self.assertEqual(phylo_tree.tree_args, {'tree_path': nhx_path})
        self.assertEqual(phylo_tree.distance_matrix, et_mip_obj.distance_matrix)
        self.assertEqual(phylo_tree.size, self.query_aln_msf_large.size)
        self.assertIsNotNone(phylo_tree.tree)
        non_terminal_nodes = phylo_tree.tree.get_nonterminals()
        self.assertEqual(len(non_terminal_nodes), self.query_aln_msf_large.size - 1)
        self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
        terminal_nodes = phylo_tree.tree.get_terminals()
        self.assertEqual(len(terminal_nodes), phylo_tree.size)
        self.evaluate_leaf_nodes(leaf_nodes=terminal_nodes, aln=self.query_aln_msf_large)

    def test3a__upgma_tree_small(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        self.assertEqual(phylo_tree.tree_method, 'upgma')
        self.assertEqual(phylo_tree.tree_args, {})
        self.assertIsNone(phylo_tree.distance_matrix)
        self.assertIsNone(phylo_tree.tree)
        self.assertIsNone(phylo_tree.size)
        phylo_tree.construct_tree(dm=dm)
        self.assertEqual(phylo_tree.tree_method, 'upgma')
        self.assertEqual(phylo_tree.tree_args, {})
        self.assertEqual(phylo_tree.distance_matrix, dm)
        self.assertEqual(phylo_tree.size, self.query_aln_fa_small.size)
        self.assertIsNotNone(phylo_tree.tree)
        non_terminal_nodes = phylo_tree.tree.get_nonterminals()
        self.assertEqual(len(non_terminal_nodes), self.query_aln_fa_small.size - 1)
        self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
        terminal_nodes = phylo_tree.tree.get_terminals()
        self.assertEqual(len(terminal_nodes), self.query_aln_fa_small.size)
        self.evaluate_leaf_nodes(terminal_nodes, self.query_aln_fa_small)

    def test3b__upgma_tree_large(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        self.assertEqual(phylo_tree.tree_method, 'upgma')
        self.assertEqual(phylo_tree.tree_args, {})
        self.assertIsNone(phylo_tree.distance_matrix)
        self.assertIsNone(phylo_tree.tree)
        self.assertIsNone(phylo_tree.size)
        phylo_tree.construct_tree(dm=dm)
        self.assertEqual(phylo_tree.tree_method, 'upgma')
        self.assertEqual(phylo_tree.tree_args, {})
        self.assertEqual(phylo_tree.distance_matrix, dm)
        self.assertEqual(phylo_tree.size, self.query_aln_fa_large.size)
        self.assertIsNotNone(phylo_tree.tree)
        non_terminal_nodes = phylo_tree.tree.get_nonterminals()
        self.assertEqual(len(non_terminal_nodes), self.query_aln_fa_large.size - 1)
        self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
        terminal_nodes = phylo_tree.tree.get_terminals()
        self.assertEqual(len(terminal_nodes), self.query_aln_fa_large.size)
        self.evaluate_leaf_nodes(terminal_nodes, self.query_aln_fa_large)

    def test4a_agglomerative_tree_small(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        self.assertEqual(phylo_tree.tree_method, 'agglomerative')
        self.assertEqual(phylo_tree.tree_args, {'affinity': 'euclidean', 'linkage': 'ward',
                                                'cache_dir': self.testing_dir})
        self.assertIsNone(phylo_tree.distance_matrix)
        self.assertIsNone(phylo_tree.tree)
        self.assertIsNone(phylo_tree.size)
        phylo_tree.construct_tree(dm=dm)
        self.assertEqual(phylo_tree.tree_method, 'agglomerative')
        self.assertEqual(phylo_tree.tree_args, {'affinity': 'euclidean', 'linkage': 'ward',
                                                'cache_dir': self.testing_dir})
        self.assertEqual(phylo_tree.distance_matrix, dm)
        self.assertEqual(phylo_tree.size, self.query_aln_fa_small.size)
        self.assertIsNotNone(phylo_tree.tree)
        non_terminal_nodes = phylo_tree.tree.get_nonterminals()
        self.assertEqual(len(non_terminal_nodes), self.query_aln_fa_small.size - 1)
        self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
        terminal_nodes = phylo_tree.tree.get_terminals()
        self.assertEqual(len(terminal_nodes), self.query_aln_fa_small.size)
        self.evaluate_leaf_nodes(terminal_nodes, self.query_aln_fa_small)

    def test4b_agglomerative_tree_large(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        self.assertEqual(phylo_tree.tree_method, 'agglomerative')
        self.assertEqual(phylo_tree.tree_args, {'affinity': 'euclidean', 'linkage': 'ward',
                                                'cache_dir': self.testing_dir})
        self.assertIsNone(phylo_tree.distance_matrix)
        self.assertIsNone(phylo_tree.tree)
        self.assertIsNone(phylo_tree.size)
        phylo_tree.construct_tree(dm=dm)
        self.assertEqual(phylo_tree.tree_method, 'agglomerative')
        self.assertEqual(phylo_tree.tree_args, {'affinity': 'euclidean', 'linkage': 'ward',
                                                'cache_dir': self.testing_dir})
        self.assertEqual(phylo_tree.distance_matrix, dm)
        self.assertEqual(phylo_tree.size, self.query_aln_fa_large.size)
        self.assertIsNotNone(phylo_tree.tree)
        non_terminal_nodes = phylo_tree.tree.get_nonterminals()
        self.assertEqual(len(non_terminal_nodes), self.query_aln_fa_large.size - 1)
        self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
        terminal_nodes = phylo_tree.tree.get_terminals()
        self.assertEqual(len(terminal_nodes), self.query_aln_fa_large.size)
        self.evaluate_leaf_nodes(terminal_nodes, self.query_aln_fa_large)

    # Construct tree tested throughout each of the specific construction method tests.

    def check_nodes(self, node1, node2):
        if node1.is_terminal():
            self.assertTrue(node2.is_terminal())
            self.assertEqual(node1.name, node2.name)
        else:
            self.assertTrue(node2.is_bifurcating())
            self.assertFalse(node2.is_terminal())
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

    def test5a_write_out_tree(self):
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
        loaded_tree.construct_tree(dm=dm)
        self.assertEqual(phylo_tree.distance_matrix, loaded_tree.distance_matrix)
        self.assertEqual(phylo_tree.size, loaded_tree.size)
        phylo_nodes = list(phylo_tree.traverse_by_rank())
        loaded_nodes = list(loaded_tree.traverse_by_rank())
        self.check_lists_of_nodes_for_equality(list1=phylo_nodes, list2=loaded_nodes)
        os.remove(nhx_fn)

    def test5b_write_out_tree(self):
        nhx_fn = os.path.join(self.testing_dir, 'UPGMA_Newcki_tree.nhx')
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        self.assertFalse(os.path.isfile(nhx_fn))
        with self.assertRaises(ValueError):
            phylo_tree.write_out_tree(filename=nhx_fn)
        self.assertFalse(os.path.isfile(nhx_fn))
        phylo_tree.construct_tree(dm=dm)
        phylo_tree.write_out_tree(filename=nhx_fn)
        self.assertTrue(os.path.isfile(nhx_fn))
        loaded_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_fn})
        loaded_tree.construct_tree(dm=dm)
        self.assertEqual(phylo_tree.distance_matrix, loaded_tree.distance_matrix)
        self.assertEqual(phylo_tree.size, loaded_tree.size)
        phylo_nodes = list(phylo_tree.traverse_by_rank())
        loaded_nodes = list(loaded_tree.traverse_by_rank())
        self.check_lists_of_nodes_for_equality(list1=phylo_nodes, list2=loaded_nodes)
        os.remove(nhx_fn)

    def get_path_length(self, node, tree):
        path = tree.get_path(node)
        dist = 0
        for node in path:
            dist += node.branch_length
        return dist

    def evaluate_top_down_traversal(self, phylo_tree):
        node_names = set()
        last_dist = 0.0
        for node in phylo_tree.traverse_top_down():
            self.assertNotIn(node.name, node_names)
            node_names.add(node.name)
            dist = self.get_path_length(node, phylo_tree.tree)
            self.assertGreaterEqual(dist, last_dist)
            last_dist = dist
        self.assertEqual(len(node_names), (phylo_tree.size * 2) - 1)

    def test6a_traverse_top_down(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_top_down_traversal(phylo_tree)

    def test6b_traverse_top_down(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_top_down_traversal(phylo_tree)

    def test6c_traverse_top_down(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
        self.evaluate_top_down_traversal(phylo_tree)

    def test6d_traverse_top_down(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_top_down_traversal(phylo_tree)

    def test6e_traverse_top_down(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_top_down_traversal(phylo_tree)

    def test6f_traverse_top_down(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
        self.evaluate_top_down_traversal(phylo_tree)

    def evaluate_bottom_up_traversal(self, phylo_tree):
        last_dist = phylo_tree.tree.root.total_branch_length()
        node_names = set()
        for node in phylo_tree.traverse_bottom_up():
            self.assertNotIn(node.name, node_names)
            node_names.add(node.name)
            dist = self.get_path_length(node, phylo_tree.tree)
            self.assertLessEqual(dist, last_dist)
            last_dist = dist
        self.assertEqual(len(node_names), (phylo_tree.size * 2) - 1)

    def test7a_traverse_bottom_up(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test7b_traverse_bottom_up(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test7c_traverse_bottom_up(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test7d_traverse_bottom_up(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test7e_traverse_bottom_up(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test7f_traverse_bottom_up(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def evaluate_by_rank_traversal(self, tree):
        nodes_added = set([])
        nodes_removed = set([])
        prev_rank = None
        rank_count = 1
        for rank in tree.traverse_by_rank():
            # Make sure the current rank has the size of the rank counter (starting with size 1 for the first rank, i.e.
            # root node only).
            self.assertEqual(len(rank), rank_count)
            if prev_rank:
                # Make sure that the two ranks differ in size by 1
                self.assertEqual(len(rank) - len(prev_rank), 1)
                # Determine the differences between the current and previous rank
                rank_set = set(rank)
                prev_set = set(prev_rank)
                # Make sure there are two new nodes in the current rank and that they have not appeared in any previous
                # rank.
                new_nodes = rank_set - prev_set
                self.assertEqual(len(new_nodes), 2)
                self.assertEqual(len(nodes_added.intersection(new_nodes)), 0)
                nodes_added |= new_nodes
                # Make sure only one node was lost (i.e. branched) from the previous rank and that it had never been
                # previously removed.
                old_nodes = prev_set - rank_set
                self.assertEqual(len(old_nodes), 1)
                self.assertEqual(len(nodes_removed.intersection(old_nodes)), 0)
                nodes_removed |= old_nodes
            prev_rank = rank
            rank_count += 1
        # Make sure that the final rank processed was the size of number of sequences the tree was built from.
        self.assertEqual(rank_count - 1, tree.size)

    def test8a_traverse_by_rank(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_by_rank_traversal(tree=phylo_tree)

    def test8b_traverse_by_rank(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_by_rank_traversal(tree=phylo_tree)

    def test8c_traverse_by_rank(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        et_mip_obj.import_phylogenetic_tree(out_dir=wetc_test_dir)
        self.evaluate_by_rank_traversal(tree=et_mip_obj.tree)

    def test8d_traverse_by_rank(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_by_rank_traversal(tree=phylo_tree)
        self.validate_upgma_tree(tree=phylo_tree, dm=dm)

    def test8e_traverse_by_rank(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_by_rank_traversal(tree=phylo_tree)

    def test8f_traverse_by_rank(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        et_mip_obj.import_phylogenetic_tree(out_dir=wetc_test_dir)
        self.evaluate_by_rank_traversal(tree=et_mip_obj.tree)

    def test9a_rename_internal_nodes(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
        self.assertEqual(phylo_tree.tree.root.name, 'Inner1')
        non_terminal_nodes = {i.name: i for i in phylo_tree.tree.get_nonterminals()}
        terminal_nodes = {t.name: t for t in phylo_tree.tree.get_terminals()}
        phylo_tree.rename_internal_nodes()
        for nt_name in non_terminal_nodes:
            self.assertEqual(nt_name, non_terminal_nodes[nt_name].name)
        for t_name in terminal_nodes:
            self.assertEqual(t_name, terminal_nodes[t_name].name)

    def test9b_rename_internal_nodes(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
        self.assertEqual(phylo_tree.tree.root.name, 'Inner1')
        non_terminal_nodes = {i.name: i for i in phylo_tree.tree.get_nonterminals()}
        terminal_nodes = {t.name: t for t in phylo_tree.tree.get_terminals()}
        phylo_tree.rename_internal_nodes()
        for nt_name in non_terminal_nodes:
            self.assertEqual(nt_name, non_terminal_nodes[nt_name].name)
        for t_name in terminal_nodes:
            self.assertEqual(t_name, terminal_nodes[t_name].name)

    def test9c_rename_internal_nodes(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.assertNotEqual(phylo_tree.tree.root.name, 'Inner1')
        non_terminal_nodes = {i.name: i for i in phylo_tree.tree.get_nonterminals()}
        terminal_nodes = {t.name: t for t in phylo_tree.tree.get_terminals()}
        phylo_tree.rename_internal_nodes()
        self.assertEqual(phylo_tree.tree.root.name, 'Inner1')
        count_equal = 0
        count_not_equal = 0
        for nt_name in non_terminal_nodes:
            if nt_name == non_terminal_nodes[nt_name].name:
                count_equal += 1
            else:
                count_not_equal += 1
        self.assertGreaterEqual(count_not_equal, count_equal)
        for t_name in terminal_nodes:
            self.assertEqual(t_name, terminal_nodes[t_name].name)

    def test9d_rename_internal_nodes(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.assertNotEqual(phylo_tree.tree.root.name, 'Inner1')
        non_terminal_nodes = {i.name: i for i in phylo_tree.tree.get_nonterminals()}
        terminal_nodes = {t.name: t for t in phylo_tree.tree.get_terminals()}
        phylo_tree.rename_internal_nodes()
        self.assertEqual(phylo_tree.tree.root.name, 'Inner1')
        count_equal = 0
        count_not_equal = 0
        for nt_name in non_terminal_nodes:
            if nt_name == non_terminal_nodes[nt_name].name:
                count_equal += 1
            else:
                count_not_equal += 1
        self.assertGreaterEqual(count_not_equal, count_equal)
        for t_name in terminal_nodes:
            self.assertEqual(t_name, terminal_nodes[t_name].name)

    def test9e_rename_internal_nodes(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.assertEqual(phylo_tree.tree.root.name, 'Inner1')
        non_terminal_nodes = {i.name: i for i in phylo_tree.tree.get_nonterminals()}
        terminal_nodes = {t.name: t for t in phylo_tree.tree.get_terminals()}
        phylo_tree.rename_internal_nodes()
        for nt_name in non_terminal_nodes:
            self.assertEqual(nt_name, non_terminal_nodes[nt_name].name)
        for t_name in terminal_nodes:
            self.assertEqual(t_name, terminal_nodes[t_name].name)

    def test9f_rename_internal_nodes(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.assertEqual(phylo_tree.tree.root.name, 'Inner1')
        non_terminal_nodes = {i.name: i for i in phylo_tree.tree.get_nonterminals()}
        terminal_nodes = {t.name: t for t in phylo_tree.tree.get_terminals()}
        phylo_tree.rename_internal_nodes()
        for nt_name in non_terminal_nodes:
            self.assertEqual(nt_name, non_terminal_nodes[nt_name].name)
        for t_name in terminal_nodes:
            self.assertEqual(t_name, terminal_nodes[t_name].name)

    def evaulate_rank_group_assignments(self, assignment, alignment):
        expected_terminals = set(alignment.seq_order)
        previous_nodes = set([])
        for rank in range(1, alignment.size + 1):
            # Make sure that each rank is represented in the assignment
            self.assertTrue(rank in assignment)
            all_roots = set([])
            all_terminals = []
            for group in range(1, rank + 1):
                # Make sure that each group is present in the rank assignments
                self.assertTrue(group in assignment[rank])
                # Make sure that each rank/group assignment has a root node assignment
                self.assertTrue('node' in assignment[rank][group])
                all_roots.add(assignment[rank][group]['node'])
                # Make sure that each rank/group assignment has a list of associated terminal nodes
                self.assertTrue('terminals' in assignment[rank][group])
                all_terminals += assignment[rank][group]['terminals']
            # Make sure that all root nodes at a given rank or unique and have a count equal to the rank
            self.assertEqual(len(all_roots), rank)
            # Make sure the correct number of nodes branch at each rank
            unique_to_prev = previous_nodes - all_roots
            unique_to_curr = all_roots - previous_nodes
            if rank == 1:
                self.assertEqual(len(unique_to_prev), 0)
                self.assertEqual(len(unique_to_curr), 1)
            else:
                self.assertEqual(len(unique_to_prev), 1)
                self.assertEqual(len(unique_to_curr), 2)
            # Make sure that all sequences in the alignment are represented in the terminals of this rank
            self.assertEqual(len(all_terminals), alignment.size)
            all_terminals_set = set(all_terminals)
            # Make sure those terminal nodes are all unique
            self.assertEqual(len(all_terminals), len(all_terminals_set))
            # Explicitly check that all sequences in the alignment are in the set of terminal nodes
            self.assertTrue(expected_terminals == all_terminals_set)
            previous_nodes = all_roots

    def test10a_assign_rank_group_small(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
        assignments = phylo_tree.assign_group_rank()
        self.evaulate_rank_group_assignments(assignment=assignments, alignment=self.query_aln_msf_small)

    def test10b_assign_rank_group_small(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
        assignments = phylo_tree.assign_group_rank()
        self.evaulate_rank_group_assignments(assignment=assignments, alignment=self.query_aln_msf_large)

    def test10c_assign_rank_group_small(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        assignments = phylo_tree.assign_group_rank()
        self.evaulate_rank_group_assignments(assignment=assignments, alignment=self.query_aln_fa_small)

    def test10d_assign_rank_group_small(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        assignments = phylo_tree.assign_group_rank()
        self.evaulate_rank_group_assignments(assignment=assignments, alignment=self.query_aln_fa_large)

    def test10e_assign_rank_group_small(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        assignments = phylo_tree.assign_group_rank()
        self.evaulate_rank_group_assignments(assignment=assignments, alignment=self.query_aln_fa_small)

    def test10f_assign_rank_group_small(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        assignments = phylo_tree.assign_group_rank()
        self.evaulate_rank_group_assignments(assignment=assignments, alignment=self.query_aln_fa_large)

    def test10g_assign_rank_group_small(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        et_mip_obj.import_phylogenetic_tree(out_dir=wetc_test_dir, file_name='etc_out.nhx')
        et_mip_obj.import_assignments(out_dir=wetc_test_dir)
        self.evaulate_rank_group_assignments(assignment=et_mip_obj.rank_group_assignments,
                                             alignment=self.query_aln_msf_small)

    def test10h_assign_rank_group_small(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        et_mip_obj.import_phylogenetic_tree(out_dir=wetc_test_dir, file_name='etc_out.nhx')
        et_mip_obj.import_assignments(out_dir=wetc_test_dir)
        self.evaulate_rank_group_assignments(assignment=et_mip_obj.rank_group_assignments,
                                             alignment=self.query_aln_msf_large)

    def validate_upgma_tree(self, tree, dm, verbose=False):
        reverse_rank_traversal = list(tree.traverse_by_rank())[::-1]
        internal_dm = deepcopy(dm)
        count = 1
        while count < len(dm):
            position_node = {i: name for i, name in enumerate(internal_dm.names)}
            # Get nodes and their names for the previous rank
            prev_rank = reverse_rank_traversal[count - 1]
            prev_rank_names = set([node.name for node in prev_rank])
            # Get nodes and their names for the current rank
            curr_rank = reverse_rank_traversal[count]
            curr_rank_names = set([node.name for node in curr_rank])
            # Determine which nodes from the previous rank were joined to create a new node in the current rank (there
            # should always be two)
            joined_nodes = list(prev_rank_names - curr_rank_names)
            self.assertEqual(len(joined_nodes), 2)
            # Determine which node in the current rank was the product of joining nodes (there should be only one)
            resulting_node = list(curr_rank_names - prev_rank_names)
            self.assertEqual(len(resulting_node), 1)
            # Get the distance matrix positions the minimum score
            dm_array = np.array(internal_dm)
            min_score = np.min(dm_array[np.tril_indices(len(internal_dm), k=-1)])
            positions = np.where(dm_array == min_score)
            match = False
            if verbose:
                print('#' * 100)
                print('Current Rank: {}'.format(count))
                print(prev_rank_names)
                print(curr_rank_names)
                print(joined_nodes)
                print(resulting_node)
                print(internal_dm[joined_nodes[0], joined_nodes[1]])
                print(min_score)
                print(positions)
            for i in range(len(positions[0])):
                pos_i = positions[0][i]
                pos_j = positions[1][i]
                name_i = position_node[pos_i]
                name_j = position_node[pos_j]
                if verbose:
                    print(name_i)
                    print(name_j)
                    print(internal_dm[name_i, name_j])
                if name_i == joined_nodes[0] and name_j == joined_nodes[1]:
                    match = True
                    for k in range(len(internal_dm)):
                        if k != pos_i and k != pos_j:
                            internal_dm[pos_j, k] = (internal_dm[pos_i, k] + internal_dm[pos_j, k]) * 1.0 / 2
                    internal_dm.names[pos_j] = resulting_node[0]
                    del (internal_dm[pos_i])
                    break
            self.assertTrue(match, internal_dm)
            count += 1

    def test11a_validate_upgma_tree(self):
        calculator = AlignmentDistanceCalculator()
        _, dm, _, _ = calculator.get_et_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.validate_upgma_tree(tree=phylo_tree, dm=dm, verbose=False)

    def test11b_validate_upgma_tree(self):
        calculator = AlignmentDistanceCalculator()
        _, dm, _, _ = calculator.get_et_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.validate_upgma_tree(tree=phylo_tree, dm=dm, verbose=False)

    def test11c_validate_upgma_tree(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
        self.validate_upgma_tree(tree=phylo_tree, dm=et_mip_obj.distance_matrix, verbose=False)

    def test11d_validate_upgma_tree(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
        self.validate_upgma_tree(tree=phylo_tree, dm=et_mip_obj.distance_matrix, verbose=False)

    def test11e_validate_upgma_tree(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        et_mip_obj.import_phylogenetic_tree(out_dir=wetc_test_dir)
        self.validate_upgma_tree(tree=et_mip_obj.tree, dm=et_mip_obj.distance_matrix, verbose=True)

    def test11f_validate_upgma_tree(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        et_mip_obj.import_distance_matrices(out_dir=wetc_test_dir)
        et_mip_obj.import_phylogenetic_tree(out_dir=wetc_test_dir)
        self.validate_upgma_tree(tree=et_mip_obj.tree, dm=et_mip_obj.distance_matrix, verbose=True)

########################################################################################################################
########################################################################################################################

    def evaluate_rank(self, rank, dict1, dict2):
        # Logic for this test:
        # Iterate over the groups in both rank dictionaries, get their root node and terminal node lists, if the root
        # nodes match and the terminal lists are equivalent remove these groups from the set of keys for both rank
        # dictionaries. Ensure that all groups have been visited.
        d1_keys = set(dict1.keys())
        d2_keys = set(dict2.keys())
        for k1 in dict1:
            if k1 not in d1_keys:
                continue
            node1 = dict1[k1]['node']
            terminals1 = dict1[k1]['terminals']
            for k2 in dict2:
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
            self.assertEqual(len(d1_keys), len(d2_keys))
            self.assertEqual(len(d1_keys), rank - k1)
        self.assertEqual(len(d1_keys), 0)
        self.assertEqual(len(d2_keys), 0)

    def test11a_compare_assignments_small(self):
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
        self.assertEqual(len(assignments.keys()), len(et_mip_obj.rank_group_assignments.keys()))
        for rank in assignments:
            self.assertTrue(rank in et_mip_obj.rank_group_assignments)
            self.assertEqual(len(assignments[rank].keys()), len(et_mip_obj.rank_group_assignments[rank].keys()))
            self.evaluate_rank(rank, assignments[rank], et_mip_obj.rank_group_assignments[rank])

    def test11b_compare_assignments_large(self):
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
