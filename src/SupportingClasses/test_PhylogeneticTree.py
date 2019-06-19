import os
import numpy as np
from time import time
from re import compile
from copy import deepcopy
from shutil import rmtree
from unittest import TestCase
from multiprocessing import cpu_count
from Bio.Phylo.TreeConstruction import DistanceCalculator
from ETMIPWrapper import ETMIPWrapper
from SeqAlignment import SeqAlignment
from DataSetGenerator import DataSetGenerator
from PhylogeneticTree import PhylogeneticTree
from AlignmentDistanceCalculator import AlignmentDistanceCalculator


class TestPhylogeneticTree(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.max_threads = cpu_count() - 2
        cls.max_target_seqs = 150
        cls.testing_dir = os.path.abspath('../Test/')
        cls.input_path = os.path.join(cls.testing_dir, 'Input')
        cls.protein_list_path = os.path.join(cls.input_path, 'ProteinLists')
        if not os.path.isdir(cls.protein_list_path):
            os.makedirs(cls.protein_list_path)
        cls.small_structure_id = '7hvp'
        cls.large_structure_id = '2zxe'
        cls.protein_list_fn = os.path.join(cls.protein_list_path, 'Test_Set.txt')
        structure_ids = [cls.small_structure_id, cls.large_structure_id]
        with open(cls.protein_list_fn, 'wb') as test_list_handle:
            for structure_id in structure_ids:
                test_list_handle.write('{}{}\n'.format(structure_id, 'A'))
        cls.data_set = DataSetGenerator(input_path=cls.input_path)
        cls.data_set.build_pdb_alignment_dataset(protein_list_fn='Test_Set.txt', num_threads=cls.max_threads,
                                                 max_target_seqs=cls.max_target_seqs)
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

    # @classmethod
    # def tearDownClass(cls):
        # rmtree(cls.input_path)

    # def setUp(self):

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
        path = tree.tree.get_path(node)
        dist = sum([x.branch_length for x in path])
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

    def check_lists_of_nodes_for_equality(self, list1, list2):
        self.assertEqual(len(list1), len(list2))
        for i in range(len(list1)):
            node1 = list1[i]
            node2 = list2[i]
            if node1.is_terminal():
                self.assertTrue(node2.is_terminal())
                self.assertEqual(node1.name, node2.name)
            else:
                self.assertTrue(node2.is_nonterminal())
                self.assertEqual(set([x.name for x in node1.get_terminals()]),
                                 set([x.name for x in node2.get_terminals()]))

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
        phylo_tree.construct_tree(dm=None)
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
        phylo_tree.construct_tree(dm=None)
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
        phylo_tree.construct_tree(dm=None)
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
        phylo_tree.construct_tree(dm=None)
        self.evaluate_top_down_traversal(phylo_tree)

    def test5a_traverse_bottom_up(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test5b_traverse_bottom_up(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test5c_traverse_bottom_up(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=None)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test5d_traverse_bottom_up(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test5e_traverse_bottom_up(self):
        calculator = AlignmentDistanceCalculator()
        dm = calculator.get_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
                                                          'cache_dir': self.testing_dir})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test5f_traverse_bottom_up(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        phylo_tree.construct_tree(dm=None)
        self.evaluate_bottom_up_traversal(phylo_tree)

    def test7a_build_ETC_tree_small(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        etc_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        etc_tree.construct_tree(dm=None)
        calculator = AlignmentDistanceCalculator(model='blosum62')
        dm = calculator.get_et_distance(self.query_aln_fa_small.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        etc_nodes = [etc_tree.traverse_top_down()]
        phylo_nodes = [phylo_tree.traverse_top_down()]
        self.check_lists_of_nodes_for_equality(etc_nodes, phylo_nodes)

    def test7b_build_ETC_tree_large(self):
        wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
        if not os.path.isdir(wetc_test_dir):
            os.makedirs(wetc_test_dir)
        et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
        et_mip_obj.calculate_scores(out_dir=wetc_test_dir, delete_files=False)
        nhx_path = os.path.join(wetc_test_dir, 'etc_out.nhx')
        etc_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
        etc_tree.construct_tree(dm=None)
        calculator = AlignmentDistanceCalculator(model='blosum62')
        dm = calculator.get_et_distance(self.query_aln_fa_large.alignment)
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        etc_nodes = [etc_tree.traverse_top_down()]
        phylo_nodes = [phylo_tree.traverse_top_down()]
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
        loaded_tree.construct_tree(dm=None)
        self.check_lists_of_nodes_for_equality(list1=[phylo_tree.traverse_top_down()],
                                               list2=[loaded_tree.traverse_top_down()])
        os.remove(nhx_fn)