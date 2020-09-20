import os
import re
import unittest
from unittest import TestCase
import numpy as np
from time import time
from re import compile
from shutil import rmtree
from copy import deepcopy
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceMatrix
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from test_Base import TestBase
from ETMIPWrapper import ETMIPWrapper
from SeqAlignment import SeqAlignment
from PhylogeneticTree import PhylogeneticTree
from EvolutionaryTraceAlphabet import FullIUPACProtein
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor


protein_seq1 = SeqRecord(id='seq1', seq=Seq('MET---', alphabet=FullIUPACProtein()))
protein_seq2 = SeqRecord(id='seq2', seq=Seq('M-TREE', alphabet=FullIUPACProtein()))
protein_seq3 = SeqRecord(id='seq3', seq=Seq('M-FREE', alphabet=FullIUPACProtein()))
protein_msa = MultipleSeqAlignment(records=[protein_seq1, protein_seq2, protein_seq3], alphabet=FullIUPACProtein())
adc = AlignmentDistanceCalculator(model='identity')
dm = adc.get_distance(msa=protein_msa, processes=2)
min_dm = DistanceMatrix(names=['seq1', 'seq2', 'seq3'])


class TestPhylogeneticTreeInit(TestCase):

    def test_init_default(self):
        phylo_tree = PhylogeneticTree()
        self.assertIsNone(phylo_tree.distance_matrix)
        self.assertEqual(phylo_tree.tree_method, 'upgma')
        self.assertEqual(phylo_tree.tree_args, {})
        self.assertIsNone(phylo_tree.tree)
        self.assertIsNone(phylo_tree.size)

    def test_init_upgma(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        self.assertIsNone(phylo_tree.distance_matrix)
        self.assertEqual(phylo_tree.tree_method, 'upgma')
        self.assertEqual(phylo_tree.tree_args, {})
        self.assertIsNone(phylo_tree.tree)
        self.assertIsNone(phylo_tree.size)

    def test_init_et(self):
        phylo_tree = PhylogeneticTree(tree_building_method='et', tree_building_args={})
        self.assertIsNone(phylo_tree.distance_matrix)
        self.assertEqual(phylo_tree.tree_method, 'et')
        self.assertEqual(phylo_tree.tree_args, {})
        self.assertIsNone(phylo_tree.tree)
        self.assertIsNone(phylo_tree.size)

    def test_init_agglomerative(self):
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'cache_dir': os.getcwd(),
                                                          'affinity': 'euclidean',
                                                          'linkage': 'ward'})
        self.assertIsNone(phylo_tree.distance_matrix)
        self.assertEqual(phylo_tree.tree_method, 'agglomerative')
        self.assertEqual(phylo_tree.tree_args, {'cache_dir': os.getcwd(),
                                                'affinity': 'euclidean',
                                                'linkage': 'ward'})
        self.assertIsNone(phylo_tree.tree)
        self.assertIsNone(phylo_tree.size)

    def test_init_custom(self):
        phylo_tree = PhylogeneticTree(tree_building_method='custom',
                                      tree_building_args={'tree_path': os.path.join(os.getcwd(), 'test.nhx')})
        self.assertIsNone(phylo_tree.distance_matrix)
        self.assertEqual(phylo_tree.tree_method, 'custom')
        self.assertEqual(phylo_tree.tree_args, {'tree_path': os.path.join(os.getcwd(), 'test.nhx')})
        self.assertIsNone(phylo_tree.tree)
        self.assertIsNone(phylo_tree.size)

    def test_init_other(self):
        phylo_tree = PhylogeneticTree(tree_building_method='fake',
                                      tree_building_args={'foo': 'bar'})
        self.assertIsNone(phylo_tree.distance_matrix)
        self.assertEqual(phylo_tree.tree_method, 'fake')
        self.assertEqual(phylo_tree.tree_args, {'foo': 'bar'})
        self.assertIsNone(phylo_tree.tree)
        self.assertIsNone(phylo_tree.size)


class TestPhylogeneticTreeConstructTree(TestCase):

    def test__upgma_tree(self):
        phylo_tree = PhylogeneticTree()
        phylo_tree.distance_matrix = dm
        tree = phylo_tree._upgma_tree()
        self.assertIsNone(phylo_tree.size)
        self.assertIsNone(phylo_tree.tree)
        self.assertIsNotNone(tree)
        first_children = tree.root.clades
        self.assertEqual(len(first_children), 2)
        for n1 in first_children:
            if n1.is_terminal():
                self.assertEqual(n1.name, 'seq1')
            else:
                second_children = n1.clades
                self.assertEqual(len(second_children), 2)
                for n2 in second_children:
                    self.assertTrue(n2.name in {'seq2', 'seq3'})
        
    def test__upgma_tree_failure_no_distance_matrix(self):
        with self.assertRaises(TypeError):
            phylo_tree = PhylogeneticTree()
            phylo_tree._upgma_tree()

    def test__et_tree(self):
        phylo_tree = PhylogeneticTree()
        phylo_tree.distance_matrix = dm
        tree = phylo_tree._et_tree()
        self.assertIsNone(phylo_tree.size)
        self.assertIsNone(phylo_tree.tree)
        self.assertIsNotNone(tree)
        first_children = tree.root.clades
        self.assertEqual(len(first_children), 2)
        for n1 in first_children:
            if n1.is_terminal():
                self.assertEqual(n1.name, 'seq1')
            else:
                second_children = n1.clades
                self.assertEqual(len(second_children), 2)
                for n2 in second_children:
                    self.assertTrue(n2.name in {'seq2', 'seq3'})

    def test__et_tree_failure_no_distance_matrix(self):
        with self.assertRaises(TypeError):
            phylo_tree = PhylogeneticTree()
            phylo_tree._et_tree()

    # def test__agglomerative_tree(self):
    # def test__agglomerative_tree_failure_bad_dir(self):
    # def test__agglomerative_tree_failure_no_linkage(self):
    # def test__agglomerative_tree_failure_no_affinity(self):
    # def test__agglomerative_tree_no_distance_matrix(self):
    #
    # def test__custom_tree(self):
    # def test__custom_tree_failure_no_path(self):
    # def test__custom_tree_failure_no_distance_matrix(self):
    #
    # def test_construct_tree_upgma(self):
    # def test_construct_tree_upgma_failure_no_distance_matrix(self):
    # def test_construct_tree_et(self):
    # def test_construct_tree_et_failure_no_distance_matrix(self):
    # def test_construct_tree_agglomerative(self):
    # def test_construct_tree_agglomerative_failure_bad_dir(self):
    # def test_construct_tree_agglomerative_failure_no_linkage(self):
    # def test_construct_tree_agglomerative_failure_no_affinity(self):
    # def test_construct_tree_agglomerative_failure_no_distance_matrix(self):
    # def test_construct_tree_custom(self):
    # def test_construct_tree_custom_failure_bad_path(self):
    # def test_construct_tree_custom_failure_no_distance_matrix(self):
    # def test_construct_tree_failure_bad_method(self):
# class TestPhylogeneticTree(TestBase):
#
#     @classmethod
#     def setUpClass(cls):
#         super(TestPhylogeneticTree, cls).setUpClass()
#         cls.query_aln_fa_small = SeqAlignment(file_name=cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln'],
#                                                query_id=cls.small_structure_id)
#         cls.query_aln_fa_small.import_alignment()
#         cls.query_aln_fa_large = SeqAlignment(file_name=cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln'],
#                                                query_id=cls.large_structure_id)
#         cls.query_aln_fa_large.import_alignment()
#         cls.query_aln_msf_small = deepcopy(cls.query_aln_fa_small)
#         cls.query_aln_msf_small.file_name = cls.data_set.protein_data[cls.small_structure_id]['Final_MSF_Aln']
#         cls.query_aln_msf_large = deepcopy(cls.query_aln_fa_large)
#         cls.query_aln_msf_large.file_name = cls.data_set.protein_data[cls.large_structure_id]['Final_MSF_Aln']
#         cls.out_small_dir = os.path.join(cls.testing_dir, cls.small_structure_id)
#         cls.out_large_dir = os.path.join(cls.testing_dir, cls.large_structure_id)
#
#     def tearDown(self):
#         if os.path.exists('./identity.pkl'):
#             os.remove('./identity.pkl')
#         cache_dir = os.path.join(self.testing_dir, 'joblib')
#         if os.path.exists(cache_dir):
#             rmtree(cache_dir)
#         wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test')
#         # if os.path.exists(wetc_test_dir):
#         #     rmtree(wetc_test_dir)
#
#     def test1_init(self):
#         phylo_tree = PhylogeneticTree()
#         self.assertEqual(phylo_tree.tree_method, 'upgma')
#         self.assertEqual(phylo_tree.tree_args, {})
#         self.assertIsNone(phylo_tree.distance_matrix)
#         self.assertIsNone(phylo_tree.tree)
#         self.assertIsNone(phylo_tree.size)
#
#     def evaluate_internal_nodes(self, internal_nodes):
#         non_terminal_pattern = compile(r'^Inner(\d+)$')
#         nodes = set()
#         for n in internal_nodes:
#             match = non_terminal_pattern.match(n.name)
#             self.assertIsNotNone(match)
#             self.assertNotIn(match.group(1), nodes)
#             nodes.add(match.group(1))
#
#     def evaluate_leaf_nodes(self, leaf_nodes, aln):
#         leaves = set()
#         for l in leaf_nodes:
#             self.assertIn(l.name, aln.seq_order)
#             self.assertNotIn(l.name, leaves)
#             leaves.add(l.name)
#
#     def evaluate__custome_tree(self, query_id, aln_fn, aln):
#         out_dir = os.path.join(self.testing_dir, query_id)
#         et_mip_obj = ETMIPWrapper(query=query_id, aln_file=aln_fn, out_dir=out_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         # Distance matrix is imported inside of et_mip_obj using the correct path.
#         nhx_path = os.path.join(out_dir, 'etc_out_intET.nhx')
#         phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_path})
#         self.assertEqual(phylo_tree.tree_method, 'custom')
#         self.assertEqual(phylo_tree.tree_args, {'tree_path': nhx_path})
#         self.assertIsNone(phylo_tree.distance_matrix)
#         self.assertIsNone(phylo_tree.tree)
#         self.assertIsNone(phylo_tree.size)
#         phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
#         self.assertEqual(phylo_tree.tree_method, 'custom')
#         self.assertEqual(phylo_tree.tree_args, {'tree_path': nhx_path})
#         self.assertEqual(phylo_tree.distance_matrix, et_mip_obj.distance_matrix)
#         self.assertEqual(phylo_tree.size, aln.size)
#         self.assertIsNotNone(phylo_tree.tree)
#         non_terminal_nodes = phylo_tree.tree.get_nonterminals()
#         self.assertEqual(len(non_terminal_nodes), aln.size - 1)
#         self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
#         terminal_nodes = phylo_tree.tree.get_terminals()
#         self.assertEqual(len(terminal_nodes), phylo_tree.size)
#         self.evaluate_leaf_nodes(leaf_nodes=terminal_nodes, aln=aln)
#
#     def test2a__custom_tree(self):
#         self.evaluate__custome_tree(query_id=self.query_aln_fa_small.query_id, aln_fn=self.query_aln_fa_small.file_name,
#                                     aln=self.query_aln_fa_small)
#
#     def test2b__custom_tree(self):
#         self.evaluate__custome_tree(query_id=self.query_aln_fa_large.query_id, aln_fn=self.query_aln_fa_large.file_name,
#                                     aln=self.query_aln_fa_large)
#
#     def check_nodes(self, node1, node2):
#         if node1.is_terminal():
#             self.assertTrue(node2.is_terminal(), 'Node1: {} vs Node2: {}'.format(node1.name, node2.name))
#             self.assertEqual(node1.name, node2.name)
#         else:
#             self.assertTrue(node2.is_bifurcating())
#             self.assertFalse(node2.is_terminal(), 'Node1: {} vs Node2: {}'.format(node1.name, node2.name))
#             self.assertEqual(set([x.name for x in node1.get_terminals()]),
#                              set([x.name for x in node2.get_terminals()]))
#
#     def evaluate__upgma_tree(self, aln):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(aln.alignment)
#         phylo_tree = PhylogeneticTree()
#         self.assertEqual(phylo_tree.tree_method, 'upgma')
#         self.assertEqual(phylo_tree.tree_args, {})
#         self.assertIsNone(phylo_tree.distance_matrix)
#         self.assertIsNone(phylo_tree.tree)
#         self.assertIsNone(phylo_tree.size)
#         start = time()
#         phylo_tree.construct_tree(dm=dm)
#         end = time()
#         print('Current implementation: {} min'.format((end - start) / 60.0))
#         self.assertEqual(phylo_tree.tree_method, 'upgma')
#         self.assertEqual(phylo_tree.tree_args, {})
#         self.assertEqual(phylo_tree.distance_matrix, dm)
#         self.assertEqual(phylo_tree.size, aln.size)
#         self.assertIsNotNone(phylo_tree.tree)
#         non_terminal_nodes = phylo_tree.tree.get_nonterminals()
#         self.assertEqual(len(non_terminal_nodes), aln.size - 1)
#         self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
#         terminal_nodes = phylo_tree.tree.get_terminals()
#         self.assertEqual(len(terminal_nodes), aln.size)
#         self.evaluate_leaf_nodes(terminal_nodes, aln)
#         constructor = DistanceTreeConstructor()
#         start2 = time()
#         upgma_tree = constructor.upgma(distance_matrix=dm)
#         end2 = time()
#         print('Official implementation: {} min'.format((end2 - start2) / 60.0))
#         phylo_tree_official = PhylogeneticTree()
#         phylo_tree_official.tree = upgma_tree
#         phylo_tree_official.size = len(dm)
#         phylo_tree_official.rename_internal_nodes()
#         py_iter = phylo_tree.traverse_by_rank()
#         official_iter = phylo_tree_official.traverse_by_rank()
#         try:
#             official_nodes = next(official_iter)
#         except StopIteration:
#             official_nodes = None
#         try:
#             py_nodes = next(py_iter)
#         except StopIteration:
#             py_nodes = None
#         while official_nodes and py_nodes:
#             if official_nodes is None:
#                 self.assertIsNone(py_nodes)
#             else:
#                 sorted_official_nodes = sorted(official_nodes, key=compare_nodes_key(compare_nodes))
#                 sorted_py_nodes = sorted(py_nodes, key=compare_nodes_key(compare_nodes))
#                 self.assertEqual(len(sorted_official_nodes), len(sorted_py_nodes))
#                 for i in range(len(sorted_py_nodes)):
#                     self.check_nodes(sorted_official_nodes[i], sorted_py_nodes[i])
#             try:
#                 official_nodes = next(official_iter)
#             except StopIteration:
#                 official_nodes = None
#             try:
#                 py_nodes = next(py_iter)
#             except StopIteration:
#                 py_nodes = None
#
#     def test3a__upgma_tree(self):
#         self.evaluate__upgma_tree(aln=self.query_aln_fa_small)
#
#     def test3b__upgma_tree(self):
#         self.evaluate__upgma_tree(aln=self.query_aln_fa_large)
#
#     def evaluate__et_tree(self, aln):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(aln.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='et')
#         self.assertEqual(phylo_tree.tree_method, 'et')
#         self.assertEqual(phylo_tree.tree_args, {})
#         self.assertIsNone(phylo_tree.distance_matrix)
#         self.assertIsNone(phylo_tree.tree)
#         self.assertIsNone(phylo_tree.size)
#         phylo_tree.construct_tree(dm=dm)
#         self.assertEqual(phylo_tree.tree_method, 'et')
#         self.assertEqual(phylo_tree.tree_args, {})
#         self.assertEqual(phylo_tree.distance_matrix, dm)
#         self.assertEqual(phylo_tree.size, aln.size)
#         self.assertIsNotNone(phylo_tree.tree)
#         non_terminal_nodes = phylo_tree.tree.get_nonterminals()
#         self.assertEqual(len(non_terminal_nodes), aln.size - 1)
#         self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
#         terminal_nodes = phylo_tree.tree.get_terminals()
#         self.assertEqual(len(terminal_nodes), aln.size)
#         self.evaluate_leaf_nodes(terminal_nodes, aln)
#
#     def test4a__et_tree(self):
#         self.evaluate__et_tree(aln=self.query_aln_fa_small)
#
#     def test4b__et_tree(self):
#         self.evaluate__et_tree(aln=self.query_aln_fa_large)
#
#     def evaluate__agglomerative_tree(self, aln):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(aln.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
#                                       tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
#                                                           'cache_dir': self.testing_dir})
#         self.assertEqual(phylo_tree.tree_method, 'agglomerative')
#         self.assertEqual(phylo_tree.tree_args, {'affinity': 'euclidean', 'linkage': 'ward',
#                                                 'cache_dir': self.testing_dir})
#         self.assertIsNone(phylo_tree.distance_matrix)
#         self.assertIsNone(phylo_tree.tree)
#         self.assertIsNone(phylo_tree.size)
#         phylo_tree.construct_tree(dm=dm)
#         self.assertEqual(phylo_tree.tree_method, 'agglomerative')
#         self.assertEqual(phylo_tree.tree_args, {'affinity': 'euclidean', 'linkage': 'ward',
#                                                 'cache_dir': self.testing_dir})
#         self.assertEqual(phylo_tree.distance_matrix, dm)
#         self.assertEqual(phylo_tree.size, aln.size)
#         self.assertIsNotNone(phylo_tree.tree)
#         non_terminal_nodes = phylo_tree.tree.get_nonterminals()
#         self.assertEqual(len(non_terminal_nodes), aln.size - 1)
#         self.evaluate_internal_nodes(internal_nodes=non_terminal_nodes)
#         terminal_nodes = phylo_tree.tree.get_terminals()
#         self.assertEqual(len(terminal_nodes), aln.size)
#         self.evaluate_leaf_nodes(terminal_nodes, aln)
#
#     def test5a_agglomerative_tree(self):
#         self.evaluate__agglomerative_tree(aln=self.query_aln_fa_small)
#
#     def test5b_agglomerative_tree(self):
#         self.evaluate__agglomerative_tree(aln=self.query_aln_fa_large)
#
#     # Construct tree tested throughout each of the specific construction method tests.
#
#     def check_lists_of_nodes_for_equality(self, list1, list2):
#         self.assertEqual(len(list1), len(list2))
#         for i in range(len(list1)):
#             group1 = list1[i]
#             group1 = sorted(group1, key=compare_nodes_key(compare_nodes))
#             group2 = list2[i]
#             group2 = sorted(group2, key=compare_nodes_key(compare_nodes))
#             self.assertEqual(len(group1), len(group2))
#             for j in range(len(group1)):
#                 node1 = group1[j]
#                 node2 = group2[j]
#                 self.check_nodes(node1=node1, node2=node2)
#
#     def evaluate_write_out_tree(self, save_fn, aln):
#         nhx_fn = os.path.join(self.testing_dir, save_fn)
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(aln.alignment)
#         phylo_tree = PhylogeneticTree()
#         self.assertFalse(os.path.isfile(nhx_fn))
#         with self.assertRaises(ValueError):
#             phylo_tree.write_out_tree(filename=nhx_fn)
#         self.assertFalse(os.path.isfile(nhx_fn))
#         phylo_tree.construct_tree(dm=dm)
#         phylo_tree.write_out_tree(filename=nhx_fn)
#         self.assertTrue(os.path.isfile(nhx_fn))
#         loaded_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': nhx_fn})
#         loaded_tree.construct_tree(dm=dm)
#         self.assertEqual(phylo_tree.distance_matrix, loaded_tree.distance_matrix)
#         self.assertEqual(phylo_tree.size, loaded_tree.size)
#         phylo_nodes = list(phylo_tree.traverse_by_rank())
#         loaded_nodes = list(loaded_tree.traverse_by_rank())
#         self.check_lists_of_nodes_for_equality(list1=phylo_nodes, list2=loaded_nodes)
#         os.remove(nhx_fn)
#
#     def test6a_write_out_tree(self):
#         self.evaluate_write_out_tree(save_fn='UPGMA_Newcki_tree.nhx', aln=self.query_aln_fa_small)
#
#     def test6b_write_out_tree(self):
#         self.evaluate_write_out_tree(save_fn='UPGMA_Newcki_tree.nhx', aln=self.query_aln_fa_large)
#
#     def get_path_length(self, node, tree):
#         path = tree.get_path(node)
#         dist = 0
#         for node in path:
#             dist += node.branch_length
#         return dist
#
#     def evaluate_top_down_traversal(self, phylo_tree):
#         node_names = set()
#         last_dist = 0.0
#         for node in phylo_tree.traverse_top_down():
#             self.assertNotIn(node.name, node_names)
#             node_names.add(node.name)
#             dist = self.get_path_length(node, phylo_tree.tree)
#             self.assertGreaterEqual(dist, last_dist)
#             last_dist = dist
#         self.assertEqual(len(node_names), (phylo_tree.size * 2) - 1)
#         self.assertTrue('Inner1' in node_names)
#         self.assertTrue('Inner{}'.format(phylo_tree.size - 1) in node_names)
#
#     def test7a_traverse_top_down(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree()
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_top_down_traversal(phylo_tree)
#
#     def test7b_traverse_top_down(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='et')
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_top_down_traversal(phylo_tree)
#
#     def test7c_traverse_top_down(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
#                                       tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
#                                                           'cache_dir': self.testing_dir})
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_top_down_traversal(phylo_tree)
#
#     def test7d_traverse_top_down(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_small.query_id, aln_file=self.query_aln_fa_small.file_name,
#                                   out_dir=self.out_small_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#         self.evaluate_top_down_traversal(et_mip_obj.tree)
#
#     def test7e_traverse_top_down(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree()
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_top_down_traversal(phylo_tree)
#
#     def test7f_traverse_top_down(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='et')
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_top_down_traversal(phylo_tree)
#
#     def test7g_traverse_top_down(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
#                                       tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
#                                                           'cache_dir': self.testing_dir})
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_top_down_traversal(phylo_tree)
#
#     def test7h_traverse_top_down(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_large.query_id, aln_file=self.query_aln_fa_large.file_name,
#                                   out_dir=self.out_large_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#         self.evaluate_top_down_traversal(et_mip_obj.tree)
#
#     def evaluate_bottom_up_traversal(self, phylo_tree):
#         last_dist = phylo_tree.tree.root.total_branch_length()
#         node_names = set()
#         for node in phylo_tree.traverse_bottom_up():
#             self.assertNotIn(node.name, node_names)
#             node_names.add(node.name)
#             dist = self.get_path_length(node, phylo_tree.tree)
#             self.assertLessEqual(dist, last_dist)
#             last_dist = dist
#         self.assertEqual(len(node_names), (phylo_tree.size * 2) - 1)
#         self.assertTrue('Inner1' in node_names)
#         self.assertTrue('Inner{}'.format(phylo_tree.size - 1) in node_names)
#
#     def test8a_traverse_bottom_up(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree()
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_bottom_up_traversal(phylo_tree)
#
#     def test8b_traverse_bottom_up(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='et')
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_bottom_up_traversal(phylo_tree)
#
#     def test8c_traverse_bottom_up(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
#                                       tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
#                                                           'cache_dir': self.testing_dir})
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_bottom_up_traversal(phylo_tree)
#
#     def test8d_traverse_bottom_up(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_small.query_id, aln_file=self.query_aln_fa_small.file_name,
#                                   out_dir=self.out_small_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#         self.evaluate_bottom_up_traversal(et_mip_obj.tree)
#
#     def test8e_traverse_bottom_up(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree()
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_bottom_up_traversal(phylo_tree)
#
#     def test8f_traverse_bottom_up(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='et')
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_bottom_up_traversal(phylo_tree)
#
#     def test8g_traverse_bottom_up(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
#                                       tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
#                                                           'cache_dir': self.testing_dir})
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_bottom_up_traversal(phylo_tree)
#
#     def test8h_traverse_bottom_up(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_large.query_id, aln_file=self.query_aln_fa_large.file_name,
#                                   out_dir=self.out_large_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#         self.evaluate_bottom_up_traversal(et_mip_obj.tree)
#
#     def evaluate_by_rank_traversal(self, tree):
#         nodes_added = set([])
#         nodes_removed = set([])
#         prev_rank = None
#         rank_count = 1
#         for rank in tree.traverse_by_rank():
#             # Make sure the current rank has the size of the rank counter (starting with size 1 for the first rank, i.e.
#             # root node only).
#             self.assertEqual(len(rank), rank_count)
#             if prev_rank:
#                 # Make sure that the two ranks differ in size by 1
#                 self.assertEqual(len(rank) - len(prev_rank), 1)
#                 # Determine the differences between the current and previous rank
#                 rank_set = set(rank)
#                 prev_set = set(prev_rank)
#                 # Make sure there are two new nodes in the current rank and that they have not appeared in any previous
#                 # rank.
#                 new_nodes = rank_set - prev_set
#                 self.assertEqual(len(new_nodes), 2)
#                 self.assertEqual(len(nodes_added.intersection(new_nodes)), 0)
#                 nodes_added |= new_nodes
#                 # Make sure only one node was lost (i.e. branched) from the previous rank and that it had never been
#                 # previously removed.
#                 old_nodes = prev_set - rank_set
#                 self.assertEqual(len(old_nodes), 1)
#                 self.assertEqual(len(nodes_removed.intersection(old_nodes)), 0)
#                 nodes_removed |= old_nodes
#             prev_rank = rank
#             rank_count += 1
#         # Make sure that the final rank processed was the size of number of sequences the tree was built from.
#         self.assertEqual(rank_count - 1, tree.size)
#
#     def test9a_traverse_by_rank(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree()
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_by_rank_traversal(tree=phylo_tree)
#         self.validate_upgma_tree(tree=phylo_tree, dm=dm)
#
#     def test9b_traverse_by_rank(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='et')
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_by_rank_traversal(tree=phylo_tree)
#
#     def test9c_traverse_by_rank(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
#                                       tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
#                                                           'cache_dir': self.testing_dir})
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_by_rank_traversal(tree=phylo_tree)
#
#     def test9d_traverse_by_rank(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_small.query_id, aln_file=self.query_aln_fa_small.file_name,
#                                   out_dir=self.out_small_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         self.evaluate_by_rank_traversal(tree=et_mip_obj.tree)
#
#     def test9e_traverse_by_rank(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree()
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_by_rank_traversal(tree=phylo_tree)
#         self.validate_upgma_tree(tree=phylo_tree, dm=dm)
#
#     def test9f_traverse_by_rank(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='et')
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_by_rank_traversal(tree=phylo_tree)
#
#     def test9g_traverse_by_rank(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
#                                       tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
#                                                           'cache_dir': self.testing_dir})
#         phylo_tree.construct_tree(dm=dm)
#         self.evaluate_by_rank_traversal(tree=phylo_tree)
#
#     def test9h_traverse_by_rank(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_large.query_id, aln_file=self.query_aln_fa_large.file_name,
#                                   out_dir=self.out_large_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         self.evaluate_by_rank_traversal(tree=et_mip_obj.tree)
#
#     def evaluate_rename_internal_nodes(self, phylo_tree, flip):
#         old_inner = {i.name: i for i in phylo_tree.tree.get_nonterminals()}
#         old_terminal = {t.name: t for t in phylo_tree.tree.get_terminals()}
#         phylo_tree.rename_internal_nodes()
#         try:
#             for nt_name in old_inner:
#                 self.assertEqual(nt_name,  old_inner[nt_name].name, (nt_name, ))
#         except AssertionError:
#             count_equal = 0
#             count_not_equal = 0
#             for nt_name in old_inner:
#                 if nt_name == old_inner[nt_name].name:
#                     count_equal += 1
#                 else:
#                     count_not_equal += 1
#                 self.assertGreaterEqual(count_not_equal, count_equal)
#         for t_name in old_terminal:
#             self.assertEqual(t_name, old_terminal[t_name].name)
#         new_inner = {i.name: i for i in phylo_tree.tree.get_nonterminals()}
#         new_terminal = {t.name: t for t in phylo_tree.tree.get_terminals()}
#         for nt_name in new_inner:
#             self.assertEqual(nt_name,  new_inner[nt_name].name)
#         for t_name in new_terminal:
#             self.assertEqual(t_name, new_terminal[t_name].name)
#         self.assertEqual(len(old_inner), len(new_inner))
#         if flip:
#             for name in old_inner:
#                 expected_name = 'Inner{}'.format(phylo_tree.size - int(re.match('^Inner([0-9]+)$', name).group(1)))
#                 self.assertTrue(old_inner[name] is new_inner[expected_name], "{} and {} do not match nodes!".format(
#                     name, expected_name))
#         else:
#             for name in old_inner:
#                 self.assertTrue(old_inner[name] is new_inner[name])
#         self.assertEqual(len(old_terminal), len(new_terminal))
#         for name in old_terminal:
#             self.assertTrue(old_terminal[name] is new_terminal[name])
#
#     def test10a_rename_internal_nodes(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_small.query_id, aln_file=self.query_aln_fa_small.file_name,
#                                   out_dir=self.out_small_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#         # expected_tree_fn = os.path.join(wetc_test_dir, '{}.nhx'.format('etc_out_intET'))
#         expected_tree_fn = os.path.join(self.out_small_dir, '{}.nhx'.format('etc_out_intET'))
#         # However, now that the construct_tree method renames internal nodes, this test would not work as intended so
#         # the construction is repeated using the specific (custom_tree) method.
#         et_mip_obj.tree.size = len(et_mip_obj.distance_matrix)
#         et_mip_obj.tree.tree = et_mip_obj.tree._custom_tree(tree_path=expected_tree_fn)
#         self.evaluate_rename_internal_nodes(phylo_tree=et_mip_obj.tree, flip=False)
#
#     def test10b_rename_internal_nodes(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_large.query_id, aln_file=self.query_aln_fa_large.file_name,
#                                   out_dir=self.out_large_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#         # expected_tree_fn = os.path.join(wetc_test_dir, '{}.nhx'.format('etc_out_intET'))
#         expected_tree_fn = os.path.join(self.out_small_dir, '{}.nhx'.format('etc_out_intET'))
#         # However, now that the construct_tree method renames internal nodes, this test would not work as intended so
#         # the construction is repeated using the specific (custom_tree) method.
#         et_mip_obj.tree.size = len(et_mip_obj.distance_matrix)
#         et_mip_obj.tree.tree = et_mip_obj.tree._custom_tree(tree_path=expected_tree_fn)
#         self.evaluate_rename_internal_nodes(phylo_tree=et_mip_obj.tree, flip=False)
#
#     def test10c_rename_internal_nodes(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree()
#         # Now that the construct_tree method renames internal nodes, this test would not work as intended so the
#         # construction is performed manually using the specific (upgma_tree) method.
#         phylo_tree.distance_matrix = dm
#         phylo_tree.size = len(dm)
#         phylo_tree.tree = phylo_tree._upgma_tree()
#         self.evaluate_rename_internal_nodes(phylo_tree=phylo_tree, flip=True)
#
#     def test10d_rename_internal_nodes(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree()
#         # Now that the construct_tree method renames internal nodes, this test would not work as intended so the
#         # construction is performed manually using the specific (upgma_tree) method.
#         phylo_tree.distance_matrix = dm
#         phylo_tree.size = len(dm)
#         phylo_tree.tree = phylo_tree._upgma_tree()
#         self.evaluate_rename_internal_nodes(phylo_tree=phylo_tree, flip=True)
#
#     def test10e_rename_internal_nodes(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='et')
#         # Now that the construct_tree method renames internal nodes, this test would not work as intended so the
#         # construction is performed manually using the specific (et_tree) method.
#         phylo_tree.distance_matrix = dm
#         phylo_tree.size = len(dm)
#         phylo_tree.tree = phylo_tree._et_tree()
#         self.evaluate_rename_internal_nodes(phylo_tree=phylo_tree, flip=False)
#
#     def test10f_rename_internal_nodes(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='et')
#         # Now that the construct_tree method renames internal nodes, this test would not work as intended so the
#         # construction is performed manually using the specific (et_tree) method.
#         phylo_tree.distance_matrix = dm
#         phylo_tree.size = len(dm)
#         phylo_tree.tree = phylo_tree._et_tree()
#         self.evaluate_rename_internal_nodes(phylo_tree=phylo_tree, flip=False)
#
#     def test10g_rename_internal_nodes(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
#                                       tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
#                                                           'cache_dir': self.testing_dir})
#         # Now that the construct_tree method renames internal nodes, this test would not work as intended so the
#         # construction is performed manually using the specific (agglomerative_clustering) method.
#         phylo_tree.distance_matrix = dm
#         phylo_tree.size = len(dm)
#         phylo_tree.tree = phylo_tree._agglomerative_clustering(affinity='euclidean', linkage='ward')
#         self.evaluate_rename_internal_nodes(phylo_tree=phylo_tree, flip=False)
#
#     def test10h_rename_internal_nodes(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
#                                       tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
#                                                           'cache_dir': self.testing_dir})
#         # Now that the construct_tree method renames internal nodes, this test would not work as intended so the
#         # construction is performed manually using the specific (agglomerative_clustering) method.
#         phylo_tree.distance_matrix = dm
#         phylo_tree.size = len(dm)
#         phylo_tree.tree = phylo_tree._agglomerative_clustering(affinity='euclidean', linkage='ward')
#         self.evaluate_rename_internal_nodes(phylo_tree=phylo_tree, flip=False)
#
#     def evaulate_rank_group_assignments(self, assignment, alignment):
#         expected_terminals = set(alignment.seq_order)
#         previous_nodes = set([])
#         for rank in range(1, alignment.size + 1):
#             # Make sure that each rank is represented in the assignment
#             self.assertTrue(rank in assignment)
#             all_roots = set([])
#             all_terminals = []
#             for group in range(1, rank + 1):
#                 # Make sure that each group is present in the rank assignments
#                 self.assertTrue(group in assignment[rank])
#                 # Make sure that each rank/group assignment has a root node assignment
#                 self.assertTrue('node' in assignment[rank][group])
#                 all_roots.add(assignment[rank][group]['node'])
#                 # Make sure that each rank/group assignment has a list of associated terminal nodes
#                 self.assertTrue('terminals' in assignment[rank][group])
#                 all_terminals += assignment[rank][group]['terminals']
#                 # Make sure that each rank/group assignment has a list of descendants associated
#                 self.assertTrue('descendants' in assignment[rank][group])
#                 if assignment[rank][group]['node'].is_terminal():
#                     self.assertIsNone(assignment[rank][group]['descendants'])
#                 else:
#                     self.assertEqual(set([node.name for node in assignment[rank][group]['node'].clades]),
#                                      set([node.name for node in assignment[rank][group]['descendants']]))
#             # Make sure that all root nodes at a given rank or unique and have a count equal to the rank
#             self.assertEqual(len(all_roots), rank)
#             # Make sure the correct number of nodes branch at each rank
#             unique_to_prev = previous_nodes - all_roots
#             unique_to_curr = all_roots - previous_nodes
#             if rank == 1:
#                 self.assertEqual(len(unique_to_prev), 0)
#                 self.assertEqual(len(unique_to_curr), 1)
#             else:
#                 self.assertEqual(len(unique_to_prev), 1)
#                 self.assertEqual(len(unique_to_curr), 2)
#             # Make sure that all sequences in the alignment are represented in the terminals of this rank
#             self.assertEqual(len(all_terminals), alignment.size)
#             all_terminals_set = set(all_terminals)
#             # Make sure those terminal nodes are all unique
#             self.assertEqual(len(all_terminals), len(all_terminals_set))
#             # Explicitly check that all sequences in the alignment are in the set of terminal nodes
#             self.assertTrue(expected_terminals == all_terminals_set)
#             # Explicitly check that all descendants from from the previous rank are represented in the root nodes
#             previous_nodes = all_roots
#
#     def test11a_assign_rank_group_small(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_small.query_id, aln_file=self.query_aln_fa_small.file_name,
#                                   out_dir=self.out_small_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#         assignments = et_mip_obj.tree.assign_group_rank()
#         self.evaulate_rank_group_assignments(assignment=assignments, alignment=self.query_aln_msf_small)
#
#     def test11b_assign_rank_group_large(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_large.query_id, aln_file=self.query_aln_fa_large.file_name,
#                                   out_dir=self.out_large_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#         assignments = et_mip_obj.tree.assign_group_rank()
#         self.evaulate_rank_group_assignments(assignment=assignments, alignment=self.query_aln_msf_large)
#
#     def test11c_assign_rank_group_small(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree()
#         phylo_tree.construct_tree(dm=dm)
#         assignments = phylo_tree.assign_group_rank()
#         self.evaulate_rank_group_assignments(assignment=assignments, alignment=self.query_aln_fa_small)
#
#     def test11d_assign_rank_group_large(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree()
#         phylo_tree.construct_tree(dm=dm)
#         assignments = phylo_tree.assign_group_rank()
#         self.evaulate_rank_group_assignments(assignment=assignments, alignment=self.query_aln_fa_large)
#
#     def test11e_assign_rank_group_small(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='et')
#         phylo_tree.construct_tree(dm=dm)
#         assignments = phylo_tree.assign_group_rank()
#         self.evaulate_rank_group_assignments(assignment=assignments, alignment=self.query_aln_fa_small)
#
#     def test11f_assign_rank_group_large(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='et')
#         phylo_tree.construct_tree(dm=dm)
#         assignments = phylo_tree.assign_group_rank()
#         self.evaulate_rank_group_assignments(assignment=assignments, alignment=self.query_aln_fa_large)
#
#     def test11g_assign_rank_group_small(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
#                                       tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
#                                                           'cache_dir': self.testing_dir})
#         phylo_tree.construct_tree(dm=dm)
#         assignments = phylo_tree.assign_group_rank()
#         self.evaulate_rank_group_assignments(assignment=assignments, alignment=self.query_aln_fa_small)
#
#     def test11h_assign_rank_group_large(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
#                                       tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
#                                                           'cache_dir': self.testing_dir})
#         phylo_tree.construct_tree(dm=dm)
#         assignments = phylo_tree.assign_group_rank()
#         self.evaulate_rank_group_assignments(assignment=assignments, alignment=self.query_aln_fa_large)
#
#     def test11i_assign_rank_group_small(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_small.query_id, aln_file=self.query_aln_fa_small.file_name,
#                                   out_dir=self.out_small_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#         self.evaulate_rank_group_assignments(assignment=et_mip_obj.rank_group_assignments,
#                                              alignment=self.query_aln_msf_small)
#
#     def test11j_assign_rank_group_large(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_large.query_id, aln_file=self.query_aln_fa_large.file_name,
#                                   out_dir=self.out_large_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#         self.evaulate_rank_group_assignments(assignment=et_mip_obj.rank_group_assignments,
#                                              alignment=self.query_aln_msf_large)
#
#     def evaluate_rank_group_assignments_custom_ranks(self, assignments):
#         visited_nodes = set([])
#         ranks = sorted(assignments.keys(), reverse=True)
#         node_ranks = {}
#         for i in range(len(ranks)):
#             rank = ranks[i]
#             node_ranks[rank] = set([])
#             descendants = set([])
#             for group in assignments[rank]:
#                 node = assignments[rank][group]['node']
#                 self.assertEqual(set(assignments[rank][group]['terminals']),
#                                  set([t.name for t in node.get_terminals()]))
#                 node_ranks[rank].add(node.name)
#                 if i > 0 and not node.is_terminal() and node.name not in visited_nodes:
#                     curr_d = set([d.name for d in assignments[rank][group]['descendants']])
#                     descendants |= set([d.name for d in assignments[rank][group]['descendants']])
#             visited_nodes |= node_ranks[rank]
#             prev_rank = ranks[i - 1]
#             if i > 0:
#                 self.assertTrue(descendants.issubset(node_ranks[prev_rank]))
#                 rank_diff = prev_rank - rank
#                 max_size_diff = rank_diff * 2
#                 unique_prev = node_ranks[prev_rank] - node_ranks[rank]
#                 self.assertLessEqual(len(unique_prev), max_size_diff)
#                 self.assertGreaterEqual(len(unique_prev), 2)
#                 unique_curr = node_ranks[rank] - node_ranks[prev_rank]
#                 self.assertLessEqual(len(unique_curr), rank_diff)
#                 self.assertGreaterEqual(len(unique_curr), 1)
#             else:
#                 self.assertEqual(descendants, set([]))
#                 self.assertEqual(len(node_ranks[rank]), rank)
#
#     def test11k_assign_rank_group_small(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_small.query_id, aln_file=self.query_aln_fa_small.file_name,
#                                   out_dir=self.out_small_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#         assignments = et_mip_obj.tree.assign_group_rank(ranks=[1, 2, 3, 5, 7, 10, 25])
#         self.evaluate_rank_group_assignments_custom_ranks(assignments=assignments)
#
#     def test11l_assign_rank_group_large(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_large.query_id, aln_file=self.query_aln_fa_large.file_name,
#                                   out_dir=self.out_large_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#         assignments = et_mip_obj.tree.assign_group_rank()
#         self.evaluate_rank_group_assignments_custom_ranks(assignments=assignments)
#
#     def test11m_assign_rank_group_small(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree()
#         phylo_tree.construct_tree(dm=dm)
#         assignments = phylo_tree.assign_group_rank()
#         self.evaluate_rank_group_assignments_custom_ranks(assignments=assignments)
#
#     def test11n_assign_rank_group_large(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree()
#         phylo_tree.construct_tree(dm=dm)
#         assignments = phylo_tree.assign_group_rank()
#         self.evaluate_rank_group_assignments_custom_ranks(assignments=assignments)
#
#     def test11o_assign_rank_group_small(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='et')
#         phylo_tree.construct_tree(dm=dm)
#         assignments = phylo_tree.assign_group_rank()
#         self.evaluate_rank_group_assignments_custom_ranks(assignments=assignments)
#
#     def test11p_assign_rank_group_large(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='et')
#         phylo_tree.construct_tree(dm=dm)
#         assignments = phylo_tree.assign_group_rank()
#         self.evaluate_rank_group_assignments_custom_ranks(assignments=assignments)
#
#     def test11q_assign_rank_group_small(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
#                                       tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
#                                                           'cache_dir': self.testing_dir})
#         phylo_tree.construct_tree(dm=dm)
#         assignments = phylo_tree.assign_group_rank()
#         self.evaluate_rank_group_assignments_custom_ranks(assignments=assignments)
#
#     def test11r_assign_rank_group_large(self):
#         calculator = AlignmentDistanceCalculator()
#         dm = calculator.get_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
#                                       tree_building_args={'affinity': 'euclidean', 'linkage': 'ward',
#                                                           'cache_dir': self.testing_dir})
#         phylo_tree.construct_tree(dm=dm)
#         assignments = phylo_tree.assign_group_rank()
#         self.evaluate_rank_group_assignments_custom_ranks(assignments=assignments)
#
#     def validate_upgma_tree(self, tree, dm, verbose=False):
#         reverse_rank_traversal = list(tree.traverse_by_rank())[::-1]
#         internal_dm = deepcopy(dm)
#         count = 1
#         while count < len(dm):
#             position_node = {i: name for i, name in enumerate(internal_dm.names)}
#             # Get nodes and their names for the previous rank
#             prev_rank = reverse_rank_traversal[count - 1]
#             prev_rank_names = set([node.name for node in prev_rank])
#             # Get nodes and their names for the current rank
#             curr_rank = reverse_rank_traversal[count]
#             curr_rank_names = set([node.name for node in curr_rank])
#             # Determine which nodes from the previous rank were joined to create a new node in the current rank (there
#             # should always be two)
#             joined_nodes = list(prev_rank_names - curr_rank_names)
#             # joined_nodes = list(curr_rank_names - prev_rank_names)
#             self.assertEqual(len(joined_nodes), 2)
#             # Determine which node in the current rank was the product of joining nodes (there should be only one)
#             resulting_node = list(curr_rank_names - prev_rank_names)
#             # resulting_node = list(prev_rank_names - curr_rank_names)
#             self.assertEqual(len(resulting_node), 1)
#             # Get the distance matrix positions the minimum score
#             dm_array = np.array(internal_dm)
#             min_score = np.min(dm_array[np.tril_indices(len(internal_dm), k=-1)])
#             positions = np.where(dm_array == min_score)
#             match = False
#             if verbose:
#                 print('#' * 100)
#                 print('Current Rank: {}'.format(count))
#                 print(prev_rank_names)
#                 print(curr_rank_names)
#                 print(joined_nodes)
#                 print(resulting_node)
#                 print([curr_rank[x].clades for x in range(len(curr_rank)) if curr_rank[x].name == resulting_node[0]])
#                 print(internal_dm[joined_nodes[0], joined_nodes[1]])
#                 print(np.where(dm_array == internal_dm[joined_nodes[0], joined_nodes[1]]))
#                 print(min_score)
#                 print(positions)
#             for i in range(len(positions[0])):
#                 pos_i = int(positions[0][i])
#                 pos_j = int(positions[1][i])
#                 name_i = position_node[pos_i]
#                 name_j = position_node[pos_j]
#                 if verbose:
#                     print(name_i)
#                     print(name_j)
#                     print(internal_dm[name_i, name_j])
#                 if name_i == joined_nodes[0] and name_j == joined_nodes[1]:
#                     match = True
#                     for k in range(len(internal_dm)):
#                         if k != pos_i and k != pos_j:
#                             internal_dm[pos_j, k] = (internal_dm[pos_i, k] + internal_dm[pos_j, k]) * 1.0 / 2
#                     internal_dm.names[pos_j] = resulting_node[0]
#                     del (internal_dm[pos_i])
#                     break
#             self.assertTrue(match)
#             count += 1
#
#     def test12a_validate_upgma_tree(self):
#         calculator = AlignmentDistanceCalculator()
#         _, dm, _, _ = calculator.get_et_distance(self.query_aln_fa_small.alignment)
#         phylo_tree = PhylogeneticTree()
#         phylo_tree.construct_tree(dm=dm)
#         self.validate_upgma_tree(tree=phylo_tree, dm=dm, verbose=False)
#
#     def test12b_validate_upgma_tree(self):
#         calculator = AlignmentDistanceCalculator()
#         _, dm, _, _ = calculator.get_et_distance(self.query_aln_fa_large.alignment)
#         phylo_tree = PhylogeneticTree()
#         phylo_tree.construct_tree(dm=dm)
#         self.validate_upgma_tree(tree=phylo_tree, dm=dm, verbose=False)
#
#     # Tree building method 'et' does not produce a true UPGMA tree so it does not validate.
#
#     # def test12c_validate_upgma_tree(self):
#     #     calculator = AlignmentDistanceCalculator()
#     #     _, dm, _, _ = calculator.get_et_distance(self.query_aln_fa_small.alignment)
#     #     phylo_tree = PhylogeneticTree(tree_building_method='et')
#     #     phylo_tree.construct_tree(dm=dm)
#     #     self.validate_upgma_tree(tree=phylo_tree, dm=dm, verbose=False)
#     #
#     # def test12d_validate_upgma_tree(self):
#     #     calculator = AlignmentDistanceCalculator()
#     #     _, dm, _, _ = calculator.get_et_distance(self.query_aln_fa_large.alignment)
#     #     phylo_tree = PhylogeneticTree(tree_building_method='et')
#     #     phylo_tree.construct_tree(dm=dm)
#     #     self.validate_upgma_tree(tree=phylo_tree, dm=dm, verbose=False)
#
#     def test12e_validate_upgma_tree(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_small.query_id, aln_file=self.query_aln_fa_small.file_name,
#                                   out_dir=self.out_small_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         # Distance matrix is imported inside of et_mip_obj the correct path.
#         phylo_tree = PhylogeneticTree()
#         phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
#         self.validate_upgma_tree(tree=phylo_tree, dm=et_mip_obj.distance_matrix, verbose=False)
#
#     def test12f_validate_upgma_tree(self):
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_large.query_id, aln_file=self.query_aln_fa_large.file_name,
#                                   out_dir=self.out_large_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         # Distance matrix is imported inside of et_mip_obj the correct path.
#         phylo_tree = PhylogeneticTree()
#         phylo_tree.construct_tree(dm=et_mip_obj.distance_matrix)
#         self.validate_upgma_tree(tree=phylo_tree, dm=et_mip_obj.distance_matrix, verbose=False)
#
#     # WETC tree building method (replicated by the tree building method in PhylogeneticTree 'et') does not produce a
#     # true UPGMA tree so it does not validate.
#
#     # def test12g_validate_upgma_tree(self):
#     #     wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.small_structure_id)
#     #     if not os.path.isdir(wetc_test_dir):
#     #         os.makedirs(wetc_test_dir)
#     #     et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_small)
#     #     et_mip_obj.calculate_scores(method='intET', out_dir=wetc_test_dir, delete_files=False)
#     #     # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#     #     self.validate_upgma_tree(tree=et_mip_obj.tree, dm=et_mip_obj.distance_matrix, verbose=True)
#     #
#     # def test12h_validate_upgma_tree(self):
#     #     wetc_test_dir = os.path.join(self.testing_dir, 'WETC_Test', self.large_structure_id)
#     #     if not os.path.isdir(wetc_test_dir):
#     #         os.makedirs(wetc_test_dir)
#     #     et_mip_obj = ETMIPWrapper(alignment=self.query_aln_msf_large)
#     #     et_mip_obj.calculate_scores(method='intET', out_dir=wetc_test_dir, delete_files=False)
#     #     # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#     #     self.validate_upgma_tree(tree=et_mip_obj.tree, dm=et_mip_obj.distance_matrix, verbose=True)
#
#     def evaluate_rank(self, assignments1, assignments2):
#         self.assertEqual(len(assignments1.keys()), len(assignments2.keys()))
#         for rank in assignments1:
#             self.assertTrue(rank in assignments2)
#             self.assertEqual(len(assignments1[rank].keys()), len(assignments2[rank].keys()))
#             # Logic for this test:
#             # Iterate over the groups in both rank dictionaries, get their root node and terminal node lists, if the root
#             # nodes match and the terminal lists are equivalent remove these groups from the set of keys for both rank
#             # dictionaries. Ensure that all groups have been visited.
#             d1_keys = set(assignments1[rank].keys())
#             d2_keys = set(assignments2[rank].keys())
#             for k1 in assignments1[rank]:
#                 if k1 not in d1_keys:
#                     continue
#                 node1 = assignments1[rank][k1]['node']
#                 terminals1 = assignments1[rank][k1]['terminals']
#                 for k2 in assignments2[rank]:
#                     if k2 not in d2_keys:
#                         continue
#                     node2 = assignments2[rank][k2]['node']
#                     terminals2 = assignments2[rank][k2]['terminals']
#                     if set(terminals1) == set(terminals2):
#                         self.check_nodes(node1, node2)
#                     else:
#                         continue
#                     d1_keys.remove(k1)
#                     d2_keys.remove(k2)
#                     break
#                 self.assertEqual(len(d1_keys), len(d2_keys))
#                 self.assertEqual(len(d1_keys), rank - k1)
#             self.assertEqual(len(d1_keys), 0)
#             self.assertEqual(len(d2_keys), 0)
#
#     def test13a_compare_assignments_small(self):
#         start = time()
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_small.query_id, aln_file=self.query_aln_fa_small.file_name,
#                                   out_dir=self.out_small_dir)
#         inter1 = time()
#         print('Initialization took {} min'.format((inter1 - start) / 60.0))
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         inter2 = time()
#         print('Score calculation took {} min'.format((inter2 - inter1) / 60.0))
#         # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#         assignments = et_mip_obj.tree.assign_group_rank()
#         self.evaluate_rank(assignments1=assignments, assignments2=et_mip_obj.rank_group_assignments)
#
#     def test13b_compare_assignments_large(self):
#         start = time()
#         et_mip_obj = ETMIPWrapper(query=self.query_aln_fa_large.query_id, aln_file=self.query_aln_fa_large.file_name,
#                                   out_dir=self.out_large_dir)
#         inter1 = time()
#         print('Initializatoin took {} min'.format((inter1 - start) / 60.0))
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         inter2 = time()
#         print('Score calculation took {} min'.format((inter2 - inter1) / 60.0))
#         # Tree is imported inside of et_mip_obj using PhylogeneticTree with method custom and the correct path.
#         assignments = et_mip_obj.tree.assign_group_rank()
#         self.evaluate_rank(assignments1=assignments, assignments2=et_mip_obj.rank_group_assignments)
#
#     def compare_tree_and_wetc_tree(self, p_id, fa_aln):
#         out_dir =  os.path.join(self.testing_dir, p_id)
#         et_mip_obj = ETMIPWrapper(query=fa_aln.query_id, aln_file=fa_aln.file_name,
#                                   out_dir=out_dir)
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         calculator = AlignmentDistanceCalculator(model='blosum62')
#         _, dm, _, _ = calculator.get_et_distance(fa_aln.remove_gaps().alignment)
#         phylo_tree = PhylogeneticTree(tree_building_method='et')
#         phylo_tree.construct_tree(dm=dm)
#         wetc_iter = et_mip_obj.tree.traverse_by_rank()
#         py_iter = phylo_tree.traverse_by_rank()
#         try:
#             wetc_nodes = next(wetc_iter)
#         except StopIteration:
#             wetc_nodes = None
#         try:
#             py_nodes = next(py_iter)
#         except StopIteration:
#             py_nodes = None
#         count = 1
#         while wetc_nodes and py_nodes:
#             count += 1
#             if wetc_nodes is None:
#                 self.assertIsNone(py_nodes)
#             else:
#                 sorted_wetc_nodes = sorted(wetc_nodes, key=compare_nodes_key(compare_nodes))
#                 sorted_py_nodes = sorted(py_nodes, key=compare_nodes_key(compare_nodes))
#                 self.assertEqual(len(sorted_wetc_nodes), len(sorted_py_nodes))
#                 for i in range(len(sorted_py_nodes)):
#                     try:
#                         self.check_nodes(sorted_wetc_nodes[i], sorted_py_nodes[i])
#                     except AssertionError as e:
#                         raise AssertionError("ERRORED ON i={}\nWETC NODE:{} WITH CHILDREN {} and {}\nPY NODE:{} with CHILDREN {} and {}".format(
#                             i, sorted_wetc_nodes[i], sorted_wetc_nodes[i].clades[0], sorted_wetc_nodes[i].clades[1], sorted_py_nodes[i], sorted_py_nodes[i].clades[0], sorted_py_nodes[i].clades[1])) from e
#             try:
#                 wetc_nodes = next(wetc_iter)
#             except StopIteration:
#                 wetc_nodes = None
#             try:
#                 py_nodes = next(py_iter)
#             except StopIteration:
#                 py_nodes = None
#
#     def test14a_compare_to_wetc_tree_small(self):
#         self.compare_tree_and_wetc_tree(p_id=self.small_structure_id, fa_aln=self.query_aln_fa_small)
#
#     def test14b_compare_to_wetc_tree_small(self):
#         self.compare_tree_and_wetc_tree(p_id=self.large_structure_id, fa_aln=self.query_aln_fa_large)
#
#
# def compare_nodes_key(compare_nodes):
#     """Taken from: https://docs.python.org/3/howto/sorting.html"""
#     class K:
#         def __init__(self, obj, *args):
#             self.obj = obj
#
#         def __lt__(self, other):
#             return compare_nodes(self.obj, other.obj) < 0
#
#         def __gt__(self, other):
#             return compare_nodes(self.obj, other.obj) > 0
#
#         def __eq__(self, other):
#             return compare_nodes(self.obj, other.obj) == 0
#
#         def __le__(self, other):
#             return compare_nodes(self.obj, other.obj) <= 0
#
#         def __ge__(self, other):
#             return compare_nodes(self.obj, other.obj) >= 0
#
#         def __ne__(self, other):
#             return compare_nodes(self.obj, other.obj) != 0
#     return K
#
#
# def compare_nodes(node1, node2):
#     if node1.is_terminal and not node2.is_terminal():
#         return -1
#     elif not node1.is_terminal() and node2.is_terminal():
#         return 1
#     else:
#         if node1.name < node2.name:
#             return 1
#         elif node1.name > node2.name:
#             return -1
#         else:
#             return 0


if __name__ == '__main__':
    unittest.main()
