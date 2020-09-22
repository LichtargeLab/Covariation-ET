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
from PhylogeneticTree import PhylogeneticTree, get_path_length
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
        phylo_tree = PhylogeneticTree(tree_building_method='et', tree_building_args={})
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
            phylo_tree = PhylogeneticTree(tree_building_method='et', tree_building_args={})
            phylo_tree._et_tree()

    def test__agglomerative_tree(self):
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'cache_dir': os.getcwd(),
                                                          'affinity': 'euclidean',
                                                          'linkage': 'ward'})
        phylo_tree.distance_matrix = dm
        tree = phylo_tree._agglomerative_clustering(cache_dir=phylo_tree.tree_args['cache_dir'],
                                                    affinity=phylo_tree.tree_args['affinity'],
                                                    linkage=phylo_tree.tree_args['linkage'])
        self.assertIsNone(phylo_tree.size)
        self.assertIsNone(phylo_tree.tree)
        cache_dir_path = os.path.join(os.getcwd(), 'joblib')
        self.assertTrue(os.path.isdir(cache_dir_path))
        rmtree(cache_dir_path)
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

    def test__agglomerative_tree_new_cache_dir(self):
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'cache_dir': os.path.join(os.getcwd(), 'test'),
                                                          'affinity': 'euclidean',
                                                          'linkage': 'ward'})
        phylo_tree.distance_matrix = dm
        tree = phylo_tree._agglomerative_clustering(cache_dir=phylo_tree.tree_args['cache_dir'],
                                                    affinity=phylo_tree.tree_args['affinity'],
                                                    linkage=phylo_tree.tree_args['linkage'])
        self.assertIsNone(phylo_tree.size)
        self.assertIsNone(phylo_tree.tree)
        cache_dir_path = os.path.join(os.getcwd(), 'test')
        self.assertTrue(os.path.isdir(cache_dir_path))
        rmtree(cache_dir_path)
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

    def test__agglomerative_tree_failure_no_linkage(self):
        with self.assertRaises(ValueError):
            phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                          tree_building_args={'cache_dir': None,
                                                              'affinity': None,
                                                              'linkage': 'ward'})
            phylo_tree.distance_matrix = dm
            phylo_tree._agglomerative_clustering(cache_dir=phylo_tree.tree_args['cache_dir'],
                                                 affinity=phylo_tree.tree_args['affinity'],
                                                 linkage=phylo_tree.tree_args['linkage'])

    def test__agglomerative_tree_failure_no_affinity(self):
        with self.assertRaises(ValueError):
            phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                          tree_building_args={'cache_dir': None,
                                                              'affinity': 'euclidean',
                                                              'linkage': None})
            phylo_tree.distance_matrix = dm
            phylo_tree._agglomerative_clustering(cache_dir=phylo_tree.tree_args['cache_dir'],
                                                 affinity=phylo_tree.tree_args['affinity'],
                                                 linkage=phylo_tree.tree_args['linkage'])

    def test__agglomerative_tree_no_distance_matrix(self):
        with self.assertRaises(TypeError):
            phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                          tree_building_args={'cache_dir': None,
                                                              'affinity': 'euclidean',
                                                              'linkage': 'ward'})
            phylo_tree._agglomerative_clustering(cache_dir=phylo_tree.tree_args['cache_dir'],
                                                 affinity=phylo_tree.tree_args['affinity'],
                                                 linkage=phylo_tree.tree_args['linkage'])

    def test__custom_tree(self):
        test_tree_path = os.path.join(os.getcwd(), 'test.nhx')
        with open(os.path.join(os.getcwd(), 'test.nhx'), 'w') as handle:
            handle.write('(seq1:0.1,(seq2:0.05,seq3:0.05)Inner2:0.29167)Inner1:0.00000;')
        phylo_tree = PhylogeneticTree(tree_building_method='custom',
                                      tree_building_args={'tree_path': test_tree_path})
        phylo_tree.distance_matrix = min_dm
        tree = phylo_tree._custom_tree(tree_path=phylo_tree.tree_args['tree_path'])
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
        os.remove(test_tree_path)

    def test__custom_tree_no_distance_matrix(self):
        test_tree_path = os.path.join(os.getcwd(), 'test.nhx')
        with open(os.path.join(os.getcwd(), 'test.nhx'), 'w') as handle:
            handle.write('(seq1:0.1,(seq2:0.05,seq3:0.05)Inner2:0.29167)Inner1:0.00000;')
        phylo_tree = PhylogeneticTree(tree_building_method='custom',
                                      tree_building_args={'tree_path': test_tree_path})
        tree = phylo_tree._custom_tree(tree_path=phylo_tree.tree_args['tree_path'])
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
        os.remove(test_tree_path)

    def test__custom_tree_failure_no_path(self):
        with self.assertRaises(TypeError):
            phylo_tree = PhylogeneticTree(tree_building_method='custom',
                                          tree_building_args={'tree_path': None})
            phylo_tree.distance_matrix = min_dm
            phylo_tree._custom_tree(tree_path=phylo_tree.tree_args['tree_path'])

    def test_construct_tree_upgma(self):
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.assertIsNotNone(phylo_tree.size)
        self.assertEqual(phylo_tree.size, 3)
        self.assertIsNotNone(phylo_tree.tree)
        first_children = phylo_tree.tree.root.clades
        self.assertEqual(len(first_children), 2)
        for n1 in first_children:
            if n1.is_terminal():
                self.assertEqual(n1.name, 'seq1')
            else:
                second_children = n1.clades
                self.assertEqual(len(second_children), 2)
                for n2 in second_children:
                    self.assertTrue(n2.name in {'seq2', 'seq3'})

    def test_construct_tree_upgma_failure_no_distance_matrix(self):
        with self.assertRaises(ValueError):
            phylo_tree = PhylogeneticTree()
            phylo_tree.construct_tree(dm=None)

    def test_construct_tree_et(self):
        phylo_tree = PhylogeneticTree(tree_building_method='et', tree_building_args={})

        phylo_tree.construct_tree(dm=dm)
        self.assertIsNotNone(phylo_tree.size)
        self.assertEqual(phylo_tree.size, 3)
        self.assertIsNotNone(phylo_tree.tree)
        first_children = phylo_tree.tree.root.clades
        self.assertEqual(len(first_children), 2)
        for n1 in first_children:
            if n1.is_terminal():
                self.assertEqual(n1.name, 'seq1')
            else:
                second_children = n1.clades
                self.assertEqual(len(second_children), 2)
                for n2 in second_children:
                    self.assertTrue(n2.name in {'seq2', 'seq3'})

    def test_construct_tree_et_failure_no_distance_matrix(self):
        with self.assertRaises(ValueError):
            phylo_tree = PhylogeneticTree(tree_building_method='et', tree_building_args={})
            phylo_tree.construct_tree(dm=None)

    def test_construct_tree_agglomerative(self):
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'cache_dir': os.getcwd(),
                                                          'affinity': 'euclidean',
                                                          'linkage': 'ward'})
        phylo_tree.construct_tree(dm=dm)
        self.assertIsNotNone(phylo_tree.size)
        self.assertEqual(phylo_tree.size, 3)
        self.assertIsNotNone(phylo_tree.tree)
        cache_dir_path = os.path.join(os.getcwd(), 'joblib')
        self.assertTrue(os.path.isdir(cache_dir_path))
        rmtree(cache_dir_path)
        first_children = phylo_tree.tree.root.clades
        self.assertEqual(len(first_children), 2)
        for n1 in first_children:
            if n1.is_terminal():
                self.assertEqual(n1.name, 'seq1')
            else:
                second_children = n1.clades
                self.assertEqual(len(second_children), 2)
                for n2 in second_children:
                    self.assertTrue(n2.name in {'seq2', 'seq3'})

    def test_construct_tree_agglomerative_new_cache_dir(self):
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'cache_dir': os.path.join(os.getcwd(), 'test'),
                                                          'affinity': 'euclidean',
                                                          'linkage': 'ward'})
        phylo_tree.construct_tree(dm=dm)
        self.assertIsNotNone(phylo_tree.size)
        self.assertIsNotNone(phylo_tree.tree)
        cache_dir_path = os.path.join(os.getcwd(), 'test')
        self.assertTrue(os.path.isdir(cache_dir_path))
        rmtree(cache_dir_path)
        first_children = phylo_tree.tree.root.clades
        self.assertEqual(len(first_children), 2)
        for n1 in first_children:
            if n1.is_terminal():
                self.assertEqual(n1.name, 'seq1')
            else:
                second_children = n1.clades
                self.assertEqual(len(second_children), 2)
                for n2 in second_children:
                    self.assertTrue(n2.name in {'seq2', 'seq3'})

    def test_construct_tree_agglomerative_failure_no_linkage(self):
        with self.assertRaises(ValueError):
            phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                          tree_building_args={'cache_dir': None,
                                                              'affinity': None,
                                                              'linkage': 'ward'})
            phylo_tree.construct_tree(dm=dm)

    def test_construct_tree_agglomerative_failure_no_affinity(self):
        with self.assertRaises(ValueError):
            phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                          tree_building_args={'cache_dir': None,
                                                              'affinity': 'euclidean',
                                                              'linkage': None})
            phylo_tree.construct_tree(dm=dm)

    def test_construct_tree_agglomerative_failure_no_distance_matrix(self):
        with self.assertRaises(ValueError):
            phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                          tree_building_args={'cache_dir': None,
                                                              'affinity': 'euclidean',
                                                              'linkage': 'ward'})
            phylo_tree.construct_tree(dm=None)

    def test_construct_tree_custom(self):
        test_tree_path = os.path.join(os.getcwd(), 'test.nhx')
        with open(os.path.join(os.getcwd(), 'test.nhx'), 'w') as handle:
            handle.write('(seq1:0.1,(seq2:0.05,seq3:0.05)Inner2:0.29167)Inner1:0.00000;')
        phylo_tree = PhylogeneticTree(tree_building_method='custom',
                                      tree_building_args={'tree_path': test_tree_path})
        phylo_tree.construct_tree(dm=min_dm)
        self.assertIsNotNone(phylo_tree.size)
        self.assertEqual(phylo_tree.size, 3)
        self.assertIsNotNone(phylo_tree.tree)
        first_children = phylo_tree.tree.root.clades
        self.assertEqual(len(first_children), 2)
        for n1 in first_children:
            if n1.is_terminal():
                self.assertEqual(n1.name, 'seq1')
            else:
                second_children = n1.clades
                self.assertEqual(len(second_children), 2)
                for n2 in second_children:
                    self.assertTrue(n2.name in {'seq2', 'seq3'})
        os.remove(test_tree_path)

    def test_construct_tree_custom_failure_bad_path(self):
        with self.assertRaises(TypeError):
            phylo_tree = PhylogeneticTree(tree_building_method='custom',
                                          tree_building_args={'tree_path': None})
            phylo_tree.construct_tree(dm=min_dm)

    def test_construct_tree_custom_failure_no_distance_matrix(self):
        test_tree_path = os.path.join(os.getcwd(), 'test.nhx')
        with open(os.path.join(os.getcwd(), 'test.nhx'), 'w') as handle:
            handle.write('(seq1:0.1,(seq2:0.05,seq3:0.05)Inner2:0.29167)Inner1:0.00000;')
        with self.assertRaises(ValueError):
            phylo_tree = PhylogeneticTree(tree_building_method='custom',
                                          tree_building_args={'tree_path': test_tree_path})
            phylo_tree.construct_tree(dm=None)
        os.remove(test_tree_path)

    def test_construct_tree_failure_bad_method(self):
        with self.assertRaises(KeyError):
            phylo_tree = PhylogeneticTree(tree_building_method='fake',
                                          tree_building_args={'foo': 'bar'})
            phylo_tree.construct_tree(dm=dm)


class TestPhylogeneticTreeWriteOutTree(TestCase):

    def test_write_out_upgma_tree(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        tree_path = os.path.join(os.getcwd(), 'test.nhx')
        self.assertFalse(os.path.isfile(tree_path))
        phylo_tree.write_out_tree(filename=tree_path)
        self.assertTrue(os.path.isfile(tree_path))
        phylo_tree2 = PhylogeneticTree(tree_building_method='custom',
                                       tree_building_args={'tree_path': tree_path})
        phylo_tree2.construct_tree(dm=min_dm)
        self.assertEqual(phylo_tree.size, phylo_tree2.size)
        self.assertEqual(phylo_tree.tree.root.name, phylo_tree2.tree.root.name)
        self.assertEqual(phylo_tree.tree.root.branch_length, phylo_tree2.tree.root.branch_length)
        first_children = phylo_tree.tree.root.clades
        first_children2 = phylo_tree2.tree.root.clades
        first_children_combined = zip(first_children, first_children2)
        for c1, c2 in first_children_combined:
            self.assertEqual(c1.is_terminal(), c2.is_terminal())
            self.assertEqual(c1.name, c2.name)
            self.assertLessEqual(c1.branch_length - c2.branch_length, 1E-5)
            if not c1.is_terminal():
                second_children = c1.clades
                second_children2 = c2.clades
                second_children_combined = zip(second_children, second_children2)
                for l1, l2 in second_children_combined:
                    self.assertTrue(l1.is_terminal())
                    self.assertTrue(l2.is_terminal())
                    self.assertEqual(l1.name, l2.name)
                    self.assertLessEqual(l1.branch_length - l2.branch_length, 1E-5)
        os.remove(tree_path)

    def test_write_out_et_tree(self):
        phylo_tree = PhylogeneticTree(tree_building_method='et', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        tree_path = os.path.join(os.getcwd(), 'test.nhx')
        self.assertFalse(os.path.isfile(tree_path))
        phylo_tree.write_out_tree(filename=tree_path)
        self.assertTrue(os.path.isfile(tree_path))
        phylo_tree2 = PhylogeneticTree(tree_building_method='custom',
                                       tree_building_args={'tree_path': tree_path})
        phylo_tree2.construct_tree(dm=min_dm)
        self.assertEqual(phylo_tree.size, phylo_tree2.size)
        self.assertEqual(phylo_tree.tree.root.name, phylo_tree2.tree.root.name)
        self.assertEqual(phylo_tree.tree.root.branch_length, phylo_tree2.tree.root.branch_length)
        first_children = phylo_tree.tree.root.clades
        first_children2 = phylo_tree2.tree.root.clades
        first_children_combined = zip(first_children, first_children2)
        for c1, c2 in first_children_combined:
            self.assertEqual(c1.is_terminal(), c2.is_terminal())
            self.assertEqual(c1.name, c2.name)
            self.assertLessEqual(c1.branch_length - c2.branch_length, 1E-5)
            if not c1.is_terminal():
                second_children = c1.clades
                second_children2 = c2.clades
                second_children_combined = zip(second_children, second_children2)
                for l1, l2 in second_children_combined:
                    self.assertTrue(l1.is_terminal())
                    self.assertTrue(l2.is_terminal())
                    self.assertEqual(l1.name, l2.name)
                    self.assertLessEqual(l1.branch_length - l2.branch_length, 1E-5)
        os.remove(tree_path)

    def test_write_out_agglomerative_tree(self):
        cache_dir_path = os.path.join(os.getcwd(), 'test')
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'cache_dir': cache_dir_path,
                                                          'affinity': 'euclidean',
                                                          'linkage': 'ward'})
        phylo_tree.construct_tree(dm=dm)
        tree_path = os.path.join(os.getcwd(), 'test.nhx')
        self.assertFalse(os.path.isfile(tree_path))
        phylo_tree.write_out_tree(filename=tree_path)
        self.assertTrue(os.path.isfile(tree_path))
        phylo_tree2 = PhylogeneticTree(tree_building_method='custom',
                                       tree_building_args={'tree_path': tree_path})
        phylo_tree2.construct_tree(dm=min_dm)
        self.assertEqual(phylo_tree.size, phylo_tree2.size)
        self.assertEqual(phylo_tree.tree.root.name, phylo_tree2.tree.root.name)
        self.assertEqual(phylo_tree.tree.root.branch_length, phylo_tree2.tree.root.branch_length)
        first_children = phylo_tree.tree.root.clades
        first_children2 = phylo_tree2.tree.root.clades
        first_children_combined = zip(first_children, first_children2)
        for c1, c2 in first_children_combined:
            self.assertEqual(c1.is_terminal(), c2.is_terminal())
            self.assertEqual(c1.name, c2.name)
            self.assertLessEqual(c1.branch_length - c2.branch_length, 1E-5)
            if not c1.is_terminal():
                second_children = c1.clades
                second_children2 = c2.clades
                second_children_combined = zip(second_children, second_children2)
                for l1, l2 in second_children_combined:
                    self.assertTrue(l1.is_terminal())
                    self.assertTrue(l2.is_terminal())
                    self.assertEqual(l1.name, l2.name)
                    self.assertLessEqual(l1.branch_length - l2.branch_length, 1E-5)
        os.remove(tree_path)
        rmtree(cache_dir_path)

    def test_write_out_custom_tree(self):
        test_tree_path = os.path.join(os.getcwd(), 'test.nhx')
        with open(os.path.join(os.getcwd(), 'test.nhx'), 'w') as handle:
            handle.write('(seq1:0.1,(seq2:0.05,seq3:0.05)Inner2:0.29167)Inner1:0.00000;')
        phylo_tree = PhylogeneticTree(tree_building_method='custom',
                                      tree_building_args={'tree_path': test_tree_path})
        phylo_tree.construct_tree(dm=min_dm)
        tree_path = os.path.join(os.getcwd(), 'test2.nhx')
        self.assertFalse(os.path.isfile(tree_path))
        phylo_tree.write_out_tree(filename=tree_path)
        self.assertTrue(os.path.isfile(tree_path))
        phylo_tree2 = PhylogeneticTree(tree_building_method='custom',
                                       tree_building_args={'tree_path': tree_path})
        phylo_tree2.construct_tree(dm=min_dm)
        self.assertEqual(phylo_tree.size, phylo_tree2.size)
        self.assertEqual(phylo_tree.tree.root.name, phylo_tree2.tree.root.name)
        self.assertEqual(phylo_tree.tree.root.branch_length, phylo_tree2.tree.root.branch_length)
        first_children = phylo_tree.tree.root.clades
        first_children2 = phylo_tree2.tree.root.clades
        first_children_combined = zip(first_children, first_children2)
        for c1, c2 in first_children_combined:
            self.assertEqual(c1.is_terminal(), c2.is_terminal())
            self.assertEqual(c1.name, c2.name)
            self.assertLessEqual(c1.branch_length - c2.branch_length, 1E-5)
            if not c1.is_terminal():
                second_children = c1.clades
                second_children2 = c2.clades
                second_children_combined = zip(second_children, second_children2)
                for l1, l2 in second_children_combined:
                    self.assertTrue(l1.is_terminal())
                    self.assertTrue(l2.is_terminal())
                    self.assertEqual(l1.name, l2.name)
                    self.assertLessEqual(l1.branch_length - l2.branch_length, 1E-5)
        os.remove(tree_path)
        os.remove(test_tree_path)

    def test_write_out_failure_no_tree(self):
        with self.assertRaises(ValueError):
            phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
            tree_path = os.path.join(os.getcwd(), 'test.nhx')
            self.assertFalse(os.path.isfile(tree_path))
            phylo_tree.write_out_tree(filename=tree_path)


class TestPhylogeneticTreeTraversal(TestCase):

    def test_traverse_top_down(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        gen = phylo_tree.traverse_top_down()
        nodes = [n.name for n in gen]
        expected_nodes = ['Inner1', 'Inner2', 'seq1', 'seq2', 'seq3']
        self.assertEqual(nodes, expected_nodes)

    def test_traverse_bottom_up(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        print(phylo_tree.tree)
        gen = phylo_tree.traverse_bottom_up()
        nodes = [n.name for n in gen]
        expected_nodes = ['seq1', 'seq2', 'seq3', 'Inner2', 'Inner1']
        self.assertEqual(nodes, expected_nodes)

    def test_traverse_by_rank(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        gen = phylo_tree.traverse_by_rank()
        expected_nodes = [['Inner1'], ['Inner2', 'seq1'], ['seq3', 'seq2', 'seq1']]
        for i, nodes in enumerate(gen):
            node_names = [n.name for n in nodes]
            self.assertEqual(node_names, expected_nodes[i])


class TestPhylogeneticTreeRenameInternalNodes(TestCase):

    def test_rename_internal_nodes_upgma(self):
        phylo_tree = PhylogeneticTree()
        phylo_tree.distance_matrix = dm
        tree = phylo_tree._upgma_tree()
        phylo_tree.tree = tree
        node_names_1 = [n.name for n in phylo_tree.traverse_top_down()]
        phylo_tree.size = len(dm)
        phylo_tree.rename_internal_nodes()
        node_names_2 = [n.name for n in phylo_tree.traverse_top_down()]
        self.assertNotEqual(node_names_1, node_names_2)
        expected_nodes = ['Inner1', 'Inner2', 'seq1', 'seq2', 'seq3']
        self.assertEqual(node_names_2, expected_nodes)

    def test_rename_internal_nodes_et(self):
        phylo_tree = PhylogeneticTree(tree_building_method='et', tree_building_args={})
        phylo_tree.distance_matrix = dm
        tree = phylo_tree._et_tree()
        phylo_tree.tree = tree
        node_names_1 = [n.name for n in phylo_tree.traverse_top_down()]
        phylo_tree.size = len(dm)
        phylo_tree.rename_internal_nodes()
        node_names_2 = [n.name for n in phylo_tree.traverse_top_down()]
        self.assertEqual(node_names_1, node_names_2)
        expected_nodes = ['Inner1', 'Inner2', 'seq1', 'seq2', 'seq3']
        self.assertEqual(node_names_2, expected_nodes)

    def test_rename_internal_nodes_agglomerative(self):
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'cache_dir': os.getcwd(),
                                                          'affinity': 'euclidean',
                                                          'linkage': 'ward'})
        phylo_tree.distance_matrix = dm
        tree = phylo_tree._agglomerative_clustering(cache_dir=phylo_tree.tree_args['cache_dir'],
                                                    affinity=phylo_tree.tree_args['affinity'],
                                                    linkage=phylo_tree.tree_args['linkage'])
        cache_dir_path = os.path.join(os.getcwd(), 'joblib')
        rmtree(cache_dir_path)
        phylo_tree.tree = tree
        node_names_1 = [n.name for n in phylo_tree.traverse_top_down()]
        phylo_tree.size = len(dm)
        phylo_tree.rename_internal_nodes()
        node_names_2 = [n.name for n in phylo_tree.traverse_top_down()]
        self.assertEqual(node_names_1, node_names_2)
        expected_nodes = ['Inner1', 'Inner2', 'seq1', 'seq2', 'seq3']
        self.assertEqual(node_names_2, expected_nodes)

    def test_rename_internal_nodes_custom(self):
        test_tree_path = os.path.join(os.getcwd(), 'test.nhx')
        with open(os.path.join(os.getcwd(), 'test.nhx'), 'w') as handle:
            handle.write('(seq1:0.15,(seq2:0.05,seq3:0.05)Inner1:0.1)Inner2:0.00000;')
        phylo_tree = PhylogeneticTree(tree_building_method='custom',
                                      tree_building_args={'tree_path': test_tree_path})
        phylo_tree.distance_matrix = min_dm
        tree = phylo_tree._custom_tree(tree_path=phylo_tree.tree_args['tree_path'])
        phylo_tree.tree = tree
        node_names_1 = [n.name for n in phylo_tree.traverse_top_down()]
        phylo_tree.size = len(dm)
        phylo_tree.rename_internal_nodes()
        node_names_2 = [n.name for n in phylo_tree.traverse_top_down()]
        self.assertNotEqual(node_names_1, node_names_2)
        expected_nodes = ['Inner1', 'Inner2', 'seq1', 'seq2', 'seq3']
        self.assertEqual(node_names_2, expected_nodes)
        os.remove(test_tree_path)

    def test_rename_internal_nodes_failure_no_tree(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        with self.assertRaises(ValueError):
            phylo_tree.rename_internal_nodes()


class TestPhylogeneticTreeAssignRank(TestCase):

    def test_assign_group_rank_none(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        rank_dict = phylo_tree.assign_group_rank(ranks=None)
        expected_rank_dict = {1: {1: {'node': 'Inner1',
                                      'terminals': ['seq3', 'seq2', 'seq1'],
                                      'descendants': ['Inner2', 'seq1']}},
                              2: {1: {'node': 'Inner2',
                                      'terminals': ['seq3', 'seq2'],
                                      'descendants': ['seq2', 'seq3']},
                                  2: {'node': 'seq1',
                                      'terminals': ['seq1'],
                                      'descendants': None}},
                              3: {1: {'node': 'seq3',
                                      'terminals': ['seq3'],
                                      'descendants': None},
                                  2: {'node': 'seq2',
                                      'terminals': ['seq2'],
                                      'descendants': None},
                                  3: {'node': 'seq1',
                                      'terminals': ['seq1'],
                                      'descendants': None}}}
        print(rank_dict)
        self.assertEqual(len(rank_dict), len(expected_rank_dict))
        for rank in rank_dict:
            print(rank)
            self.assertTrue(rank in expected_rank_dict)
            self.assertEqual(len(rank_dict[rank]), len(expected_rank_dict[rank]))
            for group in rank_dict[rank]:
                print(f'\t{group}')
                self.assertTrue(group in expected_rank_dict[rank])
                self.assertEqual(len(rank_dict[rank][group]), len(expected_rank_dict[rank][group]))
                for field in rank_dict[rank][group]:
                    print(f'\t\t{field}')
                    self.assertTrue(field in expected_rank_dict[rank][group])
                    if field == 'node':
                        self.assertEqual(rank_dict[rank][group][field].name, expected_rank_dict[rank][group][field])
                    else:
                        try:
                            for i in range(len(rank_dict[rank][group][field])):
                                if field == 'descendants':
                                    self.assertTrue(rank_dict[rank][group][field][i].name in
                                                    expected_rank_dict[rank][group][field])
                                else:
                                    self.assertTrue(rank_dict[rank][group][field][i] in
                                                    expected_rank_dict[rank][group][field])
                        except TypeError:
                            self.assertIsNone(rank_dict[rank][group][field])
                            self.assertIsNone(expected_rank_dict[rank][group][field])

    def test_assign_group_rank_all(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        rank_dict = phylo_tree.assign_group_rank(ranks={1, 2, 3})
        expected_rank_dict = {1: {1: {'node': 'Inner1',
                                      'terminals': ['seq3', 'seq2', 'seq1'],
                                      'descendants': ['Inner2', 'seq1']}},
                              2: {1: {'node': 'Inner2',
                                      'terminals': ['seq3', 'seq2'],
                                      'descendants': ['seq2', 'seq3']},
                                  2: {'node': 'seq1',
                                      'terminals': ['seq1'],
                                      'descendants': None}},
                              3: {1: {'node': 'seq3',
                                      'terminals': ['seq3'],
                                      'descendants': None},
                                  2: {'node': 'seq2',
                                      'terminals': ['seq2'],
                                      'descendants': None},
                                  3: {'node': 'seq1',
                                      'terminals': ['seq1'],
                                      'descendants': None}}}
        print(rank_dict)
        self.assertEqual(len(rank_dict), len(expected_rank_dict))
        for rank in rank_dict:
            print(rank)
            self.assertTrue(rank in expected_rank_dict)
            self.assertEqual(len(rank_dict[rank]), len(expected_rank_dict[rank]))
            for group in rank_dict[rank]:
                print(f'\t{group}')
                self.assertTrue(group in expected_rank_dict[rank])
                self.assertEqual(len(rank_dict[rank][group]), len(expected_rank_dict[rank][group]))
                for field in rank_dict[rank][group]:
                    print(f'\t\t{field}')
                    self.assertTrue(field in expected_rank_dict[rank][group])
                    if field == 'node':
                        self.assertEqual(rank_dict[rank][group][field].name, expected_rank_dict[rank][group][field])
                    else:
                        try:
                            for i in range(len(rank_dict[rank][group][field])):
                                if field == 'descendants':
                                    self.assertTrue(rank_dict[rank][group][field][i].name in
                                                    expected_rank_dict[rank][group][field])
                                else:
                                    self.assertTrue(rank_dict[rank][group][field][i] in
                                                    expected_rank_dict[rank][group][field])
                        except TypeError:
                            self.assertIsNone(rank_dict[rank][group][field])
                            self.assertIsNone(expected_rank_dict[rank][group][field])

    def test_assign_group_rank_not_root(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        rank_dict = phylo_tree.assign_group_rank(ranks={2, 3})
        expected_rank_dict = {1: {1: {'node': 'Inner1',
                                      'terminals': ['seq3', 'seq2', 'seq1'],
                                      'descendants': ['Inner2', 'seq1']}},
                              2: {1: {'node': 'Inner2',
                                      'terminals': ['seq3', 'seq2'],
                                      'descendants': ['seq2', 'seq3']},
                                  2: {'node': 'seq1',
                                      'terminals': ['seq1'],
                                      'descendants': None}},
                              3: {1: {'node': 'seq3',
                                      'terminals': ['seq3'],
                                      'descendants': None},
                                  2: {'node': 'seq2',
                                      'terminals': ['seq2'],
                                      'descendants': None},
                                  3: {'node': 'seq1',
                                      'terminals': ['seq1'],
                                      'descendants': None}}}
        print(rank_dict)
        self.assertEqual(len(rank_dict), len(expected_rank_dict))
        for rank in rank_dict:
            print(rank)
            self.assertTrue(rank in expected_rank_dict)
            self.assertEqual(len(rank_dict[rank]), len(expected_rank_dict[rank]))
            for group in rank_dict[rank]:
                print(f'\t{group}')
                self.assertTrue(group in expected_rank_dict[rank])
                self.assertEqual(len(rank_dict[rank][group]), len(expected_rank_dict[rank][group]))
                for field in rank_dict[rank][group]:
                    print(f'\t\t{field}')
                    self.assertTrue(field in expected_rank_dict[rank][group])
                    if field == 'node':
                        self.assertEqual(rank_dict[rank][group][field].name, expected_rank_dict[rank][group][field])
                    else:
                        try:
                            for i in range(len(rank_dict[rank][group][field])):
                                if field == 'descendants':
                                    self.assertTrue(rank_dict[rank][group][field][i].name in
                                                    expected_rank_dict[rank][group][field])
                                else:
                                    self.assertTrue(rank_dict[rank][group][field][i] in
                                                    expected_rank_dict[rank][group][field])
                        except TypeError:
                            self.assertIsNone(rank_dict[rank][group][field])
                            self.assertIsNone(expected_rank_dict[rank][group][field])

    def test_assign_group_rank_not_leaves(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        rank_dict = phylo_tree.assign_group_rank(ranks={1, 2})
        expected_rank_dict = {1: {1: {'node': 'Inner1',
                                      'terminals': ['seq3', 'seq2', 'seq1'],
                                      'descendants': ['Inner2', 'seq1']}},
                              2: {1: {'node': 'Inner2',
                                      'terminals': ['seq3', 'seq2'],
                                      'descendants': None},
                                  2: {'node': 'seq1',
                                      'terminals': ['seq1'],
                                      'descendants': None}}}
        print(rank_dict)
        self.assertEqual(len(rank_dict), len(expected_rank_dict))
        for rank in rank_dict:
            print(rank)
            self.assertTrue(rank in expected_rank_dict)
            self.assertEqual(len(rank_dict[rank]), len(expected_rank_dict[rank]))
            for group in rank_dict[rank]:
                print(f'\t{group}')
                self.assertTrue(group in expected_rank_dict[rank])
                self.assertEqual(len(rank_dict[rank][group]), len(expected_rank_dict[rank][group]))
                for field in rank_dict[rank][group]:
                    print(f'\t\t{field}')
                    self.assertTrue(field in expected_rank_dict[rank][group])
                    if field == 'node':
                        self.assertEqual(rank_dict[rank][group][field].name, expected_rank_dict[rank][group][field])
                    else:
                        try:
                            for i in range(len(rank_dict[rank][group][field])):
                                if field == 'descendants':
                                    self.assertTrue(rank_dict[rank][group][field][i].name in
                                                    expected_rank_dict[rank][group][field])
                                else:
                                    self.assertTrue(rank_dict[rank][group][field][i] in
                                                    expected_rank_dict[rank][group][field])
                        except TypeError:
                            self.assertIsNone(rank_dict[rank][group][field])
                            self.assertIsNone(expected_rank_dict[rank][group][field])

    def test_assign_group_rank_not_intermediate(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        rank_dict = phylo_tree.assign_group_rank(ranks={1, 3})
        expected_rank_dict = {1: {1: {'node': 'Inner1',
                                      'terminals': ['seq3', 'seq2', 'seq1'],
                                      'descendants': ['seq1', 'seq2', 'seq3']}},
                              3: {1: {'node': 'seq3',
                                      'terminals': ['seq3'],
                                      'descendants': None},
                                  2: {'node': 'seq2',
                                      'terminals': ['seq2'],
                                      'descendants': None},
                                  3: {'node': 'seq1',
                                      'terminals': ['seq1'],
                                      'descendants': None}}}
        print(rank_dict)
        self.assertEqual(len(rank_dict), len(expected_rank_dict))
        for rank in rank_dict:
            print(rank)
            self.assertTrue(rank in expected_rank_dict)
            self.assertEqual(len(rank_dict[rank]), len(expected_rank_dict[rank]))
            for group in rank_dict[rank]:
                print(f'\t{group}')
                self.assertTrue(group in expected_rank_dict[rank])
                self.assertEqual(len(rank_dict[rank][group]), len(expected_rank_dict[rank][group]))
                for field in rank_dict[rank][group]:
                    print(f'\t\t{field}')
                    self.assertTrue(field in expected_rank_dict[rank][group])
                    if field == 'node':
                        self.assertEqual(rank_dict[rank][group][field].name, expected_rank_dict[rank][group][field])
                    else:
                        try:
                            for i in range(len(rank_dict[rank][group][field])):
                                if field == 'descendants':
                                    self.assertTrue(rank_dict[rank][group][field][i].name in
                                                    expected_rank_dict[rank][group][field])
                                else:
                                    self.assertTrue(rank_dict[rank][group][field][i] in
                                                    expected_rank_dict[rank][group][field])
                        except TypeError:
                            self.assertIsNone(rank_dict[rank][group][field])
                            self.assertIsNone(expected_rank_dict[rank][group][field])

    def test_assign_group_rank_failure_rank_0(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        with self.assertRaises(ValueError):
            phylo_tree.assign_group_rank(ranks={0, 1, 2, 3})

    def test_assign_group_rank_failure_no_tree(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        with self.assertRaises(ValueError):
            phylo_tree.assign_group_rank()


class TestPhylogeneticTreeGetPathLength(TestCase):

    def test_get_path_length_empty_path(self):
        length = get_path_length([])
        self.assertEqual(length, 0)

    def test_get_path_length_single_path1(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        length = get_path_length([phylo_tree.tree.root])
        self.assertEqual(length, 0)

    def test_get_path_length_single_path2(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        length = get_path_length([phylo_tree.tree.root.clades[0]])
        self.assertLessEqual(length - 0.2916, 1E-4)

    def test_get_path_length_single_path3(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        length = get_path_length([phylo_tree.tree.root.clades[0].clades[0]])
        self.assertLessEqual(length - 0.083, 1E-3)

    def test_get_path_length_double_path1(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        length = get_path_length([phylo_tree.tree.root, phylo_tree.tree.root.clades[0]])
        self.assertLessEqual(length - 0.2916, 1E-4)

    def test_get_path_length_double_path2(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        length = get_path_length([phylo_tree.tree.root.clades[0], phylo_tree.tree.root.clades[0].clades[0]])
        self.assertLessEqual(length - 0.375, 1E-3)

    def test_get_path_length_triple(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        length = get_path_length([phylo_tree.tree.root, phylo_tree.tree.root.clades[0],
                                  phylo_tree.tree.root.clades[0].clades[0]])
        self.assertLessEqual(length - 0.375, 1E-3)

    def test_get_path_length_failure_broken_path(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        with self.assertRaises(ValueError):
            get_path_length([phylo_tree.tree.root, phylo_tree.tree.root.clades[0].clades[0]])


if __name__ == '__main__':
    unittest.main()
