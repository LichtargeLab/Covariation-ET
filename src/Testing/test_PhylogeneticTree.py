import os
import sys
import unittest
from shutil import rmtree
from unittest import TestCase

#
from dotenv import find_dotenv, load_dotenv
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)
source_code_path = os.path.join(os.environ.get('PROJECT_PATH'), 'src')
# Add the project path to the python path so the required clases can be imported
if source_code_path not in sys.path:
    sys.path.append(os.path.join(os.environ.get('PROJECT_PATH'), 'src'))
#

from SupportingClasses.PhylogeneticTree import PhylogeneticTree, get_path_length
from Testing.test_Base import protein_seq1, protein_seq2, protein_seq3, protein_msa, min_dm
from Testing.test_Base import id_dm as dm


class TestPhylogeneticTreeInit(TestCase):

    def evaluate_init(self, ptree, method, args):
        self.assertIsNone(ptree.distance_matrix)
        self.assertEqual(ptree.tree_method, method)
        self.assertEqual(ptree.tree_args, args)
        self.assertIsNone(ptree.tree)
        self.assertIsNone(ptree.size)

    def test_init_default(self):
        phylo_tree = PhylogeneticTree()
        self.evaluate_init(phylo_tree, 'upgma', {})

    def test_init_upgma(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        self.evaluate_init(phylo_tree, 'upgma', {})

    def test_init_et(self):
        phylo_tree = PhylogeneticTree(tree_building_method='et', tree_building_args={})
        self.evaluate_init(phylo_tree, 'et', {})

    def test_init_agglomerative(self):
        m_args = {'cache_dir': os.getcwd(), 'affinity': 'euclidean', 'linkage': 'ward'}
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args=m_args)
        self.evaluate_init(phylo_tree, 'agglomerative', m_args)

    def test_init_custom(self):
        phylo_tree = PhylogeneticTree(tree_building_method='custom',
                                      tree_building_args={'tree_path': os.path.join(os.getcwd(), 'test.nhx')})
        self.evaluate_init(phylo_tree, 'custom', {'tree_path': os.path.join(os.getcwd(), 'test.nhx')})

    def test_init_other(self):
        phylo_tree = PhylogeneticTree(tree_building_method='fake',
                                      tree_building_args={'foo': 'bar'})
        self.evaluate_init(phylo_tree, 'fake', {'foo': 'bar'})


class TestPhylogeneticTreeConstructTree(TestCase):

    def evaluate_private_construct_method(self, ptree, tree):
        self.assertIsNone(ptree.size)
        self.assertIsNone(ptree.tree)
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

    def test__upgma_tree(self):
        phylo_tree = PhylogeneticTree()
        phylo_tree.distance_matrix = dm
        tree = phylo_tree._upgma_tree()
        self.evaluate_private_construct_method(phylo_tree, tree)
        
    def test__upgma_tree_failure_no_distance_matrix(self):
        with self.assertRaises(TypeError):
            phylo_tree = PhylogeneticTree()
            phylo_tree._upgma_tree()

    def test__et_tree(self):
        phylo_tree = PhylogeneticTree(tree_building_method='et', tree_building_args={})
        phylo_tree.distance_matrix = dm
        tree = phylo_tree._et_tree()
        self.evaluate_private_construct_method(phylo_tree, tree)

    def test__et_tree_failure_no_distance_matrix(self):
        with self.assertRaises(TypeError):
            phylo_tree = PhylogeneticTree(tree_building_method='et', tree_building_args={})
            phylo_tree._et_tree()

    def test__agglomerative_tree(self):
        m_args = {'cache_dir': os.getcwd(), 'affinity': 'euclidean', 'linkage': 'ward'}
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative', tree_building_args=m_args)
        phylo_tree.distance_matrix = dm
        tree = phylo_tree._agglomerative_clustering(cache_dir=m_args['cache_dir'],
                                                    affinity=m_args['affinity'],
                                                    linkage=m_args['linkage'])
        self.evaluate_private_construct_method(phylo_tree, tree)

    def test__agglomerative_tree_new_cache_dir(self):
        m_args = {'cache_dir': os.path.join(os.getcwd(), 'test'), 'affinity': 'euclidean', 'linkage': 'ward'}
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative', tree_building_args=m_args)
        phylo_tree.distance_matrix = dm
        tree = phylo_tree._agglomerative_clustering(cache_dir=m_args['cache_dir'],
                                                    affinity=m_args['affinity'],
                                                    linkage=m_args['linkage'])
        self.evaluate_private_construct_method(phylo_tree, tree)

    def test__agglomerative_tree_failure_no_linkage(self):
        m_args = {'cache_dir': None, 'affinity': None, 'linkage': 'ward'}
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative', tree_building_args=m_args)
        phylo_tree.distance_matrix = dm
        with self.assertRaises(ValueError):
            phylo_tree._agglomerative_clustering(cache_dir=m_args['cache_dir'],
                                                 affinity=m_args['affinity'],
                                                 linkage=m_args['linkage'])

    def test__agglomerative_tree_failure_no_affinity(self):
        m_args = {'cache_dir': None, 'affinity': 'euclidean', 'linkage': None}
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative', tree_building_args=m_args)
        phylo_tree.distance_matrix = dm
        with self.assertRaises(ValueError):
            phylo_tree._agglomerative_clustering(cache_dir=m_args['cache_dir'],
                                                 affinity=m_args['affinity'],
                                                 linkage=m_args['linkage'])

    def test__agglomerative_tree_no_distance_matrix(self):
        m_args = {'cache_dir': None, 'affinity': 'euclidean', 'linkage': 'ward'}
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative', tree_building_args=m_args)
        with self.assertRaises(TypeError):
            phylo_tree._agglomerative_clustering(cache_dir=m_args['cache_dir'],
                                                 affinity=m_args['affinity'],
                                                 linkage=m_args['linkage'])

    def test__custom_tree(self):
        test_tree_path = os.path.join(os.getcwd(), 'test.nhx')
        with open(os.path.join(os.getcwd(), 'test.nhx'), 'w') as handle:
            handle.write('(seq1:0.1,(seq2:0.05,seq3:0.05)Inner2:0.29167)Inner1:0.00000;')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': test_tree_path})
        phylo_tree.distance_matrix = min_dm
        tree = phylo_tree._custom_tree(tree_path=phylo_tree.tree_args['tree_path'])
        self.evaluate_private_construct_method(phylo_tree, tree)
        os.remove(test_tree_path)

    def test__custom_tree_no_distance_matrix(self):
        test_tree_path = os.path.join(os.getcwd(), 'test.nhx')
        with open(os.path.join(os.getcwd(), 'test.nhx'), 'w') as handle:
            handle.write('(seq1:0.1,(seq2:0.05,seq3:0.05)Inner2:0.29167)Inner1:0.00000;')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': test_tree_path})
        tree = phylo_tree._custom_tree(tree_path=phylo_tree.tree_args['tree_path'])
        self.evaluate_private_construct_method(phylo_tree, tree)
        os.remove(test_tree_path)

    def test__custom_tree_failure_no_path(self):
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': None})
        phylo_tree.distance_matrix = min_dm
        with self.assertRaises(TypeError):
            phylo_tree._custom_tree(tree_path=phylo_tree.tree_args['tree_path'])

    def evaluate_construct_method(self, p_tree):
        self.assertIsNotNone(p_tree.size)
        self.assertEqual(p_tree.size, 3)
        self.assertIsNotNone(p_tree.tree)
        first_children = p_tree.tree.root.clades
        self.assertEqual(len(first_children), 2)
        for n1 in first_children:
            if n1.is_terminal():
                self.assertEqual(n1.name, 'seq1')
            else:
                second_children = n1.clades
                self.assertEqual(len(second_children), 2)
                for n2 in second_children:
                    self.assertTrue(n2.name in {'seq2', 'seq3'})

    def test_construct_tree_upgma(self):
        phylo_tree = PhylogeneticTree()
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_construct_method(phylo_tree)

    def test_construct_tree_upgma_failure_no_distance_matrix(self):
        phylo_tree = PhylogeneticTree()
        with self.assertRaises(ValueError):
            phylo_tree.construct_tree(dm=None)

    def test_construct_tree_et(self):
        phylo_tree = PhylogeneticTree(tree_building_method='et', tree_building_args={})

        phylo_tree.construct_tree(dm=dm)
        self.evaluate_construct_method(phylo_tree)

    def test_construct_tree_et_failure_no_distance_matrix(self):
        phylo_tree = PhylogeneticTree(tree_building_method='et', tree_building_args={})
        with self.assertRaises(ValueError):
            phylo_tree.construct_tree(dm=None)

    def test_construct_tree_agglomerative(self):
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'cache_dir': os.getcwd(), 'affinity': 'euclidean',
                                                          'linkage': 'ward'})
        phylo_tree.construct_tree(dm=dm)
        cache_dir_path = os.path.join(os.getcwd(), 'joblib')
        self.assertTrue(os.path.isdir(cache_dir_path))
        rmtree(cache_dir_path)
        self.evaluate_construct_method(phylo_tree)

    def test_construct_tree_agglomerative_new_cache_dir(self):
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'cache_dir': os.path.join(os.getcwd(), 'test'),
                                                          'affinity': 'euclidean', 'linkage': 'ward'})
        phylo_tree.construct_tree(dm=dm)
        cache_dir_path = os.path.join(os.getcwd(), 'test')
        self.assertTrue(os.path.isdir(cache_dir_path))
        rmtree(cache_dir_path)
        self.evaluate_construct_method(phylo_tree)

    def test_construct_tree_agglomerative_failure_no_linkage(self):
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'cache_dir': None, 'affinity': None, 'linkage': 'ward'})
        with self.assertRaises(ValueError):
            phylo_tree.construct_tree(dm=dm)

    def test_construct_tree_agglomerative_failure_no_affinity(self):
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'cache_dir': None, 'affinity': 'euclidean', 'linkage': None})
        with self.assertRaises(ValueError):
            phylo_tree.construct_tree(dm=dm)

    def test_construct_tree_agglomerative_failure_no_distance_matrix(self):
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'cache_dir': None, 'affinity': 'euclidean', 'linkage': 'ward'})
        with self.assertRaises(ValueError):
            phylo_tree.construct_tree(dm=None)

    def test_construct_tree_custom(self):
        test_tree_path = os.path.join(os.getcwd(), 'test.nhx')
        with open(os.path.join(os.getcwd(), 'test.nhx'), 'w') as handle:
            handle.write('(seq1:0.1,(seq2:0.05,seq3:0.05)Inner2:0.29167)Inner1:0.00000;')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': test_tree_path})
        phylo_tree.construct_tree(dm=min_dm)
        self.evaluate_construct_method(phylo_tree)
        os.remove(test_tree_path)

    def test_construct_tree_custom_failure_bad_path(self):
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': None})
        with self.assertRaises(TypeError):
            phylo_tree.construct_tree(dm=min_dm)

    def test_construct_tree_custom_failure_no_distance_matrix(self):
        test_tree_path = os.path.join(os.getcwd(), 'test.nhx')
        with open(os.path.join(os.getcwd(), 'test.nhx'), 'w') as handle:
            handle.write('(seq1:0.1,(seq2:0.05,seq3:0.05)Inner2:0.29167)Inner1:0.00000;')
        phylo_tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': test_tree_path})
        with self.assertRaises(ValueError):
            phylo_tree.construct_tree(dm=None)
        os.remove(test_tree_path)

    def test_construct_tree_failure_bad_method(self):
        phylo_tree = PhylogeneticTree(tree_building_method='fake', tree_building_args={'foo': 'bar'})
        with self.assertRaises(KeyError):
            phylo_tree.construct_tree(dm=dm)


class TestPhylogeneticTreeWriteOutTree(TestCase):

    def evaluate_write_out_tree(self, p_tree, t_path):
        self.assertFalse(os.path.isfile(t_path))
        p_tree.write_out_tree(filename=t_path)
        self.assertTrue(os.path.isfile(t_path))
        phylo_tree2 = PhylogeneticTree(tree_building_method='custom',
                                       tree_building_args={'tree_path': t_path})
        phylo_tree2.construct_tree(dm=min_dm)
        self.assertEqual(p_tree.size, phylo_tree2.size)
        self.assertEqual(p_tree.tree.root.name, phylo_tree2.tree.root.name)
        self.assertEqual(p_tree.tree.root.branch_length, phylo_tree2.tree.root.branch_length)
        first_children = p_tree.tree.root.clades
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
        os.remove(t_path)

    def test_write_out_upgma_tree(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        tree_path = os.path.join(os.getcwd(), 'test.nhx')
        self.evaluate_write_out_tree(phylo_tree, tree_path)

    def test_write_out_et_tree(self):
        phylo_tree = PhylogeneticTree(tree_building_method='et', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        tree_path = os.path.join(os.getcwd(), 'test.nhx')
        self.evaluate_write_out_tree(phylo_tree, tree_path)

    def test_write_out_agglomerative_tree(self):
        cache_dir_path = os.path.join(os.getcwd(), 'test')
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'cache_dir': cache_dir_path,
                                                          'affinity': 'euclidean',
                                                          'linkage': 'ward'})
        phylo_tree.construct_tree(dm=dm)
        tree_path = os.path.join(os.getcwd(), 'test.nhx')
        self.evaluate_write_out_tree(phylo_tree, tree_path)
        rmtree(cache_dir_path)

    def test_write_out_custom_tree(self):
        test_tree_path = os.path.join(os.getcwd(), 'test.nhx')
        with open(os.path.join(os.getcwd(), 'test.nhx'), 'w') as handle:
            handle.write('(seq1:0.1,(seq2:0.05,seq3:0.05)Inner2:0.29167)Inner1:0.00000;')
        phylo_tree = PhylogeneticTree(tree_building_method='custom',
                                      tree_building_args={'tree_path': test_tree_path})
        phylo_tree.construct_tree(dm=min_dm)
        tree_path = os.path.join(os.getcwd(), 'test2.nhx')
        self.evaluate_write_out_tree(phylo_tree, tree_path)
        os.remove(test_tree_path)

    def test_write_out_failure_no_tree(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        tree_path = os.path.join(os.getcwd(), 'test.nhx')
        self.assertFalse(os.path.isfile(tree_path))
        with self.assertRaises(ValueError):
            phylo_tree.write_out_tree(filename=tree_path)


class TestPhylogeneticTreeTraversal(TestCase):

    def evaluate_traversal(self, dist_mat, expectation, method):
        p_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        p_tree.construct_tree(dm=dist_mat)
        if method == 'top_down':
            gen = p_tree.traverse_top_down()
        elif method == 'bottom_up':
            gen = p_tree.traverse_bottom_up()
        elif method == 'by_rank':
            gen = p_tree.traverse_by_rank()
        else:
            raise ValueError('Bad traversal method.')
        nodes = []
        for level in gen:
            try:
                nodes.append(level.name)
            except AttributeError:
                nodes.append([x.name for x in level])
        self.assertEqual(nodes, expectation)

    def test_traverse_top_down(self):
        expected_nodes = ['Inner1', 'Inner2', 'seq1', 'seq2', 'seq3']
        self.evaluate_traversal(dm, expected_nodes, method='top_down')

    def test_traverse_bottom_up(self):
        expected_nodes = ['seq1', 'seq2', 'seq3', 'Inner2', 'Inner1']
        self.evaluate_traversal(dm, expected_nodes, method='bottom_up')

    def test_traverse_by_rank(self):
        expected_nodes = [['Inner1'], ['Inner2', 'seq1'], ['seq3', 'seq2', 'seq1']]
        self.evaluate_traversal(dm, expected_nodes, method='by_rank')


class TestPhylogeneticTreeRenameInternalNodes(TestCase):

    def evaluate_rename_internal_nodes(self, p_tree, tree, equality, expectation):
        p_tree.tree = tree
        node_names_1 = [n.name for n in p_tree.traverse_top_down()]
        p_tree.size = len(dm)
        p_tree.rename_internal_nodes()
        node_names_2 = [n.name for n in p_tree.traverse_top_down()]
        if equality:
            self.assertEqual(node_names_1, node_names_2)
        else:
            self.assertNotEqual(node_names_1, node_names_2)
        self.assertEqual(node_names_2, expectation)

    def test_rename_internal_nodes_upgma(self):
        phylo_tree = PhylogeneticTree()
        phylo_tree.distance_matrix = dm
        tree = phylo_tree._upgma_tree()
        expected_nodes = ['Inner1', 'Inner2', 'seq1', 'seq2', 'seq3']
        self.evaluate_rename_internal_nodes(phylo_tree, tree, False, expected_nodes)

    def test_rename_internal_nodes_et(self):
        phylo_tree = PhylogeneticTree(tree_building_method='et', tree_building_args={})
        phylo_tree.distance_matrix = dm
        tree = phylo_tree._et_tree()
        expected_nodes = ['Inner1', 'Inner2', 'seq1', 'seq2', 'seq3']
        self.evaluate_rename_internal_nodes(phylo_tree, tree, True, expected_nodes)

    def test_rename_internal_nodes_agglomerative(self):
        phylo_tree = PhylogeneticTree(tree_building_method='agglomerative',
                                      tree_building_args={'cache_dir': os.getcwd(), 'affinity': 'euclidean',
                                                          'linkage': 'ward'})
        phylo_tree.distance_matrix = dm
        tree = phylo_tree._agglomerative_clustering(cache_dir=phylo_tree.tree_args['cache_dir'],
                                                    affinity=phylo_tree.tree_args['affinity'],
                                                    linkage=phylo_tree.tree_args['linkage'])
        expected_nodes = ['Inner1', 'Inner2', 'seq1', 'seq2', 'seq3']
        self.evaluate_rename_internal_nodes(phylo_tree, tree, True, expected_nodes)
        cache_dir_path = os.path.join(os.getcwd(), 'joblib')
        rmtree(cache_dir_path)

    def test_rename_internal_nodes_custom(self):
        test_tree_path = os.path.join(os.getcwd(), 'test.nhx')
        with open(os.path.join(os.getcwd(), 'test.nhx'), 'w') as handle:
            handle.write('(seq1:0.15,(seq2:0.05,seq3:0.05)Inner1:0.1)Inner2:0.00000;')
        phylo_tree = PhylogeneticTree(tree_building_method='custom',
                                      tree_building_args={'tree_path': test_tree_path})
        phylo_tree.distance_matrix = min_dm
        tree = phylo_tree._custom_tree(tree_path=phylo_tree.tree_args['tree_path'])
        expected_nodes = ['Inner1', 'Inner2', 'seq1', 'seq2', 'seq3']
        self.evaluate_rename_internal_nodes(phylo_tree, tree, False, expected_nodes)
        os.remove(test_tree_path)

    def test_rename_internal_nodes_failure_no_tree(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        with self.assertRaises(ValueError):
            phylo_tree.rename_internal_nodes()


class TestPhylogeneticTreeAssignRank(TestCase):

    def evaluate_group_rank_dict(self, r_dict, expectation):
        self.assertEqual(len(r_dict), len(expectation))
        for rank in r_dict:
            self.assertTrue(rank in expectation)
            self.assertEqual(len(r_dict[rank]), len(expectation[rank]))
            for group in r_dict[rank]:
                self.assertTrue(group in expectation[rank])
                self.assertEqual(len(r_dict[rank][group]), len(expectation[rank][group]))
                for field in r_dict[rank][group]:
                    self.assertTrue(field in expectation[rank][group])
                    if field == 'node':
                        self.assertEqual(r_dict[rank][group][field].name, expectation[rank][group][field])
                    else:
                        try:
                            for i in range(len(r_dict[rank][group][field])):
                                if field == 'descendants':
                                    self.assertTrue(r_dict[rank][group][field][i].name in
                                                    expectation[rank][group][field])
                                else:
                                    self.assertTrue(r_dict[rank][group][field][i] in
                                                    expectation[rank][group][field])
                        except TypeError:
                            self.assertIsNone(r_dict[rank][group][field])
                            self.assertIsNone(expectation[rank][group][field])

    def test_assign_group_rank_none(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        rank_dict = phylo_tree.assign_group_rank(ranks=None)
        expected_rank_dict = {1: {1: {'node': 'Inner1', 'terminals': ['seq3', 'seq2', 'seq1'],
                                      'descendants': ['Inner2', 'seq1']}},
                              2: {1: {'node': 'Inner2', 'terminals': ['seq3', 'seq2'], 'descendants': ['seq2', 'seq3']},
                                  2: {'node': 'seq1', 'terminals': ['seq1'], 'descendants': None}},
                              3: {1: {'node': 'seq3', 'terminals': ['seq3'], 'descendants': None},
                                  2: {'node': 'seq2', 'terminals': ['seq2'], 'descendants': None},
                                  3: {'node': 'seq1', 'terminals': ['seq1'], 'descendants': None}}}
        self.evaluate_group_rank_dict(rank_dict, expected_rank_dict)

    def test_assign_group_rank_all(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        rank_dict = phylo_tree.assign_group_rank(ranks={1, 2, 3})
        expected_rank_dict = {1: {1: {'node': 'Inner1', 'terminals': ['seq3', 'seq2', 'seq1'],
                                      'descendants': ['Inner2', 'seq1']}},
                              2: {1: {'node': 'Inner2', 'terminals': ['seq3', 'seq2'], 'descendants': ['seq2', 'seq3']},
                                  2: {'node': 'seq1', 'terminals': ['seq1'], 'descendants': None}},
                              3: {1: {'node': 'seq3', 'terminals': ['seq3'], 'descendants': None},
                                  2: {'node': 'seq2', 'terminals': ['seq2'], 'descendants': None},
                                  3: {'node': 'seq1', 'terminals': ['seq1'], 'descendants': None}}}
        self.evaluate_group_rank_dict(rank_dict, expected_rank_dict)

    def test_assign_group_rank_not_root(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        rank_dict = phylo_tree.assign_group_rank(ranks={2, 3})
        expected_rank_dict = {1: {1: {'node': 'Inner1', 'terminals': ['seq3', 'seq2', 'seq1'],
                                      'descendants': ['Inner2', 'seq1']}},
                              2: {1: {'node': 'Inner2', 'terminals': ['seq3', 'seq2'], 'descendants': ['seq2', 'seq3']},
                                  2: {'node': 'seq1', 'terminals': ['seq1'], 'descendants': None}},
                              3: {1: {'node': 'seq3', 'terminals': ['seq3'], 'descendants': None},
                                  2: {'node': 'seq2', 'terminals': ['seq2'], 'descendants': None},
                                  3: {'node': 'seq1', 'terminals': ['seq1'], 'descendants': None}}}
        self.evaluate_group_rank_dict(rank_dict, expected_rank_dict)

    def test_assign_group_rank_not_leaves(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        rank_dict = phylo_tree.assign_group_rank(ranks={1, 2})
        expected_rank_dict = {1: {1: {'node': 'Inner1', 'terminals': ['seq3', 'seq2', 'seq1'],
                                      'descendants': ['Inner2', 'seq1']}},
                              2: {1: {'node': 'Inner2', 'terminals': ['seq3', 'seq2'], 'descendants': None},
                                  2: {'node': 'seq1', 'terminals': ['seq1'], 'descendants': None}}}
        self.evaluate_group_rank_dict(rank_dict, expected_rank_dict)

    def test_assign_group_rank_not_intermediate(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        rank_dict = phylo_tree.assign_group_rank(ranks={1, 3})
        expected_rank_dict = {1: {1: {'node': 'Inner1', 'terminals': ['seq3', 'seq2', 'seq1'],
                                      'descendants': ['seq1', 'seq2', 'seq3']}},
                              3: {1: {'node': 'seq3', 'terminals': ['seq3'], 'descendants': None},
                                  2: {'node': 'seq2', 'terminals': ['seq2'], 'descendants': None},
                                  3: {'node': 'seq1', 'terminals': ['seq1'], 'descendants': None}}}
        self.evaluate_group_rank_dict(rank_dict, expected_rank_dict)

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

    def evaluate_get_path_length(self, path, expected_length):
        length = get_path_length(path)
        self.assertLessEqual((length - expected_length), 1E-3)

    def test_get_path_length_empty_path(self):
        self.evaluate_get_path_length(path=[], expected_length=0)

    def test_get_path_length_single_path1(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_get_path_length(path=[phylo_tree.tree.root], expected_length=0)

    def test_get_path_length_single_path2(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_get_path_length(path=[phylo_tree.tree.root.clades[0]], expected_length=0.2916)

    def test_get_path_length_single_path3(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_get_path_length(path=[phylo_tree.tree.root.clades[0].clades[0]], expected_length=0.083)

    def test_get_path_length_double_path1(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_get_path_length(path=[phylo_tree.tree.root, phylo_tree.tree.root.clades[0]],
                                      expected_length=0.2916)

    def test_get_path_length_double_path2(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_get_path_length(path=[phylo_tree.tree.root.clades[0], phylo_tree.tree.root.clades[0].clades[0]],
                                      expected_length=0.375)

    def test_get_path_length_triple(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        self.evaluate_get_path_length(path=[phylo_tree.tree.root, phylo_tree.tree.root.clades[0],
                                            phylo_tree.tree.root.clades[0].clades[0]], expected_length=0.375)

    def test_get_path_length_failure_broken_path(self):
        phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
        phylo_tree.construct_tree(dm=dm)
        with self.assertRaises(ValueError):
            get_path_length([phylo_tree.tree.root, phylo_tree.tree.root.clades[0].clades[0]])


if __name__ == '__main__':
    unittest.main()
