"""
Created on February 5, 2020

@author: Daniel Konecki
"""
import os
import unittest
import numpy as np
from shutil import rmtree
from dotenv import find_dotenv, load_dotenv
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)
from test_Base import TestBase
from ETMIPWrapper import ETMIPWrapper
from utils import compute_rank_and_coverage
from PhylogeneticTree import PhylogeneticTree
from AlignmentDistanceCalculator import convert_array_to_distance_matrix
from test_PhylogeneticTree import compare_nodes_key, compare_nodes


class TestETMIPWrapper(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestETMIPWrapper, cls).setUpClass()
        cls.small_fa_fn = cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln']
        cls.large_fa_fn = cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln']
        cls.out_small_dir = os.path.join(cls.testing_dir, cls.small_structure_id)
        rmtree(cls.out_small_dir, ignore_errors=True)
        cls.out_large_dir = os.path.join(cls.testing_dir, cls.large_structure_id)
        rmtree(cls.out_large_dir, ignore_errors=True)

    def evaluate_init(self, query, aln_file, out_dir, expected_length, expected_sequence):
        self.assertFalse(os.path.isdir(os.path.abspath(out_dir)))
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        self.assertEqual(wetc.out_dir, os.path.abspath(out_dir))
        self.assertTrue(os.path.isdir(os.path.abspath(out_dir)))
        self.assertEqual(wetc.query, query)
        self.assertIsNotNone(wetc.original_aln)
        self.assertGreaterEqual(wetc.original_aln.seq_length, expected_length)
        self.assertEqual(str(wetc.original_aln.query_sequence).replace('-', ''), expected_sequence)
        self.assertEqual(wetc.original_aln_fn, os.path.join(out_dir, 'Original_Alignment.fa'))
        self.assertTrue(os.path.isfile(wetc.original_aln_fn))
        self.assertIsNotNone(wetc.non_gapped_aln)
        self.assertEqual(wetc.non_gapped_aln.seq_length, expected_length)
        self.assertEqual(wetc.non_gapped_aln.query_sequence, expected_sequence)
        self.assertEqual(wetc.non_gapped_aln_fn, os.path.join(out_dir, 'Non-Gapped_Alignment.fa'))
        self.assertTrue(os.path.isfile(wetc.non_gapped_aln_fn))
        self.assertEqual(wetc.method, 'WETC')
        self.assertIsNone(wetc.scores)
        self.assertIsNone(wetc.coverages)
        self.assertIsNone(wetc.rankings)
        self.assertIsNone(wetc.time)
        self.assertIsNone(wetc.msf_aln_fn)
        self.assertIsNone(wetc.distance_matrix)
        self.assertIsNone(wetc.tree)
        self.assertIsNone(wetc.rank_group_assignments)
        self.assertIsNone(wetc.rank_scores)
        self.assertIsNone(wetc.entropy)

    # def test_1a_init(self):
    #     self.evaluate_init(query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
    #                        expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
    #                        expected_sequence=str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq))
    #
    # def test_1b_init(self):
    #     self.evaluate_init(query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
    #                        expected_length=self.data_set.protein_data[self.large_structure_id]['Length'],
    #                        expected_sequence=str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq))

    def evaluate_convert_alignment(self, query, aln_file, out_dir):
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        wetc.convert_alignment()
        self.assertTrue(os.path.isfile(os.path.join(out_dir, 'Non-Gapped_Alignment.msf')))

    # def test_2a_convert_alignment(self):
    #     self.evaluate_convert_alignment(query=self.small_structure_id, aln_file=self.small_fa_fn,
    #                                     out_dir=self.out_small_dir)
    #
    # def test_2b_convert_alignment(self):
    #     self.evaluate_convert_alignment(query=self.large_structure_id, aln_file=self.large_fa_fn,
    #                                     out_dir=self.out_large_dir)

    def evaluate_import_rank_scores(self, query, aln_file, out_dir, expected_length):
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        scores = np.random.RandomState(1234567890).rand(expected_length)
        ranks, _ = compute_rank_and_coverage(expected_length, scores, 1, 'min')
        expected_path = os.path.join(out_dir, 'etc_out.rank_id.tsv')
        with open(expected_path, 'w') as handle:
            handle.write('Position\tRank\n')
            for i in range(expected_length):
                handle.write('{}\t{}\n'.format(i, ranks[i]))
        wetc.import_rank_scores()
        diff_ranks = wetc.rank_scores - ranks
        not_passing_ranks = diff_ranks > 1E-15
        self.assertFalse(not_passing_ranks.any())

    # def test_3a_import_rank_scores(self):
    #     self.evaluate_import_rank_scores(
    #         query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
    #         expected_length=self.data_set.protein_data[self.small_structure_id]['Length'])
    #
    # def test_3b_import_rank_scores(self):
    #     self.evaluate_import_rank_scores(
    #         query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
    #         expected_length=self.data_set.protein_data[self.large_structure_id]['Length'])

    def evaluate_import_entropy_rank_scores(self, query, aln_file, out_dir, expected_length):
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        expected_path = os.path.join(out_dir, 'etc_out.rank_plain_entropy.tsv')
        expected_positions = np.array(range(1, wetc.non_gapped_aln.seq_length + 1))
        rand_state = np.random.RandomState(1234567890)
        expected_rho = rand_state.rand(expected_length)
        expected_ranks = {x: rand_state.rand(expected_length)
                          for x in range(1, wetc.non_gapped_aln.size)}
        with open(expected_path, 'w') as handle:
            handle.write('Position\t' +
                         '\t'.join(['Rank {} Entropy'.format(x) for x in range(1, wetc.non_gapped_aln.size)]) +
                         '\tRho\n')
            for i in range(expected_length):
                handle.write('{}\t'.format(expected_positions[i]) +
                             '\t'.join([str(expected_ranks[x][i]) for x in range(1, wetc.non_gapped_aln.size)]) +
                             '\t{}\n'.format(expected_rho[i]))
        wetc.import_entropy_rank_sores()
        diff_rho = wetc.rho - expected_rho
        not_passing_rho = diff_rho > 1E-15
        self.assertFalse(not_passing_rho.any())
        for i in range(1, wetc.non_gapped_aln.size):
            diff_rank_entropy = wetc.entropy[i] - expected_ranks[i]
            not_passing_rank_entropy = diff_rank_entropy > 1E-15
            self.assertFalse(not_passing_rank_entropy.any())

    # def test_4a_import_entropy_rank_scores(self):
    #     self.evaluate_import_entropy_rank_scores(
    #         query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
    #         expected_length=self.data_set.protein_data[self.small_structure_id]['Length'])
    #
    # def test_4b_import_entropy_rank_scores(self):
    #     self.evaluate_import_entropy_rank_scores(
    #         query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
    #         expected_length=self.data_set.protein_data[self.large_structure_id]['Length'])

    def evaluate_import_distance_matrices(self, query, aln_file, out_dir):
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        rand_state = np.random.RandomState(1234567890)
        expected_id_path = os.path.join(out_dir, 'etc_out.id_dist.tsv')
        expected_id_dists = rand_state.rand(wetc.non_gapped_aln.size, wetc.non_gapped_aln.size)
        expected_id_dists[np.tril_indices(wetc.non_gapped_aln.size)] = 0.0
        expected_id_dists += expected_id_dists.T
        expected_id_dists[np.arange(wetc.non_gapped_aln.size), np.arange(wetc.non_gapped_aln.size)] = 1.0
        with open(expected_id_path, 'w') as handle:
            handle.write('Seq1/Seq2\t' + '\t'.join(wetc.non_gapped_aln.seq_order) + '\n')
            for i in range(wetc.non_gapped_aln.size):
                handle.write(wetc.non_gapped_aln.seq_order[i] + '\t' +
                             '\t'.join([str(x) for x in expected_id_dists[:, i]]) + '\n')
        expected_aln_path = os.path.join(out_dir, 'etc_out.aln_dist.tsv')
        expected_aln_dists = rand_state.rand(wetc.non_gapped_aln.size, wetc.non_gapped_aln.size)
        expected_aln_dists[np.tril_indices(wetc.non_gapped_aln.size)] = 0.0
        expected_aln_dists += expected_aln_dists.T
        with open(expected_aln_path, 'w') as handle2:
            handle2.write('Seq1/Seq2\t' + '\t'.join(wetc.non_gapped_aln.seq_order) + '\n')
            for i in range(wetc.non_gapped_aln.size):
                handle2.write(wetc.non_gapped_aln.seq_order[i] + '\t' +
                              '\t'.join([str(x) for x in expected_aln_dists[:, i]]) + '\n')
        expected_debug_path = os.path.join(out_dir, 'etc_out.debug.tsv')
        expected_thresh = rand_state.randint(low=1, high=wetc.non_gapped_aln.seq_length)
        expected_count = rand_state.randint(low=1, high=wetc.non_gapped_aln.size)
        num_comp = len(np.triu_indices(n=wetc.non_gapped_aln.size)[0])
        expected_min_seq_len = rand_state.randint(low=1, high=wetc.non_gapped_aln.seq_length, size=num_comp)
        expected_id_count = rand_state.randint(low=1, high=wetc.non_gapped_aln.seq_length, size=num_comp)
        expected_thresh_count = rand_state.randint(low=1, high=wetc.non_gapped_aln.seq_length, size=num_comp)
        expected_seq1 = []
        expected_seq2 = []
        ind = 0
        with open(expected_debug_path, 'w') as handle3:
            handle3.write('% Lines starting with % are comments\n')
            handle3.write('% Threshold: {} From Count: {}\n'.format(expected_thresh, expected_count))
            handle3.write('Seq1\tSeq2\tConsensus_Seq\tMin_Seq_Length\tID_Count\tThreshold_Count\n')
            for i in range(wetc.non_gapped_aln.size):
                for j in range(i, wetc.non_gapped_aln.size):
                    handle3.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        wetc.non_gapped_aln.seq_order[i], wetc.non_gapped_aln.seq_order[j],
                        ','.join(['XX'] * expected_id_count[ind]), expected_min_seq_len[ind], expected_id_count[ind],
                        expected_thresh_count[ind]))
                    ind += 1
                    expected_seq1.append(wetc.non_gapped_aln.seq_order[i])
                    expected_seq2.append(wetc.non_gapped_aln.seq_order[j])
        aln_dist_df, id_dist_df, debug_df = wetc.import_distance_matrices()
        diff_dist_mat = np.array(wetc.distance_matrix) - expected_aln_dists
        not_passing_dist_mat = diff_dist_mat > 1E-15
        self.assertFalse(not_passing_dist_mat.any())
        diff_aln_dist = aln_dist_df.values - expected_aln_dists
        not_passing_aln_dist = diff_aln_dist > 1E-15
        self.assertFalse(not_passing_aln_dist.any())
        diff_id_dist = id_dist_df.values - expected_id_dists
        not_passing_id_dist = diff_id_dist > 1E-15
        self.assertFalse(not_passing_id_dist.any())
        self.assertEqual(list(debug_df['Seq1']), expected_seq1)
        self.assertEqual(list(debug_df['Seq2']), expected_seq2)
        diff_min_seq_len = debug_df['Min_Seq_Length'].values - expected_min_seq_len
        not_passing_min_seq_len = diff_min_seq_len > 0
        self.assertFalse(not_passing_min_seq_len.any())
        diff_id_count = debug_df['ID_Count'].values - expected_id_count
        not_passing_id_count = diff_id_count > 0
        self.assertFalse(not_passing_id_count.any())
        diff_thresh_count = debug_df['Threshold_Count'].values - expected_thresh_count
        not_passing_thresh_count = diff_thresh_count > 0
        self.assertFalse(not_passing_thresh_count.any())

    # def test_5a_import_distance_matrices(self):
    #     self.evaluate_import_distance_matrices(
    #         query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir)
    #
    # def test_5b_import_distance_matrices(self):
    #     self.evaluate_import_distance_matrices(
    #         query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir)

    def check_nodes(self, node1, node2):
        if node1.is_terminal():
            self.assertTrue(node2.is_terminal(), 'Node1: {} vs Node2: {}'.format(node1.name, node2.name))
            self.assertEqual(node1.name, node2.name)
        else:
            self.assertTrue(node2.is_bifurcating())
            self.assertFalse(node2.is_terminal(), 'Node1: {} vs Node2: {}'.format(node1.name, node2.name))
            self.assertEqual(set([x.name for x in node1.get_terminals()]),
                             set([x.name for x in node2.get_terminals()]))

    def evaluate_import_phylogenetic_tree(self, query, aln_file, out_dir):
        def test_tree_equality(wetc_tree, phylo_tree):
            wetc_iter = wetc_tree.traverse_by_rank()
            expected_iter = phylo_tree.traverse_by_rank()
            try:
                wetc_nodes = next(wetc_iter)
            except StopIteration:
                wetc_nodes = None
            try:
                expected_nodes = next(expected_iter)
            except StopIteration:
                expected_nodes = None
            count = 1
            while wetc_nodes and expected_nodes:
                count += 1
                if wetc_nodes is None:
                    self.assertIsNone(expected_nodes)
                else:
                    sorted_wetc_nodes = sorted(wetc_nodes, key=compare_nodes_key(compare_nodes))
                    sorted_py_nodes = sorted(expected_nodes, key=compare_nodes_key(compare_nodes))
                    self.assertEqual(len(sorted_wetc_nodes), len(sorted_py_nodes))
                    for i in range(len(sorted_py_nodes)):
                        try:
                            self.check_nodes(sorted_wetc_nodes[i], sorted_py_nodes[i])
                        except AssertionError as e:
                            raise AssertionError(
                                "ERRORED ON i={}\nWETC NODE:{} WITH CHILDREN {} and {}\nPY NODE:{} with CHILDREN {} and {}".format(
                                    i, sorted_wetc_nodes[i], sorted_wetc_nodes[i].clades[0],
                                    sorted_wetc_nodes[i].clades[1],
                                    sorted_py_nodes[i], sorted_py_nodes[i].clades[0],
                                    sorted_py_nodes[i].clades[1])) from e
                try:
                    wetc_nodes = next(wetc_iter)
                except StopIteration:
                    wetc_nodes = None
                try:
                    expected_nodes = next(expected_iter)
                except StopIteration:
                    expected_nodes = None

        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        rand_state = np.random.RandomState(1234567890)
        expected_id_path = os.path.join(out_dir, 'etc_out.id_dist.tsv')
        expected_id_dists = rand_state.rand(wetc.non_gapped_aln.size, wetc.non_gapped_aln.size)
        expected_id_dists[np.tril_indices(wetc.non_gapped_aln.size)] = 0.0
        expected_id_dists += expected_id_dists.T
        expected_id_dists[np.arange(wetc.non_gapped_aln.size), np.arange(wetc.non_gapped_aln.size)] = 1.0
        with open(expected_id_path, 'w') as handle:
            handle.write('Seq1/Seq2\t' + '\t'.join(wetc.non_gapped_aln.seq_order) + '\n')
            for i in range(wetc.non_gapped_aln.size):
                handle.write(wetc.non_gapped_aln.seq_order[i] + '\t' +
                             '\t'.join([str(x) for x in expected_id_dists[:, i]]) + '\n')
        expected_aln_path = os.path.join(out_dir, 'etc_out.aln_dist.tsv')
        expected_aln_dists = rand_state.rand(wetc.non_gapped_aln.size, wetc.non_gapped_aln.size)
        expected_aln_dists[np.tril_indices(wetc.non_gapped_aln.size)] = 0.0
        expected_aln_dists += expected_aln_dists.T
        with open(expected_aln_path, 'w') as handle2:
            handle2.write('Seq1/Seq2\t' + '\t'.join(wetc.non_gapped_aln.seq_order) + '\n')
            for i in range(wetc.non_gapped_aln.size):
                handle2.write(wetc.non_gapped_aln.seq_order[i] + '\t' +
                              '\t'.join([str(x) for x in expected_aln_dists[:, i]]) + '\n')
        expected_debug_path = os.path.join(out_dir, 'etc_out.debug.tsv')
        expected_thresh = rand_state.randint(low=1, high=wetc.non_gapped_aln.seq_length)
        expected_count = rand_state.randint(low=1, high=wetc.non_gapped_aln.size)
        num_comp = len(np.triu_indices(n=wetc.non_gapped_aln.size)[0])
        expected_min_seq_len = rand_state.randint(low=1, high=wetc.non_gapped_aln.seq_length, size=num_comp)
        expected_id_count = rand_state.randint(low=1, high=wetc.non_gapped_aln.seq_length, size=num_comp)
        expected_thresh_count = rand_state.randint(low=1, high=wetc.non_gapped_aln.seq_length, size=num_comp)
        expected_seq1 = []
        expected_seq2 = []
        ind = 0
        with open(expected_debug_path, 'w') as handle3:
            handle3.write('% Lines starting with % are comments\n')
            handle3.write('% Threshold: {} From Count: {}\n'.format(expected_thresh, expected_count))
            handle3.write('Seq1\tSeq2\tConsensus_Seq\tMin_Seq_Length\tID_Count\tThreshold_Count\n')
            for i in range(wetc.non_gapped_aln.size):
                for j in range(i, wetc.non_gapped_aln.size):
                    handle3.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        wetc.non_gapped_aln.seq_order[i], wetc.non_gapped_aln.seq_order[j],
                        ','.join(['XX'] * expected_id_count[ind]), expected_min_seq_len[ind], expected_id_count[ind],
                        expected_thresh_count[ind]))
                    ind += 1
                    expected_seq1.append(wetc.non_gapped_aln.seq_order[i])
                    expected_seq2.append(wetc.non_gapped_aln.seq_order[j])
        dm = convert_array_to_distance_matrix(expected_aln_dists, wetc.non_gapped_aln.seq_order)
        pg_tree = PhylogeneticTree()
        pg_tree.construct_tree(dm)
        pg_tree.write_out_tree(os.path.join(out_dir, 'etc_out.nhx'))
        wetc.import_phylogenetic_tree()
        self.assertIsNotNone(wetc.tree)
        self.assertIsNotNone(wetc.distance_matrix)
        diff_dist_mat = np.array(wetc.distance_matrix) - expected_aln_dists
        not_passing_dist_mat = diff_dist_mat > 1E-15
        self.assertFalse(not_passing_dist_mat.any())
        os.remove(expected_aln_path)
        os.remove(expected_id_path)
        os.remove(expected_debug_path)
        wetc2 = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        wetc2.import_phylogenetic_tree()
        self.assertEqual(wetc2.distance_matrix.names, wetc.non_gapped_aln.seq_order)
        diff_dist_mat2 = np.array(wetc2.distance_matrix) - np.zeros((wetc2.non_gapped_aln.size,
                                                                     wetc2.non_gapped_aln.size))
        not_passing_dist_mat2 = diff_dist_mat2 > 1E-15
        self.assertFalse(not_passing_dist_mat2.any())
        test_tree_equality(wetc_tree=wetc2.tree, phylo_tree=pg_tree)

    # def test_6a_import_phylogenetic_tree(self):
    #     self.evaluate_import_phylogenetic_tree(query=self.small_structure_id, aln_file=self.small_fa_fn,
    #                                            out_dir=self.out_small_dir)
    #
    # def test_6b_import_phylogenetic_tree(self):
    #     self.evaluate_import_phylogenetic_tree(query=self.large_structure_id, aln_file=self.large_fa_fn,
    #                                            out_dir=self.out_large_dir)

    # def evaluate_import_assignments(self, query, aln_file, out_dir):
    #     wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
    #     rand_state = np.random.RandomState(1234567890)
    #     expected_ranks = range(1, wetc.non_gapped_aln.size + 1)
    #     expected_assignments = {i: rand_state.randint(low=1, high=i, size=wetc.non_gapped_aln.size)
    #                             for i in range(2, wetc.non_gapped_aln.size + 1)}
    #     expected_assignments[1] = [1] * wetc.non_gapped_aln.size
    #     expected_root_ranks = []
    #     expected_root_groups = []
    #     expected_root_nodes = []
    #     with open(os.path.join(out_dir, 'etc_out.group.tsv'), 'w') as handle:
    #         handle.write('% Group definitions:\n')
    #         handle.write('Rank\t' + '\t'.join(wetc.non_gapped_aln.seq_order) + '\n')
    #         for i in expected_ranks:
    #             handle.write(str(i) + '\t' + '\t'.join([str(x) for x in expected_assignments[i]]) + '\n')
    #         handle.write('% Group root:\n')
    #         handle.write('Rank\tGroup\tRoot_Node\n')
    #         for i in range(1, wetc.non_gapped_aln.size + 1):
    #             ranks = [i] * (wetc.non_gapped_aln.size + 2)
    #             groups = list(range(1, wetc.non_gapped_aln.size + 2))
    #             nodes = np.zeros(wetc.non_gapped_aln.size + 2)
    #             ind = rand_state.randint(low=1, high=wetc.non_gapped_aln.size + 2, size=i)
    #             values = rand_state.randint(low=-1 * (i - 1), high=wetc.non_gapped_aln.size + 2, size=i)
    #             nodes[ind] = values
    #             expected_root_ranks += ranks
    #             expected_root_groups += groups
    #             expected_root_nodes += list(nodes)
    #             for j in range(len(ranks)):
    #                 print('J: ', j)
    #                 print('Ranks: ', ranks[j])
    #                 print('Groups: ', groups[j])
    #                 print('Nodes: ', nodes[j])
    #                 handle.write('{}\t{}\t{}\n'.format(ranks[j], groups[j], nodes[j]))
    #     wetc.import_assignments()
    #     self.assertIsNotNone(wetc.rank_group_assignments)
    #     self.assertEqual(set(expected_ranks), set(wetc.rank_group_assignments.keys()))
    #     for r in expected_ranks:
    #         self.assertEqual(set(range(1, r + 1)), set(wetc.rank_group_assignments[r].keys()))
    #         for g in wetc.rank_group_assignments[r]:
    #             self.assertEqual(set(['node', 'terminals', 'descendants']),
    #                              set(wetc.rank_group_assignments[r][g].key()))
    #             # self.assertEqual(wetc.rank_group_assignments[r][g]['node'].name, )
    #             # self.assertEqual(wetc.rank_group_assignments[r][g]['terminals'], )
    #             # self.assertEqual(wetc.rank_group_assignments[r][g]['descendants'], )
    #
    # def test_7a_import_phylogenetic_tree(self):
    #     self.evaluate_import_assignments(query=self.small_structure_id, aln_file=self.small_fa_fn,
    #                                      out_dir=self.out_small_dir)
    #
    # def test_7b_import_phylogenetic_tree(self):
    #     self.evaluate_import_assignments(query=self.large_structure_id, aln_file=self.large_fa_fn,
    #                                      out_dir=self.out_large_dir)

    # def evaluate_import_scores(self, query, aln_file, out_dir, expected_length):
    #     evc = EVCouplingsWrapper(query=query, aln_file=aln_file, out_dir=out_dir, protocol='standard')
    #     scores = np.random.RandomState(1234567890).rand(expected_length, expected_length)
    #     scores[np.tril_indices(expected_length, 1)] = 0
    #     scores += scores.T
    #     _, coverages = compute_rank_and_coverage(expected_length, scores, 2, 'max')
    #     probabilities = 1 - coverages
    #     indices = np.triu_indices(expected_length, 1)
    #     sorted_scores, sorted_x, sorted_y, sorted_probability = zip(*sorted(zip(scores[indices], indices[0], indices[1],
    #                                                                             probabilities[indices])))
    #     expected_dir = os.path.join(out_dir, 'couplings')
    #     os.makedirs(expected_dir, exist_ok=True)
    #     expected_path = os.path.join(expected_dir, '_CouplingScores.csv')
    #     with open(expected_path, 'w') as handle:
    #         handle.write('i,A_i,j,A_j,fn,cn,segment_i,segment_j,probability\n')
    #         for i in range(len(sorted_scores)):
    #             handle.write('{},X,{},X,0,{},X,X,{}\n'.format(sorted_x[i] + 1, sorted_y[i] + 1, sorted_scores[i],
    #                                                           sorted_probability[i]))
    #     evc.import_covariance_scores(out_path=expected_path)
    #
    #     diff_expected_scores = scores - scores.T
    #     not_passing_expected_scores = diff_expected_scores > 1E-15
    #     self.assertFalse(not_passing_expected_scores.any())
    #     diff_computed_scores = evc.scores - evc.scores.T
    #     not_passing_computed_scores = diff_computed_scores > 1E-15
    #     self.assertFalse(not_passing_computed_scores.any())
    #
    #     diff_scores = evc.scores - scores
    #     not_passing_scores = diff_scores > 1E-15
    #     self.assertFalse(not_passing_scores.any())
    #     diff_probabilities = evc.probability - probabilities
    #     not_passing_protbabilities = diff_probabilities > 1E-15
    #     self.assertFalse(not_passing_protbabilities.any())
    #
    # def test_3a_import_scores(self):
    #     self.evaluate_import_scores(
    #         query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
    #         expected_length=self.data_set.protein_data[self.small_structure_id]['Length'])
    #
    # def test_3b_import_scores(self):
    #     self.evaluate_import_scores(
    #         query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
    #         expected_length=self.data_set.protein_data[self.large_structure_id]['Length'])
    #
    # def evaluate_calculator_scores(self, query, aln_file, out_dir, expected_length, expected_sequence):
    #     evc = EVCouplingsWrapper(query=query, aln_file=aln_file, out_dir=out_dir, protocol='standard')
    #     start = time()
    #     evc.calculate_scores(delete_files=False, cores=self.max_threads)
    #     end = time()
    #     expected_time = end - start
    #     self.assertEqual(evc.out_dir, os.path.abspath(out_dir))
    #     self.assertTrue(os.path.isdir(os.path.abspath(out_dir)))
    #     self.assertEqual(evc.query, query)
    #     self.assertIsNotNone(evc.original_aln)
    #     self.assertGreaterEqual(evc.original_aln.seq_length, expected_length)
    #     self.assertEqual(str(evc.original_aln.query_sequence).replace('-', ''), expected_sequence)
    #     self.assertEqual(evc.original_aln_fn, os.path.join(out_dir, 'Original_Alignment.fa'))
    #     self.assertTrue(os.path.isfile(evc.original_aln_fn))
    #     self.assertIsNotNone(evc.non_gapped_aln)
    #     self.assertEqual(evc.non_gapped_aln.seq_length, expected_length)
    #     self.assertEqual(evc.non_gapped_aln.query_sequence, expected_sequence)
    #     self.assertEqual(evc.non_gapped_aln_fn, os.path.join(out_dir, 'Non-Gapped_Alignment.fa'))
    #     self.assertTrue(os.path.isfile(evc.non_gapped_aln_fn))
    #     self.assertEqual(evc.method, 'EVCouplings')
    #     self.assertEqual(evc.protocol, 'standard')
    #     self.assertIsNotNone(evc.scores)
    #     self.assertIsNotNone(evc.probability)
    #     expected_ranks, expected_coverages = compute_rank_and_coverage(expected_length, evc.scores, 2, 'max')
    #     ranks_diff = evc.rankings - expected_ranks
    #     ranks_not_passing = ranks_diff > 0.0
    #     self.assertFalse(ranks_not_passing.any())
    #     coverages_diff = evc.coverages - expected_coverages
    #     coverages_not_passing = coverages_diff > 0.0
    #     self.assertFalse(coverages_not_passing.any())
    #     self.assertLessEqual(evc.time, expected_time)
    #     self.assertTrue(os.path.isfile(os.path.join(out_dir, 'EVCouplings.npz')))
    #
    # def test_4a_calculate_scores(self):
    #     self.evaluate_calculator_scores(
    #         query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
    #         expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
    #         expected_sequence=str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq))
    #
    # def test_4b_calculate_scores(self):
    #     self.evaluate_calculator_scores(
    #         query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
    #         expected_length=self.data_set.protein_data[self.large_structure_id]['Length'],
    #         expected_sequence=str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq))


if __name__ == '__main__':
    unittest.main()