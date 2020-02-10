"""
Created on February 5, 2020

@author: Daniel Konecki
"""
import os
import unittest
import numpy as np
import pandas as pd
from time import time
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

    def test_1a_init(self):
        self.evaluate_init(query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
                           expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
                           expected_sequence=str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq))

    def test_1b_init(self):
        self.evaluate_init(query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
                           expected_length=self.data_set.protein_data[self.large_structure_id]['Length'],
                           expected_sequence=str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq))

    def evaluate_convert_alignment(self, query, aln_file, out_dir):
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        wetc.convert_alignment()
        self.assertTrue(os.path.isfile(os.path.join(out_dir, 'Non-Gapped_Alignment.msf')))

    def test_2a_convert_alignment(self):
        self.evaluate_convert_alignment(query=self.small_structure_id, aln_file=self.small_fa_fn,
                                        out_dir=self.out_small_dir)

    def test_2b_convert_alignment(self):
        self.evaluate_convert_alignment(query=self.large_structure_id, aln_file=self.large_fa_fn,
                                        out_dir=self.out_large_dir)

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

    def test_3a_import_rank_scores(self):
        self.evaluate_import_rank_scores(
            query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
            expected_length=self.data_set.protein_data[self.small_structure_id]['Length'])

    def test_3b_import_rank_scores(self):
        self.evaluate_import_rank_scores(
            query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
            expected_length=self.data_set.protein_data[self.large_structure_id]['Length'])

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

    def test_4a_import_entropy_rank_scores(self):
        self.evaluate_import_entropy_rank_scores(
            query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
            expected_length=self.data_set.protein_data[self.small_structure_id]['Length'])

    def test_4b_import_entropy_rank_scores(self):
        self.evaluate_import_entropy_rank_scores(
            query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
            expected_length=self.data_set.protein_data[self.large_structure_id]['Length'])

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

    def test_5a_import_distance_matrices(self):
        self.evaluate_import_distance_matrices(
            query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir)

    def test_5b_import_distance_matrices(self):
        self.evaluate_import_distance_matrices(
            query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir)

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

    def test_6a_import_phylogenetic_tree(self):
        self.evaluate_import_phylogenetic_tree(query=self.small_structure_id, aln_file=self.small_fa_fn,
                                               out_dir=self.out_small_dir)

    def test_6b_import_phylogenetic_tree(self):
        self.evaluate_import_phylogenetic_tree(query=self.large_structure_id, aln_file=self.large_fa_fn,
                                               out_dir=self.out_large_dir)

    def evaluate_import_assignments(self, query, aln_file, out_dir):
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        rand_state = np.random.RandomState(1234567890)
        expected_aln_dists = rand_state.rand(wetc.non_gapped_aln.size, wetc.non_gapped_aln.size)
        expected_aln_dists[np.tril_indices(wetc.non_gapped_aln.size)] = 0.0
        expected_aln_dists += expected_aln_dists.T
        dm = convert_array_to_distance_matrix(expected_aln_dists, wetc.non_gapped_aln.seq_order)
        pg_tree = PhylogeneticTree()
        pg_tree.construct_tree(dm)
        pg_tree.write_out_tree(os.path.join(out_dir, 'etc_out.nhx'))
        expected_rank_group_assignments = pg_tree.assign_group_rank()

        with open(os.path.join(out_dir, 'etc_out.group.tsv'), 'w') as handle:
            handle.write('% Group definitions:\n')
            handle.write('Rank\t' + '\t'.join(wetc.non_gapped_aln.seq_order) + '\n')
            for r in expected_rank_group_assignments:
                curr_groups = {}
                for g in expected_rank_group_assignments[r]:
                    for t in expected_rank_group_assignments[r][g]['terminals']:
                        curr_groups[t] = g
                handle.write(str(r) + '\t' + '\t'.join([str(curr_groups[wetc.non_gapped_aln.seq_order[i]])
                                                        for i in range(wetc.non_gapped_aln.size)]) + '\n')
            handle.write('% Group root:\n')
            handle.write('Rank\tGroup\tRoot_Node\n')
            for r in range(1, wetc.non_gapped_aln.size + 2):
                for g in range(1, wetc.non_gapped_aln.size + 2):
                    curr_line = str(r) + '\t' + str(g) + '\t'
                    if (r in expected_rank_group_assignments) and (g in expected_rank_group_assignments[r]):
                        node_name = expected_rank_group_assignments[r][g]['node'].name
                        if len(expected_rank_group_assignments[r][g]['terminals']) == 1:
                            curr_num = -1 * (wetc.non_gapped_aln.seq_order.index(node_name) + 1)
                        else:
                            curr_num = int(node_name.strip('Inner'))
                    else:
                        curr_num = 0
                    curr_line += '{}\n'.format(curr_num)
                    handle.write(curr_line)
        wetc.import_assignments()
        self.assertEqual(set(wetc.rank_group_assignments.keys()), set(expected_rank_group_assignments.keys()))
        for r in sorted(wetc.rank_group_assignments.keys()):
            self.assertEqual(set(wetc.rank_group_assignments[r].keys()), set(expected_rank_group_assignments[r].keys()))
            for g in sorted(wetc.rank_group_assignments[r].keys()):
                self.assertEqual(set(wetc.rank_group_assignments[r][g].keys()),
                                 set(expected_rank_group_assignments[r][g].keys()))
                self.assertEqual(wetc.rank_group_assignments[r][g]['node'].name,
                                 expected_rank_group_assignments[r][g]['node'].name)
                self.assertEqual(set(wetc.rank_group_assignments[r][g]['terminals']),
                                 set(expected_rank_group_assignments[r][g]['terminals']))
                if wetc.rank_group_assignments[r][g]['descendants'] is None:
                    self.assertIsNone(expected_rank_group_assignments[r][g]['descendants'])
                else:
                    self.assertEqual(set([x.name for x in wetc.rank_group_assignments[r][g]['descendants']]),
                                     set([x.name for x in expected_rank_group_assignments[r][g]['descendants']]))

    def test_7a_import_phylogenetic_tree(self):
        self.evaluate_import_assignments(query=self.small_structure_id, aln_file=self.small_fa_fn,
                                         out_dir=self.out_small_dir)

    def test_7b_import_phylogenetic_tree(self):
        self.evaluate_import_assignments(query=self.large_structure_id, aln_file=self.large_fa_fn,
                                         out_dir=self.out_large_dir)

    def evaluate_import_intermediate_covariance_scores(self, query, aln_file, out_dir):
        expected_path = os.path.join(out_dir, 'etc_out.all_rank_and_group_mip.tsv')
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        wetc2 = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        rand_state = np.random.RandomState(1234567890)
        expected_aln_dists = rand_state.rand(wetc.non_gapped_aln.size, wetc.non_gapped_aln.size)
        expected_aln_dists[np.tril_indices(wetc.non_gapped_aln.size)] = 0.0
        expected_aln_dists += expected_aln_dists.T
        dm = convert_array_to_distance_matrix(expected_aln_dists, wetc.non_gapped_aln.seq_order)
        pg_tree = PhylogeneticTree()
        pg_tree.construct_tree(dm)
        wetc.tree = pg_tree
        wetc2.tree = pg_tree
        wetc.rank_group_assignments = pg_tree.assign_group_rank()
        wetc2.rank_group_assignments = pg_tree.assign_group_rank()
        data_for_file = {'i': [], 'j': [], 'rank': [], 'group': [], 'group_MI': [], 'E[MIi]': [], 'E[MIj]': [],
                         'E[MI]': [], 'group_MIp': []}
        indices = np.triu_indices(wetc.non_gapped_aln.seq_length, k=1)
        expected_scores = {}
        for r in sorted(wetc.rank_group_assignments.keys()):
            expected_scores[r] = {}
            for g in sorted(wetc.rank_group_assignments[r].keys()):
                expected_scores[r][g] = {}
                data_for_file['i'] += list(indices[0] + 1)
                data_for_file['j'] += list(indices[1] + 1)
                data_for_file['rank'] += [r] * len(indices[0])
                data_for_file['group'] += [g] * len(indices[0])
                for data in ['group_MI', 'E[MIi]', 'E[MIj]', 'E[MI]', 'group_MIp']:
                    curr_data = rand_state.rand(wetc.non_gapped_aln.size, wetc.non_gapped_aln.size)
                    curr_data[np.tril_indices(wetc.non_gapped_aln.size)] = 0.0
                    curr_data += curr_data.T
                    expected_scores[r][g][data] = curr_data
                    data_for_file[data] += list(curr_data[indices])
        pd.DataFrame(data_for_file).to_csv(expected_path, sep='\t', header=True, index=False,
                                           columns=['i', 'j', 'rank', 'group', 'group_MI', 'E[MIi]', 'E[MIj]', 'E[MI]',
                                                    'group_MIp'])
        mi_arrays1, amii_arrays1, amij_arrays1, ami_arrays1, mip_arrays1 = wetc.import_intermediate_covariance_scores()
        os.remove(expected_path)
        mi_arrays2, amii_arrays2, amij_arrays2, ami_arrays2, mip_arrays2 = wetc2.import_intermediate_covariance_scores()
        for r in sorted(wetc.rank_group_assignments.keys()):
            for g in sorted(wetc.rank_group_assignments[r].keys()):
                diff_mi1 = mi_arrays1 - expected_scores[r][g]['group_MI']
                not_passing_mi1 = diff_mi1 > 1E-15
                self.assertFalse(not_passing_mi1.any())
                diff_mi2 = mi_arrays2 - expected_scores[r][g]['group_MI']
                not_passing_mi2 = diff_mi2 > 1E-15
                self.assertFalse(not_passing_mi2.any())
                diff_amii1 = amii_arrays1 - expected_scores[r][g]['E[MIi]']
                not_passing_amii1 = diff_amii1 > 1E-15
                self.assertFalse(not_passing_amii1.any())
                diff_amii2 = amii_arrays2 - expected_scores[r][g]['E[MIi]']
                not_passing_amii2 = diff_amii2 > 1E-15
                self.assertFalse(not_passing_amii2.any())
                diff_amij1 = amij_arrays1 - expected_scores[r][g]['E[MIj]']
                not_passing_amij1 = diff_amij1 > 1E-15
                self.assertFalse(not_passing_amij1.any())
                diff_amij2 = amij_arrays2 - expected_scores[r][g]['E[MIj]']
                not_passing_amij2 = diff_amij2 > 1E-15
                self.assertFalse(not_passing_amij2.any())
                diff_ami1 = ami_arrays1 - expected_scores[r][g]['E[MI]']
                not_passing_ami1 = diff_ami1 > 1E-15
                self.assertFalse(not_passing_ami1.any())
                diff_ami2 = ami_arrays2 - expected_scores[r][g]['E[MI]']
                not_passing_ami2 = diff_ami2 > 1E-15
                self.assertFalse(not_passing_ami2.any())
                diff_mip1 = mip_arrays1 - expected_scores[r][g]['group_MIp']
                not_passing_mip1 = diff_mip1 > 1E-15
                self.assertFalse(not_passing_mip1.any())
                diff_mip2 = mip_arrays2 - expected_scores[r][g]['group_MIp']
                not_passing_mip2 = diff_mip2 > 1E-15
                self.assertFalse(not_passing_mip2.any())
                self.assertTrue(os.path.isfile(os.path.join(out_dir,
                                                            'R{}G{}_pair_group_WETC_MI_score.npz'.format(r, g))))
                self.assertTrue(os.path.isfile(os.path.join(out_dir,
                                                            'R{}G{}_pair_group_WETC_AMIi_score.npz'.format(r, g))))
                self.assertTrue(os.path.isfile(os.path.join(out_dir,
                                                            'R{}G{}_pair_group_WETC_AMIj_score.npz'.format(r, g))))
                self.assertTrue(os.path.isfile(os.path.join(out_dir,
                                                            'R{}G{}_pair_group_WETC_AMI_score.npz'.format(r, g))))
                self.assertTrue(os.path.isfile(os.path.join(out_dir,
                                                            'R{}G{}_pair_group_WETC_MIp_score.npz'.format(r, g))))

    def test_8a_import_intermediate_covariance_scores(self):
        self.evaluate_import_intermediate_covariance_scores(query=self.small_structure_id, aln_file=self.small_fa_fn,
                                                            out_dir=self.out_small_dir)

    def test_8b_import_intermediate_covariance_scores(self):
        self.evaluate_import_intermediate_covariance_scores(query=self.large_structure_id, aln_file=self.large_fa_fn,
                                                            out_dir=self.out_large_dir)

    def generate_covariance_scores(self, wetc, out_dir):
        rand_state = np.random.RandomState(1234567890)
        indices = np.triu_indices(n=wetc.non_gapped_aln.seq_length, k=1)
        expected_scores = rand_state.rand(wetc.non_gapped_aln.seq_length, wetc.non_gapped_aln.seq_length)
        expected_scores[np.tril_indices(wetc.non_gapped_aln.seq_length)] = 0.0
        expected_scores += expected_scores.T
        expected_ranks, expected_coverage = compute_rank_and_coverage(wetc.non_gapped_aln.seq_length, expected_scores,
                                                                      2, 'max')
        expected_interface = rand_state.randint(low=0, high=1, size=(wetc.non_gapped_aln.seq_length,
                                                                     wetc.non_gapped_aln.seq_length))
        expected_contacts = rand_state.randint(low=0, high=1, size=(wetc.non_gapped_aln.seq_length,
                                                                    wetc.non_gapped_aln.seq_length))
        expected_number = rand_state.randint(low=1, high=wetc.non_gapped_aln.seq_length,
                                             size=(wetc.non_gapped_aln.seq_length, wetc.non_gapped_aln.seq_length))
        expected_ave_contact = rand_state.rand(wetc.non_gapped_aln.seq_length, wetc.non_gapped_aln.seq_length)
        with open(os.path.join(out_dir, 'etc_out.tree_mip_sorted'), 'w') as handle:
            handle.write('%       Sort Res i(AA) Res j(AA)      sorted   cvg(sort)  interface      contact      number'
                         'AveContact\n')
            for x in range(len(indices[0])):
                i = indices[0][x]
                j = indices[1][x]
                handle.write('        {} {} {} {} {}      {}   {}  {}      {}      {}  {}\n'.format(
                    expected_ranks[i, j], i + 1, wetc.non_gapped_aln.query_sequence[i], j + 1,
                    wetc.non_gapped_aln.query_sequence[j], expected_scores[i, j], expected_coverage[i, j],
                    expected_interface[i, j], expected_contacts[i, j], expected_number[i, j],
                    expected_ave_contact[i, j]))
        return (expected_scores, expected_ranks, expected_coverage, expected_contacts, expected_interface,
                expected_number, expected_ave_contact, indices)

    def evaluate_import_covariance_scores(self, query, aln_file, out_dir):
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        e_scores, e_ranks, e_coverage, _, _, _, _, indices = self.generate_covariance_scores(wetc=wetc, out_dir=out_dir)
        wetc.import_covariance_scores()
        diff_scores = wetc.scores[indices] - e_scores[indices]
        not_passing_scores = diff_scores > 1E-15
        self.assertFalse(not_passing_scores.any())
        diff_coverage = wetc.coverages[indices] - e_coverage[indices]
        not_passing_coverage = diff_coverage > 1E-15
        self.assertFalse(not_passing_coverage.any())
        diff_ranking = wetc.rankings[indices] - e_ranks[indices]
        not_passing_ranking = diff_ranking > 1E-15
        self.assertFalse(not_passing_ranking.any())

    def test_9a_import_covariance_scores(self):
        self.evaluate_import_covariance_scores(query=self.small_structure_id, aln_file=self.small_fa_fn,
                                               out_dir=self.out_small_dir)

    def test_9b_import_covariance_scores(self):
        self.evaluate_import_covariance_scores(query=self.large_structure_id, aln_file=self.large_fa_fn,
                                               out_dir=self.out_large_dir)

    def generate_importance_scores(self, wetc, method, out_dir):
        rand_state = np.random.RandomState(1234567890)
        if method == 'intET':
            expected_scores = rand_state.randint(low=1, high=wetc.non_gapped_aln.size,
                                                 size=wetc.non_gapped_aln.seq_length)
        elif method == 'rvET':
            expected_scores = rand_state.rand(wetc.non_gapped_aln.seq_length)
        else:
            raise ValueError('Bad method provided.')
        expected_ranks, expected_coverage = compute_rank_and_coverage(wetc.non_gapped_aln.seq_length, expected_scores,
                                                                      1, 'min')
        expected_variability = rand_state.randint(low=1, high=23, size=wetc.non_gapped_aln.seq_length)
        with open(os.path.join(out_dir, 'etc_out.ranks'), 'w') as handle:
            handle.write('% Note: in this file % is a comment sign.\n% etc version April2019.\n% File produced on Thu '
                         'Feb  6 16:02:44 2020\n% The command line was: '
                         '/home/daniel/Documents/git/EvolutionaryTrace/src/wetc -p '
                         '/home/daniel/Documents/New_ETMIP/Test/135l/Non-Gapped_Alignment.msf -x 135l -o etc_out_ET-MIp'
                         ' -allpairs\n%\n%\n%	 RESIDUE RANKS:\n')
            if method == 'intET':
                handle.write('% alignment#  residue#      type      rank              variability           coverage\n')
            elif method == 'rvET':
                handle.write('% alignment#  residue#      type      rank              variability           rho     '
                             'coverage\n')
            else:
                raise ValueError('Bad method specified')
            for x in range(wetc.non_gapped_aln.seq_length):
                if method == 'intET':
                    handle.write(' {}  {}      {}      {}       {}          {}        {}\n'.format(
                        x + 1, x + 1, wetc.non_gapped_aln.query_sequence[x], expected_scores[x],
                        expected_variability[x], 'X' * expected_variability[x], expected_coverage[x]))
                elif method == 'rvET':
                    handle.write(' {}  {}      {}      {}       {}          {}        {}     {}\n'.format(
                        x + 1, x + 1, wetc.non_gapped_aln.query_sequence[x], expected_scores[x],
                        expected_variability[x],
                        'X' * expected_variability[x], expected_scores[x], expected_coverage[x]))
        return expected_scores, expected_ranks, expected_coverage

    def evaluate_import_et_ranks(self, method, query, aln_file, out_dir):
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        e_scores, e_ranks, e_coverage = self.generate_importance_scores(wetc=wetc, method=method, out_dir=out_dir)
        wetc.import_et_ranks(method=method)
        diff_scores = wetc.scores - e_scores
        not_passing_scores = diff_scores > 1E-15
        self.assertFalse(not_passing_scores.any())
        diff_coverage = wetc.coverages - e_coverage
        not_passing_coverage = diff_coverage > 1E-15
        self.assertFalse(not_passing_coverage.any())
        diff_ranking = wetc.rankings - e_ranks
        not_passing_ranking = diff_ranking > 1E-15
        self.assertFalse(not_passing_ranking.any())

    def test_10a_import_et_ranks(self):
        self.evaluate_import_et_ranks(query=self.small_structure_id, aln_file=self.small_fa_fn,
                                      out_dir=self.out_small_dir, method='intET')
        self.evaluate_import_et_ranks(query=self.small_structure_id, aln_file=self.small_fa_fn,
                                      out_dir=self.out_small_dir, method='rvET')

    def test_10b_import_et_ranks(self):
        self.evaluate_import_et_ranks(query=self.large_structure_id, aln_file=self.large_fa_fn,
                                      out_dir=self.out_large_dir, method='intET')
        self.evaluate_import_et_ranks(query=self.large_structure_id, aln_file=self.large_fa_fn,
                                      out_dir=self.out_large_dir, method='rvET')

    def evaluate_import_scores(self, query, aln_file, out_dir):
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        e_scores1, e_ranks1, e_coverage1 = self.generate_importance_scores(wetc=wetc, method='intET', out_dir=out_dir)
        wetc.import_scores(method='intET')
        diff_scores = wetc.scores - e_scores1
        not_passing_scores = diff_scores > 1E-15
        self.assertFalse(not_passing_scores.any())
        diff_coverage = wetc.coverages - e_coverage1
        not_passing_coverage = diff_coverage > 1E-15
        self.assertFalse(not_passing_coverage.any())
        diff_ranking = wetc.rankings - e_ranks1
        not_passing_ranking = diff_ranking > 1E-15
        self.assertFalse(not_passing_ranking.any())
        wetc.scores = None
        wetc.coverages = None
        wetc.rankings = None
        os.remove(os.path.join(out_dir, 'etc_out.ranks'))
        e_scores2, e_ranks2, e_coverage2 = self.generate_importance_scores(wetc=wetc, method='rvET', out_dir=out_dir)
        wetc.import_scores(method='rvET')
        diff_scores = wetc.scores - e_scores2
        not_passing_scores = diff_scores > 1E-15
        self.assertFalse(not_passing_scores.any())
        diff_coverage = wetc.coverages - e_coverage2
        not_passing_coverage = diff_coverage > 1E-15
        self.assertFalse(not_passing_coverage.any())
        diff_ranking = wetc.rankings - e_ranks2
        not_passing_ranking = diff_ranking > 1E-15
        self.assertFalse(not_passing_ranking.any())
        wetc.scores = None
        wetc.coverages = None
        wetc.rankings = None
        os.remove(os.path.join(out_dir, 'etc_out.ranks'))
        e_scores3, e_ranks3, e_coverage3, _, _, _, _, indices = self.generate_covariance_scores(wetc=wetc,
                                                                                                out_dir=out_dir)
        wetc.import_scores(method='ET-MIp')
        diff_scores = wetc.scores[indices] - e_scores3[indices]
        not_passing_scores = diff_scores > 1E-15
        self.assertFalse(not_passing_scores.any())
        diff_coverage = wetc.coverages[indices] - e_coverage3[indices]
        not_passing_coverage = diff_coverage > 1E-15
        self.assertFalse(not_passing_coverage.any())
        diff_ranking = wetc.rankings[indices] - e_ranks3[indices]
        not_passing_ranking = diff_ranking > 1E-15
        self.assertFalse(not_passing_ranking.any())

    def test_11a_import_scores(self):
        self.evaluate_import_scores(
            query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir)

    def test_11b_import_scores(self):
        self.evaluate_import_scores(
            query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir)

    def evaluate_calculate_scores(self, query, aln_file, out_dir, expected_length, expected_sequence):
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        for method in ['intET', 'rvET', 'ET-MIp']:
            start = time()
            wetc.calculate_scores(delete_files=False, cores=self.max_threads)
            end = time()
            expected_time = end - start
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
            self.assertIsNotNone(wetc.scores)
            if method.endswith('ET'):
                expected_ranks, expected_coverages = compute_rank_and_coverage(expected_length, wetc.scores, 1, 'min')
            else:
                expected_ranks, expected_coverages = compute_rank_and_coverage(expected_length, wetc.scores, 2, 'max')
            ranks_diff = wetc.rankings - expected_ranks
            ranks_not_passing = ranks_diff > 0.0
            self.assertFalse(ranks_not_passing.any())
            coverages_diff = wetc.coverages - expected_coverages
            coverages_not_passing = coverages_diff > 0.0
            self.assertFalse(coverages_not_passing.any())
            self.assertLessEqual(wetc.time, expected_time)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, 'WETC.npz')))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, 'WETC.pkl')))
            wetc.scores = None
            wetc.rankings = None
            wetc.coverages = None
            os.remove(os.path.join(out_dir, 'WETC.npz'))
            os.remove(os.path.join(out_dir, 'WETC.pkl'))

    def test_12a_calculate_scores(self):
        self.evaluate_calculate_scores(
            query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
            expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
            expected_sequence=str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq))

    def test_12b_calculate_scores(self):
        self.evaluate_calculate_scores(
            query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
            expected_length=self.data_set.protein_data[self.large_structure_id]['Length'],
            expected_sequence=str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq))

    def evalaute_remove_output(self, query, aln_file, out_dir):
        wetc = ETMIPWrapper(query=query, aln_file=aln_file, out_dir=out_dir)
        file_suffixes = ['aa_freqs', 'allpair_ranks', 'allpair_ranks_sorted', 'auc', 'average_ranks_sorted',
                         'covariation_matrix', 'entro.heatmap', 'entroMI.heatmap', 'mip_sorted', 'MI_sorted', 'nhx',
                         'pairs_allpair_ranks_sorted', 'pairs_average_ranks_sorted', 'pairs_mip_sorted',
                         'pairs_tree_mip_sorted', 'pairs_MI_sorted', 'pairs_tre_mip_sorted', 'pss.nhx', 'rank_matrix',
                         'ranks', 'ranks_sorted', 'rv.heatmap', 'rvMI.heatmap', 'tree_mip_matrix', 'tree_mip_sorted',
                         'tree_mip_top40_matrix']
        for i in range(len(file_suffixes)):
            for j in range(i + 1):
                with open(os.path.join(out_dir, 'etc_out.{}'.format(file_suffixes[j])), 'a'):
                    pass
                self.assertTrue(os.path.isfile(os.path.join(out_dir, 'etc_out.{}'.format(file_suffixes[j]))))
            wetc.remove_output()
            for j in range(i + 1):
                self.assertFalse(os.path.isfile(os.path.join(out_dir, 'etc_out.{}'.format(file_suffixes[j]))))

    def test_13a_remove_output(self):
        self.evaluate_remove_output(
            query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir)

    def test_13b_remove_output(self):
        self.evalaute_remove_output(
            query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir)


if __name__ == '__main__':
    unittest.main()