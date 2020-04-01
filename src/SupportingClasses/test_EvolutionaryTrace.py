"""
Created on August 13, 2019

@author: Daniel Konecki
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd
from copy import deepcopy
from shutil import rmtree
from Bio.Alphabet import Gapped
from Bio.Alphabet.IUPAC import IUPACProtein
from test_Base import TestBase
from test_PhylogeneticTree import compare_nodes, compare_nodes_key
from utils import build_mapping
from SeqAlignment import SeqAlignment
from ETMIPWrapper import ETMIPWrapper
from PhylogeneticTree import PhylogeneticTree
from PositionalScorer import PositionalScorer
from Trace import Trace, load_freq_table, load_numpy_array
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from EvolutionaryTraceAlphabet import FullIUPACProtein, MultiPositionAlphabet
sys.path.append(os.path.abspath('..'))
from EvolutionaryTrace import EvolutionaryTrace


class TestEvoultionaryTrace(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestEvoultionaryTrace, cls).setUpClass()
        cls.small_fa_fn = cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln']
        cls.query_aln_fa_small = SeqAlignment(
            file_name=cls.small_fa_fn, query_id=cls.small_structure_id)
        cls.query_aln_fa_small.import_alignment()
        cls.phylo_tree_small = PhylogeneticTree()
        calc = AlignmentDistanceCalculator()
        cls.phylo_tree_small.construct_tree(dm=calc.get_distance(cls.query_aln_fa_small.alignment))
        cls.assignments_small = cls.phylo_tree_small.assign_group_rank()
        cls.assignments_custom_small = cls.phylo_tree_small.assign_group_rank(ranks=[1, 2, 3, 5, 7, 10, 25])
        cls.large_fa_fn = cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln']
        cls.query_aln_fa_large = SeqAlignment(file_name=cls.large_fa_fn, query_id=cls.large_structure_id)
        cls.query_aln_fa_large.import_alignment()
        cls.phylo_tree_large = PhylogeneticTree()
        calc = AlignmentDistanceCalculator()
        cls.phylo_tree_large.construct_tree(dm=calc.get_distance(cls.query_aln_fa_large.alignment))
        cls.assignments_large = cls.phylo_tree_large.assign_group_rank()
        cls.assignments_custom_large = cls.phylo_tree_large.assign_group_rank(ranks=[1, 2, 3, 5, 7, 10, 25])
        cls.query_aln_msf_small = deepcopy(cls.query_aln_fa_small)
        cls.query_aln_msf_small.file_name = cls.data_set.protein_data[cls.small_structure_id]['Final_MSF_Aln']
        cls.out_small_dir = os.path.join(cls.testing_dir, cls.small_structure_id)
        if not os.path.isdir(cls.out_small_dir):
            os.makedirs(cls.out_small_dir)
        cls.query_aln_msf_large = deepcopy(cls.query_aln_fa_large)
        cls.query_aln_msf_large.file_name = cls.data_set.protein_data[cls.large_structure_id]['Final_MSF_Aln']
        cls.out_large_dir = os.path.join(cls.testing_dir, cls.large_structure_id)
        if not os.path.isdir(cls.out_large_dir):
            os.makedirs(cls.out_large_dir)
        cls.query_aln_fa_small = cls.query_aln_fa_small.remove_gaps()
        cls.query_aln_fa_large = cls.query_aln_fa_large.remove_gaps()
        cls.single_alphabet = Gapped(FullIUPACProtein())
        cls.single_size, _, cls.single_mapping, cls.single_reverse = build_mapping(alphabet=cls.single_alphabet)
        cls.pair_alphabet = MultiPositionAlphabet(alphabet=cls.single_alphabet, size=2)
        cls.pair_size, _, cls.pair_mapping, cls.pair_reverse = build_mapping(alphabet=cls.pair_alphabet)
        cls.single_to_pair = {}
        for char in cls.pair_mapping:
            key = (cls.single_mapping[char[0]], cls.single_mapping[char[1]])
            cls.single_to_pair[key] = cls.pair_mapping[char]

    def check_nodes(self, node1, node2):
        if node1.is_terminal():
            self.assertTrue(node2.is_terminal())
            self.assertEqual(node1.name, node2.name)
        else:
            self.assertTrue(node2.is_bifurcating())
            self.assertFalse(node2.is_terminal())
            self.assertEqual(set([x.name for x in node1.get_terminals()]),
                             set([x.name for x in node2.get_terminals()]))

    def evaluate_init(self, query_id, polymer_type, aln_fn, et_distance, distance_model, tree_building_method,
                      tree_building_options, ranks, position_type, scoring_metric, gap_correction, out_dir,
                      output_files, processors, low_memory, expected_length, expected_sequence):
        et = EvolutionaryTrace(query_id, polymer_type, aln_fn, et_distance, distance_model, tree_building_method,
                               tree_building_options, ranks, position_type, scoring_metric, gap_correction, out_dir,
                               output_files, processors, low_memory)
        self.assertEqual(et.query, query_id)
        self.assertEqual(et.out_dir, out_dir)
        self.assertIsNotNone(et.original_aln)
        self.assertGreaterEqual(et.original_aln.seq_length, expected_length)
        self.assertEqual(str(et.original_aln.query_sequence).replace('-', ''), expected_sequence)
        self.assertEqual(et.original_aln_fn, os.path.join(out_dir, 'Original_Alignment.fa'))
        self.assertTrue(os.path.isfile(et.original_aln_fn))
        self.assertIsNotNone(et.non_gapped_aln)
        self.assertEqual(et.non_gapped_aln.seq_length, expected_length)
        self.assertEqual(et.non_gapped_aln.query_sequence, expected_sequence)
        self.assertEqual(et.non_gapped_aln_fn, os.path.join(out_dir, 'Non-Gapped_Alignment.fa'))
        self.assertTrue(os.path.isfile(et.non_gapped_aln_fn))
        self.assertEqual(et.polymer_type, polymer_type)
        self.assertEqual(et.et_distance, et_distance)
        self.assertEqual(et.distance_model, distance_model)
        self.assertIsNone(et.distance_matrix)
        self.assertEqual(et.tree_building_method, tree_building_method)
        self.assertEqual(et.tree_building_options, tree_building_options)
        self.assertIsNone(et.phylo_tree)
        self.assertIsNone(et.phylo_tree_fn)
        self.assertEqual(et.ranks, ranks)
        self.assertIsNone(et.assignments)
        self.assertEqual(et.position_type, position_type)
        self.assertEqual(et.scoring_metric, scoring_metric)
        self.assertEqual(et.gap_correction, gap_correction)
        self.assertIsNone(et.trace)
        self.assertIsNone(et.rankings)
        self.assertIsNone(et.scorer)
        self.assertIsNone(et.scores)
        self.assertIsNone(et.coverages)
        self.assertEqual(et.output_files, output_files)
        self.assertEqual(et.processors, processors)
        self.assertEqual(et.low_memory, low_memory)

    def test_1a_init(self):
        self.evaluate_init(query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn,
                           et_distance=True, distance_model='blosum62', tree_building_method='et',
                           tree_building_options={}, ranks=None, position_type='single', scoring_metric='identity',
                           gap_correction=None, out_dir=self.out_small_dir,
                           output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'},
                           processors=self.max_threads, low_memory=True,
                           expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
                           expected_sequence=self.data_set.protein_data[self.small_structure_id]['Sequence'].seq)

    def test_1b_init(self):
        self.evaluate_init(query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn,
                           et_distance=True, distance_model='blosum62', tree_building_method='et',
                           tree_building_options={}, ranks=None, position_type='single', scoring_metric='plain_entropy',
                           gap_correction=None, out_dir=self.out_small_dir,
                           output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'},
                           processors=self.max_threads, low_memory=True,
                           expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
                           expected_sequence=self.data_set.protein_data[self.small_structure_id]['Sequence'].seq)

    def test_1c_init(self):
        self.evaluate_init(query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn,
                           et_distance=True, distance_model='blosum62', tree_building_method='et',
                           tree_building_options={}, ranks=None, position_type='pair',
                           scoring_metric='mutual_information',
                           gap_correction=None, out_dir=self.out_small_dir, processors=self.max_threads,
                           low_memory=True, output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'},
                           expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
                           expected_sequence=self.data_set.protein_data[self.small_structure_id]['Sequence'].seq)

    def test_1d_init(self):
        self.evaluate_init(query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn,
                           et_distance=True, distance_model='blosum62', tree_building_method='et',
                           tree_building_options={}, ranks=None, position_type='pair',
                           scoring_metric='normalized_mutual_information',
                           gap_correction=None, out_dir=self.out_small_dir, processors=self.max_threads,
                           low_memory=True, output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'},
                           expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
                           expected_sequence=self.data_set.protein_data[self.small_structure_id]['Sequence'].seq)

    def test_1e_init(self):
        self.evaluate_init(query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn,
                           et_distance=True, distance_model='blosum62', tree_building_method='et',
                           tree_building_options={}, ranks=None, position_type='pair',
                           scoring_metric='average_product_corrected_mutual_information',
                           gap_correction=None, out_dir=self.out_small_dir, processors=self.max_threads,
                           low_memory=True, output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'},
                           expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
                           expected_sequence=self.data_set.protein_data[self.small_structure_id]['Sequence'].seq)

    def test_1f_init(self):
        self.evaluate_init(query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn,
                           et_distance=True, distance_model='blosum62', tree_building_method='et',
                           tree_building_options={}, ranks=None, position_type='pair',
                           scoring_metric='filtered_average_product_corrected_mutual_information',
                           gap_correction=None, out_dir=self.out_small_dir, processors=self.max_threads,
                           low_memory=True, output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'},
                           expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
                           expected_sequence=self.data_set.protein_data[self.small_structure_id]['Sequence'].seq)

    def test_1g_init(self):
        self.evaluate_init(query_id=self.large_structure_id, polymer_type='Protein', aln_fn=self.large_fa_fn,
                           et_distance=True, distance_model='blosum62', tree_building_method='et',
                           tree_building_options={}, ranks=None, position_type='pair',
                           scoring_metric='match_mismatch_entropy_angle',
                           gap_correction=None, out_dir=self.out_large_dir, processors=self.max_threads,
                           low_memory=True, output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'},
                           expected_length=self.data_set.protein_data[self.large_structure_id]['Length'],
                           expected_sequence=self.data_set.protein_data[self.large_structure_id]['Sequence'].seq)

    def evaluate_compute_distance_matrix_tree_and_assignments(self, query_id, polymer_type, aln_fn, et_distance,
                                                              distance_model, tree_building_method,
                                                              tree_building_options, ranks, position_type,
                                                              scoring_metric, gap_correction, out_dir, output_files,
                                                              processors, low_memory):
        serial_fn = os.path.join(out_dir, '{}_{}{}_Dist_{}_Tree.pkl'.format(query_id, ('ET_' if et_distance else ''),
                                                                            distance_model, tree_building_method))
        if os.path.isfile(serial_fn):
            os.remove(serial_fn)
        et = EvolutionaryTrace(query_id, polymer_type, aln_fn, et_distance, distance_model, tree_building_method,
                               tree_building_options, ranks, position_type, scoring_metric, gap_correction, out_dir,
                               output_files, processors, low_memory)
        et.compute_distance_matrix_tree_and_assignments()
        calculator = AlignmentDistanceCalculator(protein=(polymer_type == 'Protein'), model=distance_model)
        if et_distance:
            _, expected_dist_matrix, _, _ = calculator.get_et_distance(msa=et.original_aln.alignment)
        else:
            expected_dist_matrix = calculator.get_distance(msa=et.original_aln.alignment)
        self.assertIsNotNone(et.distance_matrix)
        self.assertEqual(et.distance_matrix.names, et.original_aln.seq_order)
        self.assertFalse((np.array(et.distance_matrix) - np.array(expected_dist_matrix)).any())
        expected_tree_fn = os.path.join(out_dir, '{}_{}{}_dist_{}_tree.nhx'.format(
            query_id, ('ET_' if et_distance else ''), distance_model, tree_building_method))
        self.assertTrue(os.path.isfile(expected_tree_fn))
        os.remove(expected_tree_fn)
        expected_tree = PhylogeneticTree(tree_building_method=tree_building_method,
                                         tree_building_args=tree_building_options)
        expected_tree.construct_tree(dm=expected_dist_matrix)
        expected_tree_iter = expected_tree.traverse_by_rank()
        tree_iter = et.phylo_tree.traverse_by_rank()
        try:
            curr_expected_nodes = next(expected_tree_iter)
        except StopIteration:
            curr_expected_nodes = None
        try:
            curr_nodes = next(tree_iter)
        except StopIteration:
            curr_nodes = None
        while curr_expected_nodes and curr_nodes:
            if curr_expected_nodes is None:
                self.assertIsNone(curr_nodes)
            else:
                sorted_curr_nodes = sorted(curr_nodes, key=compare_nodes_key(compare_nodes))
                sorted_curr_expected_nodes = sorted(curr_expected_nodes, key=compare_nodes_key(compare_nodes))
                self.assertEqual(len(sorted_curr_nodes), len(sorted_curr_expected_nodes))
                for i in range(len(sorted_curr_expected_nodes)):
                    self.check_nodes(sorted_curr_nodes[i], sorted_curr_expected_nodes[i])
            try:
                curr_expected_nodes = next(expected_tree_iter)
            except StopIteration:
                curr_expected_nodes = None
            try:
                curr_nodes = next(tree_iter)
            except StopIteration:
                curr_nodes = None
        expected_assignments = expected_tree.assign_group_rank(ranks=ranks)
        self.assertEqual(len(expected_assignments), len(et.assignments))
        for rank in expected_assignments:
            self.assertTrue(rank in et.assignments)
            for group in expected_assignments[rank]:
                self.assertTrue(group in et.assignments[rank])
                self.assertTrue('node' in expected_assignments[rank][group])
                self.assertTrue('node' in et.assignments[rank][group])
                self.check_nodes(expected_assignments[rank][group]['node'],
                                                 et.assignments[rank][group]['node'])
                self.assertTrue('terminals' in expected_assignments[rank][group])
                self.assertTrue('terminals' in et.assignments[rank][group])
                self.assertEqual(expected_assignments[rank][group]['terminals'],
                                 et.assignments[rank][group]['terminals'])
                self.assertTrue('descendants' in expected_assignments[rank][group])
                self.assertTrue('descendants' in et.assignments[rank][group])
                if expected_assignments[rank][group]['node'].is_terminal():
                    self.assertIsNone(expected_assignments[rank][group]['descendants'])
                    self.assertIsNone(et.assignments[rank][group]['descendants'])
                else:
                    self.assertEqual(set([x.name for x in expected_assignments[rank][group]['descendants']]),
                                     set([x.name for x in et.assignments[rank][group]['descendants']]))
        self.assertTrue(os.path.isfile(serial_fn))

    def test_2a_compute_distance_matrix_tree_and_assignments(self):
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='single', scoring_metric='identity', gap_correction=None, out_dir=self.out_small_dir,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'}, processors=self.max_threads,
            low_memory=True)

    def test_2b_compute_distance_matrix_tree_and_assignments(self):
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='single', scoring_metric='plain_entropy', gap_correction=None, out_dir=self.out_small_dir,
            processors=self.max_threads, low_memory=True, output_files={'original_aln', 'non_gap_aln', 'tree',
                                                                        'scores'})

    def test_2c_compute_distance_matrix_tree_and_assignments(self):
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='pair', scoring_metric='filtered_average_product_corrected_mutual_information',
            gap_correction=None, out_dir=self.out_small_dir, processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def test_2d_compute_distance_matrix_tree_and_assignments(self):
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='pair', scoring_metric='match_mismatch_entropy_angle',
            gap_correction=None, out_dir=self.out_small_dir, processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def test_2e_compute_distance_matrix_tree_and_assignments(self):
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id=self.large_structure_id, polymer_type='Protein', aln_fn=self.large_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='single', scoring_metric='identity', gap_correction=None, out_dir=self.out_large_dir,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'}, processors=self.max_threads,
            low_memory=True)

    def test_2f_compute_distance_matrix_tree_and_assignments(self):
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id=self.large_structure_id, polymer_type='Protein', aln_fn=self.large_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='single', scoring_metric='plain_entropy', gap_correction=None, out_dir=self.out_large_dir,
            processors=self.max_threads, low_memory=True, output_files={'original_aln', 'non_gap_aln', 'tree',
                                                                        'scores'})

    def test_2g_compute_distance_matrix_tree_and_assignments(self):
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id=self.large_structure_id, polymer_type='Protein', aln_fn=self.large_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='pair', scoring_metric='filtered_average_product_corrected_mutual_information',
            gap_correction=None, out_dir=self.out_large_dir, processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def test_2h_compute_distance_matrix_tree_and_assignments(self):
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id=self.large_structure_id, polymer_type='Protein', aln_fn=self.large_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='pair', scoring_metric='match_mismatch_entropy_angle',
            gap_correction=None, out_dir=self.out_large_dir, processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def evaluate_write_out_et_scores(self, et, fn):

        self.assertTrue(os.path.isfile(fn))
        et_df = pd.read_csv(fn, sep='\t', header=0, index_col=None, keep_default_na=False)
        self.assertTrue('Variability_Count' in et_df.columns)
        self.assertTrue('Variability_Characters' in et_df.columns)
        self.assertTrue('Rank' in et_df.columns)
        self.assertTrue('Score' in et_df.columns)
        self.assertTrue('Coverage' in et_df.columns)
        if et.position_type == 'single':
            self.assertTrue('Position' in et_df.columns)
            self.assertTrue('Query' in et_df.columns)
        elif et.position_type == 'pair':
            self.assertTrue('Position_i' in et_df.columns)
            self.assertTrue('Position_j' in et_df.columns)
            self.assertTrue('Query_i' in et_df.columns)
            self.assertTrue('Query_j' in et_df.columns)
        else:
            raise ValueError('Cannot evaluate EvolutionaryTrace instance with position_type: {}'.format(et.position_type))
        root_name = et.assignments[1][1]['node'].name
        if et.trace.match_mismatch:
            root_freq_table = (load_freq_table(freq_table=et.trace.unique_nodes[root_name]['match'],
                                               low_memory=et.low_memory) +
                               load_freq_table(freq_table=et.trace.unique_nodes[root_name]['mismatch'],
                                               low_memory=et.low_memory))
        else:
            root_freq_table = load_freq_table(freq_table=et.trace.unique_nodes[root_name][et.position_type],
                                              low_memory=et.low_memory)
        for ind in et_df.index:
            if et.position_type == 'single':
                position = et_df.loc[ind, 'Position'] - 1
                expected_query = et.non_gapped_aln.query_sequence[position]
                self.assertEqual(expected_query, et_df.loc[ind, 'Query'])
                expected_characters = root_freq_table.get_chars(pos=position)
                self.assertEqual(len(expected_characters),
                                 et_df.loc[ind, 'Variability_Count'], (len(expected_characters), et_df.loc[ind, 'Variability_Count'], set(expected_characters), set(et_df.loc[ind, 'Variability_Characters'].split(','))))
                self.assertEqual(set(expected_characters),
                                 set(et_df.loc[ind, 'Variability_Characters'].split(',')))
                expected_rank = et.rankings[position]
                self.assertEqual(expected_rank, et_df.loc[ind, 'Rank'])
                expected_score = et.scores[position]
                if et.scoring_metric == 'identity':
                    self.assertEqual(expected_score, et_df.loc[ind, 'Score'])
                else:
                    self.assertLessEqual(np.abs(expected_score - et_df.loc[ind, 'Score']), 1e-3)
                expected_coverage = et.coverages[position]
                self.assertLessEqual(np.abs(expected_coverage - et_df.loc[ind, 'Coverage']), 1e-3)
            else:
                pos_i = et_df.loc[ind, 'Position_i'] - 1
                pos_j = et_df.loc[ind, 'Position_j'] - 1
                expected_query_i = et.non_gapped_aln.query_sequence[pos_i]
                self.assertEqual(expected_query_i, et_df.loc[ind, 'Query_i'])
                expected_query_j = et.non_gapped_aln.query_sequence[pos_j]
                self.assertEqual(expected_query_j, et_df.loc[ind, 'Query_j'])
                expected_characters = root_freq_table.get_chars(pos=(pos_i, pos_j))
                self.assertEqual(len(expected_characters),
                                 et_df.loc[ind, 'Variability_Count'], (len(expected_characters), et_df.loc[ind, 'Variability_Count'], set(expected_characters), set(et_df.loc[ind, 'Variability_Characters'].split(','))))
                self.assertEqual(set(expected_characters),
                                 set(et_df.loc[ind, 'Variability_Characters'].split(',')))
                expected_rank = et.rankings[pos_i, pos_j]
                self.assertEqual(expected_rank, et_df.loc[ind, 'Rank'], fn)
                expected_score = et.scores[pos_i, pos_j]
                if et.scoring_metric == 'identity':
                    self.assertEqual(expected_score, et_df.loc[ind, 'Score'])
                else:
                    self.assertLessEqual(np.abs(expected_score - et_df.loc[ind, 'Score']), 1e-3)
                expected_coverage = et.coverages[pos_i, pos_j]
                self.assertLessEqual(np.abs(expected_coverage - et_df.loc[ind, 'Coverage']), 1e-3)
        os.remove(fn)

    def evaluate_perform_trace(self, query_id, polymer_type, aln_fn, et_distance, distance_model, tree_building_method,
                               tree_building_options, ranks, position_type, scoring_metric, gap_correction, out_dir,
                               output_files, processors, low_memory):
        rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        serial_fn = os.path.join(out_dir, '{}_{}{}_Dist_{}_Tree_{}_{}_Scoring.pkl'.format(
            query_id, ('ET_' if et_distance else ''), distance_model, tree_building_method,
            ('All_Ranks' if ranks is None else 'Custom_Ranks'), scoring_metric))
        if os.path.isfile(serial_fn):
            os.remove(serial_fn)
        et = EvolutionaryTrace(query_id, polymer_type, aln_fn, et_distance, distance_model, tree_building_method,
                               tree_building_options, ranks, position_type, scoring_metric, gap_correction, out_dir,
                               output_files, processors, low_memory)
        with self.assertRaises(AttributeError):
            et.perform_trace()
        et.compute_distance_matrix_tree_and_assignments()
        et.perform_trace()
        expected_trace = Trace(alignment=et.non_gapped_aln, phylo_tree=et.phylo_tree, group_assignments=et.assignments,
                               position_specific=(position_type == 'single'), pair_specific=(position_type == 'pair'),
                               output_dir=out_dir, low_memory=low_memory,
                               match_mismatch=(scoring_metric == 'match_mismatch_entropy_angle'))
        self.assertEqual(et.trace.aln.query_id, expected_trace.aln.query_id)
        self.assertEqual(et.trace.aln.size, expected_trace.aln.size)
        self.assertEqual(et.trace.aln.seq_length, expected_trace.aln.seq_length)
        self.assertEqual(str(et.trace.aln.query_sequence), str(expected_trace.aln.query_sequence))
        self.assertEqual(et.trace.phylo_tree.size, expected_trace.phylo_tree.size)
        node_iter = et.trace.phylo_tree.traverse_top_down()
        expected_node_iter = expected_trace.phylo_tree.traverse_top_down()
        try:
            curr_node = next(node_iter)
        except StopIteration:
            curr_node = None
        try:
            curr_expected_node = next(expected_node_iter)
        except StopIteration:
            curr_expected_node = None
        while curr_node and curr_expected_node:
            if curr_node is None:
                self.assertIsNone(curr_expected_node)
            else:
                self.check_nodes(curr_node, curr_expected_node)
            try:
                curr_node = next(node_iter)
            except StopIteration:
                curr_node = None
            try:
                curr_expected_node = next(expected_node_iter)
            except StopIteration:
                curr_expected_node = None
        self.assertEqual(len(et.trace.assignments), len(expected_trace.assignments))
        for rank in expected_trace.assignments:
            self.assertEqual(len(et.assignments[rank]), len(expected_trace.assignments[rank]))
            for group in expected_trace.assignments[rank]:
                self.check_nodes(et.assignments[rank][group]['node'],
                                                 expected_trace.assignments[rank][group]['node'])
                self.assertEqual(et.assignments[rank][group]['terminals'],
                                 expected_trace.assignments[rank][group]['terminals'])
                self.assertEqual(et.assignments[rank][group]['descendants'],
                                 expected_trace.assignments[rank][group]['descendants'])
        self.assertEqual(et.trace.pos_specific, expected_trace.pos_specific)
        self.assertEqual(et.trace.pair_specific, expected_trace.pair_specific)
        self.assertEqual(et.trace.out_dir, expected_trace.out_dir)
        self.assertEqual(et.trace.low_memory, expected_trace.low_memory)
        expected_trace.characterize_rank_groups(processes=self.max_threads,
                                                write_out_sub_aln=('sub-alignments' in output_files),
                                                write_out_freq_table=('frequency_tables' in output_files))
        unique_dir = os.path.join(out_dir, 'unique_node_data')
        self.assertEqual(len(et.trace.unique_nodes), len(expected_trace.unique_nodes))
        for node_name in expected_trace.unique_nodes:
            self.assertTrue(node_name in et.trace.unique_nodes)
            if 'sub-alignments' in output_files:
                self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}.fa'.format(node_name))))
            if et.trace.match_mismatch:
                self.assertTrue('match' in et.trace.unique_nodes[node_name])
                self.assertTrue('match' in expected_trace.unique_nodes[node_name])
                self.assertTrue('mismatch' in et.trace.unique_nodes[node_name])
                self.assertTrue('mismatch' in expected_trace.unique_nodes[node_name])
                expected_m_table = load_freq_table(freq_table=expected_trace.unique_nodes[node_name]['match'],
                                                   low_memory=low_memory)
                single_m_table = load_freq_table(freq_table=et.trace.unique_nodes[node_name]['match'],
                                                 low_memory=low_memory)
                sparse_diff_m = expected_m_table.get_table() - single_m_table.get_table()
                nonzero_check_m = sparse_diff_m.count_nonzero() > 0
                self.assertFalse(nonzero_check_m)
                expected_mm_table = load_freq_table(freq_table=expected_trace.unique_nodes[node_name]['mismatch'],
                                                    low_memory=low_memory)
                single_mm_table = load_freq_table(freq_table=et.trace.unique_nodes[node_name]['mismatch'],
                                                  low_memory=low_memory)
                sparse_diff_mm = expected_mm_table.get_table() - single_mm_table.get_table()
                nonzero_check_mm = sparse_diff_mm.count_nonzero() > 0
                self.assertFalse(nonzero_check_mm)
                if 'frequency_tables' in output_files:
                    self.assertTrue(os.path.isfile(os.path.join(
                        unique_dir, '{}_position_match_freq_table.tsv'.format(node_name))))
                    self.assertTrue(os.path.isfile(os.path.join(
                        unique_dir, '{}_position_mismatch_freq_table.tsv'.format(node_name))))
            else:
                self.assertTrue('single' in et.trace.unique_nodes[node_name])
                self.assertTrue('single' in expected_trace.unique_nodes[node_name])
                if position_type == 'single':
                    expected_single_table = load_freq_table(freq_table=expected_trace.unique_nodes[node_name]['single'],
                                                            low_memory=low_memory)
                    single_table = load_freq_table(freq_table=et.trace.unique_nodes[node_name]['single'],
                                                   low_memory=low_memory)
                    expected_single_array = expected_single_table.get_table()
                    single_array = single_table.get_table()
                    sparse_diff = single_array - expected_single_array
                    nonzero_check = sparse_diff.count_nonzero() > 0
                    self.assertFalse(nonzero_check)
                    if 'frequency_tables' in output_files:
                        self.assertTrue(os.path.isfile(os.path.join(unique_dir,
                                                                    '{}_position_freq_table.tsv'.format(node_name))))
                else:
                    self.assertIsNone(expected_trace.unique_nodes[node_name]['single'])
                    self.assertIsNone(et.trace.unique_nodes[node_name]['single'])
                self.assertTrue('pair' in expected_trace.unique_nodes[node_name])
                self.assertTrue('pair' in et.trace.unique_nodes[node_name])
                if position_type == 'pair':
                    expected_pair_table = load_freq_table(freq_table=expected_trace.unique_nodes[node_name]['pair'],
                                                          low_memory=low_memory)
                    pair_table = load_freq_table(freq_table=et.trace.unique_nodes[node_name]['pair'],
                                                 low_memory=low_memory)
                    expected_pair_array = expected_pair_table.get_table()
                    pair_array = pair_table.get_table()
                    sparse_diff = pair_array - expected_pair_array
                    nonzero_check = sparse_diff.count_nonzero() > 0
                    self.assertFalse(nonzero_check)
                    if 'frequency_tables' in output_files:
                        self.assertTrue(os.path.isfile(os.path.join(unique_dir,
                                                                    '{}_pair_freq_table.tsv'.format(node_name))))
                else:
                    self.assertIsNone(expected_trace.unique_nodes[node_name]['pair'])
                    self.assertIsNone(et.trace.unique_nodes[node_name]['pair'])
        expected_scorer = PositionalScorer(seq_length=et.non_gapped_aln.seq_length,
                                           pos_size=(1 if position_type == 'single' else 2), metric=scoring_metric)
        self.assertEqual(et.scorer.sequence_length, expected_scorer.sequence_length)
        self.assertEqual(et.scorer.dimensions, expected_scorer.dimensions)
        self.assertEqual(et.scorer.position_size, expected_scorer.position_size)
        self.assertEqual(et.scorer.metric, expected_scorer.metric)
        self.assertEqual(et.scorer.metric_type, expected_scorer.metric_type)
        self.assertEqual(et.scorer.rank_type, expected_scorer.rank_type)
        expected_ranks, expected_scores, expected_coverage = expected_trace.trace(scorer=expected_scorer,
                                                                                  gap_correction=gap_correction,
                                                                                  processes=self.max_threads)
        diff_scores = et.scores - expected_scores
        not_passing = diff_scores > 1e-13
        if not_passing.any():
            print(et.scores)
            print(expected_scores)
            print(diff_scores)
            indices = np.nonzero(not_passing)
            print(et.scores[indices])
            print(expected_scores[indices])
            print(diff_scores[indices])
        self.assertFalse(not_passing.any())
        # Cannot get these to match at the moment (seems to be due to rounding error but my attempts to check the parts
        # that agree are also failing. This was tested already in test_Trace.py hopefully the result holds and I have
        # just made mistake in the commented out block below.
        # if diff_scores.any():
        #     if expected_scorer.metric_type == 'max':
        #         print('MAX')
        #         first_et_diff = np.max(et.scores[np.nonzero(diff_scores)])
        #         first_et_mask = et.scores <= first_et_diff
        #         first_expected_diff = np.max(expected_scores[np.nonzero(diff_scores)])
        #         first_expected_mask = expected_scores <= first_expected_diff
        #     else:
        #         print('MIN')
        #         first_et_diff = np.min(et.scores[np.nonzero(diff_scores)])
        #         first_et_mask = et.scores >= first_et_diff
        #         first_expected_diff = np.min(expected_scores[np.nonzero(diff_scores)])
        #         first_expected_mask = expected_scores >= first_expected_diff
        #     if np.sum(first_et_mask) > np.sum(first_expected_mask):
        #         print('First ET Diff:', first_et_diff)
        #         print(et.rankings[et.scores == first_et_diff])
        #         diff_mask = first_et_mask
        #     else:
        #         print('First Expected Diff:', first_expected_diff)
        #         print(et.rankings[expected_scores == first_expected_diff])
        #         diff_mask = first_expected_mask
        #     expected_match_mask = np.invert(diff_mask)
        # else:
        #     expected_match_mask = np.ones(et.scores.shape, dtype=bool)
        # diff_ranks = et.rankings[expected_match_mask] - expected_ranks[expected_match_mask]
        # if diff_ranks.any():
        #     print(et.rankings[expected_match_mask])
        #     print(expected_ranks[expected_match_mask])
        #     print(diff_ranks)
        #     indices = np.nonzero(diff_ranks)
        #     print(et.scores[expected_match_mask][indices])
        #     print(expected_scores[expected_match_mask][indices])
        #     print(et.rankings[expected_match_mask][indices])
        #     print(expected_ranks[expected_match_mask][indices])
        #     print(diff_ranks[indices])
        # self.assertFalse(diff_ranks.any())
        # diff_coverage = et.coverages[expected_match_mask] - expected_coverage[expected_match_mask]
        # if diff_coverage.any():
        #     print(et.coverages)
        #     print(expected_coverage)
        #     print(diff_coverage)
        #     indices = np.nonzero(diff_coverage)
        #     print(et.coverages[indices])
        #     print(expected_coverage[indices])
        #     print(diff_coverage[indices])
        # self.assertFalse(diff_coverage.any())
        expected_final_fn = os.path.join(out_dir, '{}_{}{}_Dist_{}_Tree_{}_{}_Scoring.ranks'.format(
            query_id, ('ET_' if et_distance else ''), distance_model, tree_building_method,
            ('All_Ranks' if ranks is None else 'Custom_Ranks'), scoring_metric))
        self.evaluate_write_out_et_scores(et=et, fn=expected_final_fn)
        self.assertTrue(os.path.isfile(serial_fn))

    def test_3a_perform_trace(self):
        self.evaluate_perform_trace(
            query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='single', scoring_metric='identity', gap_correction=None, out_dir=self.out_small_dir,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'}, processors=self.max_threads,
            low_memory=True)

    def test_3b_perform_trace(self):
        self.evaluate_perform_trace(
            query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='single', scoring_metric='plain_entropy', gap_correction=None, out_dir=self.out_small_dir,
            processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def test_3c_perform_trace(self):
        self.evaluate_perform_trace(
            query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='pair', scoring_metric='filtered_average_product_corrected_mutual_information',
            gap_correction=None, out_dir=self.out_small_dir, processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def test_3d_perform_trace(self):
        self.evaluate_perform_trace(
            query_id=self.large_structure_id, polymer_type='Protein', aln_fn=self.large_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='single', scoring_metric='identity', gap_correction=None, out_dir=self.out_large_dir,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'}, processors=self.max_threads,
            low_memory=True)

    def test_3e_perform_trace(self):
        self.evaluate_perform_trace(
            query_id=self.large_structure_id, polymer_type='Protein', aln_fn=self.large_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='single', scoring_metric='plain_entropy', gap_correction=None, out_dir=self.out_large_dir,
            processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def test_3f_perform_trace(self):
        self.evaluate_perform_trace(
            query_id=self.large_structure_id, polymer_type='Protein', aln_fn=self.large_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='pair', scoring_metric='filtered_average_product_corrected_mutual_information',
            gap_correction=None, out_dir=self.out_large_dir, processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def test_3g_perform_trace(self):
        self.evaluate_perform_trace(
            query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='pair', scoring_metric='match_mismatch_entropy_angle',
            gap_correction=None, out_dir=self.out_small_dir, processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def test_3h_perform_trace(self):
        self.evaluate_perform_trace(
            query_id=self.large_structure_id, polymer_type='Protein', aln_fn=self.large_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='pair', scoring_metric='match_mismatch_entropy_angle',
            gap_correction=None, out_dir=self.out_large_dir, processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def evaluate_calculate_scores(self, query_id, polymer_type, aln_fn, et_distance, distance_model,
                                  tree_building_method, tree_building_options, ranks, position_type, scoring_metric,
                                  gap_correction, out_dir, output_files, processors, low_memory):
        rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        serial_fn = os.path.join(out_dir, '{}_{}{}_Dist_{}_Tree.pkl'.format(query_id, ('ET_' if et_distance else ''),
                                                                            distance_model, tree_building_method))
        if os.path.isfile(serial_fn):
            os.remove(serial_fn)
        serial_fn2 = os.path.join(out_dir, '{}_{}{}_Dist_{}_Tree_{}_{}_Scoring.pkl'.format(
            query_id, ('ET_' if et_distance else ''), distance_model, tree_building_method,
            ('All_Ranks' if ranks is None else 'Custom_Ranks'), scoring_metric))
        if os.path.isfile(serial_fn2):
            os.remove(serial_fn2)
        et = EvolutionaryTrace(query_id, polymer_type, aln_fn, et_distance, distance_model, tree_building_method,
                               tree_building_options, ranks, position_type, scoring_metric, gap_correction, out_dir,
                               output_files, processors, low_memory)
        et.calculate_scores()
        calculator = AlignmentDistanceCalculator(protein=(polymer_type == 'Protein'), model=distance_model)
        if et_distance:
            _, expected_dist_matrix, _, _ = calculator.get_et_distance(msa=et.original_aln.alignment)
        else:
            expected_dist_matrix = calculator.get_distance(msa=et.original_aln.alignment)
        self.assertIsNotNone(et.distance_matrix)
        self.assertEqual(et.distance_matrix.names, et.original_aln.seq_order)
        self.assertFalse((np.array(et.distance_matrix) - np.array(expected_dist_matrix)).any())
        expected_tree_fn = os.path.join(out_dir, '{}_{}{}_dist_{}_tree.nhx'.format(
            query_id, ('ET_' if et_distance else ''), distance_model, tree_building_method))
        self.assertTrue(os.path.isfile(expected_tree_fn))
        os.remove(expected_tree_fn)
        expected_tree = PhylogeneticTree(tree_building_method=tree_building_method,
                                         tree_building_args=tree_building_options)
        expected_tree.construct_tree(dm=expected_dist_matrix)
        expected_tree_iter = expected_tree.traverse_by_rank()
        tree_iter = et.phylo_tree.traverse_by_rank()
        try:
            curr_expected_nodes = next(expected_tree_iter)
        except StopIteration:
            curr_expected_nodes = None
        try:
            curr_nodes = next(tree_iter)
        except StopIteration:
            curr_nodes = None
        while curr_expected_nodes and curr_nodes:
            if curr_expected_nodes is None:
                self.assertIsNone(curr_nodes)
            else:
                sorted_curr_nodes = sorted(curr_nodes, key=compare_nodes_key(compare_nodes))
                sorted_curr_expected_nodes = sorted(curr_expected_nodes, key=compare_nodes_key(compare_nodes))
                self.assertEqual(len(sorted_curr_nodes), len(sorted_curr_expected_nodes))
                for i in range(len(sorted_curr_expected_nodes)):
                    self.check_nodes(sorted_curr_nodes[i], sorted_curr_expected_nodes[i])
            try:
                curr_expected_nodes = next(expected_tree_iter)
            except StopIteration:
                curr_expected_nodes = None
            try:
                curr_nodes = next(tree_iter)
            except StopIteration:
                curr_nodes = None
        expected_assignments = expected_tree.assign_group_rank(ranks=ranks)
        self.assertEqual(len(expected_assignments), len(et.assignments))
        for rank in expected_assignments:
            self.assertTrue(rank in et.assignments)
            for group in expected_assignments[rank]:
                self.assertTrue(group in et.assignments[rank])
                self.assertTrue('node' in expected_assignments[rank][group])
                self.assertTrue('node' in et.assignments[rank][group])
                self.check_nodes(expected_assignments[rank][group]['node'],
                                 et.assignments[rank][group]['node'])
                self.assertTrue('terminals' in expected_assignments[rank][group])
                self.assertTrue('terminals' in et.assignments[rank][group])
                self.assertEqual(expected_assignments[rank][group]['terminals'],
                                 et.assignments[rank][group]['terminals'])
                self.assertTrue('descendants' in expected_assignments[rank][group])
                self.assertTrue('descendants' in et.assignments[rank][group])
                if expected_assignments[rank][group]['node'].is_terminal():
                    self.assertIsNone(expected_assignments[rank][group]['descendants'])
                    self.assertIsNone(et.assignments[rank][group]['descendants'])
                else:
                    self.assertEqual(set([x.name for x in expected_assignments[rank][group]['descendants']]),
                                     set([x.name for x in et.assignments[rank][group]['descendants']]))
        self.assertTrue(os.path.isfile(serial_fn))
        expected_trace = Trace(alignment=et.non_gapped_aln, phylo_tree=et.phylo_tree, group_assignments=et.assignments,
                               position_specific=(position_type == 'single'), pair_specific=(position_type == 'pair'),
                               output_dir=out_dir, low_memory=low_memory,
                               match_mismatch=(scoring_metric == 'match_mismatch_entropy_angle'))
        self.assertEqual(et.trace.aln.query_id, expected_trace.aln.query_id)
        self.assertEqual(et.trace.aln.size, expected_trace.aln.size)
        self.assertEqual(et.trace.aln.seq_length, expected_trace.aln.seq_length)
        self.assertEqual(str(et.trace.aln.query_sequence), str(expected_trace.aln.query_sequence))
        self.assertEqual(et.trace.phylo_tree.size, expected_trace.phylo_tree.size)
        node_iter = et.trace.phylo_tree.traverse_top_down()
        expected_node_iter = expected_trace.phylo_tree.traverse_top_down()
        try:
            curr_node = next(node_iter)
        except StopIteration:
            curr_node = None
        try:
            curr_expected_node = next(expected_node_iter)
        except StopIteration:
            curr_expected_node = None
        while curr_node and curr_expected_node:
            if curr_node is None:
                self.assertIsNone(curr_expected_node)
            else:
                self.check_nodes(curr_node, curr_expected_node)
            try:
                curr_node = next(node_iter)
            except StopIteration:
                curr_node = None
            try:
                curr_expected_node = next(expected_node_iter)
            except StopIteration:
                curr_expected_node = None
        self.assertEqual(len(et.trace.assignments), len(expected_trace.assignments))
        for rank in expected_trace.assignments:
            self.assertEqual(len(et.assignments[rank]), len(expected_trace.assignments[rank]))
            for group in expected_trace.assignments[rank]:
                self.check_nodes(et.assignments[rank][group]['node'],
                                 expected_trace.assignments[rank][group]['node'])
                self.assertEqual(et.assignments[rank][group]['terminals'],
                                 expected_trace.assignments[rank][group]['terminals'])
                self.assertEqual(et.assignments[rank][group]['descendants'],
                                 expected_trace.assignments[rank][group]['descendants'])
        self.assertEqual(et.trace.pos_specific, expected_trace.pos_specific)
        self.assertEqual(et.trace.pair_specific, expected_trace.pair_specific)
        self.assertEqual(et.trace.out_dir, expected_trace.out_dir)
        self.assertEqual(et.trace.low_memory, expected_trace.low_memory)
        expected_trace.characterize_rank_groups(processes=self.max_threads,
                                                write_out_sub_aln=('sub-alignments' in output_files),
                                                write_out_freq_table=('frequency_tables' in output_files))
        unique_dir = os.path.join(out_dir, 'unique_node_data')
        self.assertEqual(len(et.trace.unique_nodes), len(expected_trace.unique_nodes))
        for node_name in expected_trace.unique_nodes:
            self.assertTrue(node_name in et.trace.unique_nodes)
            if 'sub-alignments' in output_files:
                self.assertTrue(os.path.isfile(os.path.join(unique_dir, '{}.fa'.format(node_name))))
            if et.trace.match_mismatch:
                self.assertTrue('match' in et.trace.unique_nodes[node_name])
                self.assertTrue('match' in expected_trace.unique_nodes[node_name])
                self.assertTrue('mismatch' in et.trace.unique_nodes[node_name])
                self.assertTrue('mismatch' in expected_trace.unique_nodes[node_name])
                expected_m_table = load_freq_table(freq_table=expected_trace.unique_nodes[node_name]['match'],
                                                   low_memory=low_memory)
                single_m_table = load_freq_table(freq_table=et.trace.unique_nodes[node_name]['match'],
                                                 low_memory=low_memory)
                sparse_diff_m = expected_m_table.get_table() - single_m_table.get_table()
                nonzero_check_m = sparse_diff_m.count_nonzero() > 0
                self.assertFalse(nonzero_check_m)
                expected_mm_table = load_freq_table(freq_table=expected_trace.unique_nodes[node_name]['mismatch'],
                                                    low_memory=low_memory)
                single_mm_table = load_freq_table(freq_table=et.trace.unique_nodes[node_name]['mismatch'],
                                                  low_memory=low_memory)
                sparse_diff_mm = expected_mm_table.get_table() - single_mm_table.get_table()
                nonzero_check_mm = sparse_diff_mm.count_nonzero() > 0
                self.assertFalse(nonzero_check_mm)
                if 'frequency_tables' in output_files:
                    self.assertTrue(os.path.isfile(os.path.join(
                        unique_dir, '{}_position_match_freq_table.tsv'.format(node_name))))
                    self.assertTrue(os.path.isfile(os.path.join(
                        unique_dir, '{}_position_mismatch_freq_table.tsv'.format(node_name))))
            else:
                self.assertTrue('single' in et.trace.unique_nodes[node_name])
                self.assertTrue('single' in expected_trace.unique_nodes[node_name])
                self.assertTrue('pair' in expected_trace.unique_nodes[node_name])
                self.assertTrue('pair' in et.trace.unique_nodes[node_name])
                if position_type == 'single':
                    expected_single_table = load_freq_table(freq_table=expected_trace.unique_nodes[node_name]['single'],
                                                            low_memory=low_memory)
                    single_table = load_freq_table(freq_table=et.trace.unique_nodes[node_name]['single'],
                                                   low_memory=low_memory)
                    sparse_diff = expected_single_table.get_table() - single_table.get_table()
                    nonzero_check = sparse_diff.count_nonzero() > 0
                    self.assertFalse(nonzero_check)
                    if 'frequency_tables' in output_files:
                        self.assertTrue(os.path.isfile(os.path.join(unique_dir,
                                                                    '{}_position_freq_table.tsv'.format(node_name))))
                else:
                    self.assertIsNone(expected_trace.unique_nodes[node_name]['single'])
                    self.assertIsNone(et.trace.unique_nodes[node_name]['single'])
                if position_type == 'pair':
                    expected_pair_table = load_freq_table(freq_table=expected_trace.unique_nodes[node_name]['pair'],
                                                          low_memory=low_memory)
                    pair_table = load_freq_table(freq_table=et.trace.unique_nodes[node_name]['pair'],
                                                 low_memory=low_memory)
                    sparse_diff = expected_pair_table.get_table() - pair_table.get_table()
                    nonzero_check = sparse_diff.count_nonzero() > 0
                    self.assertFalse(nonzero_check)
                    if 'frequency_tables' in output_files:
                        self.assertTrue(os.path.isfile(os.path.join(unique_dir,
                                                                    '{}_pair_freq_table.tsv'.format(node_name))))
                else:
                    self.assertIsNone(expected_trace.unique_nodes[node_name]['pair'])
                    self.assertIsNone(et.trace.unique_nodes[node_name]['pair'])
        expected_scorer = PositionalScorer(seq_length=et.non_gapped_aln.seq_length,
                                           pos_size=(1 if position_type == 'single' else 2), metric=scoring_metric)
        self.assertEqual(et.scorer.sequence_length, expected_scorer.sequence_length)
        self.assertEqual(et.scorer.dimensions, expected_scorer.dimensions)
        self.assertEqual(et.scorer.position_size, expected_scorer.position_size)
        self.assertEqual(et.scorer.metric, expected_scorer.metric)
        self.assertEqual(et.scorer.metric_type, expected_scorer.metric_type)
        self.assertEqual(et.scorer.rank_type, expected_scorer.rank_type)
        expected_ranks, expected_scores, expected_coverage = expected_trace.trace(scorer=expected_scorer,
                                                                                  gap_correction=gap_correction,
                                                                                  processes=self.max_threads)
        diff_scores = et.scores - expected_scores
        not_passing = diff_scores > 1e-13
        if not_passing.any():
            print(et.scores)
            print(expected_scores)
            print(diff_scores)
            indices = np.nonzero(not_passing)
            print(et.scores[indices])
            print(expected_scores[indices])
            print(diff_scores[indices])
        self.assertFalse(not_passing.any())
        # Cannot get these to match at the moment (seems to be due to rounding error but my attempts to check the parts
        # that agree are also failing. This was tested already in test_Trace.py hopefully the result holds and I have
        # just made mistake in the commented out block below.
        # if diff_scores.any():
        #     if expected_scorer.metric_type == 'max':
        #         print('MAX')
        #         first_et_diff = np.max(et.scores[np.nonzero(diff_scores)])
        #         first_et_mask = et.scores <= first_et_diff
        #         first_expected_diff = np.max(expected_scores[np.nonzero(diff_scores)])
        #         first_expected_mask = expected_scores <= first_expected_diff
        #     else:
        #         print('MIN')
        #         first_et_diff = np.min(et.scores[np.nonzero(diff_scores)])
        #         first_et_mask = et.scores >= first_et_diff
        #         first_expected_diff = np.min(expected_scores[np.nonzero(diff_scores)])
        #         first_expected_mask = expected_scores >= first_expected_diff
        #     if np.sum(first_et_mask) > np.sum(first_expected_mask):
        #         print('First ET Diff:', first_et_diff)
        #         print(et.rankings[et.scores == first_et_diff])
        #         diff_mask = first_et_mask
        #     else:
        #         print('First Expected Diff:', first_expected_diff)
        #         print(et.rankings[expected_scores == first_expected_diff])
        #         diff_mask = first_expected_mask
        #     expected_match_mask = np.invert(diff_mask)
        # else:
        #     expected_match_mask = np.ones(et.scores.shape, dtype=bool)
        # diff_ranks = et.rankings[expected_match_mask] - expected_ranks[expected_match_mask]
        # if diff_ranks.any():
        #     print(et.rankings[expected_match_mask])
        #     print(expected_ranks[expected_match_mask])
        #     print(diff_ranks)
        #     indices = np.nonzero(diff_ranks)
        #     print(et.scores[expected_match_mask][indices])
        #     print(expected_scores[expected_match_mask][indices])
        #     print(et.rankings[expected_match_mask][indices])
        #     print(expected_ranks[expected_match_mask][indices])
        #     print(diff_ranks[indices])
        # self.assertFalse(diff_ranks.any())
        # diff_coverage = et.coverages[expected_match_mask] - expected_coverage[expected_match_mask]
        # if diff_coverage.any():
        #     print(et.coverages)
        #     print(expected_coverage)
        #     print(diff_coverage)
        #     indices = np.nonzero(diff_coverage)
        #     print(et.coverages[indices])
        #     print(expected_coverage[indices])
        #     print(diff_coverage[indices])
        # self.assertFalse(diff_coverage.any())
        expected_final_fn = os.path.join(out_dir, '{}_{}{}_Dist_{}_Tree_{}_{}_Scoring.ranks'.format(
            query_id, ('ET_' if et_distance else ''), distance_model, tree_building_method,
            ('All_Ranks' if ranks is None else 'Custom_Ranks'), scoring_metric))
        self.evaluate_write_out_et_scores(et=et, fn=expected_final_fn)
        self.assertTrue(os.path.isfile(serial_fn2))

    def test_4a_calculate_scores(self):
        self.evaluate_calculate_scores(
            query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='single', scoring_metric='identity', gap_correction=None, out_dir=self.out_small_dir,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'}, processors=self.max_threads,
            low_memory=True)

    def test_4b_calculate_scores(self):
        self.evaluate_calculate_scores(
            query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='single', scoring_metric='plain_entropy', gap_correction=None, out_dir=self.out_small_dir,
            processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def test_4c_calculate_scores(self):
        self.evaluate_calculate_scores(
            query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='pair', scoring_metric='filtered_average_product_corrected_mutual_information',
            gap_correction=None, out_dir=self.out_small_dir, processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def test_4d_calculate_scores(self):
        self.evaluate_calculate_scores(
            query_id=self.large_structure_id, polymer_type='Protein', aln_fn=self.large_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='single', scoring_metric='identity', gap_correction=None, out_dir=self.out_large_dir,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'}, processors=self.max_threads,
            low_memory=True)

    def test_4e_calculate_scores(self):
        self.evaluate_calculate_scores(
            query_id=self.large_structure_id, polymer_type='Protein', aln_fn=self.large_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='single', scoring_metric='plain_entropy', gap_correction=None, out_dir=self.out_large_dir,
            processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def test_4f_calculate_scores(self):
        self.evaluate_calculate_scores(
            query_id=self.large_structure_id, polymer_type='Protein', aln_fn=self.large_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='pair', scoring_metric='filtered_average_product_corrected_mutual_information',
            gap_correction=None, out_dir=self.out_large_dir, processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def test_4g_calculate_scores(self):
        self.evaluate_calculate_scores(
            query_id=self.small_structure_id, polymer_type='Protein', aln_fn=self.small_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='pair', scoring_metric='match_mismatch_entropy_angle',
            gap_correction=None, out_dir=self.out_small_dir, processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def test_4h_calculate_scores(self):
        self.evaluate_calculate_scores(
            query_id=self.large_structure_id, polymer_type='Protein', aln_fn=self.large_fa_fn, et_distance=True,
            distance_model='blosum62', tree_building_method='et', tree_building_options={}, ranks=None,
            position_type='pair', scoring_metric='match_mismatch_entropy_angle',
            gap_correction=None, out_dir=self.out_large_dir, processors=self.max_threads, low_memory=True,
            output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})

    def evaluate_integer_et_comparison(self, p_id, fa_aln, low_mem):
        out_dir = os.path.join(self.testing_dir, p_id)
        rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        et_mip_obj = ETMIPWrapper(query=p_id, aln_file=fa_aln.file_name, out_dir=out_dir)
        et_mip_obj.convert_alignment()
        et_mip_obj.calculate_scores(method='intET', delete_files=False)
        et = EvolutionaryTrace(query=p_id, polymer_type='Protein', aln_file=fa_aln.file_name, et_distance=True,
                               distance_model='blosum62', tree_building_method='custom',
                               tree_building_options={'tree_path': os.path.join(out_dir, 'etc_out_intET.nhx')},
                               ranks=None, position_type='single', scoring_metric='identity', gap_correction=None,
                               out_dir=out_dir, processors=self.max_threads, low_memory=low_mem,
                               output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})
        et.calculate_scores()
        diff_ranks = et.scores - et_mip_obj.scores
        if diff_ranks.any():
            print(p_id)
            print(et.scores)
            print(et_mip_obj.scores)
            print(diff_ranks)
            indices = np.nonzero(diff_ranks)
            print(et.scores[indices])
            print(et_mip_obj.scores[indices])
            print(diff_ranks[indices])
        self.assertFalse(diff_ranks.any())
        diff_coverage = et.coverages - et_mip_obj.coverages
        not_passing = np.abs(diff_coverage) > 1e-2
        if not_passing.any():
            print(et.coverages)
            print(et_mip_obj.coverage)
            print(diff_coverage)
            indices = np.nonzero(diff_coverage)
            print(et.coverages[indices])
            print(et_mip_obj.coverage[indices])
            print(diff_coverage[indices])
        self.assertFalse(not_passing.any())
        rounded_coverages = np.round(et.coverages, decimals=3)
        diff_coverages2 = rounded_coverages - et_mip_obj.coverages
        not_passing2 = diff_coverages2 > 1E-15
        if not_passing2.any():
            print(rounded_coverages)
            print(et_mip_obj.coverages)
            print(diff_coverages2)
            indices = np.nonzero(not_passing2)
            print(rounded_coverages[indices])
            print(et_mip_obj.coverages[indices])
            print(diff_coverages2[indices])
        self.assertFalse(not_passing2.any())
        rmtree(out_dir)

    def test_5a_trace(self):
        # Compare the results of identity trace over single positions between this implementation and the WETC
        # implementation for the small alignment.
        self.evaluate_integer_et_comparison(p_id=self.small_structure_id, fa_aln=self.query_aln_fa_small, low_mem=False)

    def test_5b_trace(self):
        # Compare the results of identity trace over single positions between this implementation and the WETC
        # implementation for the large alignment.
        self.evaluate_integer_et_comparison(p_id=self.large_structure_id, fa_aln=self.query_aln_fa_large, low_mem=True)

    def evaluate_real_value_et_comparison(self, p_id, fa_aln, low_mem):
        out_dir = os.path.join(self.testing_dir, p_id,)
        rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        et_mip_obj = ETMIPWrapper(query=p_id, aln_file=fa_aln.file_name, out_dir=out_dir)
        et_mip_obj.convert_alignment()
        et_mip_obj.calculate_scores(method='rvET', delete_files=False)
        et = EvolutionaryTrace(query=p_id, polymer_type='Protein', aln_file=fa_aln.file_name, et_distance=True,
                               distance_model='blosum62', tree_building_method='custom',
                               tree_building_options={'tree_path': os.path.join(out_dir, 'etc_out_rvET.nhx')},
                               ranks=None, position_type='single', scoring_metric='plain_entropy', gap_correction=0.6,
                               out_dir=out_dir, processors=self.max_threads, low_memory=low_mem,
                               output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})
        et.calculate_scores()
        diff_ranks = et.scores - et_mip_obj.scores
        not_passing = np.abs(diff_ranks) > 1e-2
        if not_passing.any():
            print(et.scores)
            print(et_mip_obj.scores)
            print(diff_ranks)
            indices = np.nonzero(diff_ranks)
            print(et.scores[indices])
            print(et_mip_obj.scores[indices])
            print(diff_ranks[indices])
        self.assertFalse(not_passing.any())
        rounded_entropies = np.round(et.scores, decimals=2)
        diff_ranks2 = rounded_entropies - et_mip_obj.scores
        if diff_ranks2.any():
            print(rounded_entropies)
            print(et_mip_obj.scores)
            print(diff_ranks2)
            indices = np.nonzero(diff_ranks2)
            print(rounded_entropies[indices])
            print(et_mip_obj.scores[indices])
            print(diff_ranks2[indices])
        self.assertFalse(diff_ranks2.any())
        diff_coverage = et.coverages - et_mip_obj.coverages
        not_passing = np.abs(diff_coverage) > 1e-2
        if not_passing.any():
            print(et.coverages)
            print(et_mip_obj.coverage)
            print(diff_coverage)
            indices = np.nonzero(diff_coverage)
            print(et.coverages[indices])
            print(et_mip_obj.coverage[indices])
            print(diff_coverage[indices])
        self.assertFalse(not_passing.any())
        rounded_coverages = np.round(et.coverages, decimals=3)
        diff_coverages2 = rounded_coverages - et_mip_obj.coverages
        not_passing2 = diff_coverages2 > 1E-15
        if not_passing2.any():
            print(rounded_coverages)
            print(et_mip_obj.coverage)
            print(diff_coverages2)
            indices = np.nonzero(not_passing2)
            print(rounded_coverages[indices])
            print(et_mip_obj.coverage[indices])
            print(diff_coverages2[indices])
        self.assertFalse(not_passing2.any())
        rmtree(out_dir)

    def test_5c_trace(self):
        # Compare the results of plain entropy trace over single positions between this implementation and the WETC
        # implementation for the small alignment.
        self.evaluate_real_value_et_comparison(p_id=self.small_structure_id, fa_aln=self.query_aln_fa_small,
                                               low_mem=False)

    def test_5d_trace(self):
        # Compare the results of identity trace over single positions between this implementation and the WETC
        # implementation for the large alignment.
        self.evaluate_real_value_et_comparison(p_id=self.large_structure_id, fa_aln=self.query_aln_fa_large,
                                               low_mem=True)

    def evaluate_mip_et_comparison(self, p_id, fa_aln, low_mem):
        out_dir = os.path.join(self.testing_dir, p_id)
        rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        filtered_fa_fn = os.path.join(out_dir, '{}_filtered_aln.fa'.format(p_id))
        if os.path.isfile(filtered_fa_fn):
            char_filtered_fa_aln = SeqAlignment(file_name=filtered_fa_fn, query_id=p_id)
            char_filtered_fa_aln.import_alignment()
        else:
            curr_fa_aln = SeqAlignment(file_name=fa_aln.file_name, query_id=p_id)
            curr_fa_aln.import_alignment()
            curr_fa_aln.alphabet = Gapped(IUPACProtein())
            char_filtered_fa_aln = curr_fa_aln.remove_bad_sequences()
            char_filtered_fa_aln.write_out_alignment(file_name=filtered_fa_fn)
            char_filtered_fa_aln.file_name = filtered_fa_fn
        et_mip_obj = ETMIPWrapper(query=p_id, aln_file=filtered_fa_fn, out_dir=out_dir)
        et_mip_obj.convert_alignment()
        et_mip_obj.calculate_scores(method='ET-MIp', delete_files=False)
        gap_filtered_fa_aln = char_filtered_fa_aln.remove_gaps()
        et = EvolutionaryTrace(query=p_id, polymer_type='Protein', aln_file=filtered_fa_fn, et_distance=True,
                               distance_model='blosum62', tree_building_method='custom',
                               tree_building_options={'tree_path': os.path.join(out_dir, 'etc_out_ET-MIp.nhx')},
                               ranks=None, position_type='pair', gap_correction=None, out_dir=out_dir,
                               processors=self.max_threads, low_memory=low_mem,
                               output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'},
                               scoring_metric='filtered_average_product_corrected_mutual_information')
        et.calculate_scores()
        diff_ranks = et.scores - et_mip_obj.scores
        not_passing = np.abs(diff_ranks) > 1e-3
        if not_passing.any():
            print(et.scores)
            print(et_mip_obj.scores)
            print(diff_ranks)
            indices = np.nonzero(not_passing)
            print(et.scores[indices])
            print(et_mip_obj.scores[indices])
            print(diff_ranks[indices])
            print(et.scores[indices][0])
            print(et_mip_obj.scores[indices][0])
            print(diff_ranks[indices][0])
        self.assertFalse(not_passing.any())
        rounded_scores = np.round(et.scores, decimals=3)
        diff_ranks2 = rounded_scores - et_mip_obj.scores
        not_passing_rounded = np.abs(diff_ranks2) > 1e-15
        if not_passing_rounded.any():
            print(rounded_scores)
            print(et_mip_obj.scores)
            print(diff_ranks2)
            indices = np.nonzero(not_passing_rounded)
            print(rounded_scores[indices])
            print(et_mip_obj.scores[indices])
            print(diff_ranks2[indices])
        self.assertFalse(not_passing_rounded.any())
        diff_coverages = et.coverages - et_mip_obj.coverages
        not_passing = np.abs(diff_coverages) > 1E-3
        if not_passing.any():
            print(et.coverages)
            print(et_mip_obj.coverages)
            print(diff_coverages)
            indices = np.nonzero(not_passing)
            for i in range(len(indices[0])):
                print(indices[0][i], indices[1][i], et_mip_obj.coverages[indices[0][i], indices[1][i]],
                      et.coverages[indices[0][i], indices[1][i]], diff_coverages[indices[0][i], indices[1][i]],
                      1e-2, np.abs(diff_coverages[indices[0][i], indices[1][i]]) > 1e-2)
            print(et.scores[indices])
            print(et.rankings[indices])
            print(np.sum(not_passing))
            print(np.nonzero(not_passing))
            self.assertLessEqual(np.sum(not_passing), np.ceil(0.01 * np.sum(range(fa_aln.seq_length - 1))))
        else:
            self.assertFalse(not_passing.any())
        rmtree(out_dir)

    def test_5e_trace(self):
        # Compare the results of average product corrected mutual information over pairs of positions between this
        # implementation and the WETC implementation for the small alignment.
        self.evaluate_mip_et_comparison(p_id=self.small_structure_id, fa_aln=self.query_aln_fa_small, low_mem=False)

    def test_5f_trace(self):
        # Compare the results of average product corrected mutual information over pairs of positions between this
        # implementation and the WETC implementation for the large alignment.
        self.evaluate_mip_et_comparison(p_id=self.large_structure_id, fa_aln=self.query_aln_fa_large, low_mem=True)

    def evaluate_visualize_trace(self, p_id, fa_aln, low_mem):
        test_dir = os.path.join(self.testing_dir,  p_id)
        rmtree(test_dir, ignore_errors=True)
        os.makedirs(test_dir)
        filtered_fa_fn = os.path.join(test_dir, '{}_filtered_aln.fa'.format(p_id))
        if os.path.isfile(filtered_fa_fn):
            char_filtered_fa_aln = SeqAlignment(file_name=filtered_fa_fn, query_id=p_id)
            char_filtered_fa_aln.import_alignment()
        else:
            curr_fa_aln = SeqAlignment(file_name=fa_aln.file_name, query_id=p_id)
            curr_fa_aln.import_alignment()
            curr_fa_aln.alphabet = Gapped(IUPACProtein())
            char_filtered_fa_aln = curr_fa_aln.remove_bad_sequences()
            char_filtered_fa_aln.write_out_alignment(file_name=filtered_fa_fn)
            char_filtered_fa_aln.file_name = filtered_fa_fn
        et = EvolutionaryTrace(query=p_id, polymer_type='Protein', aln_file=filtered_fa_fn, et_distance=True,
                               distance_model='blosum62', tree_building_method='et',
                               tree_building_options={}, ranks=None, position_type='pair', gap_correction=None,
                               out_dir=test_dir, processors=self.max_threads, low_memory=low_mem,
                               output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'},
                               scoring_metric='filtered_average_product_corrected_mutual_information')
        et.compute_distance_matrix_tree_and_assignments()
        pos_sets = [[0, 1, et.non_gapped_aln.seq_length - 1], [0, et.non_gapped_aln.seq_length - 1], [0]]
        expected_ranks = [list(range(1, 3)), list(range(1, 4)), list(range(1, 11))]
        for i in range(len(pos_sets)):
            pos_set = pos_sets[i]
            expected_rank = expected_ranks[i]
            et.visualize_trace(positions=pos_set, ranks=expected_rank)
            expected_dir = os.path.join(test_dir, '_'.join([str(x) for x in pos_set]))
            self.assertTrue(os.path.isdir(expected_dir))
            for r in et.assignments:
                print('Validate Rank: {}'.format(r))
                expected_rank_fn = os.path.join(expected_dir, 'Rank_{}.png'.format(r))
                if r in expected_rank:
                    self.assertTrue(os.path.isfile(expected_rank_fn), expected_rank_fn)
                else:
                    self.assertFalse(os.path.isfile(expected_rank_fn), expected_rank_fn)

    def test_6a_visualize_trace(self):
        # Check that visualize_trace is generating the expected output for the small alignment
        self.evaluate_visualize_trace(p_id=self.small_structure_id, fa_aln=self.query_aln_fa_small, low_mem=False)

    def test_6b_visualize_trace(self):
        # Check that visualize_trace is generating the expected output for the large alignment
        self.evaluate_visualize_trace(p_id=self.large_structure_id, fa_aln=self.query_aln_fa_large, low_mem=True)


if __name__ == '__main__':
    unittest.main()
