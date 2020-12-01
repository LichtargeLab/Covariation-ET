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
from unittest import TestCase
from Bio.Alphabet import Gapped
from Bio.Alphabet.IUPAC import IUPACProtein
from test_Base import processes as max_processes
from test_Base import (protein_aln, compare_nodes_key, compare_nodes, protein_phylo_tree, pro_single_ft, pro_pair_ft,
                       protein_mm_freq_tables, protein_mm_table, pro_pair_alpha_size, pro_pair_map, pro_pair_rev)
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from Trace import Trace, load_freq_table, load_numpy_array
from PhylogeneticTree import PhylogeneticTree
from PositionalScorer import PositionalScorer
from FrequencyTable import FrequencyTable
from SeqAlignment import SeqAlignment
sys.path.append(os.path.abspath('..'))
from EvolutionaryTrace import EvolutionaryTrace, init_var_pool, get_var_pool


def check_nodes(test_case, node1, node2):
    if node1.is_terminal():
        test_case.assertTrue(node2.is_terminal())
        test_case.assertEqual(node1.name, node2.name)
    else:
        test_case.assertTrue(node2.is_bifurcating())
        test_case.assertFalse(node2.is_terminal())
        test_case.assertEqual(set([x.name for x in node1.get_terminals()]),
                              set([x.name for x in node2.get_terminals()]))


class TestEvolutionaryTraceInit(TestCase):

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
        os.remove(et.original_aln_fn)
        os.remove(et.non_gapped_aln_fn)

    def test_evolutionary_trace_init(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='single', scoring_metric='identity', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_blosum62(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='blosum62', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='single', scoring_metric='identity', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_blosum62_et_dist(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=True,
                           distance_model='blosum62', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='single', scoring_metric='identity', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_et_tree(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='et', tree_building_options={},
                           ranks=None, position_type='single', scoring_metric='identity', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_agglomerative_tree(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='agglomerative',
                           tree_building_options={'cache_dir': None, 'affinity': 'euclidean', 'linkage': 'ward'},
                           ranks=None, position_type='single', scoring_metric='identity', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_custom_tree(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='custom',
                           tree_building_options={'tree_path': os.path.join(test_dir, 'custom_tree.nhx')},
                           ranks=None, position_type='single', scoring_metric='identity', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_all_ranks(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=[1, 2, 3], position_type='single', scoring_metric='identity', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_first_rank(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=[1], position_type='single', scoring_metric='identity', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_top_ranks(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=[1, 2], position_type='single', scoring_metric='identity', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_skip_middle_rank(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=[1, 3], position_type='single', scoring_metric='identity', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_pair(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='identity', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_gap_correction(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='single', scoring_metric='identity', gap_correction=0.6,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_all_output_files(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='single', scoring_metric='identity', gap_correction=None,
                           out_dir=test_dir,
                           output_files=['original_aln', 'non-gap_aln', 'tree', 'sub-alignment', 'frequency_tables',
                                         'scores'], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_common_output_files(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='single', scoring_metric='identity', gap_correction=None,
                           out_dir=test_dir, output_files=['original_aln', 'non-gap_aln', 'tree'], processors=1,
                           low_memory=False, expected_length=3, expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_multiprocess(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='single', scoring_metric='identity', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=2, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_low_memory(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='single', scoring_metric='identity', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=True, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_entropy(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='single', scoring_metric='plain_entropy', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_mi(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='mutual_information', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_nmi(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='normalized_mutual_information',
                           gap_correction=None, out_dir=test_dir, output_files=[], processors=1, low_memory=False,
                           expected_length=3, expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_mip(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair',
                           scoring_metric='average_product_corrected_mutual_information', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_fmip(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair',
                           scoring_metric='filtered_average_product_corrected_mutual_information', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_mcm_(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='match_count', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_m_mc(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='mismatch_count', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_mcmcr(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='match_mismatch_count_ratio',
                           gap_correction=None, out_dir=test_dir, output_files=[], processors=1, low_memory=False,
                           expected_length=3, expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_mcmca(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='match_mismatch_count_angle',
                           gap_correction=None, out_dir=test_dir, output_files=[], processors=1, low_memory=False,
                           expected_length=3, expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_mem_(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='match_entropy', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_m_me(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='mismatch_entropy', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_memer(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='match_mismatch_entropy_ratio',
                           gap_correction=None, out_dir=test_dir, output_files=[], processors=1, low_memory=False,
                           expected_length=3, expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_memea(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='match_mismatch_entropy_angle',
                           gap_correction=None, out_dir=test_dir, output_files=[], processors=1, low_memory=False,
                           expected_length=3, expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_mdm_(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='match_diversity', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_m_md(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='mismatch_diversity', gap_correction=None,
                           out_dir=test_dir, output_files=[], processors=1, low_memory=False, expected_length=3,
                           expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_mdmdr(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='match_mismatch_diversity_ratio',
                           gap_correction=None, out_dir=test_dir, output_files=[], processors=1, low_memory=False,
                           expected_length=3, expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_mdmda(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='match_mismatch_diversity_angle',
                           gap_correction=None, out_dir=test_dir, output_files=[], processors=1, low_memory=False,
                           expected_length=3, expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_mdmer(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='match_diversity_mismatch_entropy_ratio',
                           gap_correction=None, out_dir=test_dir, output_files=[], processors=1, low_memory=False,
                           expected_length=3, expected_sequence='MET')
        rmtree(test_dir)

    def test_evolutionary_trace_init_mdmea(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_init(query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False,
                           distance_model='identity', tree_building_method='upgma', tree_building_options={},
                           ranks=None, position_type='pair', scoring_metric='match_diversity_mismatch_entropy_angle',
                           gap_correction=None, out_dir=test_dir, output_files=[], processors=1, low_memory=False,
                           expected_length=3, expected_sequence='MET')
        rmtree(test_dir)


class TestEvolutionaryTraceComputeDistanceMatrixTreeAndAssignments(TestCase):

    def evaluate_compute_distance_matrix_tree_and_assignments(self, query_id, polymer_type, aln_fn, et_distance,
                                                              distance_model, tree_building_method,
                                                              tree_building_options, ranks, position_type,
                                                              scoring_metric, gap_correction, out_dir, output_files,
                                                              processors, low_memory):
            serial_fn = os.path.join(out_dir, '{}_{}{}_Dist_{}_Tree.pkl'.format(query_id,
                                                                                ('ET_' if et_distance else ''),
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
                        check_nodes(self, sorted_curr_nodes[i], sorted_curr_expected_nodes[i])
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
                    check_nodes(self, expected_assignments[rank][group]['node'],
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
                        if expected_assignments[rank][group]['descendants'] is None:
                            self.assertIsNone(et.assignments[rank][group]['descendants'])
                        else:
                            self.assertEqual(set([x.name for x in expected_assignments[rank][group]['descendants']]),
                                             set([x.name for x in et.assignments[rank][group]['descendants']]))
            self.assertTrue(os.path.isfile(serial_fn))

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=['tree'], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_blosum62(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='blosum62',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=['tree'], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_blosum62_et_dist(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=True, distance_model='blosum62',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=['tree'], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_et_tree(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='et', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=['tree'], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_agglomerative_tree(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='agglomerative',
            tree_building_options={'cache_dir': None, 'affinity': 'euclidean', 'linkage': 'ward'},
            ranks=None, position_type='single', scoring_metric='identity', gap_correction=None, out_dir=test_dir,
            output_files=['tree'], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_custom_tree(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        tree_fn = os.path.join(test_dir, 'custom_tree.nhx')
        protein_phylo_tree.write_out_tree(tree_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='custom', tree_building_options={'tree_path': tree_fn}, ranks=None,
            position_type='single', scoring_metric='identity', gap_correction=None, out_dir=test_dir,
            output_files=['tree'], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_all_ranks(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=[1, 2, 3], position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=['tree'], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_first_rank(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=[1], position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=['tree'], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_top_ranks(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=[1, 2], position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=['tree'], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_skip_middle_rank(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=[1, 3], position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=['tree'], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_pair(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=['tree'], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_gap_correction(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=0.6, out_dir=test_dir, output_files=['tree'], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_all_output_files(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir,
            output_files=['original_aln', 'non-gap_aln', 'tree', 'sub-alignment', 'frequency_tables', 'scores'],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_common_output_files(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir,
            output_files=['original_aln', 'non-gap_aln', 'tree'], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_multiprocess(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=['tree'], processors=2,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_low_memory(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=['tree'], processors=1,
            low_memory=True)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_entropy(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='plain_entropy', gap_correction=None, out_dir=test_dir, output_files=['tree'],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_mi(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='mutual_information', gap_correction=None, out_dir=test_dir, output_files=['tree'],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_nmi(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='normalized_mutual_information', gap_correction=None, out_dir=test_dir,
            output_files=['tree'], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_mip(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='average_product_corrected_mutual_information', gap_correction=None, out_dir=test_dir,
            output_files=['tree'], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_fmip(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='filtered_average_product_corrected_mutual_information', gap_correction=None,
            out_dir=test_dir, output_files=['tree'], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_mcm_(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_count', gap_correction=None, out_dir=test_dir, output_files=['tree'],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_m_mc(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='mismatch_count', gap_correction=None, out_dir=test_dir, output_files=['tree'],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_mcmcr(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_count_ratio', gap_correction=None, out_dir=test_dir,
            output_files=['tree'], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_mcmca(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_count_angle', gap_correction=None, out_dir=test_dir,
            output_files=['tree'], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_mem_(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_entropy', gap_correction=None, out_dir=test_dir, output_files=['tree'],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_m_me(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='mismatch_entropy', gap_correction=None, out_dir=test_dir, output_files=['tree'],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_memer(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_entropy_ratio', gap_correction=None, out_dir=test_dir,
            output_files=['tree'], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_memea(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_entropy_angle', gap_correction=None, out_dir=test_dir,
            output_files=['tree'], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_mdm_(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_diversity', gap_correction=None, out_dir=test_dir, output_files=['tree'],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_m_md(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='mismatch_diversity', gap_correction=None, out_dir=test_dir, output_files=['tree'],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_mdmdr(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_diversity_ratio', gap_correction=None, out_dir=test_dir,
            output_files=['tree'], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_mdmda(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_diversity_angle', gap_correction=None, out_dir=test_dir,
            output_files=['tree'], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_mdmer(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_diversity_mismatch_entropy_ratio', gap_correction=None, out_dir=test_dir,
            output_files=['tree'], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_compute_distance_matrix_tree_and_assignments_mdmea(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_compute_distance_matrix_tree_and_assignments(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_diversity_mismatch_entropy_angle', gap_correction=None, out_dir=test_dir,
            output_files=['tree'], processors=1, low_memory=False)
        rmtree(test_dir)


class TestEvolutionaryTracePerformTraceAndWriteOutETScores(TestCase):

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
            raise ValueError('Cannot evaluate EvolutionaryTrace instance with position_type: {}'.format(
                et.position_type))
        root_name = et.assignments[1][1]['node'].name
        if et.trace.match_mismatch:
            root_freq_table = (load_freq_table(freq_table=et.trace.unique_nodes[root_name]['match'],
                                               low_memory=et.low_memory) +
                               load_freq_table(freq_table=et.trace.unique_nodes[root_name]['mismatch'],
                                               low_memory=et.low_memory))
        else:
            root_freq_table = load_freq_table(freq_table=et.trace.unique_nodes[root_name]['freq_table'],
                                              low_memory=et.low_memory)
        for ind in et_df.index:
            if et.position_type == 'single':
                position = et_df.loc[ind, 'Position'] - 1
                expected_query = et.non_gapped_aln.query_sequence[position]
                self.assertEqual(expected_query, et_df.loc[ind, 'Query'])
                expected_characters = root_freq_table.get_chars(pos=position)
                self.assertEqual(len(expected_characters), et_df.loc[ind, 'Variability_Count'],
                                 (len(expected_characters), et_df.loc[ind, 'Variability_Count'],
                                  set(expected_characters), set(et_df.loc[ind, 'Variability_Characters'].split(','))))
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
                self.assertEqual(len(expected_characters), et_df.loc[ind, 'Variability_Count'],
                                 (len(expected_characters), et_df.loc[ind, 'Variability_Count'],
                                  set(expected_characters), set(et_df.loc[ind, 'Variability_Characters'].split(','))))
                self.assertEqual(set(expected_characters),
                                 set(et_df.loc[ind, 'Variability_Characters'].split(',')))
                expected_rank = et.rankings[pos_i, pos_j]
                self.assertEqual(expected_rank, et_df.loc[ind, 'Rank'], fn)
                expected_score = et.scores[pos_i, pos_j]
                if et.scoring_metric == 'identity':
                    self.assertEqual(expected_score, et_df.loc[ind, 'Score'])
                else:
                    if expected_score > 1E14:
                        self.assertLessEqual(np.abs(expected_score - et_df.loc[ind, 'Score']), 35)
                    else:
                        self.assertLessEqual(np.abs(expected_score - et_df.loc[ind, 'Score']), 1e-3)
                expected_coverage = et.coverages[pos_i, pos_j]
                self.assertLessEqual(np.abs(expected_coverage - et_df.loc[ind, 'Coverage']), 1e-3)
        os.remove(fn)

    def evaluate_perform_trace(self, query_id, polymer_type, aln_fn, et_distance, distance_model,
                               tree_building_method, tree_building_options, ranks, position_type, scoring_metric,
                               gap_correction, out_dir, output_files, processors, low_memory):
        serial_fn = os.path.join(out_dir, '{}_{}{}_Dist_{}_Tree_{}_{}_Scoring.pkl'.format(
            query_id, ('ET_' if et_distance else ''), distance_model, tree_building_method,
            ('All_Ranks' if ranks is None else 'Custom_Ranks'), scoring_metric))
        if os.path.isfile(serial_fn):
            os.remove(serial_fn)
        et = EvolutionaryTrace(query_id, polymer_type, aln_fn, et_distance, distance_model, tree_building_method,
                               tree_building_options, ranks, position_type, scoring_metric, gap_correction, out_dir,
                               output_files, processors, low_memory)
        # with self.assertRaises(AttributeError):
        with self.assertRaises(ValueError):
            et.perform_trace()
        et.compute_distance_matrix_tree_and_assignments()
        et.perform_trace()
        expected_trace = Trace(alignment=et.non_gapped_aln, phylo_tree=et.phylo_tree,
                               group_assignments=et.assignments,
                               pos_size=(1 if position_type == 'single' else 2), output_dir=out_dir,
                               low_memory=low_memory,
                               match_mismatch=(('match' in scoring_metric) or ('mismatch' in scoring_metric)))
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
                check_nodes(self, curr_node, curr_expected_node)
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
                check_nodes(self, et.assignments[rank][group]['node'],
                                 expected_trace.assignments[rank][group]['node'])
                self.assertEqual(et.assignments[rank][group]['terminals'],
                                 expected_trace.assignments[rank][group]['terminals'])
                self.assertEqual(et.assignments[rank][group]['descendants'],
                                 expected_trace.assignments[rank][group]['descendants'])
        self.assertEqual(et.trace.pos_size, expected_trace.pos_size)
        self.assertEqual(et.trace.out_dir, expected_trace.out_dir)
        self.assertEqual(et.trace.low_memory, expected_trace.low_memory)
        expected_trace.characterize_rank_groups(processes=max_processes,
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
                self.assertTrue('freq_table' in et.trace.unique_nodes[node_name])
                self.assertTrue('freq_table' in expected_trace.unique_nodes[node_name])
                expected_single_table = load_freq_table(
                    freq_table=expected_trace.unique_nodes[node_name]['freq_table'], low_memory=low_memory)
                single_table = load_freq_table(freq_table=et.trace.unique_nodes[node_name]['freq_table'],
                                               low_memory=low_memory)
                expected_single_array = expected_single_table.get_table()
                single_array = single_table.get_table()
                sparse_diff = single_array - expected_single_array
                nonzero_check = sparse_diff.count_nonzero() > 0
                self.assertFalse(nonzero_check)
                if 'frequency_tables' in output_files:
                    self.assertTrue(os.path.isfile(os.path.join(unique_dir,
                                                                f'{node_name}_{position_type}_freq_table.tsv')))
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
                                                                                  processes=max_processes)
        diff_scores = et.scores - expected_scores
        not_passing = diff_scores > 1e-13
        not_passing[expected_scores > 1E14] = diff_scores[expected_scores > 1E14] > 35
        if not_passing.any():
            print(et.scores)
            print(expected_scores)
            print(diff_scores)
            indices = np.nonzero(not_passing)
            print(et.scores[indices])
            print(expected_scores[indices])
            print(diff_scores[indices])
        self.assertFalse(not_passing.any())
        expected_match_mask = np.ones(et.scores.shape, dtype=bool)
        diff_ranks = et.rankings[expected_match_mask] - expected_ranks[expected_match_mask]
        if diff_ranks.any():
            print(et.rankings[expected_match_mask])
            print(expected_ranks[expected_match_mask])
            print(diff_ranks)
            indices = np.nonzero(diff_ranks)
            print(et.scores[expected_match_mask][indices])
            print(expected_scores[expected_match_mask][indices])
            print(et.rankings[expected_match_mask][indices])
            print(expected_ranks[expected_match_mask][indices])
            print(diff_ranks[indices])
        self.assertFalse(diff_ranks.any())
        diff_coverage = et.coverages[expected_match_mask] - expected_coverage[expected_match_mask]
        if diff_coverage.any():
            print(et.coverages)
            print(expected_coverage)
            print(diff_coverage)
            indices = np.nonzero(diff_coverage)
            print(et.coverages[indices])
            print(expected_coverage[indices])
            print(diff_coverage[indices])
        self.assertFalse(diff_coverage.any())
        expected_final_fn = os.path.join(out_dir, '{}_{}{}_Dist_{}_Tree_{}_{}_Scoring.ranks'.format(
            query_id, ('ET_' if et_distance else ''), distance_model, tree_building_method,
            ('All_Ranks' if ranks is None else 'Custom_Ranks'), scoring_metric))
        self.evaluate_write_out_et_scores(et=et, fn=expected_final_fn)
        self.assertTrue(os.path.isfile(serial_fn))

    def test_evolutionary_trace_perform_trace(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_blosum62(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='blosum62',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_blosum62_et_dist(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=True, distance_model='blosum62',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_et_tree(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='et', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_agglomerative_tree(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='agglomerative',
            tree_building_options={'cache_dir': None, 'affinity': 'euclidean', 'linkage': 'ward'},
            ranks=None, position_type='single', scoring_metric='identity', gap_correction=None, out_dir=test_dir,
            output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_custom_tree(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        tree_fn = os.path.join(test_dir, 'custom_tree.nhx')
        protein_phylo_tree.write_out_tree(tree_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='custom', tree_building_options={'tree_path': tree_fn}, ranks=None,
            position_type='single', scoring_metric='identity', gap_correction=None, out_dir=test_dir,
            output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_all_ranks(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=[1, 2, 3], position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_first_rank(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=[1], position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_top_ranks(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=[1, 2], position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_skip_middle_rank(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=[1, 3], position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_pair(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_gap_correction(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=0.6, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_all_output_files(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir,
            output_files=['original_aln', 'non-gap_aln', 'tree', 'sub-alignment', 'frequency_tables', 'scores'],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_common_output_files(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir,
            output_files=['original_aln', 'non-gap_aln', 'tree'], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_multiprocess(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=2,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_low_memory(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=True)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_entropy(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='plain_entropy', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_mi(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='mutual_information', gap_correction=None, out_dir=test_dir, output_files=[],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_nmi(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='normalized_mutual_information', gap_correction=None, out_dir=test_dir, output_files=[],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_mip(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='average_product_corrected_mutual_information', gap_correction=None, out_dir=test_dir,
            output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_fmip(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='filtered_average_product_corrected_mutual_information', gap_correction=None,
            out_dir=test_dir, output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_mcm_(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_count', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_m_mc(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='mismatch_count', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_mcmcr(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_count_ratio', gap_correction=None, out_dir=test_dir, output_files=[],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_mcmca(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_count_angle', gap_correction=None, out_dir=test_dir, output_files=[],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_mem_(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_entropy', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_m_me(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='mismatch_entropy', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_memer(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_entropy_ratio', gap_correction=None, out_dir=test_dir, output_files=[],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_memea(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_entropy_angle', gap_correction=None, out_dir=test_dir, output_files=[],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_mdm_(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_diversity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_m_md(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='mismatch_diversity', gap_correction=None, out_dir=test_dir, output_files=[],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_mdmdr(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_diversity_ratio', gap_correction=None, out_dir=test_dir,
            output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_mdmda(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_diversity_angle', gap_correction=None, out_dir=test_dir,
            output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_mdmer(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_diversity_mismatch_entropy_ratio', gap_correction=None, out_dir=test_dir,
            output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_perform_trace_mdmea(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_perform_trace(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_diversity_mismatch_entropy_angle', gap_correction=None, out_dir=test_dir,
            output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)


class TestEvolutionaryTraceCalculateScores(TestCase):

    def evaluate_calculate_scores(self, query_id, polymer_type, aln_fn, et_distance, distance_model,
                                  tree_building_method, tree_building_options, ranks, position_type, scoring_metric,
                                  gap_correction, out_dir, output_files, processors, low_memory):
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
        self.assertTrue(os.path.isfile(serial_fn))
        expected_trace = Trace(alignment=et.non_gapped_aln, phylo_tree=et.phylo_tree, group_assignments=et.assignments,
                               pos_size=(1 if position_type == 'single' else 2), output_dir=out_dir,
                               low_memory=low_memory,
                               match_mismatch=(('match' in scoring_metric) or ('mismatch' in scoring_metric)))
        expected_trace.characterize_rank_groups(processes=max_processes,
                                                write_out_sub_aln=('sub-alignments' in output_files),
                                                write_out_freq_table=('frequency_tables' in output_files))
        expected_scorer = PositionalScorer(seq_length=et.non_gapped_aln.seq_length,
                                           pos_size=(1 if position_type == 'single' else 2), metric=scoring_metric)
        expected_ranks, expected_scores, expected_coverage = expected_trace.trace(scorer=expected_scorer,
                                                                                  gap_correction=gap_correction,
                                                                                  processes=max_processes)
        diff_scores = et.scores - expected_scores
        not_passing = diff_scores > 1e-13
        not_passing[expected_scores > 1E14] = diff_scores[expected_scores > 1E14] > 35
        if not_passing.any():
            print(et.scores)
            print(expected_scores)
            print(diff_scores)
            indices = np.nonzero(not_passing)
            print(et.scores[indices])
            print(expected_scores[indices])
            print(diff_scores[indices])
        self.assertFalse(not_passing.any())
        expected_match_mask = np.ones(et.scores.shape, dtype=bool)
        diff_ranks = et.rankings[expected_match_mask] - expected_ranks[expected_match_mask]
        if diff_ranks.any():
            print(et.rankings[expected_match_mask])
            print(expected_ranks[expected_match_mask])
            print(diff_ranks)
            indices = np.nonzero(diff_ranks)
            print(et.scores[expected_match_mask][indices])
            print(expected_scores[expected_match_mask][indices])
            print(et.rankings[expected_match_mask][indices])
            print(expected_ranks[expected_match_mask][indices])
            print(diff_ranks[indices])
        self.assertFalse(diff_ranks.any())
        diff_coverage = et.coverages[expected_match_mask] - expected_coverage[expected_match_mask]
        if diff_coverage.any():
            print(et.coverages)
            print(expected_coverage)
            print(diff_coverage)
            indices = np.nonzero(diff_coverage)
            print(et.coverages[indices])
            print(expected_coverage[indices])
            print(diff_coverage[indices])
        self.assertFalse(diff_coverage.any())
        expected_final_fn = os.path.join(out_dir, '{}_{}{}_Dist_{}_Tree_{}_{}_Scoring.ranks'.format(
            query_id, ('ET_' if et_distance else ''), distance_model, tree_building_method,
            ('All_Ranks' if ranks is None else 'Custom_Ranks'), scoring_metric))
        self.assertTrue(os.path.isfile(expected_final_fn))
        self.assertTrue(os.path.isfile(serial_fn2))
        self.assertIsNotNone(et.time)

    def test_evolutionary_trace_calculate_scores(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_blosum62(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='blosum62',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_blosum62_et_dist(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=True, distance_model='blosum62',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_et_tree(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='et', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_agglomerative_tree(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='agglomerative',
            tree_building_options={'cache_dir': None, 'affinity': 'euclidean', 'linkage': 'ward'},
            ranks=None, position_type='single', scoring_metric='identity', gap_correction=None, out_dir=test_dir,
            output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_custom_tree(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        tree_fn = os.path.join(test_dir, 'custom_tree.nhx')
        protein_phylo_tree.write_out_tree(tree_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='custom', tree_building_options={'tree_path': tree_fn}, ranks=None,
            position_type='single', scoring_metric='identity', gap_correction=None, out_dir=test_dir,
            output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_all_ranks(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=[1, 2, 3], position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_first_rank(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=[1], position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_top_ranks(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=[1, 2], position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_skip_middle_rank(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=[1, 3], position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_pair(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_gap_correction(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=0.6, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_all_output_files(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir,
            output_files=['original_aln', 'non-gap_aln', 'tree', 'sub-alignment', 'frequency_tables', 'scores'],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_common_output_files(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir,
            output_files=['original_aln', 'non-gap_aln', 'tree'], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_multiprocess(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=2,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_low_memory(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='identity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=True)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_entropy(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='single',
            scoring_metric='plain_entropy', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_mi(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='mutual_information', gap_correction=None, out_dir=test_dir, output_files=[],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_nmi(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='normalized_mutual_information', gap_correction=None, out_dir=test_dir, output_files=[],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_mip(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='average_product_corrected_mutual_information', gap_correction=None, out_dir=test_dir,
            output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_fmip(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='filtered_average_product_corrected_mutual_information', gap_correction=None,
            out_dir=test_dir, output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_mcm_(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_count', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_m_mc(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='mismatch_count', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_mcmcr(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_count_ratio', gap_correction=None, out_dir=test_dir, output_files=[],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_mcmca(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_count_angle', gap_correction=None, out_dir=test_dir, output_files=[],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_mem_(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_entropy', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_m_me(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='mismatch_entropy', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_memer(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_entropy_ratio', gap_correction=None, out_dir=test_dir, output_files=[],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_memea(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_entropy_angle', gap_correction=None, out_dir=test_dir, output_files=[],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_mdm_(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_diversity', gap_correction=None, out_dir=test_dir, output_files=[], processors=1,
            low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_m_md(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='mismatch_diversity', gap_correction=None, out_dir=test_dir, output_files=[],
            processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_mdmdr(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_diversity_ratio', gap_correction=None, out_dir=test_dir,
            output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_mdmda(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_mismatch_diversity_angle', gap_correction=None, out_dir=test_dir,
            output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_mdmer(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_diversity_mismatch_entropy_ratio', gap_correction=None, out_dir=test_dir,
            output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)

    def test_evolutionary_trace_calculate_scores_mdmea(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        os.mkdir(test_dir)
        aln_fn = os.path.join(test_dir, 'test_protein_aln.fasta')
        protein_aln.write_out_alignment(aln_fn)
        self.evaluate_calculate_scores(
            query_id='seq1', polymer_type='Protein', aln_fn=aln_fn, et_distance=False, distance_model='identity',
            tree_building_method='upgma', tree_building_options={}, ranks=None, position_type='pair',
            scoring_metric='match_diversity_mismatch_entropy_angle', gap_correction=None, out_dir=test_dir,
            output_files=[], processors=1, low_memory=False)
        rmtree(test_dir)


class TestEvolutionaryTraceVisualizeTrace(TestCase):

    def evaluate_visualize_trace(self, out_dir, p_id, fa_aln, low_mem, expected_pos_set, expected_rank):
        test_dir = os.path.join(out_dir, p_id)
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
                               out_dir=test_dir, processors=max_processes, low_memory=low_mem,
                               output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'},
                               scoring_metric='filtered_average_product_corrected_mutual_information')
        et.compute_distance_matrix_tree_and_assignments()
        et.visualize_trace(positions=expected_pos_set, ranks=expected_rank)
        expected_dir = os.path.join(test_dir, '_'.join([str(x) for x in expected_pos_set]))
        self.assertTrue(os.path.isdir(expected_dir))
        for r in et.assignments:
            print('Validate Rank: {}'.format(r))
            expected_rank_fn = os.path.join(expected_dir, 'Rank_{}.png'.format(r))
            if r in expected_rank:
                self.assertTrue(os.path.isfile(expected_rank_fn), expected_rank_fn)
            else:
                self.assertFalse(os.path.isfile(expected_rank_fn), expected_rank_fn)

    def test_visualize_trace_full_ranks_3_pos(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=False,
                                      expected_pos_set=[0, 1, 2], expected_rank=[1, 2, 3])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_full_ranks_3_pos_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=True,
                                      expected_pos_set=[0, 1, 2], expected_rank=[1, 2, 3])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_no_intermediate_rank_3_pos(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=False,
                                      expected_pos_set=[0, 1, 2], expected_rank=[1, 3])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_no_intermediate_rank_3_pos_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=True,
                                      expected_pos_set=[0, 1, 2], expected_rank=[1, 3])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_top_ranks_3_pos(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=False,
                                      expected_pos_set=[0, 1, 2], expected_rank=[1, 2])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_top_ranks_3_pos_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=True,
                                      expected_pos_set=[0, 1, 2], expected_rank=[1, 2])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_single_rank_3_pos(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=False,
                                      expected_pos_set=[0, 1, 2], expected_rank=[1])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_single_rank_3_pos_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=True,
                                      expected_pos_set=[0, 1, 2], expected_rank=[1])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_full_ranks_2_pos(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=False,
                                      expected_pos_set=[0, 2], expected_rank=[1, 2, 3])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_full_ranks_2_pos_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=True,
                                      expected_pos_set=[0, 2], expected_rank=[1, 2, 3])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_no_intermediate_rank_2_pos(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=False,
                                      expected_pos_set=[0, 2], expected_rank=[1, 3])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_no_intermediate_rank_2_pos_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=True,
                                      expected_pos_set=[0, 2], expected_rank=[1, 3])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_top_ranks_2_pos(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=False,
                                      expected_pos_set=[0, 2], expected_rank=[1, 2])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_top_ranks_2_pos_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=True,
                                      expected_pos_set=[0, 2], expected_rank=[1, 2])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_single_rank_2_pos(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=False,
                                      expected_pos_set=[0, 2], expected_rank=[1])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_single_rank_2_pos_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=True,
                                      expected_pos_set=[0, 2], expected_rank=[1])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_full_ranks_1_pos(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=False,
                                      expected_pos_set=[0], expected_rank=[1, 2, 3])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_full_ranks_1_pos_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=True,
                                      expected_pos_set=[0], expected_rank=[1, 2, 3])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_no_intermediate_rank_1_pos(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=False,
                                      expected_pos_set=[0], expected_rank=[1, 3])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_no_intermediate_rank_1_pos_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=True,
                                      expected_pos_set=[0], expected_rank=[1, 3])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_top_ranks_1_pos(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=False,
                                      expected_pos_set=[0], expected_rank=[1, 2])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_top_ranks_1_pos_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=True,
                                      expected_pos_set=[0], expected_rank=[1, 2])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_single_rank_1_pos(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=False,
                                      expected_pos_set=[0], expected_rank=[1])
        os.remove(protein_aln.file_name)

    def test_visualize_trace_single_rank_1_pos_low_mem(self):
        test_dir = os.path.join(os.getcwd(), 'test_case')
        protein_aln.write_out_alignment(protein_aln.file_name)
        self.evaluate_visualize_trace(out_dir=test_dir, p_id='seq1', fa_aln=protein_aln, low_mem=True,
                                      expected_pos_set=[0], expected_rank=[1])
        os.remove(protein_aln.file_name)


class TestEvolutionaryTraceGetVarPool(TestCase):

    def evaluate_get_var_pool(self, freq_table, mm_bool):
        init_var_pool(aln=protein_aln, frequency_table=freq_table)
        for pos in freq_table.get_positions():
            curr_pos, curr_query, curr_chars, curr_char_count = get_var_pool(pos=(pos, ) if type(pos) == int else pos)
            if type(pos) == int:
                self.assertEqual(curr_pos, (pos, ))
                self.assertEqual(curr_query, (protein_aln.query_sequence[pos], ))
            else:
                self.assertEqual(curr_pos, pos)
                self.assertEqual(curr_query, (protein_aln.query_sequence[pos[0]], protein_aln.query_sequence[pos[1]]))
            characters = set()
            for s_i in range(protein_aln.size):
                if mm_bool:
                    for s_j in range(s_i + 1, protein_aln.size):
                        if type(pos) == int:
                            curr_char = protein_aln.alignment[s_i, pos] + protein_aln.alignment[s_j, pos]
                        else:
                            curr_char = ((protein_aln.alignment[s_i, pos[0]] + protein_aln.alignment[s_i, pos[1]]) +
                                         (protein_aln.alignment[s_j, pos[0]] + protein_aln.alignment[s_j, pos[1]]))
                        if curr_char not in characters:
                            characters.add(curr_char)
                else:
                    if type(pos) == int:
                        curr_char = protein_aln.alignment[s_i, pos]
                    else:
                        curr_char = protein_aln.alignment[s_i, pos[0]] + protein_aln.alignment[s_i, pos[1]]
                    if curr_char not in characters:
                        characters.add(curr_char)
            self.assertEqual(characters, set(curr_chars.split(',')))
            self.assertEqual(curr_char_count, len(characters))

    def test_get_var_pool_single_pos(self):
        self.evaluate_get_var_pool(freq_table=pro_single_ft, mm_bool=False)

    def test_get_var_pool_double_pos(self):
        self.evaluate_get_var_pool(freq_table=pro_pair_ft, mm_bool=False)

    def test_get_var_pool_match_mismatch_single_pos(self):
        protein_mm_freq_tables_small = {'match': FrequencyTable(alphabet_size=pro_pair_alpha_size, mapping=pro_pair_map,
                                                                reverse_mapping=pro_pair_rev, seq_len=6, pos_size=1)}
        protein_mm_freq_tables_small['match'].mapping = pro_pair_map
        protein_mm_freq_tables_small['match'].set_depth(3)
        protein_mm_freq_tables_small['mismatch'] = deepcopy(protein_mm_freq_tables_small['match'])
        for pos in protein_mm_freq_tables_small['match'].get_positions():
            char_dict = {'match': {}, 'mismatch': {}}
            for i in range(3):
                for j in range(i + 1, 3):
                    status, stat_char = protein_mm_table.get_status_and_character(pos=pos, seq_ind1=i, seq_ind2=j)
                    if stat_char not in char_dict[status]:
                        char_dict[status][stat_char] = 0
                    char_dict[status][stat_char] += 1
            for m in char_dict:
                for curr_char in char_dict[m]:
                    protein_mm_freq_tables_small[m]._increment_count(pos=pos, char=curr_char,
                                                                     amount=char_dict[m][curr_char])
        for m in ['match', 'mismatch']:
            protein_mm_freq_tables_small[m].finalize_table()
        self.evaluate_get_var_pool(freq_table=(protein_mm_freq_tables_small['match'] +
                                               protein_mm_freq_tables_small['mismatch']),
                                   mm_bool=True)

    def test_get_var_pool_match_mismatch_double_double_pos(self):
        self.evaluate_get_var_pool(freq_table=protein_mm_freq_tables['match'] + protein_mm_freq_tables['mismatch'],
                                   mm_bool=True)

# class TestEvoultionaryTrace(TestBase):
#
#     @classmethod
#     def setUpClass(cls):
#         super(TestEvoultionaryTrace, cls).setUpClass()
#         cls.small_fa_fn = cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln']
#         cls.query_aln_fa_small = SeqAlignment(
#             file_name=cls.small_fa_fn, query_id=cls.small_structure_id)
#         cls.query_aln_fa_small.import_alignment()
#         cls.phylo_tree_small = PhylogeneticTree()
#         calc = AlignmentDistanceCalculator()
#         cls.phylo_tree_small.construct_tree(dm=calc.get_distance(cls.query_aln_fa_small.alignment))
#         cls.assignments_small = cls.phylo_tree_small.assign_group_rank()
#         cls.assignments_custom_small = cls.phylo_tree_small.assign_group_rank(ranks=[1, 2, 3, 5, 7, 10, 25])
#         cls.large_fa_fn = cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln']
#         cls.query_aln_fa_large = SeqAlignment(file_name=cls.large_fa_fn, query_id=cls.large_structure_id)
#         cls.query_aln_fa_large.import_alignment()
#         cls.phylo_tree_large = PhylogeneticTree()
#         calc = AlignmentDistanceCalculator()
#         cls.phylo_tree_large.construct_tree(dm=calc.get_distance(cls.query_aln_fa_large.alignment))
#         cls.assignments_large = cls.phylo_tree_large.assign_group_rank()
#         cls.assignments_custom_large = cls.phylo_tree_large.assign_group_rank(ranks=[1, 2, 3, 5, 7, 10, 25])
#         cls.query_aln_msf_small = deepcopy(cls.query_aln_fa_small)
#         cls.query_aln_msf_small.file_name = cls.data_set.protein_data[cls.small_structure_id]['Final_MSF_Aln']
#         cls.out_small_dir = os.path.join(cls.testing_dir, cls.small_structure_id)
#         if not os.path.isdir(cls.out_small_dir):
#             os.makedirs(cls.out_small_dir)
#         cls.query_aln_msf_large = deepcopy(cls.query_aln_fa_large)
#         cls.query_aln_msf_large.file_name = cls.data_set.protein_data[cls.large_structure_id]['Final_MSF_Aln']
#         cls.out_large_dir = os.path.join(cls.testing_dir, cls.large_structure_id)
#         if not os.path.isdir(cls.out_large_dir):
#             os.makedirs(cls.out_large_dir)
#         cls.query_aln_fa_small = cls.query_aln_fa_small.remove_gaps()
#         cls.query_aln_fa_large = cls.query_aln_fa_large.remove_gaps()
#         cls.single_alphabet = Gapped(FullIUPACProtein())
#         cls.single_size, _, cls.single_mapping, cls.single_reverse = build_mapping(alphabet=cls.single_alphabet)
#         cls.pair_alphabet = MultiPositionAlphabet(alphabet=cls.single_alphabet, size=2)
#         cls.pair_size, _, cls.pair_mapping, cls.pair_reverse = build_mapping(alphabet=cls.pair_alphabet)
#         cls.single_to_pair = {}
#         for char in cls.pair_mapping:
#             key = (cls.single_mapping[char[0]], cls.single_mapping[char[1]])
#             cls.single_to_pair[key] = cls.pair_mapping[char]
#
#     def evaluate_integer_et_comparison(self, p_id, fa_aln, low_mem):
#         out_dir = os.path.join(self.testing_dir, p_id)
#         rmtree(out_dir, ignore_errors=True)
#         os.makedirs(out_dir)
#         et_mip_obj = ETMIPWrapper(query=p_id, aln_file=fa_aln.file_name, out_dir=out_dir)
#         et_mip_obj.convert_alignment()
#         et_mip_obj.calculate_scores(method='intET', delete_files=False)
#         et = EvolutionaryTrace(query=p_id, polymer_type='Protein', aln_file=fa_aln.file_name, et_distance=True,
#                                distance_model='blosum62', tree_building_method='custom',
#                                tree_building_options={'tree_path': os.path.join(out_dir, 'etc_out_intET.nhx')},
#                                ranks=None, position_type='single', scoring_metric='identity', gap_correction=None,
#                                out_dir=out_dir, processors=self.max_threads, low_memory=low_mem,
#                                output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})
#         et.calculate_scores()
#         diff_ranks = et.scores - et_mip_obj.scores
#         if diff_ranks.any():
#             print(p_id)
#             print(et.scores)
#             print(et_mip_obj.scores)
#             print(diff_ranks)
#             indices = np.nonzero(diff_ranks)
#             print(et.scores[indices])
#             print(et_mip_obj.scores[indices])
#             print(diff_ranks[indices])
#         self.assertFalse(diff_ranks.any())
#         diff_coverage = et.coverages - et_mip_obj.coverages
#         not_passing = np.abs(diff_coverage) > 1e-2
#         if not_passing.any():
#             print(et.coverages)
#             print(et_mip_obj.coverage)
#             print(diff_coverage)
#             indices = np.nonzero(diff_coverage)
#             print(et.coverages[indices])
#             print(et_mip_obj.coverage[indices])
#             print(diff_coverage[indices])
#         self.assertFalse(not_passing.any())
#         rounded_coverages = np.round(et.coverages, decimals=3)
#         diff_coverages2 = rounded_coverages - et_mip_obj.coverages
#         not_passing2 = diff_coverages2 > 1E-15
#         if not_passing2.any():
#             print(rounded_coverages)
#             print(et_mip_obj.coverages)
#             print(diff_coverages2)
#             indices = np.nonzero(not_passing2)
#             print(rounded_coverages[indices])
#             print(et_mip_obj.coverages[indices])
#             print(diff_coverages2[indices])
#         self.assertFalse(not_passing2.any())
#         rmtree(out_dir)
#
#     def test_5a_trace(self):
#         # Compare the results of identity trace over single positions between this implementation and the WETC
#         # implementation for the small alignment.
#         self.evaluate_integer_et_comparison(p_id=self.small_structure_id, fa_aln=self.query_aln_fa_small, low_mem=False)
#
#     def test_5b_trace(self):
#         # Compare the results of identity trace over single positions between this implementation and the WETC
#         # implementation for the large alignment.
#         self.evaluate_integer_et_comparison(p_id=self.large_structure_id, fa_aln=self.query_aln_fa_large, low_mem=True)
#
#     def evaluate_real_value_et_comparison(self, p_id, fa_aln, low_mem):
#         out_dir = os.path.join(self.testing_dir, p_id,)
#         rmtree(out_dir, ignore_errors=True)
#         os.makedirs(out_dir)
#         et_mip_obj = ETMIPWrapper(query=p_id, aln_file=fa_aln.file_name, out_dir=out_dir)
#         et_mip_obj.convert_alignment()
#         et_mip_obj.calculate_scores(method='rvET', delete_files=False)
#         et = EvolutionaryTrace(query=p_id, polymer_type='Protein', aln_file=fa_aln.file_name, et_distance=True,
#                                distance_model='blosum62', tree_building_method='custom',
#                                tree_building_options={'tree_path': os.path.join(out_dir, 'etc_out_rvET.nhx')},
#                                ranks=None, position_type='single', scoring_metric='plain_entropy', gap_correction=0.6,
#                                out_dir=out_dir, processors=self.max_threads, low_memory=low_mem,
#                                output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})
#         et.calculate_scores()
#         diff_ranks = et.scores - et_mip_obj.scores
#         not_passing = np.abs(diff_ranks) > 1e-2
#         if not_passing.any():
#             print(et.scores)
#             print(et_mip_obj.scores)
#             print(diff_ranks)
#             indices = np.nonzero(diff_ranks)
#             print(et.scores[indices])
#             print(et_mip_obj.scores[indices])
#             print(diff_ranks[indices])
#         self.assertFalse(not_passing.any())
#         rounded_entropies = np.round(et.scores, decimals=2)
#         diff_ranks2 = rounded_entropies - et_mip_obj.scores
#         if diff_ranks2.any():
#             print(rounded_entropies)
#             print(et_mip_obj.scores)
#             print(diff_ranks2)
#             indices = np.nonzero(diff_ranks2)
#             print(rounded_entropies[indices])
#             print(et_mip_obj.scores[indices])
#             print(diff_ranks2[indices])
#         self.assertFalse(diff_ranks2.any())
#         diff_coverage = et.coverages - et_mip_obj.coverages
#         not_passing = np.abs(diff_coverage) > 1e-2
#         if not_passing.any():
#             print(et.coverages)
#             print(et_mip_obj.coverage)
#             print(diff_coverage)
#             indices = np.nonzero(diff_coverage)
#             print(et.coverages[indices])
#             print(et_mip_obj.coverage[indices])
#             print(diff_coverage[indices])
#         self.assertFalse(not_passing.any())
#         rounded_coverages = np.round(et.coverages, decimals=3)
#         diff_coverages2 = rounded_coverages - et_mip_obj.coverages
#         not_passing2 = diff_coverages2 > 1E-15
#         if not_passing2.any():
#             print(rounded_coverages)
#             print(et_mip_obj.coverage)
#             print(diff_coverages2)
#             indices = np.nonzero(not_passing2)
#             print(rounded_coverages[indices])
#             print(et_mip_obj.coverage[indices])
#             print(diff_coverages2[indices])
#         self.assertFalse(not_passing2.any())
#         rmtree(out_dir)
#
#     def test_5c_trace(self):
#         # Compare the results of plain entropy trace over single positions between this implementation and the WETC
#         # implementation for the small alignment.
#         self.evaluate_real_value_et_comparison(p_id=self.small_structure_id, fa_aln=self.query_aln_fa_small,
#                                                low_mem=False)
#
#     def test_5d_trace(self):
#         # Compare the results of identity trace over single positions between this implementation and the WETC
#         # implementation for the large alignment.
#         self.evaluate_real_value_et_comparison(p_id=self.large_structure_id, fa_aln=self.query_aln_fa_large,
#                                                low_mem=True)
#
#     def evaluate_mip_et_comparison(self, p_id, fa_aln, low_mem):
#         out_dir = os.path.join(self.testing_dir, p_id)
#         rmtree(out_dir, ignore_errors=True)
#         os.makedirs(out_dir)
#         filtered_fa_fn = os.path.join(out_dir, '{}_filtered_aln.fa'.format(p_id))
#         if os.path.isfile(filtered_fa_fn):
#             char_filtered_fa_aln = SeqAlignment(file_name=filtered_fa_fn, query_id=p_id)
#             char_filtered_fa_aln.import_alignment()
#         else:
#             curr_fa_aln = SeqAlignment(file_name=fa_aln.file_name, query_id=p_id)
#             curr_fa_aln.import_alignment()
#             curr_fa_aln.alphabet = Gapped(IUPACProtein())
#             char_filtered_fa_aln = curr_fa_aln.remove_bad_sequences()
#             char_filtered_fa_aln.write_out_alignment(file_name=filtered_fa_fn)
#             char_filtered_fa_aln.file_name = filtered_fa_fn
#         et_mip_obj = ETMIPWrapper(query=p_id, aln_file=filtered_fa_fn, out_dir=out_dir)
#         et_mip_obj.convert_alignment()
#         et_mip_obj.calculate_scores(method='ET-MIp', delete_files=False)
#         et = EvolutionaryTrace(query=p_id, polymer_type='Protein', aln_file=filtered_fa_fn, et_distance=True,
#                                distance_model='blosum62', tree_building_method='custom',
#                                tree_building_options={'tree_path': os.path.join(out_dir, 'etc_out_ET-MIp.nhx')},
#                                ranks=None, position_type='pair', gap_correction=None, out_dir=out_dir,
#                                processors=self.max_threads, low_memory=low_mem,
#                                output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'},
#                                scoring_metric='filtered_average_product_corrected_mutual_information')
#         et.calculate_scores()
#         diff_ranks = et.scores - et_mip_obj.scores
#         not_passing = np.abs(diff_ranks) > 1e-3
#         if not_passing.any():
#             print(et.scores)
#             print(et_mip_obj.scores)
#             print(diff_ranks)
#             indices = np.nonzero(not_passing)
#             print(et.scores[indices])
#             print(et_mip_obj.scores[indices])
#             print(diff_ranks[indices])
#             print(et.scores[indices][0])
#             print(et_mip_obj.scores[indices][0])
#             print(diff_ranks[indices][0])
#         self.assertFalse(not_passing.any())
#         rounded_scores = np.round(et.scores, decimals=3)
#         diff_ranks2 = rounded_scores - et_mip_obj.scores
#         not_passing_rounded = np.abs(diff_ranks2) > 1e-15
#         if not_passing_rounded.any():
#             print(rounded_scores)
#             print(et_mip_obj.scores)
#             print(diff_ranks2)
#             indices = np.nonzero(not_passing_rounded)
#             print(rounded_scores[indices])
#             print(et_mip_obj.scores[indices])
#             print(diff_ranks2[indices])
#         self.assertFalse(not_passing_rounded.any())
#         diff_coverages = et.coverages - et_mip_obj.coverages
#         not_passing = np.abs(diff_coverages) > 1E-3
#         if not_passing.any():
#             print(et.coverages)
#             print(et_mip_obj.coverages)
#             print(diff_coverages)
#             indices = np.nonzero(not_passing)
#             for i in range(len(indices[0])):
#                 print(indices[0][i], indices[1][i], et_mip_obj.coverages[indices[0][i], indices[1][i]],
#                       et.coverages[indices[0][i], indices[1][i]], diff_coverages[indices[0][i], indices[1][i]],
#                       1e-2, np.abs(diff_coverages[indices[0][i], indices[1][i]]) > 1e-2)
#             print(et.scores[indices])
#             print(et.rankings[indices])
#             print(np.sum(not_passing))
#             print(np.nonzero(not_passing))
#             self.assertLessEqual(np.sum(not_passing), np.ceil(0.01 * np.sum(range(fa_aln.seq_length - 1))))
#         else:
#             self.assertFalse(not_passing.any())
#         rmtree(out_dir)
#
#     def test_5e_trace(self):
#         # Compare the results of average product corrected mutual information over pairs of positions between this
#         # implementation and the WETC implementation for the small alignment.
#         self.evaluate_mip_et_comparison(p_id=self.small_structure_id, fa_aln=self.query_aln_fa_small, low_mem=False)
#
#     def test_5f_trace(self):
#         # Compare the results of average product corrected mutual information over pairs of positions between this
#         # implementation and the WETC implementation for the large alignment.
#         self.evaluate_mip_et_comparison(p_id=self.large_structure_id, fa_aln=self.query_aln_fa_large, low_mem=True)


if __name__ == '__main__':
    unittest.main()
