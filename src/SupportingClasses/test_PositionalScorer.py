"""
Created on July 12, 2019

@author: Daniel Konecki
"""
import unittest
import numpy as np
from copy import deepcopy
from Bio.Alphabet import Gapped
from test_Base import TestBase
from utils import build_mapping
from SeqAlignment import SeqAlignment
from FrequencyTable import FrequencyTable
from PhylogeneticTree import PhylogeneticTree
from MatchMismatchTable import MatchMismatchTable
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from EvolutionaryTraceAlphabet import FullIUPACProtein, MultiPositionAlphabet
from PositionalScorer import (PositionalScorer, rank_integer_value_score, rank_real_value_score, group_identity_score,
                              group_plain_entropy_score, mutual_information_computation, group_mutual_information_score,
                              group_normalized_mutual_information_score, average_product_correction,
                              filtered_average_product_correction, ratio_computation, angle_computation,
                              diversity_computation, group_match_mismatch_entropy_ratio,
                              group_match_mismatch_entropy_angle, group_match_diversity_mismatch_entropy_ratio,
                              group_match_diversity_mismatch_entropy_angle)


class TestPositionalScorer(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestPositionalScorer, cls).setUpClass()

        cls.single_alphabet = Gapped(FullIUPACProtein())
        cls.single_size, _, cls.single_mapping, cls.single_reverse = build_mapping(alphabet=cls.single_alphabet)
        cls.pair_alphabet = MultiPositionAlphabet(alphabet=cls.single_alphabet, size=2)
        cls.pair_size, _, cls.pair_mapping, cls.pair_reverse = build_mapping(alphabet=cls.pair_alphabet)
        cls.single_to_pair = {}
        for char in cls.pair_mapping:
            key = (cls.single_mapping[char[0]], cls.single_mapping[char[1]])
            cls.single_to_pair[key] = cls.pair_mapping[char]
        cls.quad_alphabet = MultiPositionAlphabet(alphabet=cls.single_alphabet, size=4)
        cls.quad_size, _, cls.quad_mapping, cls.quad_reverse = build_mapping(alphabet=cls.quad_alphabet)
        cls.single_to_quad = {}
        for char in cls.quad_mapping:
            key = (cls.single_mapping[char[0]], cls.single_mapping[char[1]], cls.single_mapping[char[2]],
                   cls.single_mapping[char[3]])
            cls.single_to_quad[key] = cls.quad_mapping[char]

        cls.query_aln_fa_small = SeqAlignment(
            file_name=cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln'],
            query_id=cls.small_structure_id)
        cls.query_aln_fa_small.import_alignment()
        cls.phylo_tree_small = PhylogeneticTree()
        calc = AlignmentDistanceCalculator()
        cls.phylo_tree_small.construct_tree(dm=calc.get_distance(cls.query_aln_fa_small.alignment))
        cls.query_aln_fa_small = cls.query_aln_fa_small.remove_gaps()
        cls.terminals = {x.name: {'aln': cls.query_aln_fa_small.generate_sub_alignment(sequence_ids=[x.name]),
                                  'node': x} for x in cls.phylo_tree_small.tree.get_terminals()}
        for x in cls.terminals:
            single, pair = cls.terminals[x]['aln'].characterize_positions(
                single=True, pair=True, single_size=cls.single_size, single_mapping=cls.single_mapping,
                single_reverse=cls.single_reverse, pair_size=cls.pair_size, pair_mapping=cls.pair_mapping,
                pair_reverse=cls.pair_reverse)
            cls.terminals[x]['single'] = single
            cls.terminals[x]['pair'] = pair
        potential_parents = set()
        for t in cls.terminals:
            path = cls.phylo_tree_small.tree.get_path(t)
            if len(path) >= 2:
                potential_parents.add(path[-2])
        cls.first_parents = {}
        for parent in potential_parents:
            if parent.clades[0].is_terminal() and parent.clades[1].is_terminal():
                cls.first_parents[parent.name] = {'node': parent, 'aln': cls.query_aln_fa_small.generate_sub_alignment(
                    sequence_ids=[y.name for y in parent.clades])}
                cls.first_parents[parent.name]['single'] = (cls.terminals[parent.clades[0].name]['single'] +
                                                            cls.terminals[parent.clades[1].name]['single'])
                cls.first_parents[parent.name]['pair'] = (cls.terminals[parent.clades[0].name]['pair'] +
                                                          cls.terminals[parent.clades[1].name]['pair'])
        cls.seq_len = cls.query_aln_fa_small.seq_length
        cls.mm_table = MatchMismatchTable(seq_len=cls.query_aln_fa_small.seq_length,
                                          num_aln=cls.query_aln_fa_small._alignment_to_num(cls.single_mapping),
                                          single_alphabet_size=cls.single_size, single_mapping=cls.single_mapping,
                                          single_reverse_mapping=cls.single_reverse, larger_alphabet_size=cls.quad_size,
                                          larger_alphabet_mapping=cls.quad_mapping,
                                          larger_alphabet_reverse_mapping=cls.quad_reverse,
                                          single_to_larger_mapping=cls.single_to_quad, pos_size=2)
        cls.mm_table.identify_matches_mismatches()
        for x in cls.terminals:
            cls.terminals[x]['match'] = FrequencyTable(alphabet_size=cls.quad_size, mapping=cls.quad_mapping,
                                                       reverse_mapping=cls.quad_reverse,
                                                       seq_len=cls.query_aln_fa_small.seq_length, pos_size=2)
            cls.terminals[x]['match'].mapping = cls.quad_mapping
            cls.terminals[x]['match'].set_depth(1)
            cls.terminals[x]['mismatch'] = deepcopy(cls.terminals[x]['match'])
            for pos in cls.terminals[x]['match'].get_positions():
                char_dict = {'match': {}, 'mismatch': {}}
                for i in range(cls.terminals[x]['aln'].size):
                    s1 = cls.query_aln_fa_small.seq_order.index(cls.terminals[x]['aln'].seq_order[i])
                    for j in range(i + 1, cls.terminals[x]['aln'].size):
                        s2 = cls.query_aln_fa_small.seq_order.index(cls.terminals[x]['aln'].seq_order[j])
                        status, char = cls.mm_table.get_status_and_character(pos=pos, seq_ind1=s1, seq_ind2=s2)
                        if char not in char_dict[status]:
                            char_dict[status][char] = 0
                        char_dict[status][char] += 1
                for m in char_dict:
                    for char in char_dict[m]:
                        cls.terminals[x][m]._increment_count(pos=pos, char=char, amount=char_dict[m][char])
            for m in ['match', 'mismatch']:
                cls.terminals[x][m].finalize_table()
        for x in cls.first_parents:
            cls.first_parents[x]['match'] = FrequencyTable(alphabet_size=cls.quad_size, mapping=cls.quad_mapping,
                                                           reverse_mapping=cls.quad_reverse,
                                                           seq_len=cls.query_aln_fa_small.seq_length, pos_size=2)
            cls.first_parents[x]['match'].mapping = cls.quad_mapping
            cls.first_parents[x]['match'].set_depth(((cls.first_parents[x]['aln'].size**2) -
                                                     cls.first_parents[x]['aln'].size) / 2.0)
            cls.first_parents[x]['mismatch'] = deepcopy(cls.first_parents[x]['match'])
            for pos in cls.first_parents[x]['match'].get_positions():
                char_dict = {'match': {}, 'mismatch': {}}
                for i in range(cls.first_parents[x]['aln'].size):
                    s1 = cls.query_aln_fa_small.seq_order.index(cls.first_parents[x]['aln'].seq_order[i])
                    for j in range(i + 1, cls.first_parents[x]['aln'].size):
                        s2 = cls.query_aln_fa_small.seq_order.index(cls.first_parents[x]['aln'].seq_order[j])
                        status, char = cls.mm_table.get_status_and_character(pos=pos, seq_ind1=s1, seq_ind2=s2)
                        if char not in char_dict[status]:
                            char_dict[status][char] = 0
                        char_dict[status][char] += 1
                for m in char_dict:
                    for char in char_dict[m]:
                        cls.first_parents[x][m]._increment_count(pos=pos, char=char, amount=char_dict[m][char])
            for m in ['match', 'mismatch']:
                cls.first_parents[x][m].finalize_table()

    def evaluate__init(self, seq_len, pos_size, metric, metric_type, rank_type):
        if metric == 'fake':
            with self.assertRaises(ValueError):
                PositionalScorer(seq_length=seq_len, pos_size=pos_size, metric=metric)
        else:
            pos_scorer = PositionalScorer(seq_length=seq_len, pos_size=pos_size, metric=metric)
            self.assertEqual(pos_scorer.sequence_length, seq_len)
            self.assertEqual(pos_scorer.position_size, pos_size)
            if pos_size == 1:
                self.assertEqual(pos_scorer.dimensions, (seq_len,))
            else:
                self.assertEqual(pos_scorer.dimensions, (seq_len, seq_len))
            self.assertEqual(pos_scorer.metric, metric)
            self.assertEqual(pos_scorer.metric_type, metric_type)
            self.assertEqual(pos_scorer.rank_type, rank_type)

    def test1a_init(self):
        self.evaluate__init(seq_len=self.seq_len, pos_size=1, metric='identity', metric_type='integer', rank_type='min')

    def test1b_init(self):
        self.evaluate__init(seq_len=self.seq_len, pos_size=2, metric='identity', metric_type='integer', rank_type='min')

    def test1c_init(self):
        self.evaluate__init(seq_len=self.seq_len, pos_size=1, metric='fake', metric_type='integer', rank_type='min')

    def test1d_init(self):
        self.evaluate__init(seq_len=self.seq_len, pos_size=2, metric='fake', metric_type='integer', rank_type='min')

    def test1e_init(self):
        self.evaluate__init(seq_len=self.seq_len, pos_size=1, metric='plain_entropy', metric_type='real',
                            rank_type='min')

    def test1f_init(self):
        self.evaluate__init(seq_len=self.seq_len, pos_size=2, metric='plain_entropy', metric_type='real',
                            rank_type='min')

    def test1g_init(self):
        self.evaluate__init(seq_len=self.seq_len, pos_size=2, metric='mutual_information', metric_type='real',
                            rank_type='max')

    def test1h_init(self):
        self.evaluate__init(seq_len=self.seq_len, pos_size=2, metric='normalized_mutual_information',
                            metric_type='real', rank_type='max')

    def test1i_init(self):
        self.evaluate__init(seq_len=self.seq_len, pos_size=2, metric='average_product_corrected_mutual_information',
                            metric_type='real', rank_type='max')

    def test1j_init(self):
        self.evaluate__init(seq_len=self.seq_len, pos_size=2,
                            metric='filtered_average_product_corrected_mutual_information',
                            metric_type='real', rank_type='max')

    def test1k_init(self):
        self.evaluate__init(seq_len=self.seq_len, pos_size=2, metric='match_mismatch_entropy_angle', metric_type='real',
                            rank_type='min')

    def evaluate_score_group_ambiguous(self, node_dict, metric):
        pos_scorer_single = PositionalScorer(seq_length=self.seq_len, pos_size=1, metric=metric)
        dim_single = (self.seq_len,)
        pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric=metric)
        dim_pair = (self.seq_len, self.seq_len)
        for x in node_dict:
            group_single_scores = pos_scorer_single.score_group(freq_table=node_dict[x]['single'])
            group_pair_scores = pos_scorer_pair.score_group(freq_table=node_dict[x]['pair'])
            if metric == 'identity':
                single_scores = group_identity_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
            elif metric == 'plain_entropy':
                single_scores = group_plain_entropy_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
            else:
                raise ValueError('Cannot test metric: {} in evaluate_score_group_ambiguous'.format(metric))
            if metric == 'identity':
                pair_scores = group_identity_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
            elif metric == 'plain_entropy':
                pair_scores = group_plain_entropy_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
            else:
                raise ValueError('Cannot test metric: {} in evaluate_score_group_ambiguous'.format(metric))
            for i in range(node_dict[x]['aln'].seq_length):
                self.assertEqual(group_single_scores[i], single_scores[i])
                for j in range(i, node_dict[x]['aln'].seq_length):
                    self.assertEqual(group_pair_scores[i, j], pair_scores[i, j])

    def test2a_score_group(self):
        # Score group using identity metric terminals
        self.evaluate_score_group_ambiguous(node_dict=self.terminals, metric='identity')

    def test2b_score_group(self):
        # Score group using identity metric parents
        self.evaluate_score_group_ambiguous(node_dict=self.first_parents, metric='identity')

    def test2c_score_group(self):
        # Score group using plain entropy metric terminals
        self.evaluate_score_group_ambiguous(node_dict=self.terminals, metric='plain_entropy')

    def test2d_score_group(self):
        # Score group using plain entropy metric parents
        self.evaluate_score_group_ambiguous(node_dict=self.first_parents, metric='plain_entropy')

    def evaluate_score_group_mi(self, node_dict, metric):
        pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric=metric)
        dim_pair = (self.seq_len, self.seq_len)
        for x in node_dict:
            group_pair_scores = pos_scorer_pair.score_group(freq_table=node_dict[x]['pair'])
            if metric == 'mutual_information':
                pair_scores = group_mutual_information_score(freq_table=node_dict[x]['pair'],
                                                             dimensions=dim_pair)
            elif metric == 'normalized_mutual_information':
                pair_scores = group_normalized_mutual_information_score(freq_table=node_dict[x]['pair'],
                                                                        dimensions=dim_pair)
            else:
                raise ValueError('Cannot test metric: {} in evaluate_score_group_mi'.format(metric))
            for i in range(node_dict[x]['aln'].seq_length):
                for j in range(i, node_dict[x]['aln'].seq_length):
                    self.assertEqual(group_pair_scores[i, j], pair_scores[i, j])

    def test2e_score_group(self):
        # Score group using mutual information metric terminals
        self.evaluate_score_group_mi(node_dict=self.terminals, metric='mutual_information')

    def test2f_score_group(self):
        # Score group using mutual information metric parents
        self.evaluate_score_group_mi(node_dict=self.first_parents, metric='mutual_information')

    def test2g_score_group(self):
        # Score group using normalized mutual information metric terminals
        self.evaluate_score_group_mi(node_dict=self.terminals, metric='normalized_mutual_information')

    def test2h_score_group(self):
        # Score group using normalized mutual information metric parents
        self.evaluate_score_group_mi(node_dict=self.first_parents, metric='normalized_mutual_information')

    def evaluate_score_group_average_product_corrected_mi(self, node_dict):
        pos_scorer = PositionalScorer(seq_length=self.seq_len, pos_size=2,
                                      metric='average_product_corrected_mutual_information')
        for x in node_dict:
            freq_table = node_dict[x]['pair']
            pair_dims = (self.seq_len, self.seq_len)
            mi_matrix = group_mutual_information_score(freq_table=freq_table, dimensions=pair_dims)
            column_sums = {}
            column_counts = {}
            matrix_sums = 0.0
            total_count = 0.0
            for pos in freq_table.get_positions():
                if pos[0] == pos[1]:
                    continue
                if pos[0] not in column_sums:
                    column_sums[pos[0]] = 0.0
                    column_counts[pos[0]] = 0.0
                if pos[1] not in column_sums:
                    column_sums[pos[1]] = 0.0
                    column_counts[pos[1]] = 0.0
                mi = mi_matrix[pos[0]][pos[1]]
                column_sums[pos[0]] += mi
                column_sums[pos[1]] += mi
                column_counts[pos[0]] += 1
                column_counts[pos[1]] += 1
                matrix_sums += mi
                total_count += 1
            expected_apc = np.zeros((self.seq_len, self.seq_len))
            if total_count == 0.0:
                matrix_average = 0.0
            else:
                matrix_average = matrix_sums / total_count
            if matrix_average != 0.0:
                column_averages = {}
                for key in column_sums:
                    column_averages[key] = column_sums[key] / column_counts[key]
                for pos in freq_table.get_positions():
                    if pos[0] == pos[1]:
                        continue
                    apc_numerator = column_averages[pos[0]] * column_averages[pos[1]]
                    apc_correction = apc_numerator / matrix_average
                    expected_apc[pos[0]][pos[1]] = mi_matrix[pos[0]][pos[1]] - apc_correction
            apc = pos_scorer.score_group(freq_table=freq_table)
            diff = apc - expected_apc
            not_passing = diff > 1E-13
            if not_passing.any():
                print(apc)
                print(expected_apc)
                print(diff)
                indices = np.nonzero(not_passing)
                print(apc[indices])
                print(expected_apc[indices])
                print(diff[indices])
            self.assertTrue(not not_passing.any())

    def test2i_score_group(self):
        # Score group using average product correction mutual information metric terminals
        self.evaluate_score_group_average_product_corrected_mi(node_dict=self.terminals)

    def test2j_score_group(self):
        # Score group using average product correction mutual information metric parents
        self.evaluate_score_group_average_product_corrected_mi(node_dict=self.first_parents)

    def evaluate_score_group_match_entropy_mismatch_entropy_ratio(self, node_dict):
        pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric='match_mismatch_entropy_ratio')
        dim_pair = (self.seq_len, self.seq_len)
        for x in node_dict:
            freq_table_dict = {'match': node_dict[x]['match'], 'mismatch': node_dict[x]['mismatch']}
            group_pair_scores = pos_scorer_pair.score_group(freq_table=freq_table_dict)
            pair_scores = group_match_mismatch_entropy_ratio(freq_tables=freq_table_dict, dimensions=dim_pair)
            for i in range(node_dict[x]['aln'].seq_length):
                for j in range(i, node_dict[x]['aln'].seq_length):
                    self.assertEqual(group_pair_scores[i, j], pair_scores[i, j])

    def test2k_score_group_angle(self):
        self.evaluate_score_group_match_entropy_mismatch_entropy_ratio(node_dict=self.terminals)

    def test2l_score_group_angle(self):
        self.evaluate_score_group_match_entropy_mismatch_entropy_ratio(node_dict=self.first_parents)

    def evaluate_score_group_match_entropy_mismatch_entropy_angle(self, node_dict):
        pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric='match_mismatch_entropy_angle')
        dim_pair = (self.seq_len, self.seq_len)
        for x in node_dict:
            freq_table_dict = {'match': node_dict[x]['match'], 'mismatch': node_dict[x]['mismatch']}
            group_pair_scores = pos_scorer_pair.score_group(freq_table=freq_table_dict)
            pair_scores = group_match_mismatch_entropy_angle(freq_tables=freq_table_dict, dimensions=dim_pair)
            for i in range(node_dict[x]['aln'].seq_length):
                for j in range(i, node_dict[x]['aln'].seq_length):
                    self.assertEqual(group_pair_scores[i, j], pair_scores[i, j])

    def test2m_score_group_angle(self):
        self.evaluate_score_group_match_entropy_mismatch_entropy_angle(node_dict=self.terminals)

    def test2n_score_group_angle(self):
        self.evaluate_score_group_match_entropy_mismatch_entropy_angle(node_dict=self.first_parents)

    def evaluate_score_group_match_diversity_mismatch_entropy_ratio(self, node_dict):
        pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2,
                                           metric='match_diversity_mismatch_entropy_ratio')
        dim_pair = (self.seq_len, self.seq_len)
        for x in node_dict:
            freq_table_dict = {'match': node_dict[x]['match'], 'mismatch': node_dict[x]['mismatch']}
            group_pair_scores = pos_scorer_pair.score_group(freq_table=freq_table_dict)
            pair_scores = group_match_diversity_mismatch_entropy_ratio(freq_tables=freq_table_dict, dimensions=dim_pair)
            for i in range(node_dict[x]['aln'].seq_length):
                for j in range(i, node_dict[x]['aln'].seq_length):
                    self.assertEqual(group_pair_scores[i, j], pair_scores[i, j])

    def test2o_score_group_angle(self):
        self.evaluate_score_group_match_entropy_mismatch_entropy_ratio(node_dict=self.terminals)

    def test2p_score_group_angle(self):
        self.evaluate_score_group_match_entropy_mismatch_entropy_ratio(node_dict=self.first_parents)

    def evaluate_score_group_match_diversity_mismatch_entropy_angle(self, node_dict):
        pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2,
                                           metric='match_diversity_mismatch_entropy_angle')
        dim_pair = (self.seq_len, self.seq_len)
        for x in node_dict:
            freq_table_dict = {'match': node_dict[x]['match'], 'mismatch': node_dict[x]['mismatch']}
            group_pair_scores = pos_scorer_pair.score_group(freq_table=freq_table_dict)
            pair_scores = group_match_diversity_mismatch_entropy_angle(freq_tables=freq_table_dict, dimensions=dim_pair)
            for i in range(node_dict[x]['aln'].seq_length):
                for j in range(i, node_dict[x]['aln'].seq_length):
                    self.assertEqual(group_pair_scores[i, j], pair_scores[i, j])

    def test2q_score_group_angle(self):
        self.evaluate_score_group_match_diversity_mismatch_entropy_angle(node_dict=self.terminals)

    def test2r_score_group_angle(self):
        self.evaluate_score_group_match_diversity_mismatch_entropy_angle(node_dict=self.first_parents)

    def evaluate_score_rank_ambiguous(self, node_dict, metric, single=True, pair=True):
        group_scores_single = []
        group_scores_pair = []
        if single:
            pos_scorer_single = PositionalScorer(seq_length=self.seq_len, pos_size=1, metric=metric)
        if pair:
            pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric=metric)
        for x in node_dict:
            if ('ratio' in metric) or ('angle' in metric):
                if single:
                    raise ValueError(f'{metric} not intended for single position measurement.')
                if pair:
                    group_score_pair = pos_scorer_pair.score_group(freq_table=node_dict[x])
                    group_scores_pair.append(group_score_pair)
            else:
                if single:
                    group_score_single = pos_scorer_single.score_group(freq_table=node_dict[x]['single'])
                    group_scores_single.append(group_score_single)
                if pair:
                    group_score_pair = pos_scorer_pair.score_group(freq_table=node_dict[x]['pair'])
                    group_scores_pair.append(group_score_pair)
        rank = max(len(group_scores_single), len(group_scores_pair))
        if single:
            group_scores_single = np.sum(np.stack(group_scores_single, axis=0), axis=0)
            if metric == 'identity':
                expected_rank_score_single = rank_integer_value_score(score_matrix=group_scores_single, rank=rank)
            else:
                expected_rank_score_single = rank_real_value_score(score_matrix=group_scores_single, rank=rank)
            rank_score_single = pos_scorer_single.score_rank(score_tensor=group_scores_single, rank=rank)
            diff_single = rank_score_single - expected_rank_score_single
            self.assertTrue(not diff_single.any())
        if pair:
            group_scores_pair = np.sum(np.stack(group_scores_pair, axis=0), axis=0)
            if metric == 'identity':
                expected_rank_score_pair = rank_integer_value_score(score_matrix=group_scores_pair, rank=rank)
            else:
                expected_rank_score_pair = rank_real_value_score(score_matrix=group_scores_pair, rank=rank)
            expected_rank_score_pair = np.triu(expected_rank_score_pair, k=1)
            rank_score_pair = pos_scorer_pair.score_rank(score_tensor=group_scores_pair, rank=rank)
            diff_pair = rank_score_pair - expected_rank_score_pair
            self.assertTrue(not diff_pair.any())

    def test3a_score_rank(self):
        # Score rank using identity metric terminals
        self.evaluate_score_rank_ambiguous(node_dict=self.terminals, metric='identity')

    def test3b_score_rank(self):
        # Score rank using identity metric parents
        self.evaluate_score_rank_ambiguous(node_dict=self.first_parents, metric='identity')

    def test3c_score_rank(self):
        # Score rank using plain entropy metric terminals
        self.evaluate_score_rank_ambiguous(node_dict=self.terminals, metric='plain_entropy')

    def test3d_score_rank(self):
        # Score rank using plain entropy metric parents
        self.evaluate_score_rank_ambiguous(node_dict=self.first_parents, metric='plain_entropy')

    def test3e_score_rank(self):
        # Score rank using mutual information metric terminals
        self.evaluate_score_rank_ambiguous(node_dict=self.terminals, metric='mutual_information', single=False)

    def test3f_score_rank(self):
        # Score rank using mutual information metric parents
        self.evaluate_score_rank_ambiguous(node_dict=self.first_parents, metric='mutual_information', single=False)

    def test3g_score_rank(self):
        # Score rank using normalized mutual information metric terminal
        self.evaluate_score_rank_ambiguous(node_dict=self.terminals, metric='normalized_mutual_information',
                                           single=False)

    def test3h_score_rank(self):
        # Score rank using normalized mutual information metric parents
        self.evaluate_score_rank_ambiguous(node_dict=self.first_parents, metric='normalized_mutual_information',
                                           single=False)

    def test3i_score_rank(self):
        # Score rank using average product corrected mutual information metric terminals
        self.evaluate_score_rank_ambiguous(node_dict=self.terminals,
                                           metric='average_product_corrected_mutual_information',
                                           single=False)

    def test3j_score_rank(self):
        # Score rank using average product corrected mutual information metric parents
        self.evaluate_score_rank_ambiguous(node_dict=self.first_parents,
                                           metric='average_product_corrected_mutual_information',
                                           single=False)

    def test3k_score_rank(self):
        # Score rank using average product corrected mutual information metric terminals
        self.evaluate_score_rank_ambiguous(node_dict=self.terminals,
                                           metric='filtered_average_product_corrected_mutual_information',
                                           single=False)

    def test3l_score_rank(self):
        # Score rank using average product corrected mutual information metric parents
        self.evaluate_score_rank_ambiguous(node_dict=self.first_parents,
                                           metric='filtered_average_product_corrected_mutual_information',
                                           single=False)

    def test3m_score_rank(self):
        # Score rank using average product corrected mutual information metric terminals
        self.evaluate_score_rank_ambiguous(node_dict=self.terminals,
                                           metric='match_mismatch_entropy_ratio',
                                           single=False)

    def test3n_score_rank(self):
        # Score rank using average product corrected mutual information metric parents
        self.evaluate_score_rank_ambiguous(node_dict=self.first_parents,
                                           metric='match_mismatch_entropy_ratio',
                                           single=False)

    def test3o_score_rank(self):
        # Score rank using average product corrected mutual information metric terminals
        self.evaluate_score_rank_ambiguous(node_dict=self.terminals,
                                           metric='match_mismatch_entropy_angle',
                                           single=False)

    def test3p_score_rank(self):
        # Score rank using average product corrected mutual information metric parents
        self.evaluate_score_rank_ambiguous(node_dict=self.first_parents,
                                           metric='match_mismatch_entropy_angle',
                                           single=False)

    def test3q_score_rank(self):
        # Score rank using average product corrected mutual information metric terminals
        self.evaluate_score_rank_ambiguous(node_dict=self.terminals,
                                           metric='match_diversity_mismatch_entropy_ratio',
                                           single=False)

    def test3r_score_rank(self):
        # Score rank using average product corrected mutual information metric parents
        self.evaluate_score_rank_ambiguous(node_dict=self.first_parents,
                                           metric='match_diversity_mismatch_entropy_ratio',
                                           single=False)

    def test3s_score_rank(self):
        # Score rank using average product corrected mutual information metric terminals
        self.evaluate_score_rank_ambiguous(node_dict=self.terminals,
                                           metric='match_diversity_mismatch_entropy_angle',
                                           single=False)

    def test3t_score_rank(self):
        # Score rank using average product corrected mutual information metric parents
        self.evaluate_score_rank_ambiguous(node_dict=self.first_parents,
                                           metric='match_diversity_mismatch_entropy_angle',
                                           single=False)

    def evaluate_rank_integer_value_score(self, node_dict):
        group_single_scores = []
        dim_single = (self.seq_len,)
        group_pair_scores = []
        dim_pair = (self.seq_len, self.seq_len)
        for x in node_dict:
            group_single_score = group_identity_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
            group_pair_score = group_identity_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
            group_single_scores.append(group_single_score)
            group_pair_scores.append(group_pair_score)
        rank = max(len(group_single_scores), len(group_pair_scores))
        group_single_scores = np.sum(np.stack(group_single_scores, axis=0), axis=0)
        rank_single_scores = rank_integer_value_score(score_matrix=group_single_scores, rank=rank)
        expected_rank_single_scores = np.zeros(dim_single)
        for i in range(node_dict[x]['aln'].seq_length):
            if group_single_scores[i] != 0:
                expected_rank_single_scores[i] = 1
        diff_single = rank_single_scores - expected_rank_single_scores
        self.assertTrue(not diff_single.any())
        group_pair_scores = np.sum(np.stack(group_pair_scores, axis=0), axis=0)
        rank_pair_scores = rank_integer_value_score(score_matrix=group_pair_scores, rank=rank)
        expected_rank_pair_scores = np.zeros((self.seq_len, self.seq_len))
        for i in range(node_dict[x]['aln'].seq_length):
            for j in range(i, node_dict[x]['aln'].seq_length):
                if group_pair_scores[i, j] != 0:
                    expected_rank_pair_scores[i, j] = 1
        diff_pair = rank_pair_scores - expected_rank_pair_scores
        self.assertTrue(not diff_pair.any())

    def test4a_rank_integer_value_score(self):
        # Metric=Identity, Alignment Size=1
        self.evaluate_rank_integer_value_score(node_dict=self.terminals)

    def test4b_rank_integer_value_score(self):
        # Metric=Identity, Alignment Size=2
        self.evaluate_rank_integer_value_score(node_dict=self.first_parents)

    def evaluate_rank_real_value_score(self, node_dict):
        group_single_scores = []
        dim_single = (self.seq_len,)
        group_pair_scores = []
        dim_pair = (self.seq_len, self.seq_len)
        for x in node_dict:
            group_single_score = group_identity_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
            group_pair_score = group_identity_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
            group_single_scores.append(group_single_score)
            group_pair_scores.append(group_pair_score)
        rank = max(len(group_single_scores), len(group_pair_scores))
        group_single_scores = np.sum(np.stack(group_single_scores, axis=0), axis=0)
        rank_single_scores = rank_real_value_score(score_matrix=group_single_scores, rank=rank)
        expected_rank_single_scores = np.zeros(self.seq_len)
        for i in range(node_dict[x]['aln'].seq_length):
            expected_rank_single_scores[i] += (1.0 / rank) * group_single_scores[i]
        diff_single = rank_single_scores - expected_rank_single_scores
        not_passing_single = diff_single > 1E-16
        self.assertTrue(not not_passing_single.any())
        group_pair_scores = np.sum(np.stack(group_pair_scores, axis=0), axis=0)
        rank_pair_scores = rank_real_value_score(score_matrix=group_pair_scores, rank=rank)
        expected_rank_pair_scores = np.zeros((self.seq_len, self.seq_len))
        for i in range(node_dict[x]['aln'].seq_length):
            for j in range(i, node_dict[x]['aln'].seq_length):
                expected_rank_pair_scores[i, j] += (1.0 / rank) * group_pair_scores[i, j]
        diff_pair = rank_pair_scores - expected_rank_pair_scores
        not_passing_pair = diff_pair > 1E-16
        self.assertTrue(not not_passing_pair.any())

    def test5a_rank_real_value_score(self):
        self.evaluate_rank_real_value_score(node_dict=self.terminals)

    def test5b_rank_real_value_score(self):
        self.evaluate_rank_real_value_score(node_dict=self.first_parents)

    def evaluate_group_identity_score(self, node_dict):
        for x in node_dict:
            dim_single = (self.seq_len,)
            single_scores = group_identity_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
            dim_pair = (self.seq_len, self.seq_len)
            pair_scores = group_identity_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
            for i in range(node_dict[x]['aln'].seq_length):
                expected_single_score = 0
                single_char = None
                for k in range(node_dict[x]['aln'].size):
                    curr_single_char = node_dict[x]['aln'].alignment[k, i]
                    if single_char is None:
                        single_char = curr_single_char
                    else:
                        if single_char != curr_single_char:
                            expected_single_score = 1
                self.assertEqual(single_scores[i], expected_single_score)
                for j in range(i, node_dict[x]['aln'].seq_length):
                    if i == j:
                        self.assertEqual(single_scores[i], pair_scores[i, j])
                        continue
                    expected_pair_score = 0
                    pair_char = None
                    for k in range(node_dict[x]['aln'].size):
                        curr_pair_char = node_dict[x]['aln'].alignment[k, i] + node_dict[x]['aln'].alignment[k, j]
                        if pair_char is None:
                            pair_char = curr_pair_char
                        else:
                            if pair_char != curr_pair_char:
                                expected_pair_score = 1
                    self.assertEqual(pair_scores[i, j], expected_pair_score)

    def test6a_group_identity_score(self):
        self.evaluate_group_identity_score(node_dict=self.terminals)

    def test6b_group_identity_score(self):
        self.evaluate_group_identity_score(node_dict=self.first_parents)

    def evaluate_group_plain_entropy_score(self, node_dict):
        for x in node_dict:
            dim_single = (self.seq_len,)
            single_scores = group_plain_entropy_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
            dim_pair = (self.seq_len, self.seq_len)
            pair_scores = group_plain_entropy_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
            for i in range(node_dict[x]['aln'].seq_length):
                single_chars = {}
                for k in range(node_dict[x]['aln'].size):
                    curr_single_char = node_dict[x]['aln'].alignment[k, i]
                    if curr_single_char not in single_chars:
                        single_chars[curr_single_char] = 0.0
                    single_chars[curr_single_char] += 1.0
                expected_single_score = 0.0
                for c in single_chars:
                    frequency = single_chars[c] / node_dict[x]['aln'].size
                    expected_single_score -= frequency * np.log(frequency)
                self.assertEqual(single_scores[i], expected_single_score)
                for j in range(i, node_dict[x]['aln'].seq_length):
                    if i == j:
                        self.assertEqual(single_scores[i], pair_scores[i, j])
                        continue
                    pair_chars = {}
                    for k in range(node_dict[x]['aln'].size):
                        curr_pair_char = node_dict[x]['aln'].alignment[k, i] + node_dict[x]['aln'].alignment[k, j]
                        if curr_pair_char not in pair_chars:
                            pair_chars[curr_pair_char] = 0.0
                        pair_chars[curr_pair_char] += 1.0
                    expected_pair_score = 0.0
                    for c in pair_chars:
                        frequency = pair_chars[c] / node_dict[x]['aln'].size
                        expected_pair_score -= frequency * np.log(frequency)
                    self.assertEqual(pair_scores[i, j], expected_pair_score)

    def test7a_group_plain_entropy_score(self):
        self.evaluate_group_plain_entropy_score(node_dict=self.terminals)

    def test7b_group_plain_entropy_score(self):
        self.evaluate_group_plain_entropy_score(node_dict=self.first_parents)

    def evaluate_mutual_information_computation(self, node_dict):
        for x in node_dict:
            dim_single = (self.seq_len,)
            dim_pair = (self.seq_len, self.seq_len)
            mi_values = mutual_information_computation(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
            single_entropies = group_plain_entropy_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
            pair_joint_entropies = group_plain_entropy_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
            for i in range(node_dict[x]['aln'].seq_length):
                for j in range(i + 1, node_dict[x]['aln'].seq_length):
                    hi = single_entropies[i]
                    hj = single_entropies[j]
                    hij = pair_joint_entropies[i, j]
                    mi = (hi + hj) - hij
                    self.assertEqual(mi_values[0][i, j], hi)
                    self.assertEqual(mi_values[1][i, j], hj)
                    self.assertEqual(mi_values[2][i, j], hij)
                    self.assertEqual(mi_values[3][i, j], mi)

    def test8a_mutual_information_computation(self):
        self.evaluate_mutual_information_computation(node_dict=self.terminals)

    def test8b_mutual_information_computation(self):
        self.evaluate_mutual_information_computation(node_dict=self.first_parents)

    def evaluate_group_mutual_information_score(self, node_dict):
        for x in node_dict:
            dim_single = (self.seq_len,)
            dim_pair = (self.seq_len, self.seq_len)
            mi_values = group_mutual_information_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
            single_entropies = group_plain_entropy_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
            pair_joint_entropies = group_plain_entropy_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
            for i in range(node_dict[x]['aln'].seq_length):
                for j in range(i + 1, node_dict[x]['aln'].seq_length):
                    hi = single_entropies[i]
                    hj = single_entropies[j]
                    hij = pair_joint_entropies[i, j]
                    expected_mi = (hi + hj) - hij
                    self.assertEqual(mi_values[i, j], expected_mi)

    def test9a_group_mutual_information_score(self):
        self.evaluate_group_mutual_information_score(node_dict=self.terminals)

    def test9b_group_mutual_information_score(self):
        self.evaluate_group_mutual_information_score(node_dict=self.first_parents)

    def evaluate_group_normalized_mutual_information_score(self, node_dict):
        for x in node_dict:
            dim_single = (self.seq_len,)
            dim_pair = (self.seq_len, self.seq_len)
            nmi_values = group_normalized_mutual_information_score(freq_table=node_dict[x]['pair'],
                                                                   dimensions=dim_pair)
            single_entropies = group_plain_entropy_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
            pair_joint_entropies = group_plain_entropy_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
            for i in range(node_dict[x]['aln'].seq_length):
                for j in range(i + 1, node_dict[x]['aln'].seq_length):
                    hi = single_entropies[i]
                    hj = single_entropies[j]
                    hij = pair_joint_entropies[i, j]
                    mi = (hi + hj) - hij
                    normalization = np.mean([hi, hj])
                    if hi == hj == 0.0:
                        expected_nmi = 1.0
                    elif normalization == 0.0:
                        expected_nmi = 0.0
                    else:
                        expected_nmi = mi / normalization
                    self.assertEqual(nmi_values[i, j], expected_nmi)

    def test10a_group_normalized_mutual_information_score(self):
        self.evaluate_group_normalized_mutual_information_score(node_dict=self.terminals)

    def test10b_group_normalized_mutual_information_score(self):
        self.evaluate_group_normalized_mutual_information_score(node_dict=self.first_parents)

    def evaluate_average_product_correction(self, node_dict):
        for x in node_dict:
            pair_dim = (self.seq_len, self.seq_len)
            mi_matrix = group_mutual_information_score(freq_table=node_dict[x]['pair'], dimensions=pair_dim)
            total_sum = 0.0
            total_count = 0.0
            column_sums = np.zeros(node_dict[x]['aln'].seq_length)
            column_counts = np.zeros(node_dict[x]['aln'].seq_length)
            for i in range(node_dict[x]['aln'].seq_length):
                for j in range(i + 1, node_dict[x]['aln'].seq_length):
                    mi = mi_matrix[i, j]
                    total_sum += mi
                    total_count += 1
                    column_sums[i] += mi
                    column_counts[i] += 1
                    column_sums[j] += mi
                    column_counts[j] += 1
            expected_apc_matrix = np.zeros((node_dict[x]['aln'].seq_length, node_dict[x]['aln'].seq_length))
            if total_count > 0:
                total_average = total_sum / total_count
                column_average = column_sums / column_counts
                if total_average > 0:
                    for i in range(node_dict[x]['aln'].seq_length):
                        for j in range(i + 1, node_dict[x]['aln'].seq_length):
                            numerator = column_average[i] * column_average[j]
                            correction_factor = numerator / total_average
                            expected_apc_matrix[i, j] = mi_matrix[i, j] - correction_factor
            apc_matrix = average_product_correction(mutual_information_matrix=mi_matrix)
            diff = apc_matrix - expected_apc_matrix
            not_passing = diff > 1E-13
            self.assertTrue(not not_passing.any())

    def test11a_average_product_correction(self):
        self.evaluate_average_product_correction(node_dict=self.terminals)

    def test11b_average_product_correction(self):
        self.evaluate_average_product_correction(node_dict=self.first_parents)

    def evaluate_filtered_average_product_correction(self, node_dict):
        for x in node_dict:
            pair_dim = (self.seq_len, self.seq_len)
            mi_matrix = group_mutual_information_score(freq_table=node_dict[x]['pair'], dimensions=pair_dim)
            total_sum = 0.0
            total_count = 0.0
            column_sums = np.zeros(node_dict[x]['aln'].seq_length)
            column_counts = np.zeros(node_dict[x]['aln'].seq_length)
            for i in range(node_dict[x]['aln'].seq_length):
                for j in range(i + 1, node_dict[x]['aln'].seq_length):
                    mi = mi_matrix[i, j]
                    total_sum += mi
                    total_count += 1
                    column_sums[i] += mi
                    column_counts[i] += 1
                    column_sums[j] += mi
                    column_counts[j] += 1
            expected_apc_matrix = np.zeros((node_dict[x]['aln'].seq_length, node_dict[x]['aln'].seq_length))
            if total_count > 0:
                total_average = total_sum / total_count
                column_average = column_sums / column_counts
                if total_average > 0:
                    for i in range(node_dict[x]['aln'].seq_length):
                        for j in range(i + 1, node_dict[x]['aln'].seq_length):
                            numerator = column_average[i] * column_average[j]
                            correction_factor = numerator / total_average
                            if mi_matrix[i, j] > 0.0001:
                                expected_apc_matrix[i, j] = mi_matrix[i, j] - correction_factor
            apc_matrix = filtered_average_product_correction(mutual_information_matrix=mi_matrix)
            diff = apc_matrix - expected_apc_matrix
            not_passing = diff > 1E-13
            self.assertTrue(not not_passing.any())

    def test11c_heuristic_average_product_correction(self):
        self.evaluate_filtered_average_product_correction(node_dict=self.terminals)

    def test11d_heuristic_average_product_correction(self):
        self.evaluate_filtered_average_product_correction(node_dict=self.first_parents)

    def evaluate_ratio_computation(self, node_dict):
        for x in node_dict:
            dim_pair = (self.seq_len, self.seq_len)
            expected_match_entropy = group_plain_entropy_score(freq_table=node_dict[x]['match'], dimensions=dim_pair)
            expected_mismatch_entropy = group_plain_entropy_score(freq_table=node_dict[x]['mismatch'],
                                                                  dimensions=dim_pair)
            expected_ratio = expected_mismatch_entropy / expected_match_entropy
            div_by_0_indices = np.isnan(expected_ratio)
            comparable_indices = ~div_by_0_indices
            observed_ratios = ratio_computation(match_table=expected_match_entropy,
                                                mismatch_table=expected_mismatch_entropy)
            comparable_diff = observed_ratios[comparable_indices] - expected_ratio[comparable_indices]
            self.assertFalse(comparable_diff.any())
            mismatch_indices = div_by_0_indices & (expected_mismatch_entropy == 0.0)
            min_check = observed_ratios[mismatch_indices] == 0.0
            self.assertTrue(min_check.all())
            match_indices = div_by_0_indices & (expected_match_entropy == 0.0)
            match_indices = ((1 * match_indices) - (1 * mismatch_indices)).astype(bool)
            max_check = observed_ratios[match_indices] == np.tan(np.pi / 2.0)
            self.assertTrue(max_check.all())

    def test12a_ratio_computation(self):
        self.evaluate_ratio_computation(node_dict=self.terminals)

    def test12b_ratio_computation(self):
        self.evaluate_ratio_computation(node_dict=self.first_parents)

    def evaluate_group_match_mismatch_entropy_ratio(self, node_dict):
        for x in node_dict:
            dim_pair = (self.seq_len, self.seq_len)
            observed_ratios = group_match_mismatch_entropy_ratio(freq_tables={'match': node_dict[x]['match'],
                                                                              'mismatch': node_dict[x]['mismatch']},
                                                                 dimensions=dim_pair)
            expected_match_entropy = group_plain_entropy_score(freq_table=node_dict[x]['match'], dimensions=dim_pair)
            expected_mismatch_entropy = group_plain_entropy_score(freq_table=node_dict[x]['mismatch'],
                                                                  dimensions=dim_pair)
            expected_ratios = ratio_computation(match_table=expected_match_entropy,
                                                mismatch_table=expected_mismatch_entropy)
            diff = observed_ratios - expected_ratios
            self.assertFalse(diff.any())

    def test13a_group_match_mismatch_entropy_ratio(self):
        self.evaluate_group_match_mismatch_entropy_ratio(node_dict=self.terminals)

    def test13b_group_match_mismatch_entropy_ratio(self):
        self.evaluate_group_match_mismatch_entropy_ratio(node_dict=self.first_parents)

    def evaluate_angle_computation(self, node_dict):
        for x in node_dict:
            dim_pair = (self.seq_len, self.seq_len)
            expected_match_entropy = group_plain_entropy_score(freq_table=node_dict[x]['match'], dimensions=dim_pair)
            expected_mismatch_entropy = group_plain_entropy_score(freq_table=node_dict[x]['mismatch'],
                                                                  dimensions=dim_pair)
            expected_ratios = ratio_computation(match_table=expected_match_entropy,
                                                mismatch_table=expected_mismatch_entropy)
            observed_angles = angle_computation(ratios=expected_ratios)
            expected_hypotenuse = np.linalg.norm(np.stack([expected_match_entropy, expected_mismatch_entropy], axis=0),
                                                 axis=0)
            expected_sin_ratio = np.zeros(expected_match_entropy.shape)
            sin_indices = (expected_match_entropy != 0.0) | (expected_mismatch_entropy != 0)
            expected_sin_ratio[sin_indices] = expected_mismatch_entropy[sin_indices] / expected_hypotenuse[sin_indices]
            expected_sin_ratio[expected_match_entropy == 0] = np.sin(90.0)
            expected_sin_ratio[expected_mismatch_entropy == 0] = np.sin(0.0)
            expected_sin_angle = np.arcsin(expected_sin_ratio)
            sin_diff = observed_angles - expected_sin_angle
            self.assertFalse(sin_diff.any())
            cos_indices = (expected_match_entropy != 0.0) | (expected_mismatch_entropy != 0)
            expected_cos_ratio = np.zeros(expected_match_entropy.shape)
            expected_cos_ratio[cos_indices] = expected_match_entropy[cos_indices] / expected_hypotenuse[cos_indices]
            expected_cos_ratio[expected_match_entropy == 0.0] = np.cos(90.0)
            expected_cos_ratio[expected_mismatch_entropy == 0] = np.cos(0.0)
            expected_cos_angle = np.arccos(expected_cos_ratio)
            cos_diff = observed_angles - expected_cos_angle
            self.assertFalse(cos_diff.any())

    def test14a_angle_computation(self):
        self.evaluate_angle_computation(node_dict=self.terminals)

    def test14b_angle_computation(self):
        self.evaluate_angle_computation(node_dict=self.first_parents)

    def evaluate_group_match_mismatch_entropy_angle(self, node_dict):
        for x in node_dict:
            dim_pair = (self.seq_len, self.seq_len)
            observed_angles = group_match_mismatch_entropy_angle(freq_tables={'match': node_dict[x]['match'],
                                                                              'mismatch': node_dict[x]['mismatch']},
                                                                 dimensions=dim_pair)
            expected_ratios = group_match_mismatch_entropy_ratio(freq_tables=node_dict[x], dimensions=dim_pair)
            expected_angles = angle_computation(ratios=expected_ratios)
            diff = observed_angles - expected_angles
            self.assertFalse(diff.any())

    def test15a_group_match_mismatch_entropy_angle(self):
        self.evaluate_group_match_mismatch_entropy_angle(node_dict=self.terminals)

    def test15b_group_match_mismatch_entropy_angle(self):
        self.evaluate_group_match_mismatch_entropy_angle(node_dict=self.first_parents)

    def evaluate_diversity_computation(self, node_dict):
        for x in node_dict:
            dim_pair = (self.seq_len, self.seq_len)
            expected_match_entropy = group_plain_entropy_score(freq_table=node_dict[x]['match'], dimensions=dim_pair)
            expected_match_diversity = np.exp(expected_match_entropy)
            observed_match_diversity = diversity_computation(freq_table=node_dict[x]['match'], dimensions=dim_pair)
            match_diff = observed_match_diversity - expected_match_diversity
            self.assertFalse(match_diff.any())
            expected_mismatch_entropy = group_plain_entropy_score(freq_table=node_dict[x]['mismatch'],
                                                                  dimensions=dim_pair)
            expected_mismatch_diversity = np.exp(expected_mismatch_entropy)
            observed_mismatch_diversity = diversity_computation(freq_table=node_dict[x]['mismatch'],
                                                                dimensions=dim_pair)
            mismatch_diff = observed_mismatch_diversity - expected_mismatch_diversity
            self.assertFalse(mismatch_diff.any())

    def test16a_diversity_computation(self):
        self.evaluate_diversity_computation(node_dict=self.terminals)

    def test16b_diversity_computation(self):
        self.evaluate_diversity_computation(node_dict=self.first_parents)

    def evaluate_group_match_diversity_mismatch_entropy_ratio(self, node_dict):
        for x in node_dict:
            dim_pair = (self.seq_len, self.seq_len)
            observed_ratios = group_match_diversity_mismatch_entropy_ratio(
                freq_tables={'match': node_dict[x]['match'], 'mismatch': node_dict[x]['mismatch']}, dimensions=dim_pair)
            expected_match_diversity = diversity_computation(freq_table=node_dict[x]['match'], dimensions=dim_pair)
            expected_mismatch_entropy = group_plain_entropy_score(freq_table=node_dict[x]['mismatch'],
                                                                  dimensions=dim_pair)
            expected_ratios = ratio_computation(match_table=expected_match_diversity,
                                                mismatch_table=expected_mismatch_entropy)
            diff = observed_ratios - expected_ratios
            self.assertFalse(diff.any())

    def test17a_group_match_mismatch_entropy_ratio(self):
        self.evaluate_group_match_diversity_mismatch_entropy_ratio(node_dict=self.terminals)

    def test17b_group_match_mismatch_entropy_ratio(self):
        self.evaluate_group_match_diversity_mismatch_entropy_ratio(node_dict=self.first_parents)

    def evaluate_group_match_diversity_mismatch_entropy_angle(self, node_dict):
        for x in node_dict:
            dim_pair = (self.seq_len, self.seq_len)
            observed_angles = group_match_diversity_mismatch_entropy_angle(
                freq_tables={'match': node_dict[x]['match'], 'mismatch': node_dict[x]['mismatch']}, dimensions=dim_pair)
            expected_ratios = group_match_diversity_mismatch_entropy_ratio(freq_tables=node_dict[x],
                                                                           dimensions=dim_pair)
            expected_angles = angle_computation(ratios=expected_ratios)
            diff = observed_angles - expected_angles
            self.assertFalse(diff.any())

    def test18a_group_match_mismatch_entropy_angle(self):
        self.evaluate_group_match_diversity_mismatch_entropy_angle(node_dict=self.terminals)

    def test18b_group_match_mismatch_entropy_angle(self):
        self.evaluate_group_match_diversity_mismatch_entropy_angle(node_dict=self.first_parents)


if __name__ == '__main__':
    unittest.main()
