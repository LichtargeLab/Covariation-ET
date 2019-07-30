"""
Created on July 12, 2019

@author: Daniel Konecki
"""
import numpy as np
from Bio.Alphabet import Gapped
from test_Base import TestBase
from utils import build_mapping
from SeqAlignment import SeqAlignment
from PhylogeneticTree import PhylogeneticTree
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from EvolutionaryTraceAlphabet import FullIUPACProtein, MultiPositionAlphabet
from PositionalScorer import (PositionalScorer, rank_integer_value_score, rank_real_value_score, group_identity_score,
                              group_plain_entropy_score, mutual_information_computation, group_mutual_information_score,
                              group_normalized_mutual_information_score, average_product_correction)


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
        #
        cls.seq_len = cls.query_aln_fa_small.seq_length

    def test1a_init(self):
        pos_scorer = PositionalScorer(seq_length=self.seq_len, pos_size=1, metric='identity')
        self.assertEqual(pos_scorer.sequence_length, self.seq_len)
        self.assertEqual(pos_scorer.position_size, 1)
        self.assertEqual(pos_scorer.dimensions, (self.seq_len,))
        self.assertEqual(pos_scorer.metric, 'identity')
        self.assertEqual(pos_scorer.metric_type, 'integer')

    def test1b_init(self):
        pos_scorer = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric='identity')
        self.assertEqual(pos_scorer.sequence_length, self.seq_len)
        self.assertEqual(pos_scorer.position_size, 2)
        self.assertEqual(pos_scorer.dimensions, (self.seq_len, self.seq_len))
        self.assertEqual(pos_scorer.metric, 'identity')
        self.assertEqual(pos_scorer.metric_type, 'integer')

    def test1c_init(self):
        with self.assertRaises(ValueError):
            PositionalScorer(seq_length=self.seq_len, pos_size=1, metric='fake')
        with self.assertRaises(ValueError):
            PositionalScorer(seq_length=self.seq_len, pos_size=2, metric='fake')

    def test1d_init(self):
        pos_scorer = PositionalScorer(seq_length=self.seq_len, pos_size=1, metric='plain_entropy')
        self.assertEqual(pos_scorer.sequence_length, self.seq_len)
        self.assertEqual(pos_scorer.position_size, 1)
        self.assertEqual(pos_scorer.dimensions, (self.seq_len,))
        self.assertEqual(pos_scorer.metric, 'plain_entropy')
        self.assertEqual(pos_scorer.metric_type, 'real')

    def test1e_init(self):
        pos_scorer = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric='plain_entropy')
        self.assertEqual(pos_scorer.sequence_length, self.seq_len)
        self.assertEqual(pos_scorer.position_size, 2)
        self.assertEqual(pos_scorer.dimensions, (self.seq_len, self.seq_len))
        self.assertEqual(pos_scorer.metric, 'plain_entropy')
        self.assertEqual(pos_scorer.metric_type, 'real')

    def test1f_init(self):
        pos_scorer = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric='mutual_information')
        self.assertEqual(pos_scorer.sequence_length, self.seq_len)
        self.assertEqual(pos_scorer.position_size, 2)
        self.assertEqual(pos_scorer.dimensions, (self.seq_len, self.seq_len))
        self.assertEqual(pos_scorer.metric, 'mutual_information')
        self.assertEqual(pos_scorer.metric_type, 'real')

    def test1g_init(self):
        pos_scorer = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric='normalized_mutual_information')
        self.assertEqual(pos_scorer.sequence_length, self.seq_len)
        self.assertEqual(pos_scorer.position_size, 2)
        self.assertEqual(pos_scorer.dimensions, (self.seq_len, self.seq_len))
        self.assertEqual(pos_scorer.metric, 'normalized_mutual_information')
        self.assertEqual(pos_scorer.metric_type, 'real')

    def test1h_init(self):
        pos_scorer = PositionalScorer(seq_length=self.seq_len, pos_size=2,
                                      metric='average_product_corrected_mutual_information')
        self.assertEqual(pos_scorer.sequence_length, self.seq_len)
        self.assertEqual(pos_scorer.position_size, 2)
        self.assertEqual(pos_scorer.dimensions, (self.seq_len, self.seq_len))
        self.assertEqual(pos_scorer.metric, 'average_product_corrected_mutual_information')
        self.assertEqual(pos_scorer.metric_type, 'real')

    def evaluate_score_group_ambiguous(self, node_dict, metric):
        pos_scorer_single = PositionalScorer(seq_length=self.seq_len, pos_size=1, metric=metric)
        pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric=metric)
        for x in node_dict:
            group_single_scores = pos_scorer_single.score_group(freq_table=node_dict[x]['single'])
            group_pair_scores = pos_scorer_pair.score_group(freq_table=node_dict[x]['pair'])
            for i in range(node_dict[x]['aln'].seq_length):
                if metric == 'identity':
                    single_score = group_identity_score(freq_table=node_dict[x]['single'], pos=i)
                elif metric == 'plain_entropy':
                    single_score = group_plain_entropy_score(freq_table=node_dict[x]['single'], pos=i)
                else:
                    raise ValueError('Cannot test metric: {} in evaluate_score_group_ambiguous'.format(metric))
                self.assertEqual(group_single_scores[i], single_score)
                for j in range(i + 1, node_dict[x]['aln'].seq_length):
                    if metric == 'identity':
                        pair_score = group_identity_score(freq_table=node_dict[x]['pair'], pos=(i, j))
                    elif metric == 'plain_entropy':
                        pair_score = group_plain_entropy_score(freq_table=node_dict[x]['pair'], pos=(i, j))
                    else:
                        raise ValueError('Cannot test metric: {} in evaluate_score_group_ambiguous'.format(metric))
                    self.assertEqual(group_pair_scores[i, j], pair_score)

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
        for x in node_dict:
            group_pair_scores = pos_scorer_pair.score_group(freq_table=node_dict[x]['pair'])
            for i in range(node_dict[x]['aln'].seq_length):
                for j in range(i + 1, node_dict[x]['aln'].seq_length):
                    if metric == 'mutual_information':
                        pair_score = group_mutual_information_score(freq_table=node_dict[x]['pair'], pos=(i, j))
                    elif metric == 'normalized_mutual_information':
                        pair_score = group_normalized_mutual_information_score(freq_table=node_dict[x]['pair'],
                                                                               pos=(i, j))
                    else:
                        raise ValueError('Cannot test metric: {} in evaluate_score_group_mi'.format(metric))
                    self.assertEqual(group_pair_scores[i, j], pair_score)

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
            mi_matrix = np.zeros((self.seq_len, self.seq_len))
            column_sums = {}
            column_counts = {}
            matrix_sums = 0.0
            total_count = 0.0
            freq_table = node_dict[x]['pair']
            for pos in freq_table.get_positions():
                if pos[0] == pos[1]:
                    continue
                if pos[0] not in column_sums:
                    column_sums[pos[0]] = 0.0
                    column_counts[pos[0]] = 0.0
                if pos[1] not in column_sums:
                    column_sums[pos[1]] = 0.0
                    column_counts[pos[1]] = 0.0
                mi = group_mutual_information_score(freq_table=freq_table, pos=pos)
                mi_matrix[pos[0]][pos[1]] = mi
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
            not_passing = diff > 1E-14
            self.assertTrue(not not_passing.any())

    def test2i_score_group(self):
        # Score group using average product correction mutual information metric terminals
        self.evaluate_score_group_average_product_corrected_mi(node_dict=self.terminals)

    def test2j_score_group(self):
        # Score group using average product correction mutual information metric parents
        self.evaluate_score_group_average_product_corrected_mi(node_dict=self.first_parents)

    def evaluate_score_rank_ambiguous(self, node_dict, metric, single=True, pair=True):
        if single:
            pos_scorer_single = PositionalScorer(seq_length=self.seq_len, pos_size=1, metric=metric)
            group_scores_single = []
        if pair:
           pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric=metric)
           group_scores_pair = []
        for x in node_dict:
            if single:
                group_score_single = pos_scorer_single.score_group(freq_table=node_dict[x]['single'])
                group_scores_single.append(group_score_single)
            if pair:
                group_score_pair = pos_scorer_pair.score_group(freq_table=node_dict[x]['pair'])
                group_scores_pair.append(group_score_pair)
        if single:
            group_scores_single = np.stack(group_scores_single, axis=0)
            if metric == 'identity':
                expected_rank_score_single = rank_integer_value_score(score_matrix=group_scores_single)
            else:
                expected_rank_score_single = rank_real_value_score(score_matrix=group_scores_single)
            rank_score_single = pos_scorer_single.score_rank(score_tensor=group_scores_single)
            diff_single = rank_score_single - expected_rank_score_single
            self.assertTrue(not diff_single.any())
        if pair:
            group_scores_pair = np.stack(group_scores_pair, axis=0)
            if metric == 'identity':
                expected_rank_score_pair = rank_integer_value_score(score_matrix=group_scores_pair)
            else:
                expected_rank_score_pair = rank_real_value_score(score_matrix=group_scores_pair)
            rank_score_pair = pos_scorer_pair.score_rank(score_tensor=group_scores_pair)
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

    def evaluate_rank_integer_value_score(self, node_dict):
        group_single_scores = []
        group_pair_scores = []
        for x in node_dict:
            group_single_score = np.zeros(node_dict[x]['aln'].seq_length)
            group_pair_score = np.zeros((node_dict[x]['aln'].seq_length, node_dict[x]['aln'].seq_length))
            for i in range(node_dict[x]['aln'].seq_length):
                group_single_score[i] = group_identity_score(freq_table=node_dict[x]['single'], pos=i)
                for j in range(i + 1, self.seq_len):
                    group_pair_score[i, j] = group_identity_score(freq_table=node_dict[x]['pair'], pos=(i, j))
            group_single_scores.append(group_single_score)
            group_pair_scores.append(group_pair_score)
        group_single_scores = np.stack(group_single_scores, axis=0)
        rank_single_scores = rank_integer_value_score(score_matrix=group_single_scores)
        expected_rank_single_scores = np.zeros(self.seq_len)
        for i in range(node_dict[x]['aln'].seq_length):
            for k in range(group_single_scores.shape[0]):
                if group_single_scores[k, i] != 0:
                    expected_rank_single_scores[i] = 1
        diff_single = rank_single_scores - expected_rank_single_scores
        self.assertTrue(not diff_single.any())
        group_pair_scores = np.stack(group_pair_scores, axis=0)
        rank_pair_scores = rank_integer_value_score(score_matrix=group_pair_scores)
        expected_rank_pair_scores = np.zeros((self.seq_len, self.seq_len))
        for i in range(node_dict[x]['aln'].seq_length):
            for j in range(i + 1, node_dict[x]['aln'].seq_length):
                for k in range(group_pair_scores.shape[0]):
                    if group_pair_scores[k, i, j] != 0:
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
        group_pair_scores = []
        for x in node_dict:
            group_single_score = np.zeros(self.seq_len)
            group_pair_score = np.zeros((self.seq_len, self.seq_len))
            for i in range(self.seq_len):
                group_single_score[i] = group_plain_entropy_score(freq_table=node_dict[x]['single'], pos=i)
                for j in range(i + 1, self.seq_len):
                    group_pair_score[i, j] = group_plain_entropy_score(freq_table=node_dict[x]['pair'], pos=(i, j))
            group_single_scores.append(group_single_score)
            group_pair_scores.append(group_pair_score)
        group_single_scores = np.stack(group_single_scores, axis=0)
        rank_single_scores = rank_real_value_score(score_matrix=group_single_scores)
        expected_rank_single_scores = np.zeros(self.seq_len)
        rank = group_single_scores.shape[0]
        for i in range(node_dict[x]['aln'].seq_length):
            for k in range(rank):
                expected_rank_single_scores[i] += (1.0 / rank) * group_single_scores[k, i]
        diff_single = rank_single_scores - expected_rank_single_scores
        not_passing_single = diff_single > 1E-16
        self.assertTrue(not not_passing_single.any())
        group_pair_scores = np.stack(group_pair_scores, axis=0)
        rank_pair_scores = rank_real_value_score(score_matrix=group_pair_scores)
        expected_rank_pair_scores = np.zeros((self.seq_len, self.seq_len))
        rank = group_pair_scores.shape[0]
        for i in range(node_dict[x]['aln'].seq_length):
            for j in range(i + 1, node_dict[x]['aln'].seq_length):
                for k in range(rank):
                    expected_rank_pair_scores[i, j] += (1.0 / rank) * group_pair_scores[k, i, j]
        diff_pair = rank_pair_scores - expected_rank_pair_scores
        not_passing_pair = diff_pair > 1E-16
        self.assertTrue(not not_passing_pair.any())

    def test5a_rank_real_value_score(self):
        self.evaluate_rank_real_value_score(node_dict=self.terminals)

    def test5b_rank_real_value_score(self):
        self.evaluate_rank_real_value_score(node_dict=self.first_parents)

    def evaluate_group_identity_score(self, node_dict):
        for x in node_dict:
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
                single_score = group_identity_score(freq_table=node_dict[x]['single'], pos=i)
                self.assertEqual(single_score, expected_single_score)
                for j in range(i + 1, node_dict[x]['aln'].seq_length):
                    expected_pair_score = 0
                    pair_char = None
                    for k in range(node_dict[x]['aln'].size):
                        curr_pair_char = node_dict[x]['aln'].alignment[k, i] + node_dict[x]['aln'].alignment[k, j]
                        if pair_char is None:
                            pair_char = curr_pair_char
                        else:
                            if pair_char != curr_pair_char:
                                expected_pair_score = 1
                    pair_score = group_identity_score(freq_table=node_dict[x]['pair'], pos=(i, j))
                    self.assertEqual(pair_score, expected_pair_score)

    def test6a_group_identity_score(self):
        self.evaluate_group_identity_score(node_dict=self.terminals)

    def test6b_group_identity_score(self):
        self.evaluate_group_identity_score(node_dict=self.first_parents)

    def evaluate_group_plain_entropy_score(self, node_dict):
        for x in node_dict:
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
                single_score = group_plain_entropy_score(freq_table=node_dict[x]['single'], pos=i)
                self.assertEqual(single_score, expected_single_score)
                for j in range(i + 1, node_dict[x]['aln'].seq_length):
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
                    pair_score = group_plain_entropy_score(freq_table=node_dict[x]['pair'], pos=(i, j))
                    self.assertEqual(pair_score, expected_pair_score)

    def test7a_group_plain_entropy_score(self):
        self.evaluate_group_plain_entropy_score(node_dict=self.terminals)

    def test7b_group_plain_entropy_score(self):
        self.evaluate_group_plain_entropy_score(node_dict=self.first_parents)

    def evalaute_mutual_information_computation(self, node_dict):
        for x in node_dict:
            for i in range(node_dict[x]['aln'].seq_length):
                for j in range(i + 1, node_dict[x]['aln'].seq_length):
                    hi = group_plain_entropy_score(freq_table=node_dict[x]['single'], pos=i)
                    hj = group_plain_entropy_score(freq_table=node_dict[x]['single'], pos=j)
                    hij = group_plain_entropy_score(freq_table=node_dict[x]['pair'], pos=(i, j))
                    mi = (hi + hj) - hij
                    mi_values = mutual_information_computation(freq_table=node_dict[x]['pair'], pos=(i, j))
                    self.assertEqual(mi_values[0], i)
                    self.assertEqual(mi_values[1], j)
                    self.assertEqual(mi_values[2], hi)
                    self.assertEqual(mi_values[3], hj)
                    self.assertEqual(mi_values[4], hij)
                    self.assertEqual(mi_values[5], mi)

    def test8a_mutual_information_computation(self):
        self.evalaute_mutual_information_computation(node_dict=self.terminals)

    def test8b_mutual_information_computation(self):
        self.evalaute_mutual_information_computation(node_dict=self.first_parents)

    def evaluate_group_mutual_information_score(self, node_dict):
        for x in node_dict:
            for i in range(node_dict[x]['aln'].seq_length):
                for j in range(i + 1, node_dict[x]['aln'].seq_length):
                    hi = group_plain_entropy_score(freq_table=node_dict[x]['single'], pos=i)
                    hj = group_plain_entropy_score(freq_table=node_dict[x]['single'], pos=j)
                    hij = group_plain_entropy_score(freq_table=node_dict[x]['pair'], pos=(i, j))
                    expected_mi = (hi + hj) - hij
                    mi = group_mutual_information_score(freq_table=node_dict[x]['pair'], pos=(i, j))
                    self.assertEqual(mi, expected_mi)

    def test9a_group_mutual_information_score(self):
        self.evaluate_group_mutual_information_score(node_dict=self.terminals)

    def test9b_group_mutual_information_score(self):
        self.evaluate_group_mutual_information_score(node_dict=self.first_parents)

    def evaluate_group_normalized_mutual_information_score(self, node_dict):
        for x in node_dict:
            for i in range(node_dict[x]['aln'].seq_length):
                for j in range(i + 1, node_dict[x]['aln'].seq_length):
                    hi = group_plain_entropy_score(freq_table=node_dict[x]['single'], pos=i)
                    hj = group_plain_entropy_score(freq_table=node_dict[x]['single'], pos=j)
                    hij = group_plain_entropy_score(freq_table=node_dict[x]['pair'], pos=(i, j))
                    mi = (hi + hj) - hij
                    normalization = hi + hij
                    if normalization == 0.0:
                        expected_nmi = 0.0
                    else:
                        expected_nmi = mi / normalization
                    nmi = group_normalized_mutual_information_score(freq_table=node_dict[x]['pair'], pos=(i, j))
                    self.assertEqual(nmi, expected_nmi)

    def test10a_group_normalized_mutual_information_score(self):
        self.evaluate_group_normalized_mutual_information_score(node_dict=self.terminals)

    def test10b_group_normalized_mutual_information_score(self):
        self.evaluate_group_normalized_mutual_information_score(node_dict=self.first_parents)

    def evaluate_average_product_correction(self, node_dict):
        for x in node_dict:
            mi_matrix = np.zeros((node_dict[x]['aln'].seq_length, node_dict[x]['aln'].seq_length))
            total_sum = 0.0
            total_count = 0.0
            column_sums = np.zeros(node_dict[x]['aln'].seq_length)
            column_counts = np.zeros(node_dict[x]['aln'].seq_length)
            for i in range(node_dict[x]['aln'].seq_length):
                for j in range(i + 1, node_dict[x]['aln'].seq_length):
                    hi = group_plain_entropy_score(freq_table=node_dict[x]['single'], pos=i)
                    hj = group_plain_entropy_score(freq_table=node_dict[x]['single'], pos=j)
                    hij = group_plain_entropy_score(freq_table=node_dict[x]['pair'], pos=(i, j))
                    mi = (hi + hj) - hij
                    mi_matrix[i, j] = mi
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
            not_passing = diff > 1E-14
            self.assertTrue(not not_passing.any())

    def test11a_average_product_correction(self):
        self.evaluate_average_product_correction(node_dict=self.terminals)

    def test11b_average_product_correction(self):
        self.evaluate_average_product_correction(node_dict=self.first_parents)
