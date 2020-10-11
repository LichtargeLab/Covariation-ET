"""
Created on July 12, 2019

@author: Daniel Konecki
"""
import os
import unittest
import numpy as np
from copy import deepcopy
from unittest import TestCase
from Bio.Alphabet import Gapped
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from test_Base import TestBase
from utils import build_mapping
from SeqAlignment import SeqAlignment
from FrequencyTable import FrequencyTable
from PhylogeneticTree import PhylogeneticTree
from MatchMismatchTable import MatchMismatchTable
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from test_seqAlignment import generate_temp_fn, write_out_temp_fasta
from EvolutionaryTraceAlphabet import FullIUPACDNA, FullIUPACProtein, MultiPositionAlphabet
from PositionalScorer import (integer_valued_metrics, real_valued_metrics, ambiguous_metrics, single_only_metrics,
                              pair_only_metrics, min_metrics, max_metrics, PositionalScorer, rank_integer_value_score,
                              rank_real_value_score, mutual_information_computation, average_product_correction,
                              filtered_average_product_correction, ratio_computation, angle_computation,
                              diversity_computation,
                              group_identity_score, group_plain_entropy_score,
                              group_mutual_information_score, group_normalized_mutual_information_score,
                              count_computation, group_match_count_score, group_mismatch_count_score,
                              group_match_mismatch_count_ratio, group_match_mismatch_count_angle,
                              group_match_entropy_score, group_mismatch_entropy_score,
                              group_match_mismatch_entropy_ratio, group_match_mismatch_entropy_angle,
                              group_match_diversity_score, group_mismatch_diversity_score,
                              group_match_mismatch_diversity_ratio, group_match_mismatch_diversity_angle,
                              group_match_diversity_mismatch_entropy_ratio,
                              group_match_diversity_mismatch_entropy_angle)

dna_alpha = Gapped(FullIUPACDNA())
dna_alpha_size, _, dna_map, dna_rev = build_mapping(dna_alpha)
protein_alpha = Gapped(FullIUPACProtein())
protein_alpha_size, _, protein_map, protein_rev = build_mapping(protein_alpha)
pair_dna_alpha = MultiPositionAlphabet(dna_alpha, size=2)
dna_pair_alpha_size, _, dna_pair_map, dna_pair_rev = build_mapping(pair_dna_alpha)
dna_single_to_pair = np.zeros((max(dna_map.values()) + 1, max(dna_map.values()) + 1))
for char in dna_pair_map:
    dna_single_to_pair[dna_map[char[0]], dna_map[char[1]]] = dna_pair_map[char]
pair_protein_alpha = MultiPositionAlphabet(protein_alpha, size=2)
pro_pair_alpha_size, _, pro_pair_map, pro_pair_rev = build_mapping(pair_protein_alpha)
quad_protein_alpha = MultiPositionAlphabet(protein_alpha, size=4)
pro_quad_alpha_size, _, pro_quad_map, pro_quad_rev = build_mapping(quad_protein_alpha)
pro_single_to_pair = np.zeros((max(protein_map.values()) + 1, max(protein_map.values()) + 1))
for char in pro_pair_map:
    pro_single_to_pair[protein_map[char[0]], protein_map[char[1]]] = pro_pair_map[char]
pro_single_to_quad = {}
for char in pro_quad_map:
    key = (protein_map[char[0]], protein_map[char[1]], protein_map[char[2]], protein_map[char[3]])
    pro_single_to_quad[key] = pro_quad_map[char]
protein_seq1 = SeqRecord(id='seq1', seq=Seq('MET---', alphabet=FullIUPACProtein()))
protein_seq2 = SeqRecord(id='seq2', seq=Seq('M-TREE', alphabet=FullIUPACProtein()))
protein_seq3 = SeqRecord(id='seq3', seq=Seq('M-FREE', alphabet=FullIUPACProtein()))
protein_msa = MultipleSeqAlignment(records=[protein_seq1, protein_seq2, protein_seq3], alphabet=FullIUPACProtein())
dna_seq1 = SeqRecord(id='seq1', seq=Seq('ATGGAGACT---------', alphabet=FullIUPACDNA()))
dna_seq2 = SeqRecord(id='seq2', seq=Seq('ATG---ACTAGAGAGGAG', alphabet=FullIUPACDNA()))
dna_seq3 = SeqRecord(id='seq3', seq=Seq('ATG---TTTAGAGAGGAG', alphabet=FullIUPACDNA()))
dna_msa = MultipleSeqAlignment(records=[dna_seq1, dna_seq2, dna_seq3], alphabet=FullIUPACDNA())

aln_fn = write_out_temp_fasta(
                out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
aln = SeqAlignment(aln_fn, 'seq1', polymer_type='Protein')
aln.import_alignment()
os.remove(aln_fn)
num_aln = aln._alignment_to_num(mapping=protein_map)

pro_single_ft = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
pro_single_ft.characterize_alignment(num_aln=num_aln)

pro_pair_ft = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
pro_pair_ft.characterize_alignment(num_aln=num_aln, single_to_pair=pro_single_to_pair)

mm_table = MatchMismatchTable(seq_len=6, num_aln=num_aln, single_alphabet_size=protein_alpha_size,
                              single_mapping=protein_map, single_reverse_mapping=protein_rev,
                              larger_alphabet_size=pro_quad_alpha_size,
                              larger_mapping=pro_quad_map, larger_reverse_mapping=pro_quad_rev,
                              single_to_larger_mapping=pro_single_to_quad, pos_size=2)
mm_table.identify_matches_mismatches()
mm_freq_tables = {'match': FrequencyTable(alphabet_size=pro_quad_alpha_size, mapping=pro_quad_map,
                                          reverse_mapping=pro_quad_rev, seq_len=6, pos_size=2)}
mm_freq_tables['match'].mapping = pro_quad_map
mm_freq_tables['match'].set_depth(3)
mm_freq_tables['mismatch'] = deepcopy(mm_freq_tables['match'])
for pos in mm_freq_tables['match'].get_positions():
    char_dict = {'match': {}, 'mismatch': {}}
    for i in range(3):
        for j in range(i + 1, 3):
            status, stat_char = mm_table.get_status_and_character(pos=pos, seq_ind1=i, seq_ind2=j)
            if stat_char not in char_dict[status]:
                char_dict[status][stat_char] = 0
            char_dict[status][stat_char] += 1
    for m in char_dict:
        for curr_char in char_dict[m]:
            mm_freq_tables[m]._increment_count(pos=pos, char=curr_char,
                                               amount=char_dict[m][curr_char])
for m in ['match', 'mismatch']:
    mm_freq_tables[m].finalize_table()


class TestPositionalScorerPackageVariables(TestCase):

    def test_value_type_metrics(self):
        self.assertEqual(len(integer_valued_metrics.intersection(real_valued_metrics)), 0)

    def test_pos_type_metrics(self):
        self.assertEqual(len(ambiguous_metrics.intersection(single_only_metrics)), 0)
        self.assertEqual(len(ambiguous_metrics.intersection(pair_only_metrics)), 0)
        self.assertEqual(len(single_only_metrics.intersection(pair_only_metrics)), 0)

    def test_direction_type_metrics(self):
        self.assertEqual(len(min_metrics.intersection(max_metrics)), 0)

    def test_metric_type_counts(self):
        value_metric_count = len(integer_valued_metrics.union(real_valued_metrics))
        pos_metric_count = len(ambiguous_metrics.union(single_only_metrics.union(pair_only_metrics)))
        direction_metric_count = len(min_metrics.union(max_metrics))
        self.assertEqual(value_metric_count, pos_metric_count)
        self.assertEqual(value_metric_count, direction_metric_count)


class TestPositionalScorerInit(TestCase):

    def test__init_pos_single_metric_identity(self):
        ps = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 1)
        self.assertEqual(ps.dimensions, (6, ))
        self.assertEqual(ps.metric_type, 'integer')
        self.assertEqual(ps.rank_type, 'min')
        self.assertEqual(ps.metric, 'identity')

    def test__init_pos_single_metric_plain_entropy(self):
        ps = PositionalScorer(seq_length=6, pos_size=1, metric='plain_entropy')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 1)
        self.assertEqual(ps.dimensions, (6,))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'min')
        self.assertEqual(ps.metric, 'plain_entropy')

    def test__init_failure_pos_single_metric_mutual_information(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='mutual_information')

    def test__init_failure_pos_single_metric_normalized_mutual_information(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='normalized_mutual_information')

    def test__init_failure_pos_single_metric_average_product_corrected_mutual_information(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='average_product_corrected_mutual_information')

    def test__init_failure_pos_single_metric_filtered_average_product_corrected_mutual_information(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='filtered_average_product_corrected_mutual_information')

    def test__init_failure_pos_single_metric_match_count(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='match_count')

    def test__init_failure_pos_single_metric_mismatch_count(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='mismatch_count')

    def test__init_failure_pos_single_metric_match_mismatch_count_ratio(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='match_mismatch_count_ratio')

    def test__init_failure_pos_single_metric_match_mismatch_count_angle(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='match_mismatch_count_angle')

    def test__init_failure_pos_single_metric_match_entropy(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='match_entropy')

    def test__init_failure_pos_single_metric_mismatch_entropy(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='mismatch_entropy')

    def test__init_failure_pos_single_metric_match_mismatch_entropy_ratio(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='match_mismatch_entropy_ratio')

    def test__init_failure_pos_single_metric_match_mismatch_entropy_angle(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='match_mismatch_entropy_angle')

    def test__init_failure_pos_single_metric_match_diversity(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='match_diversity')

    def test__init_failure_pos_single_metric_mismatch_diversity(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='mismatch_diversity')

    def test__init_failure_pos_single_metric_match_mismatch_diversity_ratio(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='match_mismatch_diversity_ratio')

    def test__init_failure_pos_single_metric_match_mismatch_diversity_angle(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='match_mismatch_diversity_angle')

    def test__init_failure_pos_single_metric_match_diversity_mismatch_entropy_ratio(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='match_diversity_mismatch_entropy_ratio')

    def test__init_failure_pos_single_metric_match_diversity_mismatch_entropy_angle(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='match_diversity_mismatch_entropy_angle')

    def test__init_pos_pair_metric_identity(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'integer')
        self.assertEqual(ps.rank_type, 'min')
        self.assertEqual(ps.metric, 'identity')

    def test__init_pos_pair_metric_plain_entropy(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'min')
        self.assertEqual(ps.metric, 'plain_entropy')

    def test__init_pos_pair_metric_mutual_information(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mutual_information')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'max')
        self.assertEqual(ps.metric, 'mutual_information')

    def test__init_pos_pair_metric_normalized_mutual_information(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='normalized_mutual_information')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'max')
        self.assertEqual(ps.metric, 'normalized_mutual_information')

    def test__init_pos_pair_metric_average_product_corrected_mutual_information(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='average_product_corrected_mutual_information')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'max')
        self.assertEqual(ps.metric, 'average_product_corrected_mutual_information')

    def test__init_pos_pair_metric_filtered_average_product_corrected_mutual_information(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='filtered_average_product_corrected_mutual_information')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'max')
        self.assertEqual(ps.metric, 'filtered_average_product_corrected_mutual_information')

    def test__init_pos_pair_metric_match_count(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_count')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'max')
        self.assertEqual(ps.metric, 'match_count')

    def test__init_pos_pair_metric_mismatch_count(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_count')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'min')
        self.assertEqual(ps.metric, 'mismatch_count')

    def test__init_pos_pair_metric_match_mismatch_count_ratio(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_ratio')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'min')
        self.assertEqual(ps.metric, 'match_mismatch_count_ratio')

    def test__init_pos_pair_metric_match_mismatch_count_angle(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_angle')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'min')
        self.assertEqual(ps.metric, 'match_mismatch_count_angle')

    def test__init_pos_pair_metric_match_entropy(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_entropy')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'max')
        self.assertEqual(ps.metric, 'match_entropy')

    def test__init_pos_pair_metric_mismatch_entropy(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_entropy')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'min')
        self.assertEqual(ps.metric, 'mismatch_entropy')

    def test__init_pos_pair_metric_match_mismatch_entropy_ratio(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_ratio')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'min')
        self.assertEqual(ps.metric, 'match_mismatch_entropy_ratio')

    def test__init_pos_pair_metric_match_mismatch_entropy_angle(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_angle')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'min')
        self.assertEqual(ps.metric, 'match_mismatch_entropy_angle')

    def test__init_pos_pair_metric_match_diversity(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'max')
        self.assertEqual(ps.metric, 'match_diversity')

    def test__init_pos_pair_metric_mismatch_diversity(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_diversity')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'min')
        self.assertEqual(ps.metric, 'mismatch_diversity')

    def test__init_pos_pair_metric_match_mismatch_diversity_ratio(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_ratio')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'min')
        self.assertEqual(ps.metric, 'match_mismatch_diversity_ratio')

    def test__init_pos_pair_metric_match_mismatch_diversity_angle(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_angle')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'min')
        self.assertEqual(ps.metric, 'match_mismatch_diversity_angle')

    def test__init_pos_pair_metric_match_diversity_mismatch_entropy_ratio(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_ratio')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'min')
        self.assertEqual(ps.metric, 'match_diversity_mismatch_entropy_ratio')

    def test__init_pos_pair_metric_match_diversity_mismatch_entropy_angle(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_angle')
        self.assertEqual(ps.sequence_length, 6)
        self.assertEqual(ps.position_size, 2)
        self.assertEqual(ps.dimensions, (6, 6))
        self.assertEqual(ps.metric_type, 'real')
        self.assertEqual(ps.rank_type, 'min')
        self.assertEqual(ps.metric, 'match_diversity_mismatch_entropy_angle')

    def test__init_failure_bad_metric_pos_single(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=1, metric='foobar')

    def test__init_failure_bad_metric_pos_pair(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=2, metric='foobar')

    def test__init_failure_bad_pos_size(self):
        with self.assertRaises(ValueError):
            ps = PositionalScorer(seq_length=6, pos_size=100, metric='identity')


class TestPositionalScorerGroupIdentityScore(TestCase):

    def test_group_identity_score_single(self):
        final = group_identity_score(freq_table=pro_single_ft, dimensions=(6, ))
        expected_final = np.array([0, 1, 1, 1, 1, 1])
        self.assertFalse((final - expected_final).any())

    def test_group_identity_score_pair(self):
        final = group_identity_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        expected_final = np.array([[0, 1, 1, 1, 1, 1],
                                   [0, 1, 1, 1, 1, 1],
                                   [0, 0, 1, 1, 1, 1],
                                   [0, 0, 0, 1, 1, 1],
                                   [0, 0, 0, 0, 1, 1],
                                   [0, 0, 0, 0, 0, 1]])
        self.assertFalse((final - expected_final).any())

    def test_group_identity_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_identity_score(freq_table=None, dimensions=(6, ))

    def test_group_identity_score_failure_wrong_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_identity_score(freq_table=pro_single_ft, dimensions=(6, 6))

    def test_group_identity_score_failure_wrong_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_identity_score(freq_table=pro_pair_ft, dimensions=(6, ))


class TestPositionalScorerGroupPlainEntropyScore(TestCase):

    def test_group_plain_entropy_score_single(self):
        final = group_plain_entropy_score(freq_table=pro_single_ft, dimensions=(6, ))
        score_3 = 0.0
        score_1_2 = -1.0 * (((2.0 / 3) * np.log(2.0 / 3)) + ((1.0 / 3) * np.log(1.0 / 3)))
        expected_final = np.array([score_3, score_1_2, score_1_2, score_1_2, score_1_2, score_1_2])
        self.assertFalse((final - expected_final).any())

    def test_group_plain_entropy_score_pair(self):
        final = group_plain_entropy_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        score_3 = 0.0
        score_1_2 = -1.0 * (((2.0 / 3) * np.log(2.0 / 3)) + ((1.0 / 3) * np.log(1.0 / 3)))
        score_1_1_1 = -1.0 * (3 * ((1.0 / 3) * np.log(1.0 / 3)))
        expected_final = np.array([[score_3, score_1_2, score_1_2, score_1_2, score_1_2, score_1_2],
                                   [0.0, score_1_2, score_1_1_1, score_1_2, score_1_2, score_1_2],
                                   [0.0, 0.0, score_1_2, score_1_1_1, score_1_1_1, score_1_1_1],
                                   [0.0, 0.0, 0.0, score_1_2, score_1_2, score_1_2],
                                   [0.0, 0.0, 0.0, 0.0, score_1_2, score_1_2],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, score_1_2]])
        self.assertFalse((final - expected_final).any())

    def test_group_plain_entropy_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_plain_entropy_score(freq_table=None, dimensions=(6, ))

    def test_group_plain_entropy_score_failure_wrong_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_plain_entropy_score(freq_table=pro_single_ft, dimensions=(6, 6))

    def test_group_plain_entropy_score_failure_wrong_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_plain_entropy_score(freq_table=pro_pair_ft, dimensions=(6, ))


class TestPositionalScorerGroupMutualInformation(TestCase):

    def test_mutual_information_computation(self):
        e_i, e_j, e_ij, mi = mutual_information_computation(freq_table=pro_pair_ft, dimensions=(6, 6))
        score_3 = 0.0
        score_1_2 = -1.0 * (((2.0 / 3) * np.log(2.0 / 3)) + ((1.0 / 3) * np.log(1.0 / 3)))
        expected_e_i = np.array([[0.0, score_3, score_3, score_3, score_3, score_3],
                                 [0.0, 0.0, score_1_2, score_1_2, score_1_2, score_1_2],
                                 [0.0, 0.0, 0.0, score_1_2, score_1_2, score_1_2],
                                 [0.0, 0.0, 0.0, 0.0, score_1_2, score_1_2],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, score_1_2],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertFalse((e_i - expected_e_i).any())
        expected_e_j = np.array([[0.0, score_1_2, score_1_2, score_1_2, score_1_2, score_1_2],
                                 [0.0, 0.0, score_1_2, score_1_2, score_1_2, score_1_2],
                                 [0.0, 0.0, 0.0, score_1_2, score_1_2, score_1_2],
                                 [0.0, 0.0, 0.0, 0.0, score_1_2, score_1_2],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, score_1_2],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertFalse((e_j - expected_e_j).any())
        score_1_1_1 = -1.0 * (3 * ((1.0 / 3) * np.log(1.0 / 3)))
        expected_e_ij = np.array([[0.0, score_1_2, score_1_2, score_1_2, score_1_2, score_1_2],
                                  [0.0, 0.0, score_1_1_1, score_1_2, score_1_2, score_1_2],
                                  [0.0, 0.0, 0.0, score_1_1_1, score_1_1_1, score_1_1_1],
                                  [0.0, 0.0, 0, 0.0, score_1_2, score_1_2],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, score_1_2],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertFalse((e_ij - expected_e_ij).any())
        expected_mi = (expected_e_i + expected_e_j) - expected_e_ij
        self.assertFalse((mi - expected_mi).any())

    def test_mutual_information_computation_failure_single_pos_input(self):
        with self.assertRaises(ValueError):
            mutual_information_computation(freq_table=pro_single_ft, dimensions=(6, ))

    def test_mutual_information_computation_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            mutual_information_computation(freq_table=None, dimensions=(6, ))

    def test_mutual_information_computation_failure_mismatch_dimensions_large(self):
        with self.assertRaises(ValueError):
            mutual_information_computation(freq_table=pro_single_ft, dimensions=(6, 6))

    def test_mutual_information_computation_failure_mismatch_dimensions_small(self):
        with self.assertRaises(ValueError):
            mutual_information_computation(freq_table=pro_pair_ft, dimensions=(6,))

    def test_group_mutual_information_score(self):
        _, _, _, expected_mi = mutual_information_computation(freq_table=pro_pair_ft, dimensions=(6, 6))
        mi = group_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        self.assertFalse((mi - expected_mi).any())

    def test_group_mutual_information_score_failure_single_pos_input(self):
        with self.assertRaises(ValueError):
            group_mutual_information_score(freq_table=pro_single_ft, dimensions=(6,))

    def test_group_mutual_information_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_mutual_information_score(freq_table=None, dimensions=(6,))

    def test_group_mutual_information_score_failure_mismatch_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_mutual_information_score(freq_table=pro_single_ft, dimensions=(6, 6))

    def test_group_mutual_information_score_failure_mismatch_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6,))

    def test_group_normalized_mutual_information_score(self):
        expected_e_i, expected_e_j, _, expected_mi = mutual_information_computation(freq_table=pro_pair_ft,
                                                                                    dimensions=(6, 6))
        nmi = group_normalized_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        triu_ind = np.triu_indices(n=nmi.shape[0], k=1)
        for x in range(nmi.shape[0]):
            i = triu_ind[0][x]
            j = triu_ind[1][x]
            e_i = expected_e_i[i, j]
            e_j = expected_e_j[i, j]
            if (e_i == 0.0) and (e_j == 0.0):
                self.assertEqual(nmi[i, j], 1.0)
            else:
                norm = np.mean([e_i, e_j])
                if norm == 0.0:
                    self.assertEqual(nmi[i, j], 0.0)
                else:
                    self.assertEqual(nmi[i, j], expected_mi[i, j] / norm)
        self.assertFalse((np.tril(nmi) - np.zeros(nmi.shape)).any())

    def test_group_normalized_mutual_information_score_failure_single_pos_input(self):
        with self.assertRaises(ValueError):
            group_normalized_mutual_information_score(freq_table=pro_single_ft, dimensions=(6,))

    def test_group_normalized_mutual_information_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_normalized_mutual_information_score(freq_table=None, dimensions=(6,))

    def test_group_normalized_mutual_information_score_failure_mismatch_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_normalized_mutual_information_score(freq_table=pro_single_ft, dimensions=(6, 6))

    def test_group_normalized_mutual_information_score_failure_mismatch_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_normalized_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6,))

    def test_average_product_correction(self):
        expected_e_i, expected_e_j, _, expected_mi = mutual_information_computation(freq_table=pro_pair_ft,
                                                                                    dimensions=(6, 6))
        expected_column_row_sum = np.zeros(6)
        for i in range(6):
            for j in range(6):
                if i > j:
                    curr = expected_mi[j, i]
                elif i == j:
                    continue
                else:
                    curr = expected_mi[i, j]
                expected_column_row_sum[i] += curr
        expected_column_row_mean = expected_column_row_sum / (6.0 - 1.0)
        expected_mi_mean = np.sum(expected_mi) * (2.0 / (6.0 * (6.0 - 1.0)))
        expected_apc_numerator = np.zeros((6, 6))
        expected_apc = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                if i >= j:
                    continue
                expected_apc_numerator[i, j] = (expected_column_row_mean[i] * expected_column_row_mean[j])
                expected_apc[i, j] = expected_apc_numerator[i, j] / expected_mi_mean
        expected_final = expected_mi - expected_apc
        apc_mi = average_product_correction(expected_mi)
        self.assertFalse((apc_mi - expected_final).any())

    def test_average_product_correction_failure_mean_zero_with_nonzero_values(self):
        mi_zero = np.zeros((6, 6))
        mi_zero[[0, 1], [4, 5]] = [1, -1]
        with self.assertRaises(ValueError):
            average_product_correction(mutual_information_matrix=mi_zero)

    def test_average_product_correction_all_zeros(self):
        mi_zero = np.zeros((6, 6))
        apc = average_product_correction(mutual_information_matrix=mi_zero)
        self.assertFalse((mi_zero - apc).any())

    def test_average_product_correction_failure_no_input(self):
        with self.assertRaises(ValueError):
            average_product_correction(mutual_information_matrix=None)

    def test_average_product_correction_failure_rectangular_matrix(self):
        mi = group_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        temp = np.zeros((mi.shape[0], mi.shape[1] + 1))
        temp[:, :6] += mi
        with self.assertRaises(ValueError):
            average_product_correction(mutual_information_matrix=temp)

    def test_average_product_correction_failure_full_matrix(self):
        mi = group_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        with self.assertRaises(ValueError):
            average_product_correction(mutual_information_matrix=mi + mi.T)

    def test_average_product_correction_failure_lower_triangle_matrix(self):
        mi = group_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        with self.assertRaises(ValueError):
            average_product_correction(mutual_information_matrix=mi.T)

    def test_filtered_average_product_correction(self):
        expected_e_i, expected_e_j, _, expected_mi = mutual_information_computation(freq_table=pro_pair_ft,
                                                                                    dimensions=(6, 6))
        expected_column_row_sum = np.zeros(6)
        for i in range(6):
            for j in range(6):
                if i > j:
                    curr = expected_mi[j, i]
                elif i == j:
                    continue
                else:
                    curr = expected_mi[i, j]
                expected_column_row_sum[i] += curr
        expected_column_row_mean = expected_column_row_sum / (6.0 - 1.0)
        expected_mi_mean = np.sum(expected_mi) * (2.0 / (6.0 * (6.0 - 1.0)))
        expected_apc_numerator = np.zeros((6, 6))
        expected_apc = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                if i >= j:
                    continue
                expected_apc_numerator[i, j] = (expected_column_row_mean[i] * expected_column_row_mean[j])
                expected_apc[i, j] = expected_apc_numerator[i, j] / expected_mi_mean
        expected_final = expected_mi - expected_apc
        expected_final[expected_mi <= 0.0001] = 0.0
        apc_mi = filtered_average_product_correction(expected_mi)
        self.assertFalse((apc_mi - expected_final).any())

    def test_filtered_average_product_correction_guaranteed_filter(self):
        expected_e_i, expected_e_j, _, expected_mi = mutual_information_computation(freq_table=pro_pair_ft,
                                                                                    dimensions=(6, 6))
        expected_mi[-2, -1] = 0.00001
        expected_column_row_sum = np.zeros(6)
        for i in range(6):
            for j in range(6):
                if i > j:
                    curr = expected_mi[j, i]
                elif i == j:
                    continue
                else:
                    curr = expected_mi[i, j]
                expected_column_row_sum[i] += curr
        expected_column_row_mean = expected_column_row_sum / (6.0 - 1.0)
        expected_mi_mean = np.sum(expected_mi) * (2.0 / (6.0 * (6.0 - 1.0)))
        expected_apc_numerator = np.zeros((6, 6))
        expected_apc = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                if i >= j:
                    continue
                expected_apc_numerator[i, j] = (expected_column_row_mean[i] * expected_column_row_mean[j])
                expected_apc[i, j] = expected_apc_numerator[i, j] / expected_mi_mean
        expected_final = expected_mi - expected_apc
        expected_final[expected_mi <= 0.0001] = 0.0
        apc_mi = filtered_average_product_correction(expected_mi)
        self.assertFalse((apc_mi - expected_final).any())

    def test_filtered_average_product_correction_failure_mean_zero_with_nonzero_values(self):
        mi_zero = np.zeros((6, 6))
        mi_zero[[0, 1], [4, 5]] = [1, -1]
        with self.assertRaises(ValueError):
            apc = filtered_average_product_correction(mutual_information_matrix=mi_zero)

    def test_filtered_average_product_correction_all_zeros(self):
        mi_zero = np.zeros((6, 6))
        apc = filtered_average_product_correction(mutual_information_matrix=mi_zero)
        self.assertFalse((mi_zero - apc).any())

    def test_filtered_average_product_correction_failure_no_input(self):
        with self.assertRaises(ValueError):
            filtered_average_product_correction(mutual_information_matrix=None)

    def test_filtered_average_product_correction_failure_rectangular_matrix(self):
        mi = group_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        temp = np.zeros((mi.shape[0], mi.shape[1] + 1))
        temp[:, :6] += mi
        with self.assertRaises(ValueError):
            filtered_average_product_correction(mutual_information_matrix=temp)

    def test_filtered_average_product_correction_failure_full_matrix(self):
        mi = group_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        with self.assertRaises(ValueError):
            filtered_average_product_correction(mutual_information_matrix=mi + mi.T)

    def test_filtered_average_product_correction_failure_lower_triangle_matrix(self):
        mi = group_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        with self.assertRaises(ValueError):
            filtered_average_product_correction(mutual_information_matrix=mi.T)


class TestPositionalScorerCountComputation(TestCase):

    def test_count_computation_pair(self):
        expected_final = np.zeros((6, 6))
        for p1 in range(6):  # position 1 in pair1
            for p2 in range(6):  # position 2 in pair1
                if p1 < p2:
                    for s1 in range(3):  # sequence 1 in comparison
                        for s2 in range(s1 + 1, 3):  # sequence 2 in comparison
                            curr_stat, _ = mm_table.get_status_and_character(pos=(p1, p2), seq_ind1=s1, seq_ind2=s2)

                            if curr_stat == 'match':
                                expected_final[p1, p2] += 1
        final = count_computation(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        self.assertFalse((final - expected_final).any())

    def test_count_computation_failure_single(self):
        with self.assertRaises(ValueError):
            count_computation(freq_table=pro_single_ft, dimensions=(6,))

    def test_count_computation_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            count_computation(freq_table=None, dimensions=(6, 6))

    def test_count_computation_failure_wrong_dimensions_large(self):
        with self.assertRaises(ValueError):
            count_computation(freq_table=pro_single_ft, dimensions=(6, 6))

    def test_count_computation_failure_wrong_dimensions_small(self):
        with self.assertRaises(ValueError):
            count_computation(freq_table=pro_pair_ft, dimensions=(6,))


class TestPositionalScorerDiversityComputation(TestCase):

    def test_diversity_computation_pair(self):
        final = diversity_computation(freq_table=pro_pair_ft, dimensions=(6, 6))
        score_3 = np.exp(0.0)
        score_1_2 = np.exp(-1.0 * (((2.0 / 3) * np.log(2.0 / 3)) + ((1.0 / 3) * np.log(1.0 / 3))))
        score_1_1_1 = np.exp(-1.0 * (3 * ((1.0 / 3) * np.log(1.0 / 3))))
        expected_final = np.array([[0.0, score_1_2, score_1_2, score_1_2, score_1_2, score_1_2],
                                   [0.0, 0.0, score_1_1_1, score_1_2, score_1_2, score_1_2],
                                   [0.0, 0.0, 0.0, score_1_1_1, score_1_1_1, score_1_1_1],
                                   [0.0, 0.0, 0.0, 0.0, score_1_2, score_1_2],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, score_1_2],
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertFalse((final - expected_final).any())

    def test_diversity_computation_failure_single(self):
        with self.assertRaises(ValueError):
            diversity_computation(freq_table=pro_single_ft, dimensions=(6,))

    def test_diversity_computation_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            diversity_computation(freq_table=None, dimensions=(6, 6))

    def test_diversity_computation_failure_wrong_dimensions_large(self):
        with self.assertRaises(ValueError):
            diversity_computation(freq_table=pro_single_ft, dimensions=(6, 6))

    def test_diversity_computation_failure_wrong_dimensions_small(self):
        with self.assertRaises(ValueError):
            diversity_computation(freq_table=pro_pair_ft, dimensions=(6,))


class TestPositionalScorerRatioComputation(TestCase):

    def test_ratio_computation(self):
        expected_value = np.tan(np.pi / 2.0)
        match_entropy = group_plain_entropy_score(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        mismatch_entropy = group_plain_entropy_score(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        # Ensure at least one instance of Case 1
        match_entropy[0, -1] = 0.0
        mismatch_entropy[0, -1] = 0.5
        # Ensure at least two instance of Case 3
        mismatch_entropy[[0, 1], [-2, -1]] = 0.0
        match_entropy[[0, 1], [-2, -1]] = 0.5
        ratio_mat = ratio_computation(match_table=match_entropy, mismatch_table=mismatch_entropy)
        for i in range(6):
            for j in range(6):
                match_val = match_entropy[i, j]
                mismatch_val = mismatch_entropy[i, j]
                curr_val = ratio_mat[i, j]
                if (match_val == 0.0) and (mismatch_val != 0.0):  # Case 1
                    self.assertEqual(curr_val, expected_value)
                elif ((match_val != 0.0) and (mismatch_val == 0.0) or  # Case 2
                      (match_val == 0.0) and (mismatch_val == 0.0)):
                    self.assertEqual(curr_val, 0.0)
                else:  # Case 3
                    self.assertEqual(curr_val, mismatch_val / match_val)

    def test_ratio_computation_match_zeros(self):
        ratio_mat = ratio_computation(match_table=np.zeros((6, 6)), mismatch_table=np.random.rand(6, 6))
        expected_value = np.tan(np.pi / 2.0)
        expected_ratio_mat = np.ones((6, 6)) * expected_value
        self.assertFalse((ratio_mat - expected_ratio_mat).any())

    def test_ratio_computation_mismatch_zeros(self):
        expected_mat = np.zeros((6, 6))
        ratio_mat = ratio_computation(match_table=np.random.rand(6, 6), mismatch_table=expected_mat)
        self.assertFalse((ratio_mat - expected_mat).any())

    def test_ratio_computation_both_zeros(self):
        expected_mat = np.zeros((6, 6))
        ratio_mat = ratio_computation(match_table=expected_mat, mismatch_table=expected_mat)
        self.assertFalse((ratio_mat - expected_mat).any())

    def test_ratio_computation_failure_no_match_table(self):
        with self.assertRaises(AttributeError):
            ratio_computation(match_table=None, mismatch_table=np.random.rand(6, 6))

    def test_ratio_computation_failure_no_mismatch_table(self):
        with self.assertRaises(TypeError):
            ratio_computation(match_table=np.random.rand(6, 6), mismatch_table=None)

    def test_ratio_computation_failure_no_tables(self):
        with self.assertRaises(AttributeError):
            ratio_computation(match_table=None, mismatch_table=None)

    def test_ratio_computation_failure_table_size_difference(self):
        match_temp = np.random.rand(4, 4)
        mismatch_temp = np.random.rand(6, 6)
        with self.assertRaises(IndexError):
            ratio_computation(match_table=match_temp, mismatch_table=mismatch_temp)

    def test_ratio_computation_failure_table_size_difference2(self):
        match_temp = np.random.rand(6, 6)
        mismatch_temp = np.random.rand(4, 4)
        with self.assertRaises(IndexError):
            ratio_computation(match_table=match_temp, mismatch_table=mismatch_temp)


class TestPositionalScorerAngleComputation(TestCase):

    def test_angle_computation(self):
        match_entropy = group_plain_entropy_score(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        mismatch_entropy = group_plain_entropy_score(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        # Ensure at least one instance of Case 1
        match_entropy[0, -1] = 0.0
        mismatch_entropy[0, -1] = 0.5
        # Ensure at least two instance of Case 3
        mismatch_entropy[[0, 1], [-2, -1]] = 0.0
        match_entropy[[0, 1], [-2, -1]] = 0.5
        ratio_mat = ratio_computation(match_table=match_entropy, mismatch_table=mismatch_entropy)
        angle_mat = angle_computation(ratio_mat)
        for i in range(6):
            for j in range(6):
                match_val = match_entropy[i, j]
                mismatch_val = mismatch_entropy[i, j]
                curr_val = angle_mat[i, j]
                if (match_val == 0.0) and (mismatch_val != 0.0):  # Case 1
                    self.assertEqual(curr_val, np.pi / 2.0)
                elif ((match_val != 0.0) and (mismatch_val == 0.0) or  # Case 2
                      (match_val == 0.0) and (mismatch_val == 0.0)):
                    self.assertEqual(curr_val, 0.0)
                else:  # Case 3
                    self.assertEqual(curr_val, np.arctan(mismatch_val / match_val))

    def test_angle_computation_match_zeros(self):
        ratio_mat = ratio_computation(match_table=np.zeros((6, 6)), mismatch_table=np.random.rand(6, 6))
        angle_mat = angle_computation(ratio_mat)
        expected_value = np.pi / 2.0
        expected_angle_mat = np.ones((6, 6)) * expected_value
        self.assertFalse((angle_mat - expected_angle_mat).any())

    def test_angle_computation_mismatch_zeros(self):
        expected_mat = np.zeros((6, 6))
        ratio_mat = ratio_computation(match_table=np.random.rand(6, 6), mismatch_table=expected_mat)
        angle_mat = angle_computation(ratio_mat)
        self.assertFalse((angle_mat - expected_mat).any())

    def test_angle_computation_both_zeros(self):
        expected_mat = np.zeros((6, 6))
        ratio_mat = ratio_computation(match_table=expected_mat, mismatch_table=expected_mat)
        angle_mat = angle_computation(ratio_mat)
        self.assertFalse((angle_mat - expected_mat).any())

    def test_angle_computation_failure_no_ratio_table(self):
        with self.assertRaises(AttributeError):
            angle_computation(ratios=None)


class TestPositionalScorerMatchMismatchCountScores(TestCase):

    def test_group_match_count_score(self):
        expected_counts = count_computation(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        counts = group_match_count_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        triu_ind = np.triu_indices(n=counts.shape[0], k=1)
        for x in range(counts.shape[0]):
            i = triu_ind[0][x]
            j = triu_ind[1][x]
            self.assertEqual(counts[i, j], expected_counts[i, j])
        self.assertFalse((np.tril(counts) - np.zeros(counts.shape)).any())

    def test_group_match_count_score_failure_single_pos_input(self):
        with self.assertRaises(ValueError):
            group_match_count_score(freq_tables={'match': pro_single_ft, 'mismatch': None}, dimensions=(6,))

    def test_group_match_count_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_match_count_score(freq_tables={'match': None, 'mismatch': None}, dimensions=(6,))

    def test_group_match_count_score_failure_mismatch_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_match_count_score(freq_tables={'match': pro_single_ft, 'mismatch': None}, dimensions=(6, 6))

    def test_group_match_count_score_failure_mismatch_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_match_count_score(freq_tables=mm_freq_tables, dimensions=(6,))

    def test_group_mismatch_count_score(self):
        expected_counts = count_computation(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
        counts = group_mismatch_count_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        triu_ind = np.triu_indices(n=counts.shape[0], k=1)
        for x in range(counts.shape[0]):
            i = triu_ind[0][x]
            j = triu_ind[1][x]
            self.assertEqual(counts[i, j], expected_counts[i, j])
        self.assertFalse((np.tril(counts) - np.zeros(counts.shape)).any())

    def test_group_mismatch_count_score_failure_single_pos_input(self):
        with self.assertRaises(ValueError):
            group_mismatch_count_score(freq_tables={'match': None, 'mismatch': pro_single_ft}, dimensions=(6,))

    def test_group_mismatch_count_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_mismatch_count_score(freq_tables={'match': None, 'mismatch': None}, dimensions=(6, 6))

    def test_group_mismatch_count_score_failure_mismatch_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_mismatch_count_score(freq_tables={'match': None, 'mismatch': pro_single_ft}, dimensions=(6, 6))

    def test_group_mismatch_count_score_failure_mismatch_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_mismatch_count_score(freq_tables=mm_freq_tables, dimensions=(6,))

    def test_group_match_mismatch_count_ratio_score(self):
        expected_value = np.tan(np.pi / 2.0)
        expected_match_count = group_match_count_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        expected_mismatch_count = group_mismatch_count_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        # This test does not cover all cases because the simple frequency tables used do not cover all cases. All cases
        # should be covered from the ratio_computation test but it would be good to make that test explicit from this
        # function as well.
        ratio_mat = group_match_mismatch_count_ratio(freq_tables=mm_freq_tables, dimensions=(6, 6))
        for i in range(6):
            for j in range(6):
                match_val = expected_match_count[i, j]
                mismatch_val = expected_mismatch_count[i, j]
                curr_val = ratio_mat[i, j]
                if (match_val == 0.0) and (mismatch_val != 0.0):  # Case 1
                    self.assertEqual(curr_val, expected_value)
                elif ((match_val != 0.0) and (mismatch_val == 0.0) or  # Case 2
                      (match_val == 0.0) and (mismatch_val == 0.0)):
                    self.assertEqual(curr_val, 0.0)
                else:  # Case 3
                    self.assertEqual(curr_val, mismatch_val / match_val)

    def test_group_match_mismatch_count_ratio_match_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': mm_freq_tables['mismatch']}
        mismatch_counts = group_mismatch_count_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        ratio_mat = group_match_mismatch_count_ratio(freq_tables=temp_tables, dimensions=(6, 6))
        expected_value = np.tan(np.pi / 2.0)
        expected_ratio_mat = np.zeros((6, 6))
        expected_ratio_mat[np.nonzero(mismatch_counts)] = 1.0
        expected_ratio_mat *= expected_value
        self.assertFalse((ratio_mat - expected_ratio_mat).any())

    def test_group_match_mismatch_count_ratio_mismatch_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp_table}
        expected_mat = np.zeros((6, 6))
        ratio_mat = group_match_mismatch_count_ratio(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((ratio_mat - expected_mat).any())

    def test_group_match_mismatch_count_ratio_both_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': temp_table}
        expected_mat = np.zeros((6, 6))
        ratio_mat = group_match_mismatch_count_ratio(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((ratio_mat - expected_mat).any())

    def test_group_match_mismatch_count_ratio_failure_no_match_table(self):
        temp_tables = {'match': None, 'mismatch': mm_freq_tables['mismatch']}
        with self.assertRaises(AttributeError):
            group_match_mismatch_count_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_count_ratio_failure_no_mismatch_table(self):
        temp_tables = {'match': mm_freq_tables['mismatch'], 'mismatch': None}
        with self.assertRaises(AttributeError):
            group_match_mismatch_count_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_count_ratio_failure_no_tables(self):
        with self.assertRaises(AttributeError):
            group_match_mismatch_count_ratio(freq_tables={'match': None, 'mismatch': None}, dimensions=(6, 6))

    def test_group_match_mismatch_count_ratio_failure_table_size_difference(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp}
        with self.assertRaises(ValueError):
            group_match_mismatch_count_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_count_ratio_failure_table_size_difference2(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': temp, 'mismatch': mm_freq_tables['match']}
        with self.assertRaises(ValueError):
            group_match_mismatch_count_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_count_angle(self):
        expected_match_count = group_match_count_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        expected_mismatch_count = group_mismatch_count_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        # This test does not cover all cases because the simple frequency tables used do not cover all cases. All cases
        # should be covered from the ratio_computation test but it would be good to make that test explicit from this
        # function as well.
        angle_mat = group_match_mismatch_count_angle(freq_tables=mm_freq_tables, dimensions=(6, 6))
        for i in range(6):
            for j in range(6):
                match_val = expected_match_count[i, j]
                mismatch_val = expected_mismatch_count[i, j]
                curr_val = angle_mat[i, j]
                if (match_val == 0.0) and (mismatch_val != 0.0):  # Case 1
                    self.assertEqual(curr_val, np.pi / 2.0)
                elif ((match_val != 0.0) and (mismatch_val == 0.0) or  # Case 2
                      (match_val == 0.0) and (mismatch_val == 0.0)):
                    self.assertEqual(curr_val, 0.0)
                else:  # Case 3
                    self.assertEqual(curr_val, np.arctan(mismatch_val / match_val))

    def test_group_match_mismatch_count_angle_match_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': mm_freq_tables['mismatch']}
        mismatch_counts = group_mismatch_count_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        angle_mat = group_match_mismatch_count_angle(freq_tables=temp_tables, dimensions=(6, 6))
        expected_value = np.pi / 2.0
        expected_angle_mat = np.zeros((6, 6))
        expected_angle_mat[np.nonzero(mismatch_counts)] = 1.0
        expected_angle_mat *= expected_value
        self.assertFalse((angle_mat - expected_angle_mat).any())

    def test_group_match_mismatch_count_angle_mismatch_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp_table}
        expected_mat = np.zeros((6, 6))
        angle_mat = group_match_mismatch_count_angle(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((angle_mat - expected_mat).any())

    def test_group_match_mismatch_count_angle_both_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': temp_table}
        expected_mat = np.zeros((6, 6))
        angle_mat = group_match_mismatch_count_angle(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((angle_mat - expected_mat).any())

    def test_group_match_mismatch_count_angle_failure_no_match_table(self):
        temp_tables = {'match': None, 'mismatch': mm_freq_tables['mismatch']}
        with self.assertRaises(AttributeError):
            group_match_mismatch_count_angle(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_count_angle_failure_no_mismatch_table(self):
        temp_tables = {'match': mm_freq_tables['mismatch'], 'mismatch': None}
        with self.assertRaises(AttributeError):
            group_match_mismatch_count_angle(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_count_angle_failure_no_tables(self):
        with self.assertRaises(AttributeError):
            group_match_mismatch_count_angle(freq_tables={'match': None, 'mismatch': None}, dimensions=(6, 6))

    def test_group_match_mismatch_count_angle_failure_table_size_difference(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp}
        with self.assertRaises(ValueError):
            group_match_mismatch_count_angle(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_count_angle_failure_table_size_difference2(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': temp, 'mismatch': mm_freq_tables['match']}
        with self.assertRaises(ValueError):
            group_match_mismatch_count_angle(freq_tables=temp_tables, dimensions=(6, 6))


class TestPositionalScorerMatchMismatchEntropyScores(TestCase):

    def test_group_match_entropy_score(self):
        expected_entropy = group_plain_entropy_score(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        entropy = group_match_entropy_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        triu_ind = np.triu_indices(n=entropy.shape[0], k=1)
        for x in range(entropy.shape[0]):
            i = triu_ind[0][x]
            j = triu_ind[1][x]
            self.assertEqual(entropy[i, j], expected_entropy[i, j])
        self.assertFalse((np.tril(entropy) - np.zeros(entropy.shape)).any())

    def test_group_match_entropy_score_failure_single_pos_input(self):
        with self.assertRaises(ValueError):
            group_match_entropy_score(freq_tables={'match': pro_single_ft, 'mismatch': None}, dimensions=(6,))

    def test_group_match_entropy_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_match_entropy_score(freq_tables={'match': None, 'mismatch': None}, dimensions=(6, 6))

    def test_group_match_entropy_score_failure_mismatch_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_match_entropy_score(freq_tables={'match': pro_single_ft, 'mismatch': None}, dimensions=(6, 6))

    def test_group_match_entropy_score_failure_mismatch_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_match_entropy_score(freq_tables=mm_freq_tables, dimensions=(6,))

    def test_group_mismatch_entropy_score(self):
        expected_entropy = group_plain_entropy_score(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
        entropy = group_mismatch_entropy_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        triu_ind = np.triu_indices(n=entropy.shape[0], k=1)
        for x in range(entropy.shape[0]):
            i = triu_ind[0][x]
            j = triu_ind[1][x]
            self.assertEqual(entropy[i, j], expected_entropy[i, j])
        self.assertFalse((np.tril(entropy) - np.zeros(entropy.shape)).any())

    def test_group_mismatch_entropy_score_failure_single_pos_input(self):
        with self.assertRaises(ValueError):
            group_mismatch_entropy_score(freq_tables={'match': None, 'mismatch': pro_single_ft}, dimensions=(6,))

    def test_group_mismatch_entropy_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_mismatch_entropy_score(freq_tables={'match': None, 'mismatch': None}, dimensions=(6, 6))

    def test_group_mismatch_entropy_score_failure_mismatch_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_mismatch_entropy_score(freq_tables={'match': None, 'mismatch': pro_single_ft}, dimensions=(6, 6))

    def test_group_mismatch_entropy_score_failure_mismatch_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_mismatch_entropy_score(freq_tables=mm_freq_tables, dimensions=(6,))

    def test_group_match_mismatch_entropy_ratio_score(self):
        expected_value = np.tan(np.pi / 2.0)
        expected_match_count = group_match_entropy_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        expected_mismatch_count = group_mismatch_entropy_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        # This test does not cover all cases because the simple frequency tables used do not cover all cases. All cases
        # should be covered from the ratio_computation test but it would be good to make that test explicit from this
        # function as well.
        ratio_mat = group_match_mismatch_entropy_ratio(freq_tables=mm_freq_tables, dimensions=(6, 6))
        for i in range(6):
            for j in range(6):
                match_val = expected_match_count[i, j]
                mismatch_val = expected_mismatch_count[i, j]
                curr_val = ratio_mat[i, j]
                if (match_val == 0.0) and (mismatch_val != 0.0):  # Case 1
                    self.assertEqual(curr_val, expected_value)
                elif ((match_val != 0.0) and (mismatch_val == 0.0) or  # Case 2
                      (match_val == 0.0) and (mismatch_val == 0.0)):
                    self.assertEqual(curr_val, 0.0)
                else:  # Case 3
                    self.assertEqual(curr_val, mismatch_val / match_val)

    def test_group_match_mismatch_entropy_ratio_match_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': mm_freq_tables['mismatch']}
        mismatch_counts = group_mismatch_entropy_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        ratio_mat = group_match_mismatch_entropy_ratio(freq_tables=temp_tables, dimensions=(6, 6))
        expected_value = np.tan(np.pi / 2.0)
        expected_ratio_mat = np.zeros((6, 6))
        expected_ratio_mat[np.nonzero(mismatch_counts)] = 1.0
        expected_ratio_mat *= expected_value
        self.assertFalse((ratio_mat - expected_ratio_mat).any())

    def test_group_match_mismatch_count_ratio_mismatch_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp_table}
        expected_mat = np.zeros((6, 6))
        ratio_mat = group_match_mismatch_entropy_ratio(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((ratio_mat - expected_mat).any())

    def test_group_match_mismatch_entropy_ratio_both_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': temp_table}
        expected_mat = np.zeros((6, 6))
        ratio_mat = group_match_mismatch_entropy_ratio(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((ratio_mat - expected_mat).any())

    def test_group_match_mismatch_entropy_ratio_failure_no_match_table(self):
        temp_tables = {'match': None, 'mismatch': mm_freq_tables['mismatch']}
        with self.assertRaises(AttributeError):
            group_match_mismatch_entropy_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_entropy_ratio_failure_no_mismatch_table(self):
        temp_tables = {'match': mm_freq_tables['mismatch'], 'mismatch': None}
        with self.assertRaises(AttributeError):
            group_match_mismatch_entropy_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_entropy_ratio_failure_no_tables(self):
        with self.assertRaises(AttributeError):
            group_match_mismatch_entropy_ratio(freq_tables={'match': None, 'mismatch': None}, dimensions=(6, 6))

    def test_group_match_mismatch_entropy_ratio_failure_table_size_difference(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp}
        with self.assertRaises(ValueError):
            group_match_mismatch_entropy_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_entropy_ratio_failure_table_size_difference2(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': temp, 'mismatch': mm_freq_tables['match']}
        with self.assertRaises(ValueError):
            group_match_mismatch_entropy_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_entropy_angle(self):
        expected_match_entropy = group_match_entropy_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        expected_mismatch_entropy = group_mismatch_entropy_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        # This test does not cover all cases because the simple frequency tables used do not cover all cases. All cases
        # should be covered from the ratio_computation test but it would be good to make that test explicit from this
        # function as well.
        angle_mat = group_match_mismatch_entropy_angle(freq_tables=mm_freq_tables, dimensions=(6, 6))
        for i in range(6):
            for j in range(6):
                match_val = expected_match_entropy[i, j]
                mismatch_val = expected_mismatch_entropy[i, j]
                curr_val = angle_mat[i, j]
                if (match_val == 0.0) and (mismatch_val != 0.0):  # Case 1
                    self.assertEqual(curr_val, np.pi / 2.0)
                elif ((match_val != 0.0) and (mismatch_val == 0.0) or  # Case 2
                      (match_val == 0.0) and (mismatch_val == 0.0)):
                    self.assertEqual(curr_val, 0.0)
                else:  # Case 3
                    self.assertEqual(curr_val, np.arctan(mismatch_val / match_val))

    def test_group_match_mismatch_entropy_angle_match_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': mm_freq_tables['mismatch']}
        mismatch_counts = group_mismatch_entropy_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        angle_mat = group_match_mismatch_entropy_angle(freq_tables=temp_tables, dimensions=(6, 6))
        expected_value = np.pi / 2.0
        expected_angle_mat = np.zeros((6, 6))
        expected_angle_mat[np.nonzero(mismatch_counts)] = 1.0
        expected_angle_mat *= expected_value
        self.assertFalse((angle_mat - expected_angle_mat).any())

    def test_group_match_mismatch_entropy_angle_mismatch_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp_table}
        expected_mat = np.zeros((6, 6))
        angle_mat = group_match_mismatch_entropy_angle(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((angle_mat - expected_mat).any())

    def test_group_match_mismatch_entropy_angle_both_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': temp_table}
        expected_mat = np.zeros((6, 6))
        angle_mat = group_match_mismatch_entropy_angle(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((angle_mat - expected_mat).any())

    def test_group_match_mismatch_entropy_angle_failure_no_match_table(self):
        temp_tables = {'match': None, 'mismatch': mm_freq_tables['mismatch']}
        with self.assertRaises(AttributeError):
            group_match_mismatch_entropy_angle(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_entropy_angle_failure_no_mismatch_table(self):
        temp_tables = {'match': mm_freq_tables['mismatch'], 'mismatch': None}
        with self.assertRaises(AttributeError):
            group_match_mismatch_entropy_angle(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_entropy_angle_failure_no_tables(self):
        with self.assertRaises(AttributeError):
            group_match_mismatch_entropy_angle(freq_tables={'match': None, 'mismatch': None}, dimensions=(6, 6))

    def test_group_match_mismatch_entropy_angle_failure_table_size_difference(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp}
        with self.assertRaises(ValueError):
            group_match_mismatch_entropy_angle(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_entropy_angle_failure_table_size_difference2(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': temp, 'mismatch': mm_freq_tables['match']}
        with self.assertRaises(ValueError):
            group_match_mismatch_entropy_angle(freq_tables=temp_tables, dimensions=(6, 6))


class TestPositionalScorerMatchMismatchDiversityScores(TestCase):

    def test_group_match_diversity_score(self):
        expected_diversity = diversity_computation(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        diversity = group_match_diversity_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        triu_ind = np.triu_indices(n=diversity.shape[0], k=1)
        for x in range(diversity.shape[0]):
            i = triu_ind[0][x]
            j = triu_ind[1][x]
            self.assertEqual(diversity[i, j], expected_diversity[i, j])
        self.assertFalse((np.tril(diversity) - np.zeros(diversity.shape)).any())

    def test_group_match_diversity_score_failure_single_pos_input(self):
        with self.assertRaises(ValueError):
            group_match_diversity_score(freq_tables={'match': pro_single_ft, 'mismatch': None}, dimensions=(6,))

    def test_group_match_diversity_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_match_diversity_score(freq_tables={'match': None, 'mismatch': None}, dimensions=(6, 6))

    def test_group_match_diversity_score_failure_mismatch_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_match_diversity_score(freq_tables={'match': pro_single_ft, 'mismatch': None}, dimensions=(6, 6))

    def test_group_match_diversity_score_failure_mismatch_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_match_diversity_score(freq_tables=mm_freq_tables, dimensions=(6,))

    def test_group_mismatch_diversity_score(self):
        expected_diversity = diversity_computation(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
        diversity = group_mismatch_diversity_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        triu_ind = np.triu_indices(n=diversity.shape[0], k=1)
        for x in range(diversity.shape[0]):
            i = triu_ind[0][x]
            j = triu_ind[1][x]
            self.assertEqual(diversity[i, j], expected_diversity[i, j])
        self.assertFalse((np.tril(diversity) - np.zeros(diversity.shape)).any())

    def test_group_mismatch_diversity_score_failure_single_pos_input(self):
        with self.assertRaises(ValueError):
            group_mismatch_diversity_score(freq_tables={'match': None, 'mismatch': pro_single_ft}, dimensions=(6,))

    def test_group_mismatch_diversity_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_mismatch_diversity_score(freq_tables={'match': None, 'mismatch': None}, dimensions=(6, 6))

    def test_group_mismatch_diversity_score_failure_mismatch_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_mismatch_diversity_score(freq_tables={'match': None, 'mismatch': pro_single_ft}, dimensions=(6, 6))

    def test_group_mismatch_diversity_score_failure_mismatch_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_mismatch_diversity_score(freq_tables=mm_freq_tables, dimensions=(6,))

    def test_group_match_mismatch_diversity_ratio_score(self):
        expected_value = np.tan(np.pi / 2.0)
        expected_match_diversity = group_match_diversity_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        expected_mismatch_diversity = group_mismatch_diversity_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        # This test does not cover all cases because the simple frequency tables used do not cover all cases. All cases
        # should be covered from the ratio_computation test but it would be good to make that test explicit from this
        # function as well.
        ratio_mat = group_match_mismatch_diversity_ratio(freq_tables=mm_freq_tables, dimensions=(6, 6))
        for i in range(6):
            for j in range(6):
                match_val = expected_match_diversity[i, j]
                mismatch_val = expected_mismatch_diversity[i, j]
                curr_val = ratio_mat[i, j]
                if (match_val == 0.0) and (mismatch_val != 0.0):  # Case 1
                    self.assertEqual(curr_val, expected_value)
                elif ((match_val != 0.0) and (mismatch_val == 0.0) or  # Case 2
                      (match_val == 0.0) and (mismatch_val == 0.0)):
                    self.assertEqual(curr_val, 0.0)
                else:  # Case 3
                    self.assertEqual(curr_val, mismatch_val / match_val)

    def test_group_match_mismatch_diversity_ratio_match_zeros(self):
        # This is different than all other metrics testing against this case because the lowest possible value for
        # diversity is 1.0.
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': mm_freq_tables['mismatch']}
        mismatch_diversity = group_mismatch_diversity_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        ratio_mat = group_match_mismatch_diversity_ratio(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((ratio_mat - mismatch_diversity).any())

    def test_group_match_mismatch_diversity_ratio_mismatch_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp_table}
        match_diversity = group_match_diversity_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        expected_mat = np.ones((6, 6))
        expected_mat /= match_diversity
        expected_mat = np.triu(expected_mat, k=1)
        ratio_mat = group_match_mismatch_diversity_ratio(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((ratio_mat - expected_mat).any())

    def test_group_match_mismatch_diversity_ratio_both_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': temp_table}
        expected_mat = np.ones((6, 6))
        expected_mat = np.triu(expected_mat, k=1)
        ratio_mat = group_match_mismatch_diversity_ratio(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((ratio_mat - expected_mat).any())

    def test_group_match_mismatch_diversity_ratio_failure_no_match_table(self):
        temp_tables = {'match': None, 'mismatch': mm_freq_tables['mismatch']}
        with self.assertRaises(AttributeError):
            group_match_mismatch_diversity_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_diversity_ratio_failure_no_mismatch_table(self):
        temp_tables = {'match': mm_freq_tables['mismatch'], 'mismatch': None}
        with self.assertRaises(AttributeError):
            group_match_mismatch_diversity_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_diversity_ratio_failure_no_tables(self):
        with self.assertRaises(AttributeError):
            group_match_mismatch_diversity_ratio(freq_tables={'match': None, 'mismatch': None}, dimensions=(6, 6))

    def test_group_match_mismatch_diversity_ratio_failure_table_size_difference(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp}
        with self.assertRaises(ValueError):
            group_match_mismatch_diversity_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_diversity_ratio_failure_table_size_difference2(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': temp, 'mismatch': mm_freq_tables['match']}
        with self.assertRaises(ValueError):
            group_match_mismatch_diversity_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_diversity_angle(self):
        expected_match_diversity = group_match_diversity_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        expected_mismatch_diversity = group_mismatch_diversity_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        # This test does not cover all cases because the simple frequency tables used do not cover all cases. All cases
        # should be covered from the ratio_computation test but it would be good to make that test explicit from this
        # function as well.
        angle_mat = group_match_mismatch_diversity_angle(freq_tables=mm_freq_tables, dimensions=(6, 6))
        for i in range(6):
            for j in range(6):
                match_val = expected_match_diversity[i, j]
                mismatch_val = expected_mismatch_diversity[i, j]
                curr_val = angle_mat[i, j]
                if (match_val == 0.0) and (mismatch_val != 0.0):  # Case 1
                    self.assertEqual(curr_val, np.pi / 2.0)
                elif ((match_val != 0.0) and (mismatch_val == 0.0) or  # Case 2
                      (match_val == 0.0) and (mismatch_val == 0.0)):
                    self.assertEqual(curr_val, 0.0)
                else:  # Case 3
                    self.assertEqual(curr_val, np.arctan(mismatch_val / match_val))

    def test_group_match_mismatch_diversity_angle_match_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': mm_freq_tables['mismatch']}
        mismatch_diversity = group_mismatch_diversity_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        angle_mat = group_match_mismatch_diversity_angle(freq_tables=temp_tables, dimensions=(6, 6))
        expected_angle_mat = angle_computation(ratios=mismatch_diversity)
        self.assertFalse((angle_mat - expected_angle_mat).any())

    def test_group_match_mismatch_diversity_angle_mismatch_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp_table}
        match_diversity = group_match_diversity_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        expected_mat = np.triu(np.arctan(np.ones((6, 6)) / match_diversity), k=1)
        angle_mat = group_match_mismatch_diversity_angle(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((angle_mat - expected_mat).any())

    def test_group_match_mismatch_diversity_angle_both_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': temp_table}
        expected_mat = np.arctan(np.triu(np.ones((6, 6)), k=1))
        angle_mat = group_match_mismatch_diversity_angle(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((angle_mat - expected_mat).any())

    def test_group_match_mismatch_diversity_angle_failure_no_match_table(self):
        temp_tables = {'match': None, 'mismatch': mm_freq_tables['mismatch']}
        with self.assertRaises(AttributeError):
            group_match_mismatch_diversity_angle(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_diversity_angle_failure_no_mismatch_table(self):
        temp_tables = {'match': mm_freq_tables['mismatch'], 'mismatch': None}
        with self.assertRaises(AttributeError):
            group_match_mismatch_diversity_angle(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_diversity_angle_failure_no_tables(self):
        with self.assertRaises(AttributeError):
            group_match_mismatch_diversity_angle(freq_tables={'match': None, 'mismatch': None}, dimensions=(6, 6))

    def test_group_match_mismatch_diversity_angle_failure_table_size_difference(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp}
        with self.assertRaises(ValueError):
            group_match_mismatch_diversity_angle(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_mismatch_diversity_angle_failure_table_size_difference2(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': temp, 'mismatch': mm_freq_tables['match']}
        with self.assertRaises(ValueError):
            group_match_mismatch_diversity_angle(freq_tables=temp_tables, dimensions=(6, 6))


class TestPositionalScorerMatchDiversityMismatchEntropyScores(TestCase):

    def test_group_match_diversity_mismatch_entropy_ratio_score(self):
        expected_value = np.tan(np.pi / 2.0)
        expected_match_diversity = group_match_diversity_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        expected_mismatch_entropy = group_mismatch_entropy_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        # This test does not cover all cases because the simple frequency tables used do not cover all cases. All cases
        # should be covered from the ratio_computation test but it would be good to make that test explicit from this
        # function as well.
        ratio_mat = group_match_diversity_mismatch_entropy_ratio(freq_tables=mm_freq_tables, dimensions=(6, 6))
        for i in range(6):
            for j in range(6):
                match_val = expected_match_diversity[i, j]
                mismatch_val = expected_mismatch_entropy[i, j]
                curr_val = ratio_mat[i, j]
                if (match_val == 0.0) and (mismatch_val != 0.0):  # Case 1
                    self.assertEqual(curr_val, expected_value)
                elif ((match_val != 0.0) and (mismatch_val == 0.0) or  # Case 2
                      (match_val == 0.0) and (mismatch_val == 0.0)):
                    self.assertEqual(curr_val, 0.0)
                else:  # Case 3
                    self.assertEqual(curr_val, mismatch_val / match_val)

    def test_group_match_diversity_mismatch_entropy_ratio_match_zeros(self):
        # This is different than all other metrics testing against this case because the lowest possible value for
        # diversity is 1.0.
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': mm_freq_tables['mismatch']}
        mismatch_entropy = group_mismatch_entropy_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        ratio_mat = group_match_diversity_mismatch_entropy_ratio(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((ratio_mat - mismatch_entropy).any())

    def test_group_match_diversity_mismatch_entropy_ratio_mismatch_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp_table}
        expected_mat = np.zeros((6, 6))
        ratio_mat = group_match_diversity_mismatch_entropy_ratio(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((ratio_mat - expected_mat).any())

    def test_group_match_diversity_mismatch_entropy_ratio_both_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': temp_table}
        expected_mat = np.zeros((6, 6))
        ratio_mat = group_match_diversity_mismatch_entropy_ratio(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((ratio_mat - expected_mat).any())

    def test_group_match_diversity_mismatch_entropy_ratio_failure_no_match_table(self):
        temp_tables = {'match': None, 'mismatch': mm_freq_tables['mismatch']}
        with self.assertRaises(AttributeError):
            group_match_diversity_mismatch_entropy_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_diversity_mismatch_entropy_ratio_failure_no_mismatch_table(self):
        temp_tables = {'match': mm_freq_tables['mismatch'], 'mismatch': None}
        with self.assertRaises(AttributeError):
            group_match_diversity_mismatch_entropy_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_diversity_mismatch_entropy_ratio_failure_no_tables(self):
        with self.assertRaises(AttributeError):
            group_match_diversity_mismatch_entropy_ratio(freq_tables={'match': None, 'mismatch': None}, dimensions=(6, 6))

    def test_group_match_diversity_mismatch_entropy_ratio_failure_table_size_difference(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp}
        with self.assertRaises(ValueError):
            group_match_diversity_mismatch_entropy_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_diversity_mismatch_entropy_ratio_failure_table_size_difference2(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': temp, 'mismatch': mm_freq_tables['match']}
        with self.assertRaises(ValueError):
            group_match_diversity_mismatch_entropy_ratio(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_diversity_mismatch_entropy_angle(self):
        expected_match_diversity = group_match_diversity_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        expected_mismatch_entropy = group_mismatch_entropy_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        # This test does not cover all cases because the simple frequency tables used do not cover all cases. All cases
        # should be covered from the ratio_computation test but it would be good to make that test explicit from this
        # function as well.
        angle_mat = group_match_diversity_mismatch_entropy_angle(freq_tables=mm_freq_tables, dimensions=(6, 6))
        for i in range(6):
            for j in range(6):
                match_val = expected_match_diversity[i, j]
                mismatch_val = expected_mismatch_entropy[i, j]
                curr_val = angle_mat[i, j]
                if (match_val == 0.0) and (mismatch_val != 0.0):  # Case 1
                    self.assertEqual(curr_val, np.pi / 2.0)
                elif ((match_val != 0.0) and (mismatch_val == 0.0) or  # Case 2
                      (match_val == 0.0) and (mismatch_val == 0.0)):
                    self.assertEqual(curr_val, 0.0)
                else:  # Case 3
                    self.assertEqual(curr_val, np.arctan(mismatch_val / match_val))

    def test_group_match_diversity_mismatch_entropy_angle_match_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': mm_freq_tables['mismatch']}
        mismatch_entropy = group_mismatch_entropy_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        angle_mat = group_match_diversity_mismatch_entropy_angle(freq_tables=temp_tables, dimensions=(6, 6))
        expected_angle_mat = angle_computation(ratios=mismatch_entropy)
        self.assertFalse((angle_mat - expected_angle_mat).any())

    def test_group_match_diversity_mismatch_entropy_angle_mismatch_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp_table}
        expected_mat = np.zeros((6, 6))
        angle_mat = group_match_diversity_mismatch_entropy_angle(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((angle_mat - expected_mat).any())

    def test_group_match_diversity_mismatch_entropy_angle_both_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': temp_table, 'mismatch': temp_table}
        expected_mat = np.zeros((6, 6))  # np.arctan(np.triu(np.ones((6, 6)), k=1))
        angle_mat = group_match_diversity_mismatch_entropy_angle(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((angle_mat - expected_mat).any())

    def test_group_match_diversity_mismatch_entropy_angle_failure_no_match_table(self):
        temp_tables = {'match': None, 'mismatch': mm_freq_tables['mismatch']}
        with self.assertRaises(AttributeError):
            group_match_diversity_mismatch_entropy_angle(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_diversity_mismatch_entropy_angle_failure_no_mismatch_table(self):
        temp_tables = {'match': mm_freq_tables['mismatch'], 'mismatch': None}
        with self.assertRaises(AttributeError):
            group_match_diversity_mismatch_entropy_angle(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_diversity_mismatch_entropy_angle_failure_no_tables(self):
        with self.assertRaises(AttributeError):
            group_match_diversity_mismatch_entropy_angle(freq_tables={'match': None, 'mismatch': None}, dimensions=(6, 6))

    def test_group_match_diversity_mismatch_entropy_angle_failure_table_size_difference(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp}
        with self.assertRaises(ValueError):
            group_match_diversity_mismatch_entropy_angle(freq_tables=temp_tables, dimensions=(6, 6))

    def test_group_match_diversity_mismatch_entropy_angle_failure_table_size_difference2(self):
        new_aln_fn = write_out_temp_fasta(
            out_str=f'>seq1\n{str(protein_seq1.seq[:4])}\n>seq2\n{str(protein_seq2.seq[:4])}\n>seq3\n{str(protein_seq3.seq[:4])}')
        new_aln = SeqAlignment(new_aln_fn, 'seq1', polymer_type='Protein')
        new_aln.import_alignment()
        os.remove(new_aln_fn)
        new_num_aln = new_aln._alignment_to_num(mapping=protein_map)
        temp = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 4, 2)
        temp.characterize_alignment(num_aln=new_num_aln, single_to_pair=pro_single_to_pair)
        temp_tables = {'match': temp, 'mismatch': mm_freq_tables['match']}
        with self.assertRaises(ValueError):
            group_match_diversity_mismatch_entropy_angle(freq_tables=temp_tables, dimensions=(6, 6))


class TestPositionalScorerRankIntegerValueScore(TestCase):

    def test_rank_integer_value_score_all_zeros_1d(self):
        score_mat = np.zeros(6)
        ranks = rank_integer_value_score(score_matrix=score_mat, rank=1)
        self.assertFalse((ranks - score_mat).any())

    def test_rank_integer_value_all_zeros_2d(self):
        score_mat = np.zeros((6, 6))
        ranks = rank_integer_value_score(score_matrix=score_mat, rank=1)
        self.assertFalse((ranks - score_mat).any())

    def test_rank_integer_value_score_all_integers_1d(self):
        score_mat = np.random.randint(low=1, high=10, size=6)  # Produce random non-integer values.
        score_mat[:2] *= -1  # Ensure that some are negative.
        score_mat[2:4] = 0  # Ensure that some are zeros.
        ranks = rank_integer_value_score(score_matrix=score_mat, rank=1)
        self.assertFalse((ranks[2:4] - 0).any())  # Assert middle values are zeros.
        self.assertFalse((ranks[:2] - 1).any())  # Assert negative values evaluated to 1.
        self.assertFalse((ranks[4:] - 1).any())  # Assert positive values evaluated to 1.

    def test_rank_integer_value_score_all_integers_2d(self):
        score_mat = np.random.randint(low=1, high=9, size=36).reshape((6, 6))  # Produce random non-integer values.
        score_mat[list(range(6)), list(range(6))] *= -1  # Ensure that some are negative.
        score_mat = np.triu(score_mat)  # Ensure that the lower triangle is all zeros.
        ranks = rank_integer_value_score(score_matrix=score_mat, rank=1)
        self.assertFalse((ranks[np.tril_indices(n=6, k=-1)] - 0).any())  # Assert lower triangle is all zeros.
        self.assertFalse((ranks[list(range(6)), list(range(6))] - 1).any())  # Assert negative values evaluated to 1.
        self.assertFalse((ranks[np.triu_indices(n=6, k=1)] - 1).any())  # Assert positive values evaluated to 1.

    def test_rank_integer_value_score_all_real_valued_1d(self):
        score_mat = np.random.rand(6)  # Produce random non-integer values.
        score_mat[:2] *= -1  # Ensure that some are negative.
        score_mat[2:4] = 0  # Ensure that some are zeros.
        ranks = rank_integer_value_score(score_matrix=score_mat, rank=1)
        self.assertFalse((ranks[2:4] - 0).any())  # Assert middle values are zeros.
        self.assertFalse((ranks[:2] - 1).any())  # Assert negative values evaluated to 1.
        self.assertFalse((ranks[4:] - 1).any())  # Assert positive values evaluated to 1.

    def test_rank_integer_value_score_all_real_valued_2d(self):
        score_mat = np.random.rand(6, 6)  # Produce random non-integer values.
        score_mat[list(range(6)), list(range(6))] *= -1  # Ensure that some are negative.
        score_mat = np.triu(score_mat)  # Ensure that the lower triangle is all zeros.
        ranks = rank_integer_value_score(score_matrix=score_mat, rank=1)
        self.assertFalse((ranks[np.tril_indices(n=6, k=-1)] - 0).any())  # Assert lower triangle is all zeros.
        self.assertFalse((ranks[list(range(6)), list(range(6))] - 1).any())  # Assert negative values evaluated to 1.
        self.assertFalse((ranks[np.triu_indices(n=6, k=1)] - 1).any())  # Assert positive values evaluated to 1.

    def test_rank_integer_value_score_no_score_matrix(self):
        with self.assertRaises(ValueError):
            rank_integer_value_score(score_matrix=None, rank=1)

    def test_rank_integer_value_score_no_rank(self):
        with self.assertRaises(ValueError):
            rank_integer_value_score(score_matrix=np.random.rand(6, 6), rank=None)


class TestPositionalScorerRankRealValueScore(TestCase):

    def test_rank_real_value_score_all_zeros_1d_r1(self):
        score_mat = np.zeros(6)
        ranks = rank_real_value_score(score_matrix=score_mat, rank=1)
        self.assertFalse((ranks - score_mat).any())

    def test_rank_real_value_score_all_zeros_1d_r2(self):
        score_mat = np.zeros(6)
        ranks = rank_real_value_score(score_matrix=score_mat, rank=2)
        self.assertFalse((ranks - score_mat).any())

    def test_rank_real_value_score_all_zeros_2d_r1(self):
        score_mat = np.zeros((6, 6))
        ranks = rank_real_value_score(score_matrix=score_mat, rank=1)
        self.assertFalse((ranks - score_mat).any())

    def test_rank_real_value_score_all_zeros_2d_r2(self):
        score_mat = np.zeros((6, 6))
        ranks = rank_real_value_score(score_matrix=score_mat, rank=2)
        self.assertFalse((ranks - score_mat).any())

    def test_rank_real_value_score_all_integers_1d_r1(self):
        score_mat = np.random.randint(low=1, high=10, size=6)  # Produce random non-integer values.
        score_mat[:2] *= -1  # Ensure that some are negative.
        score_mat[2:4] = 0  # Ensure that some are zeros.
        ranks = rank_real_value_score(score_matrix=score_mat, rank=1)
        self.assertFalse((ranks - score_mat).any())

    def test_rank_real_value_score_all_integers_1d_r2(self):
        score_mat = np.random.randint(low=1, high=10, size=6)  # Produce random non-integer values.
        score_mat[:2] *= -1  # Ensure that some are negative.
        score_mat[2:4] = 0  # Ensure that some are zeros.
        ranks = rank_real_value_score(score_matrix=score_mat, rank=2)
        expected_mat = score_mat / 2.0
        self.assertFalse((ranks - expected_mat).any())

    def test_rank_real_value_score_all_integers_2d_r1(self):
        score_mat = np.random.randint(low=1, high=9, size=36).reshape((6, 6))  # Produce random non-integer values.
        score_mat[list(range(6)), list(range(6))] *= -1  # Ensure that some are negative.
        score_mat = np.triu(score_mat)  # Ensure that the lower triangle is all zeros.
        ranks = rank_real_value_score(score_matrix=score_mat, rank=1)
        self.assertFalse((ranks - score_mat).any())

    def test_rank_real_value_score_all_integers_2d_r2(self):
        score_mat = np.random.randint(low=1, high=9, size=36).reshape((6, 6))  # Produce random non-integer values.
        score_mat[list(range(6)), list(range(6))] *= -1  # Ensure that some are negative.
        score_mat = np.triu(score_mat)  # Ensure that the lower triangle is all zeros.
        ranks = rank_real_value_score(score_matrix=score_mat, rank=2)
        expected_mat = score_mat / 2.0
        self.assertFalse((ranks - expected_mat).any())

    def test_rank_real_value_score_all_real_valued_1d_r1(self):
        score_mat = np.random.rand(6)  # Produce random non-integer values.
        score_mat[:2] *= -1  # Ensure that some are negative.
        score_mat[2:4] = 0  # Ensure that some are zeros.
        ranks = rank_real_value_score(score_matrix=score_mat, rank=1)
        self.assertFalse((ranks - score_mat).any())

    def test_rank_real_value_score_all_real_valued_1d_r2(self):
        score_mat = np.random.rand(6)  # Produce random non-integer values.
        score_mat[:2] *= -1  # Ensure that some are negative.
        score_mat[2:4] = 0  # Ensure that some are zeros.
        ranks = rank_real_value_score(score_matrix=score_mat, rank=2)
        expected_mat = score_mat / 2.0
        self.assertFalse((ranks - expected_mat).any())

    def test_rank_real_value_score_all_real_valued_2d_r1(self):
        score_mat = np.random.rand(6, 6)  # Produce random non-integer values.
        score_mat[list(range(6)), list(range(6))] *= -1  # Ensure that some are negative.
        score_mat = np.triu(score_mat)  # Ensure that the lower triangle is all zeros.
        ranks = rank_real_value_score(score_matrix=score_mat, rank=1)
        self.assertFalse((ranks - score_mat).any())

    def test_rank_real_value_score_all_real_valued_2d_r2(self):
        score_mat = np.random.rand(6, 6)  # Produce random non-integer values.
        score_mat[list(range(6)), list(range(6))] *= -1  # Ensure that some are negative.
        score_mat = np.triu(score_mat)  # Ensure that the lower triangle is all zeros.
        ranks = rank_real_value_score(score_matrix=score_mat, rank=2)
        expected_mat = score_mat / 2.0
        self.assertFalse((ranks - expected_mat).any())

    def test_rank_real_value_score_no_score_matrix(self):
        with self.assertRaises(ValueError):
            rank_real_value_score(score_matrix=None, rank=1)

    def test_rank_real_value_score_no_rank(self):
        with self.assertRaises(ValueError):
            rank_real_value_score(score_matrix=np.random.rand(6, 6), rank=None)


class TestPositionalScorerScoreGroup(TestCase):

    def test_score_group_identity_pos_single(self):
        ps = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        scores = ps.score_group(freq_table=pro_single_ft)
        expected_scores = group_identity_score(freq_table=pro_single_ft, dimensions=(6, ))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_identity_failure_mismatch_large(self):
        ps = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_pair_ft)

    def test_score_group_identity_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        scores = ps.score_group(freq_table=pro_pair_ft)
        expected_scores = group_identity_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_identity_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)

    def test_score_group_plain_entropy_pos_single(self):
        ps = PositionalScorer(seq_length=6, pos_size=1, metric='plain_entropy')
        scores = ps.score_group(freq_table=pro_single_ft)
        expected_scores = group_plain_entropy_score(freq_table=pro_single_ft, dimensions=(6, ))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_plain_entropy_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)

    def test_score_group_plain_entropy_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        scores = ps.score_group(freq_table=pro_pair_ft)
        expected_scores = group_plain_entropy_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_mutual_information_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mutual_information')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)

    def test_score_group_mutual_information_entropy_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mutual_information')
        scores = ps.score_group(freq_table=pro_pair_ft)
        expected_scores = group_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_normalized_mutual_information_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='normalized_mutual_information')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)

    def test_score_group_normalized_mutual_information_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='normalized_mutual_information')
        scores = ps.score_group(freq_table=pro_pair_ft)
        expected_scores = group_normalized_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_average_product_corrected_mutual_information_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='average_product_corrected_mutual_information')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)

    def test_score_group_average_product_corrected_mutual_information_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='average_product_corrected_mutual_information')
        scores = ps.score_group(freq_table=pro_pair_ft)
        expected_intermediate = group_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        expected_scores = average_product_correction(expected_intermediate)
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_filtered_average_product_corrected_mutual_information_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='filtered_average_product_corrected_mutual_information')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)

    def test_score_group_filtered_average_product_corrected_mutual_information_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='filtered_average_product_corrected_mutual_information')
        scores = ps.score_group(freq_table=pro_pair_ft)
        expected_intermediate = group_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        expected_scores = filtered_average_product_correction(expected_intermediate)
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_match_count_failure_single_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_count')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': None})

    def test_score_group_match_count_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_count')
        scores = ps.score_group(freq_table=mm_freq_tables)
        expected_scores = group_match_count_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_mismatch_count_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_count')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': None, 'mismatch': pro_single_ft})

    def test_score_group_mismatch_count_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_count')
        scores = ps.score_group(freq_table=mm_freq_tables)
        expected_scores = group_mismatch_count_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_match_mismatch_count_ratio_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_ratio')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})

    def test_score_group_match_mismatch_count_ratio_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_ratio')
        scores = ps.score_group(freq_table=mm_freq_tables)
        expected_scores = group_match_mismatch_count_ratio(freq_tables=mm_freq_tables, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_match_mismatch_count_angle_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_angle')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})

    def test_score_group_match_mismatch_count_angle_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_angle')
        scores = ps.score_group(freq_table=mm_freq_tables)
        expected_scores = group_match_mismatch_count_angle(freq_tables=mm_freq_tables, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_match_entropy_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_entropy')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': None})

    def test_score_group_match_entropy_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_entropy')
        scores = ps.score_group(freq_table=mm_freq_tables)
        expected_scores = group_match_entropy_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_mismatch_entropy_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_entropy')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': None, 'mismatch': pro_single_ft})

    def test_score_group_mismatch_entropy_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_entropy')
        scores = ps.score_group(freq_table=mm_freq_tables)
        expected_scores = group_mismatch_entropy_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_match_mismatch_entropy_ratio_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_ratio')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})

    def test_score_group_match_mismatch_entropy_ratio_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_ratio')
        scores = ps.score_group(freq_table=mm_freq_tables)
        expected_scores = group_match_mismatch_entropy_ratio(freq_tables=mm_freq_tables, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_match_mismatch_entropy_angle_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_angle')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})

    def test_score_group_match_mismatch_entropy_angle_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_angle')
        scores = ps.score_group(freq_table=mm_freq_tables)
        expected_scores = group_match_mismatch_entropy_angle(freq_tables=mm_freq_tables, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_match_diversity_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': None})

    def test_score_group_match_diversity_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity')
        scores = ps.score_group(freq_table=mm_freq_tables)
        expected_scores = group_match_diversity_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_mismatch_diversity_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_diversity')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': None, 'mismatch': pro_single_ft})

    def test_score_group_mismatch_diversity_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_diversity')
        scores = ps.score_group(freq_table=mm_freq_tables)
        expected_scores = group_mismatch_diversity_score(freq_tables=mm_freq_tables, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_match_mismatch_diversity_ratio_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_ratio')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})

    def test_score_group_match_mismatch_diversity_ratio_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_ratio')
        scores = ps.score_group(freq_table=mm_freq_tables)
        expected_scores = group_match_mismatch_diversity_ratio(freq_tables=mm_freq_tables, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_match_mismatch_diversity_angle_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_angle')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})

    def test_score_group_match_mismatch_diversity_angle_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_angle')
        scores = ps.score_group(freq_table=mm_freq_tables)
        expected_scores = group_match_mismatch_diversity_angle(freq_tables=mm_freq_tables, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_match_diversity_mismatch_entropy_ratio_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_ratio')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})

    def test_score_group_match_diversity_mismatch_entropy_ratio_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_ratio')
        scores = ps.score_group(freq_table=mm_freq_tables)
        expected_scores = group_match_diversity_mismatch_entropy_ratio(freq_tables=mm_freq_tables, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_match_diversity_mismatch_entropy_angle_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_angle')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})

    def test_score_group_match_diversity_mismatch_entropy_angle_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_angle')
        scores = ps.score_group(freq_table=mm_freq_tables)
        expected_scores = group_match_diversity_mismatch_entropy_angle(freq_tables=mm_freq_tables, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())


class TestPositionalScorerScoreRank(TestCase):

    def test_score_rank_identity_pos_single(self):
        ps = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        scores = ps.score_group(freq_table=pro_single_ft)
        ranks = ps.score_rank(score_tensor=scores, rank=2)
        expected_ranks = rank_integer_value_score(scores, 2)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_identity_failure_mismatch_large(self):
        ps = PositionalScorer(seq_length=6, pos_size=1, metric='identity')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4), 2)

    def test_score_rank_identity_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        scores = ps.score_group(freq_table=pro_pair_ft)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_integer_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_identity_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_plain_entropy_pos_single(self):
        ps = PositionalScorer(seq_length=6, pos_size=1, metric='plain_entropy')
        scores = ps.score_group(freq_table=pro_single_ft)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = rank_real_value_score(scores, 2)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_plain_entropy_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4), 2)

    def test_score_rank_plain_entropy_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        scores = ps.score_group(freq_table=pro_pair_ft)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_mutual_information_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mutual_information')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_mutual_information_entropy_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mutual_information')
        scores = ps.score_group(freq_table=pro_pair_ft)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = rank_real_value_score(scores, 2)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_normalized_mutual_information_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='normalized_mutual_information')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_normalized_mutual_information_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='normalized_mutual_information')
        scores = ps.score_group(freq_table=pro_pair_ft)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = rank_real_value_score(scores, 2)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_average_product_corrected_mutual_information_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='average_product_corrected_mutual_information')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4,4), 2)

    def test_score_rank_average_product_corrected_mutual_information_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='average_product_corrected_mutual_information')
        scores = ps.score_group(freq_table=pro_pair_ft)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2))
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_filtered_average_product_corrected_mutual_information_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='filtered_average_product_corrected_mutual_information')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_filtered_average_product_corrected_mutual_information_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='filtered_average_product_corrected_mutual_information')
        scores = ps.score_group(freq_table=pro_pair_ft)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_match_count_failure_single_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_count')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_match_count_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_count')
        scores = ps.score_group(freq_table=mm_freq_tables)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_mismatch_count_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_count')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_mismatch_count_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_count')
        scores = ps.score_group(freq_table=mm_freq_tables)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_match_mismatch_count_ratio_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_ratio')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_match_mismatch_count_ratio_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_ratio')
        scores = ps.score_group(freq_table=mm_freq_tables)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_match_mismatch_count_angle_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_angle')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_match_mismatch_count_angle_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_angle')
        scores = ps.score_group(freq_table=mm_freq_tables)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_match_entropy_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_entropy')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_match_entropy_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_entropy')
        scores = ps.score_group(freq_table=mm_freq_tables)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_mismatch_entropy_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_entropy')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_mismatch_entropy_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_entropy')
        scores = ps.score_group(freq_table=mm_freq_tables)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_match_mismatch_entropy_ratio_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_ratio')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_match_mismatch_entropy_ratio_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_ratio')
        scores = ps.score_group(freq_table=mm_freq_tables)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_match_mismatch_entropy_angle_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_angle')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_match_mismatch_entropy_angle_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_angle')
        scores = ps.score_group(freq_table=mm_freq_tables)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_match_diversity_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_group_match_diversity_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity')
        scores = ps.score_group(freq_table=mm_freq_tables)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_rank_group_mismatch_diversity_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_diversity')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_rank_group_mismatch_diversity_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_diversity')
        scores = ps.score_group(freq_table=mm_freq_tables)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_match_mismatch_diversity_ratio_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_ratio')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_match_mismatch_diversity_ratio_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_ratio')
        scores = ps.score_group(freq_table=mm_freq_tables)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_match_mismatch_diversity_angle_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_angle')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_match_mismatch_diversity_angle_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_angle')
        scores = ps.score_group(freq_table=mm_freq_tables)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_match_diversity_mismatch_entropy_ratio_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_ratio')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_match_diversity_mismatch_entropy_ratio_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_ratio')
        scores = ps.score_group(freq_table=mm_freq_tables)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())

    def test_score_rank_match_diversity_mismatch_entropy_angle_failure_mismatch_small(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_angle')
        with self.assertRaises(ValueError):
            ps.score_rank(np.random.rand(4, 4), 2)

    def test_score_rank_match_diversity_mismatch_entropy_angle_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_angle')
        scores = ps.score_group(freq_table=mm_freq_tables)
        ranks = ps.score_rank(scores, 2)
        expected_ranks = np.triu(rank_real_value_score(scores, 2), k=1)
        self.assertFalse((ranks - expected_ranks).any())


if __name__ == '__main__':
    unittest.main()
