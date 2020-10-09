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
        counts = group_match_count_score(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        triu_ind = np.triu_indices(n=counts.shape[0], k=1)
        for x in range(counts.shape[0]):
            i = triu_ind[0][x]
            j = triu_ind[1][x]
            self.assertEqual(counts[i, j], expected_counts[i, j])
        self.assertFalse((np.tril(counts) - np.zeros(counts.shape)).any())

    def test_group_match_count_score_failure_single_pos_input(self):
        with self.assertRaises(ValueError):
            group_match_count_score(freq_table=pro_single_ft, dimensions=(6,))

    def test_group_match_count_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_match_count_score(freq_table=None, dimensions=(6,))

    def test_group_match_count_score_failure_mismatch_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_match_count_score(freq_table=pro_single_ft, dimensions=(6, 6))

    def test_group_match_count_score_failure_mismatch_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_match_count_score(freq_table=mm_freq_tables['match'], dimensions=(6,))

    def test_group_mismatch_count_score(self):
        expected_counts = count_computation(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
        counts = group_mismatch_count_score(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
        triu_ind = np.triu_indices(n=counts.shape[0], k=1)
        for x in range(counts.shape[0]):
            i = triu_ind[0][x]
            j = triu_ind[1][x]
            self.assertEqual(counts[i, j], expected_counts[i, j])
        self.assertFalse((np.tril(counts) - np.zeros(counts.shape)).any())

    def test_group_mismatch_count_score_failure_single_pos_input(self):
        with self.assertRaises(ValueError):
            group_mismatch_count_score(freq_table=pro_single_ft, dimensions=(6,))

    def test_group_mismatch_count_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_mismatch_count_score(freq_table=None, dimensions=(6, 6))

    def test_group_mismatch_count_score_failure_mismatch_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_mismatch_count_score(freq_table=pro_single_ft, dimensions=(6, 6))

    def test_group_mismatch_count_score_failure_mismatch_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_mismatch_count_score(freq_table=mm_freq_tables['match'], dimensions=(6,))

    def test_group_match_mismatch_count_ratio_score(self):
        expected_value = np.tan(np.pi / 2.0)
        expected_match_count = count_computation(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        expected_mismatch_count = count_computation(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
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
        mismatch_counts = group_mismatch_count_score(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
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
        expected_value = np.tan(np.pi / 2.0)
        expected_match_count = count_computation(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        expected_mismatch_count = count_computation(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
        # This test does not cover all cases because the simple frequency tables used do not cover all cases. All cases
        # should be covered from the ratio_computation test but it would be good to make that test explicit from this
        # function as well.
        expected_ratios = group_match_mismatch_count_ratio(freq_tables=mm_freq_tables, dimensions=(6, 6))
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
        mismatch_counts = group_mismatch_count_score(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
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
        entropy = group_match_entropy_score(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        triu_ind = np.triu_indices(n=entropy.shape[0], k=1)
        for x in range(entropy.shape[0]):
            i = triu_ind[0][x]
            j = triu_ind[1][x]
            self.assertEqual(entropy[i, j], expected_entropy[i, j])
        self.assertFalse((np.tril(entropy) - np.zeros(entropy.shape)).any())

    def test_group_match_entropy_score_failure_single_pos_input(self):
        with self.assertRaises(ValueError):
            group_match_entropy_score(freq_table=pro_single_ft, dimensions=(6,))

    def test_group_match_entropy_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_match_entropy_score(freq_table=None, dimensions=(6, 6))

    def test_group_match_entropy_score_failure_mismatch_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_match_entropy_score(freq_table=pro_single_ft, dimensions=(6, 6))

    def test_group_match_entropy_score_failure_mismatch_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_match_entropy_score(freq_table=mm_freq_tables['match'], dimensions=(6,))

    def test_group_mismatch_entropy_score(self):
        expected_entropy = group_plain_entropy_score(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
        entropy = group_mismatch_entropy_score(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
        triu_ind = np.triu_indices(n=entropy.shape[0], k=1)
        for x in range(entropy.shape[0]):
            i = triu_ind[0][x]
            j = triu_ind[1][x]
            self.assertEqual(entropy[i, j], expected_entropy[i, j])
        self.assertFalse((np.tril(entropy) - np.zeros(entropy.shape)).any())

    def test_group_mismatch_entropy_score_failure_single_pos_input(self):
        with self.assertRaises(ValueError):
            group_mismatch_entropy_score(freq_table=pro_single_ft, dimensions=(6,))

    def test_group_mismatch_entropy_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_mismatch_entropy_score(freq_table=None, dimensions=(6, 6))

    def test_group_mismatch_entropy_score_failure_mismatch_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_mismatch_entropy_score(freq_table=pro_single_ft, dimensions=(6, 6))

    def test_group_mismatch_entropy_score_failure_mismatch_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_mismatch_entropy_score(freq_table=mm_freq_tables['match'], dimensions=(6,))

    def test_group_match_mismatch_entropy_ratio_score(self):
        expected_value = np.tan(np.pi / 2.0)
        expected_match_count = group_plain_entropy_score(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        expected_mismatch_count = group_plain_entropy_score(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
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
        mismatch_counts = group_mismatch_entropy_score(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
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
        expected_value = np.tan(np.pi / 2.0)
        expected_match_entropy = group_plain_entropy_score(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        expected_mismatch_entropy = group_plain_entropy_score(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
        # This test does not cover all cases because the simple frequency tables used do not cover all cases. All cases
        # should be covered from the ratio_computation test but it would be good to make that test explicit from this
        # function as well.
        expected_ratios = group_match_mismatch_entropy_ratio(freq_tables=mm_freq_tables, dimensions=(6, 6))
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
        mismatch_counts = group_mismatch_entropy_score(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
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
        diversity = group_match_diversity_score(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        triu_ind = np.triu_indices(n=diversity.shape[0], k=1)
        for x in range(diversity.shape[0]):
            i = triu_ind[0][x]
            j = triu_ind[1][x]
            self.assertEqual(diversity[i, j], expected_diversity[i, j])
        self.assertFalse((np.tril(diversity) - np.zeros(diversity.shape)).any())

    def test_group_match_diversity_score_failure_single_pos_input(self):
        with self.assertRaises(ValueError):
            group_match_diversity_score(freq_table=pro_single_ft, dimensions=(6,))

    def test_group_match_diversity_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_match_diversity_score(freq_table=None, dimensions=(6, 6))

    def test_group_match_diversity_score_failure_mismatch_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_match_diversity_score(freq_table=pro_single_ft, dimensions=(6, 6))

    def test_group_match_diversity_score_failure_mismatch_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_match_diversity_score(freq_table=mm_freq_tables['match'], dimensions=(6,))

    def test_group_mismatch_diversity_score(self):
        expected_diversity = diversity_computation(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
        diversity = group_mismatch_diversity_score(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
        triu_ind = np.triu_indices(n=diversity.shape[0], k=1)
        for x in range(diversity.shape[0]):
            i = triu_ind[0][x]
            j = triu_ind[1][x]
            self.assertEqual(diversity[i, j], expected_diversity[i, j])
        self.assertFalse((np.tril(diversity) - np.zeros(diversity.shape)).any())

    def test_group_mismatch_diversity_score_failure_single_pos_input(self):
        with self.assertRaises(ValueError):
            group_mismatch_diversity_score(freq_table=pro_single_ft, dimensions=(6,))

    def test_group_mismatch_diversity_score_failure_no_freq_table(self):
        with self.assertRaises(AttributeError):
            group_mismatch_diversity_score(freq_table=None, dimensions=(6, 6))

    def test_group_mismatch_diversity_score_failure_mismatch_dimensions_large(self):
        with self.assertRaises(ValueError):
            group_mismatch_diversity_score(freq_table=pro_single_ft, dimensions=(6, 6))

    def test_group_mismatch_diversity_score_failure_mismatch_dimensions_small(self):
        with self.assertRaises(ValueError):
            group_mismatch_diversity_score(freq_table=mm_freq_tables['match'], dimensions=(6,))

    def test_group_match_mismatch_diversity_ratio_score(self):
        expected_value = np.tan(np.pi / 2.0)
        expected_match_diversity = diversity_computation(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        expected_mismatch_diversity = diversity_computation(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
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
        mismatch_diversity = diversity_computation(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
        ratio_mat = group_match_mismatch_diversity_ratio(freq_tables=temp_tables, dimensions=(6, 6))
        self.assertFalse((ratio_mat - mismatch_diversity).any())

    def test_group_match_mismatch_diversity_ratio_mismatch_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp_table}
        match_diversity = diversity_computation(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
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
        expected_match_diversity = diversity_computation(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        expected_mismatch_diversity = diversity_computation(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
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
        mismatch_diversity = group_mismatch_diversity_score(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
        angle_mat = group_match_mismatch_diversity_angle(freq_tables=temp_tables, dimensions=(6, 6))
        expected_angle_mat = angle_computation(ratios=mismatch_diversity)
        self.assertFalse((angle_mat - expected_angle_mat).any())

    def test_group_match_mismatch_diversity_angle_mismatch_zeros(self):
        temp_table = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
        temp_table.set_depth(3.0)
        temp_table.finalize_table()
        temp_tables = {'match': mm_freq_tables['match'], 'mismatch': temp_table}
        match_diversity = group_match_diversity_score(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
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
        expected_match_diversity = group_match_diversity_score(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        expected_mismatch_entropy = group_mismatch_entropy_score(freq_table=mm_freq_tables['mismatch'],
                                                                   dimensions=(6, 6))
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
        mismatch_entropy = group_mismatch_entropy_score(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
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
        expected_match_diversity = diversity_computation(freq_table=mm_freq_tables['match'], dimensions=(6, 6))
        expected_mismatch_entropy = group_mismatch_entropy_score(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
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
        mismatch_entropy = group_mismatch_entropy_score(freq_table=mm_freq_tables['mismatch'], dimensions=(6, 6))
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

    def test_score_group_identity_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='identity')
        scores = ps.score_group(freq_table=pro_pair_ft)
        expected_scores = group_identity_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_plain_entropy_pos_single(self):
        ps = PositionalScorer(seq_length=6, pos_size=1, metric='plain_entropy')
        scores = ps.score_group(freq_table=pro_single_ft)
        expected_scores = group_plain_entropy_score(freq_table=pro_single_ft, dimensions=(6, ))
        self.assertFalse((scores - expected_scores).any())

    def test_score_group_plain_entropy_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)

    def test_score_group_plain_entropy_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='plain_entropy')
        scores = ps.score_group(freq_table=pro_pair_ft)
        expected_scores = group_plain_entropy_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_mutual_information_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mutual_information')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)

    def test_score_mutual_information_entropy_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mutual_information')
        scores = ps.score_group(freq_table=pro_pair_ft)
        expected_scores = group_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_normalized_mutual_information_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='normalized_mutual_information')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)

    def test_score_normalized_mutual_information_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='normalized_mutual_information')
        scores = ps.score_group(freq_table=pro_pair_ft)
        expected_scores = group_normalized_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_average_product_corrected_mutual_information_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='average_product_corrected_mutual_information')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)

    def test_score_average_product_corrected_mutual_information_pos_pair(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='average_product_corrected_mutual_information')
        scores = ps.score_group(freq_table=pro_pair_ft)
        expected_scores = average_product_corrected_mutual_information_score(freq_table=pro_pair_ft, dimensions=(6, 6))
        self.assertFalse((scores - expected_scores).any())

    def test_score_filtered_average_product_corrected_mutual_information_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='filtered_average_product_corrected_mutual_information')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)

    # def test_score_filtered_average_product_corrected_mutual_information_pos_pair(self):
    #     ps = PositionalScorer(seq_length=6, pos_size=2, metric='filtered_average_product_corrected_mutual_information')
    #
    def test_score_group_match_count_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_count')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)

    # def test_score_group_match_count_pos_pair(self):
    #     ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_count')
    #
    def test_score_group_mismatch_count_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_count')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)

    # def test_score_group_mismatch_count_pos_pair(self):
    #     ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_count')
    #
    def test_score_group_match_mismatch_count_ratio_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_ratio')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})

    # def test_score_group_match_mismatch_count_ratio_pos_pair(self):
    #     ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_ratio')
    #
    def test_score_group_match_mismatch_count_angle_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_angle')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})
    # def test_score_group_match_mismatch_count_angle_pos_pair(self):
    #     ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_count_angle')
    #
    def test_score_group_match_entropy_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_entropy')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)

    # def test_score_group_match_entropy_pos_pair(self):
    #     ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_entropy')
    #
    def test_score_group_mismatch_entropy_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_entropy')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)
    # def test_score_group_mismatch_entropy_pos_pair(self):
    #     ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_entropy')
    #
    def test_score_group_match_mismatch_entropy_ratio_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_ratio')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})
    # def test_score_group_match_mismatch_entropy_ratio_pos_pair(self):
    #     ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_ratio')
    #
    def test_score_group_match_mismatch_entropy_angle_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_angle')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})
    # def test_score_group_match_mismatch_entropy_angle_pos_pair(self):
    #     ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_entropy_angle')
    #
    def test_score_group_match_diversity_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)
    # def test_score_group_match_diversity_pos_pair(self):
    #     ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity')
    #
    def test_score_group_mismatch_diversity_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_diversity')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table=pro_single_ft)
    # def test_score_group_mismatch_diversity_pos_pair(self):
    #     ps = PositionalScorer(seq_length=6, pos_size=2, metric='mismatch_diversity')
    #
    def test_score_group_match_mismatch_diversity_ratio_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_ratio')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})
    # def test_score_group_match_mismatch_diversity_ratio_pos_pair(self):
    #     ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_ratio')
    #
    def test_score_group_match_mismatch_diversity_angle_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_angle')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})
    # def test_score_group_match_mismatch_diversity_angle_pos_pair(self):
    #     ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_mismatch_diversity_angle')
    #
    def test_score_group_match_diversity_mismatch_entropy_ratio_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_ratio')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})
    # def test_score_group_match_diversity_mismatch_entropy_ratio_pos_pair(self):
    #     ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_ratio')
    #
    def test_score_group_match_diversity_mismatch_entropy_angle_failure_single_freq_table(self):
        ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_angle')
        with self.assertRaises(ValueError):
            ps.score_group(freq_table={'match': pro_single_ft, 'mismatch': pro_single_ft})
    # def test_score_group_match_diversity_mismatch_entropy_angle_pos_pair(self):
    #     ps = PositionalScorer(seq_length=6, pos_size=2, metric='match_diversity_mismatch_entropy_angle')

# class TestPositionalScorerScoreRank(TestCase):


# class TestPositionalScorer(TestBase):
#
#     @classmethod
#     def setUpClass(cls):
#         super(TestPositionalScorer, cls).setUpClass()
#
#         cls.single_alphabet = Gapped(FullIUPACProtein())
#         cls.single_size, _, cls.single_mapping, cls.single_reverse = build_mapping(alphabet=cls.single_alphabet)
#         cls.pair_alphabet = MultiPositionAlphabet(alphabet=cls.single_alphabet, size=2)
#         cls.pair_size, _, cls.pair_mapping, cls.pair_reverse = build_mapping(alphabet=cls.pair_alphabet)
#         cls.single_to_pair = {}
#         for char in cls.pair_mapping:
#             key = (cls.single_mapping[char[0]], cls.single_mapping[char[1]])
#             cls.single_to_pair[key] = cls.pair_mapping[char]
#         cls.quad_alphabet = MultiPositionAlphabet(alphabet=cls.single_alphabet, size=4)
#         cls.quad_size, _, cls.quad_mapping, cls.quad_reverse = build_mapping(alphabet=cls.quad_alphabet)
#         cls.single_to_quad = {}
#         for char in cls.quad_mapping:
#             key = (cls.single_mapping[char[0]], cls.single_mapping[char[1]], cls.single_mapping[char[2]],
#                    cls.single_mapping[char[3]])
#             cls.single_to_quad[key] = cls.quad_mapping[char]
#
#         cls.query_aln_fa_small = SeqAlignment(
#             file_name=cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln'],
#             query_id=cls.small_structure_id)
#         cls.query_aln_fa_small.import_alignment()
#         cls.phylo_tree_small = PhylogeneticTree()
#         calc = AlignmentDistanceCalculator()
#         cls.phylo_tree_small.construct_tree(dm=calc.get_distance(cls.query_aln_fa_small.alignment))
#         cls.query_aln_fa_small = cls.query_aln_fa_small.remove_gaps()
#         cls.terminals = {x.name: {'aln': cls.query_aln_fa_small.generate_sub_alignment(sequence_ids=[x.name]),
#                                   'node': x} for x in cls.phylo_tree_small.tree.get_terminals()}
#         for x in cls.terminals:
#             single, pair = cls.terminals[x]['aln'].characterize_positions(
#                 single=True, pair=True, single_size=cls.single_size, single_mapping=cls.single_mapping,
#                 single_reverse=cls.single_reverse, pair_size=cls.pair_size, pair_mapping=cls.pair_mapping,
#                 pair_reverse=cls.pair_reverse)
#             cls.terminals[x]['single'] = single
#             cls.terminals[x]['pair'] = pair
#         potential_parents = set()
#         for t in cls.terminals:
#             path = cls.phylo_tree_small.tree.get_path(t)
#             if len(path) >= 2:
#                 potential_parents.add(path[-2])
#         cls.first_parents = {}
#         for parent in potential_parents:
#             if parent.clades[0].is_terminal() and parent.clades[1].is_terminal():
#                 cls.first_parents[parent.name] = {'node': parent, 'aln': cls.query_aln_fa_small.generate_sub_alignment(
#                     sequence_ids=[y.name for y in parent.clades])}
#                 cls.first_parents[parent.name]['single'] = (cls.terminals[parent.clades[0].name]['single'] +
#                                                             cls.terminals[parent.clades[1].name]['single'])
#                 cls.first_parents[parent.name]['pair'] = (cls.terminals[parent.clades[0].name]['pair'] +
#                                                           cls.terminals[parent.clades[1].name]['pair'])
#         cls.seq_len = cls.query_aln_fa_small.seq_length
#         cls.mm_table = MatchMismatchTable(seq_len=cls.query_aln_fa_small.seq_length,
#                                           num_aln=cls.query_aln_fa_small._alignment_to_num(cls.single_mapping),
#                                           single_alphabet_size=cls.single_size, single_mapping=cls.single_mapping,
#                                           single_reverse_mapping=cls.single_reverse, larger_alphabet_size=cls.quad_size,
#                                           larger_alphabet_mapping=cls.quad_mapping,
#                                           larger_alphabet_reverse_mapping=cls.quad_reverse,
#                                           single_to_larger_mapping=cls.single_to_quad, pos_size=2)
#         cls.mm_table.identify_matches_mismatches()
#         for x in cls.terminals:
#             cls.terminals[x]['match'] = FrequencyTable(alphabet_size=cls.quad_size, mapping=cls.quad_mapping,
#                                                        reverse_mapping=cls.quad_reverse,
#                                                        seq_len=cls.query_aln_fa_small.seq_length, pos_size=2)
#             cls.terminals[x]['match'].mapping = cls.quad_mapping
#             cls.terminals[x]['match'].set_depth(1)
#             cls.terminals[x]['mismatch'] = deepcopy(cls.terminals[x]['match'])
#             for pos in cls.terminals[x]['match'].get_positions():
#                 char_dict = {'match': {}, 'mismatch': {}}
#                 for i in range(cls.terminals[x]['aln'].size):
#                     s1 = cls.query_aln_fa_small.seq_order.index(cls.terminals[x]['aln'].seq_order[i])
#                     for j in range(i + 1, cls.terminals[x]['aln'].size):
#                         s2 = cls.query_aln_fa_small.seq_order.index(cls.terminals[x]['aln'].seq_order[j])
#                         status, char = cls.mm_table.get_status_and_character(pos=pos, seq_ind1=s1, seq_ind2=s2)
#                         if char not in char_dict[status]:
#                             char_dict[status][char] = 0
#                         char_dict[status][char] += 1
#                 for m in char_dict:
#                     for char in char_dict[m]:
#                         cls.terminals[x][m]._increment_count(pos=pos, char=char, amount=char_dict[m][char])
#             for m in ['match', 'mismatch']:
#                 cls.terminals[x][m].finalize_table()
#         for x in cls.first_parents:
#             cls.first_parents[x]['match'] = FrequencyTable(alphabet_size=cls.quad_size, mapping=cls.quad_mapping,
#                                                            reverse_mapping=cls.quad_reverse,
#                                                            seq_len=cls.query_aln_fa_small.seq_length, pos_size=2)
#             cls.first_parents[x]['match'].mapping = cls.quad_mapping
#             cls.first_parents[x]['match'].set_depth(((cls.first_parents[x]['aln'].size**2) -
#                                                      cls.first_parents[x]['aln'].size) / 2.0)
#             cls.first_parents[x]['mismatch'] = deepcopy(cls.first_parents[x]['match'])
#             for pos in cls.first_parents[x]['match'].get_positions():
#                 char_dict = {'match': {}, 'mismatch': {}}
#                 for i in range(cls.first_parents[x]['aln'].size):
#                     s1 = cls.query_aln_fa_small.seq_order.index(cls.first_parents[x]['aln'].seq_order[i])
#                     for j in range(i + 1, cls.first_parents[x]['aln'].size):
#                         s2 = cls.query_aln_fa_small.seq_order.index(cls.first_parents[x]['aln'].seq_order[j])
#                         status, char = cls.mm_table.get_status_and_character(pos=pos, seq_ind1=s1, seq_ind2=s2)
#                         if char not in char_dict[status]:
#                             char_dict[status][char] = 0
#                         char_dict[status][char] += 1
#                 for m in char_dict:
#                     for char in char_dict[m]:
#                         cls.first_parents[x][m]._increment_count(pos=pos, char=char, amount=char_dict[m][char])
#             for m in ['match', 'mismatch']:
#                 cls.first_parents[x][m].finalize_table()
#
#     def evaluate__init(self, seq_len, pos_size, metric, metric_type, rank_type):
#         if metric == 'fake':
#             with self.assertRaises(ValueError):
#                 PositionalScorer(seq_length=seq_len, pos_size=pos_size, metric=metric)
#         else:
#             pos_scorer = PositionalScorer(seq_length=seq_len, pos_size=pos_size, metric=metric)
#             self.assertEqual(pos_scorer.sequence_length, seq_len)
#             self.assertEqual(pos_scorer.position_size, pos_size)
#             if pos_size == 1:
#                 self.assertEqual(pos_scorer.dimensions, (seq_len,))
#             else:
#                 self.assertEqual(pos_scorer.dimensions, (seq_len, seq_len))
#             self.assertEqual(pos_scorer.metric, metric)
#             self.assertEqual(pos_scorer.metric_type, metric_type)
#             self.assertEqual(pos_scorer.rank_type, rank_type)
#
#     def test1a_init(self):
#         self.evaluate__init(seq_len=self.seq_len, pos_size=1, metric='identity', metric_type='integer', rank_type='min')
#
#     def test1b_init(self):
#         self.evaluate__init(seq_len=self.seq_len, pos_size=2, metric='identity', metric_type='integer', rank_type='min')
#
#     def test1c_init(self):
#         self.evaluate__init(seq_len=self.seq_len, pos_size=1, metric='fake', metric_type='integer', rank_type='min')
#
#     def test1d_init(self):
#         self.evaluate__init(seq_len=self.seq_len, pos_size=2, metric='fake', metric_type='integer', rank_type='min')
#
#     def test1e_init(self):
#         self.evaluate__init(seq_len=self.seq_len, pos_size=1, metric='plain_entropy', metric_type='real',
#                             rank_type='min')
#
#     def test1f_init(self):
#         self.evaluate__init(seq_len=self.seq_len, pos_size=2, metric='plain_entropy', metric_type='real',
#                             rank_type='min')
#
#     def test1g_init(self):
#         self.evaluate__init(seq_len=self.seq_len, pos_size=2, metric='mutual_information', metric_type='real',
#                             rank_type='max')
#
#     def test1h_init(self):
#         self.evaluate__init(seq_len=self.seq_len, pos_size=2, metric='normalized_mutual_information',
#                             metric_type='real', rank_type='max')
#
#     def test1i_init(self):
#         self.evaluate__init(seq_len=self.seq_len, pos_size=2, metric='average_product_corrected_mutual_information',
#                             metric_type='real', rank_type='max')
#
#     def test1j_init(self):
#         self.evaluate__init(seq_len=self.seq_len, pos_size=2,
#                             metric='filtered_average_product_corrected_mutual_information',
#                             metric_type='real', rank_type='max')
#
#     def test1k_init(self):
#         self.evaluate__init(seq_len=self.seq_len, pos_size=2, metric='match_mismatch_entropy_angle', metric_type='real',
#                             rank_type='min')
#
#     def evaluate_score_group_ambiguous(self, node_dict, metric):
#         pos_scorer_single = PositionalScorer(seq_length=self.seq_len, pos_size=1, metric=metric)
#         dim_single = (self.seq_len,)
#         pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric=metric)
#         dim_pair = (self.seq_len, self.seq_len)
#         for x in node_dict:
#             group_single_scores = pos_scorer_single.score_group(freq_table=node_dict[x]['single'])
#             group_pair_scores = pos_scorer_pair.score_group(freq_table=node_dict[x]['pair'])
#             if metric == 'identity':
#                 single_scores = group_identity_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
#             elif metric == 'plain_entropy':
#                 single_scores = group_plain_entropy_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
#             else:
#                 raise ValueError('Cannot test metric: {} in evaluate_score_group_ambiguous'.format(metric))
#             if metric == 'identity':
#                 pair_scores = group_identity_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
#             elif metric == 'plain_entropy':
#                 pair_scores = group_plain_entropy_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
#             else:
#                 raise ValueError('Cannot test metric: {} in evaluate_score_group_ambiguous'.format(metric))
#             for i in range(node_dict[x]['aln'].seq_length):
#                 self.assertEqual(group_single_scores[i], single_scores[i])
#                 for j in range(i, node_dict[x]['aln'].seq_length):
#                     self.assertEqual(group_pair_scores[i, j], pair_scores[i, j])
#
#     def test2a_score_group(self):
#         # Score group using identity metric terminals
#         self.evaluate_score_group_ambiguous(node_dict=self.terminals, metric='identity')
#
#     def test2b_score_group(self):
#         # Score group using identity metric parents
#         self.evaluate_score_group_ambiguous(node_dict=self.first_parents, metric='identity')
#
#     def test2c_score_group(self):
#         # Score group using plain entropy metric terminals
#         self.evaluate_score_group_ambiguous(node_dict=self.terminals, metric='plain_entropy')
#
#     def test2d_score_group(self):
#         # Score group using plain entropy metric parents
#         self.evaluate_score_group_ambiguous(node_dict=self.first_parents, metric='plain_entropy')
#
#     def evaluate_score_group_mi(self, node_dict, metric):
#         pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric=metric)
#         dim_pair = (self.seq_len, self.seq_len)
#         for x in node_dict:
#             group_pair_scores = pos_scorer_pair.score_group(freq_table=node_dict[x]['pair'])
#             if metric == 'mutual_information':
#                 pair_scores = group_mutual_information_score(freq_table=node_dict[x]['pair'],
#                                                              dimensions=dim_pair)
#             elif metric == 'normalized_mutual_information':
#                 pair_scores = group_normalized_mutual_information_score(freq_table=node_dict[x]['pair'],
#                                                                         dimensions=dim_pair)
#             else:
#                 raise ValueError('Cannot test metric: {} in evaluate_score_group_mi'.format(metric))
#             for i in range(node_dict[x]['aln'].seq_length):
#                 for j in range(i, node_dict[x]['aln'].seq_length):
#                     self.assertEqual(group_pair_scores[i, j], pair_scores[i, j])
#
#     def test2e_score_group(self):
#         # Score group using mutual information metric terminals
#         self.evaluate_score_group_mi(node_dict=self.terminals, metric='mutual_information')
#
#     def test2f_score_group(self):
#         # Score group using mutual information metric parents
#         self.evaluate_score_group_mi(node_dict=self.first_parents, metric='mutual_information')
#
#     def test2g_score_group(self):
#         # Score group using normalized mutual information metric terminals
#         self.evaluate_score_group_mi(node_dict=self.terminals, metric='normalized_mutual_information')
#
#     def test2h_score_group(self):
#         # Score group using normalized mutual information metric parents
#         self.evaluate_score_group_mi(node_dict=self.first_parents, metric='normalized_mutual_information')
#
#     def evaluate_score_group_average_product_corrected_mi(self, node_dict):
#         pos_scorer = PositionalScorer(seq_length=self.seq_len, pos_size=2,
#                                       metric='average_product_corrected_mutual_information')
#         for x in node_dict:
#             freq_table = node_dict[x]['pair']
#             pair_dims = (self.seq_len, self.seq_len)
#             mi_matrix = group_mutual_information_score(freq_table=freq_table, dimensions=pair_dims)
#             column_sums = {}
#             column_counts = {}
#             matrix_sums = 0.0
#             total_count = 0.0
#             for pos in freq_table.get_positions():
#                 if pos[0] == pos[1]:
#                     continue
#                 if pos[0] not in column_sums:
#                     column_sums[pos[0]] = 0.0
#                     column_counts[pos[0]] = 0.0
#                 if pos[1] not in column_sums:
#                     column_sums[pos[1]] = 0.0
#                     column_counts[pos[1]] = 0.0
#                 mi = mi_matrix[pos[0]][pos[1]]
#                 column_sums[pos[0]] += mi
#                 column_sums[pos[1]] += mi
#                 column_counts[pos[0]] += 1
#                 column_counts[pos[1]] += 1
#                 matrix_sums += mi
#                 total_count += 1
#             expected_apc = np.zeros((self.seq_len, self.seq_len))
#             if total_count == 0.0:
#                 matrix_average = 0.0
#             else:
#                 matrix_average = matrix_sums / total_count
#             if matrix_average != 0.0:
#                 column_averages = {}
#                 for key in column_sums:
#                     column_averages[key] = column_sums[key] / column_counts[key]
#                 for pos in freq_table.get_positions():
#                     if pos[0] == pos[1]:
#                         continue
#                     apc_numerator = column_averages[pos[0]] * column_averages[pos[1]]
#                     apc_correction = apc_numerator / matrix_average
#                     expected_apc[pos[0]][pos[1]] = mi_matrix[pos[0]][pos[1]] - apc_correction
#             apc = pos_scorer.score_group(freq_table=freq_table)
#             diff = apc - expected_apc
#             not_passing = diff > 1E-13
#             if not_passing.any():
#                 print(apc)
#                 print(expected_apc)
#                 print(diff)
#                 indices = np.nonzero(not_passing)
#                 print(apc[indices])
#                 print(expected_apc[indices])
#                 print(diff[indices])
#             self.assertTrue(not not_passing.any())
#
#     def test2i_score_group(self):
#         # Score group using average product correction mutual information metric terminals
#         self.evaluate_score_group_average_product_corrected_mi(node_dict=self.terminals)
#
#     def test2j_score_group(self):
#         # Score group using average product correction mutual information metric parents
#         self.evaluate_score_group_average_product_corrected_mi(node_dict=self.first_parents)
#
#     def evaluate_score_group_match_entropy_mismatch_entropy_ratio(self, node_dict):
#         pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric='match_mismatch_entropy_ratio')
#         dim_pair = (self.seq_len, self.seq_len)
#         for x in node_dict:
#             freq_table_dict = {'match': node_dict[x]['match'], 'mismatch': node_dict[x]['mismatch']}
#             group_pair_scores = pos_scorer_pair.score_group(freq_table=freq_table_dict)
#             pair_scores = group_match_mismatch_entropy_ratio(freq_tables=freq_table_dict, dimensions=dim_pair)
#             for i in range(node_dict[x]['aln'].seq_length):
#                 for j in range(i, node_dict[x]['aln'].seq_length):
#                     self.assertEqual(group_pair_scores[i, j], pair_scores[i, j])
#
#     def test2k_score_group_angle(self):
#         self.evaluate_score_group_match_entropy_mismatch_entropy_ratio(node_dict=self.terminals)
#
#     def test2l_score_group_angle(self):
#         self.evaluate_score_group_match_entropy_mismatch_entropy_ratio(node_dict=self.first_parents)
#
#     def evaluate_score_group_match_entropy_mismatch_entropy_angle(self, node_dict):
#         pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric='match_mismatch_entropy_angle')
#         dim_pair = (self.seq_len, self.seq_len)
#         for x in node_dict:
#             freq_table_dict = {'match': node_dict[x]['match'], 'mismatch': node_dict[x]['mismatch']}
#             group_pair_scores = pos_scorer_pair.score_group(freq_table=freq_table_dict)
#             pair_scores = group_match_mismatch_entropy_angle(freq_tables=freq_table_dict, dimensions=dim_pair)
#             for i in range(node_dict[x]['aln'].seq_length):
#                 for j in range(i, node_dict[x]['aln'].seq_length):
#                     self.assertEqual(group_pair_scores[i, j], pair_scores[i, j])
#
#     def test2m_score_group_angle(self):
#         self.evaluate_score_group_match_entropy_mismatch_entropy_angle(node_dict=self.terminals)
#
#     def test2n_score_group_angle(self):
#         self.evaluate_score_group_match_entropy_mismatch_entropy_angle(node_dict=self.first_parents)
#
#     def evaluate_score_group_match_diversity_mismatch_entropy_ratio(self, node_dict):
#         pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2,
#                                            metric='match_diversity_mismatch_entropy_ratio')
#         dim_pair = (self.seq_len, self.seq_len)
#         for x in node_dict:
#             freq_table_dict = {'match': node_dict[x]['match'], 'mismatch': node_dict[x]['mismatch']}
#             group_pair_scores = pos_scorer_pair.score_group(freq_table=freq_table_dict)
#             pair_scores = group_match_diversity_mismatch_entropy_ratio(freq_tables=freq_table_dict, dimensions=dim_pair)
#             for i in range(node_dict[x]['aln'].seq_length):
#                 for j in range(i, node_dict[x]['aln'].seq_length):
#                     self.assertEqual(group_pair_scores[i, j], pair_scores[i, j])
#
#     def test2o_score_group_angle(self):
#         self.evaluate_score_group_match_entropy_mismatch_entropy_ratio(node_dict=self.terminals)
#
#     def test2p_score_group_angle(self):
#         self.evaluate_score_group_match_entropy_mismatch_entropy_ratio(node_dict=self.first_parents)
#
#     def evaluate_score_group_match_diversity_mismatch_entropy_angle(self, node_dict):
#         pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2,
#                                            metric='match_diversity_mismatch_entropy_angle')
#         dim_pair = (self.seq_len, self.seq_len)
#         for x in node_dict:
#             freq_table_dict = {'match': node_dict[x]['match'], 'mismatch': node_dict[x]['mismatch']}
#             group_pair_scores = pos_scorer_pair.score_group(freq_table=freq_table_dict)
#             pair_scores = group_match_diversity_mismatch_entropy_angle(freq_tables=freq_table_dict, dimensions=dim_pair)
#             for i in range(node_dict[x]['aln'].seq_length):
#                 for j in range(i, node_dict[x]['aln'].seq_length):
#                     self.assertEqual(group_pair_scores[i, j], pair_scores[i, j])
#
#     def test2q_score_group_angle(self):
#         self.evaluate_score_group_match_diversity_mismatch_entropy_angle(node_dict=self.terminals)
#
#     def test2r_score_group_angle(self):
#         self.evaluate_score_group_match_diversity_mismatch_entropy_angle(node_dict=self.first_parents)
#
#     def evaluate_score_rank_ambiguous(self, node_dict, metric, single=True, pair=True):
#         group_scores_single = []
#         group_scores_pair = []
#         if single:
#             pos_scorer_single = PositionalScorer(seq_length=self.seq_len, pos_size=1, metric=metric)
#         if pair:
#             pos_scorer_pair = PositionalScorer(seq_length=self.seq_len, pos_size=2, metric=metric)
#         for x in node_dict:
#             if ('ratio' in metric) or ('angle' in metric):
#                 if single:
#                     raise ValueError(f'{metric} not intended for single position measurement.')
#                 if pair:
#                     group_score_pair = pos_scorer_pair.score_group(freq_table=node_dict[x])
#                     group_scores_pair.append(group_score_pair)
#             else:
#                 if single:
#                     group_score_single = pos_scorer_single.score_group(freq_table=node_dict[x]['single'])
#                     group_scores_single.append(group_score_single)
#                 if pair:
#                     group_score_pair = pos_scorer_pair.score_group(freq_table=node_dict[x]['pair'])
#                     group_scores_pair.append(group_score_pair)
#         rank = max(len(group_scores_single), len(group_scores_pair))
#         if single:
#             group_scores_single = np.sum(np.stack(group_scores_single, axis=0), axis=0)
#             if metric == 'identity':
#                 expected_rank_score_single = rank_integer_value_score(score_matrix=group_scores_single, rank=rank)
#             else:
#                 expected_rank_score_single = rank_real_value_score(score_matrix=group_scores_single, rank=rank)
#             rank_score_single = pos_scorer_single.score_rank(score_tensor=group_scores_single, rank=rank)
#             diff_single = rank_score_single - expected_rank_score_single
#             self.assertTrue(not diff_single.any())
#         if pair:
#             group_scores_pair = np.sum(np.stack(group_scores_pair, axis=0), axis=0)
#             if metric == 'identity':
#                 expected_rank_score_pair = rank_integer_value_score(score_matrix=group_scores_pair, rank=rank)
#             else:
#                 expected_rank_score_pair = rank_real_value_score(score_matrix=group_scores_pair, rank=rank)
#             expected_rank_score_pair = np.triu(expected_rank_score_pair, k=1)
#             rank_score_pair = pos_scorer_pair.score_rank(score_tensor=group_scores_pair, rank=rank)
#             diff_pair = rank_score_pair - expected_rank_score_pair
#             self.assertTrue(not diff_pair.any())
#
#     def test3a_score_rank(self):
#         # Score rank using identity metric terminals
#         self.evaluate_score_rank_ambiguous(node_dict=self.terminals, metric='identity')
#
#     def test3b_score_rank(self):
#         # Score rank using identity metric parents
#         self.evaluate_score_rank_ambiguous(node_dict=self.first_parents, metric='identity')
#
#     def test3c_score_rank(self):
#         # Score rank using plain entropy metric terminals
#         self.evaluate_score_rank_ambiguous(node_dict=self.terminals, metric='plain_entropy')
#
#     def test3d_score_rank(self):
#         # Score rank using plain entropy metric parents
#         self.evaluate_score_rank_ambiguous(node_dict=self.first_parents, metric='plain_entropy')
#
#     def test3e_score_rank(self):
#         # Score rank using mutual information metric terminals
#         self.evaluate_score_rank_ambiguous(node_dict=self.terminals, metric='mutual_information', single=False)
#
#     def test3f_score_rank(self):
#         # Score rank using mutual information metric parents
#         self.evaluate_score_rank_ambiguous(node_dict=self.first_parents, metric='mutual_information', single=False)
#
#     def test3g_score_rank(self):
#         # Score rank using normalized mutual information metric terminal
#         self.evaluate_score_rank_ambiguous(node_dict=self.terminals, metric='normalized_mutual_information',
#                                            single=False)
#
#     def test3h_score_rank(self):
#         # Score rank using normalized mutual information metric parents
#         self.evaluate_score_rank_ambiguous(node_dict=self.first_parents, metric='normalized_mutual_information',
#                                            single=False)
#
#     def test3i_score_rank(self):
#         # Score rank using average product corrected mutual information metric terminals
#         self.evaluate_score_rank_ambiguous(node_dict=self.terminals,
#                                            metric='average_product_corrected_mutual_information',
#                                            single=False)
#
#     def test3j_score_rank(self):
#         # Score rank using average product corrected mutual information metric parents
#         self.evaluate_score_rank_ambiguous(node_dict=self.first_parents,
#                                            metric='average_product_corrected_mutual_information',
#                                            single=False)
#
#     def test3k_score_rank(self):
#         # Score rank using average product corrected mutual information metric terminals
#         self.evaluate_score_rank_ambiguous(node_dict=self.terminals,
#                                            metric='filtered_average_product_corrected_mutual_information',
#                                            single=False)
#
#     def test3l_score_rank(self):
#         # Score rank using average product corrected mutual information metric parents
#         self.evaluate_score_rank_ambiguous(node_dict=self.first_parents,
#                                            metric='filtered_average_product_corrected_mutual_information',
#                                            single=False)
#
#     def test3m_score_rank(self):
#         # Score rank using average product corrected mutual information metric terminals
#         self.evaluate_score_rank_ambiguous(node_dict=self.terminals,
#                                            metric='match_mismatch_entropy_ratio',
#                                            single=False)
#
#     def test3n_score_rank(self):
#         # Score rank using average product corrected mutual information metric parents
#         self.evaluate_score_rank_ambiguous(node_dict=self.first_parents,
#                                            metric='match_mismatch_entropy_ratio',
#                                            single=False)
#
#     def test3o_score_rank(self):
#         # Score rank using average product corrected mutual information metric terminals
#         self.evaluate_score_rank_ambiguous(node_dict=self.terminals,
#                                            metric='match_mismatch_entropy_angle',
#                                            single=False)
#
#     def test3p_score_rank(self):
#         # Score rank using average product corrected mutual information metric parents
#         self.evaluate_score_rank_ambiguous(node_dict=self.first_parents,
#                                            metric='match_mismatch_entropy_angle',
#                                            single=False)
#
#     def test3q_score_rank(self):
#         # Score rank using average product corrected mutual information metric terminals
#         self.evaluate_score_rank_ambiguous(node_dict=self.terminals,
#                                            metric='match_diversity_mismatch_entropy_ratio',
#                                            single=False)
#
#     def test3r_score_rank(self):
#         # Score rank using average product corrected mutual information metric parents
#         self.evaluate_score_rank_ambiguous(node_dict=self.first_parents,
#                                            metric='match_diversity_mismatch_entropy_ratio',
#                                            single=False)
#
#     def test3s_score_rank(self):
#         # Score rank using average product corrected mutual information metric terminals
#         self.evaluate_score_rank_ambiguous(node_dict=self.terminals,
#                                            metric='match_diversity_mismatch_entropy_angle',
#                                            single=False)
#
#     def test3t_score_rank(self):
#         # Score rank using average product corrected mutual information metric parents
#         self.evaluate_score_rank_ambiguous(node_dict=self.first_parents,
#                                            metric='match_diversity_mismatch_entropy_angle',
#                                            single=False)
#
#     def evaluate_rank_integer_value_score(self, node_dict):
#         group_single_scores = []
#         dim_single = (self.seq_len,)
#         group_pair_scores = []
#         dim_pair = (self.seq_len, self.seq_len)
#         for x in node_dict:
#             group_single_score = group_identity_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
#             group_pair_score = group_identity_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
#             group_single_scores.append(group_single_score)
#             group_pair_scores.append(group_pair_score)
#         rank = max(len(group_single_scores), len(group_pair_scores))
#         group_single_scores = np.sum(np.stack(group_single_scores, axis=0), axis=0)
#         rank_single_scores = rank_integer_value_score(score_matrix=group_single_scores, rank=rank)
#         expected_rank_single_scores = np.zeros(dim_single)
#         for i in range(node_dict[x]['aln'].seq_length):
#             if group_single_scores[i] != 0:
#                 expected_rank_single_scores[i] = 1
#         diff_single = rank_single_scores - expected_rank_single_scores
#         self.assertTrue(not diff_single.any())
#         group_pair_scores = np.sum(np.stack(group_pair_scores, axis=0), axis=0)
#         rank_pair_scores = rank_integer_value_score(score_matrix=group_pair_scores, rank=rank)
#         expected_rank_pair_scores = np.zeros((self.seq_len, self.seq_len))
#         for i in range(node_dict[x]['aln'].seq_length):
#             for j in range(i, node_dict[x]['aln'].seq_length):
#                 if group_pair_scores[i, j] != 0:
#                     expected_rank_pair_scores[i, j] = 1
#         diff_pair = rank_pair_scores - expected_rank_pair_scores
#         self.assertTrue(not diff_pair.any())
#
#     def test4a_rank_integer_value_score(self):
#         # Metric=Identity, Alignment Size=1
#         self.evaluate_rank_integer_value_score(node_dict=self.terminals)
#
#     def test4b_rank_integer_value_score(self):
#         # Metric=Identity, Alignment Size=2
#         self.evaluate_rank_integer_value_score(node_dict=self.first_parents)
#
#     def evaluate_rank_real_value_score(self, node_dict):
#         group_single_scores = []
#         dim_single = (self.seq_len,)
#         group_pair_scores = []
#         dim_pair = (self.seq_len, self.seq_len)
#         for x in node_dict:
#             group_single_score = group_identity_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
#             group_pair_score = group_identity_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
#             group_single_scores.append(group_single_score)
#             group_pair_scores.append(group_pair_score)
#         rank = max(len(group_single_scores), len(group_pair_scores))
#         group_single_scores = np.sum(np.stack(group_single_scores, axis=0), axis=0)
#         rank_single_scores = rank_real_value_score(score_matrix=group_single_scores, rank=rank)
#         expected_rank_single_scores = np.zeros(self.seq_len)
#         for i in range(node_dict[x]['aln'].seq_length):
#             expected_rank_single_scores[i] += (1.0 / rank) * group_single_scores[i]
#         diff_single = rank_single_scores - expected_rank_single_scores
#         not_passing_single = diff_single > 1E-16
#         self.assertTrue(not not_passing_single.any())
#         group_pair_scores = np.sum(np.stack(group_pair_scores, axis=0), axis=0)
#         rank_pair_scores = rank_real_value_score(score_matrix=group_pair_scores, rank=rank)
#         expected_rank_pair_scores = np.zeros((self.seq_len, self.seq_len))
#         for i in range(node_dict[x]['aln'].seq_length):
#             for j in range(i, node_dict[x]['aln'].seq_length):
#                 expected_rank_pair_scores[i, j] += (1.0 / rank) * group_pair_scores[i, j]
#         diff_pair = rank_pair_scores - expected_rank_pair_scores
#         not_passing_pair = diff_pair > 1E-16
#         self.assertTrue(not not_passing_pair.any())
#
#     def test5a_rank_real_value_score(self):
#         self.evaluate_rank_real_value_score(node_dict=self.terminals)
#
#     def test5b_rank_real_value_score(self):
#         self.evaluate_rank_real_value_score(node_dict=self.first_parents)
#
#     def evaluate_group_identity_score(self, node_dict):
#         for x in node_dict:
#             dim_single = (self.seq_len,)
#             single_scores = group_identity_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
#             dim_pair = (self.seq_len, self.seq_len)
#             pair_scores = group_identity_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
#             for i in range(node_dict[x]['aln'].seq_length):
#                 expected_single_score = 0
#                 single_char = None
#                 for k in range(node_dict[x]['aln'].size):
#                     curr_single_char = node_dict[x]['aln'].alignment[k, i]
#                     if single_char is None:
#                         single_char = curr_single_char
#                     else:
#                         if single_char != curr_single_char:
#                             expected_single_score = 1
#                 self.assertEqual(single_scores[i], expected_single_score)
#                 for j in range(i, node_dict[x]['aln'].seq_length):
#                     if i == j:
#                         self.assertEqual(single_scores[i], pair_scores[i, j])
#                         continue
#                     expected_pair_score = 0
#                     pair_char = None
#                     for k in range(node_dict[x]['aln'].size):
#                         curr_pair_char = node_dict[x]['aln'].alignment[k, i] + node_dict[x]['aln'].alignment[k, j]
#                         if pair_char is None:
#                             pair_char = curr_pair_char
#                         else:
#                             if pair_char != curr_pair_char:
#                                 expected_pair_score = 1
#                     self.assertEqual(pair_scores[i, j], expected_pair_score)
#
#     def test6a_group_identity_score(self):
#         self.evaluate_group_identity_score(node_dict=self.terminals)
#
#     def test6b_group_identity_score(self):
#         self.evaluate_group_identity_score(node_dict=self.first_parents)
#
#     def evaluate_group_plain_entropy_score(self, node_dict):
#         for x in node_dict:
#             dim_single = (self.seq_len,)
#             single_scores = group_plain_entropy_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
#             dim_pair = (self.seq_len, self.seq_len)
#             pair_scores = group_plain_entropy_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
#             for i in range(node_dict[x]['aln'].seq_length):
#                 single_chars = {}
#                 for k in range(node_dict[x]['aln'].size):
#                     curr_single_char = node_dict[x]['aln'].alignment[k, i]
#                     if curr_single_char not in single_chars:
#                         single_chars[curr_single_char] = 0.0
#                     single_chars[curr_single_char] += 1.0
#                 expected_single_score = 0.0
#                 for c in single_chars:
#                     frequency = single_chars[c] / node_dict[x]['aln'].size
#                     expected_single_score -= frequency * np.log(frequency)
#                 self.assertEqual(single_scores[i], expected_single_score)
#                 for j in range(i, node_dict[x]['aln'].seq_length):
#                     if i == j:
#                         self.assertEqual(single_scores[i], pair_scores[i, j])
#                         continue
#                     pair_chars = {}
#                     for k in range(node_dict[x]['aln'].size):
#                         curr_pair_char = node_dict[x]['aln'].alignment[k, i] + node_dict[x]['aln'].alignment[k, j]
#                         if curr_pair_char not in pair_chars:
#                             pair_chars[curr_pair_char] = 0.0
#                         pair_chars[curr_pair_char] += 1.0
#                     expected_pair_score = 0.0
#                     for c in pair_chars:
#                         frequency = pair_chars[c] / node_dict[x]['aln'].size
#                         expected_pair_score -= frequency * np.log(frequency)
#                     self.assertEqual(pair_scores[i, j], expected_pair_score)
#
#     def test7a_group_plain_entropy_score(self):
#         self.evaluate_group_plain_entropy_score(node_dict=self.terminals)
#
#     def test7b_group_plain_entropy_score(self):
#         self.evaluate_group_plain_entropy_score(node_dict=self.first_parents)
#
#     def evaluate_mutual_information_computation(self, node_dict):
#         for x in node_dict:
#             dim_single = (self.seq_len,)
#             dim_pair = (self.seq_len, self.seq_len)
#             mi_values = mutual_information_computation(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
#             single_entropies = group_plain_entropy_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
#             pair_joint_entropies = group_plain_entropy_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
#             for i in range(node_dict[x]['aln'].seq_length):
#                 for j in range(i + 1, node_dict[x]['aln'].seq_length):
#                     hi = single_entropies[i]
#                     hj = single_entropies[j]
#                     hij = pair_joint_entropies[i, j]
#                     mi = (hi + hj) - hij
#                     self.assertEqual(mi_values[0][i, j], hi)
#                     self.assertEqual(mi_values[1][i, j], hj)
#                     self.assertEqual(mi_values[2][i, j], hij)
#                     self.assertEqual(mi_values[3][i, j], mi)
#
#     def test8a_mutual_information_computation(self):
#         self.evaluate_mutual_information_computation(node_dict=self.terminals)
#
#     def test8b_mutual_information_computation(self):
#         self.evaluate_mutual_information_computation(node_dict=self.first_parents)
#
#     def evaluate_group_mutual_information_score(self, node_dict):
#         for x in node_dict:
#             dim_single = (self.seq_len,)
#             dim_pair = (self.seq_len, self.seq_len)
#             mi_values = group_mutual_information_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
#             single_entropies = group_plain_entropy_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
#             pair_joint_entropies = group_plain_entropy_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
#             for i in range(node_dict[x]['aln'].seq_length):
#                 for j in range(i + 1, node_dict[x]['aln'].seq_length):
#                     hi = single_entropies[i]
#                     hj = single_entropies[j]
#                     hij = pair_joint_entropies[i, j]
#                     expected_mi = (hi + hj) - hij
#                     self.assertEqual(mi_values[i, j], expected_mi)
#
#     def test9a_group_mutual_information_score(self):
#         self.evaluate_group_mutual_information_score(node_dict=self.terminals)
#
#     def test9b_group_mutual_information_score(self):
#         self.evaluate_group_mutual_information_score(node_dict=self.first_parents)
#
#     def evaluate_group_normalized_mutual_information_score(self, node_dict):
#         for x in node_dict:
#             dim_single = (self.seq_len,)
#             dim_pair = (self.seq_len, self.seq_len)
#             nmi_values = group_normalized_mutual_information_score(freq_table=node_dict[x]['pair'],
#                                                                    dimensions=dim_pair)
#             single_entropies = group_plain_entropy_score(freq_table=node_dict[x]['single'], dimensions=dim_single)
#             pair_joint_entropies = group_plain_entropy_score(freq_table=node_dict[x]['pair'], dimensions=dim_pair)
#             for i in range(node_dict[x]['aln'].seq_length):
#                 for j in range(i + 1, node_dict[x]['aln'].seq_length):
#                     hi = single_entropies[i]
#                     hj = single_entropies[j]
#                     hij = pair_joint_entropies[i, j]
#                     mi = (hi + hj) - hij
#                     normalization = np.mean([hi, hj])
#                     if hi == hj == 0.0:
#                         expected_nmi = 1.0
#                     elif normalization == 0.0:
#                         expected_nmi = 0.0
#                     else:
#                         expected_nmi = mi / normalization
#                     self.assertEqual(nmi_values[i, j], expected_nmi)
#
#     def test10a_group_normalized_mutual_information_score(self):
#         self.evaluate_group_normalized_mutual_information_score(node_dict=self.terminals)
#
#     def test10b_group_normalized_mutual_information_score(self):
#         self.evaluate_group_normalized_mutual_information_score(node_dict=self.first_parents)
#
#     def evaluate_average_product_correction(self, node_dict):
#         for x in node_dict:
#             pair_dim = (self.seq_len, self.seq_len)
#             mi_matrix = group_mutual_information_score(freq_table=node_dict[x]['pair'], dimensions=pair_dim)
#             total_sum = 0.0
#             total_count = 0.0
#             column_sums = np.zeros(node_dict[x]['aln'].seq_length)
#             column_counts = np.zeros(node_dict[x]['aln'].seq_length)
#             for i in range(node_dict[x]['aln'].seq_length):
#                 for j in range(i + 1, node_dict[x]['aln'].seq_length):
#                     mi = mi_matrix[i, j]
#                     total_sum += mi
#                     total_count += 1
#                     column_sums[i] += mi
#                     column_counts[i] += 1
#                     column_sums[j] += mi
#                     column_counts[j] += 1
#             expected_apc_matrix = np.zeros((node_dict[x]['aln'].seq_length, node_dict[x]['aln'].seq_length))
#             if total_count > 0:
#                 total_average = total_sum / total_count
#                 column_average = column_sums / column_counts
#                 if total_average > 0:
#                     for i in range(node_dict[x]['aln'].seq_length):
#                         for j in range(i + 1, node_dict[x]['aln'].seq_length):
#                             numerator = column_average[i] * column_average[j]
#                             correction_factor = numerator / total_average
#                             expected_apc_matrix[i, j] = mi_matrix[i, j] - correction_factor
#             apc_matrix = average_product_correction(mutual_information_matrix=mi_matrix)
#             diff = apc_matrix - expected_apc_matrix
#             not_passing = diff > 1E-13
#             self.assertTrue(not not_passing.any())
#
#     def test11a_average_product_correction(self):
#         self.evaluate_average_product_correction(node_dict=self.terminals)
#
#     def test11b_average_product_correction(self):
#         self.evaluate_average_product_correction(node_dict=self.first_parents)
#
#     def evaluate_filtered_average_product_correction(self, node_dict):
#         for x in node_dict:
#             pair_dim = (self.seq_len, self.seq_len)
#             mi_matrix = group_mutual_information_score(freq_table=node_dict[x]['pair'], dimensions=pair_dim)
#             total_sum = 0.0
#             total_count = 0.0
#             column_sums = np.zeros(node_dict[x]['aln'].seq_length)
#             column_counts = np.zeros(node_dict[x]['aln'].seq_length)
#             for i in range(node_dict[x]['aln'].seq_length):
#                 for j in range(i + 1, node_dict[x]['aln'].seq_length):
#                     mi = mi_matrix[i, j]
#                     total_sum += mi
#                     total_count += 1
#                     column_sums[i] += mi
#                     column_counts[i] += 1
#                     column_sums[j] += mi
#                     column_counts[j] += 1
#             expected_apc_matrix = np.zeros((node_dict[x]['aln'].seq_length, node_dict[x]['aln'].seq_length))
#             if total_count > 0:
#                 total_average = total_sum / total_count
#                 column_average = column_sums / column_counts
#                 if total_average > 0:
#                     for i in range(node_dict[x]['aln'].seq_length):
#                         for j in range(i + 1, node_dict[x]['aln'].seq_length):
#                             numerator = column_average[i] * column_average[j]
#                             correction_factor = numerator / total_average
#                             if mi_matrix[i, j] > 0.0001:
#                                 expected_apc_matrix[i, j] = mi_matrix[i, j] - correction_factor
#             apc_matrix = filtered_average_product_correction(mutual_information_matrix=mi_matrix)
#             diff = apc_matrix - expected_apc_matrix
#             not_passing = diff > 1E-13
#             self.assertTrue(not not_passing.any())
#
#     def test11c_heuristic_average_product_correction(self):
#         self.evaluate_filtered_average_product_correction(node_dict=self.terminals)
#
#     def test11d_heuristic_average_product_correction(self):
#         self.evaluate_filtered_average_product_correction(node_dict=self.first_parents)
#
#     def evaluate_ratio_computation(self, node_dict):
#         for x in node_dict:
#             dim_pair = (self.seq_len, self.seq_len)
#             expected_match_entropy = group_plain_entropy_score(freq_table=node_dict[x]['match'], dimensions=dim_pair)
#             expected_mismatch_entropy = group_plain_entropy_score(freq_table=node_dict[x]['mismatch'],
#                                                                   dimensions=dim_pair)
#             expected_ratio = expected_mismatch_entropy / expected_match_entropy
#             div_by_0_indices = np.isnan(expected_ratio)
#             comparable_indices = ~div_by_0_indices
#             observed_ratios = ratio_computation(match_table=expected_match_entropy,
#                                                 mismatch_table=expected_mismatch_entropy)
#             comparable_diff = observed_ratios[comparable_indices] - expected_ratio[comparable_indices]
#             self.assertFalse(comparable_diff.any())
#             mismatch_indices = div_by_0_indices & (expected_mismatch_entropy == 0.0)
#             min_check = observed_ratios[mismatch_indices] == 0.0
#             self.assertTrue(min_check.all())
#             match_indices = div_by_0_indices & (expected_match_entropy == 0.0)
#             match_indices = ((1 * match_indices) - (1 * mismatch_indices)).astype(bool)
#             max_check = observed_ratios[match_indices] == np.tan(np.pi / 2.0)
#             self.assertTrue(max_check.all())
#
#     def test12a_ratio_computation(self):
#         self.evaluate_ratio_computation(node_dict=self.terminals)
#
#     def test12b_ratio_computation(self):
#         self.evaluate_ratio_computation(node_dict=self.first_parents)
#
#     def evaluate_group_match_mismatch_entropy_ratio(self, node_dict):
#         for x in node_dict:
#             dim_pair = (self.seq_len, self.seq_len)
#             observed_ratios = group_match_mismatch_entropy_ratio(freq_tables={'match': node_dict[x]['match'],
#                                                                               'mismatch': node_dict[x]['mismatch']},
#                                                                  dimensions=dim_pair)
#             expected_match_entropy = group_plain_entropy_score(freq_table=node_dict[x]['match'], dimensions=dim_pair)
#             expected_mismatch_entropy = group_plain_entropy_score(freq_table=node_dict[x]['mismatch'],
#                                                                   dimensions=dim_pair)
#             expected_ratios = ratio_computation(match_table=expected_match_entropy,
#                                                 mismatch_table=expected_mismatch_entropy)
#             diff = observed_ratios - expected_ratios
#             self.assertFalse(diff.any())
#
#     def test13a_group_match_mismatch_entropy_ratio(self):
#         self.evaluate_group_match_mismatch_entropy_ratio(node_dict=self.terminals)
#
#     def test13b_group_match_mismatch_entropy_ratio(self):
#         self.evaluate_group_match_mismatch_entropy_ratio(node_dict=self.first_parents)
#
#     def evaluate_angle_computation(self, node_dict):
#         for x in node_dict:
#             dim_pair = (self.seq_len, self.seq_len)
#             expected_match_entropy = group_plain_entropy_score(freq_table=node_dict[x]['match'], dimensions=dim_pair)
#             expected_mismatch_entropy = group_plain_entropy_score(freq_table=node_dict[x]['mismatch'],
#                                                                   dimensions=dim_pair)
#             expected_ratios = ratio_computation(match_table=expected_match_entropy,
#                                                 mismatch_table=expected_mismatch_entropy)
#             observed_angles = angle_computation(ratios=expected_ratios)
#             expected_hypotenuse = np.linalg.norm(np.stack([expected_match_entropy, expected_mismatch_entropy], axis=0),
#                                                  axis=0)
#             expected_sin_ratio = np.zeros(expected_match_entropy.shape)
#             sin_indices = (expected_match_entropy != 0.0) | (expected_mismatch_entropy != 0)
#             expected_sin_ratio[sin_indices] = expected_mismatch_entropy[sin_indices] / expected_hypotenuse[sin_indices]
#             expected_sin_ratio[expected_match_entropy == 0] = np.sin(90.0)
#             expected_sin_ratio[expected_mismatch_entropy == 0] = np.sin(0.0)
#             expected_sin_angle = np.arcsin(expected_sin_ratio)
#             sin_diff = observed_angles - expected_sin_angle
#             self.assertFalse(sin_diff.any())
#             cos_indices = (expected_match_entropy != 0.0) | (expected_mismatch_entropy != 0)
#             expected_cos_ratio = np.zeros(expected_match_entropy.shape)
#             expected_cos_ratio[cos_indices] = expected_match_entropy[cos_indices] / expected_hypotenuse[cos_indices]
#             expected_cos_ratio[expected_match_entropy == 0.0] = np.cos(90.0)
#             expected_cos_ratio[expected_mismatch_entropy == 0] = np.cos(0.0)
#             expected_cos_angle = np.arccos(expected_cos_ratio)
#             cos_diff = observed_angles - expected_cos_angle
#             self.assertFalse(cos_diff.any())
#
#     def test14a_angle_computation(self):
#         self.evaluate_angle_computation(node_dict=self.terminals)
#
#     def test14b_angle_computation(self):
#         self.evaluate_angle_computation(node_dict=self.first_parents)
#
#     def evaluate_group_match_mismatch_entropy_angle(self, node_dict):
#         for x in node_dict:
#             dim_pair = (self.seq_len, self.seq_len)
#             observed_angles = group_match_mismatch_entropy_angle(freq_tables={'match': node_dict[x]['match'],
#                                                                               'mismatch': node_dict[x]['mismatch']},
#                                                                  dimensions=dim_pair)
#             expected_ratios = group_match_mismatch_entropy_ratio(freq_tables=node_dict[x], dimensions=dim_pair)
#             expected_angles = angle_computation(ratios=expected_ratios)
#             diff = observed_angles - expected_angles
#             self.assertFalse(diff.any())
#
#     def test15a_group_match_mismatch_entropy_angle(self):
#         self.evaluate_group_match_mismatch_entropy_angle(node_dict=self.terminals)
#
#     def test15b_group_match_mismatch_entropy_angle(self):
#         self.evaluate_group_match_mismatch_entropy_angle(node_dict=self.first_parents)
#
#     def evaluate_diversity_computation(self, node_dict):
#         for x in node_dict:
#             dim_pair = (self.seq_len, self.seq_len)
#             expected_match_entropy = group_plain_entropy_score(freq_table=node_dict[x]['match'], dimensions=dim_pair)
#             expected_match_diversity = np.exp(expected_match_entropy)
#             observed_match_diversity = diversity_computation(freq_table=node_dict[x]['match'], dimensions=dim_pair)
#             match_diff = observed_match_diversity - expected_match_diversity
#             self.assertFalse(match_diff.any())
#             expected_mismatch_entropy = group_plain_entropy_score(freq_table=node_dict[x]['mismatch'],
#                                                                   dimensions=dim_pair)
#             expected_mismatch_diversity = np.exp(expected_mismatch_entropy)
#             observed_mismatch_diversity = diversity_computation(freq_table=node_dict[x]['mismatch'],
#                                                                 dimensions=dim_pair)
#             mismatch_diff = observed_mismatch_diversity - expected_mismatch_diversity
#             self.assertFalse(mismatch_diff.any())
#
#     def test16a_diversity_computation(self):
#         self.evaluate_diversity_computation(node_dict=self.terminals)
#
#     def test16b_diversity_computation(self):
#         self.evaluate_diversity_computation(node_dict=self.first_parents)
#
#     def evaluate_group_match_diversity_mismatch_entropy_ratio(self, node_dict):
#         for x in node_dict:
#             dim_pair = (self.seq_len, self.seq_len)
#             observed_ratios = group_match_diversity_mismatch_entropy_ratio(
#                 freq_tables={'match': node_dict[x]['match'], 'mismatch': node_dict[x]['mismatch']}, dimensions=dim_pair)
#             expected_match_diversity = diversity_computation(freq_table=node_dict[x]['match'], dimensions=dim_pair)
#             expected_mismatch_entropy = group_plain_entropy_score(freq_table=node_dict[x]['mismatch'],
#                                                                   dimensions=dim_pair)
#             expected_ratios = ratio_computation(match_table=expected_match_diversity,
#                                                 mismatch_table=expected_mismatch_entropy)
#             diff = observed_ratios - expected_ratios
#             self.assertFalse(diff.any())
#
#     def test17a_group_match_mismatch_entropy_ratio(self):
#         self.evaluate_group_match_diversity_mismatch_entropy_ratio(node_dict=self.terminals)
#
#     def test17b_group_match_mismatch_entropy_ratio(self):
#         self.evaluate_group_match_diversity_mismatch_entropy_ratio(node_dict=self.first_parents)
#
#     def evaluate_group_match_diversity_mismatch_entropy_angle(self, node_dict):
#         for x in node_dict:
#             dim_pair = (self.seq_len, self.seq_len)
#             observed_angles = group_match_diversity_mismatch_entropy_angle(
#                 freq_tables={'match': node_dict[x]['match'], 'mismatch': node_dict[x]['mismatch']}, dimensions=dim_pair)
#             expected_ratios = group_match_diversity_mismatch_entropy_ratio(freq_tables=node_dict[x],
#                                                                            dimensions=dim_pair)
#             expected_angles = angle_computation(ratios=expected_ratios)
#             diff = observed_angles - expected_angles
#             self.assertFalse(diff.any())
#
#     def test18a_group_match_mismatch_entropy_angle(self):
#         self.evaluate_group_match_diversity_mismatch_entropy_angle(node_dict=self.terminals)
#
#     def test18b_group_match_mismatch_entropy_angle(self):
#         self.evaluate_group_match_diversity_mismatch_entropy_angle(node_dict=self.first_parents)


if __name__ == '__main__':
    unittest.main()
