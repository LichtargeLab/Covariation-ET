"""
Created on July 12, 2019

@author: Daniel Konecki
"""
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from test_Base import TestBase
from SeqAlignment import SeqAlignment
from PhylogeneticTree import PhylogeneticTree
from EvolutionaryTraceAlphabet import FullIUPACProtein
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from PositionalScorer import PositionalScorer, group_identity_score, rank_identity_score


class TestTrace(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestTrace, cls).setUpClass()
        # cls.query_aln_fa_small = SeqAlignment(
        #     file_name=cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln'],
        #     query_id=cls.small_structure_id)
        # cls.query_aln_fa_small.import_alignment()
        # cls.phylo_tree_small = PhylogeneticTree()
        # calc = AlignmentDistanceCalculator()
        # cls.phylo_tree_small.construct_tree(dm=calc.get_distance(cls.query_aln_fa_small.alignment))
        # cls.terminals_small = {x.name: {'aln': cls.query_aln_fa_small.generate_sub_alignment(sequence_ids=[x.name]),
        #                                 'node': x}
        #                        for x in cls.phylo_tree_small.tree.get_terminals()}
        # for x in cls.terminals_small:
        #     single, pair = cls.terminals_small[x]['aln'].characterize_positions()
        #     cls.terminals_small[x]['single'] = single
        #     cls.terminals_small[x]['pair'] = pair
        # cls.first_parents_small = {x.name: {'aln': cls.query_aln_fa_small.generate_sub_alignment(sequence_ids=[y.name for y in x.clades]),
        #                                     'node': x} for x in
        #                            set([cls.phylo_tree_small.tree.get_path(x)[-2] for x in cls.terminals_small])}
        # for x in cls.first_parents_small:
        #     cls.first_parents_small[x]['single'] = (cls.terminals_small[cls.first_parents_small[x]['node'].clades[0]]['single'] +
        #                                             cls.terminals_small[cls.first_parents_small[x]['node'].clades[1]]['single'])
        #     cls.first_parents_small[x]['pair'] = (cls.terminals_small[cls.first_parents_small[x]['node'].clades[0]]['pair'] +
        #                                           cls.terminals_small[cls.first_parents_small[x]['node'].clades[1]]['pair'])
        #
        alpha = FullIUPACProtein()
        seq1 = SeqRecord(Seq('MV', alphabet=alpha), id='Seq1')
        aln1 = MultipleSeqAlignment([seq1], alphabet=alpha)
        seq_aln1 = SeqAlignment(file_name='test', query_id='Seq1')
        seq_aln1.seq_length = 2
        seq_aln1.size = 1
        seq_aln1.seq_order = ['Seq1']
        seq_aln1.alignment = aln1
        seq_aln1.alphabet = alpha
        seq_aln1.query_sequence = seq1.seq
        seq_aln1.marked = [False]
        single1, pair1 = seq_aln1.characterize_positions()
        seq2 = SeqRecord(Seq('MG', alphabet=alpha), id='Seq2')
        aln2 = MultipleSeqAlignment([seq2], alphabet=alpha)
        seq_aln2 = SeqAlignment(file_name='test', query_id='Seq1')
        seq_aln2.seq_length = 2
        seq_aln2.size = 1
        seq_aln2.seq_order = ['Seq2']
        seq_aln2.alignment = aln2
        seq_aln2.alphabet = alpha
        seq_aln2.query_sequence = seq1.seq
        seq_aln2.marked = [False]
        single2, pair2 = seq_aln2.characterize_positions()
        seq3 = SeqRecord(Seq('MY', alphabet=alpha), id='Seq3')
        aln3 = MultipleSeqAlignment([seq3], alphabet=alpha)
        seq_aln3 = SeqAlignment(file_name='test', query_id='Seq1')
        seq_aln3.seq_length = 2
        seq_aln3.size = 1
        seq_aln3.seq_order = ['Seq3']
        seq_aln3.alignment = aln3
        seq_aln3.alphabet = alpha
        seq_aln3.query_sequence = seq1.seq
        seq_aln3.marked = [False]
        single3, pair3 = seq_aln3.characterize_positions()
        seq4 = SeqRecord(Seq('MT', alphabet=alpha), id='Seq4')
        aln4 = MultipleSeqAlignment([seq4], alphabet=alpha)
        seq_aln4 = SeqAlignment(file_name='test', query_id='Seq1')
        seq_aln4.seq_length = 2
        seq_aln4.size = 1
        seq_aln4.seq_order = ['Seq4']
        seq_aln4.alignment = aln4
        seq_aln4.alphabet = alpha
        seq_aln4.query_sequence = seq1.seq
        seq_aln4.marked = [False]
        single4, pair4 = seq_aln4.characterize_positions()
        cls.terminals = {'Seq1': {'aln': seq_aln1, 'single': single1, 'pair': pair1},
                         'Seq2': {'aln': seq_aln2, 'single': single2, 'pair': pair2},
                         'Seq3': {'aln': seq_aln3, 'single': single3, 'pair': pair3},
                         'Seq4': {'aln': seq_aln4, 'single': single4, 'pair': pair4}}
        aln5 = MultipleSeqAlignment([seq1, seq2], alphabet=alpha)
        seq_aln5 = SeqAlignment(file_name='test', query_id='Seq1')
        seq_aln5.seq_length = 2
        seq_aln5.size = 2
        seq_aln5.seq_order = ['Seq1', 'Seq2']
        seq_aln5.alignment = aln5
        seq_aln5.alphabet = alpha
        seq_aln5.query_sequence = seq1.seq
        seq_aln5.marked = [False, False]
        single5 = single1 + single2
        pair5 = pair1 + pair2
        aln6 = MultipleSeqAlignment([seq3, seq4], alphabet=alpha)
        seq_aln6 = SeqAlignment(file_name='test', query_id='Seq1')
        seq_aln6.seq_length = 2
        seq_aln6.size = 2
        seq_aln6.seq_order = ['Seq3', 'Seq4']
        seq_aln6.alignment = aln6
        seq_aln6.alphabet = alpha
        seq_aln6.query_sequence = seq1.seq
        seq_aln6.marked = [False, False]
        single6 = single3 + single4
        pair6 = pair3 + pair4
        cls.first_parents = {'Inner1': {'aln': seq_aln5, 'single': single5, 'pair': pair5},
                             'Inner2': {'aln': seq_aln6, 'single': single6, 'pair': pair6}}

    def test1a_init(self):
        pos_scorer = PositionalScorer(seq_length=self.data_set.protein_data[self.small_structure_id]['Length'],
                                      pos_size=1, metric='identity')
        self.assertEqual(pos_scorer.sequence_length, self.data_set.protein_data[self.small_structure_id]['Length'])
        self.assertEqual(pos_scorer.position_size, 1)
        self.assertEqual(pos_scorer.dimensions, (self.data_set.protein_data[self.small_structure_id]['Length'],))
        self.assertEqual(pos_scorer.metric, 'identity')

    def test1b_init(self):
        pos_scorer = PositionalScorer(seq_length=self.data_set.protein_data[self.small_structure_id]['Length'],
                                      pos_size=2, metric='identity')
        self.assertEqual(pos_scorer.sequence_length, self.data_set.protein_data[self.small_structure_id]['Length'])
        self.assertEqual(pos_scorer.position_size, 2)
        self.assertEqual(pos_scorer.dimensions, (self.data_set.protein_data[self.small_structure_id]['Length'],
                                                 self.data_set.protein_data[self.small_structure_id]['Length']))
        self.assertEqual(pos_scorer.metric, 'identity')

    def test1c_init(self):
        with self.assertRaises(ValueError):
            pos_scorer = PositionalScorer(seq_length=self.data_set.protein_data[self.small_structure_id]['Length'],
                                          pos_size=1, metric='fake')
        with self.assertRaises(ValueError):
            pos_scorer = PositionalScorer(seq_length=self.data_set.protein_data[self.small_structure_id]['Length'],
                                          pos_size=2, metric='fake')

    # def test2a_group_identity_score(self):
    #     for x in self.terminals_small:
    #         for i in range(self.query_aln_fa_small.seq_length):
    #             self.assertEqual(group_identity_score(freq_table=self.terminals_small[x]['single'], pos=i), 0)
    #             for j in range(i, self.query_aln_fa_small.seq_length):
    #                 self.assertEqual(group_identity_score(freq_table=self.terminals_small[x]['pair'], pos=(i, j)), 0)

    def test2a_group_identity_score(self):
        for x in self.terminals:
            for i in range(2):
                self.assertEqual(group_identity_score(freq_table=self.terminals[x]['single'], pos=i), 0)
                for j in range(i + 1, 2):
                    self.assertEqual(group_identity_score(freq_table=self.terminals[x]['pair'], pos=(i, j)), 0)

    # def test2b_group_identity_score(self):
    #     for x in self.first_parents_small:
    #         for i in range(self.query_aln_fa_small.seq_length):
    #             if self.first_parents_small[x]['aln'][0, i] == self.first_parents_small[x]['aln'][1, i]:
    #                 expected_score = 0
    #             else:
    #                 expected_score = 1
    #             self.assertEqual(group_identity_score(freq_table=self.terminals_small[x]['single'], pos=i),
    #                              expected_score)
    #             for j in range(i, self.query_aln_fa_small.seq_length):
    #                 if ((self.first_parents_small[x]['aln'][0, i]) + self.first_parents_small[x]['aln'][0, j] ==
    #                         (self.first_parents_small[x]['aln'][1, i] + self.first_parents_small[x]['aln'][1, j])):
    #                     expected_score = 0
    #                 else:
    #                     expected_score = 1
    #                 self.assertEqual(group_identity_score(freq_table=self.terminals_small[x]['pair'], pos=(i, j)))

    def test2b_group_identity_score(self):
        for x in self.first_parents:
            for i in range(2):
                print('I: {}'.format(i))
                if self.first_parents[x]['aln'].alignment[0, i] == self.first_parents[x]['aln'].alignment[1, i]:
                    expected_score1 = 0
                else:
                    expected_score1 = 1
                self.assertEqual(group_identity_score(freq_table=self.first_parents[x]['single'], pos=i),
                                 expected_score1)
                for j in range(i + 1, 2):
                    print('J: {}'.format(j))
                    char1 = self.first_parents[x]['aln'].alignment[0, i] + self.first_parents[x]['aln'].alignment[0, j]
                    print(char1)
                    char2 = self.first_parents[x]['aln'].alignment[1, i] + self.first_parents[x]['aln'].alignment[1, j]
                    print(char2)
                    if char1 == char2:
                        expected_score2 = 0
                    else:
                        expected_score2 = 1
                    print('Frequency Table')
                    print(self.first_parents[x]['pair'].get_table())
                    self.assertEqual(group_identity_score(freq_table=self.first_parents[x]['pair'], pos=(i, j)),
                                     expected_score2)

    def test3a_rank_identity_score(self):
        group_single_scores = []
        group_pair_scores = []
        for x in self.terminals:
            group_single_score = np.zeros(2)
            group_pair_score = np.zeros((2, 2))
            for i in range(2):
                group_single_score[i] = group_identity_score(freq_table=self.terminals[x]['single'], pos=i)
                for j in range(i + 1, 2):
                    group_pair_score[i, j] = group_identity_score(freq_table=self.terminals[x]['pair'], pos=(i, j))
            group_single_scores.append(group_single_score)
            group_pair_scores.append(group_pair_score)
        group_single_scores = np.stack(group_single_scores, axis=0)
        rank_single_scores = rank_identity_score(score_matrix=group_single_scores)
        expected_rank_single_scores = np.zeros(2)
        diff_single = rank_single_scores - expected_rank_single_scores
        self.assertTrue(not diff_single.any())
        group_pair_scores = np.stack(group_pair_scores, axis=0)
        rank_pair_scores = rank_identity_score(score_matrix=group_pair_scores)
        expected_rank_pair_scores = np.zeros((2, 2))
        diff_pair = rank_pair_scores - expected_rank_pair_scores
        self.assertTrue(not diff_pair.any())

    def test3b_rank_identity_score(self):
        group_single_scores = []
        group_pair_scores = []
        for x in self.first_parents:
            group_single_score = np.zeros(2)
            group_pair_score = np.zeros((2, 2))
            for i in range(2):
                group_single_score[i] = group_identity_score(freq_table=self.first_parents[x]['single'], pos=i)
                for j in range(i + 1, 2):
                    group_pair_score[i, j] = group_identity_score(freq_table=self.first_parents[x]['pair'], pos=(i, j))
            group_single_scores.append(group_single_score)
            group_pair_scores.append(group_pair_score)
        group_single_scores = np.stack(group_single_scores, axis=0)
        rank_single_scores = rank_identity_score(score_matrix=group_single_scores)
        expected_rank_single_scores = np.array([0, 1])
        diff_single = rank_single_scores - expected_rank_single_scores
        self.assertTrue(not diff_single.any())
        group_pair_scores = np.stack(group_pair_scores, axis=0)
        rank_pair_scores = rank_identity_score(score_matrix=group_pair_scores)
        expected_rank_pair_scores = np.zeros((2, 2))
        expected_rank_pair_scores[0, 1] = 1
        diff_pair = rank_pair_scores - expected_rank_pair_scores
        self.assertTrue(not diff_pair.any())

    # def test4a_score_group(self):