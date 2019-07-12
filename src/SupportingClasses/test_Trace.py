"""
Created on July 11, 2019

@author: Daniel Konecki
"""
from Trace import Trace
from test_Base import TestBase
from SeqAlignment import SeqAlignment
from PhylogeneticTree import PhylogeneticTree
from AlignmentDistanceCalculator import AlignmentDistanceCalculator


class TestTrace(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestTrace, cls).setUpClass()
        cls.query_aln_fa_small = SeqAlignment(
            file_name=cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln'],
            query_id=cls.small_structure_id)
        cls.query_aln_fa_small.import_alignment()
        cls.phylo_tree_small = PhylogeneticTree()
        calc = AlignmentDistanceCalculator()
        cls.phylo_tree_small.construct_tree(dm=calc.get_distance(cls.query_aln_fa_small.alignment))
        cls.assignments_small = cls.phylo_tree_small.assign_group_rank()
        cls.query_aln_fa_large = SeqAlignment(
            file_name=cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln'],
            query_id=cls.large_structure_id)
        cls.query_aln_fa_large.import_alignment()
        cls.phylo_tree_large = PhylogeneticTree()
        calc = AlignmentDistanceCalculator()
        cls.phylo_tree_large.construct_tree(dm=calc.get_distance(cls.query_aln_fa_large.alignment))
        cls.assignments_large = cls.phylo_tree_large.assign_group_rank()

    def test1a_init(self):
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                            group_assignments=self.assignments_small, position_specific=True, pair_specific=True)
        self.assertEqual(trace_small.aln.file_name, self.query_aln_fa_small.file_name)
        self.assertEqual(trace_small.aln.query_id, self.query_aln_fa_small.query_id)
        for s in range(trace_small.aln.size):
            self.assertEqual(str(trace_small.aln.alignment[s].seq), str(self.query_aln_fa_small.alignment[s].seq))
        self.assertEqual(trace_small.aln.seq_order, self.query_aln_fa_small.seq_order)
        self.assertEqual(str(trace_small.aln.query_sequence), str(self.query_aln_fa_small.query_sequence))
        self.assertEqual(trace_small.aln.seq_length, self.query_aln_fa_small.seq_length)
        self.assertEqual(trace_small.aln.size, self.query_aln_fa_small.size)
        self.assertEqual(trace_small.aln.marked, self.query_aln_fa_small.marked)
        self.assertEqual(trace_small.aln.polymer_type, self.query_aln_fa_small.polymer_type)
        self.assertTrue(isinstance(trace_small.aln.alphabet, type(self.query_aln_fa_small.alphabet)))
        self.assertEqual(len(trace_small.aln.alphabet.letters), len(self.query_aln_fa_small.alphabet.letters))
        for char in trace_small.aln.alphabet.letters:
            self.assertTrue(char in self.query_aln_fa_small.alphabet.letters)
        self.assertEqual(trace_small.tree, self.phylo_tree_small)
        self.assertEqual(trace_small.assignments, self.assignments_small)
        self.assertEqual(trace_small.pos_specific, True)
        self.assertEqual(trace_small.pair_specific, True)

    def test1b_init(self):
        trace_large = Trace(alignment=self.query_aln_fa_large, phylo_tree=self.phylo_tree_large,
                            group_assignments=self.assignments_large, position_specific=True, pair_specific=True)
        self.assertEqual(trace_large.aln.file_name, self.query_aln_fa_large.file_name)
        self.assertEqual(trace_large.aln.query_id, self.query_aln_fa_large.query_id)
        for s in range(trace_large.aln.size):
            self.assertEqual(str(trace_large.aln.alignment[s].seq), str(self.query_aln_fa_large.alignment[s].seq))
        self.assertEqual(trace_large.aln.seq_order, self.query_aln_fa_large.seq_order)
        self.assertEqual(str(trace_large.aln.query_sequence), str(self.query_aln_fa_large.query_sequence))
        self.assertEqual(trace_large.aln.seq_length, self.query_aln_fa_large.seq_length)
        self.assertEqual(trace_large.aln.size, self.query_aln_fa_large.size)
        self.assertEqual(trace_large.aln.marked, self.query_aln_fa_large.marked)
        self.assertEqual(trace_large.aln.polymer_type, self.query_aln_fa_large.polymer_type)
        self.assertTrue(isinstance(trace_large.aln.alphabet, type(self.query_aln_fa_large.alphabet)))
        self.assertEqual(len(trace_large.aln.alphabet.letters), len(self.query_aln_fa_large.alphabet.letters))
        for char in trace_large.aln.alphabet.letters:
            self.assertTrue(char in self.query_aln_fa_large.alphabet.letters)
        self.assertEqual(trace_large.tree, self.phylo_tree_large)
        self.assertEqual(trace_large.assignments, self.assignments_large)
        self.assertEqual(trace_large.pos_specific, True)
        self.assertEqual(trace_large.pair_specific, True)

    def test2a_characterize_rank_groups(self):
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                            group_assignments=self.assignments_small, position_specific=True, pair_specific=True)
        trace_small.characterize_rank_groups()
        for rank in trace_small.assignments:
            for group in trace_small.assignments[rank]:
                sub_aln = self.query_aln_fa_small.generate_sub_alignment(
                    sequence_ids=trace_small.assignments[rank][group]['terminals'])
                single_table, pair_table = sub_aln.characterize_positions(single=True, pair=True)
                self.assertTrue('single' in trace_small.assignments[rank][group])
                self.assertEqual(trace_small.assignments[rank][group]['single'].get_table(), single_table.get_table())
                self.assertTrue('pair' in trace_small.assignments[rank][group])
                self.assertEqual(trace_small.assignments[rank][group]['pair'].get_table(), pair_table.get_table())

    def test2b_characterize_rank_groups(self):
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                            group_assignments=self.assignments_small, position_specific=True, pair_specific=True)
        trace_small.characterize_rank_groups(processes=self.max_processes)
        for rank in trace_small.assignments:
            for group in trace_small.assignments[rank]:
                sub_aln = self.query_aln_fa_small.generate_sub_alignment(
                    sequence_ids=trace_small.assignments[rank][group]['terminals'])
                single_table, pair_table = sub_aln.characterize_positions(single=True, pair=True)
                self.assertTrue('single' in trace_small.assignments[rank][group])
                self.assertEqual(trace_small.assignments[rank][group]['single'].get_table(), single_table.get_table())
                self.assertTrue('pair' in trace_small.assignments[rank][group])
                self.assertEqual(trace_small.assignments[rank][group]['pair'].get_table(), pair_table.get_table())

    def test2c_characterize_rank_groups(self):
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                            group_assignments=self.assignments_small, position_specific=True, pair_specific=False)
        trace_small.characterize_rank_groups()
        for rank in trace_small.assignments:
            for group in trace_small.assignments[rank]:
                sub_aln = self.query_aln_fa_small.generate_sub_alignment(
                    sequence_ids=trace_small.assignments[rank][group]['terminals'])
                single_table, pair_table = sub_aln.characterize_positions(single=True, pair=False)
                self.assertTrue('single' in trace_small.assignments[rank][group])
                self.assertEqual(trace_small.assignments[rank][group]['single'].get_table(), single_table.get_table())
                self.assertTrue('pair' in trace_small.assignments[rank][group])
                self.assertIsNone(trace_small.assignments[rank][group]['pair'])

    def test2d_characterize_rank_groups(self):
        trace_small = Trace(alignment=self.query_aln_fa_small, phylo_tree=self.phylo_tree_small,
                            group_assignments=self.assignments_small, position_specific=False, pair_specific=True)
        trace_small.characterize_rank_groups()
        for rank in trace_small.assignments:
            for group in trace_small.assignments[rank]:
                sub_aln = self.query_aln_fa_small.generate_sub_alignment(
                    sequence_ids=trace_small.assignments[rank][group]['terminals'])
                single_table, pair_table = sub_aln.characterize_positions(single=True, pair=True)
                self.assertTrue('single' in trace_small.assignments[rank][group])
                self.assertIsNone(trace_small.assignments[rank][group]['single'])
                self.assertTrue('pair' in trace_small.assignments[rank][group])
                self.assertEqual(trace_small.assignments[rank][group]['pair'].get_table(), pair_table.get_table())
