"""
Created on February 5, 2020

@author: Daniel Konecki
"""
import os
import unittest
from shutil import rmtree
from Predictor import Predictor
from test_Base import TestBase


class TestPredictor(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestPredictor, cls).setUpClass()
        cls.small_fa_fn = cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln']
        cls.large_fa_fn = cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln']
        cls.out_small_dir = os.path.join(cls.testing_dir, cls.small_structure_id)
        rmtree(cls.out_small_dir)
        cls.out_large_dir = os.path.join(cls.testing_dir, cls.large_structure_id)
        rmtree(cls.out_large_dir)

    def evaluate_init(self, query, aln_file, out_dir, expected_length, expected_sequence):
        self.assertFalse(os.path.isdir(os.path.abspath(out_dir)))
        pred = Predictor(query=query, aln_file=aln_file, out_dir=out_dir)
        self.assertEqual(pred.out_dir, os.path.abspath(out_dir))
        self.assertTrue(os.path.isdir(os.path.abspath(out_dir)))
        self.assertEqual(pred.query, query)
        self.assertIsNotNone(pred.original_aln)
        self.assertGreaterEqual(pred.original_aln.seq_length, expected_length)
        self.assertEqual(str(pred.original_aln.query_sequence).replace('-', ''), expected_sequence)
        self.assertEqual(pred.original_aln_fn, os.path.join(out_dir, 'Original_Alignment.fa'))
        self.assertTrue(os.path.isfile(pred.original_aln_fn))
        self.assertIsNotNone(pred.non_gapped_aln)
        self.assertEqual(pred.non_gapped_aln.seq_length, expected_length)
        self.assertEqual(pred.non_gapped_aln.query_sequence, expected_sequence)
        self.assertEqual(pred.non_gapped_aln_fn, os.path.join(out_dir, 'Non-Gapped_Alignment.fa'))
        self.assertTrue(os.path.isfile(pred.non_gapped_aln_fn))
        self.assertEqual(pred.method, 'Base')
        self.assertIsNone(pred.scores)
        self.assertIsNone(pred.coverages)
        self.assertIsNone(pred.rankings)
        self.assertIsNone(pred.time)

    def test_1a_init(self):
        self.evaluate_init(query=self.small_structure_id, aln_file=self.small_fa_fn, out_dir=self.out_small_dir,
                           expected_length=self.data_set.protein_data[self.small_structure_id]['Length'],
                           expected_sequence=str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq))

    def test_1b_init(self):
        self.evaluate_init(query=self.large_structure_id, aln_file=self.large_fa_fn, out_dir=self.out_large_dir,
                           expected_length=self.data_set.protein_data[self.large_structure_id]['Length'],
                           expected_sequence=str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq))

    def evaluate_calculator_scores(self, query, aln_file, out_dir):
        pred = Predictor(query=query, aln_file=aln_file, out_dir=out_dir)
        with self.assertRaises(NotImplementedError):
            pred.calculate_scores()

    def test_2a_calculate_scores(self):
        self.evaluate_calculator_scores(query=self.small_structure_id, aln_file=self.small_fa_fn,
                                        out_dir=self.out_small_dir)

    def test_2b_calculate_scores(self):
        self.evaluate_calculator_scores(query=self.large_structure_id, aln_file=self.large_fa_fn,
                                        out_dir=self.out_large_dir)


if __name__ == '__main__':
    unittest.main()