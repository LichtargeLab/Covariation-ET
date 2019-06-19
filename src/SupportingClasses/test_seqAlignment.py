"""
Created on Nov 9, 2018

@author: daniel
"""
import os
import numpy as np
from copy import deepcopy
from shutil import rmtree
from unittest import TestCase
from multiprocessing import cpu_count
from Bio.Align import MultipleSeqAlignment
from utils import build_mapping
from test_Base import TestBase
from SeqAlignment import SeqAlignment
from DataSetGenerator import DataSetGenerator
from AlignmentDistanceCalculator import AlignmentDistanceCalculator


# class TestSeqAlignment(TestCase):
class TestSeqAlignment(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestSeqAlignment, cls).setUpClass()
        cls.query_aln_fa_small = SeqAlignment(
            file_name=cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln'],
            query_id=cls.small_structure_id)
        cls.query_aln_fa_small.import_alignment()
        cls.query_aln_fa_large = SeqAlignment(
            file_name=cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln'],
            query_id=cls.large_structure_id)
        cls.query_aln_fa_large.import_alignment()
        cls.query_aln_msf_small = deepcopy(cls.query_aln_fa_small)
        cls.query_aln_msf_small.file_name = cls.data_set.protein_data[cls.small_structure_id]['Final_MSF_Aln']
        cls.query_aln_msf_large = deepcopy(cls.query_aln_fa_large)
        cls.query_aln_msf_large.file_name = cls.data_set.protein_data[cls.large_structure_id]['Final_MSF_Aln']
        cls.save_file_small = os.path.join(cls.testing_dir, '{}_aln.pkl'.format(cls.small_structure_id))
        cls.save_file_large = os.path.join(cls.testing_dir, '{}_aln.pkl'.format(cls.large_structure_id))
        cls.aln_file_small = os.path.join(cls.testing_dir, 'test_{}.fa'.format(cls.small_structure_id))
        cls.aln_file_large = os.path.join(cls.testing_dir, 'test_{}.fa'.format(cls.large_structure_id))
        cls.save_dir_small = os.path.join(cls.testing_dir, '{}_cache'.format(cls.small_structure_id))
        cls.save_dir_large = os.path.join(cls.testing_dir, '{}_cache'.format(cls.large_structure_id))

    def tearDown(self):
        try:
            os.remove(self.save_file_small)
        except OSError:
            pass
        try:
            os.remove(self.save_file_large)
        except OSError:
            pass
        try:
            os.remove(self.aln_file_small)
        except OSError:
            pass
        try:
            os.remove(self.aln_file_large)
        except OSError:
            pass
        try:
            rmtree(self.save_dir_small)
        except OSError:
            pass
        try:
            rmtree(self.save_dir_large)
        except OSError:
            pass

    def test1a_init(self):
        with self.assertRaises(TypeError):
            SeqAlignment()
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        self.assertFalse(aln_small.file_name.startswith('..'), 'Filename set to absolute path.')
        self.assertEqual(aln_small.query_id, self.small_structure_id, 'Query ID properly changed per lab protocol.')
        self.assertIsNone(aln_small.alignment, 'alignment is correctly declared as None.')
        self.assertIsNone(aln_small.seq_order, 'seq_order is correctly declared as None.')
        self.assertIsNone(aln_small.query_sequence, 'query_sequence is correctly declared as None.')
        self.assertIsNone(aln_small.seq_length, 'seq_length is correctly declared as None.')
        self.assertIsNone(aln_small.size, 'size is correctly declared as None.')
        self.assertIsNone(aln_small.marked)
        self.assertEqual(aln_small.polymer_type, 'Protein')

    def test1b_init(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        self.assertFalse(aln_large.file_name.startswith('..'), 'Filename set to absolute path.')
        self.assertEqual(aln_large.query_id, self.large_structure_id, 'Query ID properly changed per lab protocol.')
        self.assertIsNone(aln_large.alignment, 'alignment is correctly declared as None.')
        self.assertIsNone(aln_large.seq_order, 'seq_order is correctly declared as None.')
        self.assertIsNone(aln_large.query_sequence, 'query_sequence is correctly declared as None.')
        self.assertIsNone(aln_large.seq_length, 'seq_length is correctly declared as None.')
        self.assertIsNone(aln_large.size, 'size is correctly declared as None.')
        self.assertIsNone(aln_large.marked)
        self.assertEqual(aln_large.polymer_type, 'Protein')

    def test2a_import_alignment(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        for save in [None, self.save_file_small]:
            aln_small.import_alignment(save_file=save)
            self.assertFalse(aln_small.file_name.startswith('..'), 'Filename set to absolute path.')
            self.assertEqual(aln_small.query_id, self.small_structure_id, 'Query ID properly changed per lab protocol.')
            self.assertIsInstance(aln_small.alignment, MultipleSeqAlignment, 'alignment is correctly declared as None.')
            self.assertEqual(len(aln_small.seq_order), aln_small.size)
            self.assertTrue(self.small_structure_id in aln_small.seq_order)
            self.assertEqual(str(aln_small.query_sequence).replace('-', ''),
                             str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq))
            self.assertGreaterEqual(aln_small.seq_length,
                                    self.data_set.protein_data[self.small_structure_id]['Length'])
            self.assertEqual(aln_small.size,
                             self.data_set.protein_data[self.small_structure_id]['Final_Count'])
            self.assertEqual(len(aln_small.marked), aln_small.size)
            self.assertFalse(any(aln_small.marked))
            self.assertEqual(aln_small.polymer_type, 'Protein')
            if save is None:
                self.assertFalse(os.path.isfile(self.save_file_small), 'No save performed.')
            else:
                self.assertTrue(os.path.isfile(self.save_file_small), 'Save file made correctly.')

    def test2b_import_alignment(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        for save in [None, self.save_file_large]:
            aln_large.import_alignment(save_file=save)
            self.assertFalse(aln_large.file_name.startswith('..'), 'Filename set to absolute path.')
            self.assertEqual(aln_large.query_id, self.large_structure_id, 'Query ID properly changed per lab protocol.')
            self.assertIsInstance(aln_large.alignment, MultipleSeqAlignment, 'alignment is correctly declared as None.')
            self.assertEqual(len(aln_large.seq_order), aln_large.size)
            self.assertTrue(self.large_structure_id in aln_large.seq_order)
            self.assertEqual(str(aln_large.query_sequence).replace('-', ''),
                             str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq))
            self.assertGreaterEqual(aln_large.seq_length,
                                    self.data_set.protein_data[self.large_structure_id]['Length'])
            self.assertEqual(aln_large.size,
                             self.data_set.protein_data[self.large_structure_id]['Final_Count'])
            self.assertEqual(len(aln_large.marked), aln_large.size)
            self.assertFalse(any(aln_large.marked))
            self.assertEqual(aln_large.polymer_type, 'Protein')
            if save is None:
                self.assertFalse(os.path.isfile(self.save_file_large), 'No save performed.')
            else:
                self.assertTrue(os.path.isfile(self.save_file_large), 'Save file made correctly.')

    def test3a_write_out_alignment(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        with self.assertRaises(TypeError):
            aln_small.write_out_alignment(self.aln_file_small)
        aln_small.import_alignment()
        aln_small.write_out_alignment(self.aln_file_small)
        self.assertTrue(os.path.isfile(self.aln_file_small), 'Alignment written to correct file.')
        aln_small_prime = SeqAlignment(self.aln_file_small, self.small_structure_id)
        aln_small_prime.import_alignment()
        self.assertEqual(aln_small.seq_order, aln_small_prime.seq_order)
        self.assertEqual(aln_small.query_sequence, aln_small_prime.query_sequence)
        self.assertEqual(aln_small.seq_length, aln_small_prime.seq_length)
        self.assertEqual(aln_small.size, aln_small_prime.size)

    def test3b_write_out_alignment(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        with self.assertRaises(TypeError):
            aln_large.write_out_alignment(self.aln_file_large)
        aln_large.import_alignment()
        aln_large.write_out_alignment(self.aln_file_large)
        self.assertTrue(os.path.isfile(self.aln_file_large), 'Alignment written to correct file.')
        aln_obj2_prime = SeqAlignment(self.aln_file_large, self.large_structure_id)
        aln_obj2_prime.import_alignment()
        self.assertEqual(aln_large.seq_order, aln_obj2_prime.seq_order)
        self.assertEqual(aln_large.query_sequence, aln_obj2_prime.query_sequence)
        self.assertEqual(aln_large.seq_length, aln_obj2_prime.seq_length)
        self.assertEqual(aln_large.size, aln_obj2_prime.size)

    def test4a_generate_sub_alignment(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        aln_small_halved = aln_small.size // 2
        aln_small_seqrecords1 = aln_small.seq_order[:aln_small_halved]
        aln_small_sub1 = aln_small.generate_sub_alignment(sequence_ids=aln_small_seqrecords1)
        aln_small_seqrecords2 = aln_small.seq_order[aln_small_halved:]
        aln_small_sub2 = aln_small.generate_sub_alignment(sequence_ids=aln_small_seqrecords2)
        for sub_aln in [aln_small_sub1, aln_small_sub2]:
            self.assertFalse(sub_aln.file_name.startswith('..'), 'Filename set to absolute path.')
            self.assertEqual(sub_aln.query_id, self.small_structure_id, 'Query ID properly changed per lab protocol.')
            self.assertIsInstance(sub_aln.alignment, MultipleSeqAlignment, 'alignment is correctly declared as None.')
            self.assertEqual(str(sub_aln.query_sequence).replace('-', ''),
                             str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq).replace('-', ''))
            self.assertGreaterEqual(sub_aln.seq_length, self.data_set.protein_data[self.small_structure_id]['Length'])
            self.assertFalse(any(sub_aln.marked))
        self.assertEqual(aln_small_sub1.seq_order, aln_small_seqrecords1, 'seq_order imported correctly.')
        self.assertEqual(aln_small_sub1.size, aln_small_halved, 'size is correctly determined.')
        self.assertEqual(len(aln_small_sub1.marked), aln_small_halved)
        self.assertEqual(aln_small_sub1.polymer_type, 'Protein')
        self.assertEqual(aln_small_sub2.seq_order, aln_small_seqrecords2, 'seq_order imported correctly.')
        self.assertEqual(aln_small_sub2.size, aln_small.size - aln_small_halved, 'size is correctly determined.')
        self.assertEqual(len(aln_small_sub2.marked), aln_small.size - aln_small_halved)
        self.assertFalse(any(aln_small_sub2.marked))

    def test4b_generate_sub_alignment(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        aln_large.import_alignment()
        aln_large_halved = aln_large.size // 2
        aln_large_seqrecords1 = aln_large.seq_order[:aln_large_halved]
        aln_large_sub1 = aln_large.generate_sub_alignment(sequence_ids=aln_large_seqrecords1)
        aln_large_seqrecords2 = aln_large.seq_order[aln_large_halved:]
        aln_large_sub2 = aln_large.generate_sub_alignment(sequence_ids=aln_large_seqrecords2)
        for sub_aln in [aln_large_sub1, aln_large_sub2]:
            self.assertFalse(sub_aln.file_name.startswith('..'), 'Filename set to absolute path.')
            self.assertEqual(sub_aln.query_id, self.large_structure_id, 'Query ID properly changed per lab protocol.')
            self.assertIsInstance(sub_aln.alignment, MultipleSeqAlignment, 'alignment is correctly declared as None.')
            self.assertEqual(str(sub_aln.query_sequence).replace('-', ''),
                             str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq).replace('-', ''))
            self.assertGreaterEqual(sub_aln.seq_length, self.data_set.protein_data[self.large_structure_id]['Length'])
            self.assertFalse(any(sub_aln.marked))
        self.assertEqual(aln_large_sub1.seq_order, aln_large_seqrecords1, 'seq_order imported correctly.')
        self.assertEqual(aln_large_sub1.size, aln_large_halved, 'size is correctly determined.')
        self.assertEqual(len(aln_large_sub1.marked), aln_large_halved)
        self.assertFalse(any(aln_large_sub1.marked))
        self.assertEqual(aln_large_sub2.seq_order, aln_large_seqrecords2, 'seq_order imported correctly.')
        self.assertEqual(aln_large_sub2.size, aln_large.size - aln_large_halved, 'size is correctly determined.')
        self.assertEqual(len(aln_large_sub2.marked), aln_large.size - aln_large_halved)
        self.assertFalse(any(aln_large_sub2.marked))

    def test5a__subset_columns_one_position(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        with self.assertRaises(TypeError):
            aln_small._subset_columns([0, aln_small.seq_length // 2, aln_small.seq_length - 1])
        aln_small.import_alignment()
        # One position
        aln_small_alpha = aln_small._subset_columns([0])
        self.assertEqual(len(aln_small_alpha), aln_small.size)
        for rec in aln_small_alpha:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_small.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_small.query_sequence[0])
        aln_small_beta = aln_small._subset_columns([aln_small.seq_length - 1])
        self.assertEqual(len(aln_small_beta), aln_small.size)
        for rec in aln_small_beta:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_small.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_small.query_sequence[aln_small.seq_length - 1])
        aln_small_gamma = aln_small._subset_columns([aln_small.seq_length // 2])
        self.assertEqual(len(aln_small_gamma), aln_small.size)
        for rec in aln_small_gamma:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_small.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_small.query_sequence[aln_small.seq_length // 2])

    def test5b__subset_columns_single_range(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        with self.assertRaises(TypeError):
            aln_small._subset_columns([range(5) + range(aln_small.seq_length // 2, aln_small.seq_length // 2 + 5) +
                                       range(aln_small.seq_length - 5, aln_small.seq_length)])
        aln_small.import_alignment()
        # Single Range
        aln_small_delta = aln_small._subset_columns(range(5))
        self.assertEqual(len(aln_small_delta), aln_small.size)
        for rec in aln_small_delta:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_small.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_small.query_sequence[:5])
        aln_small_epsilon = aln_small._subset_columns(range(aln_small.seq_length - 5, aln_small.seq_length))
        self.assertEqual(len(aln_small_epsilon), aln_small.size)
        for rec in aln_small_epsilon:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_small.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_small.query_sequence[-5:])
        aln_small_zeta = aln_small._subset_columns(range(aln_small.seq_length // 2, aln_small.seq_length // 2 + 5))
        self.assertEqual(len(aln_small_zeta), aln_small.size)
        for rec in aln_small_zeta:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_small.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_small.query_sequence[aln_small.seq_length // 2:
                                                                        aln_small.seq_length // 2 + 5])

    def test5c__subset_columns_range_and_position(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        with self.assertRaises(TypeError):
            aln_small._subset_columns([0] + range(aln_small.seq_length // 2, aln_small.seq_length // 2 + 5))
        aln_small.import_alignment()
        # Mixed Range and Single Position
        aln_small_eta = aln_small._subset_columns([0] + range(aln_small.seq_length // 2, aln_small.seq_length // 2 + 5))
        self.assertEqual(len(aln_small_eta), aln_small.size)
        for rec in aln_small_eta:
            self.assertEqual(len(rec.seq), 6)
            if rec.id == aln_small.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_small.query_sequence[0] +
                                 aln_small.query_sequence[aln_small.seq_length // 2: aln_small.seq_length // 2 + 5])
        aln_small_theta = aln_small._subset_columns(range(aln_small.seq_length // 2, aln_small.seq_length // 2 + 5) +
                                                  [aln_small.seq_length - 1])
        self.assertEqual(len(aln_small_theta), aln_small.size)
        for rec in aln_small_theta:
            self.assertEqual(len(rec.seq), 6)
            if rec.id == aln_small.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_small.query_sequence[aln_small.seq_length // 2:
                                                                        aln_small.seq_length // 2 + 5] +
                                 aln_small.query_sequence[aln_small.seq_length - 1])
        aln_small_iota = aln_small._subset_columns(range(5) + [aln_small.seq_length // 2] +
                                                   range(aln_small.seq_length - 5, aln_small.seq_length))
        self.assertEqual(len(aln_small_iota), aln_small.size)
        for rec in aln_small_iota:
            self.assertEqual(len(rec.seq), 11)
            if rec.id == aln_small.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_small.query_sequence[:5] +
                                 aln_small.query_sequence[aln_small.seq_length // 2] +
                                 aln_small.query_sequence[-5:])

    def test5d__subset_columns_one_position(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        with self.assertRaises(TypeError):
            aln_large._subset_columns([0, aln_large.seq_length // 2, aln_large.seq_length - 1])
        aln_large.import_alignment()
        # One position
        aln_large_alpha = aln_large._subset_columns([0])
        self.assertEqual(len(aln_large_alpha), aln_large.size)
        for rec in aln_large_alpha:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_large.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_large.query_sequence[0])
        aln_large_beta = aln_large._subset_columns([aln_large.seq_length - 1])
        self.assertEqual(len(aln_large_beta), aln_large.size)
        for rec in aln_large_beta:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_large.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_large.query_sequence[aln_large.seq_length - 1])
        aln_large_gamma = aln_large._subset_columns([aln_large.seq_length // 2])
        self.assertEqual(len(aln_large_gamma), aln_large.size)
        for rec in aln_large_gamma:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_large.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_large.query_sequence[aln_large.seq_length // 2])

    def test5e__subset_columns_single_range(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        with self.assertRaises(TypeError):
            aln_large._subset_columns(range(5) + range(aln_large.seq_length // 2, aln_large.seq_length // 2 + 5) +
                                      range(aln_large.seq_length - 5, aln_large.seq_length))
        aln_large.import_alignment()
        # Single Range
        aln_large_delta = aln_large._subset_columns(range(5))
        self.assertEqual(len(aln_large_delta), aln_large.size)
        for rec in aln_large_delta:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_large.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_large.query_sequence[0:5])
        aln_large_epsilon = aln_large._subset_columns(range(aln_large.seq_length - 5, aln_large.seq_length))
        self.assertEqual(len(aln_large_epsilon), aln_large.size)
        for rec in aln_large_epsilon:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_large.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_large.query_sequence[-5:])
        aln_large_zeta = aln_large._subset_columns(range(aln_large.seq_length // 2, aln_large.seq_length // 2 + 5))
        self.assertEqual(len(aln_large_zeta), aln_large.size)
        for rec in aln_large_zeta:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_large.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_large.query_sequence[aln_large.seq_length // 2:
                                                                        aln_large.seq_length // 2 + 5])

    def test5f__subset_columns_range_and_position(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        with self.assertRaises(TypeError):
            aln_large._subset_columns([0] + range(aln_large.seq_length // 2, aln_large.seq_length // 2 + 5))
        # Mixed Range and Single Position
        aln_large.import_alignment()
        aln_large_eta = aln_large._subset_columns([0] + range(aln_large.seq_length // 2, aln_large.seq_length // 2 + 5))
        self.assertEqual(len(aln_large_eta), aln_large.size)
        for rec in aln_large_eta:
            self.assertEqual(len(rec.seq), 6)
            if rec.id == aln_large.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_large.query_sequence[0] +
                                 aln_large.query_sequence[aln_large.seq_length // 2: aln_large.seq_length // 2 + 5])
        aln_large_theta = aln_large._subset_columns(range(aln_large.seq_length // 2, aln_large.seq_length // 2 + 5) +
                                                  [aln_large.seq_length - 1])
        self.assertEqual(len(aln_large_theta), aln_large.size)
        for rec in aln_large_theta:
            self.assertEqual(len(rec.seq), 6)
            if rec.id == aln_large.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_large.query_sequence[aln_large.seq_length // 2:
                                                                       aln_large.seq_length // 2 + 5] +
                                 aln_large.query_sequence[aln_large.seq_length - 1])
        aln_large_iota = aln_large._subset_columns(range(5) + [aln_large.seq_length // 2] +
                                                 range(aln_large.seq_length - 5, aln_large.seq_length))
        self.assertEqual(len(aln_large_iota), aln_large.size)
        for rec in aln_large_iota:
            self.assertEqual(len(rec.seq), 11)
            if rec.id == aln_large.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_large.query_sequence[:5] +
                                 aln_large.query_sequence[aln_large.seq_length // 2] +
                                 aln_large.query_sequence[-5:])

    def test6a_remove_gaps(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        with self.assertRaises(TypeError):
            aln_small.remove_gaps()
        aln_small.import_alignment()
        aln_small_prime = aln_small.remove_gaps()
        self.assertEqual(aln_small.file_name, aln_small_prime.file_name)
        self.assertEqual(aln_small.query_id, aln_small_prime.query_id)
        self.assertEqual(aln_small.size, aln_small_prime.size)
        self.assertLess(aln_small_prime.seq_length, aln_small.seq_length)
        self.assertEqual(aln_small.seq_order, aln_small_prime.seq_order)
        self.assertEqual(str(aln_small.query_sequence).replace('-', ''), str(aln_small_prime.query_sequence))
        for i in range(aln_small.size):
            self.assertEqual(aln_small_prime.seq_length, len(aln_small_prime.alignment[i].seq))

    def test6b_remove_gaps(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        with self.assertRaises(TypeError):
            aln_large.remove_gaps()
        aln_large.import_alignment()
        aln_large_prime = aln_large.remove_gaps()
        self.assertEqual(aln_large.file_name, aln_large_prime.file_name)
        self.assertEqual(aln_large.query_id, aln_large_prime.query_id)
        self.assertEqual(aln_large.seq_order, aln_large_prime.seq_order)
        self.assertEqual(aln_large_prime.query_sequence, str(aln_large.query_sequence).replace('-', ''))
        self.assertLess(aln_large_prime.seq_length, aln_large.seq_length)
        self.assertEqual(aln_large.size, aln_large_prime.size)
        for i in range(aln_large.size):
            self.assertEqual(len(aln_large_prime.alignment[i].seq), aln_large_prime.seq_length)

    def test7a_generate_positional_sub_alignment(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        for i in range(aln_small.size - 1):
            aln_small_sub = aln_small.generate_positional_sub_alignment(i, i + 1)
            self.assertEqual(aln_small.file_name, aln_small_sub.file_name)
            self.assertEqual(aln_small.query_id, aln_small_sub.query_id)
            self.assertEqual(str(aln_small_sub.query_sequence.seq),
                             aln_small.query_sequence[i] + aln_small.query_sequence[i + 1])
            self.assertEqual(aln_small.size, aln_small_sub.size)
            self.assertEqual(aln_small.seq_order, aln_small_sub.seq_order)
            self.assertEqual(aln_small.marked, aln_small_sub.marked)
            for j in range(aln_small.size):
                self.assertEqual(str(aln_small_sub.alignment[j].seq),
                                 aln_small.alignment[j].seq[i] + aln_small.alignment[j].seq[i + 1])

    def test7b_generate_positional_sub_alignment(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        aln_large.import_alignment()
        for i in range(aln_large.size - 1):
            aln_large_sub = aln_large.generate_positional_sub_alignment(i, i + 1)
            self.assertEqual(aln_large.file_name, aln_large_sub.file_name)
            self.assertEqual(aln_large.query_id, aln_large_sub.query_id)
            self.assertEqual(str(aln_large_sub.query_sequence.seq),
                             aln_large.query_sequence[i] + aln_large.query_sequence[i + 1])
            self.assertEqual(aln_large.size, aln_large_sub.size)
            self.assertEqual(aln_large.seq_order, aln_large_sub.seq_order)
            self.assertEqual(aln_large.marked, aln_large_sub.marked)
            for j in range(aln_large.size):
                self.assertEqual(str(aln_large_sub.alignment[j].seq),
                                 aln_large.alignment[j].seq[i] + aln_large.alignment[j].seq[i + 1])

    def test8a_compute_effective_alignment_size(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        with self.assertRaises(TypeError):
            aln_small.compute_effective_alignment_size()
        aln_small.import_alignment()
        print('Alphabet: {}'.format(aln_small.alignment._alphabet))
        calc = AlignmentDistanceCalculator()
        distance_mat = np.array(calc.get_distance(aln_small.alignment))
        identity_mat = 1 - np.array(distance_mat)
        effective_size = 0.0
        for i in range(aln_small.size):
            n_i = 0.0
            for j in range(aln_small.size):
                if identity_mat[i, j] >= 0.62:
                    n_i += 1.0
            effective_size += 1.0 / n_i
        self.assertLess(abs(aln_small.compute_effective_alignment_size() - effective_size), 1.0e-12)

    def test8b_compute_effective_alignment_size(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        with self.assertRaises(TypeError):
            aln_large.compute_effective_alignment_size()
        aln_large.import_alignment()
        calc = AlignmentDistanceCalculator()
        distance_mat = np.array(calc.get_distance(aln_large.alignment))
        identity_mat = 1 - distance_mat
        effective_size = 0.0
        for i in range(aln_large.size):
            n_i = 0.0
            for j in range(aln_large.size):
                if identity_mat[i, j] >= 0.62:
                    n_i += 1.0
            effective_size += 1.0 / n_i
        self.assertLess(abs(aln_large.compute_effective_alignment_size() - effective_size), 1.0e-12)

    def test9a_determine_usable_positions(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        pos, evidence = aln_small.determine_usable_positions(ratio=0.5)
        usable_pos = []
        for i in range(aln_small.seq_length):
            count = 0
            for j in range(aln_small.size):
                if aln_small.alignment[j, i] != '-':
                    count += 1
            if count >= (aln_small.size / 2):
                usable_pos.append(i)
            self.assertEqual(evidence[i], count)
        self.assertEqual(list(pos), usable_pos)

    def test9b_determine_usable_positions(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        aln_large.import_alignment()
        pos, evidence = aln_large.determine_usable_positions(ratio=0.5)
        usable_pos2 = []
        for i in range(aln_large.seq_length):
            count = 0
            for j in range(aln_large.size):
                if aln_large.alignment[j, i] != '-':
                    count += 1
            if count >= (aln_large.size / 2):
                usable_pos2.append(i)
            self.assertEqual(evidence[i], count)
        self.assertEqual(list(pos), usable_pos2)

    def test10a_identify_comparable_sequences(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        for i in range(1, aln_small.seq_length):
            comp_tup = aln_small.identify_comparable_sequences(pos1=0, pos2=i)
            col1 = aln_small.alignment[:, 0]
            col1_sub = []
            col2 = aln_small.alignment[:, i]
            col2_sub = []
            indices = []
            count = 0
            for j in range(aln_small.size):
                if (col1[j] != '-') and (col2[j] != '-'):
                    col1_sub.append(col1[j])
                    col2_sub.append(col2[j])
                    indices.append(j)
                    count += 1
            self.assertEqual(list(comp_tup[0]), col1_sub)
            self.assertEqual(list(comp_tup[1]), col2_sub)
            self.assertEqual(list(comp_tup[2]), indices)
            self.assertEqual(comp_tup[3], count)

    def test10b_identify_comparable_sequences(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        aln_large.import_alignment()
        for i in range(1, aln_large.seq_length):
            comp_tup = aln_large.identify_comparable_sequences(pos1=0, pos2=i)
            col1 = aln_large.alignment[:, 0]
            col1_sub = []
            col2 = aln_large.alignment[:, i]
            col2_sub = []
            indices = []
            count = 0
            for j in range(aln_large.size):
                if (col1[j] != '-') and (col2[j] != '-'):
                    col1_sub.append(col1[j])
                    col2_sub.append(col2[j])
                    indices.append(j)
                    count += 1
            self.assertEqual(list(comp_tup[0]), col1_sub)
            self.assertEqual(list(comp_tup[1]), col2_sub)
            self.assertEqual(list(comp_tup[2]), indices)
            self.assertEqual(comp_tup[3], count)

    def test11a_consensus_sequences(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        consensus = aln_small.consensus_sequence()
        self.assertEqual(consensus.id, 'Consensus Sequence')
        for i in range(aln_small.seq_length):
            best_count = -1
            best_aa = None
            counts = {}
            for j in range(aln_small.size):
                aa = aln_small.alignment[j, i]
                if aa not in counts:
                    counts[aa] = 0
                counts[aa] += 1
                if counts[aa] > best_count:
                    best_count = counts[aa]
                    best_aa = aa
                elif counts[aa] == best_count and aa < best_aa:
                    best_aa = aa
            self.assertEqual(consensus.seq[i], best_aa)

    def test11b_consensus_sequences(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        aln_large.import_alignment()
        consensus = aln_large.consensus_sequence()
        self.assertEqual(consensus.id, 'Consensus Sequence')
        for i in range(aln_large.seq_length):
            best_count = -1
            best_aa = None
            counts = {}
            for j in range(aln_large.size):
                aa = aln_large.alignment[j, i]
                if aa not in counts:
                    counts[aa] = 0
                counts[aa] += 1
                if counts[aa] > best_count:
                    best_count = counts[aa]
                    best_aa = aa
                elif counts[aa] == best_count and aa < best_aa:
                    best_aa = aa
            self.assertEqual(consensus.seq[i], best_aa)

    def test12a_alignment_to_num(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        _, _, mapping = build_mapping(alphabet=aln_small.alphabet)
        aln_obj1_num = aln_small._alignment_to_num(mapping=mapping)
        for i in range(aln_small.size):
            for j in range(aln_small.seq_length):
                self.assertEqual(aln_obj1_num[i, j], mapping[aln_small.alignment[i, j]])

    def test12b_alignment_to_num(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        aln_large.import_alignment()
        _, _, mapping = build_mapping(alphabet=aln_large.alphabet)
        aln_obj2_num = aln_large._alignment_to_num(mapping=mapping)
        for i in range(aln_large.size):
            for j in range(aln_large.seq_length):
                self.assertEqual(aln_obj2_num[i, j], mapping[aln_large.alignment[i, j]])

    def test13a__gap_z_score_check(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        alpha_size, gap_chars, mapping = build_mapping(alphabet=aln_small.alphabet)
        numeric_aln = aln_small._alignment_to_num(mapping=mapping)
        passing_sequences = aln_small._gap_z_score_check(z_score_cutoff=0.0, num_aln=numeric_aln, gap_num=alpha_size)
        self.assertEqual(len(passing_sequences), aln_small.size)
        gap_counts = []
        for i in range(aln_small.size):
            gap_count = 0
            for j in range(aln_small.seq_length):
                if aln_small.alignment[i, j] in gap_chars:
                    gap_count += 1
            gap_counts.append(gap_count)
        mean_count = np.mean(gap_counts)
        for i in range(aln_small.size):
            self.assertEqual(passing_sequences[i], gap_counts[i] < mean_count)

    def test13b__gap_z_score_check(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        aln_large.import_alignment()
        alpha_size, gap_chars, mapping = build_mapping(alphabet=aln_large.alphabet)
        numeric_aln = aln_large._alignment_to_num(mapping=mapping)
        passing_sequences = aln_large._gap_z_score_check(z_score_cutoff=0.0, num_aln=numeric_aln, gap_num=alpha_size)
        self.assertEqual(len(passing_sequences), aln_large.size)
        gap_counts = []
        for i in range(aln_large.size):
            gap_count = 0
            for j in range(aln_large.seq_length):
                if aln_large.alignment[i, j] in gap_chars:
                    gap_count += 1
            gap_counts.append(gap_count)
        mean_count = np.mean(gap_counts)
        for i in range(aln_large.size):
            self.assertEqual(passing_sequences[i], gap_counts[i] < mean_count)

    def test14a__gap_percentile_check(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        consensus = aln_small.consensus_sequence()
        alpha_size, gap_chars, mapping = build_mapping(alphabet=aln_small.alphabet)
        numeric_aln = aln_small._alignment_to_num(mapping=mapping)
        passing_sequences = aln_small._gap_percentile_check(percentile_cutoff=0.15, num_aln=numeric_aln,
                                                            gap_num=alpha_size, mapping=mapping)
        max_differences = np.floor(aln_small.size * 0.15)
        for i in range(aln_small.size):
            diff_count = 0
            for j in range(aln_small.seq_length):
                if (((aln_small.alignment[i, j] in gap_chars) or (consensus.seq[j] in gap_chars)) and
                        (aln_small.alignment[i, j] != consensus.seq[j])):
                    diff_count += 1
            self.assertEqual(passing_sequences[i], diff_count <= max_differences)

    def test14b__gap_percentile_check(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        aln_large.import_alignment()
        consensus = aln_large.consensus_sequence()
        alpha_size, gap_chars, mapping = build_mapping(alphabet=aln_large.alphabet)
        numeric_aln = aln_large._alignment_to_num(mapping=mapping)
        passing_sequences = aln_large._gap_percentile_check(percentile_cutoff=0.15, num_aln=numeric_aln,
                                                            gap_num=alpha_size, mapping=mapping)
        max_differences = np.floor(aln_large.size * 0.15)
        for i in range(aln_large.size):
            diff_count = 0
            for j in range(aln_large.seq_length):
                if (((aln_large.alignment[i, j] in gap_chars) or (consensus.seq[j] in gap_chars)) and
                        (aln_large.alignment[i, j] != consensus.seq[j])):
                    diff_count += 1
            self.assertEqual(passing_sequences[i], diff_count <= max_differences)

    def test15a_gap_evaluation(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        kept, removed = aln_small.gap_evaluation(size_cutoff=15, z_score_cutoff=0.0, percentile_cutoff=0.15)
        alpha_size, gap_chars, mapping = build_mapping(alphabet=aln_small.alphabet)
        self.assertEqual(len(kept) + len(removed), aln_small.size)
        gap_counts = []
        for i in range(aln_small.size):
            gap_count = 0
            for j in range(aln_small.seq_length):
                if aln_small.alignment[i, j] in gap_chars:
                    gap_count += 1
            gap_counts.append(gap_count)
        mean_count = np.mean(gap_counts)
        for i in range(aln_small.size):
            if gap_counts[i] < mean_count:
                self.assertTrue(aln_small.seq_order[i] in kept)
            else:
                self.assertTrue(aln_small.seq_order[i] in removed)

    def test15b_gap_evaluation(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        aln_small_sub = aln_small.generate_sub_alignment(aln_small.seq_order[:10])
        consensus = aln_small_sub.consensus_sequence()
        kept, removed = aln_small_sub.gap_evaluation(size_cutoff=15, z_score_cutoff=0.0, percentile_cutoff=0.15)
        alpha_size, gap_chars, _ = build_mapping(alphabet=aln_small.alphabet)
        max_differences = np.floor(aln_small_sub.size * 0.15)
        for i in range(aln_small_sub.size):
            diff_count = 0
            for j in range(aln_small_sub.seq_length):
                if (((aln_small_sub.alignment[i, j] in gap_chars) or (consensus.seq[j] in gap_chars)) and
                        (aln_small_sub.alignment[i, j] != consensus.seq[j])):
                    diff_count += 1
            if diff_count <= max_differences:
                self.assertTrue(aln_small_sub.seq_order[i] in kept)
            else:
                self.assertTrue(aln_small_sub.seq_order[i] in removed)

    def test15c_gap_evaluation(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_large.import_alignment()
        kept, removed = aln_large.gap_evaluation(size_cutoff=15, z_score_cutoff=0.0, percentile_cutoff=0.15)
        alpha_size, gap_chars, mapping = build_mapping(alphabet=aln_large.alphabet)
        self.assertEqual(len(kept) + len(removed), aln_large.size)
        gap_counts = []
        for i in range(aln_large.size):
            gap_count = 0
            for j in range(aln_large.seq_length):
                if aln_large.alignment[i, j] in gap_chars:
                    gap_count += 1
            gap_counts.append(gap_count)
        mean_count = np.mean(gap_counts)
        for i in range(aln_large.size):
            if gap_counts[i] < mean_count:
                self.assertTrue(aln_large.seq_order[i] in kept)
            else:
                self.assertTrue(aln_large.seq_order[i] in removed)

    def test15d_gap_evaluation(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_large.import_alignment()
        aln_large_sub = aln_large.generate_sub_alignment(aln_large.seq_order[:10])
        consensus = aln_large_sub.consensus_sequence()
        kept, removed = aln_large_sub.gap_evaluation(size_cutoff=15, z_score_cutoff=0.0, percentile_cutoff=0.15)
        alpha_size, gap_chars, _ = build_mapping(alphabet=aln_large.alphabet)
        max_differences = np.floor(aln_large_sub.size * 0.15)
        for i in range(aln_large_sub.size):
            diff_count = 0
            for j in range(aln_large_sub.seq_length):
                if (((aln_large_sub.alignment[i, j] in gap_chars) or (consensus.seq[j] in gap_chars)) and
                        (aln_large_sub.alignment[i, j] != consensus.seq[j])):
                    diff_count += 1
            if diff_count <= max_differences:
                self.assertTrue(aln_large_sub.seq_order[i] in kept)
            else:
                self.assertTrue(aln_large_sub.seq_order[i] in removed)

    def test16a_heatmap_plot(self):
        if not os.path.isdir(self.save_dir_small):
            os.makedirs(self.save_dir_small)
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        _, _, mapping = build_mapping(alphabet=aln_small.alphabet)
        name = '{} Alignment Visualization'.format(self.small_structure_id)
        df1a, _hm1a = aln_small.heatmap_plot(name=name, out_dir=self.save_dir_small, save=True)
        for i in range(aln_small.size):
            self.assertEqual(df1a.index[i], aln_small.seq_order[i])
            for j in range(aln_small.seq_length):
                self.assertEqual(df1a.loc[aln_small.seq_order[i], '{}:{}'.format(j, aln_small.query_sequence[j])],
                                 mapping[aln_small.alignment[i, j]])
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir_small, name.replace(' ', '_') + '.eps')))

    def test16b_heatmap_plot(self):
        if not os.path.isdir(self.save_dir_large):
            os.makedirs(self.save_dir_large)
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_large.import_alignment()
        _, _, mapping = build_mapping(alphabet=aln_large.alphabet)
        name = '{} Alignment Visualization'.format(self.large_structure_id)
        df1a, _hm1a = aln_large.heatmap_plot(name=name, out_dir=self.save_dir_large, save=True)
        for i in range(aln_large.size):
            self.assertEqual(df1a.index[i], aln_large.seq_order[i])
            for j in range(aln_large.seq_length):
                self.assertEqual(df1a.loc[aln_large.seq_order[i], '{}:{}'.format(j, aln_large.query_sequence[j])],
                                 mapping[aln_large.alignment[i, j]])
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir_large, name.replace(' ', '_') + '.eps')))

    # def test_set_tree_ordering(self):
    #
    #     def check(seq_dict, curr, prev=None):
    #         if prev is None:
    #             return True
    #         c_prev = 0
    #         c_curr = 0
    #         while (c_prev != (prev - 1)) and (c_curr != (curr - 1)):
    #             if not seq_dict[curr][c_curr].issubset(seq_dict[prev][c_prev]):
    #                 c_prev += 1
    #             if not seq_dict[curr][c_curr].issubset(seq_dict[prev][c_prev]):
    #                 return False
    #             c_curr += 1
    #         return True
    #
    #     aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
    #     with self.assertRaises(TypeError):
    #         aln_obj1.set_tree_ordering()
    #     aln_obj1.import_alignment()
    #     aln_obj1.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir1,
    #                                clustering_args={'affinity': 'euclidean', 'linkage': 'ward', 'model': 'identity'})
    #     self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
    #     self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
    #     self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
    #     for k in range(2, 5):
    #         self.assertTrue(check(aln_obj1.sequence_assignments, curr=k, prev=k-1))
    #     clusters = [1, 2, 3, 5, 7, 10, 25]
    #     aln_obj1.set_tree_ordering(tree_depth=clusters)
    #     self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
    #     self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
    #     self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
    #     for i in range(len(clusters)):
    #         self.assertTrue(check(aln_obj1.sequence_assignments, curr=clusters[i], prev=clusters[i - 1]),
    #                         'Error on i:{}, curr:{}, prev:{}'.format(i, clusters[i], clusters[i - 1]))
    #     aln_obj1.set_tree_ordering()
    #     self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
    #     self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
    #     self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
    #     for k in range(1, aln_obj1.size):
    #         self.assertTrue(check(aln_obj1.sequence_assignments, curr=k, prev=k - 1))
    #     aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
    #     with self.assertRaises(TypeError):
    #         aln_obj2._random_assignment(n_cluster=2)
    #     aln_obj2.import_alignment()
    #     aln_obj2.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir2,
    #                                clustering_args={'affinity': 'euclidean', 'linkage': 'ward', 'model': 'identity'})
    #     self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
    #     self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
    #     self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
    #     for k in range(2, 5):
    #         self.assertTrue(check(aln_obj2.sequence_assignments, curr=k, prev=k - 1))
    #     clusters = [1, 2, 3, 5, 7, 10, 25]
    #     aln_obj2.set_tree_ordering(tree_depth=clusters)
    #     self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
    #     self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
    #     self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
    #     for i in range(len(clusters)):
    #         self.assertTrue(check(aln_obj2.sequence_assignments, curr=clusters[i], prev=clusters[i - 1]))
    #     aln_obj2.set_tree_ordering()
    #     self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
    #     self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
    #     self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
    #     for k in range(1, aln_obj2.size):
    #         self.assertTrue(check(aln_obj2.sequence_assignments, curr=k, prev=k - 1))
    #     ################################################################################################################
    #     aln_obj1.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir1, clustering='random')
    #     self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
    #     # self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
    #     self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
    #     # for k in range(2, 5):
    #     #     self.assertTrue(check(aln_obj1.sequence_assignments, curr=k, prev=k - 1))
    #     clusters = [1, 2, 3, 5, 7, 10, 25]
    #     aln_obj1.set_tree_ordering(tree_depth=clusters, cache_dir=self.save_dir1, clustering='random')
    #     self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
    #     self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
    #     self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
    #     # for i in range(len(clusters)):
    #     #     self.assertTrue(check(aln_obj1.sequence_assignments, curr=clusters[i], prev=clusters[i - 1]),
    #     #                     'Error on i:{}, curr:{}, prev:{}'.format(i, clusters[i], clusters[i - 1]))
    #     aln_obj1.set_tree_ordering(clustering='random', cache_dir=self.save_dir1)
    #     self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
    #     self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
    #     self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
    #     # for k in range(1, aln_obj1.size):
    #     #     self.assertTrue(check(aln_obj1.sequence_assignments, curr=k, prev=k - 1))
    #     aln_obj2.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir2, clustering='random')
    #     self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
    #     self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
    #     self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
    #     # for k in range(2, 5):
    #     #     self.assertTrue(check(aln_obj2.sequence_assignments, curr=k, prev=k - 1))
    #     clusters = [1, 2, 3, 5, 7, 10, 25]
    #     aln_obj2.set_tree_ordering(tree_depth=clusters, cache_dir=self.save_dir2, clustering='random')
    #     self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
    #     # self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
    #     self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
    #     # for i in range(len(clusters)):
    #     #     self.assertTrue(check(aln_obj2.sequence_assignments, curr=clusters[i], prev=clusters[i - 1]))
    #     aln_obj2.set_tree_ordering(cache_dir=self.save_dir2, clustering='random')
    #     self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
    #     self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
    #     self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
    #     # for k in range(1, aln_obj2.size):
    #     #     self.assertTrue(check(aln_obj2.sequence_assignments, curr=k, prev=k - 1))
    #     ################################################################################################################
    #     aln_obj1.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir1, clustering='upgma')
    #     self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
    #     # self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
    #     self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
    #     for k in range(2, 5):
    #         self.assertTrue(check(aln_obj1.sequence_assignments, curr=k, prev=k - 1))
    #     clusters = [1, 2, 3, 5, 7, 10, 25]
    #     aln_obj1.set_tree_ordering(tree_depth=clusters, cache_dir=self.save_dir1, clustering='upgma')
    #     self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
    #     self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
    #     self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
    #     for i in range(len(clusters)):
    #         self.assertTrue(check(aln_obj1.sequence_assignments, curr=clusters[i], prev=clusters[i - 1]),
    #                         'Error on i:{}, curr:{}, prev:{}'.format(i, clusters[i], clusters[i - 1]))
    #     aln_obj1.set_tree_ordering(clustering='upgma', cache_dir=self.save_dir1)
    #     self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
    #     self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
    #     self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
    #     for k in range(1, aln_obj1.size):
    #         self.assertTrue(check(aln_obj1.sequence_assignments, curr=k, prev=k - 1))
    #     aln_obj2.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir2, clustering='upgma')
    #     self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
    #     self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
    #     self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
    #     for k in range(2, 5):
    #         self.assertTrue(check(aln_obj2.sequence_assignments, curr=k, prev=k - 1))
    #     clusters = [1, 2, 3, 5, 7, 10, 25]
    #     aln_obj2.set_tree_ordering(tree_depth=clusters, cache_dir=self.save_dir2, clustering='upgma')
    #     self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
    #     # self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
    #     self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
    #     for i in range(len(clusters)):
    #         self.assertTrue(check(aln_obj2.sequence_assignments, curr=clusters[i], prev=clusters[i - 1]))
    #     aln_obj2.set_tree_ordering(cache_dir=self.save_dir2, clustering='upgma')
    #     self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
    #     self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
    #     self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
    #     for k in range(1, aln_obj2.size):
    #         self.assertTrue(check(aln_obj2.sequence_assignments, curr=k, prev=k - 1))

    # def test_visualize_tree(self):
    #     aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
    #     with self.assertRaises(TypeError):
    #         aln_obj1.set_tree_ordering()
    #     aln_obj1.import_alignment()
    #     with self.assertRaises(ValueError):
    #         aln_obj1.visualize_tree(out_dir=self.save_dir1)
    #     aln_obj1.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir1,
    #                                clustering_args={'affinity': 'euclidean', 'linkage': 'ward', 'model': 'identity'})
    #     aln_obj1_df = aln_obj1.visualize_tree(out_dir=self.save_dir1)
    #     self.assertEqual(aln_obj1.tree_order, list(aln_obj1_df.index))
    #     for i in range(1, 5):
    #         for j in aln_obj1_df.index:
    #             self.assertIn(j,  aln_obj1.sequence_assignments[i][aln_obj1_df.loc[j, i]])
    #     self.assertTrue(os.path.isfile(os.path.join(self.save_dir1, 'query_1c17A_Sequence_Assignment.csv')))
    #     self.assertTrue(os.path.isfile(os.path.join(self.save_dir1, 'query_1c17A_Sequence_Assignment.eps')))
    #     aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
    #     with self.assertRaises(TypeError):
    #         aln_obj2._random_assignment(n_cluster=2)
    #     aln_obj2.import_alignment()
    #     with self.assertRaises(ValueError):
    #         aln_obj2.visualize_tree(out_dir=self.save_dir2)
    #     aln_obj2.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir2,
    #                        clustering_args={'affinity': 'euclidean', 'linkage': 'ward', 'model': 'identity'})
    #     aln_obj2_df = aln_obj2.visualize_tree(out_dir=self.save_dir2)
    #     self.assertEqual(aln_obj2.tree_order, list(aln_obj2_df.index))
    #     for i in range(1, 5):
    #         for j in aln_obj2_df.index:
    #             self.assertIn(j,  aln_obj2.sequence_assignments[i][aln_obj2_df.loc[j, i]])
    #     self.assertTrue(os.path.isfile(os.path.join(self.save_dir2, 'query_1h1vA_Sequence_Assignment.csv')))
    #     self.assertTrue(os.path.isfile(os.path.join(self.save_dir2, 'query_1h1vA_Sequence_Assignment.eps')))
    #
    # def test_get_branch_cluster(self):
    #     aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
    #     with self.assertRaises(TypeError):
    #         aln_obj1.set_tree_ordering()
    #     aln_obj1.import_alignment()
    #     aln_obj1.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir1,
    #                                clustering_args={'affinity': 'euclidean', 'linkage': 'ward', 'model': 'identity'})
    #     for k in aln_obj1.sequence_assignments:
    #         for c in aln_obj1.sequence_assignments[k]:
    #             aln_obj1_sub = aln_obj1.get_branch_cluster(k, c)
    #             aln_obj1_sub_prime = aln_obj1.generate_sub_alignment(aln_obj1.sequence_assignments[k][c])
    #             self.assertEqual(aln_obj1_sub.file_name, aln_obj1_sub_prime.file_name)
    #             self.assertEqual(aln_obj1_sub.query_id, aln_obj1_sub_prime.query_id)
    #             self.assertEqual(aln_obj1.query_sequence, aln_obj1_sub.query_sequence)
    #             self.assertEqual(aln_obj1_sub.distance_matrix, aln_obj1_sub_prime.distance_matrix)
    #             self.assertEqual(aln_obj1_sub.sequence_assignments, aln_obj1_sub_prime.sequence_assignments)
    #             self.assertEqual(aln_obj1_sub.size, aln_obj1_sub_prime.size)
    #             self.assertEqual(aln_obj1_sub.seq_order, aln_obj1_sub_prime.seq_order)
    #             self.assertEqual(aln_obj1_sub.tree_order, aln_obj1_sub_prime.tree_order)
    #     aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
    #     with self.assertRaises(TypeError):
    #         aln_obj2._random_assignment(n_cluster=2)
    #     aln_obj2.import_alignment()
    #     aln_obj2.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir2,
    #                                clustering_args={'affinity': 'euclidean', 'linkage': 'ward', 'model': 'identity'})
    #     for k in aln_obj2.sequence_assignments:
    #         for c in aln_obj2.sequence_assignments[k]:
    #             aln_obj2_sub = aln_obj2.get_branch_cluster(k, c)
    #             aln_obj2_sub_prime = aln_obj2.generate_sub_alignment(aln_obj2.sequence_assignments[k][c])
    #             self.assertEqual(aln_obj2_sub.file_name, aln_obj2_sub_prime.file_name)
    #             self.assertEqual(aln_obj2_sub.query_id, aln_obj2_sub_prime.query_id)
    #             self.assertEqual(aln_obj2.query_sequence, aln_obj2_sub.query_sequence)
    #             self.assertEqual(aln_obj2_sub.distance_matrix, aln_obj2_sub_prime.distance_matrix)
    #             self.assertEqual(aln_obj2_sub.sequence_assignments, aln_obj2_sub_prime.sequence_assignments)
    #             self.assertEqual(aln_obj2_sub.size, aln_obj2_sub_prime.size)
    #             self.assertEqual(aln_obj2_sub.seq_order, aln_obj2_sub_prime.seq_order)
    #             self.assertEqual(aln_obj2_sub.tree_order, aln_obj2_sub_prime.tree_order)

    # def test_random_assignment(self):
    #     aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
    #     with self.assertRaises(TypeError):
    #         aln_obj1._random_assignment(n_cluster=2)
    #     aln_obj1.import_alignment()
    #     aln_obj1_clusters1 = aln_obj1._random_assignment(n_cluster=2)
    #     self.assertEqual(len(set(aln_obj1_clusters1)), 2)
    #     self.assertTrue(0 in set(aln_obj1_clusters1))
    #     self.assertTrue(1 in set(aln_obj1_clusters1))
    #     aln_obj1_clusters2 = aln_obj1._random_assignment(n_cluster=2)
    #     self.assertEqual(len(set(aln_obj1_clusters2)), 2)
    #     self.assertTrue(0 in set(aln_obj1_clusters2))
    #     self.assertTrue(1 in set(aln_obj1_clusters2))
    #     self.assertNotEqual(aln_obj1_clusters1, aln_obj1_clusters2)
    #     aln_obj1_clusters3 = aln_obj1._random_assignment(n_cluster=3, cache_dir=self.save_dir1)
    #     self.assertEqual(len(set(aln_obj1_clusters3)), 3)
    #     self.assertTrue(0 in set(aln_obj1_clusters3))
    #     self.assertTrue(1 in set(aln_obj1_clusters3))
    #     self.assertTrue(2 in set(aln_obj1_clusters3))
    #     self.assertTrue(os.path.isfile(os.path.join(self.save_dir1, 'joblib', 'K_3.pkl')))
    #     aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
    #     with self.assertRaises(TypeError):
    #         aln_obj2._random_assignment(n_cluster=2)
    #     aln_obj2.import_alignment()
    #     aln_obj2_clusters1 = aln_obj2._random_assignment(n_cluster=2)
    #     self.assertEqual(len(set(aln_obj2_clusters1)), 2)
    #     self.assertTrue(0 in set(aln_obj2_clusters1))
    #     self.assertTrue(1 in set(aln_obj2_clusters1))
    #     aln_obj2_clusters2 = aln_obj2._random_assignment(n_cluster=2)
    #     self.assertEqual(len(set(aln_obj2_clusters2)), 2)
    #     self.assertTrue(0 in set(aln_obj2_clusters2))
    #     self.assertTrue(1 in set(aln_obj2_clusters2))
    #     self.assertNotEqual(aln_obj2_clusters1, aln_obj2_clusters2)
    #     aln_obj2_clusters3 = aln_obj2._random_assignment(n_cluster=3, cache_dir=self.save_dir2)
    #     self.assertEqual(len(set(aln_obj2_clusters3)), 3)
    #     self.assertTrue(0 in set(aln_obj2_clusters3))
    #     self.assertTrue(1 in set(aln_obj2_clusters3))
    #     self.assertTrue(2 in set(aln_obj2_clusters3))
    #     self.assertTrue(os.path.isfile(os.path.join(self.save_dir2, 'joblib', 'K_3.pkl')))
    #
    # def test__re_label_clusters(self):
    #     labels_0 = [0] * 10
    #     labels_1_expected = [0] * 5 + [1] * 5
    #     labels_1_test_1 = [1] * 5 + [0] * 5
    #
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_1_test_1), labels_1_expected)
    #     labels_1_test_2 = [0] * 5 + [1] * 5
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_1_test_2), labels_1_expected)
    #     labels_2_expected = [0] * 3 + [1] * 2 + [2] * 5
    #     labels_2_test_1 = [0] * 3 + [1] * 2 + [2] * 5
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_1), labels_2_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_1), labels_2_expected)
    #     labels_2_test_2 = [0] * 3 + [2] * 2 + [1] * 5
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_2), labels_2_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_2), labels_2_expected)
    #     labels_2_test_3 = [1] * 3 + [0] * 2 + [2] * 5
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_3), labels_2_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_3), labels_2_expected)
    #     labels_2_test_4 = [1] * 3 + [2] * 2 + [0] * 5
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_4), labels_2_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_4), labels_2_expected)
    #     labels_2_test_5 = [2] * 3 + [0] * 2 + [1] * 5
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_5), labels_2_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_5), labels_2_expected)
    #     labels_2_test_6 = [2] * 3 + [1] * 2 + [0] * 5
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_6), labels_2_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_6), labels_2_expected)
    #     labels_3_expected = [0] * 3 + [1] * 2 + [2] * 2 + [3] * 3
    #     labels_3_test_1 = [0] * 3 + [1] * 2 + [2] * 2 + [3] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_1), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_1), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_1), labels_3_expected)
    #     labels_3_test_2 = [0] * 3 + [1] * 2 + [3] * 2 + [2] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_2), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_2), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_2), labels_3_expected)
    #     labels_3_test_3 = [0] * 3 + [2] * 2 + [1] * 2 + [3] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_3), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_3), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_3), labels_3_expected)
    #     labels_3_test_4 = [0] * 3 + [2] * 2 + [3] * 2 + [1] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_4), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_4), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_4), labels_3_expected)
    #     labels_3_test_5 = [0] * 3 + [3] * 2 + [1] * 2 + [2] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_5), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_5), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_5), labels_3_expected)
    #     labels_3_test_6 = [0] * 3 + [3] * 2 + [2] * 2 + [1] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_6), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_6), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_6), labels_3_expected)
    #     labels_3_test_7 = [3] * 3 + [0] * 2 + [1] * 2 + [2] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_7), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_7), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_7), labels_3_expected)
    #     labels_3_test_8 = [3] * 3 + [0] * 2 + [2] * 2 + [1] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_8), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_8), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_8), labels_3_expected)
    #     labels_3_test_9 = [3] * 3 + [1] * 2 + [2] * 2 + [0] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_9), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_9), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_9), labels_3_expected)
    #     labels_3_test_10 = [3] * 3 + [1] * 2 + [0] * 2 + [2] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_10), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_10), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_10), labels_3_expected)
    #     labels_3_test_11 = [3] * 3 + [2] * 2 + [1] * 2 + [0] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_11), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_11), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_11), labels_3_expected)
    #     labels_3_test_12 = [3] * 3 + [2] * 2 + [0] * 2 + [1] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_12), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_12), labels_3_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_12), labels_3_expected)
    #     labels_4_expected = [0] * 3 + [1] * 2 + [2, 3] + [4] * 3
    #     labels_4_test_1 = [0] * 3 + [1] * 2 + [2, 3] + [4] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_1), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_1), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_1), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_1), labels_4_expected)
    #     labels_4_test_2 = [0] * 3 + [1] * 2 + [2, 4] + [3] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_2), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_2), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_2), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_2), labels_4_expected)
    #     labels_4_test_3 = [0] * 3 + [1] * 2 + [3, 2] + [4] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_3), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_3), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_3), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_3), labels_4_expected)
    #     labels_4_test_4 = [0] * 3 + [1] * 2 + [3, 4] + [2] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_4), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_4), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_4), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_4), labels_4_expected)
    #     labels_4_test_5 = [0] * 3 + [2] * 2 + [1, 3] + [4] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_5), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_5), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_5), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_5), labels_4_expected)
    #     labels_4_test_6 = [0] * 3 + [2] * 2 + [1, 4] + [3] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_6), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_6), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_6), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_6), labels_4_expected)
    #     labels_4_test_7 = [0] * 3 + [2] * 2 + [3, 1] + [4] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_7), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_7), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_7), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_7), labels_4_expected)
    #     labels_4_test_8 = [0] * 3 + [2] * 2 + [3, 4] + [1] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_8), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_8), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_8), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_8), labels_4_expected)
    #     labels_4_test_9 = [0] * 3 + [2] * 2 + [4, 1] + [3] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_9), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_9), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_9), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_9), labels_4_expected)
    #     labels_4_test_10 = [0] * 3 + [2] * 2 + [4, 3] + [1] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_10), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_10), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_10), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_10), labels_4_expected)
    #     labels_4_test_11 = [0] * 3 + [3] * 2 + [1, 2] + [4] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_11), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_11), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_11), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_11), labels_4_expected)
    #     labels_4_test_12 = [0] * 3 + [3] * 2 + [1, 4] + [2] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_12), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_12), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_12), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_12), labels_4_expected)
    #     labels_4_test_13 = [0] * 3 + [3] * 2 + [2, 1] + [4] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_13), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_13), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_13), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_13), labels_4_expected)
    #     labels_4_test_14 = [0] * 3 + [3] * 2 + [2, 4] + [1] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_14), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_14), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_14), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_14), labels_4_expected)
    #     labels_4_test_15 = [0] * 3 + [3] * 2 + [4, 1] + [2] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_15), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_15), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_15), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_15), labels_4_expected)
    #     labels_4_test_16 = [0] * 3 + [3] * 2 + [4, 2] + [1] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_16), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_16), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_16), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_16), labels_4_expected)
    #     labels_4_test_17 = [0] * 3 + [4] * 2 + [1, 2] + [3] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_17), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_17), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_17), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_17), labels_4_expected)
    #     labels_4_test_18 = [0] * 3 + [4] * 2 + [1, 3] + [2] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_18), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_18), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_18), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_18), labels_4_expected)
    #     labels_4_test_19 = [0] * 3 + [4] * 2 + [2, 1] + [3] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_19), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_19), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_19), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_19), labels_4_expected)
    #     labels_4_test_20 = [0] * 3 + [4] * 2 + [2, 3] + [1] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_20), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_20), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_20), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_20), labels_4_expected)
    #     labels_4_test_21 = [0] * 3 + [4] * 2 + [3, 1] + [2] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_21), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_21), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_21), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_21), labels_4_expected)
    #     labels_4_test_22 = [0] * 3 + [4] * 2 + [3, 2] + [1] * 3
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_22), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_22), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_22), labels_4_expected)
    #     self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_22), labels_4_expected)
