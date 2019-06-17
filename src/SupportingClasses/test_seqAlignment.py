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
from SeqAlignment import SeqAlignment
from DataSetGenerator import DataSetGenerator


class TestSeqAlignment(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.max_threads = cpu_count() - 2
        cls.max_target_seqs = 150
        cls.testing_dir = os.path.abspath('../Test/')
        cls.input_path = os.path.join(cls.testing_dir, 'Input')
        cls.protein_list_path = os.path.join(cls.input_path, 'ProteinLists')
        if not os.path.isdir(cls.protein_list_path):
            os.makedirs(cls.protein_list_path)
        cls.small_structure_id = '7hvp'
        cls.large_structure_id = '2zxe'
        cls.protein_list_fn = os.path.join(cls.protein_list_path, 'Test_Set.txt')
        structure_ids = [cls.small_structure_id, cls.large_structure_id]
        with open(cls.protein_list_fn, 'wb') as test_list_handle:
            for structure_id in structure_ids:
                test_list_handle.write('{}{}\n'.format(structure_id, 'A'))
        cls.data_set = DataSetGenerator(input_path=cls.input_path)
        cls.data_set.build_pdb_alignment_dataset(protein_list_fn='Test_Set.txt', num_threads=cls.max_threads,
                                                 max_target_seqs=cls.max_target_seqs)
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

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(cls.save_file_small)
        except OSError:
            pass
        try:
            os.remove(cls.save_file_large)
        except OSError:
            pass

    def tearDown(self):
        try:
            os.remove(self.aln_file_small)
        except OSError:
            pass
        try:
            os.remove(self.aln_file_large)
        except OSError:
            pass

    def test_init(self):
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

    def test_import_alignment(self):
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

    def test_write_out_alignment(self):
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

    def test_generate_sub_alignment(self):
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

    def test__subset_columns(self):
        aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
        with self.assertRaises(TypeError):
            aln_obj1._subset_columns([range(5) + range(745, 749)])
        aln_obj1.import_alignment()
        # One position
        aln_obj1_alpha = aln_obj1._subset_columns([0])
        self.assertEqual(len(aln_obj1_alpha), aln_obj1.size)
        for rec in aln_obj1_alpha:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[0])
        aln_obj1_beta = aln_obj1._subset_columns([aln_obj1.seq_length - 1])
        self.assertEqual(len(aln_obj1_beta), aln_obj1.size)
        for rec in aln_obj1_beta:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[aln_obj1.seq_length - 1])
        aln_obj1_gamma = aln_obj1._subset_columns([aln_obj1.seq_length // 2])
        self.assertEqual(len(aln_obj1_gamma), aln_obj1.size)
        for rec in aln_obj1_gamma:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[aln_obj1.seq_length // 2])
        # Single Range
        aln_obj1_delta = aln_obj1._subset_columns(range(5))
        self.assertEqual(len(aln_obj1_delta), aln_obj1.size)
        for rec in aln_obj1_delta:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[:5])
        aln_obj1_epsilon = aln_obj1._subset_columns(range(aln_obj1.seq_length - 5, aln_obj1.seq_length))
        self.assertEqual(len(aln_obj1_epsilon), aln_obj1.size)
        for rec in aln_obj1_epsilon:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[-5:])
        aln_obj1_zeta = aln_obj1._subset_columns(range(aln_obj1.seq_length // 2, aln_obj1.seq_length // 2 + 5))
        self.assertEqual(len(aln_obj1_zeta), aln_obj1.size)
        for rec in aln_obj1_zeta:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[aln_obj1.seq_length // 2:
                                                                       aln_obj1.seq_length // 2 + 5])
        # Mixed Range and Single Position
        aln_obj1_eta = aln_obj1._subset_columns([0] + range(aln_obj1.seq_length // 2, aln_obj1.seq_length // 2 + 5))
        self.assertEqual(len(aln_obj1_eta), aln_obj1.size)
        for rec in aln_obj1_eta:
            self.assertEqual(len(rec.seq), 6)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[0] +
                                 aln_obj1.query_sequence[aln_obj1.seq_length // 2: aln_obj1.seq_length // 2 + 5])
        aln_obj1_theta = aln_obj1._subset_columns(range(aln_obj1.seq_length // 2, aln_obj1.seq_length // 2 + 5) +
                                                  [aln_obj1.seq_length - 1])
        self.assertEqual(len(aln_obj1_theta), aln_obj1.size)
        for rec in aln_obj1_theta:
            self.assertEqual(len(rec.seq), 6)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[aln_obj1.seq_length // 2:
                                                                       aln_obj1.seq_length // 2 + 5] +
                                 aln_obj1.query_sequence[aln_obj1.seq_length - 1])
        aln_obj1_iota = aln_obj1._subset_columns(range(5) + [aln_obj1.seq_length // 2] +
                                                 range(aln_obj1.seq_length - 5, aln_obj1.seq_length))
        self.assertEqual(len(aln_obj1_iota), aln_obj1.size)
        for rec in aln_obj1_iota:
            self.assertEqual(len(rec.seq), 11)
            if rec.id == aln_obj1.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj1.query_sequence[:5] +
                                 aln_obj1.query_sequence[aln_obj1.seq_length // 2] +
                                 aln_obj1.query_sequence[-5:])
        aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
        with self.assertRaises(TypeError):
            aln_obj2._subset_columns([range(5) + range(1501, 1506)])
        aln_obj2.import_alignment()
        # One position
        aln_obj2_alpha = aln_obj2._subset_columns([0])
        self.assertEqual(len(aln_obj2_alpha), aln_obj2.size)
        for rec in aln_obj2_alpha:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[0])
        aln_obj2_beta = aln_obj2._subset_columns([aln_obj2.seq_length - 1])
        self.assertEqual(len(aln_obj2_beta), aln_obj2.size)
        for rec in aln_obj2_beta:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[aln_obj2.seq_length - 1])
        aln_obj2_gamma = aln_obj2._subset_columns([aln_obj2.seq_length // 2])
        self.assertEqual(len(aln_obj2_gamma), aln_obj2.size)
        for rec in aln_obj2_gamma:
            self.assertEqual(len(rec.seq), 1)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[aln_obj2.seq_length // 2])
        # Single Range
        aln_obj2_delta = aln_obj2._subset_columns(range(5))
        self.assertEqual(len(aln_obj2_delta), aln_obj2.size)
        for rec in aln_obj2_delta:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[0:5])
        aln_obj2_epsilon = aln_obj2._subset_columns(range(aln_obj2.seq_length - 5, aln_obj2.seq_length))
        self.assertEqual(len(aln_obj2_epsilon), aln_obj2.size)
        for rec in aln_obj2_epsilon:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[-5:])
        aln_obj2_zeta = aln_obj2._subset_columns(range(aln_obj2.seq_length // 2, aln_obj2.seq_length // 2 + 5))
        self.assertEqual(len(aln_obj2_zeta), aln_obj2.size)
        for rec in aln_obj2_zeta:
            self.assertEqual(len(rec.seq), 5)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[aln_obj2.seq_length // 2:
                                                                       aln_obj2.seq_length // 2 + 5])
        # Mixed Range and Single Position
        aln_obj2_eta = aln_obj2._subset_columns([0] + range(aln_obj2.seq_length // 2, aln_obj2.seq_length // 2 + 5))
        self.assertEqual(len(aln_obj2_eta), aln_obj2.size)
        for rec in aln_obj2_eta:
            self.assertEqual(len(rec.seq), 6)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[0] +
                                 aln_obj2.query_sequence[aln_obj2.seq_length // 2: aln_obj2.seq_length // 2 + 5])
        aln_obj2_theta = aln_obj2._subset_columns(range(aln_obj2.seq_length // 2, aln_obj2.seq_length // 2 + 5) +
                                                  [aln_obj2.seq_length - 1])
        self.assertEqual(len(aln_obj2_theta), aln_obj2.size)
        for rec in aln_obj2_theta:
            self.assertEqual(len(rec.seq), 6)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[aln_obj2.seq_length // 2:
                                                                       aln_obj2.seq_length // 2 + 5] +
                                 aln_obj2.query_sequence[aln_obj2.seq_length - 1])
        aln_obj2_iota = aln_obj2._subset_columns(range(5) + [aln_obj2.seq_length // 2] +
                                                 range(aln_obj2.seq_length - 5, aln_obj2.seq_length))
        self.assertEqual(len(aln_obj2_iota), aln_obj2.size)
        for rec in aln_obj2_iota:
            self.assertEqual(len(rec.seq), 11)
            if rec.id == aln_obj2.query_id[1:]:
                self.assertEqual(str(rec.seq), aln_obj2.query_sequence[:5] +
                                 aln_obj2.query_sequence[aln_obj2.seq_length // 2] +
                                 aln_obj2.query_sequence[-5:])

    def test_remove_gaps(self):
        aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
        with self.assertRaises(TypeError):
            aln_obj1.remove_gaps()
        aln_obj1.import_alignment()
        aln_obj1_prime = SeqAlignment(self.aln_fn1, self.query1)
        aln_obj1_prime.import_alignment()
        aln_obj1_prime.remove_gaps()
        self.assertEqual(aln_obj1.file_name, aln_obj1_prime.file_name)
        self.assertEqual(aln_obj1.query_id, aln_obj1_prime.query_id)
        for i in range(aln_obj1.size):
            self.assertEqual(aln_obj1.alignment[i].seq, aln_obj1_prime.alignment[i].seq)
        self.assertEqual(aln_obj1.seq_order, aln_obj1_prime.seq_order)
        self.assertEqual(aln_obj1.query_sequence, aln_obj1_prime.query_sequence)
        self.assertEqual(aln_obj1.seq_length, aln_obj1_prime.seq_length)
        self.assertEqual(aln_obj1.size, aln_obj1_prime.size)
        self.assertEqual(aln_obj1.distance_matrix, aln_obj1_prime.distance_matrix)
        self.assertEqual(aln_obj1.tree_order, aln_obj1_prime.tree_order)
        self.assertEqual(aln_obj1.sequence_assignments, aln_obj1_prime.sequence_assignments)
        aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
        with self.assertRaises(TypeError):
            aln_obj2.remove_gaps()
        aln_obj2.import_alignment()
        aln_obj2_prime = SeqAlignment(self.aln_fn2, self.query2)
        aln_obj2_prime.import_alignment()
        aln_obj2_prime.remove_gaps()
        self.assertEqual(aln_obj2.file_name, aln_obj2_prime.file_name)
        self.assertEqual(aln_obj2.query_id, aln_obj2_prime.query_id)
        for i in range(aln_obj2.size):
            self.assertEqual(len(aln_obj2_prime.alignment[i].seq), 368)
        self.assertEqual(aln_obj2.seq_order, aln_obj2_prime.seq_order)
        self.assertEqual(aln_obj2_prime.query_sequence, self.nongap_sequence2)
        self.assertEqual(aln_obj2_prime.seq_length, 368)
        self.assertEqual(aln_obj2.size, aln_obj2_prime.size)
        self.assertEqual(aln_obj2.distance_matrix, aln_obj2_prime.distance_matrix)
        self.assertEqual(aln_obj2.tree_order, aln_obj2_prime.tree_order)
        self.assertEqual(aln_obj2.sequence_assignments, aln_obj2_prime.sequence_assignments)

    def test_compute_distance_matrix(self):
        # This is not a very good test, should come up with something else in the future, maybe compute the identity of
        # sequences separately and compare them here.
        aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
        with self.assertRaises(TypeError):
            aln_obj1.compute_distance_matrix(model='identity', save_dir=self.save_dir1)
        aln_obj1.import_alignment()
        # Compute distance matrix manually
        # aln_obj1_num_mat = aln_obj1._alignment_to_num(aa_dict=self.aa_dict)
        aln_obj1_num_mat = aln_obj1._alignment_to_num()
        value_matrix = np.zeros([aln_obj1.size, aln_obj1.size])
        for i in range(aln_obj1.size):
            check = aln_obj1_num_mat - aln_obj1_num_mat[i]
            value_matrix[i] = np.sum(check == 0, axis=1)
        value_matrix /= aln_obj1.seq_length
        value_matrix = 1 - value_matrix
        # Compute distance matrix using class method
        aln_obj1.compute_distance_matrix(model='identity', save_dir=self.save_dir1)
        dist_mat1 = np.array(aln_obj1.distance_matrix)
        self.assertEqual(0, np.sum(dist_mat1[range(aln_obj1.size), range(aln_obj1.size)]))
        self.assertEqual(0, np.sum(value_matrix - dist_mat1))
        aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
        with self.assertRaises(TypeError):
            aln_obj2.compute_distance_matrix(model='identity', save_dir=self.save_dir2)
        aln_obj2.import_alignment()
        # Compute distance matrix manually
        # aln_obj2_num_mat = aln_obj2._alignment_to_num(aa_dict=self.aa_dict)
        aln_obj2_num_mat = aln_obj2._alignment_to_num()
        value_matrix = np.zeros([aln_obj2.size, aln_obj2.size])
        for i in range(aln_obj2.size):
            check = aln_obj2_num_mat - aln_obj2_num_mat[i]
            value_matrix[i] = np.sum(check == 0, axis=1)
        value_matrix /= aln_obj2.seq_length
        value_matrix = 1 - value_matrix
        # Compute distance matrix using class method
        aln_obj2.compute_distance_matrix(model='identity', save_dir=self.save_dir2)
        dist_mat2 = np.array(aln_obj2.distance_matrix)
        self.assertEqual(0, np.sum(dist_mat2[range(aln_obj2.size), range(aln_obj2.size)]))
        self.assertEqual(0, np.sum(value_matrix - dist_mat2))

    def test_compute_effective_alignment_size(self):
        aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
        with self.assertRaises(TypeError):
            aln_obj1.compute_effective_alignment_size()
        aln_obj1.import_alignment()
        aln_obj1.compute_distance_matrix(model='identity', save_dir=self.save_dir1)
        identity_mat = 1 - np.array(aln_obj1.distance_matrix)
        effective_size = 0.0
        for i in range(aln_obj1.size):
            n_i = 0.0
            for j in range(aln_obj1.size):
                if identity_mat[i, j] >= 0.62:
                    n_i += 1.0
            effective_size += 1.0 / n_i
        self.assertLess(abs(aln_obj1.compute_effective_alignment_size() - effective_size), 1.0e-12)
        aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
        with self.assertRaises(TypeError):
            aln_obj2.compute_effective_alignment_size()
        aln_obj2.import_alignment()
        aln_obj2.compute_distance_matrix(model='identity', save_dir=self.save_dir2)
        identity_mat = 1 - np.array(aln_obj2.distance_matrix)
        effective_size = 0.0
        for i in range(aln_obj2.size):
            n_i = 0.0
            for j in range(aln_obj2.size):
                if identity_mat[i, j] >= 0.62:
                    n_i += 1.0
            effective_size += 1.0 / n_i
        self.assertLess(abs(aln_obj2.compute_effective_alignment_size() - effective_size), 1.0e-12)

    def test__agglomerative_clustering(self):
        aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
        with self.assertRaises(ValueError):
            aln_obj1._agglomerative_clustering(n_cluster=2)
        with self.assertRaises(TypeError):
            aln_obj1._agglomerative_clustering(n_cluster=2, model='identity')
        aln_obj1.import_alignment()
        aln_obj1.compute_distance_matrix(model='identity', save_dir=self.save_dir1)
        aln_obj1_clusters1 = aln_obj1._agglomerative_clustering(n_cluster=2)
        self.assertEqual(len(set(aln_obj1_clusters1)), 2)
        self.assertTrue(0 in set(aln_obj1_clusters1))
        self.assertTrue(1 in set(aln_obj1_clusters1))
        self.assertFalse(os.path.isdir(os.path.join(os.getcwd(), 'joblib')))
        aln_obj1_clusters2 = aln_obj1._agglomerative_clustering(n_cluster=2, cache_dir=self.save_dir1)
        self.assertEqual(len(set(aln_obj1_clusters2)), 2)
        self.assertTrue(0 in set(aln_obj1_clusters2))
        self.assertTrue(1 in set(aln_obj1_clusters2))
        self.assertTrue(os.path.isdir(os.path.join(self.save_dir1, 'joblib')))
        self.assertEqual(aln_obj1_clusters1, aln_obj1_clusters2)
        aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
        with self.assertRaises(ValueError):
            aln_obj2._agglomerative_clustering(n_cluster=2)
        with self.assertRaises(TypeError):
            aln_obj2._agglomerative_clustering(n_cluster=2, model='identity')
        aln_obj2.import_alignment()
        aln_obj2.compute_distance_matrix(model='identity', save_dir=self.save_dir2)
        aln_obj2_clusters1 = aln_obj2._agglomerative_clustering(n_cluster=2)
        self.assertEqual(len(set(aln_obj2_clusters1)), 2)
        self.assertTrue(0 in set(aln_obj2_clusters1))
        self.assertTrue(1 in set(aln_obj2_clusters1))
        self.assertFalse(os.path.isdir(os.path.join(os.getcwd(), 'joblib')))
        aln_obj2_clusters2 = aln_obj2._agglomerative_clustering(n_cluster=2, cache_dir=self.save_dir2)
        self.assertEqual(len(set(aln_obj2_clusters2)), 2)
        self.assertTrue(0 in set(aln_obj2_clusters2))
        self.assertTrue(1 in set(aln_obj2_clusters2))
        self.assertTrue(os.path.isdir(os.path.join(self.save_dir2, 'joblib')))
        self.assertEqual(aln_obj2_clusters1, aln_obj2_clusters2)

    def test__upgma_tree(self):
        aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
        with self.assertRaises(ValueError):
            aln_obj1._upgma_tree(n_cluster=2, cache_dir=self.save_dir1)
        with self.assertRaises(TypeError):
            aln_obj1._upgma_tree(n_cluster=2, model='identity', cache_dir=self.save_dir1)
        aln_obj1.import_alignment()
        aln_obj1.compute_distance_matrix(model='identity', save_dir=self.save_dir1)
        aln_obj1_clusters1 = aln_obj1._upgma_tree(n_cluster=2, cache_dir=self.save_dir1)
        self.assertEqual(len(set(aln_obj1_clusters1)), 2)
        self.assertTrue(0 in set(aln_obj1_clusters1))
        self.assertTrue(1 in set(aln_obj1_clusters1))
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir1, 'serialized_aln_upgma.pkl')))
        aln_obj1_clusters2 = aln_obj1._upgma_tree(n_cluster=2, cache_dir=self.save_dir1)
        self.assertEqual(len(set(aln_obj1_clusters2)), 2)
        self.assertTrue(0 in set(aln_obj1_clusters2))
        self.assertTrue(1 in set(aln_obj1_clusters2))
        self.assertEqual(aln_obj1_clusters1, aln_obj1_clusters2)
        aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
        with self.assertRaises(ValueError):
            aln_obj2._upgma_tree(n_cluster=2, cache_dir=self.save_dir2)
        with self.assertRaises(TypeError):
            aln_obj2._upgma_tree(n_cluster=2, model='identity', cache_dir=self.save_dir2)
        aln_obj2.import_alignment()
        aln_obj2.compute_distance_matrix(model='identity', save_dir=self.save_dir2)
        aln_obj2_clusters1 = aln_obj2._upgma_tree(n_cluster=2, cache_dir=self.save_dir2)
        self.assertEqual(len(set(aln_obj2_clusters1)), 2)
        self.assertTrue(0 in set(aln_obj2_clusters1))
        self.assertTrue(1 in set(aln_obj2_clusters1))
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir2, 'serialized_aln_upgma.pkl')))
        aln_obj2_clusters2 = aln_obj2._upgma_tree(n_cluster=2, cache_dir=self.save_dir2)
        self.assertEqual(len(set(aln_obj2_clusters2)), 2)
        self.assertTrue(0 in set(aln_obj2_clusters2))
        self.assertTrue(1 in set(aln_obj2_clusters2))
        self.assertEqual(aln_obj2_clusters1, aln_obj2_clusters2)

    def test_random_assignment(self):
        aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
        with self.assertRaises(TypeError):
            aln_obj1._random_assignment(n_cluster=2)
        aln_obj1.import_alignment()
        aln_obj1_clusters1 = aln_obj1._random_assignment(n_cluster=2)
        self.assertEqual(len(set(aln_obj1_clusters1)), 2)
        self.assertTrue(0 in set(aln_obj1_clusters1))
        self.assertTrue(1 in set(aln_obj1_clusters1))
        aln_obj1_clusters2 = aln_obj1._random_assignment(n_cluster=2)
        self.assertEqual(len(set(aln_obj1_clusters2)), 2)
        self.assertTrue(0 in set(aln_obj1_clusters2))
        self.assertTrue(1 in set(aln_obj1_clusters2))
        self.assertNotEqual(aln_obj1_clusters1, aln_obj1_clusters2)
        aln_obj1_clusters3 = aln_obj1._random_assignment(n_cluster=3, cache_dir=self.save_dir1)
        self.assertEqual(len(set(aln_obj1_clusters3)), 3)
        self.assertTrue(0 in set(aln_obj1_clusters3))
        self.assertTrue(1 in set(aln_obj1_clusters3))
        self.assertTrue(2 in set(aln_obj1_clusters3))
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir1, 'joblib', 'K_3.pkl')))
        aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        aln_obj2_clusters1 = aln_obj2._random_assignment(n_cluster=2)
        self.assertEqual(len(set(aln_obj2_clusters1)), 2)
        self.assertTrue(0 in set(aln_obj2_clusters1))
        self.assertTrue(1 in set(aln_obj2_clusters1))
        aln_obj2_clusters2 = aln_obj2._random_assignment(n_cluster=2)
        self.assertEqual(len(set(aln_obj2_clusters2)), 2)
        self.assertTrue(0 in set(aln_obj2_clusters2))
        self.assertTrue(1 in set(aln_obj2_clusters2))
        self.assertNotEqual(aln_obj2_clusters1, aln_obj2_clusters2)
        aln_obj2_clusters3 = aln_obj2._random_assignment(n_cluster=3, cache_dir=self.save_dir2)
        self.assertEqual(len(set(aln_obj2_clusters3)), 3)
        self.assertTrue(0 in set(aln_obj2_clusters3))
        self.assertTrue(1 in set(aln_obj2_clusters3))
        self.assertTrue(2 in set(aln_obj2_clusters3))
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir2, 'joblib', 'K_3.pkl')))

    def test__re_label_clusters(self):
        labels_0 = [0] * 10
        labels_1_expected = [0] * 5 + [1] * 5
        labels_1_test_1 = [1] * 5 + [0] * 5

        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_1_test_1), labels_1_expected)
        labels_1_test_2 = [0] * 5 + [1] * 5
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_1_test_2), labels_1_expected)
        labels_2_expected = [0] * 3 + [1] * 2 + [2] * 5
        labels_2_test_1 = [0] * 3 + [1] * 2 + [2] * 5
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_1), labels_2_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_1), labels_2_expected)
        labels_2_test_2 = [0] * 3 + [2] * 2 + [1] * 5
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_2), labels_2_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_2), labels_2_expected)
        labels_2_test_3 = [1] * 3 + [0] * 2 + [2] * 5
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_3), labels_2_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_3), labels_2_expected)
        labels_2_test_4 = [1] * 3 + [2] * 2 + [0] * 5
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_4), labels_2_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_4), labels_2_expected)
        labels_2_test_5 = [2] * 3 + [0] * 2 + [1] * 5
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_5), labels_2_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_5), labels_2_expected)
        labels_2_test_6 = [2] * 3 + [1] * 2 + [0] * 5
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_2_test_6), labels_2_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_2_test_6), labels_2_expected)
        labels_3_expected = [0] * 3 + [1] * 2 + [2] * 2 + [3] * 3
        labels_3_test_1 = [0] * 3 + [1] * 2 + [2] * 2 + [3] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_1), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_1), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_1), labels_3_expected)
        labels_3_test_2 = [0] * 3 + [1] * 2 + [3] * 2 + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_2), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_2), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_2), labels_3_expected)
        labels_3_test_3 = [0] * 3 + [2] * 2 + [1] * 2 + [3] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_3), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_3), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_3), labels_3_expected)
        labels_3_test_4 = [0] * 3 + [2] * 2 + [3] * 2 + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_4), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_4), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_4), labels_3_expected)
        labels_3_test_5 = [0] * 3 + [3] * 2 + [1] * 2 + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_5), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_5), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_5), labels_3_expected)
        labels_3_test_6 = [0] * 3 + [3] * 2 + [2] * 2 + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_6), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_6), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_6), labels_3_expected)
        labels_3_test_7 = [3] * 3 + [0] * 2 + [1] * 2 + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_7), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_7), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_7), labels_3_expected)
        labels_3_test_8 = [3] * 3 + [0] * 2 + [2] * 2 + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_8), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_8), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_8), labels_3_expected)
        labels_3_test_9 = [3] * 3 + [1] * 2 + [2] * 2 + [0] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_9), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_9), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_9), labels_3_expected)
        labels_3_test_10 = [3] * 3 + [1] * 2 + [0] * 2 + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_10), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_10), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_10), labels_3_expected)
        labels_3_test_11 = [3] * 3 + [2] * 2 + [1] * 2 + [0] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_11), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_11), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_11), labels_3_expected)
        labels_3_test_12 = [3] * 3 + [2] * 2 + [0] * 2 + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_3_test_12), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_3_test_12), labels_3_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_3_test_12), labels_3_expected)
        labels_4_expected = [0] * 3 + [1] * 2 + [2, 3] + [4] * 3
        labels_4_test_1 = [0] * 3 + [1] * 2 + [2, 3] + [4] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_1), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_1), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_1), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_1), labels_4_expected)
        labels_4_test_2 = [0] * 3 + [1] * 2 + [2, 4] + [3] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_2), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_2), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_2), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_2), labels_4_expected)
        labels_4_test_3 = [0] * 3 + [1] * 2 + [3, 2] + [4] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_3), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_3), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_3), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_3), labels_4_expected)
        labels_4_test_4 = [0] * 3 + [1] * 2 + [3, 4] + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_4), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_4), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_4), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_4), labels_4_expected)
        labels_4_test_5 = [0] * 3 + [2] * 2 + [1, 3] + [4] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_5), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_5), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_5), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_5), labels_4_expected)
        labels_4_test_6 = [0] * 3 + [2] * 2 + [1, 4] + [3] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_6), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_6), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_6), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_6), labels_4_expected)
        labels_4_test_7 = [0] * 3 + [2] * 2 + [3, 1] + [4] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_7), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_7), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_7), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_7), labels_4_expected)
        labels_4_test_8 = [0] * 3 + [2] * 2 + [3, 4] + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_8), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_8), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_8), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_8), labels_4_expected)
        labels_4_test_9 = [0] * 3 + [2] * 2 + [4, 1] + [3] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_9), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_9), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_9), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_9), labels_4_expected)
        labels_4_test_10 = [0] * 3 + [2] * 2 + [4, 3] + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_10), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_10), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_10), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_10), labels_4_expected)
        labels_4_test_11 = [0] * 3 + [3] * 2 + [1, 2] + [4] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_11), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_11), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_11), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_11), labels_4_expected)
        labels_4_test_12 = [0] * 3 + [3] * 2 + [1, 4] + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_12), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_12), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_12), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_12), labels_4_expected)
        labels_4_test_13 = [0] * 3 + [3] * 2 + [2, 1] + [4] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_13), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_13), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_13), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_13), labels_4_expected)
        labels_4_test_14 = [0] * 3 + [3] * 2 + [2, 4] + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_14), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_14), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_14), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_14), labels_4_expected)
        labels_4_test_15 = [0] * 3 + [3] * 2 + [4, 1] + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_15), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_15), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_15), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_15), labels_4_expected)
        labels_4_test_16 = [0] * 3 + [3] * 2 + [4, 2] + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_16), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_16), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_16), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_16), labels_4_expected)
        labels_4_test_17 = [0] * 3 + [4] * 2 + [1, 2] + [3] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_17), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_17), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_17), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_17), labels_4_expected)
        labels_4_test_18 = [0] * 3 + [4] * 2 + [1, 3] + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_18), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_18), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_18), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_18), labels_4_expected)
        labels_4_test_19 = [0] * 3 + [4] * 2 + [2, 1] + [3] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_19), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_19), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_19), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_19), labels_4_expected)
        labels_4_test_20 = [0] * 3 + [4] * 2 + [2, 3] + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_20), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_20), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_20), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_20), labels_4_expected)
        labels_4_test_21 = [0] * 3 + [4] * 2 + [3, 1] + [2] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_21), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_21), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_21), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_21), labels_4_expected)
        labels_4_test_22 = [0] * 3 + [4] * 2 + [3, 2] + [1] * 3
        self.assertEqual(SeqAlignment._re_label_clusters(labels_0, labels_4_test_22), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_1_expected, labels_4_test_22), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_2_expected, labels_4_test_22), labels_4_expected)
        self.assertEqual(SeqAlignment._re_label_clusters(labels_3_expected, labels_4_test_22), labels_4_expected)

    def test_set_tree_ordering(self):

        def check(seq_dict, curr, prev=None):
            if prev is None:
                return True
            c_prev = 0
            c_curr = 0
            while (c_prev != (prev - 1)) and (c_curr != (curr - 1)):
                if not seq_dict[curr][c_curr].issubset(seq_dict[prev][c_prev]):
                    c_prev += 1
                if not seq_dict[curr][c_curr].issubset(seq_dict[prev][c_prev]):
                    return False
                c_curr += 1
            return True

        aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
        with self.assertRaises(TypeError):
            aln_obj1.set_tree_ordering()
        aln_obj1.import_alignment()
        aln_obj1.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir1,
                                   clustering_args={'affinity': 'euclidean', 'linkage': 'ward', 'model': 'identity'})
        self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
        self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
        self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
        for k in range(2, 5):
            self.assertTrue(check(aln_obj1.sequence_assignments, curr=k, prev=k-1))
        clusters = [1, 2, 3, 5, 7, 10, 25]
        aln_obj1.set_tree_ordering(tree_depth=clusters)
        self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
        self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
        self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
        for i in range(len(clusters)):
            self.assertTrue(check(aln_obj1.sequence_assignments, curr=clusters[i], prev=clusters[i - 1]),
                            'Error on i:{}, curr:{}, prev:{}'.format(i, clusters[i], clusters[i - 1]))
        aln_obj1.set_tree_ordering()
        self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
        self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
        self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
        for k in range(1, aln_obj1.size):
            self.assertTrue(check(aln_obj1.sequence_assignments, curr=k, prev=k - 1))
        aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        aln_obj2.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir2,
                                   clustering_args={'affinity': 'euclidean', 'linkage': 'ward', 'model': 'identity'})
        self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
        self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
        self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
        for k in range(2, 5):
            self.assertTrue(check(aln_obj2.sequence_assignments, curr=k, prev=k - 1))
        clusters = [1, 2, 3, 5, 7, 10, 25]
        aln_obj2.set_tree_ordering(tree_depth=clusters)
        self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
        self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
        self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
        for i in range(len(clusters)):
            self.assertTrue(check(aln_obj2.sequence_assignments, curr=clusters[i], prev=clusters[i - 1]))
        aln_obj2.set_tree_ordering()
        self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
        self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
        self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
        for k in range(1, aln_obj2.size):
            self.assertTrue(check(aln_obj2.sequence_assignments, curr=k, prev=k - 1))
        ################################################################################################################
        aln_obj1.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir1, clustering='random')
        self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
        # self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
        self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
        # for k in range(2, 5):
        #     self.assertTrue(check(aln_obj1.sequence_assignments, curr=k, prev=k - 1))
        clusters = [1, 2, 3, 5, 7, 10, 25]
        aln_obj1.set_tree_ordering(tree_depth=clusters, cache_dir=self.save_dir1, clustering='random')
        self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
        self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
        self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
        # for i in range(len(clusters)):
        #     self.assertTrue(check(aln_obj1.sequence_assignments, curr=clusters[i], prev=clusters[i - 1]),
        #                     'Error on i:{}, curr:{}, prev:{}'.format(i, clusters[i], clusters[i - 1]))
        aln_obj1.set_tree_ordering(clustering='random', cache_dir=self.save_dir1)
        self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
        self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
        self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
        # for k in range(1, aln_obj1.size):
        #     self.assertTrue(check(aln_obj1.sequence_assignments, curr=k, prev=k - 1))
        aln_obj2.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir2, clustering='random')
        self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
        self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
        self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
        # for k in range(2, 5):
        #     self.assertTrue(check(aln_obj2.sequence_assignments, curr=k, prev=k - 1))
        clusters = [1, 2, 3, 5, 7, 10, 25]
        aln_obj2.set_tree_ordering(tree_depth=clusters, cache_dir=self.save_dir2, clustering='random')
        self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
        # self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
        self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
        # for i in range(len(clusters)):
        #     self.assertTrue(check(aln_obj2.sequence_assignments, curr=clusters[i], prev=clusters[i - 1]))
        aln_obj2.set_tree_ordering(cache_dir=self.save_dir2, clustering='random')
        self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
        self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
        self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
        # for k in range(1, aln_obj2.size):
        #     self.assertTrue(check(aln_obj2.sequence_assignments, curr=k, prev=k - 1))
        ################################################################################################################
        aln_obj1.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir1, clustering='upgma')
        self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
        # self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
        self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
        for k in range(2, 5):
            self.assertTrue(check(aln_obj1.sequence_assignments, curr=k, prev=k - 1))
        clusters = [1, 2, 3, 5, 7, 10, 25]
        aln_obj1.set_tree_ordering(tree_depth=clusters, cache_dir=self.save_dir1, clustering='upgma')
        self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
        self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
        self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
        for i in range(len(clusters)):
            self.assertTrue(check(aln_obj1.sequence_assignments, curr=clusters[i], prev=clusters[i - 1]),
                            'Error on i:{}, curr:{}, prev:{}'.format(i, clusters[i], clusters[i - 1]))
        aln_obj1.set_tree_ordering(clustering='upgma', cache_dir=self.save_dir1)
        self.assertEqual(set(aln_obj1.seq_order), set(aln_obj1.tree_order))
        self.assertNotEqual(aln_obj1.seq_order, aln_obj1.tree_order)
        self.assertTrue(check(aln_obj1.sequence_assignments, curr=1))
        for k in range(1, aln_obj1.size):
            self.assertTrue(check(aln_obj1.sequence_assignments, curr=k, prev=k - 1))
        aln_obj2.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir2, clustering='upgma')
        self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
        self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
        self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
        for k in range(2, 5):
            self.assertTrue(check(aln_obj2.sequence_assignments, curr=k, prev=k - 1))
        clusters = [1, 2, 3, 5, 7, 10, 25]
        aln_obj2.set_tree_ordering(tree_depth=clusters, cache_dir=self.save_dir2, clustering='upgma')
        self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
        # self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
        self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
        for i in range(len(clusters)):
            self.assertTrue(check(aln_obj2.sequence_assignments, curr=clusters[i], prev=clusters[i - 1]))
        aln_obj2.set_tree_ordering(cache_dir=self.save_dir2, clustering='upgma')
        self.assertEqual(set(aln_obj2.seq_order), set(aln_obj2.tree_order))
        self.assertNotEqual(aln_obj2.seq_order, aln_obj2.tree_order)
        self.assertTrue(check(aln_obj2.sequence_assignments, curr=1))
        for k in range(1, aln_obj2.size):
            self.assertTrue(check(aln_obj2.sequence_assignments, curr=k, prev=k - 1))

    def test_visualize_tree(self):
        aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
        with self.assertRaises(TypeError):
            aln_obj1.set_tree_ordering()
        aln_obj1.import_alignment()
        with self.assertRaises(ValueError):
            aln_obj1.visualize_tree(out_dir=self.save_dir1)
        aln_obj1.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir1,
                                   clustering_args={'affinity': 'euclidean', 'linkage': 'ward', 'model': 'identity'})
        aln_obj1_df = aln_obj1.visualize_tree(out_dir=self.save_dir1)
        self.assertEqual(aln_obj1.tree_order, list(aln_obj1_df.index))
        for i in range(1, 5):
            for j in aln_obj1_df.index:
                self.assertIn(j,  aln_obj1.sequence_assignments[i][aln_obj1_df.loc[j, i]])
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir1, 'query_1c17A_Sequence_Assignment.csv')))
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir1, 'query_1c17A_Sequence_Assignment.eps')))
        aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        with self.assertRaises(ValueError):
            aln_obj2.visualize_tree(out_dir=self.save_dir2)
        aln_obj2.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir2,
                           clustering_args={'affinity': 'euclidean', 'linkage': 'ward', 'model': 'identity'})
        aln_obj2_df = aln_obj2.visualize_tree(out_dir=self.save_dir2)
        self.assertEqual(aln_obj2.tree_order, list(aln_obj2_df.index))
        for i in range(1, 5):
            for j in aln_obj2_df.index:
                self.assertIn(j,  aln_obj2.sequence_assignments[i][aln_obj2_df.loc[j, i]])
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir2, 'query_1h1vA_Sequence_Assignment.csv')))
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir2, 'query_1h1vA_Sequence_Assignment.eps')))

    def test_get_branch_cluster(self):
        aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
        with self.assertRaises(TypeError):
            aln_obj1.set_tree_ordering()
        aln_obj1.import_alignment()
        aln_obj1.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir1,
                                   clustering_args={'affinity': 'euclidean', 'linkage': 'ward', 'model': 'identity'})
        for k in aln_obj1.sequence_assignments:
            for c in aln_obj1.sequence_assignments[k]:
                aln_obj1_sub = aln_obj1.get_branch_cluster(k, c)
                aln_obj1_sub_prime = aln_obj1.generate_sub_alignment(aln_obj1.sequence_assignments[k][c])
                self.assertEqual(aln_obj1_sub.file_name, aln_obj1_sub_prime.file_name)
                self.assertEqual(aln_obj1_sub.query_id, aln_obj1_sub_prime.query_id)
                self.assertEqual(aln_obj1.query_sequence, aln_obj1_sub.query_sequence)
                self.assertEqual(aln_obj1_sub.distance_matrix, aln_obj1_sub_prime.distance_matrix)
                self.assertEqual(aln_obj1_sub.sequence_assignments, aln_obj1_sub_prime.sequence_assignments)
                self.assertEqual(aln_obj1_sub.size, aln_obj1_sub_prime.size)
                self.assertEqual(aln_obj1_sub.seq_order, aln_obj1_sub_prime.seq_order)
                self.assertEqual(aln_obj1_sub.tree_order, aln_obj1_sub_prime.tree_order)
        aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        aln_obj2.set_tree_ordering(tree_depth=(2, 5), cache_dir=self.save_dir2,
                                   clustering_args={'affinity': 'euclidean', 'linkage': 'ward', 'model': 'identity'})
        for k in aln_obj2.sequence_assignments:
            for c in aln_obj2.sequence_assignments[k]:
                aln_obj2_sub = aln_obj2.get_branch_cluster(k, c)
                aln_obj2_sub_prime = aln_obj2.generate_sub_alignment(aln_obj2.sequence_assignments[k][c])
                self.assertEqual(aln_obj2_sub.file_name, aln_obj2_sub_prime.file_name)
                self.assertEqual(aln_obj2_sub.query_id, aln_obj2_sub_prime.query_id)
                self.assertEqual(aln_obj2.query_sequence, aln_obj2_sub.query_sequence)
                self.assertEqual(aln_obj2_sub.distance_matrix, aln_obj2_sub_prime.distance_matrix)
                self.assertEqual(aln_obj2_sub.sequence_assignments, aln_obj2_sub_prime.sequence_assignments)
                self.assertEqual(aln_obj2_sub.size, aln_obj2_sub_prime.size)
                self.assertEqual(aln_obj2_sub.seq_order, aln_obj2_sub_prime.seq_order)
                self.assertEqual(aln_obj2_sub.tree_order, aln_obj2_sub_prime.tree_order)

    def test_generate_positional_sub_alignment(self):
        aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
        with self.assertRaises(TypeError):
            aln_obj1.set_tree_ordering()
        aln_obj1.import_alignment()
        for i in range(aln_obj1.size - 1):
            aln_obj1_sub = aln_obj1.generate_positional_sub_alignment(i, i + 1)
            self.assertEqual(aln_obj1.file_name, aln_obj1_sub.file_name)
            self.assertEqual(aln_obj1.query_id, aln_obj1_sub.query_id)
            self.assertEqual(aln_obj1_sub.query_sequence, aln_obj1.query_sequence[i] + aln_obj1.query_sequence[i + 1])
            self.assertIsNone(aln_obj1_sub.distance_matrix)
            self.assertIsNone(aln_obj1_sub.sequence_assignments)
            self.assertEqual(aln_obj1.size, aln_obj1_sub.size)
            self.assertEqual(aln_obj1.seq_order, aln_obj1_sub.seq_order)
            self.assertEqual(aln_obj1.tree_order, aln_obj1_sub.tree_order)
            for j in range(aln_obj1.size):
                self.assertEqual(aln_obj1_sub.alignment[j].seq,
                                 aln_obj1.alignment[j].seq[i] + aln_obj1.alignment[j].seq[i + 1])
        aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        for i in range(aln_obj2.size - 1):
            aln_obj2_sub = aln_obj2.generate_positional_sub_alignment(i, i + 1)
            self.assertEqual(aln_obj2.file_name, aln_obj2_sub.file_name)
            self.assertEqual(aln_obj2.query_id, aln_obj2_sub.query_id)
            self.assertEqual(aln_obj2_sub.query_sequence, aln_obj2.query_sequence[i] + aln_obj2.query_sequence[i + 1])
            self.assertIsNone(aln_obj2_sub.distance_matrix)
            self.assertIsNone(aln_obj2_sub.sequence_assignments)
            self.assertEqual(aln_obj2.size, aln_obj2_sub.size)
            self.assertEqual(aln_obj2.seq_order, aln_obj2_sub.seq_order)
            self.assertEqual(aln_obj2.tree_order, aln_obj2_sub.tree_order)
            for j in range(aln_obj2.size):
                self.assertEqual(aln_obj2_sub.alignment[j].seq,
                                 aln_obj2.alignment[j].seq[i] + aln_obj2.alignment[j].seq[i + 1])

    def test_determine_usable_positions(self):
        aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
        with self.assertRaises(TypeError):
            aln_obj1.set_tree_ordering()
        aln_obj1.import_alignment()
        pos1, evidence1 = aln_obj1.determine_usable_positions(ratio=0.5)
        usable_pos = []
        for i in range(aln_obj1.seq_length):
            count = 0
            for j in range(aln_obj1.size):
                if aln_obj1.alignment[j, i] != '-':
                    count += 1
            if count >= (aln_obj1.size / 2):
                usable_pos.append(i)
            self.assertEqual(evidence1[i], count)
        self.assertEqual(list(pos1), usable_pos)
        aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        pos2, evidence2 = aln_obj2.determine_usable_positions(ratio=0.5)
        usable_pos2 = []
        for i in range(aln_obj2.seq_length):
            count = 0
            for j in range(aln_obj2.size):
                if aln_obj2.alignment[j, i] != '-':
                    count += 1
            if count >= (aln_obj2.size / 2):
                usable_pos2.append(i)
            self.assertEqual(evidence2[i], count)
        self.assertEqual(list(pos2), usable_pos2)

    def test_identify_comparable_sequences(self):
        aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
        with self.assertRaises(TypeError):
            aln_obj1.set_tree_ordering()
        aln_obj1.import_alignment()
        for i in range(1, aln_obj1.seq_length):
            comp_tup = aln_obj1.identify_comparable_sequences(pos1=0, pos2=i)
            col1 = aln_obj1.alignment[:, 0]
            col1_sub = []
            col2 = aln_obj1.alignment[:, i]
            col2_sub = []
            indices = []
            count = 0
            for j in range(aln_obj1.size):
                if (col1[j] != '-') and (col2[j] != '-'):
                    col1_sub.append(col1[j])
                    col2_sub.append(col2[j])
                    indices.append(j)
                    count += 1
            self.assertEqual(list(comp_tup[0]), col1_sub)
            self.assertEqual(list(comp_tup[1]), col2_sub)
            self.assertEqual(list(comp_tup[2]), indices)
            self.assertEqual(comp_tup[3], count)
        aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        for i in range(1, aln_obj2.seq_length):
            comp_tup = aln_obj2.identify_comparable_sequences(pos1=0, pos2=i)
            col1 = aln_obj2.alignment[:, 0]
            col1_sub = []
            col2 = aln_obj2.alignment[:, i]
            col2_sub = []
            indices = []
            count = 0
            for j in range(aln_obj2.size):
                if (col1[j] != '-') and (col2[j] != '-'):
                    col1_sub.append(col1[j])
                    col2_sub.append(col2[j])
                    indices.append(j)
                    count += 1
            self.assertEqual(list(comp_tup[0]), col1_sub)
            self.assertEqual(list(comp_tup[1]), col2_sub)
            self.assertEqual(list(comp_tup[2]), indices)
            self.assertEqual(comp_tup[3], count)

    def test_alignment_to_num(self):
        aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
        with self.assertRaises(TypeError):
            aln_obj1.set_tree_ordering()
        aln_obj1.import_alignment()
        # aln_obj1_num = aln_obj1._alignment_to_num(aa_dict=self.aa_dict)
        aln_obj1_num = aln_obj1._alignment_to_num()
        for i in range(aln_obj1.size):
            for j in range(aln_obj1.seq_length):
                self.assertEqual(aln_obj1_num[i, j], self.aa_dict[aln_obj1.alignment[i, j]])
        aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        # aln_obj2_num = aln_obj2._alignment_to_num(aa_dict=self.aa_dict)
        aln_obj2_num = aln_obj2._alignment_to_num()
        for i in range(aln_obj2.size):
            for j in range(aln_obj2.seq_length):
                self.assertEqual(aln_obj2_num[i, j], self.aa_dict[aln_obj2.alignment[i, j]])

    def test_heatmap_plot(self):
        aln_obj1 = SeqAlignment(self.aln_fn1, self.query1)
        with self.assertRaises(TypeError):
            aln_obj1.set_tree_ordering()
        aln_obj1.import_alignment()
        df1a, _hm1a = aln_obj1.heatmap_plot(name='1c17A Alignment Visualization Seq Order', aa_dict=self.aa_dict,
                                            out_dir=self.save_dir1, save=True)
        for i in range(aln_obj1.size):
            self.assertEqual(df1a.index[i], aln_obj1.seq_order[i])
            for j in range(aln_obj1.seq_length):
                self.assertEqual(df1a.loc[aln_obj1.seq_order[i], '{}:{}'.format(j, aln_obj1.query_sequence[j])],
                                 self.aa_dict[aln_obj1.alignment[i, j]])
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir1,
                                                    '1c17A Alignment Visualization Seq Order.eps'.replace(' ', '_'))))
        aln_obj1.set_tree_ordering(tree_depth=range(2, 10, 2), cache_dir=self.save_dir1,
                                   clustering_args={'affinity': 'euclidean', 'linkage': 'ward', 'model': 'identity'})
        df1b, _hm1b = aln_obj1.heatmap_plot(name='1c17A Alignment Visualization Tree Order', aa_dict=self.aa_dict,
                                            out_dir=self.save_dir1, save=True)
        for i in range(aln_obj1.size):
            self.assertEqual(df1b.index[i], aln_obj1.tree_order[i])
            for j in range(aln_obj1.seq_length):
                self.assertEqual(df1b.loc[aln_obj1.tree_order[i], '{}:{}'.format(j, aln_obj1.query_sequence[j])],
                                 self.aa_dict[aln_obj1.alignment[aln_obj1.seq_order.index(aln_obj1.tree_order[i]), j]])
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir1,
                                                    '1c17A Alignment Visualization Tree Order.eps'.replace(' ', '_'))))
        aln_obj2 = SeqAlignment(self.aln_fn2, self.query2)
        with self.assertRaises(TypeError):
            aln_obj2._random_assignment(n_cluster=2)
        aln_obj2.import_alignment()
        df2a, _hm2a = aln_obj2.heatmap_plot(name='1h1vA Alignment Visualization Seq Order', aa_dict=self.aa_dict,
                                            out_dir=self.save_dir2, save=True)
        for i in range(aln_obj2.size):
            self.assertEqual(df2a.index[i], aln_obj2.seq_order[i])
            for j in range(aln_obj2.seq_length):
                self.assertEqual(df2a.loc[aln_obj2.seq_order[i], '{}:{}'.format(j, aln_obj2.query_sequence[j])],
                                 self.aa_dict[aln_obj2.alignment[i, j]])
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir2,
                                                    '1h1vA Alignment Visualization Seq Order.eps'.replace(' ', '_'))))
        aln_obj2.set_tree_ordering(tree_depth=range(2, 10, 2), cache_dir=self.save_dir2,
                                   clustering_args={'affinity': 'euclidean', 'linkage': 'ward', 'model': 'identity'})
        df2b, _hm2b = aln_obj2.heatmap_plot(name='1h1vA Alignment Visualization Tree Order', aa_dict=self.aa_dict,
                                            out_dir=self.save_dir2, save=True)
        for i in range(aln_obj2.size):
            self.assertEqual(df2b.index[i], aln_obj2.tree_order[i])
            for j in range(aln_obj2.seq_length):
                self.assertEqual(df2b.loc[aln_obj2.tree_order[i], '{}:{}'.format(j, aln_obj2.query_sequence[j])],
                                 self.aa_dict[aln_obj2.alignment[aln_obj2.seq_order.index(aln_obj2.tree_order[i]), j]])
        self.assertTrue(os.path.isfile(os.path.join(self.save_dir2,
                                                    '1h1vA Alignment Visualization Tree Order.eps'.replace(' ', '_'))))
