"""
Created on Nov 9, 2018

@author: daniel
"""
import os
import unittest
import numpy as np
from copy import deepcopy
from shutil import rmtree
from Bio.Seq import Seq
from Bio.Alphabet import  Gapped
from Bio.SeqRecord import  SeqRecord
from Bio.Align import MultipleSeqAlignment
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from test_Base import TestBase
from utils import build_mapping
from SeqAlignment import SeqAlignment
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from EvolutionaryTraceAlphabet import FullIUPACProtein, MultiPositionAlphabet


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
        cls.single_letter_size, _, cls.single_letter_mapping, cls.single_letter_reverse = build_mapping(
            alphabet=Gapped(cls.query_aln_fa_small.alphabet))
        cls.pair_letter_size, _, cls.pair_letter_mapping, cls.pair_letter_reverse = build_mapping(
            alphabet=MultiPositionAlphabet(alphabet=Gapped(cls.query_aln_fa_small.alphabet), size=2))
        cls.single_to_pair = {}
        for char in cls.pair_letter_mapping:
            key = (cls.single_letter_mapping[char[0]], cls.single_letter_mapping[char[1]])
            cls.single_to_pair[key] = cls.pair_letter_mapping[char]

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
        self.assertTrue(isinstance(aln_small.alphabet, FullIUPACProtein))

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
        self.assertTrue(isinstance(aln_large.alphabet, FullIUPACProtein))

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
            self.assertTrue(isinstance(aln_small.alphabet, FullIUPACProtein))
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
            self.assertTrue(isinstance(aln_large.alphabet, FullIUPACProtein))
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
        self.assertEqual(aln_small.query_id, aln_small_prime.query_id)
        self.assertEqual(aln_small.seq_order, aln_small_prime.seq_order)
        self.assertEqual(aln_small.query_sequence, aln_small_prime.query_sequence)
        self.assertEqual(aln_small.seq_length, aln_small_prime.seq_length)
        self.assertEqual(aln_small.size, aln_small_prime.size)
        self.assertEqual(aln_small.marked, aln_small_prime.marked)
        self.assertEqual(aln_small.polymer_type, aln_small_prime.polymer_type)
        self.assertTrue(isinstance(aln_small_prime.alphabet, type(aln_small.alphabet)))
        for i in range(aln_small.size):
            self.assertEqual(aln_small.alignment[i].id, aln_small_prime.alignment[i].id)
            self.assertEqual(aln_small.alignment[i].seq, aln_small_prime.alignment[i].seq)

    def test3b_write_out_alignment(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        with self.assertRaises(TypeError):
            aln_large.write_out_alignment(self.aln_file_large)
        aln_large.import_alignment()
        aln_large.write_out_alignment(self.aln_file_large)
        self.assertTrue(os.path.isfile(self.aln_file_large), 'Alignment written to correct file.')
        aln_large_prime = SeqAlignment(self.aln_file_large, self.large_structure_id)
        aln_large_prime.import_alignment()
        self.assertEqual(aln_large.query_id, aln_large_prime.query_id)
        self.assertEqual(aln_large.seq_order, aln_large_prime.seq_order)
        self.assertEqual(aln_large.query_sequence, aln_large_prime.query_sequence)
        self.assertEqual(aln_large.seq_length, aln_large_prime.seq_length)
        self.assertEqual(aln_large.size, aln_large_prime.size)
        self.assertEqual(aln_large.marked, aln_large_prime.marked)
        self.assertEqual(aln_large.polymer_type, aln_large_prime.polymer_type)
        self.assertTrue(isinstance(aln_large_prime.alphabet, type(aln_large.alphabet)))
        for i in range(aln_large.size):
            self.assertEqual(aln_large.alignment[i].id, aln_large_prime.alignment[i].id)
            self.assertEqual(aln_large.alignment[i].seq, aln_large_prime.alignment[i].seq)

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
            self.assertEqual(aln_small.file_name, sub_aln.file_name)
            self.assertEqual(aln_small.query_id, sub_aln.query_id)
            self.assertEqual(aln_small.query_sequence, sub_aln.query_sequence)
            self.assertEqual(aln_small.seq_length, sub_aln.seq_length)
            self.assertEqual(aln_small.polymer_type, sub_aln.polymer_type)
            self.assertTrue(isinstance(sub_aln.alphabet, type(aln_small.alphabet)))
            self.assertFalse(any(sub_aln.marked))
        self.assertEqual(aln_small_sub1.seq_order, aln_small_seqrecords1, 'seq_order imported correctly.')
        self.assertEqual(aln_small_sub1.size, aln_small_halved, 'size is correctly determined.')
        self.assertEqual(len(aln_small_sub1.marked), aln_small_halved)
        self.assertEqual(aln_small_sub2.seq_order, aln_small_seqrecords2, 'seq_order imported correctly.')
        self.assertEqual(aln_small_sub2.size, aln_small.size - aln_small_halved, 'size is correctly determined.')
        self.assertEqual(len(aln_small_sub2.marked), aln_small.size - aln_small_halved)
        for i in range(aln_small.size):
            if i < aln_small_halved:
                self.assertEqual(aln_small.alignment[i].id, aln_small_sub1.alignment[i].id)
                self.assertEqual(aln_small.alignment[i].seq, aln_small_sub1.alignment[i].seq)
            else:
                self.assertEqual(aln_small.alignment[i].id, aln_small_sub2.alignment[i - aln_small_halved].id)
                self.assertEqual(aln_small.alignment[i].seq, aln_small_sub2.alignment[i - aln_small_halved].seq)

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
            self.assertEqual(aln_large.file_name, sub_aln.file_name)
            self.assertEqual(aln_large.query_id, sub_aln.query_id)
            self.assertEqual(aln_large.query_sequence, sub_aln.query_sequence)
            self.assertEqual(aln_large.seq_length, sub_aln.seq_length)
            self.assertEqual(aln_large.polymer_type, sub_aln.polymer_type)
            self.assertTrue(isinstance(sub_aln.alphabet, type(aln_large.alphabet)))
            self.assertFalse(any(sub_aln.marked))
        self.assertEqual(aln_large_sub1.seq_order, aln_large_seqrecords1, 'seq_order imported correctly.')
        self.assertEqual(aln_large_sub1.size, aln_large_halved, 'size is correctly determined.')
        self.assertEqual(len(aln_large_sub1.marked), aln_large_halved)
        self.assertEqual(aln_large_sub2.seq_order, aln_large_seqrecords2, 'seq_order imported correctly.')
        self.assertEqual(aln_large_sub2.size, aln_large.size - aln_large_halved, 'size is correctly determined.')
        self.assertEqual(len(aln_large_sub2.marked), aln_large.size - aln_large_halved)
        for i in range(aln_large.size):
            if i < aln_large_halved:
                self.assertEqual(aln_large.alignment[i].id, aln_large_sub1.alignment[i].id)
                self.assertEqual(aln_large.alignment[i].seq, aln_large_sub1.alignment[i].seq)
            else:
                self.assertEqual(aln_large.alignment[i].id, aln_large_sub2.alignment[i - aln_large_halved].id)
                self.assertEqual(aln_large.alignment[i].seq, aln_large_sub2.alignment[i - aln_large_halved].seq)

    def test5a__subset_columns_one_position(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        with self.assertRaises(TypeError):
            aln_small._subset_columns([0, aln_small.seq_length // 2, aln_small.seq_length - 1])
        aln_small.import_alignment()
        # One position - Start
        aln_small_alpha = aln_small._subset_columns([0])
        self.assertEqual(len(aln_small_alpha), aln_small.size)
        # One position - End
        aln_small_beta = aln_small._subset_columns([aln_small.seq_length - 1])
        self.assertEqual(len(aln_small_beta), aln_small.size)
        # One position - Middle
        aln_small_gamma = aln_small._subset_columns([aln_small.seq_length // 2])
        self.assertEqual(len(aln_small_gamma), aln_small.size)
        for i in range(aln_small.size):
            self.assertEqual(aln_small.alignment[i].id, aln_small_alpha[i].id)
            self.assertEqual(aln_small.alignment[i].id, aln_small_beta[i].id)
            self.assertEqual(aln_small.alignment[i].id, aln_small_gamma[i].id)
            self.assertEqual(len(aln_small_alpha[i].seq), 1)
            self.assertEqual(len(aln_small_beta[i].seq), 1)
            self.assertEqual(len(aln_small_gamma[i].seq), 1)
            self.assertEqual(str(aln_small.alignment[i].seq)[0], str(aln_small_alpha[i].seq))
            self.assertEqual(str(aln_small.alignment[i].seq)[aln_small.seq_length - 1], str(aln_small_beta[i].seq))
            self.assertEqual(str(aln_small.alignment[i].seq)[aln_small.seq_length // 2], str(aln_small_gamma[i].seq))

    def test5b__subset_columns_single_range(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        with self.assertRaises(TypeError):
            aln_small._subset_columns([range(5) + range(aln_small.seq_length // 2, aln_small.seq_length // 2 + 5) +
                                       range(aln_small.seq_length - 5, aln_small.seq_length)])
        aln_small.import_alignment()
        # Single Range - Start
        delta_range = range(5)
        aln_small_delta = aln_small._subset_columns(delta_range)
        self.assertEqual(len(aln_small_delta), aln_small.size)
        # Single Range - End
        epsilon_range = range(aln_small.seq_length - 5, aln_small.seq_length)
        aln_small_epsilon = aln_small._subset_columns(epsilon_range)
        self.assertEqual(len(aln_small_epsilon), aln_small.size)
        # Single Range - Middle
        zeta_range = range(aln_small.seq_length // 2, aln_small.seq_length // 2 + 5)
        aln_small_zeta = aln_small._subset_columns(zeta_range)
        self.assertEqual(len(aln_small_zeta), aln_small.size)
        for i in range(aln_small.size):
            self.assertEqual(aln_small.alignment[i].id, aln_small_delta[i].id)
            self.assertEqual(aln_small.alignment[i].id, aln_small_epsilon[i].id)
            self.assertEqual(aln_small.alignment[i].id, aln_small_zeta[i].id)
            self.assertEqual(len(aln_small_delta[i].seq), 5)
            self.assertEqual(len(aln_small_epsilon[i].seq), 5)
            self.assertEqual(len(aln_small_zeta[i].seq), 5)
            expected_delta = subset_string(str(aln_small.alignment[i].seq), delta_range)
            self.assertEqual(str(aln_small_delta[i].seq), expected_delta)
            expected_epsilon = subset_string(str(aln_small.alignment[i].seq), epsilon_range)
            self.assertEqual(str(aln_small_epsilon[i].seq), expected_epsilon)
            expected_zeta = subset_string(str(aln_small.alignment[i].seq), zeta_range)
            self.assertEqual(str(aln_small_zeta[i].seq), expected_zeta)

    def test5c__subset_columns_range_and_position(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        with self.assertRaises(TypeError):
            aln_small._subset_columns([0] + range(aln_small.seq_length // 2, aln_small.seq_length // 2 + 5))
        aln_small.import_alignment()
        # Mixed Range and Single Position - Start and Middle
        eta_pos = [0] + list(range(aln_small.seq_length // 2, aln_small.seq_length // 2 + 5))
        aln_small_eta = aln_small._subset_columns(eta_pos)
        self.assertEqual(len(aln_small_eta), aln_small.size)
        # Mixed Range and Single Position - Middle and End
        theta_pos = list(range(aln_small.seq_length // 2, aln_small.seq_length // 2 + 5)) + [aln_small.seq_length - 1]
        aln_small_theta = aln_small._subset_columns(theta_pos)
        self.assertEqual(len(aln_small_theta), aln_small.size)
        # Mixed Range and Single Position - Start, Middle, and End
        iota_pos = (list(range(5)) + [aln_small.seq_length // 2] +
                    list(range(aln_small.seq_length - 5, aln_small.seq_length)))
        aln_small_iota = aln_small._subset_columns(iota_pos)
        self.assertEqual(len(aln_small_iota), aln_small.size)
        for i in range(aln_small.size):
            self.assertEqual(aln_small.alignment[i].id, aln_small_eta[i].id)
            self.assertEqual(aln_small.alignment[i].id, aln_small_theta[i].id)
            self.assertEqual(aln_small.alignment[i].id, aln_small_iota[i].id)
            self.assertEqual(len(aln_small_eta[i].seq), 6)
            self.assertEqual(len(aln_small_theta[i].seq), 6)
            self.assertEqual(len(aln_small_iota[i].seq), 11)
            expected_eta = subset_string(str(aln_small.alignment[i].seq), eta_pos)
            self.assertEqual(str(aln_small_eta[i].seq), expected_eta)
            expected_theta = subset_string(str(aln_small.alignment[i].seq), theta_pos)
            self.assertEqual(str(aln_small_theta[i].seq), expected_theta)
            expected_iota = subset_string(str(aln_small.alignment[i].seq), iota_pos)
            self.assertEqual(str(aln_small_iota[i].seq), expected_iota)

    def test5d__subset_columns_one_position(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        with self.assertRaises(TypeError):
            aln_large._subset_columns([0])
        aln_large.import_alignment()
        # One position - Start
        aln_large_alpha = aln_large._subset_columns([0])
        self.assertEqual(len(aln_large_alpha), aln_large.size)
        # One position - End
        aln_large_beta = aln_large._subset_columns([aln_large.seq_length - 1])
        self.assertEqual(len(aln_large_beta), aln_large.size)
        # One position - Middle
        aln_large_gamma = aln_large._subset_columns([aln_large.seq_length // 2])
        self.assertEqual(len(aln_large_gamma), aln_large.size)
        for i in range(aln_large.size):
            self.assertEqual(aln_large.alignment[i].id, aln_large_alpha[i].id)
            self.assertEqual(aln_large.alignment[i].id, aln_large_beta[i].id)
            self.assertEqual(aln_large.alignment[i].id, aln_large_gamma[i].id)
            self.assertEqual(len(aln_large_alpha[i].seq), 1)
            self.assertEqual(len(aln_large_beta[i].seq), 1)
            self.assertEqual(len(aln_large_gamma[i].seq), 1)
            self.assertEqual(str(aln_large.alignment[i].seq)[0], str(aln_large_alpha[i].seq))
            self.assertEqual(str(aln_large.alignment[i].seq)[aln_large.seq_length - 1], str(aln_large_beta[i].seq))
            self.assertEqual(str(aln_large.alignment[i].seq)[aln_large.seq_length // 2], str(aln_large_gamma[i].seq))

    def test5e__subset_columns_single_range(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        with self.assertRaises(TypeError):
            aln_large._subset_columns(range(5) + range(aln_large.seq_length // 2, aln_large.seq_length // 2 + 5) +
                                      range(aln_large.seq_length - 5, aln_large.seq_length))
        aln_large.import_alignment()
        # Single Range - Start
        delta_range = range(5)
        aln_large_delta = aln_large._subset_columns(delta_range)
        self.assertEqual(len(aln_large_delta), aln_large.size)
        # Single Range - End
        epsilon_range = range(aln_large.seq_length - 5, aln_large.seq_length)
        aln_large_epsilon = aln_large._subset_columns(epsilon_range)
        self.assertEqual(len(aln_large_epsilon), aln_large.size)
        # Single Range - Middle
        zeta_range = range(aln_large.seq_length // 2, aln_large.seq_length // 2 + 5)
        aln_large_zeta = aln_large._subset_columns(zeta_range)
        self.assertEqual(len(aln_large_zeta), aln_large.size)
        for i in range(aln_large.size):
            self.assertEqual(aln_large.alignment[i].id, aln_large_delta[i].id)
            self.assertEqual(aln_large.alignment[i].id, aln_large_epsilon[i].id)
            self.assertEqual(aln_large.alignment[i].id, aln_large_zeta[i].id)
            self.assertEqual(len(aln_large_delta[i].seq), 5)
            self.assertEqual(len(aln_large_epsilon[i].seq), 5)
            self.assertEqual(len(aln_large_zeta[i].seq), 5)
            expected_delta = subset_string(str(aln_large.alignment[i].seq), delta_range)
            self.assertEqual(str(aln_large_delta[i].seq), expected_delta)
            expected_epsilon = subset_string(str(aln_large.alignment[i].seq), epsilon_range)
            self.assertEqual(str(aln_large_epsilon[i].seq), expected_epsilon)
            expected_zeta = subset_string(str(aln_large.alignment[i].seq), zeta_range)
            self.assertEqual(str(aln_large_zeta[i].seq), expected_zeta)

    def test5f__subset_columns_range_and_position(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        with self.assertRaises(TypeError):
            aln_large._subset_columns([0] + range(aln_large.seq_length // 2, aln_large.seq_length // 2 + 5))
        aln_large.import_alignment()
        # Mixed Range and Single Position - Start and Middle
        eta_pos = [0] + list(range(aln_large.seq_length // 2, aln_large.seq_length // 2 + 5))
        aln_large_eta = aln_large._subset_columns(eta_pos)
        self.assertEqual(len(aln_large_eta), aln_large.size)
        # Mixed Range and Single Position - Middle and End
        theta_pos = list(range(aln_large.seq_length // 2, aln_large.seq_length // 2 + 5)) + [aln_large.seq_length - 1]
        aln_large_theta = aln_large._subset_columns(theta_pos)
        self.assertEqual(len(aln_large_theta), aln_large.size)
        # Mixed Range and Single Position - Start, Middle, and End
        iota_pos = (list(range(5)) + [aln_large.seq_length // 2] +
                    list(range(aln_large.seq_length - 5, aln_large.seq_length)))
        aln_large_iota = aln_large._subset_columns(iota_pos)
        self.assertEqual(len(aln_large_iota), aln_large.size)
        for i in range(aln_large.size):
            self.assertEqual(aln_large.alignment[i].id, aln_large_eta[i].id)
            self.assertEqual(aln_large.alignment[i].id, aln_large_theta[i].id)
            self.assertEqual(aln_large.alignment[i].id, aln_large_iota[i].id)
            self.assertEqual(len(aln_large_eta[i].seq), 6)
            self.assertEqual(len(aln_large_theta[i].seq), 6)
            self.assertEqual(len(aln_large_iota[i].seq), 11)
            expected_eta = subset_string(str(aln_large.alignment[i].seq), eta_pos)
            self.assertEqual(str(aln_large_eta[i].seq), expected_eta)
            expected_theta = subset_string(str(aln_large.alignment[i].seq), theta_pos)
            self.assertEqual(str(aln_large_theta[i].seq), expected_theta)
            expected_iota = subset_string(str(aln_large.alignment[i].seq), iota_pos)
            self.assertEqual(str(aln_large_iota[i].seq), expected_iota)

    def test6a_remove_gaps(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        with self.assertRaises(TypeError):
            aln_small.remove_gaps()
        aln_small.import_alignment()
        aln_small_prime = aln_small.remove_gaps()
        ungapped_pos = [i for i, char in enumerate(aln_small.query_sequence) if char != '-']
        self.assertEqual(aln_small.query_id, aln_small_prime.query_id)
        self.assertEqual(aln_small.seq_order, aln_small_prime.seq_order)
        self.assertEqual(str(aln_small_prime.query_sequence), subset_string(aln_small.query_sequence, ungapped_pos))
        self.assertEqual(aln_small_prime.seq_length, len(aln_small_prime.query_sequence))
        self.assertEqual(aln_small.size, aln_small_prime.size)
        self.assertEqual(aln_small.marked, aln_small_prime.marked)
        self.assertEqual(aln_small.polymer_type, aln_small_prime.polymer_type)
        self.assertTrue(isinstance(aln_small_prime.alphabet, type(aln_small.alphabet)))
        for i in range(aln_small.size):
            self.assertEqual(aln_small.alignment[i].id, aln_small_prime.alignment[i].id)
            self.assertEqual(aln_small_prime.alignment[i].seq, subset_string(aln_small.alignment[i].seq, ungapped_pos))

    def test6b_remove_gaps(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        with self.assertRaises(TypeError):
            aln_large.remove_gaps()
        aln_large.import_alignment()
        aln_large_prime = aln_large.remove_gaps()
        ungapped_pos = [i for i, char in enumerate(aln_large.query_sequence) if char != '-']
        self.assertEqual(aln_large.query_id, aln_large_prime.query_id)
        self.assertEqual(aln_large.seq_order, aln_large_prime.seq_order)
        self.assertEqual(str(aln_large_prime.query_sequence), subset_string(aln_large.query_sequence, ungapped_pos))
        self.assertEqual(aln_large_prime.seq_length, len(aln_large_prime.query_sequence))
        self.assertEqual(aln_large.size, aln_large_prime.size)
        self.assertEqual(aln_large.marked, aln_large_prime.marked)
        self.assertEqual(aln_large.polymer_type, aln_large_prime.polymer_type)
        self.assertTrue(isinstance(aln_large_prime.alphabet, type(aln_large.alphabet)))
        for i in range(aln_large.size):
            self.assertEqual(aln_large.alignment[i].id, aln_large_prime.alignment[i].id)
            self.assertEqual(aln_large_prime.alignment[i].seq, subset_string(aln_large.alignment[i].seq, ungapped_pos))

    def test7a_remove_bad_sequences(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        with self.assertRaises(TypeError):
            aln_small.remove_bad_sequences()
        aln_small.import_alignment()
        aln_small.size += 2
        to_remove1 = SeqRecord(id='Test1', seq=Seq(str(aln_small.alignment[0].seq)[:-1] + 'U',
                                                   alphabet=FullIUPACProtein()))
        to_remove2 = SeqRecord(id='Test2', seq=Seq('U' * aln_small.seq_length, alphabet=FullIUPACProtein()))
        aln_small.alignment.append(to_remove1)
        aln_small.alignment.append(to_remove2)
        aln_small.seq_order += ['Test1', 'Test2']
        aln_small.marked += [False, False]
        self.assertEqual(aln_small.size, self.data_set.protein_data[self.small_structure_id]['Final_Count'] + 2)
        self.assertEqual(len(aln_small.seq_order), aln_small.size)
        self.assertEqual(len(aln_small.marked), aln_small.size)
        self.assertEqual(len(aln_small.alignment),
                         self.data_set.protein_data[self.small_structure_id]['Final_Count'] + 2)
        self.assertTrue('U' in aln_small.alignment[aln_small.size-2].seq)
        self.assertTrue('U' in aln_small.alignment[aln_small.size-1].seq)
        aln_small_prime = aln_small.remove_bad_sequences()
        self.assertEqual(aln_small_prime.query_id, aln_small.query_id)
        self.assertEqual(aln_small_prime.seq_order, aln_small.seq_order[:-2])
        self.assertEqual(aln_small_prime.query_sequence, aln_small.query_sequence)
        self.assertEqual(aln_small_prime.seq_length, aln_small.seq_length)
        self.assertEqual(aln_small_prime.size, self.data_set.protein_data[self.small_structure_id]['Final_Count'])
        self.assertEqual(aln_small_prime.marked, aln_small.marked[:-2])
        self.assertEqual(aln_small_prime.polymer_type, aln_small.polymer_type)
        self.assertTrue(isinstance(aln_small_prime.alphabet, type(aln_small.alphabet)))
        for i in range(aln_small_prime.size):
            self.assertEqual(aln_small.alignment[i].id, aln_small_prime.alignment[i].id)
            self.assertEqual(aln_small_prime.alignment[i].seq, aln_small.alignment[i].seq)

    def test7b_remove_bad_sequences(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        with self.assertRaises(TypeError):
            aln_large.remove_bad_sequences()
        aln_large.import_alignment()
        aln_large.size += 2
        to_remove1 = SeqRecord(id='Test1', seq=Seq(str(aln_large.alignment[0].seq)[:-1] + 'U',
                                                   alphabet=FullIUPACProtein()))
        to_remove2 = SeqRecord(id='Test2', seq=Seq('U' * aln_large.seq_length, alphabet=FullIUPACProtein()))
        aln_large.alignment.append(to_remove1)
        aln_large.alignment.append(to_remove2)
        aln_large.seq_order += ['Test1', 'Test2']
        aln_large.marked += [False, False]
        self.assertEqual(aln_large.size, self.data_set.protein_data[self.large_structure_id]['Final_Count'] + 2)
        self.assertEqual(len(aln_large.seq_order), aln_large.size)
        self.assertEqual(len(aln_large.marked), aln_large.size)
        self.assertEqual(len(aln_large.alignment),
                         self.data_set.protein_data[self.large_structure_id]['Final_Count'] + 2)
        self.assertTrue('U' in aln_large.alignment[aln_large.size - 2].seq)
        self.assertTrue('U' in aln_large.alignment[aln_large.size - 1].seq)
        aln_large_prime = aln_large.remove_bad_sequences()
        self.assertEqual(aln_large_prime.query_id, aln_large.query_id)
        self.assertEqual(aln_large_prime.seq_order, aln_large.seq_order[:-2])
        self.assertEqual(aln_large_prime.query_sequence, aln_large.query_sequence)
        self.assertEqual(aln_large_prime.seq_length, aln_large.seq_length)
        self.assertEqual(aln_large_prime.size, self.data_set.protein_data[self.large_structure_id]['Final_Count'])
        self.assertEqual(aln_large_prime.marked, aln_large.marked[:-2])
        self.assertEqual(aln_large_prime.polymer_type, aln_large.polymer_type)
        self.assertTrue(isinstance(aln_large_prime.alphabet, type(aln_large.alphabet)))
        for i in range(aln_large_prime.size):
            self.assertEqual(aln_large.alignment[i].id, aln_large_prime.alignment[i].id)
            self.assertEqual(aln_large_prime.alignment[i].seq, aln_large.alignment[i].seq)

    def test8a_generate_positional_sub_alignment(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        for i in range(aln_small.size - 1):
            aln_small_sub = aln_small.generate_positional_sub_alignment(i, i + 1)
            self.assertEqual(aln_small.file_name, aln_small_sub.file_name)
            self.assertEqual(aln_small.query_id, aln_small_sub.query_id)
            self.assertEqual(aln_small.seq_order, aln_small_sub.seq_order)
            self.assertEqual(str(aln_small_sub.query_sequence.seq),
                             aln_small.query_sequence[i] + aln_small.query_sequence[i + 1])
            self.assertEqual(aln_small_sub.seq_length, 2)
            self.assertEqual(aln_small.size, aln_small_sub.size)
            self.assertEqual(aln_small.marked, aln_small_sub.marked)
            self.assertEqual(aln_small.polymer_type, aln_small_sub.polymer_type)
            self.assertTrue(isinstance(aln_small_sub.alphabet, type(aln_small.alphabet)))
            for j in range(aln_small.size):
                self.assertEqual(str(aln_small_sub.alignment[j].seq),
                                 aln_small.alignment[j].seq[i] + aln_small.alignment[j].seq[i + 1])

    def test8b_generate_positional_sub_alignment(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        aln_large.import_alignment()
        for i in range(aln_large.size - 1):
            aln_large_sub = aln_large.generate_positional_sub_alignment(i, i + 1)
            self.assertEqual(aln_large.file_name, aln_large_sub.file_name)
            self.assertEqual(aln_large.query_id, aln_large_sub.query_id)
            self.assertEqual(aln_large.seq_order, aln_large_sub.seq_order)
            self.assertEqual(str(aln_large_sub.query_sequence.seq),
                             aln_large.query_sequence[i] + aln_large.query_sequence[i + 1])
            self.assertEqual(aln_large_sub.seq_length, 2)
            self.assertEqual(aln_large.size, aln_large_sub.size)
            self.assertEqual(aln_large.marked, aln_large_sub.marked)
            self.assertEqual(aln_large.polymer_type, aln_large_sub.polymer_type)
            self.assertTrue(isinstance(aln_large_sub.alphabet, type(aln_large.alphabet)))
            for j in range(aln_large.size):
                self.assertEqual(str(aln_large_sub.alignment[j].seq),
                                 aln_large.alignment[j].seq[i] + aln_large.alignment[j].seq[i + 1])

    def test9a_compute_effective_alignment_size(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        with self.assertRaises(TypeError):
            aln_small.compute_effective_alignment_size()
        aln_small.import_alignment()
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

    def test9b_compute_effective_alignment_size(self):
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

    def test10a_determine_usable_positions(self):
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

    def test10b_determine_usable_positions(self):
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

    def test11a_identify_comparable_sequences(self):
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

    def test11b_identify_comparable_sequences(self):
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

    def test12a_consensus_sequences(self):
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
                    if aa == '-':
                        pass
                    else:
                        best_aa = aa
            self.assertEqual(consensus.seq[i], best_aa)

    def test12b_consensus_sequences(self):
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
                    if aa == '-':
                        pass
                    else:
                        best_aa = aa
            self.assertEqual(consensus.seq[i], best_aa)

    def test13a__alignment_to_num(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        _, _, mapping, _ = build_mapping(alphabet=aln_small.alphabet)
        aln_obj1_num = aln_small._alignment_to_num(mapping=mapping)
        for i in range(aln_small.size):
            for j in range(aln_small.seq_length):
                self.assertEqual(aln_obj1_num[i, j], mapping[aln_small.alignment[i, j]])

    def test13b__alignment_to_num(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        aln_large.import_alignment()
        _, _, mapping, _ = build_mapping(alphabet=aln_large.alphabet)
        aln_obj2_num = aln_large._alignment_to_num(mapping=mapping)
        for i in range(aln_large.size):
            for j in range(aln_large.seq_length):
                self.assertEqual(aln_obj2_num[i, j], mapping[aln_large.alignment[i, j]])

    def test14a__gap_z_score_check(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        alpha_size, gap_chars, mapping, _ = build_mapping(alphabet=aln_small.alphabet)
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

    def test14b__gap_z_score_check(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        aln_large.import_alignment()
        alpha_size, gap_chars, mapping, _ = build_mapping(alphabet=aln_large.alphabet)
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

    def test15a__gap_percentile_check(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        consensus = aln_small.consensus_sequence()
        alpha_size, gap_chars, mapping, _ = build_mapping(alphabet=aln_small.alphabet)
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

    def test15b__gap_percentile_check(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.large_structure_id]['Final_FA_Aln'],
                                 query_id=self.large_structure_id)
        aln_large.import_alignment()
        consensus = aln_large.consensus_sequence()
        alpha_size, gap_chars, mapping, _ = build_mapping(alphabet=aln_large.alphabet)
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

    def test16a_gap_evaluation(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        kept, removed = aln_small.gap_evaluation(size_cutoff=15, z_score_cutoff=0.0, percentile_cutoff=0.15)
        alpha_size, gap_chars, mapping, _ = build_mapping(alphabet=aln_small.alphabet)
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

    def test16b_gap_evaluation(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_small.import_alignment()
        aln_small_sub = aln_small.generate_sub_alignment(aln_small.seq_order[:10])
        consensus = aln_small_sub.consensus_sequence()
        kept, removed = aln_small_sub.gap_evaluation(size_cutoff=15, z_score_cutoff=0.0, percentile_cutoff=0.15)
        alpha_size, gap_chars, _, _ = build_mapping(alphabet=aln_small.alphabet)
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

    def test16c_gap_evaluation(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_large.import_alignment()
        kept, removed = aln_large.gap_evaluation(size_cutoff=15, z_score_cutoff=0.0, percentile_cutoff=0.15)
        alpha_size, gap_chars, mapping, _ = build_mapping(alphabet=aln_large.alphabet)
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

    def test16d_gap_evaluation(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        aln_large.import_alignment()
        aln_large_sub = aln_large.generate_sub_alignment(aln_large.seq_order[:10])
        consensus = aln_large_sub.consensus_sequence()
        kept, removed = aln_large_sub.gap_evaluation(size_cutoff=15, z_score_cutoff=0.0, percentile_cutoff=0.15)
        alpha_size, gap_chars, _, _ = build_mapping(alphabet=aln_large.alphabet)
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

    def evaluate_heatmap_plot(self, aln, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        aln.import_alignment()
        _, _, mapping, _ = build_mapping(alphabet=aln.alphabet)
        name = '{} Alignment Visualization'.format(aln.query_id)
        fig = plt.figure(figsize=(0.25 * aln.seq_length + 0.7 + 0.7,
                                  0.25 * aln.size + 0.7 + 0.7))
        gs = GridSpec(nrows=1, ncols=1)
        plotting_ax = fig.add_subplot(gs[0, 0])
        expected_path = os.path.join(save_dir, name.replace(' ', '_') + '.eps')
        for save in [True, False]:
            for ax in [None, plotting_ax]:
                print('Plotting with save: {} and ax: {}'.format(save, ax))
                df, hm = aln.heatmap_plot(name=name, out_dir=save_dir, save=save, ax=ax)
                for i in range(aln.size):
                    self.assertEqual(df.index[i], aln.seq_order[i])
                    for j in range(aln.seq_length):
                        self.assertEqual(df.loc[aln.seq_order[i],
                                                '{}:{}'.format(j, aln.query_sequence[j])], mapping[aln.alignment[i, j]])
                if ax:
                    self.assertEqual(ax, hm)
                    ax.clear()
                else:
                    self.assertIsNotNone(hm)
                    self.assertNotEqual(ax, hm)
                if save:
                    self.assertTrue(os.path.isfile(expected_path))
                    os.remove(expected_path)
                else:
                    self.assertFalse(os.path.isfile(expected_path))

    def test17a_heatmap_plot(self):
        aln_small = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        self.evaluate_heatmap_plot(aln=aln_small, save_dir=self.save_dir_small)

    def test17b_heatmap_plot(self):
        aln_large = SeqAlignment(file_name=self.data_set.protein_data[self.small_structure_id]['Final_FA_Aln'],
                                 query_id=self.small_structure_id)
        self.evaluate_heatmap_plot(aln=aln_large, save_dir=self.save_dir_large)

    def test18a_characterize_positions(self):
        # Test single position only for the small sequence
        aln_small_sub = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[self.small_structure_id])
        single_pos = self.query_aln_fa_small.seq_length
        single_table, pair_table = aln_small_sub.characterize_positions(
            single=True, pair=False, single_size=self.single_letter_size, single_mapping=self.single_letter_mapping,
            single_reverse=self.single_letter_reverse, pair_size=self.pair_letter_size,
            pair_mapping=self.pair_letter_mapping, pair_reverse=self.pair_letter_reverse)
        single_table2, pair_table2 = aln_small_sub.characterize_positions2(
            single=True, pair=False, single_letter_size=self.single_letter_size,
            single_letter_mapping=self.single_letter_mapping, single_letter_reverse=self.single_letter_reverse,
            pair_letter_size=self.pair_letter_size, pair_letter_mapping=self.pair_letter_mapping,
            pair_letter_reverse=self.pair_letter_reverse, single_to_pair=self.single_to_pair)
        self.assertEqual(single_table.get_positions(), list(range(self.query_aln_fa_small.seq_length)))
        self.assertEqual(single_table2.get_positions(), list(range(self.query_aln_fa_small.seq_length)))
        table1 = single_table.get_table()
        position_sums = np.sum(table1, axis=1)
        self.assertFalse((position_sums > 1).any())
        self.assertEqual(np.sum(position_sums), single_pos)
        character_sums = np.sum(table1, axis=0)
        self.assertFalse((character_sums > self.query_aln_fa_small.seq_length).any())
        self.assertEqual(np.sum(character_sums), single_pos)
        table2 = single_table2.get_table()
        position_sums2 = np.sum(table2, axis=1)
        self.assertFalse((position_sums2 > 1).any())
        self.assertEqual(np.sum(position_sums2), single_pos)
        character_sums2 = np.sum(table2, axis=0)
        self.assertFalse((character_sums2 > single_pos).any())
        self.assertEqual(np.sum(character_sums2), single_pos)
        self.assertIsNone(pair_table)
        self.assertIsNone(pair_table2)

    def test18b_characterize_positions(self):
        # Test single position only for the large sequence
        aln_large_sub = self.query_aln_fa_large.generate_sub_alignment(sequence_ids=[self.large_structure_id])
        single_pos = aln_large_sub.seq_length
        single_table, pair_table = aln_large_sub.characterize_positions(
            single=True, pair=False, single_size=self.single_letter_size, single_mapping=self.single_letter_mapping,
            single_reverse=self.single_letter_reverse, pair_size=self.pair_letter_size,
            pair_mapping=self.pair_letter_mapping, pair_reverse=self.pair_letter_reverse)
        single_table2, pair_table2 = aln_large_sub.characterize_positions2(
            single=True, pair=False, single_letter_size=self.single_letter_size,
            single_letter_mapping=self.single_letter_mapping, single_letter_reverse=self.single_letter_reverse,
            pair_letter_size=self.pair_letter_size, pair_letter_mapping=self.pair_letter_mapping,
            pair_letter_reverse=self.pair_letter_reverse, single_to_pair=self.single_to_pair)
        self.assertEqual(single_table.get_positions(), list(range(self.query_aln_fa_large.seq_length)))
        self.assertEqual(single_table2.get_positions(), list(range(self.query_aln_fa_large.seq_length)))
        table1 = single_table.get_table()
        position_sums = np.sum(table1, axis=1)
        self.assertFalse((position_sums > 1).any())
        self.assertEqual(np.sum(position_sums), single_pos)
        character_sums = np.sum(table1, axis=0)
        self.assertFalse((character_sums > single_pos).any())
        self.assertEqual(np.sum(character_sums), single_pos)
        table2 = single_table2.get_table()
        position_sums2 = np.sum(table2, axis=1)
        self.assertFalse((position_sums2 > 1).any())
        self.assertEqual(np.sum(position_sums2), single_pos)
        character_sums2 = np.sum(table2, axis=0)
        self.assertFalse((character_sums2 > single_pos).any())
        self.assertEqual(np.sum(character_sums2), single_pos)
        self.assertIsNone(pair_table)
        self.assertIsNone(pair_table2)

    def test18c_characterize_positions(self):
        # Test pair position only for the small sequence
        aln_small_sub = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[self.small_structure_id])
        pair_pos = np.sum(range(self.query_aln_fa_small.seq_length + 1))
        single_table, pair_table = aln_small_sub.characterize_positions(
            single=False, pair=True, single_size=self.single_letter_size, single_mapping=self.single_letter_mapping,
            single_reverse=self.single_letter_reverse, pair_size=self.pair_letter_size,
            pair_mapping=self.pair_letter_mapping, pair_reverse=self.pair_letter_reverse)
        single_table2, pair_table2 = aln_small_sub.characterize_positions2(
            single=False, pair=True, single_letter_size=self.single_letter_size,
            single_letter_mapping=self.single_letter_mapping, single_letter_reverse=self.single_letter_reverse,
            pair_letter_size=self.pair_letter_size, pair_letter_mapping=self.pair_letter_mapping,
            pair_letter_reverse=self.pair_letter_reverse, single_to_pair=self.single_to_pair)
        self.assertIsNone(single_table)
        self.assertIsNone(single_table2)
        positions = []
        for i in range(self.query_aln_fa_small.seq_length):
            for j in range(i, self.query_aln_fa_small.seq_length):
                position = (i, j)
                positions.append(position)
        self.assertEqual(pair_table.get_positions(), positions)
        self.assertEqual(pair_table2.get_positions(), positions)
        table1 = pair_table.get_table()
        position_sums = np.sum(table1, axis=1)
        self.assertFalse((position_sums > 1).any())
        self.assertEqual(np.sum(position_sums), pair_pos)
        character_sums = np.sum(table1, axis=0)
        self.assertFalse((character_sums > pair_pos).any())
        self.assertEqual(np.sum(character_sums), pair_pos)
        table2 = pair_table2.get_table()
        position_sums2 = np.sum(table2, axis=1)
        self.assertFalse((position_sums2 > 1).any())
        self.assertEqual(np.sum(position_sums2), pair_pos)
        character_sums2 = np.sum(table2, axis=0)
        self.assertFalse((character_sums2 > pair_pos).any())
        self.assertEqual(np.sum(character_sums2), pair_pos)

    def test18d_characterize_positions(self):
        # Test pair position only for the large sequence
        aln_large_sub = self.query_aln_fa_large.generate_sub_alignment(sequence_ids=[self.large_structure_id])
        pair_pos = np.sum(range(aln_large_sub.seq_length + 1))
        single_table, pair_table = aln_large_sub.characterize_positions(
            single=False, pair=True, single_size=self.single_letter_size, single_mapping=self.single_letter_mapping,
            single_reverse=self.single_letter_reverse, pair_size=self.pair_letter_size,
            pair_mapping=self.pair_letter_mapping, pair_reverse=self.pair_letter_reverse)
        single_table2, pair_table2 = aln_large_sub.characterize_positions2(
            single=False, pair=True, single_letter_size=self.single_letter_size,
            single_letter_mapping=self.single_letter_mapping, single_letter_reverse=self.single_letter_reverse,
            pair_letter_size=self.pair_letter_size, pair_letter_mapping=self.pair_letter_mapping,
            pair_letter_reverse=self.pair_letter_reverse, single_to_pair=self.single_to_pair)
        self.assertIsNone(single_table)
        self.assertIsNone(single_table2)
        positions = []
        for i in range(self.query_aln_fa_large.seq_length):
            for j in range(i, self.query_aln_fa_large.seq_length):
                position = (i, j)
                positions.append(position)
        self.assertEqual(pair_table.get_positions(), positions)
        self.assertEqual(pair_table2.get_positions(), positions)
        table1 = pair_table.get_table()
        position_sums = np.sum(table1, axis=1)
        self.assertFalse((position_sums > 1).any())
        self.assertEqual(np.sum(position_sums), pair_pos)
        character_sums = np.sum(table1, axis=0)
        self.assertFalse((character_sums > pair_pos).any())
        self.assertEqual(np.sum(character_sums), pair_pos)
        table2 = pair_table2.get_table()
        position_sums2 = np.sum(table2, axis=1)
        self.assertFalse((position_sums2 > 1).any())
        self.assertEqual(np.sum(position_sums2), pair_pos)
        character_sums2 = np.sum(table2, axis=0)
        self.assertFalse((character_sums2 > pair_pos).any())
        self.assertEqual(np.sum(character_sums2), pair_pos)

    def test18e_characterize_positions(self):
        # Test single position and pairs only for the small sequence
        aln_small_sub = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[self.small_structure_id])
        single_pos = aln_small_sub.seq_length
        pair_pos = np.sum(range(aln_small_sub.seq_length + 1))
        single_table, pair_table = aln_small_sub.characterize_positions(
            single=True, pair=True, single_size=self.single_letter_size, single_mapping=self.single_letter_mapping,
            single_reverse=self.single_letter_reverse, pair_size=self.pair_letter_size,
            pair_mapping=self.pair_letter_mapping, pair_reverse=self.pair_letter_reverse)
        single_table2, pair_table2 = aln_small_sub.characterize_positions2(
            single=True, pair=True, single_letter_size=self.single_letter_size,
            single_letter_mapping=self.single_letter_mapping, single_letter_reverse=self.single_letter_reverse,
            pair_letter_size=self.pair_letter_size, pair_letter_mapping=self.pair_letter_mapping,
            pair_letter_reverse=self.pair_letter_reverse, single_to_pair=self.single_to_pair)
        self.assertEqual(single_table.get_positions(), list(range(self.query_aln_fa_small.seq_length)))
        self.assertEqual(single_table2.get_positions(), list(range(self.query_aln_fa_small.seq_length)))
        table1 = single_table.get_table()
        position_sums = np.sum(table1, axis=1)
        self.assertFalse((position_sums > 1).any())
        self.assertEqual(np.sum(position_sums), single_pos)
        character_sums = np.sum(table1, axis=0)
        self.assertFalse((character_sums > single_pos).any())
        self.assertEqual(np.sum(character_sums), aln_small_sub.size * single_pos)
        table2 = single_table2.get_table()
        position_sums2 = np.sum(table2, axis=1)
        self.assertFalse((position_sums2 > 1).any())
        self.assertEqual(np.sum(position_sums2), single_pos)
        character_sums2 = np.sum(table2, axis=0)
        self.assertFalse((character_sums2 > single_pos).any())
        self.assertEqual(np.sum(character_sums2), aln_small_sub.size * single_pos)
        positions = []
        for i in range(self.query_aln_fa_small.seq_length):
            for j in range(i, self.query_aln_fa_small.seq_length):
                position = (i, j)
                positions.append(position)
        self.assertEqual(pair_table.get_positions(), positions)
        self.assertEqual(pair_table2.get_positions(), positions)
        table1 = pair_table.get_table()
        position_sums = np.sum(table1, axis=1)
        self.assertFalse((position_sums > 1).any())
        self.assertEqual(np.sum(position_sums), pair_pos)
        character_sums = np.sum(table1, axis=0)
        self.assertFalse((character_sums > pair_pos).any())
        self.assertEqual(np.sum(character_sums), aln_small_sub.size * pair_pos)
        table2 = pair_table2.get_table()
        position_sums2 = np.sum(table2, axis=1)
        self.assertFalse((position_sums2 > 1).any())
        self.assertEqual(np.sum(position_sums2), pair_pos)
        character_sums2 = np.sum(table2, axis=0)
        self.assertFalse((character_sums2 > pair_pos).any())
        self.assertEqual(np.sum(character_sums2), aln_small_sub.size * pair_pos)

    def test18f_characterize_positions(self):
        # Test single position and pairs only for the large sequence
        aln_large_sub = self.query_aln_fa_large.generate_sub_alignment(sequence_ids=[self.large_structure_id])
        single_pos = aln_large_sub.seq_length
        pair_pos = np.sum(range(aln_large_sub.seq_length + 1))
        single_table, pair_table = aln_large_sub.characterize_positions(
            single=True, pair=True, single_size=self.single_letter_size, single_mapping=self.single_letter_mapping,
            single_reverse=self.single_letter_reverse, pair_size=self.pair_letter_size,
            pair_mapping=self.pair_letter_mapping, pair_reverse=self.pair_letter_reverse)
        single_table2, pair_table2 = aln_large_sub.characterize_positions2(
            single=True, pair=True, single_letter_size=self.single_letter_size,
            single_letter_mapping=self.single_letter_mapping, single_letter_reverse=self.single_letter_reverse,
            pair_letter_size=self.pair_letter_size, pair_letter_mapping=self.pair_letter_mapping,
            pair_letter_reverse=self.pair_letter_reverse, single_to_pair=self.single_to_pair)
        self.assertEqual(single_table.get_positions(), list(range(self.query_aln_fa_large.seq_length)))
        self.assertEqual(single_table2.get_positions(), list(range(self.query_aln_fa_large.seq_length)))
        table1 = single_table.get_table()
        position_sums = np.sum(table1, axis=1)
        self.assertFalse((position_sums > 1).any())
        self.assertEqual(np.sum(position_sums), single_pos)
        character_sums = np.sum(table1, axis=0)
        self.assertFalse((character_sums > single_pos).any())
        self.assertEqual(np.sum(character_sums), aln_large_sub.size * single_pos)
        table2 = single_table2.get_table()
        position_sums2 = np.sum(table2, axis=1)
        self.assertFalse((position_sums2 > 1).any())
        self.assertEqual(np.sum(position_sums2), single_pos)
        character_sums2 = np.sum(table2, axis=0)
        self.assertFalse((character_sums2 > single_pos).any())
        self.assertEqual(np.sum(character_sums2), aln_large_sub.size * single_pos)
        positions = []
        for i in range(self.query_aln_fa_large.seq_length):
            for j in range(i, self.query_aln_fa_large.seq_length):
                position = (i, j)
                positions.append(position)
        self.assertEqual(pair_table.get_positions(), positions)
        self.assertEqual(pair_table2.get_positions(), positions)
        table1 = pair_table.get_table()
        position_sums = np.sum(table1, axis=1)
        self.assertFalse((position_sums > 1).any())
        self.assertEqual(np.sum(position_sums), pair_pos)
        character_sums = np.sum(table1, axis=0)
        self.assertFalse((character_sums > pair_pos).any())
        self.assertEqual(np.sum(character_sums), aln_large_sub.size * pair_pos)
        table2 = pair_table2.get_table()
        position_sums2 = np.sum(table2, axis=1)
        self.assertFalse((position_sums2 > 1).any())
        self.assertEqual(np.sum(position_sums2), pair_pos)
        character_sums2 = np.sum(table2, axis=0)
        self.assertFalse((character_sums2 > pair_pos).any())
        self.assertEqual(np.sum(character_sums2), aln_large_sub.size * pair_pos)

    def test18g_characterize_positions(self):
        # Test combination of two alignments frequency tables
        seq_order = self.query_aln_fa_small.seq_order
        aln_small_sub1 = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[seq_order[0]])
        single_table1, pair_table1 = aln_small_sub1.characterize_positions(
            single=True, pair=True, single_size=self.single_letter_size, single_mapping=self.single_letter_mapping,
            single_reverse=self.single_letter_reverse, pair_size=self.pair_letter_size,
            pair_mapping=self.pair_letter_mapping, pair_reverse=self.pair_letter_reverse)
        single_table1b, pair_table1b = aln_small_sub1.characterize_positions2(
            single=True, pair=True, single_letter_size=self.single_letter_size,
            single_letter_mapping=self.single_letter_mapping, single_letter_reverse=self.single_letter_reverse,
            pair_letter_size=self.pair_letter_size, pair_letter_mapping=self.pair_letter_mapping,
            pair_letter_reverse=self.pair_letter_reverse, single_to_pair=self.single_to_pair)
        aln_small_sub2 = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=[seq_order[1]])
        single_table2, pair_table2 = aln_small_sub2.characterize_positions(
            single=True, pair=True, single_size=self.single_letter_size, single_mapping=self.single_letter_mapping,
            single_reverse=self.single_letter_reverse, pair_size=self.pair_letter_size,
            pair_mapping=self.pair_letter_mapping, pair_reverse=self.pair_letter_reverse)
        single_table2b, pair_table2b = aln_small_sub2.characterize_positions2(
            single=True, pair=True, single_letter_size=self.single_letter_size,
            single_letter_mapping=self.single_letter_mapping, single_letter_reverse=self.single_letter_reverse,
            pair_letter_size=self.pair_letter_size, pair_letter_mapping=self.pair_letter_mapping,
            pair_letter_reverse=self.pair_letter_reverse, single_to_pair=self.single_to_pair)
        aln_small_sub3 = self.query_aln_fa_small.generate_sub_alignment(sequence_ids=seq_order[:2])
        single_table3, pair_table3 = aln_small_sub3.characterize_positions(
            single=True, pair=True, single_size=self.single_letter_size, single_mapping=self.single_letter_mapping,
            single_reverse=self.single_letter_reverse, pair_size=self.pair_letter_size,
            pair_mapping=self.pair_letter_mapping, pair_reverse=self.pair_letter_reverse)
        single_table3b, pair_table3b = aln_small_sub3.characterize_positions2(
            single=True, pair=True, single_letter_size=self.single_letter_size,
            single_letter_mapping=self.single_letter_mapping, single_letter_reverse=self.single_letter_reverse,
            pair_letter_size=self.pair_letter_size, pair_letter_mapping=self.pair_letter_mapping,
            pair_letter_reverse=self.pair_letter_reverse, single_to_pair=self.single_to_pair)
        single_combined = single_table1 + single_table2
        single_combinedb = single_table1b + single_table2b
        self.assertEqual(single_table3.mapping, single_combined.mapping)
        self.assertTrue(single_table3.reverse_mapping, single_combined.reverse_mapping)
        self.assertEqual(single_table3.num_pos, single_combined.num_pos)
        self.assertEqual(single_table3.position_size, single_combined.position_size)
        diff = single_table3.get_table() - single_combined.get_table()
        self.assertFalse(diff.toarray().any())
        self.assertEqual(single_table3.get_positions(), single_combined.get_positions())
        self.assertEqual(single_table3b.mapping, single_combinedb.mapping)
        self.assertTrue(single_table3b.reverse_mapping, single_combinedb.reverse_mapping)
        self.assertEqual(single_table3b.num_pos, single_combinedb.num_pos)
        self.assertEqual(single_table3b.position_size, single_combinedb.position_size)
        diff = single_table3b.get_table() - single_combinedb.get_table()
        self.assertFalse(diff.toarray().any())
        self.assertEqual(single_table3b.get_positions(), single_combinedb.get_positions())
        pair_combined = pair_table1 + pair_table2
        pair_combinedb = pair_table1 + pair_table2
        self.assertEqual(pair_table3.mapping, pair_combined.mapping)
        self.assertEqual(pair_table3.reverse_mapping, pair_combined.reverse_mapping)
        self.assertEqual(pair_table3.num_pos, pair_combined.num_pos)
        self.assertEqual(pair_table3.position_size, pair_combined.position_size)
        diff2 = pair_table3.get_table() - pair_combined.get_table()
        self.assertFalse(diff2.toarray().any())
        self.assertEqual(pair_table3.get_positions(), pair_combined.get_positions())
        self.assertEqual(pair_table3b.mapping, pair_combinedb.mapping)
        self.assertEqual(pair_table3b.reverse_mapping, pair_combinedb.reverse_mapping)
        self.assertEqual(pair_table3b.num_pos, pair_combinedb.num_pos)
        self.assertEqual(pair_table3b.position_size, pair_combinedb.position_size)
        diff2 = pair_table3b.get_table() - pair_combinedb.get_table()
        self.assertFalse(diff2.toarray().any())
        self.assertEqual(pair_table3b.get_positions(), pair_combinedb.get_positions())

    def test18h_characterize_positions(self):
        # Test combination of two alignments frequency tables
        large_aln = self.query_aln_fa_large.remove_gaps()
        seq_order = large_aln.seq_order
        aln_large_sub1 = large_aln.generate_sub_alignment(sequence_ids=[seq_order[0]])
        single_table1, pair_table1 = aln_large_sub1.characterize_positions(
            single=True, pair=True, single_size=self.single_letter_size, single_mapping=self.single_letter_mapping,
            single_reverse=self.single_letter_reverse, pair_size=self.pair_letter_size,
            pair_mapping=self.pair_letter_mapping, pair_reverse=self.pair_letter_reverse)
        single_table1b, pair_table1b = aln_large_sub1.characterize_positions2(
            single=True, pair=True, single_letter_size=self.single_letter_size,
            single_letter_mapping=self.single_letter_mapping, single_letter_reverse=self.single_letter_reverse,
            pair_letter_size=self.pair_letter_size, pair_letter_mapping=self.pair_letter_mapping,
            pair_letter_reverse=self.pair_letter_reverse, single_to_pair=self.single_to_pair)
        aln_large_sub2 = large_aln.generate_sub_alignment(sequence_ids=[seq_order[1]])
        single_table2, pair_table2 = aln_large_sub2.characterize_positions(
            single=True, pair=True, single_size=self.single_letter_size, single_mapping=self.single_letter_mapping,
            single_reverse=self.single_letter_reverse, pair_size=self.pair_letter_size,
            pair_mapping=self.pair_letter_mapping, pair_reverse=self.pair_letter_reverse)
        single_table2b, pair_table2b = aln_large_sub2.characterize_positions2(
            single=True, pair=True, single_letter_size=self.single_letter_size,
            single_letter_mapping=self.single_letter_mapping, single_letter_reverse=self.single_letter_reverse,
            pair_letter_size=self.pair_letter_size, pair_letter_mapping=self.pair_letter_mapping,
            pair_letter_reverse=self.pair_letter_reverse, single_to_pair=self.single_to_pair)
        aln_large_sub3 = large_aln.generate_sub_alignment(sequence_ids=seq_order[:2])
        single_table3, pair_table3 = aln_large_sub3.characterize_positions(
            single=True, pair=True, single_size=self.single_letter_size, single_mapping=self.single_letter_mapping,
            single_reverse=self.single_letter_reverse, pair_size=self.pair_letter_size,
            pair_mapping=self.pair_letter_mapping, pair_reverse=self.pair_letter_reverse)
        single_table3b, pair_table3b = aln_large_sub3.characterize_positions2(
            single=True, pair=True, single_letter_size=self.single_letter_size,
            single_letter_mapping=self.single_letter_mapping, single_letter_reverse=self.single_letter_reverse,
            pair_letter_size=self.pair_letter_size, pair_letter_mapping=self.pair_letter_mapping,
            pair_letter_reverse=self.pair_letter_reverse, single_to_pair=self.single_to_pair)
        single_combined = single_table1 + single_table2
        single_combinedb = single_table1b + single_table2b
        self.assertEqual(single_table3.mapping, single_combined.mapping)
        self.assertEqual(single_table3.reverse_mapping, single_combined.reverse_mapping)
        self.assertEqual(single_table3.num_pos, single_combined.num_pos)
        self.assertEqual(single_table3.position_size, single_combined.position_size)
        diff = single_table3.get_table() - single_combined.get_table()
        self.assertFalse(diff.toarray().any())
        self.assertEqual(single_table3.get_depth(), single_combined.get_depth())
        self.assertEqual(single_table3.get_positions(), single_combined.get_positions())
        self.assertEqual(single_table3b.mapping, single_combinedb.mapping)
        self.assertEqual(single_table3b.reverse_mapping, single_combinedb.reverse_mapping)
        self.assertEqual(single_table3b.num_pos, single_combinedb.num_pos)
        self.assertEqual(single_table3b.position_size, single_combinedb.position_size)
        diff = single_table3b.get_table() - single_combinedb.get_table()
        self.assertFalse(diff.toarray().any())
        self.assertEqual(single_table3b.get_depth(), single_combinedb.get_depth())
        self.assertEqual(single_table3b.get_positions(), single_combinedb.get_positions())
        pair_combined = pair_table1 + pair_table2
        pair_combinedb = pair_table1b + pair_table2b
        self.assertEqual(pair_table3.mapping, pair_combined.mapping)
        self.assertEqual(pair_table3.reverse_mapping, pair_combined.reverse_mapping)
        self.assertEqual(pair_table3.num_pos, pair_combined.num_pos)
        self.assertEqual(pair_table3.position_size, pair_combined.position_size)
        diff2 = pair_table3.get_table() - pair_combined.get_table()
        self.assertFalse(diff2.toarray().any())
        self.assertEqual(pair_table3.get_depth(), pair_combined.get_depth())
        self.assertEqual(pair_table3.get_positions(), pair_combined.get_positions())
        self.assertEqual(pair_table3b.mapping, pair_combinedb.mapping)
        self.assertEqual(pair_table3b.reverse_mapping, pair_combinedb.reverse_mapping)
        self.assertEqual(pair_table3b.num_pos, pair_combinedb.num_pos)
        self.assertEqual(pair_table3b.position_size, pair_combinedb.position_size)
        diff2 = pair_table3b.get_table() - pair_combinedb.get_table()
        self.assertFalse(diff2.toarray().any())
        self.assertEqual(pair_table3b.get_depth(), pair_combinedb.get_depth())
        self.assertEqual(pair_table3b.get_positions(), pair_combinedb.get_positions())


def subset_string(in_str, positions):
    new_str = ''
    for i in positions:
        new_str += in_str[i]
    return new_str


if __name__ == '__main__':
    unittest.main()
