"""
Created onJune 19, 2019

@author: daniel
"""
import os
from unittest import TestCase
from multiprocessing import cpu_count
from DataSetGenerator import DataSetGenerator


class TestBase(TestCase):

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
        # cls.query_aln_fa_small = SeqAlignment(
        #     file_name=cls.data_set.protein_data[cls.small_structure_id]['Final_FA_Aln'],
        #     query_id=cls.small_structure_id)
        # cls.query_aln_fa_small.import_alignment()
        # cls.query_aln_fa_large = SeqAlignment(
        #     file_name=cls.data_set.protein_data[cls.large_structure_id]['Final_FA_Aln'],
        #     query_id=cls.large_structure_id)
        # cls.query_aln_fa_large.import_alignment()
        # cls.query_aln_msf_small = deepcopy(cls.query_aln_fa_small)
        # cls.query_aln_msf_small.file_name = cls.data_set.protein_data[cls.small_structure_id]['Final_MSF_Aln']
        # cls.query_aln_msf_large = deepcopy(cls.query_aln_fa_large)
        # cls.query_aln_msf_large.file_name = cls.data_set.protein_data[cls.large_structure_id]['Final_MSF_Aln']
        # cls.save_file_small = os.path.join(cls.testing_dir, '{}_aln.pkl'.format(cls.small_structure_id))
        # cls.save_file_large = os.path.join(cls.testing_dir, '{}_aln.pkl'.format(cls.large_structure_id))
        # cls.aln_file_small = os.path.join(cls.testing_dir, 'test_{}.fa'.format(cls.small_structure_id))
        # cls.aln_file_large = os.path.join(cls.testing_dir, 'test_{}.fa'.format(cls.large_structure_id))
        # cls.save_dir_small = os.path.join(cls.testing_dir, '{}_cache'.format(cls.small_structure_id))
        # cls.save_dir_large = os.path.join(cls.testing_dir, '{}_cache'.format(cls.large_structure_id))