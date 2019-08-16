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
        cls.testing_dir = os.environ.get('TEST_PATH')
        cls.input_path = os.path.join(cls.testing_dir, 'Input')
        cls.protein_list_path = os.path.join(cls.input_path, 'ProteinLists')
        if not os.path.isdir(cls.protein_list_path):
            os.makedirs(cls.protein_list_path)
        cls.small_structure_id = '7hvp'
        cls.large_structure_id = '2zxe'
        cls.protein_list_fn = os.path.join(cls.protein_list_path, 'Test_Set.txt')
        structure_ids = [cls.small_structure_id, cls.large_structure_id]
        with open(cls.protein_list_fn, 'w') as test_list_handle:
            for structure_id in structure_ids:
                test_list_handle.write('{}{}\n'.format(structure_id, 'A'))
        cls.data_set = DataSetGenerator(input_path=cls.input_path)
        cls.data_set.build_pdb_alignment_dataset(protein_list_fn='Test_Set.txt', num_threads=cls.max_threads,
                                                 max_target_seqs=cls.max_target_seqs)

    @classmethod
    def tearDownClass(cls):
        # rmtree(cls.input_path)
        del cls.max_threads
        del cls.max_target_seqs
        del cls.testing_dir
        del cls.input_path
        del cls.protein_list_path
        del cls.small_structure_id
        del cls.large_structure_id
        del cls.protein_list_fn
        del cls.data_set
