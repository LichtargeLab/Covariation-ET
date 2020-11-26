"""
Created onJune 19, 2019

@author: daniel
"""
import os
import numpy as np
from datetime import datetime
from Bio.Seq import Seq
from Bio.Alphabet import Gapped
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceMatrix
from EvolutionaryTraceAlphabet import FullIUPACDNA, FullIUPACProtein, MultiPositionAlphabet
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from utils import build_mapping
from unittest import TestCase
from multiprocessing import cpu_count
from DataSetGenerator import DataSetGenerator


def generate_temp_fn(suffix):
    return f'temp_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.{suffix}'


def write_out_temp_fn(suffix, out_str=None):
    fn = generate_temp_fn(suffix=suffix)
    with open(fn, 'a') as handle:
        os.utime(fn)
        if out_str:
            handle.write(out_str)
    return fn


# Variables to be used by tests, some of these variables rely on classes which are being tested, only the precursors
# data to a given class will be used in the tests of that class.

protein_short_seq = SeqRecord(id='seq1', seq=Seq('MET', alphabet=FullIUPACProtein()))
protein_seq1 = SeqRecord(id='seq1', seq=Seq('MET---', alphabet=FullIUPACProtein()))
protein_seq2 = SeqRecord(id='seq2', seq=Seq('M-TREE', alphabet=FullIUPACProtein()))
protein_seq3 = SeqRecord(id='seq3', seq=Seq('M-FREE', alphabet=FullIUPACProtein()))
protein_seq4 = SeqRecord(id='seq4', seq=Seq('------', alphabet=FullIUPACProtein()))
protein_msa = MultipleSeqAlignment(records=[protein_seq1, protein_seq2, protein_seq3], alphabet=FullIUPACProtein())
dna_seq1 = SeqRecord(id='seq1', seq=Seq('ATGGAGACT---------', alphabet=FullIUPACDNA()))
dna_seq2 = SeqRecord(id='seq2', seq=Seq('ATG---ACTAGAGAGGAG', alphabet=FullIUPACDNA()))
dna_seq3 = SeqRecord(id='seq3', seq=Seq('ATG---TTTAGAGAGGAG', alphabet=FullIUPACDNA()))
dna_msa = MultipleSeqAlignment(records=[dna_seq1, dna_seq2, dna_seq3], alphabet=FullIUPACDNA())

min_dm = DistanceMatrix(names=['seq1', 'seq2', 'seq3'])
id_adc = AlignmentDistanceCalculator(model='identity')
id_dm = id_adc.get_distance(msa=protein_msa, processes=2)

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
pro_single_to_pair = np.zeros((max(protein_map.values()) + 1, max(protein_map.values()) + 1), dtype=np.int)
pro_single_to_pair_map = {}
for char in pro_pair_map:
    pro_single_to_pair[protein_map[char[0]], protein_map[char[1]]] = pro_pair_map[char]
    pro_single_to_pair_map[(protein_map[char[0]], protein_map[char[1]])] = pro_pair_map[char]


# class TestBase(TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.max_threads = cpu_count() - 2
#         cls.max_target_seqs = 150
#         cls.testing_dir = os.environ.get('TEST_PATH')
#         cls.input_path = os.path.join(cls.testing_dir, 'Input')
#         cls.protein_list_path = os.path.join(cls.input_path, 'ProteinLists')
#         if not os.path.isdir(cls.protein_list_path):
#             os.makedirs(cls.protein_list_path)
#         cls.small_structure_id = '135l'
#         cls.large_structure_id = '1bol'
#         cls.protein_list_fn = os.path.join(cls.protein_list_path, 'Test_Set.txt')
#         structure_ids = [cls.small_structure_id, cls.large_structure_id]
#         with open(cls.protein_list_fn, 'w') as test_list_handle:
#             for structure_id in structure_ids:
#                 test_list_handle.write('{}{}\n'.format(structure_id, 'A'))
#         cls.data_set = DataSetGenerator(input_path=cls.input_path)
#         cls.data_set.build_pdb_alignment_dataset(protein_list_fn='Test_Set.txt', processes=cls.max_threads,
#                                                  max_target_seqs=cls.max_target_seqs)
#
#     @classmethod
#     def tearDownClass(cls):
#         # rmtree(cls.input_path)
#         del cls.max_threads
#         del cls.max_target_seqs
#         del cls.testing_dir
#         del cls.input_path
#         del cls.protein_list_path
#         del cls.small_structure_id
#         del cls.large_structure_id
#         del cls.protein_list_fn
#         del cls.data_set
