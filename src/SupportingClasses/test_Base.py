"""
Created onJune 19, 2019

@author: daniel
"""
import os
import numpy as np
from copy import deepcopy
from datetime import datetime
from Bio.Seq import Seq
from Bio.Alphabet import Gapped
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceMatrix
from EvolutionaryTraceAlphabet import FullIUPACDNA, FullIUPACProtein, MultiPositionAlphabet
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from MatchMismatchTable import MatchMismatchTable
from PhylogeneticTree import PhylogeneticTree
from FrequencyTable import FrequencyTable
from SeqAlignment import SeqAlignment
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


def compare_nodes_key(compare_nodes):
    """Taken from: https://docs.python.org/3/howto/sorting.html"""
    class K:
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return compare_nodes(self.obj, other.obj) < 0

        def __gt__(self, other):
            return compare_nodes(self.obj, other.obj) > 0

        def __eq__(self, other):
            return compare_nodes(self.obj, other.obj) == 0

        def __le__(self, other):
            return compare_nodes(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return compare_nodes(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return compare_nodes(self.obj, other.obj) != 0
    return K


def compare_nodes(node1, node2):
    if node1.is_terminal and not node2.is_terminal():
        return -1
    elif not node1.is_terminal() and node2.is_terminal():
        return 1
    else:
        if node1.name < node2.name:
            return 1
        elif node1.name > node2.name:
            return -1
        else:
            return 0


# Variables to be used by tests, some of these variables rely on classes which are being tested, only the precursors
# data to a given class will be used in the tests of that class.
processes = 2

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

dna_alpha = Gapped(FullIUPACDNA())
dna_alpha_size, _, dna_map, dna_rev = build_mapping(dna_alpha)
protein_alpha = Gapped(FullIUPACProtein())
protein_alpha_size, _, protein_map, protein_rev = build_mapping(protein_alpha)
pair_dna_alpha = MultiPositionAlphabet(dna_alpha, size=2)
dna_pair_alpha_size, _, dna_pair_map, dna_pair_rev = build_mapping(pair_dna_alpha)
quad_dna_alpha = MultiPositionAlphabet(dna_alpha, size=4)
dna_quad_alpha_size, _, dna_quad_map, dna_quad_rev = build_mapping(quad_dna_alpha)
dna_single_to_pair = np.zeros((max(dna_map.values()) + 1, max(dna_map.values()) + 1), dtype=np.int32)
dna_pair_mismatch = np.zeros(dna_pair_alpha_size, dtype=np.bool_)
dna_single_to_pair_map = {}
for char in dna_pair_map:
    dna_single_to_pair[dna_map[char[0]], dna_map[char[1]]] = dna_pair_map[char]
    dna_single_to_pair_map[(dna_map[char[0]], dna_map[char[1]])] = dna_pair_map[char]
    if dna_map[char[0]] != dna_map[char[1]]:
        dna_pair_mismatch[dna_pair_map[char]] = True
dna_single_to_quad_map = {}
dna_pair_to_quad = np.zeros((max(dna_pair_map.values()) + 1, max(dna_pair_map.values()) + 1), dtype=np.int32)
dna_quad_mismatch = np.zeros(dna_quad_alpha_size, dtype=np.bool_)
for char in dna_quad_map:
    dna_single_to_quad_map[(dna_map[char[0]], dna_map[char[1]], dna_map[char[2]], dna_map[char[3]])] = dna_quad_map[char]
    dna_pair_to_quad[dna_pair_map[char[:2]], dna_pair_map[char[2:]]] = dna_quad_map[char]
    if (((dna_map[char[0]] == dna_map[char[2]]) and (dna_map[char[1]] != dna_map[char[3]])) or
            ((dna_map[char[0]] != dna_map[char[2]]) and (dna_map[char[1]] == dna_map[char[3]]))):
        dna_quad_mismatch[dna_quad_map[char]] = True
pair_protein_alpha = MultiPositionAlphabet(protein_alpha, size=2)
pro_pair_alpha_size, _, pro_pair_map, pro_pair_rev = build_mapping(pair_protein_alpha)
quad_protein_alpha = MultiPositionAlphabet(protein_alpha, size=4)
pro_quad_alpha_size, _, pro_quad_map, pro_quad_rev = build_mapping(quad_protein_alpha)
pro_single_to_pair = np.zeros((max(protein_map.values()) + 1, max(protein_map.values()) + 1), dtype=np.int32)
pro_pair_mismatch = np.zeros(pro_pair_alpha_size, dtype=np.bool_)
pro_single_to_pair_map = {}
for char in pro_pair_map:
    pro_single_to_pair[protein_map[char[0]], protein_map[char[1]]] = pro_pair_map[char]
    pro_single_to_pair_map[(protein_map[char[0]], protein_map[char[1]])] = pro_pair_map[char]
    if protein_map[char[0]] != protein_map[char[1]]:
        pro_pair_mismatch[pro_pair_map[char]] = True
pro_single_to_quad_map = {}
pro_pair_to_quad = np.zeros((max(pro_pair_map.values()) + 1, max(pro_pair_map.values()) + 1), dtype=np.int32)
pro_quad_mismatch = np.zeros(pro_quad_alpha_size, dtype=np.bool_)
for char in pro_quad_map:
    key = (protein_map[char[0]], protein_map[char[1]], protein_map[char[2]], protein_map[char[3]])
    pro_single_to_quad_map[key] = pro_quad_map[char]
    pro_pair_to_quad[pro_pair_map[char[:2]], pro_pair_map[char[2:]]] = pro_quad_map[char]
    if (((protein_map[char[0]] == protein_map[char[2]]) and (protein_map[char[1]] != protein_map[char[3]])) or
            ((protein_map[char[0]] != protein_map[char[2]]) and (protein_map[char[1]] == protein_map[char[3]]))):
        pro_quad_mismatch[pro_quad_map[char]] = True

protein_aln_fn = write_out_temp_fn(suffix='fasta',
                                   out_str=f'>seq1\n{str(protein_seq1.seq)}\n>seq2\n{str(protein_seq2.seq)}\n>seq3\n{str(protein_seq3.seq)}')
protein_aln = SeqAlignment(protein_aln_fn, 'seq1', polymer_type='Protein')
protein_aln.import_alignment()
os.remove(protein_aln_fn)
protein_num_aln = protein_aln._alignment_to_num(mapping=protein_map)

min_dm = DistanceMatrix(names=['seq1', 'seq2', 'seq3'])
adc = AlignmentDistanceCalculator(model='identity')
id_dm = adc.get_distance(msa=protein_msa, processes=processes)

protein_phylo_tree = PhylogeneticTree(tree_building_method='upgma', tree_building_args={})
protein_phylo_tree.construct_tree(dm=id_dm)
protein_rank_dict = protein_phylo_tree.assign_group_rank(ranks=None)

pro_single_ft = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
pro_single_ft.characterize_alignment(num_aln=protein_num_aln)
pro_single_ft_i2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
pro_single_ft_i2.characterize_alignment(num_aln=protein_num_aln[[1, 2], :])
pro_single_ft_s1 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
pro_single_ft_s1.characterize_alignment(num_aln=np.array([protein_num_aln[0, :]]))
pro_single_ft_s2 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
pro_single_ft_s2.characterize_alignment(num_aln=np.array([protein_num_aln[1, :]]))
pro_single_ft_s3 = FrequencyTable(protein_alpha_size, protein_map, protein_rev, 6, 1)
pro_single_ft_s3.characterize_alignment(num_aln=np.array([protein_num_aln[2, :]]))

pro_pair_ft = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
pro_pair_ft.characterize_alignment(num_aln=protein_num_aln, single_to_pair=pro_single_to_pair)
pro_pair_ft_i2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
pro_pair_ft_i2.characterize_alignment(num_aln=protein_num_aln[[1, 2], :], single_to_pair=pro_single_to_pair)
pro_pair_ft_s1 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
pro_pair_ft_s1.characterize_alignment(num_aln=np.array([protein_num_aln[0, :]]), single_to_pair=pro_single_to_pair)
pro_pair_ft_s2 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
pro_pair_ft_s2.characterize_alignment(num_aln=np.array([protein_num_aln[1, :]]), single_to_pair=pro_single_to_pair)
pro_pair_ft_s3 = FrequencyTable(pro_pair_alpha_size, pro_pair_map, pro_pair_rev, 6, 2)
pro_pair_ft_s3.characterize_alignment(num_aln=np.array([protein_num_aln[2, :]]), single_to_pair=pro_single_to_pair)

protein_mm_table = MatchMismatchTable(seq_len=6, num_aln=protein_num_aln[[2, 1, 0], :], single_alphabet_size=protein_alpha_size,
                                      single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                      larger_alphabet_size=pro_pair_alpha_size, larger_mapping=pro_pair_map,
                                      larger_reverse_mapping=pro_pair_rev,
                                      single_to_larger_mapping=pro_single_to_pair_map, pos_size=1)
protein_mm_table.identify_matches_mismatches()

protein_mm_table_large = MatchMismatchTable(seq_len=6, num_aln=protein_num_aln[[2, 1, 0], :], single_alphabet_size=protein_alpha_size,
                                            single_mapping=protein_map, single_reverse_mapping=protein_rev,
                                            larger_alphabet_size=pro_quad_alpha_size,
                                            larger_mapping=pro_quad_map, larger_reverse_mapping=pro_quad_rev,
                                            single_to_larger_mapping=pro_single_to_quad_map, pos_size=2)
protein_mm_table_large.identify_matches_mismatches()

protein_mm_freq_tables = {'match': FrequencyTable(alphabet_size=pro_quad_alpha_size, mapping=pro_quad_map,
                                                  reverse_mapping=pro_quad_rev, seq_len=6, pos_size=2)}
protein_mm_freq_tables['match'].mapping = pro_quad_map
protein_mm_freq_tables['match'].set_depth(3)
protein_mm_freq_tables['mismatch'] = deepcopy(protein_mm_freq_tables['match'])
for pos in protein_mm_freq_tables['match'].get_positions():
    char_dict = {'match': {}, 'mismatch': {}}
    for i in range(3):
        for j in range(i + 1, 3):
            status, stat_char = protein_mm_table_large.get_status_and_character(pos=pos, seq_ind1=i, seq_ind2=j)
            if stat_char not in char_dict[status]:
                char_dict[status][stat_char] = 0
            char_dict[status][stat_char] += 1
    for m in char_dict:
        for curr_char in char_dict[m]:
            protein_mm_freq_tables[m]._increment_count(pos=pos, char=curr_char,
                                                       amount=char_dict[m][curr_char])
for m in ['match', 'mismatch']:
    protein_mm_freq_tables[m].finalize_table()

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