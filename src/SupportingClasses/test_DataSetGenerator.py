"""
Created on May 28, 2019

@author: Daniel Konecki
"""
import os
from shutil import rmtree
from unittest import TestCase
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet.IUPAC import ExtendedIUPACProtein
from multiprocessing import cpu_count
from DataSetGenerator import (DataSetGenerator, import_protein_list, download_pdb, parse_query_sequence,
                              blast_query_sequence, filter_blast_sequences, align_sequences)


class TestDataSetGenerator(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.max_target_seqs = 500
        cls.max_threads = cpu_count() - 2
        cls.input_path = os.path.join(os.path.abspath('../Test/'), 'Input')
        cls.protein_list_path = os.path.join(cls.input_path, 'ProteinLists')
        if not os.path.isdir(cls.protein_list_path):
            os.makedirs(cls.protein_list_path)
        cls.small_structure_id = '7hvp'
        cls.small_query_seq = SeqRecord(Seq('PQITLWQRPLVTIRIGGQLKEALLDTGADDTVLEEMNLPGKWKPKMIGGIGGFIKVRQYDQIPVEIGHKAIGTV'
                                            'LVGPTPVNIIGRNLLTQIGTLNF', alphabet=ExtendedIUPACProtein),
                                        id=cls.small_structure_id, description='Target Query')
        cls.large_structure_id = '2zxe'
        cls.large_query_seq = SeqRecord(Seq(
            'LDELKKEVSMDDHKLSLDELHNKYGTDLTRGLTNARAKEILARDGPNSLTPPPTTPEWIKFCRQLFGGFSILLWIGAILCFLAYGIQAATEDEPANDNLYLGVVLS'
            'TVVIVTGCFSYYQEAKSSRIMDSFKNMVPQQALVIRDGEKSTINAEFVVAGDLVEVKGGDRIPADLRIISAHGCKVDNSSLTGESEPQTRSPEFSSENPLETRNIA'
            'FFSTNCVEGTARGVVVYTGDRTVMGRIATLASGLEVGRTPIAIEIEHFIHIITGVAVFLGVSFFILSLILGYSWLEAVIFLIGIIVANVPEGLLATVTVCLTLTAK'
            'RMARKNCLVKNLEAVETLGSTSTICSDKTGTLTQNRMTVAHMWFDNQIHEADTTENQSGAAFDKTSATWSALSRIAALCNRAVFQAGQDNVPILKRSVAGDASESA'
            'LLKCIELCCGSVQGMRDRNPKIVEIPFNSTNKYQLSIHENEKSSESRYLLVMKGAPERILDRCSTILLNGAEEPLKEDMKEAFQNAYLELGGLGERVLGFCHFALP'
            'EDKYNEGYPFDADEPNFPTTDLCFVGLMAMIDPPRAAVPDAVGKCRSAGIKVIMVTGDHPITAKAIAKGVGIISEGNETIEDIAARLNIPIGQVNPRDAKACVVHG'
            'SDLKDLSTEVLDDILHYHTEIVFARTSPQQKLIIVEGCQRQGAIVAVTGDGVNDSPALKKADIGVAMGISGSDVSKQAADMILLDDNFASIVTGVEEGRLIFDNLK'
            'KSIAYTLTSNIPEITPFLVFIIGNVPLPLGTVTILCIDLGTDMVPAISLAYEQAESDIMKRQPRNPKTDKLVNERLISMAYGQIGMIQALGGFFSYFVILAENGFL'
            'PMDLIGKRVRWDDRWISDVEDSFGQQWTYEQRKIVEFTCHTSFFISIVVVQWADLIICKTRRNSIFQQGMKNKILIFGLFEETALAAFLSYCPGTDVALRMYPLKP'
            'SWWFCAFPYSLIIFLYDEMRRFIIRRSPGGWVEQETYY', alphabet=ExtendedIUPACProtein), id=cls.large_structure_id,
            description='Target Query')
        cls.protein_list_name = 'Test_Set.txt'
        cls.protein_list_fn = os.path.join(cls.protein_list_path, cls.protein_list_name)
        cls.pdb_path = os.path.join(cls.input_path, 'PDB')
        cls.expected_pdb_fn_small = os.path.join(cls.pdb_path, '{}'.format(cls.small_structure_id[1:3]),
                                                 'pdb{}.ent'.format(cls.small_structure_id))
        cls.expected_pdb_fn_large = os.path.join(cls.pdb_path, '{}'.format(cls.large_structure_id[1:3]),
                                                 'pdb{}.ent'.format(cls.large_structure_id))
        cls.sequence_path = os.path.join(cls.input_path, 'Sequences')
        cls.expected_seq_fn_small = os.path.join(cls.sequence_path, '{}.fasta'.format(cls.small_structure_id))
        cls.expected_seq_fn_large = os.path.join(cls.sequence_path, '{}.fasta'.format(cls.large_structure_id))
        cls.blast_path = os.path.join(cls.input_path, 'BLAST')
        cls.expected_blast_fn_small = os.path.join(cls.blast_path, '{}.xml'.format(cls.small_structure_id))
        cls.expected_blast_fn_large = os.path.join(cls.blast_path, '{}.xml'.format(cls.large_structure_id))
        cls.filtered_blast_path = os.path.join(cls.input_path, 'Filtered_BLAST')
        cls.expected_filtered_blast_fn_small = os.path.join(cls.filtered_blast_path,
                                                            '{}.fasta'.format(cls.small_structure_id))
        cls.expected_filtered_blast_fn_large = os.path.join(cls.filtered_blast_path,
                                                            '{}.fasta'.format(cls.large_structure_id))
        cls.alignment_path = os.path.join(cls.input_path, 'Alignments')
        cls.expected_msf_fn_small = os.path.join(cls.alignment_path, '{}.msf'.format(cls.small_structure_id))
        cls.expected_fa_fn_small = os.path.join(cls.alignment_path, '{}.fasta'.format(cls.small_structure_id))
        cls.expected_msf_fn_large = os.path.join(cls.alignment_path, '{}.msf'.format(cls.large_structure_id))
        cls.expected_fa_fn_large = os.path.join(cls.alignment_path, '{}.fasta'.format(cls.large_structure_id))
        structure_ids = [cls.small_structure_id, cls.large_structure_id]
        with open(cls.protein_list_fn, 'wb') as test_list_handle:
            for structure_id in structure_ids:
                test_list_handle.write('{}{}\n'.format(structure_id, 'A'))

    @classmethod
    def tearDownClass(cls):
        # rmtree(cls.input_path)
        del cls.protein_list_fn
        del cls.large_query_seq
        del cls.large_structure_id
        del cls.small_query_seq
        del cls.small_structure_id
        del cls.protein_list_path
        del cls.input_path

    def test1_import_protein_list(self):
        with self.assertRaises(IOError):
            import_protein_list(protein_list_fn=self.protein_list_name)
        protein_dict = import_protein_list(protein_list_fn=self.protein_list_fn)
        self.assertTrue(self.small_structure_id in protein_dict)
        self.assertEqual(protein_dict[self.small_structure_id]['Chain'], 'A')
        self.assertTrue(self.large_structure_id in protein_dict)
        self.assertEqual(protein_dict[self.large_structure_id]['Chain'], 'A')

    def test2_download_pdb(self):
        if os.path.isdir(self.pdb_path):
            rmtree(self.pdb_path)
        pdb_fn_small = download_pdb(pdb_path=self.pdb_path, protein_id=self.small_structure_id)
        self.assertTrue(os.path.isdir(self.pdb_path))

        self.assertEqual(pdb_fn_small, self.expected_pdb_fn_small)
        self.assertTrue(os.path.isfile(self.expected_pdb_fn_small))
        pdb_fn_large = download_pdb(pdb_path=self.pdb_path, protein_id=self.large_structure_id)
        self.assertEqual(pdb_fn_large, self.expected_pdb_fn_large)
        self.assertTrue(os.path.isfile(self.expected_pdb_fn_large))

    def test3_parse_query_sequence(self):
        if os.path.isdir(self.sequence_path):
            rmtree(self.sequence_path)
        if not os.path.isdir(self.pdb_path):
            self.test2__download_pdb()
        seq_small, len_small, seq_fn_small = parse_query_sequence(protein_id=self.small_structure_id, chain_id='A',
                                                                  sequence_path=self.sequence_path,
                                                                  pdb_fn=self.expected_pdb_fn_small)
        self.assertTrue(os.path.isdir(self.sequence_path))
        self.assertEqual(str(self.small_query_seq.seq), str(seq_small.seq))
        self.assertEqual(len_small, len(seq_small))
        self.assertEqual(seq_fn_small, self.expected_seq_fn_small)
        seq_large, len_large, seq_fn_large = parse_query_sequence(protein_id=self.large_structure_id, chain_id='A',
                                                                  sequence_path=self.sequence_path,
                                                                  pdb_fn=self.expected_pdb_fn_large)
        self.assertTrue(os.path.isdir(self.sequence_path))
        self.assertEqual(str(self.large_query_seq.seq), str(seq_large.seq))
        self.assertEqual(len_large, len(seq_large))
        self.assertEqual(seq_fn_large, self.expected_seq_fn_large)

    def test4a_blast_query_sequence_single_thread(self):
        if os.path.isdir(self.blast_path):
            rmtree(self.blast_path)
        if not os.path.isdir(self.sequence_path):
            self.test3__parse_query_sequence()
        if os.path.isfile(self.expected_blast_fn_small):
            os.remove(self.expected_blast_fn_small)
        count_small, blast_fn_small = blast_query_sequence(
            protein_id=self.small_structure_id, blast_path=self.blast_path, sequence_fn=self.expected_seq_fn_small,
            evalue=0.05, num_threads=1, max_target_seqs=100, database='nr', remote=False)
        self.assertTrue(os.path.isdir(self.blast_path))
        self.assertEqual(blast_fn_small, self.expected_blast_fn_small)
        self.assertTrue(os.path.isfile(blast_fn_small))
        self.assertLessEqual(count_small, 100)
        if os.path.isfile(self.expected_blast_fn_large):
            os.remove(self.expected_blast_fn_large)
        count_large, blast_fn_large = blast_query_sequence(
            protein_id=self.large_structure_id, blast_path=self.blast_path, sequence_fn=self.expected_seq_fn_large,
            evalue=0.05, num_threads=1, max_target_seqs=100, database='nr', remote=False)
        self.assertTrue(os.path.isdir(self.blast_path))
        self.assertEqual(blast_fn_large, self.expected_blast_fn_large)
        self.assertTrue(os.path.isfile(blast_fn_large))
        self.assertLessEqual(count_large, 100)

    def test4b__blast_query_sequence_multi_thread(self):
        if os.path.isdir(self.blast_path):
            rmtree(self.blast_path)
        if not os.path.isdir(self.sequence_path):
            self.test3__parse_query_sequence()
        if os.path.isfile(self.expected_blast_fn_small):
            os.remove(self.expected_blast_fn_small)
        count_small, blast_fn_small = blast_query_sequence(
            protein_id=self.small_structure_id, blast_path=self.blast_path, sequence_fn=self.expected_seq_fn_small,
            evalue=0.05, num_threads=self.max_threads, max_target_seqs=100, database='nr', remote=False)
        self.assertTrue(os.path.isdir(self.blast_path))
        self.assertEqual(blast_fn_small, self.expected_blast_fn_small)
        self.assertTrue(os.path.isfile(blast_fn_small))
        self.assertLessEqual(count_small, 100)
        if os.path.isfile(self.expected_blast_fn_large):
            os.remove(self.expected_blast_fn_large)
        count_large, blast_fn_large = blast_query_sequence(protein_id=self.large_structure_id, blast_path=self.blast_path,
                                              sequence_fn=self.expected_seq_fn_large, evalue=0.05,
                                              num_threads=self.max_threads, max_target_seqs=100, database='nr',
                                              remote=False)
        self.assertTrue(os.path.isdir(self.blast_path))
        self.assertEqual(blast_fn_large, self.expected_blast_fn_large)
        self.assertTrue(os.path.isfile(blast_fn_large))
        self.assertLessEqual(count_large, 100)

    def test4c__blast_query_sequence_remote(self):
        if os.path.isdir(self.blast_path):
            rmtree(self.blast_path)
        if not os.path.isdir(self.sequence_path):
            self.test3__parse_query_sequence()
        if os.path.isfile(self.expected_blast_fn_small):
            os.remove(self.expected_blast_fn_small)
        count_small, blast_fn_small = blast_query_sequence(
            protein_id=self.small_structure_id, blast_path=self.blast_path, sequence_fn=self.expected_seq_fn_small,
            evalue=0.05, num_threads=self.max_threads, max_target_seqs=100, database='nr', remote=True)
        self.assertTrue(os.path.isdir(self.blast_path))
        self.assertEqual(blast_fn_small, self.expected_blast_fn_small)
        self.assertTrue(os.path.isfile(blast_fn_small))
        self.assertLessEqual(count_small, 100)
        if os.path.isfile(self.expected_blast_fn_large):
            os.remove(self.expected_blast_fn_large)
        count_large, blast_fn_large = blast_query_sequence(
            protein_id=self.large_structure_id, blast_path=self.blast_path, sequence_fn=self.expected_seq_fn_large,
            evalue=0.05, num_threads=self.max_threads, max_target_seqs=100, database='nr', remote=False)
        self.assertTrue(os.path.isdir(self.blast_path))
        self.assertEqual(blast_fn_large, self.expected_blast_fn_large)
        self.assertTrue(os.path.isfile(blast_fn_large))
        self.assertLessEqual(count_large, 100)

    def test5a_filter_blast_sequences(self):
        if os.path.isdir(self.filtered_blast_path):
            rmtree(self.filtered_blast_path)
        if not os.path.isdir(self.blast_path):
            self.test4b__blast_query_sequence_multi_thread()
        if os.path.isfile(self.expected_filtered_blast_fn_small):
            os.remove(self.expected_filtered_blast_fn_small)
        num_seqs_small, pileup_fn_small = filter_blast_sequences(
            protein_id=self.small_structure_id, filter_path=self.filtered_blast_path,
            blast_fn=self.expected_blast_fn_small, query_seq=self.small_query_seq, e_value_threshold=0.05,
            min_fraction=0.7, min_identity=40, max_identity=98)
        self.assertGreaterEqual(num_seqs_small, 0)
        self.assertEqual(pileup_fn_small, self.expected_filtered_blast_fn_small)
        if os.path.isfile(self.expected_filtered_blast_fn_large):
            os.remove(self.expected_filtered_blast_fn_large)
        num_seqs_large, pileup_fn_large = filter_blast_sequences(
            protein_id=self.large_structure_id, filter_path=self.filtered_blast_path,
            blast_fn=self.expected_blast_fn_large, query_seq=self.large_query_seq, e_value_threshold=0.05,
            min_fraction=0.7, min_identity=40, max_identity=98)
        self.assertGreaterEqual(num_seqs_large, 0)
        self.assertEqual(pileup_fn_large, self.expected_filtered_blast_fn_large)

    def test5b_filter_blast_sequences(self):
        if os.path.isdir(self.filtered_blast_path):
            rmtree(self.filtered_blast_path)
        if not os.path.isdir(self.blast_path):
            self.test4b__blast_query_sequence_multi_thread()
        if os.path.isfile(self.expected_filtered_blast_fn_small):
            os.remove(self.expected_filtered_blast_fn_small)
        num_seqs_small1, pileup_fn_small1 = filter_blast_sequences(
            protein_id=self.small_structure_id, filter_path=self.filtered_blast_path,
            blast_fn=self.expected_blast_fn_small, query_seq=self.small_query_seq, e_value_threshold=0.05,
            min_fraction=0.7, min_identity=40, max_identity=98)
        num_seqs_small2, pileup_fn_small2 = filter_blast_sequences(
            protein_id=self.small_structure_id, filter_path=self.filtered_blast_path,
            blast_fn=self.expected_blast_fn_small, query_seq=self.small_query_seq, e_value_threshold=0.05,
            min_fraction=0.7, min_identity=40, max_identity=98)
        self.assertEqual(num_seqs_small1, num_seqs_small2)
        self.assertEqual(pileup_fn_small1, pileup_fn_small2)
        num_seqs_large1, pileup_fn_large1 = filter_blast_sequences(
            protein_id=self.large_structure_id, filter_path=self.filtered_blast_path,
            blast_fn=self.expected_blast_fn_large, query_seq=self.large_query_seq, e_value_threshold=0.05,
            min_fraction=0.7, min_identity=40, max_identity=98)
        num_seqs_large2, pileup_fn_large2 = filter_blast_sequences(
            protein_id=self.large_structure_id, filter_path=self.filtered_blast_path,
            blast_fn=self.expected_blast_fn_large, query_seq=self.large_query_seq, e_value_threshold=0.05,
            min_fraction=0.7, min_identity=40, max_identity=98)
        self.assertGreaterEqual(num_seqs_large1, num_seqs_large2)
        self.assertEqual(pileup_fn_large1, pileup_fn_large2)

    def test6a_align_sequences(self):
        if os.path.isdir(self.alignment_path):
            rmtree(self.alignment_path)
        if not os.path.isdir(self.filtered_blast_path):
            self.test5a_filter_blast_sequences()
        if os.path.isfile(self.expected_msf_fn_small):
            os.remove(self.expected_msf_fn_small)
        if os.path.isfile(self.expected_fa_fn_small):
            os.remove(self.expected_fa_fn_small)
        fa_fn_small, msf_fn_small = align_sequences(protein_id=self.small_structure_id,
                                                    alignment_path=self.alignment_path,
                                                    pileup_fn=self.expected_filtered_blast_fn_small)
        self.assertTrue(os.path.isfile(self.expected_fa_fn_small))
        self.assertEqual(fa_fn_small, self.expected_fa_fn_small)
        self.assertTrue(os.path.isfile(self.expected_msf_fn_small))
        self.assertEqual(msf_fn_small, self.expected_msf_fn_small)
        if os.path.isfile(self.expected_msf_fn_large):
            os.remove(self.expected_msf_fn_large)
        if os.path.isfile(self.expected_fa_fn_large):
            os.remove(self.expected_fa_fn_large)
        fa_fn_large, msf_fn_large = align_sequences(protein_id=self.large_structure_id,
                                                    alignment_path=self.alignment_path,
                                                    pileup_fn=self.expected_filtered_blast_fn_large)
        self.assertTrue(os.path.isfile(self.expected_msf_fn_large))
        self.assertEqual(msf_fn_large, self.expected_msf_fn_large)
        self.assertTrue(os.path.isfile(self.expected_fa_fn_large))
        self.assertEqual(fa_fn_large, self.expected_fa_fn_large)

    def test6b_align_sequences_fasta_only(self):
        if os.path.isdir(self.alignment_path):
            rmtree(self.alignment_path)
        if not os.path.isdir(self.filtered_blast_path):
            self.test5a_filter_blast_sequences()
        if os.path.isfile(self.expected_msf_fn_small):
            os.remove(self.expected_msf_fn_small)
        if os.path.isfile(self.expected_fa_fn_small):
            os.remove(self.expected_fa_fn_small)
        fa_fn_small, msf_fn_small = align_sequences(protein_id=self.small_structure_id,
                                                    alignment_path=self.alignment_path,
                                                    pileup_fn=self.expected_filtered_blast_fn_small, msf=False)
        self.assertIsNone(msf_fn_small)
        self.assertTrue(os.path.isfile(self.expected_fa_fn_small))
        self.assertEqual(fa_fn_small, self.expected_fa_fn_small)
        if os.path.isfile(self.expected_msf_fn_large):
            os.remove(self.expected_msf_fn_large)
        if os.path.isfile(self.expected_fa_fn_large):
            os.remove(self.expected_fa_fn_large)
        fa_fn_large, msf_fn_large = align_sequences(protein_id=self.large_structure_id,
                                                    alignment_path=self.alignment_path,
                                                    pileup_fn=self.expected_filtered_blast_fn_large, msf=False)
        self.assertTrue(os.path.isfile(self.expected_fa_fn_large))
        self.assertEqual(fa_fn_large, self.expected_fa_fn_large)
        self.assertIsNone(msf_fn_large)

    def test6c_align_sequences_msf_only(self):
        if os.path.isdir(self.alignment_path):
            rmtree(self.alignment_path)
        if not os.path.isdir(self.filtered_blast_path):
            self.test5a_filter_blast_sequences()
        if os.path.isfile(self.expected_msf_fn_small):
            os.remove(self.expected_msf_fn_small)
        if os.path.isfile(self.expected_fa_fn_small):
            os.remove(self.expected_fa_fn_small)
        fa_fn_small, msf_fn_small = align_sequences(protein_id=self.small_structure_id,
                                                    alignment_path=self.alignment_path,
                                                    pileup_fn=self.expected_filtered_blast_fn_small, fasta=False)
        self.assertIsNone(fa_fn_small)
        self.assertTrue(os.path.isfile(self.expected_msf_fn_small))
        self.assertEqual(msf_fn_small, self.expected_msf_fn_small)
        if os.path.isfile(self.expected_msf_fn_large):
            os.remove(self.expected_msf_fn_large)
        if os.path.isfile(self.expected_fa_fn_large):
            os.remove(self.expected_fa_fn_large)
        fa_fn_large, msf_fn_large = align_sequences(protein_id=self.large_structure_id,
                                                    alignment_path=self.alignment_path,
                                                    pileup_fn=self.expected_filtered_blast_fn_large, fasta=False)
        self.assertTrue(os.path.isfile(self.expected_msf_fn_large))
        self.assertEqual(msf_fn_large, self.expected_msf_fn_large)
        self.assertIsNone(fa_fn_large)

    # def test1_init(self):
    #     test_generator = DataSetGenerator(protein_list='Test_Set.txt', input_path=self.input_path)
    #     self.assertTrue(test_generator.input_path == self.input_path)
    #     self.assertTrue(test_generator.file_name == os.path.basename(self.protein_list_fn))
    #     self.assertEqual(len(test_generator.protein_data), 2)
    #     self.assertTrue(self.small_structure_id in test_generator.protein_data)
    #     self.assertTrue(self.large_structure_id in test_generator.protein_data)
    #
    # def test7_build_dataset(self):
    #     for curr_fn in os.listdir(self.input_path):
    #         curr_dir = os.path.join(self.input_path, curr_fn)
    #         if os.path.isdir(curr_dir) and curr_fn != 'ProteinLists':
    #             rmtree(curr_dir)
    #     test_generator = DataSetGenerator(protein_list='Test_Set.txt', input_path=self.input_path)
    #     test_generator.build_dataset(num_threads=self.max_threads, max_target_seqs=self.max_target_seqs)
    #     pdb_path = os.path.join(self.input_path, 'PDB')
    #     expected_pdb_fn_small = os.path.join(pdb_path, '{}'.format(self.small_structure_id[1:3]),
    #                                          'pdb{}.ent'.format(self.small_structure_id))
    #     self.assertTrue(test_generator.protein_data[self.small_structure_id]['PDB_Path'] == expected_pdb_fn_small)
    #     expected_pdb_fn_large = os.path.join(pdb_path, '{}'.format(self.large_structure_id[1:3]),
    #                                      'pdb{}.ent'.format(self.large_structure_id))
    #     self.assertTrue(test_generator.protein_data[self.large_structure_id]['PDB_Path'] == expected_pdb_fn_large)
    #     sequence_path = os.path.join(self.input_path, 'Sequences')
    #     expexcted_query_fn_small = os.path.join(sequence_path, '{}.fasta'.format(self.small_structure_id))
    #     self.assertEqual(str(test_generator.protein_data[self.small_structure_id]['Query_Sequence'].seq),
    #                      str(self.small_query_seq.seq))
    #     self.assertEqual(test_generator.protein_data[self.small_structure_id]['Sequence_Length'],
    #                      len(self.small_query_seq.seq))
    #     self.assertEqual(test_generator.protein_data[self.small_structure_id]['Fasta_File'], expexcted_query_fn_small)
    #     expexcted_query_fn_large = os.path.join(sequence_path, '{}.fasta'.format(self.large_structure_id))
    #     self.assertEqual(str(test_generator.protein_data[self.large_structure_id]['Query_Sequence'].seq),
    #                      str(self.large_query_seq.seq))
    #     self.assertEqual(test_generator.protein_data[self.large_structure_id]['Sequence_Length'],
    #                      len(self.large_query_seq.seq))
    #     self.assertEqual(test_generator.protein_data[self.large_structure_id]['Fasta_File'], expexcted_query_fn_large)
    #     blast_path = os.path.join(self.input_path, 'BLAST')
    #     expected_blast_fn_small = os.path.join(blast_path, '{}.xml'.format(self.small_structure_id))
    #     self.assertEqual(test_generator.protein_data[self.small_structure_id]['BLAST_File'], expected_blast_fn_small)
    #     expected_blast_fn_large = os.path.join(blast_path, '{}.xml'.format(self.large_structure_id))
    #     self.assertEqual(test_generator.protein_data[self.large_structure_id]['BLAST_File'], expected_blast_fn_large)
    #     pileup_path = os.path.join(self.input_path, 'Pileups')
    #     expected_pileup_fn_small = os.path.join(pileup_path, '{}.fasta'.format(self.small_structure_id))
    #     alignment_path = os.path.join(self.input_path, 'Alignments')
    #     expected_msf_fn_small = os.path.join(alignment_path, '{}.msf'.format(self.small_structure_id))
    #     expected_fa_fn_small = os.path.join(alignment_path, '{}.fasta'.format(self.small_structure_id))
    #     if test_generator.protein_data[self.small_structure_id]['Pileup_File']:
    #         self.assertEqual(test_generator.protein_data[self.small_structure_id]['Pileup_File'],
    #                          expected_pileup_fn_small)
    #         self.assertEqual(test_generator.protein_data[self.small_structure_id]['MSF_File'], expected_msf_fn_small)
    #         self.assertEqual(test_generator.protein_data[self.small_structure_id]['FA_File'], expected_fa_fn_small)
    #     else:
    #         self.assertIsNone(test_generator.protein_data[self.small_structure_id]['Pileup_File'])
    #         self.assertIsNone(test_generator.protein_data[self.small_structure_id]['MSF_File'])
    #         self.assertIsNone(test_generator.protein_data[self.small_structure_id]['FA_File'])
    #     expected_pileup_fn_large = os.path.join(pileup_path, '{}.fasta'.format(self.large_structure_id))
    #     expected_msf_fn_large = os.path.join(alignment_path, '{}.msf'.format(self.large_structure_id))
    #     expected_fa_fn_large = os.path.join(alignment_path, '{}.fasta'.format(self.large_structure_id))
    #     if test_generator.protein_data[self.large_structure_id]['Pileup_File']:
    #         self.assertEqual(test_generator.protein_data[self.large_structure_id]['Pileup_File'],
    #                          expected_pileup_fn_large)
    #         self.assertEqual(test_generator.protein_data[self.large_structure_id]['MSF_File'], expected_msf_fn_large)
    #         self.assertEqual(test_generator.protein_data[self.large_structure_id]['FA_File'], expected_fa_fn_large)
    #     else:
    #         self.assertIsNone(test_generator.protein_data[self.large_structure_id]['Pileup_File'])
    #         self.assertIsNone(test_generator.protein_data[self.large_structure_id]['MSF_File'])
    #         self.assertIsNone(test_generator.protein_data[self.large_structure_id]['FA_File'])




