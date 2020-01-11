"""
Created on May 28, 2019

@author: Daniel Konecki
"""
import os
import unittest
from time import time
from shutil import rmtree
from unittest import TestCase
from Bio.Seq import Seq
from Bio.SeqIO import write
from Bio.SeqRecord import SeqRecord
from multiprocessing import cpu_count, Lock
from dotenv import find_dotenv, load_dotenv
from EvolutionaryTraceAlphabet import FullIUPACProtein
from DataSetGenerator import (DataSetGenerator, import_protein_list, download_pdb, parse_query_sequence,
                              init_pdb_processing_pool, pdb_processing, blast_query_sequence, filter_blast_sequences,
                              align_sequences, identity_filter, init_filtering_and_alignment_pool,
                              filtering_and_alignment)
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)


class TestDataSetGenerator(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.max_target_seqs = 100
        cls.max_e_value = 0.05
        cls.local_database = 'customuniref90.fasta'
        cls.remote_database = 'nr'
        cls.max_threads = cpu_count() - 2
        cls.input_path = os.path.join(os.environ.get("TEST_PATH"), 'Input_Temp')
        cls.protein_list_path = os.path.join(cls.input_path, 'ProteinLists')
        if not os.path.isdir(cls.protein_list_path):
            os.makedirs(cls.protein_list_path)
        cls.small_structure_id = '135l'
        cls.small_query_seq_uniprot = SeqRecord(
            Seq('MRSLLILVLCFLPLAALGKVYGRCELAAAMKRLGLDNYRGYSLGNWVCAAKFESNFNTHATNRNTDGSTDYGILQINSRWWCNDGRTPGSKNLCNIPCSALL'
                'SSDITASVNCAKKIASGGNGMNAWVAWRNRCKGTDVHAWIRGCRL', alphabet=FullIUPACProtein), id=cls.small_structure_id,
            description='Target Query: Chain: A: From UniProt Accession: LYSC_MELGA')
        cls.large_structure_id = '1bol'
        cls.large_query_seq_uniprot = SeqRecord(Seq(
            'MKAVLALATLIGSTLASSCSSTALSCSNSANSDTCCSPEYGLVVLNMQWAPGYGPDNAFTLHGLWPDKCSGAYAPSGGCDSNRASSSIASVIKSKDSSLYNSMLTY'
            'WPSNQGNNNVFWSHEWSKHGTCVSTYDPDCYDNYEEGEDIVDYFQKAMDLRSQYNVYKAFSSNGITPGGTYTATEMQSAIESYFGAKAKIDCSSGTLSDVALYFYV'
            'RGRDTYVITDALSTGSCSGDVEYPTK', alphabet=FullIUPACProtein), id=cls.large_structure_id,
            description='Target Query: Chain: A: From UniProt Accession: RNRH_RHINI')
        cls.protein_list_name = 'Test_Set.txt'
        cls.protein_list_fn = os.path.join(cls.protein_list_path, cls.protein_list_name)
        cls.pdb_path = os.path.join(cls.input_path, 'PDB')
        cls.expected_pdb_fn_small = os.path.join(cls.pdb_path, '{}'.format(cls.small_structure_id[1:3]),
                                                 'pdb{}.ent'.format(cls.small_structure_id))
        cls.expected_pdb_fn_large = os.path.join(cls.pdb_path, '{}'.format(cls.large_structure_id[1:3]),
                                                 'pdb{}.ent'.format(cls.large_structure_id))
        cls.sequence_path = os.path.join(cls.input_path, 'Sequences')
        cls.expected_seq_fn = os.path.join(cls.sequence_path, 'Test_Set.fasta')
        cls.expected_seq_fn_small = os.path.join(cls.sequence_path, '{}.fasta'.format(cls.small_structure_id))
        cls.expected_seq_fn_large = os.path.join(cls.sequence_path, '{}.fasta'.format(cls.large_structure_id))
        cls.blast_path = os.path.join(cls.input_path, 'BLAST')
        cls.expected_blast_fn = os.path.join(cls.blast_path, 'Test_Set_All_Seqs.xml')
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
        cls.filtered_alignment_path = os.path.join(cls.input_path, 'Filtered_Alignment')
        cls.expected_filtered_aln_fn_small = os.path.join(cls.filtered_alignment_path,
                                                          '{}.fasta'.format(cls.small_structure_id))
        cls.expected_filtered_aln_fn_large = os.path.join(cls.filtered_alignment_path,
                                                          '{}.fasta'.format(cls.large_structure_id))
        cls.final_alignment_path = os.path.join(cls.input_path, 'Final_Alignments')
        cls.expected_final_msf_fn_small = os.path.join(cls.final_alignment_path,
                                                       '{}.msf'.format(cls.small_structure_id))
        cls.expected_final_fa_fn_small = os.path.join(cls.final_alignment_path,
                                                      '{}.fasta'.format(cls.small_structure_id))
        cls.expected_final_msf_fn_large = os.path.join(cls.final_alignment_path,
                                                       '{}.msf'.format(cls.large_structure_id))
        cls.expected_final_fa_fn_large = os.path.join(cls.final_alignment_path,
                                                      '{}.fasta'.format(cls.large_structure_id))
        structure_ids = [cls.small_structure_id, cls.large_structure_id]
        with open(cls.protein_list_fn, 'w') as test_list_handle:
            for structure_id in structure_ids:
                test_list_handle.write('{}{}\n'.format(structure_id, 'A'))

    # @classmethod
    # def tearDownClass(cls):
        # rmtree(cls.input_path)
        # del cls.protein_list_fn
        # del cls.large_query_seq
        # del cls.large_structure_id
        # del cls.small_query_seq
        # del cls.small_structure_id
        # del cls.protein_list_path
        # del cls.input_path

    def test1_init(self):
        test_generator = DataSetGenerator(input_path=self.input_path)
        self.assertTrue(os.path.isdir(self.pdb_path))
        self.assertTrue(os.path.isdir(self.sequence_path))
        self.assertTrue(os.path.isdir(self.blast_path))
        self.assertTrue(os.path.isdir(self.filtered_blast_path))
        self.assertTrue(os.path.isdir(self.alignment_path))
        self.assertTrue(os.path.isdir(self.filtered_alignment_path))
        self.assertTrue(os.path.isdir(self.final_alignment_path))
        self.assertEqual(test_generator.pdb_path, self.pdb_path)
        self.assertEqual(test_generator.sequence_path, self.sequence_path)
        self.assertEqual(test_generator.blast_path, self.blast_path)
        self.assertEqual(test_generator.filtered_blast_path, self.filtered_blast_path)
        self.assertEqual(test_generator.alignment_path, self.alignment_path)
        self.assertEqual(test_generator.filtered_alignment_path, self.filtered_alignment_path)
        self.assertEqual(test_generator.final_alignment_path, self.final_alignment_path)

    def test2_build_pdb_alignment_dataset(self):
        for fn in os.listdir(self.input_path):
            if fn != os.path.basename(self.protein_list_path):
                rmtree(os.path.join(self.input_path, fn))
        test_generator = DataSetGenerator(input_path=self.input_path)
        test_generator.build_pdb_alignment_dataset(protein_list_fn=self.protein_list_fn, processes=self.max_threads,
                                                   max_target_seqs=self.max_target_seqs)
        self.assertTrue(self.small_structure_id in test_generator.protein_data)
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['Chain'], 'A')
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['PDB'], self.expected_pdb_fn_small)
        self.assertTrue(os.path.isfile(self.expected_pdb_fn_small))
        self.assertEqual(str(test_generator.protein_data[self.small_structure_id]['Sequence'].seq),
                         str(self.small_query_seq_uniprot.seq))
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['Length'],
                         len(str(self.small_query_seq_uniprot.seq)))
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['Seq_Fasta'], self.expected_seq_fn_small)
        self.assertTrue(os.path.isfile(self.expected_seq_fn_small))
        self.assertLessEqual(test_generator.protein_data[self.small_structure_id]['BLAST_Hits'], self.max_target_seqs)
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['BLAST'], self.expected_blast_fn)

        self.assertTrue(os.path.isfile(self.expected_blast_fn))
        self.assertLessEqual(test_generator.protein_data[self.small_structure_id]['Filter_Count'],
                             test_generator.protein_data[self.small_structure_id]['BLAST_Hits'] + 1)
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['Filtered_BLAST'],
                         self.expected_filtered_blast_fn_small)
        self.assertTrue(os.path.isfile(self.expected_filtered_blast_fn_small))
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['MSF_Aln'], self.expected_msf_fn_small)
        self.assertTrue(os.path.isfile(self.expected_msf_fn_small))
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['FA_Aln'], self.expected_fa_fn_small)
        self.assertTrue(os.path.isfile(self.expected_fa_fn_small))
        self.assertLessEqual(test_generator.protein_data[self.small_structure_id]['Final_Count'],
                             test_generator.protein_data[self.small_structure_id]['Filter_Count'])
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['Filtered_Alignment'],
                         self.expected_filtered_aln_fn_small)
        self.assertTrue(os.path.isfile(self.expected_filtered_aln_fn_small))
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['Final_MSF_Aln'],
                         self.expected_final_msf_fn_small)
        self.assertTrue(os.path.isfile(self.expected_final_msf_fn_small))
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['Final_FA_Aln'],
                         self.expected_final_fa_fn_small)
        self.assertTrue(os.path.isfile(self.expected_final_fa_fn_small))
        self.assertTrue(self.large_structure_id in test_generator.protein_data)
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['Chain'], 'A')
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['PDB'], self.expected_pdb_fn_large)
        self.assertTrue(os.path.isfile(self.expected_pdb_fn_large))
        self.assertEqual(str(test_generator.protein_data[self.large_structure_id]['Sequence'].seq),
                         str(self.large_query_seq_uniprot.seq))
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['Length'],
                         len(str(self.large_query_seq_uniprot.seq)))
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['Seq_Fasta'], self.expected_seq_fn_large)
        self.assertTrue(os.path.isfile(self.expected_seq_fn_large))
        self.assertLessEqual(test_generator.protein_data[self.large_structure_id]['BLAST_Hits'], self.max_target_seqs)
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['BLAST'], self.expected_blast_fn)
        self.assertTrue(os.path.isfile(self.expected_blast_fn))
        self.assertLessEqual(test_generator.protein_data[self.large_structure_id]['Filter_Count'],
                             test_generator.protein_data[self.large_structure_id]['BLAST_Hits'] + 1)
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['Filtered_BLAST'],
                         self.expected_filtered_blast_fn_large)
        self.assertTrue(os.path.isfile(self.expected_filtered_blast_fn_large))
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['MSF_Aln'], self.expected_msf_fn_large)
        self.assertTrue(os.path.isfile(self.expected_msf_fn_large))
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['FA_Aln'], self.expected_fa_fn_large)
        self.assertTrue(os.path.isfile(self.expected_fa_fn_large))
        self.assertLessEqual(test_generator.protein_data[self.large_structure_id]['Final_Count'],
                             test_generator.protein_data[self.large_structure_id]['Filter_Count'])
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['Filtered_Alignment'],
                         self.expected_filtered_aln_fn_large)
        self.assertTrue(os.path.isfile(self.expected_filtered_aln_fn_large))
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['Final_MSF_Aln'],
                         self.expected_final_msf_fn_large)
        self.assertTrue(os.path.isfile(self.expected_final_msf_fn_large))
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['Final_FA_Aln'],
                         self.expected_final_fa_fn_large)
        self.assertTrue(os.path.isfile(self.expected_final_fa_fn_large))
        self.assertTrue(os.path.isfile(self.expected_seq_fn))

    def test3_import_protein_list(self):
        with self.assertRaises(IOError):
            import_protein_list(protein_list_fn=self.protein_list_name)
        protein_dict = import_protein_list(protein_list_fn=self.protein_list_fn)
        self.assertTrue(self.small_structure_id in protein_dict)
        self.assertEqual(protein_dict[self.small_structure_id]['Chain'], 'A')
        self.assertTrue(self.large_structure_id in protein_dict)
        self.assertEqual(protein_dict[self.large_structure_id]['Chain'], 'A')

    def test4a_pdb_processing_download_pdb(self):
        if os.path.isdir(self.pdb_path):
            rmtree(self.pdb_path)
        pdb_fn_small = download_pdb(pdb_path=self.pdb_path, protein_id=self.small_structure_id)
        self.assertTrue(os.path.isdir(self.pdb_path))

        self.assertEqual(pdb_fn_small, self.expected_pdb_fn_small)
        self.assertTrue(os.path.isfile(self.expected_pdb_fn_small))
        pdb_fn_large = download_pdb(pdb_path=self.pdb_path, protein_id=self.large_structure_id)
        self.assertEqual(pdb_fn_large, self.expected_pdb_fn_large)
        self.assertTrue(os.path.isfile(self.expected_pdb_fn_large))

    def test4b_pdb_processing_parse_query_sequence(self):
        if os.path.isdir(self.sequence_path):
            rmtree(self.sequence_path)
        if not os.path.isdir(self.pdb_path):
            self.test4a_pdb_processing_download_pdb()
        seq_small, len_small, seq_fn_small, chain_small, unp_small = parse_query_sequence(
            protein_id=self.small_structure_id, chain_id='A', sequence_path=self.sequence_path,
            pdb_fn=self.expected_pdb_fn_small)
        self.assertTrue(os.path.isdir(self.sequence_path))
        self.assertEqual(str(self.small_query_seq_uniprot.seq), str(seq_small.seq))
        self.assertEqual(len_small, len(seq_small))
        self.assertEqual(seq_fn_small, self.expected_seq_fn_small)
        self.assertEqual(chain_small, 'A')
        self.assertIsNotNone(unp_small)
        self.assertTrue(unp_small in {'P00703', 'LYSC_MELGA'})
        seq_large, len_large, seq_fn_large, chain_large, unp_large = parse_query_sequence(
            protein_id=self.large_structure_id, chain_id='A', sequence_path=self.sequence_path,
            pdb_fn=self.expected_pdb_fn_large)
        self.assertTrue(os.path.isdir(self.sequence_path))
        self.assertEqual(str(self.large_query_seq_uniprot.seq), str(seq_large.seq))
        self.assertEqual(len_large, len(seq_large))
        self.assertEqual(seq_fn_large, self.expected_seq_fn_large)
        self.assertEqual(chain_large, 'A')
        self.assertIsNotNone(unp_large)
        self.assertTrue(unp_large in {'P08056', 'RNRH_RHINI'})

    def test4c_pdb_processing(self):
        test_lock = Lock()
        init_pdb_processing_pool(pdb_path=self.pdb_path, sequence_path=self.sequence_path, lock=test_lock,
                                 verbose=False)
        p_id, p_data = pdb_processing(in_tuple=(self.small_structure_id, 'A'))
        self.assertEqual(p_id, self.small_structure_id)
        self.assertEqual(p_data['PDB'], self.expected_pdb_fn_small)
        self.assertTrue(os.path.isfile(self.expected_pdb_fn_small))
        self.assertEqual(p_data['Chain'], 'A')
        self.assertEqual(str(p_data['Sequence'].seq), str(self.small_query_seq_uniprot.seq))
        self.assertEqual(p_data['Length'], len(self.small_query_seq_uniprot.seq))
        self.assertEqual(p_data['Seq_Fasta'], self.expected_seq_fn_small)
        self.assertTrue(os.path.isfile(self.expected_seq_fn_small))

    def test4d_pdb_processing(self):
        test_lock = Lock()
        init_pdb_processing_pool(pdb_path=self.pdb_path, sequence_path=self.sequence_path, lock=test_lock,
                                 verbose=False)
        p_id, p_data = pdb_processing(in_tuple=(self.large_structure_id, 'A'))
        self.assertEqual(p_id, self.large_structure_id)
        self.assertEqual(p_data['PDB'], self.expected_pdb_fn_large)
        self.assertTrue(os.path.isfile(self.expected_pdb_fn_large))
        self.assertEqual(p_data['Chain'], 'A')
        self.assertEqual(str(p_data['Sequence'].seq), str(self.large_query_seq_uniprot.seq))
        self.assertEqual(p_data['Length'], len(self.large_query_seq_uniprot.seq))
        self.assertEqual(p_data['Seq_Fasta'], self.expected_seq_fn_large)
        self.assertTrue(os.path.isfile(self.expected_seq_fn_large))

    def test5a_filtering_and_alignment_blast_query_sequence_single_thread(self):
        if os.path.isdir(self.blast_path):
            rmtree(self.blast_path)
        if not os.path.isdir(self.sequence_path):
            self.test4b_pdb_processing_parse_query_sequence()
        if not os.path.isfile(self.expected_seq_fn):
            with open(self.expected_seq_fn, 'w') as seq_handle:
                write([self.small_query_seq_uniprot, self.large_query_seq_uniprot], seq_handle, 'fasta')
        if os.path.isfile(self.expected_blast_fn):
            os.remove(self.expected_blast_fn)
        blast_fn_all, count_all = blast_query_sequence(
            protein_id='Test_Set_All_Seqs', blast_path=self.blast_path, sequence_fn=self.expected_seq_fn,
            evalue=self.max_e_value, processes=1, max_target_seqs=self.max_target_seqs,
            database=self.local_database, remote=False)
        self.assertTrue(os.path.isdir(self.blast_path))
        self.assertEqual(blast_fn_all, self.expected_blast_fn)
        self.assertTrue(os.path.isfile(blast_fn_all))
        self.assertLessEqual(count_all[self.small_structure_id]['BLAST_Hits'], self.max_target_seqs)
        self.assertLessEqual(count_all[self.large_structure_id]['BLAST_Hits'], self.max_target_seqs)

    def test5b_filtering_and_alignment_blast_query_sequence_multi_thread(self):
        if os.path.isdir(self.blast_path):
            rmtree(self.blast_path)
        if not os.path.isdir(self.sequence_path):
            self.test4b_pdb_processing_parse_query_sequence()
        if not os.path.isfile(self.expected_seq_fn):
            with open(self.expected_seq_fn, 'w') as seq_handle:
                write([self.small_query_seq_uniprot, self.large_query_seq_uniprot], seq_handle, 'fasta')
        if os.path.isfile(self.expected_blast_fn):
            os.remove(self.expected_blast_fn)
        blast_fn_all, count_all = blast_query_sequence(
            protein_id='Test_Set_All_Seqs', blast_path=self.blast_path, sequence_fn=self.expected_seq_fn,
            evalue=self.max_e_value, processes=self.max_threads, max_target_seqs=self.max_target_seqs,
            database=self.local_database, remote=False)
        self.assertTrue(os.path.isdir(self.blast_path))
        self.assertEqual(blast_fn_all, self.expected_blast_fn)
        self.assertTrue(os.path.isfile(blast_fn_all))
        self.assertLessEqual(count_all[self.small_structure_id]['BLAST_Hits'], self.max_target_seqs)
        self.assertLessEqual(count_all[self.large_structure_id]['BLAST_Hits'], self.max_target_seqs)

    def test5c_filtering_and_alignment_blast_query_sequence_remote(self):
        if os.path.isdir(self.blast_path):
            rmtree(self.blast_path)
        if not os.path.isdir(self.sequence_path):
            self.test4b_pdb_processing_parse_query_sequence()
        if not os.path.isfile(self.expected_seq_fn):
            with open(self.expected_seq_fn, 'w') as seq_handle:
                write([self.small_query_seq_uniprot, self.large_query_seq_uniprot], seq_handle, 'fasta')
        if os.path.isfile(self.expected_blast_fn):
            os.remove(self.expected_blast_fn)
        blast_fn_all, count_all = blast_query_sequence(
            protein_id='Test_Set_All_Seqs', blast_path=self.blast_path, sequence_fn=self.expected_seq_fn,
            evalue=self.max_e_value, processes=1, max_target_seqs=self.max_target_seqs, database=self.remote_database,
            remote=True)
        self.assertTrue(os.path.isdir(self.blast_path))
        self.assertEqual(blast_fn_all, self.expected_blast_fn)
        self.assertTrue(os.path.isfile(blast_fn_all))
        self.assertLessEqual(count_all[self.small_structure_id]['BLAST_Hits'], self.max_target_seqs)
        self.assertLessEqual(count_all[self.large_structure_id]['BLAST_Hits'], self.max_target_seqs)

    def test5d_filtering_and_alignment_filter_blast_sequences(self):
        if os.path.isdir(self.filtered_blast_path):
            rmtree(self.filtered_blast_path)
        if not os.path.isdir(self.blast_path):
            self.test5b_filtering_and_alignment_blast_query_sequence_multi_thread()
        if os.path.isfile(self.expected_filtered_blast_fn_small):
            os.remove(self.expected_filtered_blast_fn_small)
        num_seqs_small, pileup_fn_small = filter_blast_sequences(
            protein_id=self.small_structure_id, filter_path=self.filtered_blast_path,
            blast_fn=self.expected_blast_fn, query_seq=self.small_query_seq_uniprot)
        self.assertGreaterEqual(num_seqs_small, 0)
        self.assertLessEqual(num_seqs_small, self.max_target_seqs + 1)
        self.assertEqual(pileup_fn_small, self.expected_filtered_blast_fn_small)
        if os.path.isfile(self.expected_filtered_blast_fn_large):
            os.remove(self.expected_filtered_blast_fn_large)
        num_seqs_large, pileup_fn_large = filter_blast_sequences(
            protein_id=self.large_structure_id, filter_path=self.filtered_blast_path,
            blast_fn=self.expected_blast_fn, query_seq=self.large_query_seq_uniprot)
        self.assertGreaterEqual(num_seqs_large, 0)
        self.assertLessEqual(num_seqs_large, self.max_target_seqs + 1)
        self.assertEqual(pileup_fn_large, self.expected_filtered_blast_fn_large)

    def test5e_filtering_and_alignment_filter_blast_sequences_loading(self):
        if os.path.isdir(self.filtered_blast_path):
            rmtree(self.filtered_blast_path)
        if not os.path.isdir(self.blast_path):
            self.test5b_filtering_and_alignment_blast_query_sequence_multi_thread()
        if os.path.isfile(self.expected_filtered_blast_fn_small):
            os.remove(self.expected_filtered_blast_fn_small)
        num_seqs_small1, pileup_fn_small1 = filter_blast_sequences(
            protein_id=self.small_structure_id, filter_path=self.filtered_blast_path,
            blast_fn=self.expected_blast_fn, query_seq=self.small_query_seq_uniprot)
        num_seqs_small2, pileup_fn_small2 = filter_blast_sequences(
            protein_id=self.small_structure_id, filter_path=self.filtered_blast_path,
            blast_fn=self.expected_blast_fn, query_seq=self.small_query_seq_uniprot)
        self.assertEqual(num_seqs_small1, num_seqs_small2)
        self.assertEqual(pileup_fn_small1, pileup_fn_small2)
        if os.path.isfile(self.expected_filtered_blast_fn_large):
            os.remove(self.expected_filtered_blast_fn_large)
        num_seqs_large1, pileup_fn_large1 = filter_blast_sequences(
            protein_id=self.large_structure_id, filter_path=self.filtered_blast_path,
            blast_fn=self.expected_blast_fn, query_seq=self.large_query_seq_uniprot)
        num_seqs_large2, pileup_fn_large2 = filter_blast_sequences(
            protein_id=self.large_structure_id, filter_path=self.filtered_blast_path,
            blast_fn=self.expected_blast_fn, query_seq=self.large_query_seq_uniprot)
        self.assertGreaterEqual(num_seqs_large1, num_seqs_large2)
        self.assertEqual(pileup_fn_large1, pileup_fn_large2)

    def test5f_filtering_and_alignment_align_sequences(self):
        if os.path.isdir(self.alignment_path):
            rmtree(self.alignment_path)
        if not os.path.isfile(self.expected_filtered_blast_fn_small):
            self.test5d_filtering_and_alignment_filter_blast_sequences()
        if os.path.isfile(self.expected_msf_fn_small):
            os.remove(self.expected_msf_fn_small)
        if os.path.isfile(self.expected_fa_fn_small):
            os.remove(self.expected_fa_fn_small)
        msf_fn_small, fa_fn_small = align_sequences(protein_id=self.small_structure_id,
                                                    alignment_path=self.alignment_path,
                                                    pileup_fn=self.expected_filtered_blast_fn_small)
        self.assertEqual(fa_fn_small, self.expected_fa_fn_small)
        self.assertTrue(os.path.isfile(self.expected_fa_fn_small))
        self.assertEqual(msf_fn_small, self.expected_msf_fn_small)
        self.assertTrue(os.path.isfile(self.expected_msf_fn_small))
        if os.path.isfile(self.expected_msf_fn_large):
            os.remove(self.expected_msf_fn_large)
        if os.path.isfile(self.expected_fa_fn_large):
            os.remove(self.expected_fa_fn_large)
        msf_fn_large, fa_fn_large = align_sequences(protein_id=self.large_structure_id,
                                                    alignment_path=self.alignment_path,
                                                    pileup_fn=self.expected_filtered_blast_fn_large)
        self.assertTrue(os.path.isfile(self.expected_msf_fn_large))
        self.assertEqual(msf_fn_large, self.expected_msf_fn_large)
        self.assertTrue(os.path.isfile(self.expected_fa_fn_large))
        self.assertEqual(fa_fn_large, self.expected_fa_fn_large)

    def test5g_filtering_and_alignment_align_sequences_fasta_only(self):
        if os.path.isdir(self.alignment_path):
            rmtree(self.alignment_path)
        if not os.path.isdir(self.filtered_blast_path):
            self.test5d_filtering_and_alignment_filter_blast_sequences()
        if os.path.isfile(self.expected_msf_fn_small):
            os.remove(self.expected_msf_fn_small)
        if os.path.isfile(self.expected_fa_fn_small):
            os.remove(self.expected_fa_fn_small)
        msf_fn_small, fa_fn_small = align_sequences(protein_id=self.small_structure_id,
                                                    alignment_path=self.alignment_path,
                                                    pileup_fn=self.expected_filtered_blast_fn_small, msf=False)
        self.assertIsNone(msf_fn_small)
        self.assertTrue(os.path.isfile(self.expected_fa_fn_small))
        self.assertEqual(fa_fn_small, self.expected_fa_fn_small)
        if os.path.isfile(self.expected_msf_fn_large):
            os.remove(self.expected_msf_fn_large)
        if os.path.isfile(self.expected_fa_fn_large):
            os.remove(self.expected_fa_fn_large)
        msf_fn_large, fa_fn_large = align_sequences(protein_id=self.large_structure_id,
                                                    alignment_path=self.alignment_path,
                                                    pileup_fn=self.expected_filtered_blast_fn_large, msf=False)
        self.assertTrue(os.path.isfile(self.expected_fa_fn_large))
        self.assertEqual(fa_fn_large, self.expected_fa_fn_large)
        self.assertIsNone(msf_fn_large)

    def test5h_filtering_and_alignment_align_sequences_msf_only(self):
        if os.path.isdir(self.alignment_path):
            rmtree(self.alignment_path)
        if not os.path.isdir(self.filtered_blast_path):
            self.test5d_filtering_and_alignment_filter_blast_sequences()
        if os.path.isfile(self.expected_msf_fn_small):
            os.remove(self.expected_msf_fn_small)
        if os.path.isfile(self.expected_fa_fn_small):
            os.remove(self.expected_fa_fn_small)
        msf_fn_small, fa_fn_small = align_sequences(protein_id=self.small_structure_id,
                                                    alignment_path=self.alignment_path,
                                                    pileup_fn=self.expected_filtered_blast_fn_small, fasta=False)
        self.assertIsNone(fa_fn_small)
        self.assertTrue(os.path.isfile(self.expected_msf_fn_small))
        self.assertEqual(msf_fn_small, self.expected_msf_fn_small)
        if os.path.isfile(self.expected_msf_fn_large):
            os.remove(self.expected_msf_fn_large)
        if os.path.isfile(self.expected_fa_fn_large):
            os.remove(self.expected_fa_fn_large)
        msf_fn_large, fa_fn_large = align_sequences(protein_id=self.large_structure_id,
                                                    alignment_path=self.alignment_path,
                                                    pileup_fn=self.expected_filtered_blast_fn_large, fasta=False)
        self.assertTrue(os.path.isfile(self.expected_msf_fn_large))
        self.assertEqual(msf_fn_large, self.expected_msf_fn_large)
        self.assertIsNone(fa_fn_large)

    def test5i_filtering_and_alignment_identity_filter(self):
        if os.path.isdir(self.filtered_alignment_path):
            rmtree(self.filtered_alignment_path)
        if (not os.path.isdir(self.alignment_path) or not os.path.isfile(self.expected_fa_fn_small) or
                not os.path.isfile(self.expected_fa_fn_large)):
            self.test5f_filtering_and_alignment_align_sequences()
        if not os.path.isdir(self.filtered_blast_path):
            self.test5d_filtering_and_alignment_filter_blast_sequences()
        max_count_small, _ = filter_blast_sequences(
            protein_id=self.small_structure_id, filter_path=self.filtered_blast_path,
            blast_fn=self.expected_blast_fn_small, query_seq=self.small_query_seq_uniprot)
        count_small, filtered_aln_fn_small = identity_filter(
            protein_id=self.small_structure_id, filter_path=self.filtered_alignment_path,
            alignment_fn=self.expected_fa_fn_small, max_identity=0.98)
        self.assertEqual(filtered_aln_fn_small, self.expected_filtered_aln_fn_small)
        self.assertTrue(os.path.isfile(filtered_aln_fn_small))
        self.assertLessEqual(count_small, max_count_small)
        max_count_large, _ = filter_blast_sequences(
            protein_id=self.large_structure_id, filter_path=self.filtered_blast_path,
            blast_fn=self.expected_blast_fn_large, query_seq=self.large_query_seq_uniprot)
        count_large, filtered_aln_fn_large = identity_filter(
            protein_id=self.large_structure_id,  filter_path=self.filtered_alignment_path,
            alignment_fn=self.expected_fa_fn_large, max_identity=0.98)
        self.assertEqual(filtered_aln_fn_large, self.expected_filtered_aln_fn_large)
        self.assertTrue(os.path.isfile(filtered_aln_fn_small))
        self.assertLessEqual(count_large, max_count_large)

    def test5j_filtering_and_alignment(self):
        if not os.path.isfile(self.expected_blast_fn):
            self.test5b_filtering_and_alignment_blast_query_sequence_multi_thread()
        init_filtering_and_alignment_pool(max_target_seqs=self.max_target_seqs, e_value_threshold=self.max_e_value,
                                          database='customuniref90.fasta', remote=False, min_fraction=0.7,
                                          min_identity=0.40, max_identity=0.98, msf=True, fasta=True,
                                          blast_fn=self.expected_blast_fn, filtered_blast_path=self.filtered_blast_path,
                                          aln_path=self.alignment_path, filtered_aln_path=self.filtered_alignment_path,
                                          final_aln_path=self.final_alignment_path, verbose=False)
        start = time()
        p_id, p_data, p_time = filtering_and_alignment(in_tup=(self.small_structure_id, self.small_query_seq_uniprot))
        end = time()
        self.assertEqual(p_id, self.small_structure_id)
        self.assertLessEqual(p_data['Filter_Count'], self.max_target_seqs + 1)
        self.assertEqual(p_data['Filtered_BLAST'], self.expected_filtered_blast_fn_small)
        self.assertEqual(p_data['MSF_Aln'], self.expected_msf_fn_small)
        self.assertEqual(p_data['FA_Aln'], self.expected_fa_fn_small)
        self.assertLessEqual(p_data['Final_Count'], p_data['Filter_Count'])
        self.assertEqual(p_data['Filtered_Alignment'], self.expected_filtered_aln_fn_small)
        self.assertEqual(p_data['Final_MSF_Aln'], self.expected_final_msf_fn_small)
        self.assertEqual(p_data['Final_FA_Aln'], self.expected_final_fa_fn_small)
        outer_time = (end - start) / 60.0
        self.assertLessEqual(p_time, outer_time)

    def test5k_filtering_and_alignment(self):
        if not os.path.isfile(self.expected_blast_fn):
            self.test5b_filtering_and_alignment_blast_query_sequence_multi_thread()
        init_filtering_and_alignment_pool(max_target_seqs=self.max_target_seqs, e_value_threshold=self.max_e_value,
                                          database='customuniref90.fasta', remote=False, min_fraction=0.7,
                                          min_identity=0.40, max_identity=0.98, msf=True, fasta=True,
                                          blast_fn=self.expected_blast_fn, filtered_blast_path=self.filtered_blast_path,
                                          aln_path=self.alignment_path, filtered_aln_path=self.filtered_alignment_path,
                                          final_aln_path=self.final_alignment_path, verbose=False)
        start = time()
        p_id, p_data, p_time = filtering_and_alignment(in_tup=(self.large_structure_id, self.large_query_seq_uniprot))
        end = time()
        self.assertEqual(p_id, self.large_structure_id)
        self.assertLessEqual(p_data['Filter_Count'], self.max_target_seqs + 1)
        self.assertEqual(p_data['Filtered_BLAST'], self.expected_filtered_blast_fn_large)
        self.assertEqual(p_data['MSF_Aln'], self.expected_msf_fn_large)
        self.assertEqual(p_data['FA_Aln'], self.expected_fa_fn_large)
        self.assertLessEqual(p_data['Final_Count'], p_data['Filter_Count'])
        self.assertEqual(p_data['Filtered_Alignment'], self.expected_filtered_aln_fn_large)
        self.assertEqual(p_data['Final_MSF_Aln'], self.expected_final_msf_fn_large)
        self.assertEqual(p_data['Final_FA_Aln'], self.expected_final_fa_fn_large)
        outer_time = (end - start) / 60.0
        self.assertLessEqual(p_time, outer_time)


if __name__ == '__main__':
    unittest.main()
