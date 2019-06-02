"""
Created on May 28, 2019

@author: Daniel Konecki
"""
import os
from shutil import rmtree
from unittest import TestCase
from DataSetGenerator import DataSetGenerator


class TestDataSetGenerator(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.input_path = os.path.join(os.path.abspath('../Test/'), 'Input')
        cls.protein_list_path = os.path.join(cls.input_path, 'ProteinLists')
        if not os.path.isdir(cls.protein_list_path):
            os.makedirs(cls.protein_list_path)
        cls.small_structure_id = '7hvp'
        cls.large_structure_id = '2zxe'
        cls.protein_list_fn = os.path.join(cls.protein_list_path, 'Test_Set.txt')
        structure_ids = [cls.small_structure_id, cls.large_structure_id]
        with open(cls.protein_list_fn, 'wb') as test_list_handle:
            for structure_id in structure_ids:
                test_list_handle.write('{}\n'.format(structure_id))

    @classmethod
    def tearDownClass(cls):
        # rmtree(cls.input_path)
        del cls.protein_list_fn
        del cls.large_structure_id
        del cls.small_structure_id
        del cls.protein_list_path
        del cls.input_path

    # def tearDown(self):
    #     for curr_fn in os.listdir(self.input_path):
    #         curr_dir = os.path.join(self.input_path, curr_fn)
    #         if os.path.isdir(curr_dir) and curr_fn != 'ProteinLists':
    #             rmtree(curr_dir)

    def test_init(self):
        test_generator = DataSetGenerator(protein_list='Test_Set.txt', input_path=self.input_path)
        self.assertTrue(test_generator.input_path == self.input_path)
        self.assertTrue(test_generator.file_name == os.path.basename(self.protein_list_fn))
        self.assertEqual(len(test_generator.protein_data), 2)
        self.assertTrue(self.small_structure_id in test_generator.protein_data)
        self.assertTrue(self.large_structure_id in test_generator.protein_data)

    def test__download_pdb(self):
        pdb_path = os.path.join(self.input_path, 'PDB')
        if os.path.isdir(pdb_path):
            rmtree(pdb_path)
        test_generator = DataSetGenerator(protein_list='Test_Set.txt', input_path=self.input_path)
        pdb_fn_small = test_generator._download_pdb(protein_id=self.small_structure_id)
        self.assertTrue(os.path.isdir(pdb_path))
        self.assertTrue('PDB_Path' in test_generator.protein_data[self.small_structure_id])
        expected_fn_small = os.path.join(pdb_path, '{}'.format(self.small_structure_id[1:3]),
                                         'pdb{}.ent'.format(self.small_structure_id))
        self.assertTrue(test_generator.protein_data[self.small_structure_id]['PDB_Path'] == expected_fn_small)
        self.assertTrue(pdb_fn_small == expected_fn_small)
        self.assertTrue(os.path.isfile(expected_fn_small))
        pdb_fn_large = test_generator._download_pdb(protein_id=self.large_structure_id)
        self.assertTrue('PDB_Path' in test_generator.protein_data[self.large_structure_id])
        expected_fn_large = os.path.join(pdb_path, '{}'.format(self.large_structure_id[1:3]),
                                         'pdb{}.ent'.format(self.large_structure_id))
        self.assertTrue(test_generator.protein_data[self.large_structure_id]['PDB_Path'] == expected_fn_large)
        self.assertTrue(pdb_fn_large == expected_fn_large)
        self.assertTrue(os.path.isfile(expected_fn_large))

    def test__parse_query_sequence(self):
        sequence_path = os.path.join(self.input_path, 'Sequences')
        if os.path.isdir(sequence_path):
            rmtree(sequence_path)
        test_generator = DataSetGenerator(protein_list='Test_Set.txt', input_path=self.input_path)
        with self.assertRaises(KeyError):
            test_generator._parse_query_sequence(protein_id=self.small_structure_id)
        test_generator._download_pdb(protein_id=self.small_structure_id)
        seq_small, len_small, seq_fn_small = test_generator._parse_query_sequence(protein_id=self.small_structure_id)
        self.assertTrue(os.path.isdir(sequence_path))
        self.assertEqual('PQITLWQRPLVTIRIGGQLKEALLDTGADDTVLEEMNLPGKWKPKMIGGIGGFIKVRQYDQIPVEIGHKAIGTVLVGPTPVNIIGRNLLTQIG'
                         'TLNF', str(seq_small.seq))
        self.assertEqual(str(test_generator.protein_data[self.small_structure_id]['Query_Sequence'].seq),
                         str(seq_small.seq))
        self.assertEqual(len_small, 97)
        self.assertEqual(len(seq_small), len_small)
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['Sequence_Length'], len_small)
        expexcted_fn_small = os.path.join(sequence_path, '{}.fasta'.format(self.small_structure_id))
        self.assertEqual(seq_fn_small, expexcted_fn_small)
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['Fasta_File'], expexcted_fn_small)
        with self.assertRaises(KeyError):
            test_generator._parse_query_sequence(protein_id=self.large_structure_id)
        test_generator._download_pdb(protein_id=self.large_structure_id)
        seq_large, len_large, seq_fn_large = test_generator._parse_query_sequence(protein_id=self.large_structure_id)
        self.assertTrue(os.path.isdir(sequence_path))
        self.assertEqual('LDELKKEVSMDDHKLSLDELHNKYGTDLTRGLTNARAKEILARDGPNSLTPPPTTPEWIKFCRQLFGGFSILLWIGAILCFLAYGIQAATEDE'
                         'PANDNLYLGVVLSTVVIVTGCFSYYQEAKSSRIMDSFKNMVPQQALVIRDGEKSTINAEFVVAGDLVEVKGGDRIPADLRIISAHGCKVDNSS'
                         'LTGESEPQTRSPEFSSENPLETRNIAFFSTNCVEGTARGVVVYTGDRTVMGRIATLASGLEVGRTPIAIEIEHFIHIITGVAVFLGVSFFILS'
                         'LILGYSWLEAVIFLIGIIVANVPEGLLATVTVCLTLTAKRMARKNCLVKNLEAVETLGSTSTICSDKTGTLTQNRMTVAHMWFDNQIHEADTT'
                         'ENQSGAAFDKTSATWSALSRIAALCNRAVFQAGQDNVPILKRSVAGDASESALLKCIELCCGSVQGMRDRNPKIVEIPFNSTNKYQLSIHENE'
                         'KSSESRYLLVMKGAPERILDRCSTILLNGAEEPLKEDMKEAFQNAYLELGGLGERVLGFCHFALPEDKYNEGYPFDADEPNFPTTDLCFVGLM'
                         'AMIDPPRAAVPDAVGKCRSAGIKVIMVTGDHPITAKAIAKGVGIISEGNETIEDIAARLNIPIGQVNPRDAKACVVHGSDLKDLSTEVLDDIL'
                         'HYHTEIVFARTSPQQKLIIVEGCQRQGAIVAVTGDGVNDSPALKKADIGVAMGISGSDVSKQAADMILLDDNFASIVTGVEEGRLIFDNLKKS'
                         'IAYTLTSNIPEITPFLVFIIGNVPLPLGTVTILCIDLGTDMVPAISLAYEQAESDIMKRQPRNPKTDKLVNERLISMAYGQIGMIQALGGFFS'
                         'YFVILAENGFLPMDLIGKRVRWDDRWISDVEDSFGQQWTYEQRKIVEFTCHTSFFISIVVVQWADLIICKTRRNSIFQQGMKNKILIFGLFEE'
                         'TALAAFLSYCPGTDVALRMYPLKPSWWFCAFPYSLIIFLYDEMRRFIIRRSPGGWVEQETYY', str(seq_large.seq))
        self.assertEqual(str(test_generator.protein_data[self.large_structure_id]['Query_Sequence'].seq),
                         str(seq_large.seq))
        self.assertEqual(len_large, 992)
        self.assertEqual(len(seq_large), len_large)
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['Sequence_Length'], len_large)
        expexcted_fn_large = os.path.join(sequence_path, '{}.fasta'.format(self.large_structure_id))
        self.assertEqual(seq_fn_large, expexcted_fn_large)
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['Fasta_File'], expexcted_fn_large)

    def test__blast_query_sequence_single_thread(self):
        blast_path = os.path.join(self.input_path, 'BLAST')
        if os.path.isdir(blast_path):
            rmtree(blast_path)
        test_generator = DataSetGenerator(protein_list='Test_Set.txt', input_path=self.input_path)
        with self.assertRaises(KeyError):
            test_generator._blast_query_sequence(protein_id=self.small_structure_id)
        test_generator._download_pdb(protein_id=self.small_structure_id)
        test_generator._parse_query_sequence(protein_id=self.small_structure_id)
        blast_fn_small = test_generator._blast_query_sequence(protein_id=self.small_structure_id, num_threads=1,
                                                              max_target_seqs=500)
        self.assertTrue(os.path.isdir(blast_path))
        expected_fn_small = os.path.join(blast_path, '{}.xml'.format(self.small_structure_id))
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['BLAST_File'], expected_fn_small)
        self.assertEqual(blast_fn_small, expected_fn_small)
        self.assertTrue(os.path.isfile(blast_fn_small))
        with self.assertRaises(KeyError):
            test_generator._blast_query_sequence(protein_id=self.large_structure_id)
        test_generator._download_pdb(protein_id=self.large_structure_id)
        test_generator._parse_query_sequence(protein_id=self.large_structure_id)
        blast_fn_large = test_generator._blast_query_sequence(protein_id=self.large_structure_id, num_threads=1,
                                                              max_target_seqs=500)
        self.assertTrue(os.path.isdir(blast_path))
        expected_fn_large = os.path.join(blast_path, '{}.xml'.format(self.large_structure_id))
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['BLAST_File'], expected_fn_large)
        self.assertEqual(blast_fn_large, expected_fn_large)
        self.assertTrue(os.path.isfile(blast_fn_large))

    def test__blast_query_sequence_multi_thread(self):
        blast_path = os.path.join(self.input_path, 'BLAST')
        if os.path.isdir(blast_path):
            rmtree(blast_path)
        test_generator = DataSetGenerator(protein_list='Test_Set.txt', input_path=self.input_path)
        with self.assertRaises(KeyError):
            test_generator._blast_query_sequence(protein_id=self.small_structure_id)
        test_generator._download_pdb(protein_id=self.small_structure_id)
        test_generator._parse_query_sequence(protein_id=self.small_structure_id)
        expected_fn_small = os.path.join(blast_path, '{}.xml'.format(self.small_structure_id))
        if os.path.isfile(expected_fn_small):
            os.remove(expected_fn_small)
        blast_fn_small = test_generator._blast_query_sequence(protein_id=self.small_structure_id, num_threads=10,
                                                              max_target_seqs=500)
        self.assertTrue(os.path.isdir(blast_path))
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['BLAST_File'], expected_fn_small)
        self.assertEqual(blast_fn_small, expected_fn_small)
        self.assertTrue(os.path.isfile(blast_fn_small))
        with self.assertRaises(KeyError):
            test_generator._blast_query_sequence(protein_id=self.large_structure_id)
        test_generator._download_pdb(protein_id=self.large_structure_id)
        test_generator._parse_query_sequence(protein_id=self.large_structure_id)
        expected_fn_large = os.path.join(blast_path, '{}.xml'.format(self.large_structure_id))
        if os.path.isfile(expected_fn_large):
            os.remove(expected_fn_large)
        blast_fn_large = test_generator._blast_query_sequence(protein_id=self.large_structure_id, num_threads=10,
                                                              max_target_seqs=500)
        self.assertTrue(os.path.isdir(blast_path))
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['BLAST_File'], expected_fn_large)
        self.assertEqual(blast_fn_large, expected_fn_large)
        self.assertTrue(os.path.isfile(blast_fn_large))

    def test__restrict_sequences(self):
        pileup_path = os.path.join(self.input_path, 'Pileups')
        if os.path.isdir(pileup_path):
            rmtree(pileup_path)
        test_generator = DataSetGenerator(protein_list='Test_Set.txt', input_path=self.input_path)
        with self.assertRaises(KeyError):
            test_generator._restrict_sequences(protein_id=self.small_structure_id)
        test_generator._download_pdb(protein_id=self.small_structure_id)
        test_generator._parse_query_sequence(protein_id=self.small_structure_id)
        test_generator._blast_query_sequence(protein_id=self.small_structure_id, num_threads=10, max_target_seqs=2000)
        expected_fn_small = os.path.join(pileup_path, '{}.fasta'.format(self.small_structure_id))
        min_id_small, num_seqs_small, pileup_fn_small = test_generator._restrict_sequences(
            protein_id=self.small_structure_id)
        if num_seqs_small >= 125:
            self.assertLessEqual(0.95, min_id_small)
            self.assertGreaterEqual(min_id_small, 0.30)
            self.assertEqual(pileup_fn_small, expected_fn_small)
            self.assertEqual(test_generator.protein_data[self.small_structure_id]['Pileup_File'], expected_fn_small)
        else:
            self.assertGreaterEqual(min_id_small, 0.30)
            self.assertIsNone(pileup_fn_small)
            self.assertIsNone(test_generator.protein_data[self.small_structure_id]['Pileup_File'])
        with self.assertRaises(KeyError):
            test_generator._restrict_sequences(protein_id=self.large_structure_id)
        test_generator._download_pdb(protein_id=self.large_structure_id)
        test_generator._parse_query_sequence(protein_id=self.large_structure_id)
        test_generator._blast_query_sequence(protein_id=self.large_structure_id, num_threads=10, max_target_seqs=2000)
        expected_fn_large = os.path.join(pileup_path, '{}.fasta'.format(self.large_structure_id))
        min_id_large, num_seqs_large, pileup_fn_large = test_generator._restrict_sequences(
            protein_id=self.large_structure_id)
        if num_seqs_large >= 125:
            self.assertLessEqual(0.95, min_id_large)
            self.assertGreaterEqual(min_id_large, 0.30)
            self.assertEqual(pileup_fn_large, expected_fn_large)
            self.assertEqual(test_generator.protein_data[self.large_structure_id]['Pileup_File'], expected_fn_large)
        else:
            self.assertGreaterEqual(min_id_large, 0.30)
            self.assertIsNone(pileup_fn_large)
            self.assertIsNone(test_generator.protein_data[self.large_structure_id]['Pileup_File'])

    def test__restrict_sequences_loading(self):
        pileup_path = os.path.join(self.input_path, 'Pileups')
        if os.path.isdir(pileup_path):
            rmtree(pileup_path)
        test_generator = DataSetGenerator(protein_list='Test_Set.txt', input_path=self.input_path)
        test_generator._download_pdb(protein_id=self.small_structure_id)
        test_generator._parse_query_sequence(protein_id=self.small_structure_id)
        test_generator._blast_query_sequence(protein_id=self.small_structure_id, num_threads=10, max_target_seqs=2000)
        min_id_small1, num_seqs_small1, pileup_fn_small1 = test_generator._restrict_sequences(
            protein_id=self.small_structure_id)
        min_id_small2, num_seqs_small2, pileup_fn_small2 = test_generator._restrict_sequences(
            protein_id=self.small_structure_id)
        self.assertEqual(pileup_fn_small1, pileup_fn_small2)
        self.assertEqual(num_seqs_small1, num_seqs_small2)
        self.assertEqual(min_id_small1, min_id_small2)
        test_generator._download_pdb(protein_id=self.large_structure_id)
        test_generator._parse_query_sequence(protein_id=self.large_structure_id)
        test_generator._blast_query_sequence(protein_id=self.large_structure_id, num_threads=10, max_target_seqs=2000)
        min_id_large1, num_seqs_large1, pileup_fn_large1 = test_generator._restrict_sequences(
            protein_id=self.large_structure_id)
        min_id_large2, num_seqs_large2, pileup_fn_large2 = test_generator._restrict_sequences(
            protein_id=self.large_structure_id)
        self.assertEqual(pileup_fn_large1, pileup_fn_large2)
        self.assertEqual(num_seqs_large1, num_seqs_large2)
        self.assertEqual(min_id_large1, min_id_large2)

    def test__restrict_sequences_ignore_filter_size(self):
        pileup_path = os.path.join(self.input_path, 'Pileups')
        pileup_path = os.path.join(self.input_path, 'Pileups')
        if os.path.isdir(pileup_path):
            rmtree(pileup_path)
        test_generator = DataSetGenerator(protein_list='Test_Set.txt', input_path=self.input_path)
        test_generator._download_pdb(protein_id=self.small_structure_id)
        test_generator._parse_query_sequence(protein_id=self.small_structure_id)
        test_generator._blast_query_sequence(protein_id=self.small_structure_id, num_threads=10, max_target_seqs=2000)
        min_id_small1, num_seqs_small1, pileup_fn_small1 = test_generator._restrict_sequences(
            protein_id=self.small_structure_id)
        if pileup_fn_small1 is None:
            expected_fn_small = os.path.join(pileup_path, '{}.fasta'.format(self.small_structure_id))
            min_id_small2, num_seqs_small2, pileup_fn_small2 = test_generator._restrict_sequences(
                protein_id=self.small_structure_id, ignore_filter_size=True)
            self.assertEqual(min_id_small1, min_id_small2)
            self.assertEqual(num_seqs_small1, num_seqs_small2)
            self.assertEqual(pileup_fn_small2, expected_fn_small)
        test_generator._download_pdb(protein_id=self.large_structure_id)
        test_generator._parse_query_sequence(protein_id=self.large_structure_id)
        test_generator._blast_query_sequence(protein_id=self.large_structure_id, num_threads=10, max_target_seqs=2000)
        min_id_large1, num_seqs_large1, pileup_fn_large1 = test_generator._restrict_sequences(
            protein_id=self.large_structure_id)
        if pileup_fn_large1 is None:
            expected_fn_large = os.path.join(pileup_path, '{}.fasta'.format(self.large_structure_id))
            min_id_large2, num_seqs_large2, pileup_fn_large2 = test_generator._restrict_sequences(
                protein_id=self.large_structure_id, ignore_filter_size=True)
            self.assertEqual(min_id_large1, min_id_large2)
            self.assertEqual(num_seqs_large1, num_seqs_large2)
            self.assertEqual(pileup_fn_large2, expected_fn_large)

    def test_align_sequences(self):
        alignment_path = os.path.join(self.input_path, 'Alignments')
        if os.path.isdir(alignment_path):
            rmtree(alignment_path)
        test_generator = DataSetGenerator(protein_list='Test_Set.txt', input_path=self.input_path)
        # with self.assertRaises(KeyError):
        #     test_generator._align_sequences(protein_id=self.small_structure_id)
        test_generator._download_pdb(protein_id=self.small_structure_id)
        test_generator._parse_query_sequence(protein_id=self.small_structure_id)
        test_generator._blast_query_sequence(protein_id=self.small_structure_id, num_threads=10, max_target_seqs=2000)
        test_generator._restrict_sequences(protein_id=self.small_structure_id)
        expected_msf_fn_small = os.path.join(alignment_path, '{}.msf'.format(self.small_structure_id))
        expected_fa_fn_small = os.path.join(alignment_path, '{}.fasta'.format(self.small_structure_id))
        msf_fn_small, fa_fn_small = test_generator._align_sequences(protein_id=self.small_structure_id)
        if test_generator.protein_data[self.small_structure_id]['Pileup_File']:
            self.assertTrue(os.path.isfile(expected_msf_fn_small))
            self.assertEqual(msf_fn_small, expected_msf_fn_small)
            self.assertEqual(test_generator.protein_data[self.small_structure_id]['MSF_File'], expected_msf_fn_small)
            self.assertTrue(os.path.isfile(expected_fa_fn_small))
            self.assertEqual(fa_fn_small, expected_fa_fn_small)
            self.assertEqual(test_generator.protein_data[self.small_structure_id]['FA_File'], expected_fa_fn_small)
        else:
            self.assertFalse(os.path.isfile(expected_msf_fn_small))
            self.assertIsNone(test_generator.protein_data[self.small_structure_id]['MSF_File'])
            self.assertFalse(os.path.isfile(expected_fa_fn_small))
            self.assertIsNone(test_generator.protein_data[self.small_structure_id]['FA_File'])
        # with self.assertRaises(KeyError):
        #     test_generator._restrict_sequences(protein_id=self.large_structure_id)
        test_generator._download_pdb(protein_id=self.large_structure_id)
        test_generator._parse_query_sequence(protein_id=self.large_structure_id)
        test_generator._blast_query_sequence(protein_id=self.large_structure_id, num_threads=10, max_target_seqs=2000)
        test_generator._restrict_sequences(protein_id=self.large_structure_id)
        expected_msf_fn_large = os.path.join(alignment_path, '{}.msf'.format(self.large_structure_id))
        expected_fa_fn_large = os.path.join(alignment_path, '{}.fasta'.format(self.large_structure_id))
        msf_fn_large, fa_fn_large = test_generator._align_sequences(protein_id=self.large_structure_id)
        if test_generator.protein_data[self.large_structure_id]['Pileup_File']:
            self.assertTrue(os.path.isfile(expected_msf_fn_large))
            self.assertEqual(msf_fn_large, expected_msf_fn_large)
            self.assertEqual(test_generator.protein_data[self.large_structure_id]['MSF_File'], expected_msf_fn_large)
            self.assertTrue(os.path.isfile(expected_fa_fn_large))
            self.assertEqual(fa_fn_large, expected_fa_fn_large)
            self.assertEqual(test_generator.protein_data[self.large_structure_id]['FA_File'], expected_fa_fn_large)
        else:
            self.assertFalse(os.path.isfile(expected_msf_fn_large))
            self.assertIsNone(test_generator.protein_data[self.large_structure_id]['MSF_File'])
            self.assertFalse(os.path.isfile(expected_fa_fn_large))
            self.assertIsNone(test_generator.protein_data[self.large_structure_id]['FA_File'])

    def test_align_sequences_msf_only(self):
        alignment_path = os.path.join(self.input_path, 'Alignments')
        if os.path.isdir(alignment_path):
            rmtree(alignment_path)
        test_generator = DataSetGenerator(protein_list='Test_Set.txt', input_path=self.input_path)
        # with self.assertRaises(KeyError):
        #     test_generator._align_sequences(protein_id=self.small_structure_id)
        test_generator._download_pdb(protein_id=self.small_structure_id)
        test_generator._parse_query_sequence(protein_id=self.small_structure_id)
        test_generator._blast_query_sequence(protein_id=self.small_structure_id, num_threads=10, max_target_seqs=2000)
        test_generator._restrict_sequences(protein_id=self.small_structure_id)
        expected_msf_fn_small = os.path.join(alignment_path, '{}.msf'.format(self.small_structure_id))
        expected_fa_fn_small = os.path.join(alignment_path, '{}.fasta'.format(self.small_structure_id))
        msf_fn_small, fa_fn_small = test_generator._align_sequences(protein_id=self.small_structure_id, fasta=False)
        if test_generator.protein_data[self.small_structure_id]['Pileup_File']:
            self.assertTrue(os.path.isfile(expected_msf_fn_small))
            self.assertEqual(msf_fn_small, expected_msf_fn_small)
        else:
            self.assertFalse(os.path.isfile(expected_msf_fn_small))
            self.assertIsNone(msf_fn_small)
        self.assertFalse(os.path.isfile(expected_fa_fn_small))
        self.assertIsNone(fa_fn_small)
        # with self.assertRaises(KeyError):
        #     test_generator._restrict_sequences(protein_id=self.large_structure_id)
        test_generator._download_pdb(protein_id=self.large_structure_id)
        test_generator._parse_query_sequence(protein_id=self.large_structure_id)
        test_generator._blast_query_sequence(protein_id=self.large_structure_id, num_threads=10, max_target_seqs=2000)
        test_generator._restrict_sequences(protein_id=self.large_structure_id)
        expected_msf_fn_large = os.path.join(alignment_path, '{}.msf'.format(self.large_structure_id))
        expected_fa_fn_large = os.path.join(alignment_path, '{}.fasta'.format(self.large_structure_id))
        msf_fn_large, fa_fn_large = test_generator._align_sequences(protein_id=self.large_structure_id, fasta=False)
        if test_generator.protein_data[self.large_structure_id]['Pileup_File']:
            self.assertTrue(os.path.isfile(expected_msf_fn_large))
            self.assertEqual(msf_fn_large, expected_msf_fn_large)
        else:
            self.assertFalse(os.path.isfile(expected_msf_fn_large))
            self.assertIsNone(msf_fn_large)
        self.assertFalse(os.path.isfile(expected_fa_fn_large))
        self.assertIsNone(fa_fn_large)

    def test_align_sequences_fasta_only(self):
        alignment_path = os.path.join(self.input_path, 'Alignments')
        if os.path.isdir(alignment_path):
            rmtree(alignment_path)
        test_generator = DataSetGenerator(protein_list='Test_Set.txt', input_path=self.input_path)
        # with self.assertRaises(KeyError):
        #     test_generator._align_sequences(protein_id=self.small_structure_id)
        test_generator._download_pdb(protein_id=self.small_structure_id)
        test_generator._parse_query_sequence(protein_id=self.small_structure_id)
        test_generator._blast_query_sequence(protein_id=self.small_structure_id, num_threads=10, max_target_seqs=2000)
        test_generator._restrict_sequences(protein_id=self.small_structure_id)
        expected_msf_fn_small = os.path.join(alignment_path, '{}.msf'.format(self.small_structure_id))
        expected_fa_fn_small = os.path.join(alignment_path, '{}.fasta'.format(self.small_structure_id))
        msf_fn_small, fa_fn_small = test_generator._align_sequences(protein_id=self.small_structure_id, msf=False)
        if test_generator.protein_data[self.small_structure_id]['Pileup_File']:
            self.assertTrue(os.path.isfile(expected_fa_fn_small))
            self.assertEqual(fa_fn_small, expected_fa_fn_small)
        else:
            self.assertFalse(os.path.isfile(expected_fa_fn_small))
            self.assertIsNone(fa_fn_small)
        self.assertFalse(os.path.isfile(expected_msf_fn_small))
        self.assertIsNone(msf_fn_small)
        # with self.assertRaises(KeyError):
        #     test_generator._restrict_sequences(protein_id=self.large_structure_id)
        test_generator._download_pdb(protein_id=self.large_structure_id)
        test_generator._parse_query_sequence(protein_id=self.large_structure_id)
        test_generator._blast_query_sequence(protein_id=self.large_structure_id, num_threads=10, max_target_seqs=2000)
        test_generator._restrict_sequences(protein_id=self.large_structure_id)
        expected_msf_fn_large = os.path.join(alignment_path, '{}.msf'.format(self.large_structure_id))
        expected_fa_fn_large = os.path.join(alignment_path, '{}.fasta'.format(self.large_structure_id))
        msf_fn_large, fa_fn_large = test_generator._align_sequences(protein_id=self.large_structure_id, msf=False)
        if test_generator.protein_data[self.large_structure_id]['Pileup_File']:
            self.assertTrue(os.path.isfile(expected_fa_fn_large))
            self.assertEqual(fa_fn_large, expected_fa_fn_large)
        else:
            self.assertFalse(os.path.isfile(expected_fa_fn_large))
            self.assertIsNone(fa_fn_large)
        self.assertFalse(os.path.isfile(expected_msf_fn_large))
        self.assertIsNone(msf_fn_large)

    def test_build_dataset(self):
        for curr_fn in os.listdir(self.input_path):
            curr_dir = os.path.join(self.input_path, curr_fn)
            if os.path.isdir(curr_dir) and curr_fn != 'ProteinLists':
                rmtree(curr_dir)
        test_generator = DataSetGenerator(protein_list='Test_Set.txt', input_path=self.input_path)
        test_generator.build_dataset(num_threads=10, max_target_seqs=2000)
        pdb_path = os.path.join(self.input_path, 'PDB')
        expected_pdb_fn_small = os.path.join(pdb_path, '{}'.format(self.small_structure_id[1:3]),
                                             'pdb{}.ent'.format(self.small_structure_id))
        self.assertTrue(test_generator.protein_data[self.small_structure_id]['PDB_Path'] == expected_pdb_fn_small)
        expected_pdb_fn_large = os.path.join(pdb_path, '{}'.format(self.large_structure_id[1:3]),
                                         'pdb{}.ent'.format(self.large_structure_id))
        self.assertTrue(test_generator.protein_data[self.large_structure_id]['PDB_Path'] == expected_pdb_fn_large)
        sequence_path = os.path.join(self.input_path, 'Sequences')
        expexcted_query_fn_small = os.path.join(sequence_path, '{}.fasta'.format(self.small_structure_id))
        expected_sequence_small = 'PQITLWQRPLVTIRIGGQLKEALLDTGADDTVLEEMNLPGKWKPKMIGGIGGFIKVRQYDQIPVEIGHKAIGTVLVGPTPVNI'\
                                  'IGRNLLTQIGTLNF'
        self.assertEqual(str(test_generator.protein_data[self.small_structure_id]['Query_Sequence'].seq),
                         expected_sequence_small)
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['Sequence_Length'],
                         len(expected_sequence_small))
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['Fasta_File'], expexcted_query_fn_small)
        expexcted_query_fn_large = os.path.join(sequence_path, '{}.fasta'.format(self.large_structure_id))
        expected_sequence_large = 'LDELKKEVSMDDHKLSLDELHNKYGTDLTRGLTNARAKEILARDGPNSLTPPPTTPEWIKFCRQLFGGFSILLWIGAILCFLA'\
                                  'YGIQAATEDEPANDNLYLGVVLSTVVIVTGCFSYYQEAKSSRIMDSFKNMVPQQALVIRDGEKSTINAEFVVAGDLVEVKGGD'\
                                  'RIPADLRIISAHGCKVDNSSLTGESEPQTRSPEFSSENPLETRNIAFFSTNCVEGTARGVVVYTGDRTVMGRIATLASGLEVG'\
                                  'RTPIAIEIEHFIHIITGVAVFLGVSFFILSLILGYSWLEAVIFLIGIIVANVPEGLLATVTVCLTLTAKRMARKNCLVKNLEA'\
                                  'VETLGSTSTICSDKTGTLTQNRMTVAHMWFDNQIHEADTTENQSGAAFDKTSATWSALSRIAALCNRAVFQAGQDNVPILKRS'\
                                  'VAGDASESALLKCIELCCGSVQGMRDRNPKIVEIPFNSTNKYQLSIHENEKSSESRYLLVMKGAPERILDRCSTILLNGAEEP'\
                                  'LKEDMKEAFQNAYLELGGLGERVLGFCHFALPEDKYNEGYPFDADEPNFPTTDLCFVGLMAMIDPPRAAVPDAVGKCRSAGIK'\
                                  'VIMVTGDHPITAKAIAKGVGIISEGNETIEDIAARLNIPIGQVNPRDAKACVVHGSDLKDLSTEVLDDILHYHTEIVFARTSP'\
                                  'QQKLIIVEGCQRQGAIVAVTGDGVNDSPALKKADIGVAMGISGSDVSKQAADMILLDDNFASIVTGVEEGRLIFDNLKKSIAY'\
                                  'TLTSNIPEITPFLVFIIGNVPLPLGTVTILCIDLGTDMVPAISLAYEQAESDIMKRQPRNPKTDKLVNERLISMAYGQIGMIQ'\
                                  'ALGGFFSYFVILAENGFLPMDLIGKRVRWDDRWISDVEDSFGQQWTYEQRKIVEFTCHTSFFISIVVVQWADLIICKTRRNSI'\
                                  'FQQGMKNKILIFGLFEETALAAFLSYCPGTDVALRMYPLKPSWWFCAFPYSLIIFLYDEMRRFIIRRSPGGWVEQETYY'
        self.assertEqual(str(test_generator.protein_data[self.large_structure_id]['Query_Sequence'].seq),
                         expected_sequence_large)
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['Sequence_Length'],
                         len(expected_sequence_large))
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['Fasta_File'], expexcted_query_fn_large)
        blast_path = os.path.join(self.input_path, 'BLAST')
        expected_blast_fn_small = os.path.join(blast_path, '{}.xml'.format(self.small_structure_id))
        self.assertEqual(test_generator.protein_data[self.small_structure_id]['BLAST_File'], expected_blast_fn_small)
        expected_blast_fn_large = os.path.join(blast_path, '{}.xml'.format(self.large_structure_id))
        self.assertEqual(test_generator.protein_data[self.large_structure_id]['BLAST_File'], expected_blast_fn_large)
        pileup_path = os.path.join(self.input_path, 'Pileups')
        expected_pileup_fn_small = os.path.join(pileup_path, '{}.fasta'.format(self.small_structure_id))
        alignment_path = os.path.join(self.input_path, 'Alignments')
        expected_msf_fn_small = os.path.join(alignment_path, '{}.msf'.format(self.small_structure_id))
        expected_fa_fn_small = os.path.join(alignment_path, '{}.fasta'.format(self.small_structure_id))
        if test_generator.protein_data[self.small_structure_id]['Pileup_File']:
            self.assertEqual(test_generator.protein_data[self.small_structure_id]['Pileup_File'],
                             expected_pileup_fn_small)
            self.assertEqual(test_generator.protein_data[self.small_structure_id]['MSF_File'], expected_msf_fn_small)
            self.assertEqual(test_generator.protein_data[self.small_structure_id]['FA_File'], expected_fa_fn_small)
        else:
            self.assertIsNone(test_generator.protein_data[self.small_structure_id]['Pileup_File'])
            self.assertIsNone(test_generator.protein_data[self.small_structure_id]['MSF_File'])
            self.assertIsNone(test_generator.protein_data[self.small_structure_id]['FA_File'])
        expected_pileup_fn_large = os.path.join(pileup_path, '{}.fasta'.format(self.large_structure_id))
        expected_msf_fn_large = os.path.join(alignment_path, '{}.msf'.format(self.large_structure_id))
        expected_fa_fn_large = os.path.join(alignment_path, '{}.fasta'.format(self.large_structure_id))
        if test_generator.protein_data[self.large_structure_id]['Pileup_File']:
            self.assertEqual(test_generator.protein_data[self.large_structure_id]['Pileup_File'],
                             expected_pileup_fn_large)
            self.assertEqual(test_generator.protein_data[self.large_structure_id]['MSF_File'], expected_msf_fn_large)
            self.assertEqual(test_generator.protein_data[self.large_structure_id]['FA_File'], expected_fa_fn_large)
        else:
            self.assertIsNone(test_generator.protein_data[self.large_structure_id]['Pileup_File'])
            self.assertIsNone(test_generator.protein_data[self.large_structure_id]['MSF_File'])
            self.assertIsNone(test_generator.protein_data[self.large_structure_id]['FA_File'])




