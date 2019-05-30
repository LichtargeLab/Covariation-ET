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
        # os.remove(cls.protein_list_fn)
        del cls.protein_list_fn
        del cls.large_structure_id
        del cls.small_structure_id
        del cls.protein_list_path
        del cls.input_path

    # def tearDown(self):
    #     for curr_fn in os.listdir(self.input_path):
    #         curr_dir = os.path.join(self.input_path, curr_fn)
    #         if os.path.isdir(curr_dir):
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
        test_generator = DataSetGenerator(protein_list='Test_Set.txt', input_path=self.input_path)
        with self.assertRaises(KeyError):
            test_generator._blast_query_sequence(protein_id=self.small_structure_id)
        test_generator._download_pdb(protein_id=self.small_structure_id)
        test_generator._parse_query_sequence(protein_id=self.small_structure_id)
        blast_fn_small = test_generator._blast_query_sequence(protein_id=self.small_structure_id, num_threads=1,
                                                              max_target_seqs=500)
        self.assertTrue(os.path.isdir(blast_path))
        expected_fn_small = os.path.join(blast_path, '{}.xml'.format(self.small_structure_id))
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
        self.assertEqual(blast_fn_large, expected_fn_large)
        self.assertTrue(os.path.isfile(blast_fn_large))

    def test__blast_query_sequence_multi_thread(self):
        blast_path = os.path.join(self.input_path, 'BLAST')
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
        self.assertEqual(blast_fn_large, expected_fn_large)
        self.assertTrue(os.path.isfile(blast_fn_large))

    def test__restrict_sequences(self):
        pileup_path = os.path.join(self.input_path, 'Pileups')
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
        else:
            self.assertGreaterEqual(min_id_small, 0.30)
            self.assertIsNone(pileup_fn_small)
        with self.assertRaises(KeyError):
            test_generator._restrict_sequences(protein_id=self.large_structure_id)
        test_generator._download_pdb(protein_id=self.large_structure_id)
        test_generator._parse_query_sequence(protein_id=self.large_structure_id)
        test_generator._blast_query_sequence(protein_id=self.large_structure_id, num_threads=10, max_target_seqs=2000)
        expected_fn_large = os.path.join(pileup_path, '{}.fasta'.format(self.large_structure_id))
        min_id_large, num_seqs_large, pileup_fn_large = test_generator._restrict_sequences(
            protein_id=self.large_structure_id)
        if num_seqs_small >= 125:
            self.assertLessEqual(0.95, min_id_large)
            self.assertGreaterEqual(min_id_large, 0.30)
            self.assertEqual(pileup_fn_large, expected_fn_large)
        else:
            self.assertGreaterEqual(min_id_large, 0.30)
            self.assertIsNone(pileup_fn_large)
