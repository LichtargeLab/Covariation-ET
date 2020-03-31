import unittest
from Bio.PDB.Structure import Structure
from test_Base import TestBase
from PDBReference import PDBReference


class TestPDBReference(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestPDBReference, cls).setUpClass()
        cls.expected1_chain_a = 'KVYGRCELAAAMKRLGLDNYRGYSLGNWVCAAKFESNFNTHATNRNTDGSTDYGILQINSRWWCNDGRTPGSKNLCNIPCSALLS'\
                                'SDITASVNCAKKIASGGNGMNAWVAWRNRCKGTDVHAWIRGCRL'
        cls.expected1_len = 129
        cls.expected2_chain_a = 'SSCSSTALSCSNSANSDTCCSPEYGLVVLNMQWAPGYGPDNAFTLHGLWPDKCSGAYAPSGGCDSNRASSSIASVIKSKDSSLYN'\
                                'SMLTYWPSNQGNNNVFWSHEWSKHGTCVSTYDPDCYDNYEEGEDIVDYFQKAMDLRSQYNVYKAFSSNGITPGGTYTATEMQSAI'\
                                'ESYFGAKAKIDCSSGTLSDVALYFYVRGRDTYVITDALSTGSCSGDVEYPTK'
        cls.expected2_len = 222
        cls.expected1_accessions = {'UNP': {'A': ['P00703', 'LYSC_MELGA']}}
        cls.expected2_accessions = {'UNP': {'A': ['P08056', 'RNRH_RHINI']}}
        cls.expected1_seqs = {'UNP': {'A': ('P00703',
                                            'MRSLLILVLCFLPLAALGKVYGRCELAAAMKRLGLDNYRGYSLGNWVCAAKFESNFNTHATNRNTDGSTDYGIL'
                                            'QINSRWWCNDGRTPGSKNLCNIPCSALLSSDITASVNCAKKIASGGNGMNAWVAWRNRCKGTDVHAWIRGCRL')}}
        cls.expected2_seqs = {'UNP': {'A': ('P08056',
                                            'MKAVLALATLIGSTLASSCSSTALSCSNSANSDTCCSPEYGLVVLNMQWAPGYGPDNAFTLHGLWPDKCSGAYA'
                                            'PSGGCDSNRASSSIASVIKSKDSSLYNSMLTYWPSNQGNNNVFWSHEWSKHGTCVSTYDPDCYDNYEEGEDIVD'
                                            'YFQKAMDLRSQYNVYKAFSSNGITPGGTYTATEMQSAIESYFGAKAKIDCSSGTLSDVALYFYVRGRDTYVITD'
                                            'ALSTGSCSGDVEYPTK')}}
        cls.expected_gb_acc = ['5410603', 'AAD43134']
        cls.expected_gb_seq = 'MINRRYELFKDVSDADWNDWRWQVRNRIETVEELKKYIPLTKEEEEGVAQCVKSLRMAITPYYLSLIDPNDPNDPVRKQAIPTALEL'\
                              'NKAAADLEDPLHEDTDSPVPGLTHRYPDRVLLLITDMCSMYCRHCTRRRFAGQSDDSMPMERIDKAIDYIRNTPQVRDVLLSGGDAL'\
                              'LVSDETLEYIIAKLREIPHVEIVRIGSRTPVVLPQRITPELVNMLKKYHPVWLNTHFNHPNEITEESTRACQLLADAGVPLGNQSVL'\
                              'LRGVNDCVHVMKELVNKLVKIRVRPYYIYQCDLSLGLEHFRTPVSKGIEIIEGLRGHTSGYCVPTFVVDAPGGGGKTPVMPNYVISQ'\
                              'SHDKVILRNFEGVITTYSEPINYTPGCNCDVCTGKKKVHKVGVAGLLNGEGMALEPVGLERNKRHVQE'

    def evaluate__init__(self, pdb_file):
        with self.assertRaises(TypeError):
            PDBReference()
        pdb = PDBReference(pdb_file=pdb_file)
        self.assertEqual(pdb.file_name, pdb_file)
        self.assertIsNone(pdb.structure)
        self.assertIsNone(pdb.chains)
        self.assertIsNone(pdb.seq)
        self.assertIsNone(pdb.pdb_residue_list)
        self.assertIsNone(pdb.residue_pos)
        self.assertIsNone(pdb.size)
        self.assertIsNone(pdb.external_seq)

    def test1a__init__(self):
        self.evaluate__init__(pdb_file=self.data_set.protein_data[self.small_structure_id]['PDB'])

    def test1b__init__(self):
        self.evaluate__init__(pdb_file=self.data_set.protein_data[self.large_structure_id]['PDB'])

    def evaluate_import_pdb(self, pdb_file, query, expected_chain, expected_sequence, expected_len):
        pdb1 = PDBReference(pdb_file=pdb_file)
        pdb1.import_pdb(structure_id=query)
        self.assertEqual(pdb1.file_name, pdb_file)
        self.assertIsInstance(pdb1.structure, Structure)
        self.assertTrue(expected_chain in pdb1.chains)
        self.assertEqual(pdb1.seq['A'], expected_sequence)
        for i in range(1, expected_len + 1):
            self.assertGreaterEqual(pdb1.pdb_residue_list[expected_chain][i - 1], i)
        expected_dict = {pdb1.pdb_residue_list[expected_chain][i]: expected_sequence[i] for i in range(expected_len)}
        self.assertEqual(pdb1.residue_pos[expected_chain], expected_dict)
        self.assertEqual(pdb1.size[expected_chain], expected_len)
        self.assertIsNone(pdb1.external_seq)

    def test2a_import_pdb(self):
        self.evaluate_import_pdb(pdb_file=self.data_set.protein_data[self.small_structure_id]['PDB'],
                                 query=self.small_structure_id, expected_chain='A',
                                 expected_sequence=self.expected1_chain_a, expected_len=self.expected1_len)

    def test2b_import_pdb(self):
        self.evaluate_import_pdb(pdb_file=self.data_set.protein_data[self.large_structure_id]['PDB'],
                                 query=self.large_structure_id, expected_chain='A',
                                 expected_sequence=self.expected2_chain_a, expected_len=self.expected2_len)

    def evaluate__parse_external_sequence_accessions(self, pdb_fn, expected_accessions):
        accessions = PDBReference._parse_external_sequence_accessions(pdb_fn=pdb_fn)
        self.assertEqual(accessions, expected_accessions)

    def test3a__parse_external_sequence_accessions(self):
        self.evaluate__parse_external_sequence_accessions(
            pdb_fn=self.data_set.protein_data[self.small_structure_id]['PDB'],
            expected_accessions=self.expected1_accessions)

    def test3b__parse_external_sequence_accessions(self):
        self.evaluate__parse_external_sequence_accessions(
            pdb_fn=self.data_set.protein_data[self.large_structure_id]['PDB'],
            expected_accessions=self.expected2_accessions)

    def evaluate__retrieve_uniprot_seq(self, accessions, expected_seq):
        acc, seq = PDBReference._retrieve_uniprot_seq(accessions=accessions)
        self.assertEqual(acc, accessions[0])
        self.assertEqual(seq, expected_seq)

    def test4a__retrieve_uniprot_seq(self):
        self.evaluate__retrieve_uniprot_seq(accessions=self.expected1_accessions['UNP']['A'],
                                            expected_seq=self.expected1_seqs['UNP']['A'][1])

    def test4b__retrieve_uniprot_seq(self):
        self.evaluate__retrieve_uniprot_seq(accessions=self.expected2_accessions['UNP']['A'],
                                            expected_seq=self.expected2_seqs['UNP']['A'][1])

    def test5__retrieve_genebank_seq(self):
        acc, seq = PDBReference._retrieve_genbank_seq(accessions=self.expected_gb_acc)
        self.assertEqual(acc, self.expected_gb_acc[0])
        self.assertEqual(seq, self.expected_gb_seq)

    def evaluate__parse_external_sequences(self, pdb_fn, expected_sequences):
        pdb = PDBReference(pdb_file=pdb_fn)
        seqs = pdb._parse_external_sequences()
        self.assertEqual(seqs, expected_sequences)

    def test6a__parse_external_sequences(self):
        self.evaluate__parse_external_sequences(pdb_fn=self.data_set.protein_data[self.small_structure_id]['PDB'],
                                                expected_sequences=self.expected1_seqs)

    def test6b__parse_external_sequences(self):
        self.evaluate__parse_external_sequences(pdb_fn=self.data_set.protein_data[self.large_structure_id]['PDB'],
                                                expected_sequences=self.expected2_seqs)

    def evaluate_get_sequence(self, pdb_fn, query_id, chain, source, expected_accession, expected_sequence):
        pdb = PDBReference(pdb_file=pdb_fn)
        if source == 'PDB':
            with self.assertRaises(AttributeError):
                pdb.get_sequence(source=source, chain=chain)
            pdb.import_pdb(structure_id=query_id)
        acc, seq = pdb.get_sequence(source=source, chain=chain)
        self.assertEqual(acc, expected_accession)
        self.assertEqual(seq, expected_sequence)

    def test7a_get_sequence(self):
        self.evaluate_get_sequence(pdb_fn=self.data_set.protein_data[self.small_structure_id]['PDB'],
                                   query_id=self.small_structure_id, chain='A', source='PDB',
                                   expected_sequence=self.expected1_chain_a, expected_accession=self.small_structure_id)

    def test7b_get_sequence(self):
        self.evaluate_get_sequence(pdb_fn=self.data_set.protein_data[self.small_structure_id]['PDB'],
                                   query_id=self.small_structure_id, chain='A', source='UNP',
                                   expected_sequence=self.expected1_seqs['UNP']['A'][1],
                                   expected_accession=self.expected1_accessions['UNP']['A'][0])

    def test7c_get_sequence(self):
        self.evaluate_get_sequence(pdb_fn=self.data_set.protein_data[self.small_structure_id]['PDB'],
                                   query_id=self.small_structure_id, chain='A', source='GB', expected_sequence=None,
                                   expected_accession=None)

    def test7d_get_sequence(self):
        self.evaluate_get_sequence(pdb_fn=self.data_set.protein_data[self.large_structure_id]['PDB'],
                                   query_id=self.large_structure_id, chain='A', source='PDB',
                                   expected_sequence=self.expected2_chain_a, expected_accession=self.large_structure_id)

    def test7e_get_sequence(self):
        self.evaluate_get_sequence(pdb_fn=self.data_set.protein_data[self.large_structure_id]['PDB'],
                                   query_id=self.large_structure_id, chain='A', source='UNP',
                                   expected_sequence=self.expected2_seqs['UNP']['A'][1],
                                   expected_accession=self.expected2_accessions['UNP']['A'][0])

    def test7f_get_sequence(self):
        self.evaluate_get_sequence(pdb_fn=self.data_set.protein_data[self.large_structure_id]['PDB'],
                                   query_id=self.large_structure_id, chain='A', source='GB', expected_sequence=None,
                                   expected_accession=None)


if __name__ == '__main__':
    unittest.main()
