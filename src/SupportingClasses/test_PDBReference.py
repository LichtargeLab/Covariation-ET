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

    def test2a_import_pdb(self):
        self.evaluate_import_pdb(pdb_file=self.data_set.protein_data[self.small_structure_id]['PDB'],
                                 query=self.small_structure_id, expected_chain='A',
                                 expected_sequence=self.expected1_chain_a, expected_len=self.expected1_len)

    def test2b_import_pdb(self):
        self.evaluate_import_pdb(pdb_file=self.data_set.protein_data[self.large_structure_id]['PDB'],
                                 query=self.large_structure_id, expected_chain='A',
                                 expected_sequence=self.expected2_chain_a, expected_len=self.expected2_len)


if __name__ == '__main__':
    unittest.main()
