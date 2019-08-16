import unittest
from Bio.PDB.Structure import Structure
from test_Base import TestBase
from PDBReference import PDBReference


class TestPDBReference(TestBase):

    @classmethod
    def setUpClass(cls):
        super(TestPDBReference, cls).setUpClass()

    def test1__init__(self):
        with self.assertRaises(TypeError):
            PDBReference()
        pdb1 = PDBReference(pdb_file=self.data_set.protein_data[self.small_structure_id]['PDB'])
        self.assertEqual(pdb1.file_name, self.data_set.protein_data[self.small_structure_id]['PDB'])
        self.assertIsNone(pdb1.structure)
        self.assertIsNone(pdb1.chains)
        self.assertIsNone(pdb1.seq)
        self.assertIsNone(pdb1.pdb_residue_list)
        self.assertIsNone(pdb1.residue_pos)
        self.assertIsNone(pdb1.size)

    def test2a_import_pdb(self):
        pdb1 = PDBReference(pdb_file=self.data_set.protein_data[self.small_structure_id]['PDB'])
        pdb1.import_pdb(structure_id=self.small_structure_id)
        self.assertEqual(pdb1.file_name, self.data_set.protein_data[self.small_structure_id]['PDB'])
        self.assertIsInstance(pdb1.structure, Structure)
        self.assertTrue('A' in pdb1.chains)
        self.assertEqual(pdb1.seq['A'], str(self.data_set.protein_data[self.small_structure_id]['Sequence'].seq))
        for i in range(1, self.data_set.protein_data[self.small_structure_id]['Length'] + 1):
            self.assertGreaterEqual(pdb1.pdb_residue_list['A'][i - 1], i)
        expected_dict = {pdb1.pdb_residue_list['A'][i]:
                             self.data_set.protein_data[self.small_structure_id]['Sequence'].seq[i]
                                    for i in range(self.data_set.protein_data[self.small_structure_id]['Length'])}
        self.assertEqual(pdb1.residue_pos['A'], expected_dict)
        self.assertEqual(pdb1.size['A'], self.data_set.protein_data[self.small_structure_id]['Length'])

    def test2b_import_pdb(self):
        pdb2 = PDBReference(pdb_file=self.data_set.protein_data[self.large_structure_id]['PDB'])
        pdb2.import_pdb(structure_id=self.large_structure_id)
        self.assertEqual(pdb2.file_name, self.data_set.protein_data[self.large_structure_id]['PDB'])
        self.assertIsInstance(pdb2.structure, Structure)
        self.assertTrue('A' in pdb2.chains)
        self.assertEqual(pdb2.seq['A'], str(self.data_set.protein_data[self.large_structure_id]['Sequence'].seq))
        for i in range(1, self.data_set.protein_data[self.large_structure_id]['Length'] + 1):
            self.assertGreaterEqual(pdb2.pdb_residue_list['A'][i - 1], i)
        expected_dict = {pdb2.pdb_residue_list['A'][i]:
                             self.data_set.protein_data[self.large_structure_id]['Sequence'].seq[i]
                         for i in range(self.data_set.protein_data[self.large_structure_id]['Length'])}
        self.assertEqual(pdb2.residue_pos['A'], expected_dict)
        self.assertEqual(pdb2.size['A'], self.data_set.protein_data[self.large_structure_id]['Length'])


if __name__ == '__main__':
    unittest.main()
