import os
from unittest import TestCase
from Bio.PDB.Structure import Structure
from PDBReference import PDBReference


class TestPDBReference(TestCase):

    def test__init__(self):
        with self.assertRaises(TypeError):
            PDBReference()
        pdb1 = PDBReference(pdb_file='../Test/query_1c17A.pdb')
        self.assertEqual(pdb1.file_name, os.path.abspath('../Test/query_1c17A.pdb'))
        self.assertNotEqual(pdb1.file_name, '../Test/query_1c17A.pdb')
        self.assertIsNone(pdb1.structure)
        self.assertIsNone(pdb1.chains)
        self.assertIsNone(pdb1.seq)
        self.assertIsNone(pdb1.pdb_residue_list)
        self.assertIsNone(pdb1.residue_pos)
        self.assertIsNone(pdb1.size)

    def test_import_pdb(self):
        pdb1 = PDBReference(pdb_file='../Test/query_1c17A.pdb')
        pdb1.import_pdb(structure_id='query_1c17A')
        self.assertEqual(pdb1.file_name, os.path.abspath('../Test/query_1c17A.pdb'))
        self.assertIsInstance(pdb1.structure, Structure)
        self.assertEqual(pdb1.chains, set(['A']))
        self.assertEqual(pdb1.seq['A'],
                         'MENLNMDLLYMAAAVMMGLAAIGAAIGIGILGGKFLEGAARQPDLIPLLRTQFFIVMGLVDAIPMIAVGLGLYVMFAVA')
        self.assertEqual(pdb1.pdb_residue_list['A'],
                         list(range(1, 80)))
        self.assertEqual(pdb1.residue_pos['A'], {i + 1: aa for i, aa in enumerate('MENLNMDLLYMAAAVMMGLAAIGAAIGIGILGGKFL'
                                                                                  'EGAARQPDLIPLLRTQFFIVMGLVDAIPMIAVGLGL'
                                                                                  'YVMFAVA')})
        self.assertEqual(pdb1.size['A'], 79)
        pdb2 = PDBReference(pdb_file='../Test/query_1h1vA.pdb')
        pdb2.import_pdb(structure_id='query_1h1vA')
        self.assertEqual(pdb2.file_name, os.path.abspath('../Test/query_1h1vA.pdb'))
        self.assertIsInstance(pdb2.structure, Structure)
        self.assertEqual(pdb2.chains, set(['A']))
        expected_seq = 'TTALVCDNGSGLVKAGFAGDDAPRAVFPSIVGRPRHMVGMGQKDSYVGDEAQSKRGILTLKYPIEHGIITNWDDMEKIWHHTFYNELRVAPEE' \
                       'HPTLLTEAPLNPKANREKMTQIMFETFNVPAMYVAIQAVLSLYASGRTTGIVLDSGDGVTHNVPIYEGYALPHAIMRLDLAGRDLTDYLMKIL' \
                       'TERGYSFVTTAEREIVRDIKEKLCYVALDFENEMATAASSSSLEKSYELPDGQVITIGNERFRCPETLFQPSFIGMESAGIHETTYNSIMKCD' \
                       'IDIRKDLYANNVMSGGTTMYPGIADRMQKEITALAPSTMKIKIIAPPERKYSVWIGGSILASLSTFQQMWITKQEYDEAGPSIVHRKCF'
        self.assertEqual(pdb2.seq['A'], expected_seq)
        expected_positions = list(range(5, 41)) + list(range(44, 376))
        self.assertEqual(pdb2.pdb_residue_list['A'], expected_positions)
        self.assertEqual(pdb2.residue_pos['A'], {expected_positions[i]: expected_seq[i] for i in range(368)})
        self.assertEqual(pdb2.size['A'], 368)

