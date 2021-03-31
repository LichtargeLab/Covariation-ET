import os
import sys
import unittest
from datetime import datetime
from unittest import TestCase
from Bio.ExPASy import get_sprot_raw

#
from dotenv import find_dotenv, load_dotenv
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)
source_code_path = os.path.join(os.environ.get('PROJECT_PATH'), 'src')
# Add the project path to the python path so the required clases can be imported
if source_code_path not in sys.path:
    sys.path.append(os.path.join(os.environ.get('PROJECT_PATH'), 'src'))
#

from SupportingClasses.PDBReference import PDBReference, dbref_parse
from Testing.test_Base import generate_temp_fn, write_out_temp_fn

chain_a_pdb_partial = 'ATOM      9  N   GLU A   2     153.913  21.571  52.586  1.00 65.12           N  \n'\
                      'ATOM     10  CA  GLU A   2     153.617  20.553  53.599  1.00 64.46           C  \n'\
                      'ATOM     11  C   GLU A   2     153.850  21.067  55.018  1.00 63.81           C  \n'\
                      'ATOM     12  O   GLU A   2     153.177  20.639  55.960  1.00 63.64           O  \n'\
                      'ATOM     13  CB  GLU A   2     154.431  19.277  53.350  1.00 64.59           C  \n'\
                      'ATOM     14  CG  GLU A   2     153.767  18.286  52.392  1.00 64.91           C  \n'\
                      'ATOM     15  CD  GLU A   2     153.675  18.807  50.965  1.00 65.39           C  \n'\
                      'ATOM     16  OE1 GLU A   2     152.542  19.056  50.495  1.00 65.13           O  \n'\
                      'ATOM     17  OE2 GLU A   2     154.735  18.978  50.321  1.00 65.58           O  \n'\
                      'ATOM     18  N   THR A   3      24.499  13.739  37.648  1.00 12.91           N  \n'\
                      'ATOM     19  CA  THR A   3      24.278  13.068  38.914  1.00  9.39           C  \n'\
                      'ATOM     20  C   THR A   3      22.973  13.448  39.580  1.00  9.61           C  \n'\
                      'ATOM     21  O   THR A   3      22.188  12.566  39.933  1.00 10.86           O  \n'\
                      'ATOM     22  CB  THR A   3      25.405  13.382  39.929  1.00 11.10           C  \n'\
                      'ATOM     23  OG1 THR A   3      26.603  13.084  39.227  1.00 13.57           O  \n'\
                      'ATOM     24  CG2 THR A   3      25.336  12.578  41.212  1.00 15.68           C  \n'

chain_a_pdb_partial2 = 'ATOM      1  N   MET A   1     152.897  26.590  66.235  1.00 57.82           N  \n'\
                       'ATOM      2  CA  MET A   1     153.488  26.592  67.584  1.00 57.79           C  \n'\
                       'ATOM      3  C   MET A   1     153.657  28.043  68.066  1.00 57.26           C  \n'\
                       'ATOM      4  O   MET A   1     153.977  28.924  67.266  1.00 57.37           O  \n'\
                       'ATOM      5  CB  MET A   1     154.843  25.881  67.544  1.00 58.16           C  \n'\
                       'ATOM      6  CG  MET A   1     155.689  25.983  68.820  1.00 59.67           C  \n'\
                       'ATOM      7  SD  MET A   1     157.418  25.517  68.551  1.00 62.17           S  \n'\
                       'ATOM      8  CE  MET A   1     158.062  26.956  67.686  1.00 61.68           C  \n'

chain_a_pdb = chain_a_pdb_partial2 + chain_a_pdb_partial


chain_b_pdb_partial = 'ATOM     40  N   ARG B   3      24.805   9.537  22.454  1.00 12.48           N  \n'\
                      'ATOM     41  CA  ARG B   3      24.052   8.386  22.974  1.00 10.49           C  \n'\
                      'ATOM     42  C   ARG B   3      24.897   7.502  23.849  1.00 10.61           C  \n'\
                      'ATOM     43  O   ARG B   3      24.504   7.220  24.972  1.00 11.97           O  \n'\
                      'ATOM     44  CB  ARG B   3      23.506   7.549  21.793  1.00 13.55           C  \n'\
                      'ATOM     45  CG  ARG B   3      22.741   6.293  22.182  1.00 17.24           C  \n'\
                      'ATOM     46  CD  ARG B   3      22.242   5.559  20.931  1.00 13.64           C  \n'\
                      'ATOM     47  NE  ARG B   3      23.319   5.176  20.037  1.00 14.77           N  \n'\
                      'ATOM     48  CZ  ARG B   3      23.931   3.984  20.083  1.00 19.58           C  \n'\
                      'ATOM     49  NH1 ARG B   3      23.622   3.034  20.961  1.00 15.71           N  \n'\
                      'ATOM     50  NH2 ARG B   3      24.895   3.751  19.199  1.00 21.91           N  \n'\
                      'ATOM     51  N   GLU B   4     163.913  21.571  52.586  1.00 65.12           N  \n'\
                      'ATOM     52  CA  GLU B   4     163.617  20.553  53.599  1.00 64.46           C  \n'\
                      'ATOM     53  C   GLU B   4     163.850  21.067  55.018  1.00 63.81           C  \n'\
                      'ATOM     54  O   GLU B   4     163.177  20.639  55.960  1.00 63.64           O  \n'\
                      'ATOM     55  CB  GLU B   4     164.431  19.277  53.350  1.00 64.59           C  \n'\
                      'ATOM     56  CG  GLU B   4     163.767  18.286  52.392  1.00 64.91           C  \n'\
                      'ATOM     57  CD  GLU B   4     163.675  18.807  50.965  1.00 65.39           C  \n'\
                      'ATOM     58  OE1 GLU B   4     162.542  19.056  50.495  1.00 65.13           O  \n'\
                      'ATOM     59  OE2 GLU B   4     164.735  18.978  50.321  1.00 65.58           O  \n'\
                      'ATOM     60  N   GLU B   5     153.913  31.571  52.586  1.00 65.12           N  \n'\
                      'ATOM     61  CA  GLU B   5     153.617  30.553  53.599  1.00 64.46           C  \n'\
                      'ATOM     62  C   GLU B   5     153.850  31.067  55.018  1.00 63.81           C  \n'\
                      'ATOM     63  O   GLU B   5     153.177  30.639  55.960  1.00 63.64           O  \n'\
                      'ATOM     64  CB  GLU B   5     154.431  29.277  53.350  1.00 64.59           C  \n'\
                      'ATOM     65  CG  GLU B   5     153.767  28.286  52.392  1.00 64.91           C  \n'\
                      'ATOM     66  CD  GLU B   5     153.675  28.807  50.965  1.00 65.39           C  \n'\
                      'ATOM     67  OE1 GLU B   5     152.542  29.056  50.495  1.00 65.13           O  \n'\
                      'ATOM     68  OE2 GLU B   5     154.735  28.978  50.321  1.00 65.58           O  \n'
chain_b_pdb = 'ATOM     25  N   MET B   1     152.897  26.590  66.235  1.00 57.82           N  \n'\
              'ATOM     26  CA  MET B   1     153.488  26.592  67.584  1.00 57.79           C  \n'\
              'ATOM     27  C   MET B   1     153.657  28.043  68.066  1.00 57.26           C  \n'\
              'ATOM     28  O   MET B   1     153.977  28.924  67.266  1.00 57.37           O  \n'\
              'ATOM     29  CB  MET B   1     154.843  25.881  67.544  1.00 58.16           C  \n'\
              'ATOM     30  CG  MET B   1     155.689  25.983  68.820  1.00 59.67           C  \n'\
              'ATOM     31  SD  MET B   1     157.418  25.517  68.551  1.00 62.17           S  \n'\
              'ATOM     32  CE  MET B   1     158.062  26.956  67.686  1.00 61.68           C  \n' \
              'ATOM     33  N   THR B   2      24.499  13.739  37.648  1.00 12.91           N  \n' \
              'ATOM     34  CA  THR B   2      24.278  13.068  38.914  1.00  9.39           C  \n' \
              'ATOM     35  C   THR B   2      22.973  13.448  39.580  1.00  9.61           C  \n' \
              'ATOM     36  O   THR B   2      22.188  12.566  39.933  1.00 10.86           O  \n' \
              'ATOM     37  CB  THR B   2      25.405  13.382  39.929  1.00 11.10           C  \n' \
              'ATOM     38  OG1 THR B   2      26.603  13.084  39.227  1.00 13.57           O  \n' \
              'ATOM     39  CG2 THR B   2      25.336  12.578  41.212  1.00 15.68           C  \n' + chain_b_pdb_partial
chain_a_unp_id1 = 'P00703'
chain_a_unp_id2 = 'LYSC_MELGA'
chain_a_unp_seq = 'MRSLLILVLCFLPLAALGKVYGRCELAAAMKRLGLDNYRGYSLGNWVCAAKFESNFNTHATNRNTDGSTDYGILQINSRWWCNDGRTPGSKNLCNIPCS'\
                  'ALLSSDITASVNCAKKIASGGNGMNAWVAWRNRCKGTDVHAWIRGCRL'
chain_a_unp_dbref = 'DBREF  135L A    1   129  UNP    P00703   LYSC_MELGA      19    147             \n'
chain_a_unp_seqres = 'SEQRES   1 A  129  LYS VAL TYR GLY ARG CYS GLU LEU ALA ALA ALA MET LYS          \n'\
                     'SEQRES   2 A  129  ARG LEU GLY LEU ASP ASN TYR ARG GLY TYR SER LEU GLY          \n'\
                     'SEQRES   3 A  129  ASN TRP VAL CYS ALA ALA LYS PHE GLU SER ASN PHE ASN          \n'\
                     'SEQRES   4 A  129  THR HIS ALA THR ASN ARG ASN THR ASP GLY SER THR ASP          \n'\
                     'SEQRES   5 A  129  TYR GLY ILE LEU GLN ILE ASN SER ARG TRP TRP CYS ASN          \n'\
                     'SEQRES   6 A  129  ASP GLY ARG THR PRO GLY SER LYS ASN LEU CYS ASN ILE          \n'\
                     'SEQRES   7 A  129  PRO CYS SER ALA LEU LEU SER SER ASP ILE THR ALA SER          \n'\
                     'SEQRES   8 A  129  VAL ASN CYS ALA LYS LYS ILE ALA SER GLY GLY ASN GLY          \n'\
                     'SEQRES   9 A  129  MET ASN ALA TRP VAL ALA TRP ARG ASN ARG CYS LYS GLY          \n'\
                     'SEQRES  10 A  129  THR ASP VAL HIS ALA TRP ILE ARG GLY CYS ARG LEU              \n'
chain_g_unp_id1 = 'Q70Q12'
chain_g_unp_id2 = 'Q70Q12_SQUAC'
chain_g_unp_seq = 'MLGAATGLMVLVAVTQGVWAMDPEGPDNDERFTYDYYRLRVVGLIVAAVLCVIGIIILLAGKCRCKFNQNKRTRSNSGTATAQHLLQPGEATEC'
chain_g_unp_dbref = 'DBREF  2ZXE G    1    74  UNP    Q70Q12   Q70Q12_SQUAC    21     94             \n'
chain_g_unp_seqres = 'SEQRES   1 G   74  MET ASP PRO GLU GLY PRO ASP ASN ASP GLU ARG PHE THR          \n'\
                     'SEQRES   2 G   74  TYR ASP TYR TYR ARG LEU ARG VAL VAL GLY LEU ILE VAL          \n'\
                     'SEQRES   3 G   74  ALA ALA VAL LEU CYS VAL ILE GLY ILE ILE ILE LEU LEU          \n'\
                     'SEQRES   4 G   74  ALA GLY LYS CYS ARG CYS LYS PHE ASN GLN ASN LYS ARG          \n'\
                     'SEQRES   5 G   74  THR ARG SER ASN SER GLY THR ALA THR ALA GLN HIS LEU          \n'\
                     'SEQRES   6 G   74  LEU GLN PRO GLY GLU ALA THR GLU CYS                          \n'
chain_a_gb_id1 = '11497664'
chain_a_gb_id2 = 'WP_010877557'
chain_a_gb_seq = 'MKICVFHDYFGAIGGGEKVALTISKLFNADVITTDVDAVPEEFRNKVISLGETIKLPPLKQIDASLKFYFSDFPDYDFYILSGNWVMFASKRHIPNLLYC'\
                 'YTPPRAFYDLYGDYLKKRNILTKPAFILWVKFHRKWAERMLKHIDKVVCISQNIKSRCKNFWGIDAEVIYPPVETSKFKFKCYGDFWLSVNRIYPEKRIE'\
                 'LQLEVFKKLQDEKLYIVGWFSKGDHAERYARKIMKIAPDNVKFLGSVSEEELIDLYSRCKGLLCTAKDEDFGLTPIEAMASGKPVIAVNEGGFKETVINE'\
                 'KTGYLVNADVNEIIDAMKKVSKNPDKFKKDCFRRAKEFDISIFKNKIKDAIRIVKKNFKNNTC'
chain_a_gb_dbref = 'DBREF  2F9F A    1   167  GB     11497664 WP_010877557      172    338             \n'
chain_a_gb_seqres = 'SEQRES   1 A  177  MSE GLY HIS HIS HIS HIS HIS HIS SER HIS PRO VAL GLU          \n'\
                    'SEQRES   2 A  177  THR SER LYS PHE LYS PHE LYS CYS TYR GLY ASP PHE TRP          \n'\
                    'SEQRES   3 A  177  LEU SER VAL ASN ARG ILE TYR PRO GLU LYS ARG ILE GLU          \n'\
                    'SEQRES   4 A  177  LEU GLN LEU GLU VAL PHE LYS LYS LEU GLN ASP GLU LYS          \n'\
                    'SEQRES   5 A  177  LEU TYR ILE VAL GLY TRP PHE SER LYS GLY ASP HIS ALA          \n'\
                    'SEQRES   6 A  177  GLU ARG TYR ALA ARG LYS ILE MSE LYS ILE ALA PRO ASP          \n'\
                    'SEQRES   7 A  177  ASN VAL LYS PHE LEU GLY SER VAL SER GLU GLU GLU LEU          \n'\
                    'SEQRES   8 A  177  ILE ASP LEU TYR SER ARG CYS LYS GLY LEU LEU CYS THR          \n'\
                    'SEQRES   9 A  177  ALA LYS ASP GLU ASP PHE GLY LEU THR PRO ILE GLU ALA          \n'\
                    'SEQRES  10 A  177  MSE ALA SER GLY LYS PRO VAL ILE ALA VAL ASN GLU GLY          \n'\
                    'SEQRES  11 A  177  GLY PHE LYS GLU THR VAL ILE ASN GLU LYS THR GLY TYR          \n'\
                    'SEQRES  12 A  177  LEU VAL ASN ALA ASP VAL ASN GLU ILE ILE ASP ALA MSE          \n'\
                    'SEQRES  13 A  177  LYS LYS VAL SER LYS ASN PRO ASP LYS PHE LYS LYS ASP          \n'\
                    'SEQRES  14 A  177  CYS PHE ARG ARG ALA LYS GLU PHE       \n'
chain_b_gb_dbref = 'DBREF  2F9F B    1   167  GB     11497664 WP_010877557      172    338             \n'
chain_b_gb_seqres = 'SEQRES   1 B  177  MSE GLY HIS HIS HIS HIS HIS HIS SER HIS PRO VAL GLU          \n'\
                    'SEQRES   2 B  177  THR SER LYS PHE LYS PHE LYS CYS TYR GLY ASP PHE TRP          \n'\
                    'SEQRES   3 B  177  LEU SER VAL ASN ARG ILE TYR PRO GLU LYS ARG ILE GLU          \n'\
                    'SEQRES   4 B  177  LEU GLN LEU GLU VAL PHE LYS LYS LEU GLN ASP GLU LYS          \n'\
                    'SEQRES   5 B  177  LEU TYR ILE VAL GLY TRP PHE SER LYS GLY ASP HIS ALA          \n'\
                    'SEQRES   6 B  177  GLU ARG TYR ALA ARG LYS ILE MSE LYS ILE ALA PRO ASP          \n'\
                    'SEQRES   7 B  177  ASN VAL LYS PHE LEU GLY SER VAL SER GLU GLU GLU LEU          \n'\
                    'SEQRES   8 B  177  ILE ASP LEU TYR SER ARG CYS LYS GLY LEU LEU CYS THR          \n'\
                    'SEQRES   9 B  177  ALA LYS ASP GLU ASP PHE GLY LEU THR PRO ILE GLU ALA          \n'\
                    'SEQRES  10 B  177  MSE ALA SER GLY LYS PRO VAL ILE ALA VAL ASN GLU GLY          \n'\
                    'SEQRES  11 B  177  GLY PHE LYS GLU THR VAL ILE ASN GLU LYS THR GLY TYR          \n'\
                    'SEQRES  12 B  177  LEU VAL ASN ALA ASP VAL ASN GLU ILE ILE ASP ALA MSE          \n'\
                    'SEQRES  13 B  177  LYS LYS VAL SER LYS ASN PRO ASP LYS PHE LYS LYS ASP          \n'\
                    'SEQRES  14 B  177  CYS PHE ARG ARG ALA LYS GLU PHE       \n'


class TestImportPDB(TestCase):

    def evaluate_import_pdb(self, pdb, expected_struct_len, expected_chain, expected_seq, expected_res_list,
                            expected_res_pos, expected_size):
        self.assertEqual(len(pdb.structure), expected_struct_len)
        self.assertEqual(pdb.chains, expected_chain)
        self.assertEqual(pdb.seq, expected_seq)
        self.assertEqual(pdb.pdb_residue_list, expected_res_list)
        self.assertEqual(pdb.residue_pos, expected_res_pos)
        self.assertEqual(pdb.size, expected_size)
        self.assertIsNone(pdb.external_seq)

    def test_init(self):
        fn = write_out_temp_fn(suffix='pdb')
        pdb = PDBReference(pdb_file=fn)
        self.assertTrue(pdb.file_name.endswith(fn))
        os.remove(fn)

    def test_import_pdb_single_seq(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb)
        pdb = PDBReference(pdb_file=fn)
        pdb.import_pdb(structure_id='1TES')
        self.evaluate_import_pdb(pdb=pdb, expected_struct_len=1, expected_chain={'A'}, expected_seq={'A': 'MET'},
                                 expected_res_list={'A': [1, 2, 3]}, expected_res_pos={'A': {1: 'M', 2: 'E', 3: 'T'}},
                                 expected_size={'A': 3})
        os.remove(fn)

    def test_import_pdb_multiple_seq(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb + chain_b_pdb)
        pdb = PDBReference(pdb_file=fn)
        pdb.import_pdb(structure_id='1TES')
        self.evaluate_import_pdb(pdb=pdb, expected_struct_len=1, expected_chain={'A', 'B'},
                                 expected_seq={'A': 'MET', 'B': 'MTREE'},
                                 expected_res_list={'A': [1, 2, 3], 'B': [1, 2, 3, 4, 5]},
                                 expected_res_pos={'A': {1: 'M', 2: 'E', 3: 'T'},
                                                   'B': {1: 'M', 2: 'T', 3: 'R', 4: 'E', 5: 'E'}},
                                 expected_size={'A': 3, 'B': 5})
        os.remove(fn)

    def test_import_pdb_missing_positions_single_seq(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb_partial)
        pdb = PDBReference(pdb_file=fn)
        pdb.import_pdb(structure_id='1TES')
        self.evaluate_import_pdb(pdb=pdb, expected_struct_len=1, expected_chain={'A'}, expected_seq={'A': 'ET'},
                                 expected_res_list={'A': [2, 3]}, expected_res_pos={'A': {2: 'E', 3: 'T'}},
                                 expected_size={'A': 2})
        os.remove(fn)

    def test_import_pdb_missing_positions_multiple_seq(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb_partial + chain_b_pdb_partial)
        pdb = PDBReference(pdb_file=fn)
        pdb.import_pdb(structure_id='1TES')
        self.evaluate_import_pdb(pdb=pdb, expected_struct_len=1, expected_chain={'A', 'B'},
                                 expected_seq={'A': 'ET', 'B': 'REE'}, expected_res_list={'A': [2, 3], 'B': [3, 4, 5]},
                                 expected_res_pos={'A': {2: 'E', 3: 'T'}, 'B': {3: 'R', 4: 'E', 5: 'E'}},
                                 expected_size={'A': 2, 'B': 3})
        os.remove(fn)

    def test_import_pdb_save_single_seq(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb)
        save_fname = f'temp_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.pkl'
        pdb = PDBReference(pdb_file=fn)
        self.assertFalse(os.path.isfile(save_fname))
        pdb.import_pdb(structure_id='1TES', save_file=save_fname)
        self.assertTrue(os.path.isfile(save_fname))
        pdb2 = PDBReference(pdb_file=fn)
        pdb2.import_pdb(structure_id='1TES', save_file=save_fname)
        self.evaluate_import_pdb(pdb=pdb, expected_struct_len=1, expected_chain={'A'}, expected_seq={'A': 'MET'},
                                 expected_res_list={'A': [1, 2, 3]}, expected_res_pos={'A': {1: 'M', 2: 'E', 3: 'T'}},
                                 expected_size={'A': 3})
        os.remove(fn)
        os.remove(save_fname)

    def test_import_pdb_save_multiple_seq(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb + chain_b_pdb)
        save_fname = f'temp_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.pkl'
        pdb = PDBReference(pdb_file=fn)
        self.assertFalse(os.path.isfile(save_fname))
        pdb.import_pdb(structure_id='1TES', save_file=save_fname)
        self.assertTrue(os.path.isfile(save_fname))
        pdb2 = PDBReference(pdb_file=fn)
        pdb2.import_pdb(structure_id='1TES', save_file=save_fname)
        self.evaluate_import_pdb(pdb=pdb, expected_struct_len=1, expected_chain={'A', 'B'},
                                 expected_seq={'A': 'MET', 'B': 'MTREE'},
                                 expected_res_list={'A': [1, 2, 3], 'B': [1, 2, 3, 4, 5]},
                                 expected_res_pos={'A': {1: 'M', 2: 'E', 3: 'T'},
                                                   'B': {1: 'M', 2: 'T', 3: 'R', 4: 'E', 5: 'E'}},
                                 expected_size={'A': 3, 'B': 5})
        os.remove(fn)
        os.remove(save_fname)

    def test_import_pdb_misordered_numbering_seq(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb_partial2 + chain_a_pdb_partial)
        pdb = PDBReference(pdb_file=fn)
        pdb.import_pdb(structure_id='1TES')
        self.evaluate_import_pdb(pdb=pdb, expected_struct_len=1, expected_chain={'A'}, expected_seq={'A': 'MET'},
                                 expected_res_list={'A': [1, 2, 3]}, expected_res_pos={'A': {1: 'M', 2: 'E', 3: 'T'}},
                                 expected_size={'A': 3})
        os.remove(fn)

        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb_partial + chain_a_pdb_partial2)
        pdb = PDBReference(pdb_file=fn)
        pdb.import_pdb(structure_id='1TES')
        self.evaluate_import_pdb(pdb=pdb, expected_struct_len=1, expected_chain={'A'}, expected_seq={'A': 'MET'},
                                 expected_res_list={'A': [1, 2, 3]}, expected_res_pos={'A': {1: 'M', 2: 'E', 3: 'T'}},
                                 expected_size={'A': 3})
        os.remove(fn)

    def test_init_failure_none(self):
        with self.assertRaises(AttributeError):
            PDBReference(pdb_file=None)

    def test_init_failure_bad_file_path(self):
        fname = generate_temp_fn(suffix='pdb')
        with self.assertRaises(AttributeError):
            PDBReference(pdb_file=fname)

    def test_import_pdb_failure_empty_file(self):
        fn = write_out_temp_fn(suffix='pdb')
        pdb = PDBReference(pdb_file=fn)
        with self.assertRaises(ValueError):
            pdb.import_pdb(structure_id='1TES')
        os.remove(fn)


class GetPDBSequence(TestCase):

    def evaluate_get_sequence(self, seq, expected_ids, expected_seq):
        self.assertTrue(seq[0] in expected_ids)
        self.assertEqual(seq[1], expected_seq)

    def test_get_seq_pdb(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb)
        pdb = PDBReference(pdb_file=fn)
        pdb.import_pdb(structure_id='1TES')
        seq = pdb.get_sequence(chain='A', source='PDB')
        self.evaluate_get_sequence(seq=seq, expected_ids=['1TES'], expected_seq='MET')
        os.remove(fn)

    def test_get_seq_unp(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_unp_dbref + chain_a_unp_seqres)
        pdb = PDBReference(pdb_file=fn)
        seq = pdb.get_sequence(chain='A', source='UNP')
        self.evaluate_get_sequence(seq=seq, expected_ids=[chain_a_unp_id1], expected_seq=chain_a_unp_seq)
        os.remove(fn)

    def test_get_seq_unp_multiple(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=(chain_a_unp_dbref + chain_g_unp_dbref + chain_a_unp_seqres +
                                                      chain_g_unp_seqres))
        pdb = PDBReference(pdb_file=fn)
        seq = pdb.get_sequence(chain='G', source='UNP')
        self.evaluate_get_sequence(seq=seq, expected_ids=[chain_g_unp_id1, chain_g_unp_id2],
                                   expected_seq=chain_g_unp_seq)
        seq2 = pdb.get_sequence(chain='A', source='UNP')
        self.evaluate_get_sequence(seq=seq2, expected_ids=[chain_a_unp_id1, chain_a_unp_id2],
                                   expected_seq=chain_a_unp_seq)
        os.remove(fn)

    def test_get_seq_gb(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_gb_dbref + chain_a_gb_seqres)
        pdb = PDBReference(pdb_file=fn)
        seq = pdb.get_sequence(chain='A', source='GB')
        self.evaluate_get_sequence(seq=seq, expected_ids=[chain_a_gb_id1, chain_a_gb_id2], expected_seq=chain_a_gb_seq)
        os.remove(fn)

    def test_get_seq_gb_multiple(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=(chain_a_gb_dbref + chain_b_gb_dbref + chain_a_gb_seqres +
                                                      chain_b_gb_seqres))
        pdb = PDBReference(pdb_file=fn)
        seq = pdb.get_sequence(chain='B', source='GB')
        self.evaluate_get_sequence(seq=seq, expected_ids=[chain_a_gb_id1, chain_a_gb_id2], expected_seq=chain_a_gb_seq)
        seq2 = pdb.get_sequence(chain='A', source='GB')
        self.evaluate_get_sequence(seq=seq2, expected_ids=[chain_a_gb_id1, chain_a_gb_id2], expected_seq=chain_a_gb_seq)
        os.remove(fn)

    def test_get_seq_failure_bad_source(self):
        fn = write_out_temp_fn(suffix='pdb')
        pdb = PDBReference(pdb_file=fn)
        with self.assertRaises(ValueError):
            pdb.get_sequence(chain='A', source='NCBI')
        os.remove(fn)

    def test_get_seq_pdb_failure_import(self):
        fn = write_out_temp_fn(suffix='pdb')
        pdb = PDBReference(pdb_file=fn)
        with self.assertRaises(AttributeError):
            pdb.get_sequence(chain='A', source='PDB')
        os.remove(fn)

    def test_get_seq_pdb_failure_chain(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb)
        pdb = PDBReference(pdb_file=fn)
        pdb.import_pdb(structure_id='1TES')
        with self.assertRaises(KeyError):
            pdb.get_sequence(chain='B', source='PDB')
        os.remove(fn)

    def test_get_seq_unp_failure_no_entries(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb)
        pdb = PDBReference(pdb_file=fn)
        seq = pdb.get_sequence(chain='A', source='UNP')
        self.assertIsNone(seq[0])
        self.assertIsNone(seq[1])
        os.remove(fn)

    def test_get_seq_unp_failure_bad_chain(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_unp_dbref + chain_a_unp_seqres)
        pdb = PDBReference(pdb_file=fn)
        seq = pdb.get_sequence(chain='B', source='UNP')
        self.assertIsNone(seq[0])
        self.assertIsNone(seq[1])
        os.remove(fn)

    def test_get_seq_gb_failure_no_entries(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb)
        pdb = PDBReference(pdb_file=fn)
        seq = pdb.get_sequence(chain='A', source='GB')
        self.assertIsNone(seq[0])
        self.assertIsNone(seq[1])
        os.remove(fn)

    def test_get_seq_gb_failure_bad_chain(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_gb_dbref + chain_a_gb_seqres)
        pdb = PDBReference(pdb_file=fn)
        seq = pdb.get_sequence(chain='B', source='GB')
        self.assertIsNone(seq[0])
        self.assertIsNone(seq[1])
        os.remove(fn)


class TestParseUniprotHandle(TestCase):

    def test_parse_handle_fail_none(self):
        with self.assertRaises(AttributeError):
            PDBReference._parse_uniprot_handle(None)

    def test_parse_handle(self):
        handle = get_sprot_raw(chain_a_unp_id1)
        seq = PDBReference._parse_uniprot_handle(handle)
        self.assertEqual(str(seq), chain_a_unp_seq)


class TestRetrieveUniprotSeq(TestCase):

    def evaluate_retrieve_uniprot_seq(self, res, expected_id, expected_seq):
        self.assertEqual(res[0], expected_id)
        self.assertEqual(str(res[1]), expected_seq)

    def test_retrieve_acc_none(self):
        res = PDBReference._retrieve_uniprot_seq(None, None, None, None)
        self.assertIsNone(res[0])
        self.assertIsNone(res[1])

    def test_retrieve_acc_id_type1(self):
        res = PDBReference._retrieve_uniprot_seq(db_acc='P46937', db_id=None, seq_start=165, seq_end=209)
        self.evaluate_retrieve_uniprot_seq(res=res, expected_id='P46937',
                                           expected_seq='FEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLSQ')

    def test_retrieve_acc_id_type2(self):
        res = PDBReference._retrieve_uniprot_seq(db_acc=None, db_id='YAP1_HUMAN', seq_start=165, seq_end=209)
        self.evaluate_retrieve_uniprot_seq(res=res, expected_id='YAP1_HUMAN',
                                           expected_seq='FEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLSQ')

    def test_retrieve_acc_id_both_types(self):
        res = PDBReference._retrieve_uniprot_seq(db_acc='P46937', db_id='YAP1_HUMAN', seq_start=165, seq_end=209)
        self.evaluate_retrieve_uniprot_seq(res=res, expected_id='P46937',
                                           expected_seq='FEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLSQ')

    def test_retrieve_acc_id_no_seq_boundaries(self):
        res = PDBReference._retrieve_uniprot_seq(db_acc='P46937', db_id=None, seq_start=None, seq_end=None)
        self.evaluate_retrieve_uniprot_seq(res=res, expected_id='P46937',
                                           expected_seq='MDPGQQPPPQPAPQGQGQPPSQPPQGQGPPSGPGQPAPAATQAAPQAPPAGHQIVHVRGDSE'
                                                        'TDLEALFNAVMNPKTANVPQTVPMRLRKLPDSFFKPPEPKSHSRQASTDAGTAGALTPQHVR'
                                                        'AHSSPASLQLGAVSPGTLTPTGVVSGPAATPTAQHLRQSSFEIPDDVPLPAGWEMAKTSSGQ'
                                                        'RYFLNHIDQTTTWQDPRKAMLSQMNVTAPTSPPVQQNMMNSASGPLPDGWEQAMTQDGEIYY'
                                                        'INHKNKTTSWLDPRLDPRFAMNQRISQSAPVKQPPPLAPQSPQGGVMGGSNSNQQQQMRLQQ'
                                                        'LQMEKERLRLKQQELLRQAMRNINPSTANSPKCQELALRSQLPTLEQDGGTQNPVSSPGMSQ'
                                                        'ELRTMTTNSSDPFLNSGTYHSRDESTDSGLSMSSYSVPRTPDDFLNSVDEMDTGDTINQSTL'
                                                        'PSQQNRFPDYLEAIPGTNVDLGTLEGDGMNIEGEELMPSLQEALSSDILNDMESVLAATKLD'
                                                        'KESFLTWL')


class TestRetrieveGenBankSeq(TestCase):

    def evaluate_retrieve_genbank(self, res, expected_id, expected_seq):
        self.assertEqual(res[0], expected_id)
        self.assertEqual(str(res[1]), expected_seq)

    def test_retrieve_acc_none(self):
        res = PDBReference._retrieve_genbank_seq(db_acc=None, db_id=None, seq_start=None, seq_end=None)
        self.assertIsNone(res[0])
        self.assertIsNone(res[1])

    def test_retrieve_acc_id_type1(self):
        res = PDBReference._retrieve_genbank_seq(db_acc='18312750', db_id=None, seq_start=1, seq_end=302)
        self.evaluate_retrieve_genbank(res=res, expected_id='18312750',
                                       expected_seq='MSQLLQDYLNWENYILRRVDFPTSYVVEGEVVRIEAMPRLYISGMGGSGVVADLIRDFSLTWNWEV'
                                                    'EVIAVKDYFLKARDGLLIAVSYSGNTIETLYTVEYAKRRRIPAVAITTGGRLAQMGVPTVIVPKAS'
                                                    'APRAALPQLLTAALHVVAKVYGIDVKIPEGLEPPNEALIHKLVEEFQKRPTIIAAESMRGVAYRVK'
                                                    'NEFNENAKIEPSVEILPEAHHNWIEGSERAVVALTSPHIPKEHQERVKATVEIVGGSIYAVEMHPK'
                                                    'GVLSFLRDVGIASVKLAEIRGVNPLATPRIDALKRRLQ')

    def test_retrieve_acc_id_type2(self):
        res = PDBReference._retrieve_genbank_seq(db_acc=None, db_id='NP_559417', seq_start=1, seq_end=302)
        self.evaluate_retrieve_genbank(res=res, expected_id='NP_559417',
                                       expected_seq='MSQLLQDYLNWENYILRRVDFPTSYVVEGEVVRIEAMPRLYISGMGGSGVVADLIRDFSLTWNWEV'
                                                    'EVIAVKDYFLKARDGLLIAVSYSGNTIETLYTVEYAKRRRIPAVAITTGGRLAQMGVPTVIVPKAS'
                                                    'APRAALPQLLTAALHVVAKVYGIDVKIPEGLEPPNEALIHKLVEEFQKRPTIIAAESMRGVAYRVK'
                                                    'NEFNENAKIEPSVEILPEAHHNWIEGSERAVVALTSPHIPKEHQERVKATVEIVGGSIYAVEMHPK'
                                                    'GVLSFLRDVGIASVKLAEIRGVNPLATPRIDALKRRLQ')

    def test_retrieve_acc_id_both_types(self):
        res = PDBReference._retrieve_genbank_seq(db_acc='18312750', db_id='NP_559417', seq_start=1, seq_end=302)
        self.evaluate_retrieve_genbank(res=res, expected_id='18312750',
                                       expected_seq='MSQLLQDYLNWENYILRRVDFPTSYVVEGEVVRIEAMPRLYISGMGGSGVVADLIRDFSLTWNWEV'
                                                    'EVIAVKDYFLKARDGLLIAVSYSGNTIETLYTVEYAKRRRIPAVAITTGGRLAQMGVPTVIVPKAS'
                                                    'APRAALPQLLTAALHVVAKVYGIDVKIPEGLEPPNEALIHKLVEEFQKRPTIIAAESMRGVAYRVK'
                                                    'NEFNENAKIEPSVEILPEAHHNWIEGSERAVVALTSPHIPKEHQERVKATVEIVGGSIYAVEMHPK'
                                                    'GVLSFLRDVGIASVKLAEIRGVNPLATPRIDALKRRLQ')

    def test_retrieve_acc_id_no_seq_boundaries(self):
        res = PDBReference._retrieve_genbank_seq(db_acc='18312750', db_id='NP_559417', seq_start=None, seq_end=None)
        self.evaluate_retrieve_genbank(res=res, expected_id='18312750',
                                       expected_seq='MSQLLQDYLNWENYILRRVDFPTSYVVEGEVVRIEAMPRLYISGMGGSGVVADLIRDFSLTWNWEV'
                                                    'EVIAVKDYFLKARDGLLIAVSYSGNTIETLYTVEYAKRRRIPAVAITTGGRLAQMGVPTVIVPKAS'
                                                    'APRAALPQLLTAALHVVAKVYGIDVKIPEGLEPPNEALIHKLVEEFQKRPTIIAAESMRGVAYRVK'
                                                    'NEFNENAKIEPSVEILPEAHHNWIEGSERAVVALTSPHIPKEHQERVKATVEIVGGSIYAVEMHPK'
                                                    'GVLSFLRDVGIASVKLAEIRGVNPLATPRIDALKRRLQ')


class TestParseExternalSequenceAccessions(TestCase):

    def test_parse_external_seq_acc_no_entries(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb)
        acc = PDBReference._parse_external_sequence_accessions(fn)
        self.assertEqual(acc, {})
        os.remove(fn)

    def test_parse_external_seq_acc_single_unp(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_unp_dbref + chain_a_unp_seqres)
        acc = PDBReference._parse_external_sequence_accessions(fn)
        self.assertEqual(acc, {'UNP': {'A': [chain_a_unp_id1, chain_a_unp_id2]}})
        os.remove(fn)

    def test_parse_external_seq_acc_multiple_unp(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=(chain_a_unp_dbref + chain_g_unp_dbref + chain_a_unp_seqres +
                                                      chain_g_unp_seqres))
        acc = PDBReference._parse_external_sequence_accessions(fn)
        self.assertEqual(acc, {'UNP': {'A': [chain_a_unp_id1, chain_a_unp_id2],
                                       'G': [chain_g_unp_id1, chain_g_unp_id2]}})
        os.remove(fn)

    def test_parse_external_seq_acc_single_gb(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_gb_dbref + chain_a_gb_seqres)
        acc = PDBReference._parse_external_sequence_accessions(fn)
        self.assertEqual(acc, {'GB': {'A': [chain_a_gb_id1, chain_a_gb_id2]}})
        os.remove(fn)

    def test_parse_external_seq_acc_multiple_gb(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=(chain_a_gb_dbref + chain_b_gb_dbref + chain_a_gb_seqres +
                                                      chain_b_gb_seqres))
        acc = PDBReference._parse_external_sequence_accessions(fn)
        self.assertEqual(acc, {'GB': {'A': [chain_a_gb_id1, chain_a_gb_id2],
                                      'B': [chain_a_gb_id1, chain_a_gb_id2]}})
        os.remove(fn)

    def test_parse_external_seq_acc_multiple_sources(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=(chain_a_unp_dbref + chain_g_unp_dbref + chain_a_gb_dbref +
                                                      chain_b_gb_dbref + chain_a_unp_seqres + chain_g_unp_seqres +
                                                      chain_a_gb_seqres + chain_b_gb_seqres))
        acc = PDBReference._parse_external_sequence_accessions(fn)
        self.assertEqual(acc, {'UNP': {'A': [chain_a_unp_id1, chain_a_unp_id2],
                                       'G': [chain_g_unp_id1, chain_g_unp_id2]},
                               'GB': {'A': [chain_a_gb_id1, chain_a_gb_id2],
                                      'B': [chain_a_gb_id1, chain_a_gb_id2]}})
        os.remove(fn)


class TestParseExternalSequences(TestCase):

    def test_parse_external_seqs_no_entries(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb)
        pdb = PDBReference(pdb_file=fn)
        external_seqs = pdb._parse_external_sequences()
        self.assertEqual(external_seqs, {})
        os.remove(fn)

    def test_parse_external_seqs_single_unp(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_unp_dbref + chain_a_unp_seqres)
        pdb = PDBReference(pdb_file=fn)
        external_seqs = pdb._parse_external_sequences()
        self.assertEqual(external_seqs, {'UNP': {'A': (chain_a_unp_id1, chain_a_unp_seq)}})
        os.remove(fn)

    def test_parse_external_seqs_multiple_unp(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=(chain_a_unp_dbref + chain_g_unp_dbref + chain_a_unp_seqres +
                                                      chain_g_unp_seqres))
        pdb = PDBReference(pdb_file=fn)
        external_seqs = pdb._parse_external_sequences()
        self.assertEqual(external_seqs, {'UNP': {'A': (chain_a_unp_id1, chain_a_unp_seq),
                                                 'G': (chain_g_unp_id1, chain_g_unp_seq)}})
        os.remove(fn)

    def test_parse_external_seqs_single_gb(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_gb_dbref + chain_a_gb_seqres)
        pdb = PDBReference(pdb_file=fn)
        external_seqs = pdb._parse_external_sequences()
        self.assertEqual(external_seqs, {'GB': {'A': (chain_a_gb_id1, chain_a_gb_seq)}})
        os.remove(fn)

    def test_parse_external_seqs_multiple_gb(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=(chain_a_gb_dbref + chain_b_gb_dbref + chain_a_gb_seqres +
                                                      chain_b_gb_seqres))
        pdb = PDBReference(pdb_file=fn)
        external_seqs = pdb._parse_external_sequences()
        self.assertEqual(external_seqs, {'GB': {'A': (chain_a_gb_id1, chain_a_gb_seq),
                                                'B': (chain_a_gb_id1, chain_a_gb_seq)}})
        os.remove(fn)

    def test_parse_external_seqs_multiple_sources(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=(chain_a_unp_dbref + chain_g_unp_dbref + chain_a_gb_dbref +
                                                      chain_b_gb_dbref + chain_a_unp_seqres + chain_g_unp_seqres +
                                                      chain_a_gb_seqres + chain_b_gb_seqres))
        pdb = PDBReference(pdb_file=fn)
        external_seqs = pdb._parse_external_sequences()
        self.assertEqual(external_seqs, {'UNP': {'A': (chain_a_unp_id1, chain_a_unp_seq),
                                                 'G': (chain_g_unp_id1, chain_g_unp_seq)},
                                         'GB': {'A': (chain_a_gb_id1, chain_a_gb_seq),
                                                'B': (chain_a_gb_id1, chain_a_gb_seq)}})
        os.remove(fn)


class TestDBREFParse(TestCase):

    def test_dbref_parse_pdb(self):
        res = dbref_parse(dbref_line='DBREF  1USC A    1   178  PDB    1USC     1USC             1    178\n')
        self.assertEqual(res['rec_name'], 'DBREF')
        self.assertEqual(res['id_code'], '1USC')
        self.assertEqual(res['chain_id'], 'A')
        self.assertEqual(res['seq_begin'], 1)
        self.assertEqual(res['ins_begin'], '')
        self.assertEqual(res['seq_end'], 178)
        self.assertEqual(res['ins_end'], '')
        self.assertEqual(res['db'], 'PDB')
        self.assertEqual(res['db_acc'], '1USC')
        self.assertEqual(res['db_id'], '1USC')
        self.assertEqual(res['db_seq_begin'], 1)
        self.assertEqual(res['db_ins_begin'], '')
        self.assertEqual(res['db_seq_end'], 178)
        self.assertEqual(res['db_ins_end'], '')

    def test_dbref_parse_gb(self):
        res = dbref_parse(dbref_line='DBREF  1X9H A    1   302  GB     18312750 NP_559417        1    302\n')
        self.assertEqual(res['rec_name'], 'DBREF')
        self.assertEqual(res['id_code'], '1X9H')
        self.assertEqual(res['chain_id'], 'A')
        self.assertEqual(res['seq_begin'], 1)
        self.assertEqual(res['ins_begin'], '')
        self.assertEqual(res['seq_end'], 302)
        self.assertEqual(res['ins_end'], '')
        self.assertEqual(res['db'], 'GB')
        self.assertEqual(res['db_acc'], '18312750')
        self.assertEqual(res['db_id'], 'NP_559417')
        self.assertEqual(res['db_seq_begin'], 1)
        self.assertEqual(res['db_ins_begin'], '')
        self.assertEqual(res['db_seq_end'], 302)
        self.assertEqual(res['db_ins_end'], '')

    def test_dbref_parse_unp(self):
        res = dbref_parse(dbref_line='DBREF  4REX A  165   209  UNP    P46937   YAP1_HUMAN     165    209\n')
        self.assertEqual(res['rec_name'], 'DBREF')
        self.assertEqual(res['id_code'], '4REX')
        self.assertEqual(res['chain_id'], 'A')
        self.assertEqual(res['seq_begin'], 165)
        self.assertEqual(res['ins_begin'], '')
        self.assertEqual(res['seq_end'], 209)
        self.assertEqual(res['ins_end'], '')
        self.assertEqual(res['db'], 'UNP')
        self.assertEqual(res['db_acc'], 'P46937')
        self.assertEqual(res['db_id'], 'YAP1_HUMAN')
        self.assertEqual(res['db_seq_begin'], 165)
        self.assertEqual(res['db_ins_begin'], '')
        self.assertEqual(res['db_seq_end'], 209)
        self.assertEqual(res['db_ins_end'], '')

    def test_dbref_parse_fail_dbref1(self):
        with self.assertRaises(ValueError):
            dbref_parse(dbref_line='DBREF1 2J83 A   61   322  XXXXXX               YYYYYYYYYYYYYYYYYYYY\n')

    def test_dbref_parse_fail_dbref2(self):
        with self.assertRaises(ValueError):
            dbref_parse(dbref_line='DBREF2 2J83 A     ZZZZZZZZZZZZZZZZZZZZZZ     nnnnnnnnnn  mmmmmmmmmm\n')

    def test_dbref_parse_fail_empty_right_len(self):
        with self.assertRaises(ValueError):
            dbref_parse(dbref_line='                                                                   \n')

    def test_dbref_parse_fail_empty(self):
        with self.assertRaises(ValueError):
            dbref_parse(dbref_line='')


if __name__ == '__main__':
    unittest.main()
