"""
Created on May 28, 2019

@author: Daniel Konecki
"""
import os
import sys
import unittest
from time import time
from shutil import rmtree
from unittest import TestCase
from Bio.Seq import Seq
from Bio.SeqIO import write
from Bio.SeqRecord import SeqRecord
from multiprocessing import cpu_count, Lock

#
from dotenv import find_dotenv, load_dotenv
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)
source_code_path = os.path.join(os.environ.get('PROJECT_PATH'), 'src')
# Add the project path to the python path so the required classes can be imported
if source_code_path not in sys.path:
    sys.path.append(os.path.join(os.environ.get('PROJECT_PATH'), 'src'))
#

from SupportingClasses.EvolutionaryTraceAlphabet import FullIUPACProtein
from SupportingClasses.DataSetGenerator import (DataSetGenerator, import_protein_list, download_pdb,
                                                parse_query_sequence, init_pdb_processing_pool, pdb_processing,
                                                blast_query_sequence, filter_blast_sequences, align_sequences,
                                                identity_filter, init_filtering_and_alignment_pool,
                                                filtering_and_alignment)
from Testing.test_PDBReference import (chain_a_pdb, chain_a_gb_dbref, chain_b_gb_dbref, chain_a_gb_seqres,
                                       chain_b_gb_seqres, chain_a_gb_id1, chain_a_gb_id2, chain_a_gb_seq,
                                       chain_a_unp_dbref, chain_g_unp_dbref, chain_a_unp_seqres, chain_g_unp_seqres,
                                       chain_a_unp_seq, chain_g_unp_seq, chain_g_unp_seq, chain_a_unp_id1,
                                       chain_a_unp_id2, chain_g_unp_id1, chain_g_unp_id2)
from Testing.test_Base import write_out_temp_fn


class TestImportProteinList(TestCase):

    def test_empty_list(self):
        test_fn = 'test_list'
        with open(test_fn, 'w'):
            pass
        protein_list = import_protein_list(protein_list_fn=test_fn)
        self.assertIsInstance(protein_list, dict)
        self.assertEqual(protein_list, {})
        os.remove(test_fn)

    def test_single_element(self):
        test_fn = 'test_list'
        with open(test_fn, 'w') as handle:
            handle.write('4rexA')
        protein_list = import_protein_list(protein_list_fn=test_fn)
        self.assertIsInstance(protein_list, dict)
        self.assertEqual(protein_list, {'4rexA': {'PDB': '4rex', 'Chain': 'A'}})
        os.remove(test_fn)

    def test_multiple_elements(self):
        test_fn = 'test_list'
        with open(test_fn, 'w') as handle:
            handle.write('4rexA\n6cm4A')
        protein_list = import_protein_list(protein_list_fn=test_fn)
        self.assertIsInstance(protein_list, dict)
        self.assertEqual(protein_list, {'4rexA': {'PDB': '4rex', 'Chain': 'A'},
                                        '6cm4A': {'PDB': '6cm4', 'Chain': 'A'}})
        os.remove(test_fn)

    def test_bad_element(self):
        test_fn = 'test_list'
        with open(test_fn, 'w') as handle:
            handle.write('A4rex')
        with self.assertRaises(ValueError):
            import_protein_list(protein_list_fn=test_fn)
        os.remove(test_fn)

    def test_multiple_elements_one_bad_element(self):
        test_fn = 'test_list'
        with open(test_fn, 'w') as handle:
            handle.write('4rexA\nA6cm4')
        with self.assertRaises(ValueError):
            import_protein_list(protein_list_fn=test_fn)
        os.remove(test_fn)


class TestDownloadPDB(TestCase):

    def setUp(self):
        os.mkdir('PDB')

    def tearDown(self):
        rmtree('PDB')

    def test_download_pdb(self):
        expected_pdb_fn = os.path.join('PDB', 're', 'pdb4rex.ent')
        pdb_fn = download_pdb(pdb_path='PDB', pdb_id='4rex')
        self.assertEqual(pdb_fn, expected_pdb_fn)
        self.assertTrue(os.path.isfile(pdb_fn))

    def test_download_pdb_obsolete(self):
        expected_pdb_fn = os.path.join('PDB', 'obsolete', 'hr', 'pdb4hrz.ent')
        pdb_fn = download_pdb(pdb_path='PDB', pdb_id='4hrz')
        self.assertEqual(pdb_fn, expected_pdb_fn)
        self.assertTrue(os.path.isfile(pdb_fn))

    def test_download_pdb_bad_id(self):
        pdb_fn = download_pdb(pdb_path='PDB', pdb_id='xer4')
        self.assertIsNone(pdb_fn)

    def test_download_pdb_None(self):
        with self.assertRaises(ValueError):
            download_pdb(pdb_path='PDB', pdb_id=None)


class TestParseQuerySequence(TestCase):

    def setUp(self):
        os.mkdir('Seqs')

    def tearDown(self):
        rmtree('Seqs')

    def test_parse_query_sequence_PDB(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb)
        seq, seq_len, seq_fn, chain, acc = parse_query_sequence(protein_id='1tesA', pdb_id='1tes', chain_id='A',
                                                                sequence_path='Seqs/', pdb_fn=fn, sources=['PDB'])
        self.assertEqual(seq.seq, 'MET')
        self.assertEqual(seq_len, 3)
        self.assertEqual(seq_fn, 'Seqs/1tesA.fasta')
        self.assertEqual(chain, 'A')
        self.assertEqual(acc, '1tes')
        os.remove(fn)

    def test_parse_query_sequence_UNP(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_unp_dbref + chain_a_unp_seqres + chain_a_pdb)
        seq, seq_len, seq_fn, chain, acc = parse_query_sequence(protein_id='1tesA', pdb_id='1tes', chain_id='A',
                                                                sequence_path='Seqs/', pdb_fn=fn, sources=['UNP'])
        self.assertEqual(seq.seq, chain_a_unp_seq[18:147])
        self.assertEqual(seq_len, len(chain_a_unp_seq[18:147]))
        self.assertEqual(seq_fn, 'Seqs/1tesA.fasta')
        self.assertEqual(chain, 'A')
        self.assertIn(acc, [chain_a_unp_id1, chain_a_unp_id2])
        os.remove(fn)

    def test_parse_query_sequence_UNP_multiple(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=(chain_a_unp_dbref + chain_g_unp_dbref + chain_a_unp_seqres +
                                                      chain_g_unp_seqres + chain_a_pdb))
        seq, seq_len, seq_fn, chain, acc = parse_query_sequence(protein_id='1tesA', pdb_id='1tes', chain_id='A',
                                                                sequence_path='Seqs/', pdb_fn=fn, sources=['UNP'])
        self.assertEqual(seq.seq, chain_a_unp_seq[18:147])
        self.assertEqual(seq_len, len(chain_a_unp_seq[18:147]))
        self.assertEqual(seq_fn, 'Seqs/1tesA.fasta')
        self.assertEqual(chain, 'A')
        self.assertIn(acc, [chain_a_unp_id1, chain_a_unp_id2])
        seq, seq_len, seq_fn, chain, acc = parse_query_sequence(protein_id='1tesG', pdb_id='1tes', chain_id='G',
                                                                sequence_path='Seqs/', pdb_fn=fn, sources=['UNP'])
        self.assertEqual(seq.seq, chain_g_unp_seq[20:94])
        self.assertEqual(seq_len, len(chain_g_unp_seq[20:94]))
        self.assertEqual(seq_fn, 'Seqs/1tesG.fasta')
        self.assertEqual(chain, 'G')
        self.assertIn(acc, [chain_g_unp_id1, chain_g_unp_id2])
        os.remove(fn)

    def test_parse_query_sequence_GB(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_gb_dbref + chain_a_gb_seqres + chain_a_pdb)
        seq, seq_len, seq_fn, chain, acc = parse_query_sequence(protein_id='1tesA', pdb_id='1tes', chain_id='A',
                                                                sequence_path='Seqs/', pdb_fn=fn, sources=['GB'])
        self.assertEqual(seq.seq, chain_a_gb_seq[171:338])
        self.assertEqual(seq_len, len(chain_a_gb_seq[171:338]))
        self.assertEqual(seq_fn, 'Seqs/1tesA.fasta')
        self.assertEqual(chain, 'A')
        self.assertIn(acc, [chain_a_gb_id1, chain_a_gb_id2])
        os.remove(fn)

    def test_parse_query_sequence_secondary_source(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb)
        seq, seq_len, seq_fn, chain, acc = parse_query_sequence(protein_id='1tesA', pdb_id='1tes', chain_id='A',
                                                                sequence_path='Seqs/', pdb_fn=fn,
                                                                sources=['UNP', 'GB', 'PDB'])
        self.assertEqual(seq.seq, 'MET')
        self.assertEqual(seq_len, 3)
        self.assertEqual(seq_fn, 'Seqs/1tesA.fasta')
        self.assertEqual(chain, 'A')
        self.assertEqual(acc, '1tes')
        os.remove(fn)

    def test_parse_query_sequence_multiple_references_success(self):
        chain_a_dbref_first = 'DBREF  2RH1 A    1   230  UNP    P07550   ADRB2_HUMAN      1    230             \n'
        chain_a_dbref_second = 'DBREF  2RH1 A  263   365  UNP    P07550   ADRB2_HUMAN    263    365    \n'
        chain_a_seq = 'MGQPGNGSAFLLAPNGSHAPDHDVTQERDEVWVVGMGIVMSLIVLAIVFGNVLVITAIAKFERLQTVTNYFITSLACADLVMGLAVVPFGAAHIL' \
                      'MKMWTFGNFWCEFWTSIDVLCVTASIETLCVIAVDRYFAITSPFKYQSLLTKNKARVIILMVWIVSGLTSFLPIQMHWYRATHQEAINCYANETC' \
                      'CDFFTNQAYAIASSIVSFYVPLVIMVFVYSRVFQEAKRQLQKIDKSEGRFHVQNLSQVEQDGRTGHGLRRSSKFCLKEHKALKTLGIIMGTFTLC' \
                      'WLPFFIVNIVHVIQDNLIRKEVYILLNWIGYVNSGFNPLIYCRSPDFRIAFQELLCLRRSSLKAYGNGYSSNGNTGEQSGYHVEQEKENKLLCED' \
                      'LPGTEDFVGHQGTVPSDNIDSQGRNCSTNDSLL'
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_dbref_first + chain_a_dbref_second + chain_a_pdb)
        seq, seq_len, seq_fn, chain, acc = parse_query_sequence(protein_id='2rh1A', pdb_id='2rh1', chain_id='A',
                                                                sequence_path='Seqs/', pdb_fn=fn, sources=['UNP'])
        self.assertEqual(seq.seq, chain_a_seq[0:365])
        self.assertEqual(seq_len, len(chain_a_seq[0:365]))
        self.assertEqual(seq_fn, 'Seqs/2rh1A.fasta')
        self.assertEqual(chain, 'A')
        self.assertIn(acc, ['P07550', 'ADRB2_HUMAN'])
        os.remove(fn)

    def test_parse_query_sequence_multiple_references_failure(self):
        chain_a_dbref_first = 'DBREF  2RH1 A    1   230  UNP    P07550   ADRB2_HUMAN      1    230             \n'
        chain_a_dbref_second = 'DBREF  2RH1 A 1002  1161  UNP    P00720   LYS_BPT4         2    161             \n'
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_dbref_first + chain_a_dbref_second + chain_a_pdb)
        seq, seq_len, seq_fn, chain, acc = parse_query_sequence(protein_id='2rh1A', pdb_id='2rh1', chain_id='A',
                                                                sequence_path='Seqs/', pdb_fn=fn, sources=['UNP'])
        self.assertIsNone(seq)
        self.assertEqual(seq_len, 0)
        self.assertIsNone(seq_fn)
        self.assertEqual(chain, 'A')
        self.assertEqual(acc, 'INSPECT MANUALLY')
        os.remove(fn)

    def test_parse_query_sequence_fail_no_UNP(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb)
        seq, seq_len, seq_fn, chain, acc = parse_query_sequence(protein_id='1tesA', pdb_id='1tes', chain_id='A',
                                                                sequence_path='Seqs/', pdb_fn=fn,
                                                                sources=['UNP'])
        self.assertIsNone(seq)
        self.assertEqual(seq_len, 0)
        self.assertIsNone(seq_fn)
        self.assertEqual(chain, 'A')
        self.assertEqual(acc, None)
        os.remove(fn)

    def test_parse_query_sequence_fail_no_GB(self):
        fn = write_out_temp_fn(suffix='pdb', out_str=chain_a_pdb)
        seq, seq_len, seq_fn, chain, acc = parse_query_sequence(protein_id='1tesA', pdb_id='1tes', chain_id='A',
                                                                sequence_path='Seqs/', pdb_fn=fn,
                                                                sources=['GB'])
        self.assertIsNone(seq)
        self.assertEqual(seq_len, 0)
        self.assertIsNone(seq_fn)
        self.assertEqual(chain, 'A')
        self.assertEqual(acc, None)
        os.remove(fn)


class TestPDBProcessing(TestCase):

    def setUp(self):
        os.mkdir('PDB')
        os.mkdir('Seqs')

    def tearDown(self):
        rmtree('PDB')
        rmtree('Seqs')

    def test_pdb_processing_pdb(self):
        expected_pdb_fn = os.path.join('PDB', 're', 'pdb4rex.ent')
        expected_seq = 'GAMGFEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLS'
        init_pdb_processing_pool(pdb_path='PDB/', sequence_path='Seqs/', lock=Lock(), sources=['PDB'], verbose=False)
        p_id, data = pdb_processing('4rexA', '4rex', 'A')
        self.assertEqual(p_id, '4rexA')
        self.assertEqual(data['PDB_FN'], expected_pdb_fn)
        self.assertEqual(data['Chain'], 'A')
        self.assertEqual(str(data['Sequence'].seq), expected_seq)
        self.assertEqual(data['Length'], len(expected_seq))
        self.assertEqual(data['Seq_Fasta'], 'Seqs/4rexA.fasta')
        self.assertEqual(data['Accession'], '4rex')

    def test_pdb_processing_unp(self):
        expected_pdb_fn = os.path.join('PDB', 're', 'pdb4rex.ent')
        expected_seq = 'MDPGQQPPPQPAPQGQGQPPSQPPQGQGPPSGPGQPAPAATQAAPQAPPAGHQIVHVRGDSETDLEALFNAVMNPKTANVPQTVPMRLRKLPDS'\
                       'FFKPPEPKSHSRQASTDAGTAGALTPQHVRAHSSPASLQLGAVSPGTLTPTGVVSGPAATPTAQHLRQSSFEIPDDVPLPAGWEMAKTSSGQRY'\
                       'FLNHIDQTTTWQDPRKAMLSQMNVTAPTSPPVQQNMMNSASGPLPDGWEQAMTQDGEIYYINHKNKTTSWLDPRLDPRFAMNQRISQSAPVKQP'\
                       'PPLAPQSPQGGVMGGSNSNQQQQMRLQQLQMEKERLRLKQQELLRQAMRNINPSTANSPKCQELALRSQLPTLEQDGGTQNPVSSPGMSQELRT'\
                       'MTTNSSDPFLNSGTYHSRDESTDSGLSMSSYSVPRTPDDFLNSVDEMDTGDTINQSTLPSQQNRFPDYLEAIPGTNVDLGTLEGDGMNIEGEEL'\
                       'MPSLQEALSSDILNDMESVLAATKLDKESFLTWL'
        init_pdb_processing_pool(pdb_path='PDB/', sequence_path='Seqs/', lock=Lock(), sources=['UNP'], verbose=False)
        p_id, data = pdb_processing('4rexA', '4rex', 'A')
        self.assertEqual(p_id, '4rexA')
        self.assertEqual(data['PDB_FN'], expected_pdb_fn)
        self.assertEqual(data['Chain'], 'A')
        self.assertEqual(str(data['Sequence'].seq), expected_seq[164:209])
        self.assertEqual(data['Length'], len(expected_seq[164:209]))
        self.assertEqual(data['Seq_Fasta'], 'Seqs/4rexA.fasta')
        self.assertIn(data['Accession'], ['P46937', 'YAP1_HUMAN'])

    def test_pdb_processing_gb(self):
        expected_pdb_fn = os.path.join('PDB', 'x9', 'pdb1x9h.ent')
        expected_seq = 'MSQLLQDYLNWENYILRRVDFPTSYVVEGEVVRIEAMPRLYISGMGGSGVVADLIRDFSLTWNWEVEVIAVKDYFLKARDGLLIAVSYSGNTIE'\
                       'TLYTVEYAKRRRIPAVAITTGGRLAQMGVPTVIVPKASAPRAALPQLLTAALHVVAKVYGIDVKIPEGLEPPNEALIHKLVEEFQKRPTIIAAE'\
                       'SMRGVAYRVKNEFNENAKIEPSVEILPEAHHNWIEGSERAVVALTSPHIPKEHQERVKATVEIVGGSIYAVEMHPKGVLSFLRDVGIASVKLAE'\
                       'IRGVNPLATPRIDALKRRLQ'
        init_pdb_processing_pool(pdb_path='PDB/', sequence_path='Seqs/', lock=Lock(), sources=['GB'], verbose=False)
        p_id, data = pdb_processing('1x9hA', '1x9h', 'A')
        self.assertEqual(p_id, '1x9hA')
        self.assertEqual(data['PDB_FN'], expected_pdb_fn)
        self.assertEqual(data['Chain'], 'A')
        self.assertEqual(str(data['Sequence'].seq), expected_seq[0:302])
        self.assertEqual(data['Length'], len(expected_seq[0:302]))
        self.assertEqual(data['Seq_Fasta'], 'Seqs/1x9hA.fasta')
        self.assertIn(data['Accession'], ['NP_559417', '18312750'])

    def test_pdb_processing_secondary_unp(self):
        expected_pdb_fn = os.path.join('PDB', 'x9', 'pdb1x9h.ent')
        expected_seq = 'SQLLQDYLNWENYILRRVDFPTSYVVEGEVVRIEAMPRLYISGMGGSGVVADLIRDFSLTWNWEVEVIAVKDYFLKARDGLLIAVSYSGNTIET'\
                       'LYTVEYAKRRRIPAVAITTGGRLAQMGVPTVIVPKASAPRAALPQLLTAALHVVAKVYGIDVKIPEGLEPPNEALIHKLVEEFQKRPTIIAAES'\
                       'MRGVAYRVKNEFNENAKIEPSVEILPEAHHNWIEGSERAVVALTSPHIPKEHQERVKATVEIVGGSIYAVEMHPKGVLSFLRDVGIASVKLAEI'\
                       'RGVNPLATPRIDALKRRLQ'
        init_pdb_processing_pool(pdb_path='PDB/', sequence_path='Seqs/', lock=Lock(), sources=['UNP', 'PDB'],
                                 verbose=False)
        p_id, data = pdb_processing('1x9hA', '1x9h', 'A')
        self.assertEqual(p_id, '1x9hA')
        self.assertEqual(data['PDB_FN'], expected_pdb_fn)
        self.assertEqual(data['Chain'], 'A')
        self.assertEqual(str(data['Sequence'].seq), expected_seq)
        self.assertEqual(data['Length'], len(expected_seq))
        self.assertEqual(data['Seq_Fasta'], 'Seqs/1x9hA.fasta')
        self.assertEqual(data['Accession'], '1x9h')

    def test_pdb_processing_secondary_gb(self):
        expected_pdb_fn = os.path.join('PDB', 're', 'pdb4rex.ent')
        expected_seq = 'GAMGFEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLS'
        init_pdb_processing_pool(pdb_path='PDB/', sequence_path='Seqs/', lock=Lock(), sources=['GB', 'PDB'],
                                 verbose=False)
        p_id, data = pdb_processing('4rexA', '4rex', 'A')
        self.assertEqual(p_id, '4rexA')
        self.assertEqual(data['PDB_FN'], expected_pdb_fn)
        self.assertEqual(data['Chain'], 'A')
        self.assertEqual(str(data['Sequence'].seq), expected_seq)
        self.assertEqual(data['Length'], len(expected_seq))
        self.assertEqual(data['Seq_Fasta'], 'Seqs/4rexA.fasta')
        self.assertEqual(data['Accession'], '4rex')

    def test_pdb_processing_no_unp(self):
        expected_pdb_fn = os.path.join('PDB', 'x9', 'pdb1x9h.ent')
        init_pdb_processing_pool(pdb_path='PDB/', sequence_path='Seqs/', lock=Lock(), sources=['UNP'], verbose=False)
        p_id, data = pdb_processing('1x9hA', '1x9h', 'A')
        self.assertEqual(p_id, '1x9hA')
        self.assertEqual(data['PDB_FN'], expected_pdb_fn)
        self.assertEqual(data['Chain'], 'A')
        self.assertIsNone(data['Sequence'])
        self.assertEqual(data['Length'], 0)
        self.assertIsNone(data['Seq_Fasta'])
        self.assertIsNone(data['Accession'])

    def test_pdb_processing_no_gb(self):
        expected_pdb_fn = os.path.join('PDB', 're', 'pdb4rex.ent')
        init_pdb_processing_pool(pdb_path='PDB/', sequence_path='Seqs/', lock=Lock(), sources=['GB'], verbose=False)
        p_id, data = pdb_processing('4rexA', '4rex', 'A')
        self.assertEqual(p_id, '4rexA')
        self.assertEqual(data['PDB_FN'], expected_pdb_fn)
        self.assertEqual(data['Chain'], 'A')
        self.assertIsNone(data['Sequence'])
        self.assertEqual(data['Length'], 0)
        self.assertIsNone(data['Seq_Fasta'])
        self.assertIsNone(data['Accession'])

    def test_pdb_processing_complicated_dbref_failure(self):
        expected_pdb_fn = os.path.join('PDB', 'rh', 'pdb2rh1.ent')
        init_pdb_processing_pool(pdb_path='PDB/', sequence_path='Seqs/', lock=Lock(), sources=['UNP'], verbose=False)
        p_id, data = pdb_processing('2rh1A', '2rh1', 'A')
        self.assertEqual(p_id, '2rh1A')
        self.assertEqual(data['PDB_FN'], expected_pdb_fn)
        self.assertEqual(data['Chain'], 'A')
        self.assertIsNone(data['Sequence'])
        self.assertEqual(data['Length'], 0)
        self.assertIsNone(data['Seq_Fasta'])
        self.assertIn(data['Accession'], 'INSPECT MANUALLY')


class TestDataSetGeneratorIdentifyProteinSequences(TestCase):
    def setUp(self):
        os.mkdir('ProteinLists')
        with open('./ProteinLists/test.txt', 'w') as handle:
            handle.write('1uscB\n4rexA\n1x9hB\n2rh1A')

    def tearDown(self):
        rmtree('ProteinLists')
        rmtree('PDB')
        rmtree('Sequences')
        rmtree('BLAST')
        rmtree('Filtered_BLAST')
        rmtree('Alignments')
        rmtree('Filtered_Alignments')
        rmtree('Final_Alignments')

    def test_datasetgenerator_identify_protein_sequences_pdb_only_single_process(self):
        expected_seq_1usc = 'MRSYRAQGPLPGFYHYYPGVPAVVGVRVEERVNFCPAVWNTGLSADPPLFGVSISPKRFTHGLLLKARRFSASFHPFGQKDLVHWLGSH'\
                            'SGREVDKGQAPHFLGHTGVPILEGAYAAYELELLEVHTFGDHDLFVGRVVAVWEEEGLLDEKGRPKPGLALLYYGKGLYGRPAEETFAP'
        expected_seq_4rex = 'GAMGFEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLS'
        expected_seq_1x9h = 'SQLLQDYLNWENYILRRVDFPTSYVVEGEVVRIEAMPRLYISGMGGSGVVADLIRDFSLTWNWEVEVIAVKDYFLKARDGLLIAVSYSG'\
                            'NTIETLYTVEYAKRRRIPAVAITTGGRLAQMGVPTVIVPKASAPRAALPQLLTAALHVVAKVYGIDVKIPEGLEPPNEALIHKLVEEFQ'\
                            'KRPTIIAAESMRGVAYRVKNEFNENAKIEPSVEILPEAHHNWIEGSERAVVALTSPHIPKEHQERVKATVEIVGGSIYAVEMHPKGVLS'\
                            'FLRDVGIASVKLAEIRGVNPLATPRIDALKRRLQ'
        expected_seq_2rh1 = 'DEVWVVGMGIVMSLIVLAIVFGNVLVITAIAKFERLQTVTNYFITSLACADLVMGLAVVPFGAAHILMKMWTFGNFWCEFWTSIDVLCV'\
                            'TASIETLCVIAVDRYFAITSPFKYQSLLTKNKARVIILMVWIVSGLTSFLPIQMHWYRATHQEAINCYAEETCCDFFTNQAYAIASSIV'\
                            'SFYVPLVIMVFVYSRVFQEAKRQL'\
                            'KFCLKEHKALKTLGIIMGTFTLCWLPFFIVNIVHVIQDNLIRKEVYILLNWIGYVNSGFNPLIYCRSPDFRIAFQELLCL'\
                            'NIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDS'\
                            'LDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAY'
        dsg = DataSetGenerator(input_path='./')
        fn, unique, seqs = dsg.identify_protein_sequences(data_set_name='test',
                                                          protein_list_fn='./ProteinLists/test.txt', sources=['PDB'],
                                                          processes=1)
        self.assertEqual(fn, './Sequences/test.fasta')
        self.assertEqual(len(set(unique).intersection({'1uscB', '4rexA', '1x9hB', '2rh1A'})), 4)
        self.assertEqual(seqs, {expected_seq_1usc: ['1uscB'], expected_seq_4rex: ['4rexA'],
                                expected_seq_1x9h: ['1x9hB'], expected_seq_2rh1: ['2rh1A']})
        self.assertEqual(len(set(dsg.protein_data.keys()) | {'1uscB', '4rexA', '1x9hB', '2rh1A'}), 4)
        expected_dict = {'1uscB': {'PDB': '1usc', 'Chain': 'B', 'PDB_FN': './PDB/us/pdb1usc.ent',
                                   'Sequence': expected_seq_1usc, 'Length': len(expected_seq_1usc),
                                   'Seq_Fasta': './Sequences/1uscB.fasta', 'Accession': '1usc'},
                         '4rexA': {'PDB': '4rex', 'Chain': 'A', 'PDB_FN': './PDB/re/pdb4rex.ent',
                                   'Sequence': expected_seq_4rex, 'Length': len(expected_seq_4rex),
                                   'Seq_Fasta': './Sequences/4rexA.fasta', 'Accession': '4rex'},
                         '1x9hB': {'PDB': '1x9h', 'Chain': 'B', 'PDB_FN': './PDB/x9/pdb1x9h.ent',
                                   'Sequence': expected_seq_1x9h, 'Length': len(expected_seq_1x9h),
                                   'Seq_Fasta': './Sequences/1x9hB.fasta', 'Accession': '1x9h'},
                         '2rh1A': {'PDB': '2rh1', 'Chain': 'A', 'PDB_FN': './PDB/rh/pdb2rh1.ent',
                                   'Sequence': expected_seq_2rh1, 'Length': len(expected_seq_2rh1),
                                   'Seq_Fasta': './Sequences/2rh1A.fasta', 'Accession': '2rh1'}}
        for p_id in dsg.protein_data:
            for key in dsg.protein_data[p_id]:
                if key == 'Sequence':
                    self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_dict[p_id][key])
                else:
                    self.assertEqual(dsg.protein_data[p_id][key], expected_dict[p_id][key])

    def test_datasetgenerator_identify_protein_sequences_pdb_only_multiple_processes(self):
        expected_seq_1usc = 'MRSYRAQGPLPGFYHYYPGVPAVVGVRVEERVNFCPAVWNTGLSADPPLFGVSISPKRFTHGLLLKARRFSASFHPFGQKDLVHWLGSH'\
                            'SGREVDKGQAPHFLGHTGVPILEGAYAAYELELLEVHTFGDHDLFVGRVVAVWEEEGLLDEKGRPKPGLALLYYGKGLYGRPAEETFAP'
        expected_seq_4rex = 'GAMGFEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLS'
        expected_seq_1x9h = 'SQLLQDYLNWENYILRRVDFPTSYVVEGEVVRIEAMPRLYISGMGGSGVVADLIRDFSLTWNWEVEVIAVKDYFLKARDGLLIAVSYSG'\
                            'NTIETLYTVEYAKRRRIPAVAITTGGRLAQMGVPTVIVPKASAPRAALPQLLTAALHVVAKVYGIDVKIPEGLEPPNEALIHKLVEEFQ'\
                            'KRPTIIAAESMRGVAYRVKNEFNENAKIEPSVEILPEAHHNWIEGSERAVVALTSPHIPKEHQERVKATVEIVGGSIYAVEMHPKGVLS'\
                            'FLRDVGIASVKLAEIRGVNPLATPRIDALKRRLQ'
        expected_seq_2rh1 = 'DEVWVVGMGIVMSLIVLAIVFGNVLVITAIAKFERLQTVTNYFITSLACADLVMGLAVVPFGAAHILMKMWTFGNFWCEFWTSIDVLCV'\
                            'TASIETLCVIAVDRYFAITSPFKYQSLLTKNKARVIILMVWIVSGLTSFLPIQMHWYRATHQEAINCYAEETCCDFFTNQAYAIASSIV'\
                            'SFYVPLVIMVFVYSRVFQEAKRQL'\
                            'KFCLKEHKALKTLGIIMGTFTLCWLPFFIVNIVHVIQDNLIRKEVYILLNWIGYVNSGFNPLIYCRSPDFRIAFQELLCL'\
                            'NIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDS'\
                            'LDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAY'
        dsg = DataSetGenerator(input_path='./')
        fn, unique, seqs = dsg.identify_protein_sequences(data_set_name='test',
                                                          protein_list_fn='./ProteinLists/test.txt', sources=['PDB'],
                                                          processes=2)
        self.assertEqual(fn, './Sequences/test.fasta')
        self.assertEqual(len(set(unique).intersection({'1uscB', '4rexA', '1x9hB', '2rh1A'})), 4)
        self.assertEqual(seqs, {expected_seq_1usc: ['1uscB'], expected_seq_4rex: ['4rexA'],
                                expected_seq_1x9h: ['1x9hB'], expected_seq_2rh1: ['2rh1A']})
        self.assertEqual(len(set(dsg.protein_data.keys()) | {'1uscB', '4rexA', '1x9hB', '2rh1A'}), 4)
        expected_dict = {'1uscB': {'PDB': '1usc', 'Chain': 'B', 'PDB_FN': './PDB/us/pdb1usc.ent',
                                   'Sequence': expected_seq_1usc, 'Length': len(expected_seq_1usc),
                                   'Seq_Fasta': './Sequences/1uscB.fasta', 'Accession': '1usc'},
                         '4rexA': {'PDB': '4rex', 'Chain': 'A', 'PDB_FN': './PDB/re/pdb4rex.ent',
                                   'Sequence': expected_seq_4rex, 'Length': len(expected_seq_4rex),
                                   'Seq_Fasta': './Sequences/4rexA.fasta', 'Accession': '4rex'},
                         '1x9hB': {'PDB': '1x9h', 'Chain': 'B', 'PDB_FN': './PDB/x9/pdb1x9h.ent',
                                   'Sequence': expected_seq_1x9h, 'Length': len(expected_seq_1x9h),
                                   'Seq_Fasta': './Sequences/1x9hB.fasta', 'Accession': '1x9h'},
                         '2rh1A': {'PDB': '2rh1', 'Chain': 'A', 'PDB_FN': './PDB/rh/pdb2rh1.ent',
                                   'Sequence': expected_seq_2rh1, 'Length': len(expected_seq_2rh1),
                                   'Seq_Fasta': './Sequences/2rh1A.fasta', 'Accession': '2rh1'}}
        for p_id in dsg.protein_data:
            for key in dsg.protein_data[p_id]:
                if key == 'Sequence':
                    self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_dict[p_id][key])
                else:
                    self.assertEqual(dsg.protein_data[p_id][key], expected_dict[p_id][key])

    def test_datasetgenerator_identify_protein_sequences_unp_only_single_process(self):
        expected_seq_4rex = 'MDPGQQPPPQPAPQGQGQPPSQPPQGQGPPSGPGQPAPAATQAAPQAPPAGHQIVHVRGDSETDLEALFNAVMNPKTANVPQTVPMRLR'\
                            'KLPDSFFKPPEPKSHSRQASTDAGTAGALTPQHVRAHSSPASLQLGAVSPGTLTPTGVVSGPAATPTAQHLRQSSFEIPDDVPLPAGWE'\
                            'MAKTSSGQRYFLNHIDQTTTWQDPRKAMLSQMNVTAPTSPPVQQNMMNSASGPLPDGWEQAMTQDGEIYYINHKNKTTSWLDPRLDPRF'\
                            'AMNQRISQSAPVKQPPPLAPQSPQGGVMGGSNSNQQQQMRLQQLQMEKERLRLKQQELLRQAMRNINPSTANSPKCQELALRSQLPTLE'\
                            'QDGGTQNPVSSPGMSQELRTMTTNSSDPFLNSGTYHSRDESTDSGLSMSSYSVPRTPDDFLNSVDEMDTGDTINQSTLPSQQNRFPDYL'\
                            'EAIPGTNVDLGTLEGDGMNIEGEELMPSLQEALSSDILNDMESVLAATKLDKESFLTWL'
        dsg = DataSetGenerator(input_path='./')
        fn, unique, seqs = dsg.identify_protein_sequences(data_set_name='test',
                                                          protein_list_fn='./ProteinLists/test.txt', sources=['UNP'],
                                                          processes=1)
        self.assertEqual(fn, './Sequences/test.fasta')
        self.assertEqual(len(set(unique).intersection({'4rexA'})), 1)
        self.assertEqual(seqs, {expected_seq_4rex[164:209]: ['4rexA']})
        self.assertEqual(len(set(dsg.protein_data.keys()) | {'1uscB', '4rexA', '1x9hB', '2rh1A'}), 4)
        expected_dict = {'1uscB': {'PDB': '1usc', 'Chain': 'B', 'PDB_FN': './PDB/us/pdb1usc.ent',
                                   'Sequence': None, 'Length': 0, 'Seq_Fasta': None, 'Accession': None},
                         '4rexA': {'PDB': '4rex', 'Chain': 'A', 'PDB_FN': './PDB/re/pdb4rex.ent',
                                   'Sequence': expected_seq_4rex[164:209], 'Length': len(expected_seq_4rex[164:209]),
                                   'Seq_Fasta': './Sequences/4rexA.fasta', 'Accession': ['P46937', 'YAP1_HUMAN']},
                         '1x9hB': {'PDB': '1x9h', 'Chain': 'B', 'PDB_FN': './PDB/x9/pdb1x9h.ent',
                                   'Sequence': None, 'Length': 0, 'Seq_Fasta': None, 'Accession': None},
                         '2rh1A': {'PDB': '2rh1', 'Chain': 'A', 'PDB_FN': './PDB/rh/pdb2rh1.ent',
                                   'Sequence': None, 'Length': 0, 'Seq_Fasta': None, 'Accession': 'INSPECT MANUALLY'}}
        for p_id in dsg.protein_data:
            for key in dsg.protein_data[p_id]:
                if dsg.protein_data[p_id][key] is None:
                    self.assertIsNone(expected_dict[p_id][key])
                else:
                    if key == 'Sequence':
                        self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_dict[p_id][key])
                    else:
                        if key == 'Accession':
                            self.assertIn(dsg.protein_data[p_id][key], expected_dict[p_id][key])
                        else:
                            self.assertEqual(dsg.protein_data[p_id][key], expected_dict[p_id][key])

    def test_datasetgenerator_identify_protein_sequences_unp_only_multiple_processes(self):
        expected_seq_4rex = 'MDPGQQPPPQPAPQGQGQPPSQPPQGQGPPSGPGQPAPAATQAAPQAPPAGHQIVHVRGDSETDLEALFNAVMNPKTANVPQTVPMRLR' \
                            'KLPDSFFKPPEPKSHSRQASTDAGTAGALTPQHVRAHSSPASLQLGAVSPGTLTPTGVVSGPAATPTAQHLRQSSFEIPDDVPLPAGWE' \
                            'MAKTSSGQRYFLNHIDQTTTWQDPRKAMLSQMNVTAPTSPPVQQNMMNSASGPLPDGWEQAMTQDGEIYYINHKNKTTSWLDPRLDPRF' \
                            'AMNQRISQSAPVKQPPPLAPQSPQGGVMGGSNSNQQQQMRLQQLQMEKERLRLKQQELLRQAMRNINPSTANSPKCQELALRSQLPTLE' \
                            'QDGGTQNPVSSPGMSQELRTMTTNSSDPFLNSGTYHSRDESTDSGLSMSSYSVPRTPDDFLNSVDEMDTGDTINQSTLPSQQNRFPDYL' \
                            'EAIPGTNVDLGTLEGDGMNIEGEELMPSLQEALSSDILNDMESVLAATKLDKESFLTWL'
        dsg = DataSetGenerator(input_path='./')
        fn, unique, seqs = dsg.identify_protein_sequences(data_set_name='test',
                                                          protein_list_fn='./ProteinLists/test.txt', sources=['UNP'],
                                                          processes=2)
        self.assertEqual(fn, './Sequences/test.fasta')
        self.assertEqual(len(set(unique).intersection({'4rexA'})), 1)
        self.assertEqual(seqs, {expected_seq_4rex[164:209]: ['4rexA']})
        self.assertEqual(len(set(dsg.protein_data.keys()) | {'1uscB', '4rexA', '1x9hB', '2rh1A'}), 4)
        expected_dict = {'1uscB': {'PDB': '1usc', 'Chain': 'B', 'PDB_FN': './PDB/us/pdb1usc.ent',
                                   'Sequence': None, 'Length': 0, 'Seq_Fasta': None, 'Accession': None},
                         '4rexA': {'PDB': '4rex', 'Chain': 'A', 'PDB_FN': './PDB/re/pdb4rex.ent',
                                   'Sequence': expected_seq_4rex[164:209], 'Length': len(expected_seq_4rex[164:209]),
                                   'Seq_Fasta': './Sequences/4rexA.fasta', 'Accession': ['P46937', 'YAP1_HUMAN']},
                         '1x9hB': {'PDB': '1x9h', 'Chain': 'B', 'PDB_FN': './PDB/x9/pdb1x9h.ent',
                                   'Sequence': None, 'Length': 0, 'Seq_Fasta': None, 'Accession': None},
                         '2rh1A': {'PDB': '2rh1', 'Chain': 'A', 'PDB_FN': './PDB/rh/pdb2rh1.ent',
                                   'Sequence': None, 'Length': 0, 'Seq_Fasta': None, 'Accession': 'INSPECT MANUALLY'}}
        for p_id in dsg.protein_data:
            for key in dsg.protein_data[p_id]:
                if dsg.protein_data[p_id][key] is None:
                    self.assertIsNone(expected_dict[p_id][key])
                else:
                    if key == 'Sequence':
                        self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_dict[p_id][key])
                    else:
                        if key == 'Accession':
                            self.assertIn(dsg.protein_data[p_id][key], expected_dict[p_id][key])
                        else:
                            self.assertEqual(dsg.protein_data[p_id][key], expected_dict[p_id][key])

    def test_datasetgenerator_identify_protein_sequences_gb_only_single_process(self):
        expected_seq_1x9h = 'MSQLLQDYLNWENYILRRVDFPTSYVVEGEVVRIEAMPRLYISGMGGSGVVADLIRDFSLTWNWEVEVIAVKDYFLKARDGLLIAVSYS'\
                            'GNTIETLYTVEYAKRRRIPAVAITTGGRLAQMGVPTVIVPKASAPRAALPQLLTAALHVVAKVYGIDVKIPEGLEPPNEALIHKLVEEF'\
                            'QKRPTIIAAESMRGVAYRVKNEFNENAKIEPSVEILPEAHHNWIEGSERAVVALTSPHIPKEHQERVKATVEIVGGSIYAVEMHPKGVL'\
                            'SFLRDVGIASVKLAEIRGVNPLATPRIDALKRRLQ'
        dsg = DataSetGenerator(input_path='./')
        fn, unique, seqs = dsg.identify_protein_sequences(data_set_name='test',
                                                          protein_list_fn='./ProteinLists/test.txt', sources=['GB'],
                                                          processes=1)
        self.assertEqual(fn, './Sequences/test.fasta')
        self.assertEqual(len(set(unique).intersection({'1x9hB'})), 1)
        self.assertEqual(seqs, {expected_seq_1x9h[0:302]: ['1x9hB']})
        self.assertEqual(len(set(dsg.protein_data.keys()) | {'1uscB', '4rexA', '1x9hB', '2rh1A'}), 4)
        expected_dict = {'1uscB': {'PDB': '1usc', 'Chain': 'B', 'PDB_FN': './PDB/us/pdb1usc.ent',
                                   'Sequence': None, 'Length': 0, 'Seq_Fasta': None, 'Accession': None},
                         '4rexA': {'PDB': '4rex', 'Chain': 'A', 'PDB_FN': './PDB/re/pdb4rex.ent',
                                   'Sequence': None, 'Length': 0, 'Seq_Fasta': None, 'Accession': None},
                         '1x9hB': {'PDB': '1x9h', 'Chain': 'B', 'PDB_FN': './PDB/x9/pdb1x9h.ent',
                                   'Sequence': expected_seq_1x9h[0:302], 'Length': len(expected_seq_1x9h[0:302]),
                                   'Seq_Fasta': './Sequences/1x9hB.fasta', 'Accession': ['18312750', 'NP_559417']},
                         '2rh1A': {'PDB': '2rh1', 'Chain': 'A', 'PDB_FN': './PDB/rh/pdb2rh1.ent',
                                   'Sequence': None, 'Length': 0, 'Seq_Fasta': None, 'Accession': 'INSPECT MANUALLY'}}
        for p_id in dsg.protein_data:
            for key in dsg.protein_data[p_id]:
                if dsg.protein_data[p_id][key] is None:
                    self.assertIsNone(expected_dict[p_id][key])
                else:
                    if key == 'Sequence':
                        self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_dict[p_id][key])
                    else:
                        if key == 'Accession':
                            self.assertIn(dsg.protein_data[p_id][key], expected_dict[p_id][key])
                        else:
                            self.assertEqual(dsg.protein_data[p_id][key], expected_dict[p_id][key])

    def test_datasetgenerator_identify_protein_sequences_gb_only_multiple_processes(self):
        expected_seq_1x9h = 'MSQLLQDYLNWENYILRRVDFPTSYVVEGEVVRIEAMPRLYISGMGGSGVVADLIRDFSLTWNWEVEVIAVKDYFLKARDGLLIAVSYS' \
                            'GNTIETLYTVEYAKRRRIPAVAITTGGRLAQMGVPTVIVPKASAPRAALPQLLTAALHVVAKVYGIDVKIPEGLEPPNEALIHKLVEEF' \
                            'QKRPTIIAAESMRGVAYRVKNEFNENAKIEPSVEILPEAHHNWIEGSERAVVALTSPHIPKEHQERVKATVEIVGGSIYAVEMHPKGVL' \
                            'SFLRDVGIASVKLAEIRGVNPLATPRIDALKRRLQ'
        dsg = DataSetGenerator(input_path='./')
        fn, unique, seqs = dsg.identify_protein_sequences(data_set_name='test',
                                                          protein_list_fn='./ProteinLists/test.txt', sources=['GB'],
                                                          processes=2)
        self.assertEqual(fn, './Sequences/test.fasta')
        self.assertEqual(len(set(unique).intersection({'1x9hB'})), 1)
        self.assertEqual(seqs, {expected_seq_1x9h[0:302]: ['1x9hB']})
        self.assertEqual(len(set(dsg.protein_data.keys()) | {'1uscB', '4rexA', '1x9hB', '2rh1A'}), 4)
        expected_dict = {'1uscB': {'PDB': '1usc', 'Chain': 'B', 'PDB_FN': './PDB/us/pdb1usc.ent',
                                   'Sequence': None, 'Length': 0, 'Seq_Fasta': None, 'Accession': None},
                         '4rexA': {'PDB': '4rex', 'Chain': 'A', 'PDB_FN': './PDB/re/pdb4rex.ent',
                                   'Sequence': None, 'Length': 0, 'Seq_Fasta': None, 'Accession': None},
                         '1x9hB': {'PDB': '1x9h', 'Chain': 'B', 'PDB_FN': './PDB/x9/pdb1x9h.ent',
                                   'Sequence': expected_seq_1x9h[0:302], 'Length': len(expected_seq_1x9h[0:302]),
                                   'Seq_Fasta': './Sequences/1x9hB.fasta', 'Accession': ['18312750', 'NP_559417']},
                         '2rh1A': {'PDB': '2rh1', 'Chain': 'A', 'PDB_FN': './PDB/rh/pdb2rh1.ent',
                                   'Sequence': None, 'Length': 0, 'Seq_Fasta': None, 'Accession': 'INSPECT MANUALLY'}}
        for p_id in dsg.protein_data:
            for key in dsg.protein_data[p_id]:
                if dsg.protein_data[p_id][key] is None:
                    self.assertIsNone(expected_dict[p_id][key])
                else:
                    if key == 'Sequence':
                        self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_dict[p_id][key])
                    else:
                        if key == 'Accession':
                            self.assertIn(dsg.protein_data[p_id][key], expected_dict[p_id][key])
                        else:
                            self.assertEqual(dsg.protein_data[p_id][key], expected_dict[p_id][key])

    def test_datasetgenerator_identify_protein_sequences_multiple_sources_only_single_process(self):
        expected_seq_1usc = 'MRSYRAQGPLPGFYHYYPGVPAVVGVRVEERVNFCPAVWNTGLSADPPLFGVSISPKRFTHGLLLKARRFSASFHPFGQKDLVHWLGSH' \
                            'SGREVDKGQAPHFLGHTGVPILEGAYAAYELELLEVHTFGDHDLFVGRVVAVWEEEGLLDEKGRPKPGLALLYYGKGLYGRPAEETFAP'
        expected_seq_4rex = 'MDPGQQPPPQPAPQGQGQPPSQPPQGQGPPSGPGQPAPAATQAAPQAPPAGHQIVHVRGDSETDLEALFNAVMNPKTANVPQTVPMRLR' \
                            'KLPDSFFKPPEPKSHSRQASTDAGTAGALTPQHVRAHSSPASLQLGAVSPGTLTPTGVVSGPAATPTAQHLRQSSFEIPDDVPLPAGWE' \
                            'MAKTSSGQRYFLNHIDQTTTWQDPRKAMLSQMNVTAPTSPPVQQNMMNSASGPLPDGWEQAMTQDGEIYYINHKNKTTSWLDPRLDPRF' \
                            'AMNQRISQSAPVKQPPPLAPQSPQGGVMGGSNSNQQQQMRLQQLQMEKERLRLKQQELLRQAMRNINPSTANSPKCQELALRSQLPTLE' \
                            'QDGGTQNPVSSPGMSQELRTMTTNSSDPFLNSGTYHSRDESTDSGLSMSSYSVPRTPDDFLNSVDEMDTGDTINQSTLPSQQNRFPDYL' \
                            'EAIPGTNVDLGTLEGDGMNIEGEELMPSLQEALSSDILNDMESVLAATKLDKESFLTWL'
        expected_seq_1x9h = 'MSQLLQDYLNWENYILRRVDFPTSYVVEGEVVRIEAMPRLYISGMGGSGVVADLIRDFSLTWNWEVEVIAVKDYFLKARDGLLIAVSYS' \
                            'GNTIETLYTVEYAKRRRIPAVAITTGGRLAQMGVPTVIVPKASAPRAALPQLLTAALHVVAKVYGIDVKIPEGLEPPNEALIHKLVEEF' \
                            'QKRPTIIAAESMRGVAYRVKNEFNENAKIEPSVEILPEAHHNWIEGSERAVVALTSPHIPKEHQERVKATVEIVGGSIYAVEMHPKGVL' \
                            'SFLRDVGIASVKLAEIRGVNPLATPRIDALKRRLQ'
        dsg = DataSetGenerator(input_path='./')
        fn, unique, seqs = dsg.identify_protein_sequences(data_set_name='test',
                                                          protein_list_fn='./ProteinLists/test.txt',
                                                          sources=['UNP', 'GB', 'PDB'], processes=1)
        self.assertEqual(fn, './Sequences/test.fasta')
        self.assertEqual(len(set(unique).intersection({'1uscB', '4rexA', '1x9hB'})), 3)
        self.assertEqual(seqs, {expected_seq_1usc: ['1uscB'], expected_seq_4rex[164:209]: ['4rexA'],
                                expected_seq_1x9h[0:302]: ['1x9hB']})
        self.assertEqual(len(set(dsg.protein_data.keys()) | {'1uscB', '4rexA', '1x9hB', '2rh1A'}), 4)
        expected_dict = {'1uscB': {'PDB': '1usc', 'Chain': 'B', 'PDB_FN': './PDB/us/pdb1usc.ent',
                                   'Sequence': expected_seq_1usc, 'Length': len(expected_seq_1usc),
                                   'Seq_Fasta': './Sequences/1uscB.fasta', 'Accession': '1usc'},
                         '4rexA': {'PDB': '4rex', 'Chain': 'A', 'PDB_FN': './PDB/re/pdb4rex.ent',
                                   'Sequence': expected_seq_4rex[164:209], 'Length': len(expected_seq_4rex[164:209]),
                                   'Seq_Fasta': './Sequences/4rexA.fasta', 'Accession': ['P46937', 'YAP1_HUMAN']},
                         '1x9hB': {'PDB': '1x9h', 'Chain': 'B', 'PDB_FN': './PDB/x9/pdb1x9h.ent',
                                   'Sequence': expected_seq_1x9h[0:302], 'Length': len(expected_seq_1x9h[0:302]),
                                   'Seq_Fasta': './Sequences/1x9hB.fasta', 'Accession': ['18312750', 'NP_559417']},
                         '2rh1A': {'PDB': '2rh1', 'Chain': 'A', 'PDB_FN': './PDB/rh/pdb2rh1.ent',
                                   'Sequence': None, 'Length': 0, 'Seq_Fasta': None, 'Accession': 'INSPECT MANUALLY'}}
        for p_id in dsg.protein_data:
            for key in dsg.protein_data[p_id]:
                if dsg.protein_data[p_id][key] is None:
                    self.assertIsNone(expected_dict[p_id][key])
                else:
                    if key == 'Sequence':
                        self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_dict[p_id][key])
                    else:
                        if key == 'Accession':
                            self.assertIn(dsg.protein_data[p_id][key], expected_dict[p_id][key])
                        else:
                            self.assertEqual(dsg.protein_data[p_id][key], expected_dict[p_id][key])

    def test_datasetgenerator_identify_protein_sequences_multiple_sources_only_multiple_processes(self):
        expected_seq_1usc = 'MRSYRAQGPLPGFYHYYPGVPAVVGVRVEERVNFCPAVWNTGLSADPPLFGVSISPKRFTHGLLLKARRFSASFHPFGQKDLVHWLGSH' \
                            'SGREVDKGQAPHFLGHTGVPILEGAYAAYELELLEVHTFGDHDLFVGRVVAVWEEEGLLDEKGRPKPGLALLYYGKGLYGRPAEETFAP'
        expected_seq_4rex = 'MDPGQQPPPQPAPQGQGQPPSQPPQGQGPPSGPGQPAPAATQAAPQAPPAGHQIVHVRGDSETDLEALFNAVMNPKTANVPQTVPMRLR' \
                            'KLPDSFFKPPEPKSHSRQASTDAGTAGALTPQHVRAHSSPASLQLGAVSPGTLTPTGVVSGPAATPTAQHLRQSSFEIPDDVPLPAGWE' \
                            'MAKTSSGQRYFLNHIDQTTTWQDPRKAMLSQMNVTAPTSPPVQQNMMNSASGPLPDGWEQAMTQDGEIYYINHKNKTTSWLDPRLDPRF' \
                            'AMNQRISQSAPVKQPPPLAPQSPQGGVMGGSNSNQQQQMRLQQLQMEKERLRLKQQELLRQAMRNINPSTANSPKCQELALRSQLPTLE' \
                            'QDGGTQNPVSSPGMSQELRTMTTNSSDPFLNSGTYHSRDESTDSGLSMSSYSVPRTPDDFLNSVDEMDTGDTINQSTLPSQQNRFPDYL' \
                            'EAIPGTNVDLGTLEGDGMNIEGEELMPSLQEALSSDILNDMESVLAATKLDKESFLTWL'
        expected_seq_1x9h = 'MSQLLQDYLNWENYILRRVDFPTSYVVEGEVVRIEAMPRLYISGMGGSGVVADLIRDFSLTWNWEVEVIAVKDYFLKARDGLLIAVSYS' \
                            'GNTIETLYTVEYAKRRRIPAVAITTGGRLAQMGVPTVIVPKASAPRAALPQLLTAALHVVAKVYGIDVKIPEGLEPPNEALIHKLVEEF' \
                            'QKRPTIIAAESMRGVAYRVKNEFNENAKIEPSVEILPEAHHNWIEGSERAVVALTSPHIPKEHQERVKATVEIVGGSIYAVEMHPKGVL' \
                            'SFLRDVGIASVKLAEIRGVNPLATPRIDALKRRLQ'
        dsg = DataSetGenerator(input_path='./')
        fn, unique, seqs = dsg.identify_protein_sequences(data_set_name='test',
                                                          protein_list_fn='./ProteinLists/test.txt',
                                                          sources=['UNP', 'GB', 'PDB'], processes=2)
        self.assertEqual(fn, './Sequences/test.fasta')
        self.assertEqual(len(set(unique).intersection({'1uscB', '4rexA', '1x9hB'})), 3)
        self.assertEqual(seqs, {expected_seq_1usc: ['1uscB'], expected_seq_4rex[164:209]: ['4rexA'],
                                expected_seq_1x9h[0:302]: ['1x9hB']})
        self.assertEqual(len(set(dsg.protein_data.keys()) | {'1uscB', '4rexA', '1x9hB', '2rh1A'}), 4)
        expected_dict = {'1uscB': {'PDB': '1usc', 'Chain': 'B', 'PDB_FN': './PDB/us/pdb1usc.ent',
                                   'Sequence': expected_seq_1usc, 'Length': len(expected_seq_1usc),
                                   'Seq_Fasta': './Sequences/1uscB.fasta', 'Accession': '1usc'},
                         '4rexA': {'PDB': '4rex', 'Chain': 'A', 'PDB_FN': './PDB/re/pdb4rex.ent',
                                   'Sequence': expected_seq_4rex[164:209], 'Length': len(expected_seq_4rex[164:209]),
                                   'Seq_Fasta': './Sequences/4rexA.fasta', 'Accession': ['P46937', 'YAP1_HUMAN']},
                         '1x9hB': {'PDB': '1x9h', 'Chain': 'B', 'PDB_FN': './PDB/x9/pdb1x9h.ent',
                                   'Sequence': expected_seq_1x9h[0:302], 'Length': len(expected_seq_1x9h[0:302]),
                                   'Seq_Fasta': './Sequences/1x9hB.fasta', 'Accession': ['18312750', 'NP_559417']},
                         '2rh1A': {'PDB': '2rh1', 'Chain': 'A', 'PDB_FN': './PDB/rh/pdb2rh1.ent',
                                   'Sequence': None, 'Length': 0, 'Seq_Fasta': None, 'Accession': 'INSPECT MANUALLY'}}
        for p_id in dsg.protein_data:
            for key in dsg.protein_data[p_id]:
                if dsg.protein_data[p_id][key] is None:
                    self.assertIsNone(expected_dict[p_id][key])
                else:
                    if key == 'Sequence':
                        self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_dict[p_id][key])
                    else:
                        if key == 'Accession':
                            self.assertIn(dsg.protein_data[p_id][key], expected_dict[p_id][key])
                        else:
                            self.assertEqual(dsg.protein_data[p_id][key], expected_dict[p_id][key])


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
                                                   max_target_seqs=self.max_target_seqs, sources=['UNP', 'GB', 'PDB'])
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
            pdb_fn=self.expected_pdb_fn_small, sources=['UNP', 'GB', 'PDB'])
        self.assertTrue(os.path.isdir(self.sequence_path))
        self.assertEqual(str(self.small_query_seq_uniprot.seq), str(seq_small.seq))
        self.assertEqual(len_small, len(seq_small))
        self.assertEqual(seq_fn_small, self.expected_seq_fn_small)
        self.assertEqual(chain_small, 'A')
        self.assertIsNotNone(unp_small)
        self.assertTrue(unp_small in {'P00703', 'LYSC_MELGA'})
        seq_large, len_large, seq_fn_large, chain_large, unp_large = parse_query_sequence(
            protein_id=self.large_structure_id, chain_id='A', sequence_path=self.sequence_path,
            pdb_fn=self.expected_pdb_fn_large, sources=['UNP', 'GB', 'PDB'])
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
                                 sources=['UNP', 'GB', 'PDB'], verbose=False)
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
                                 sources=['UNP', 'GB', 'PDB'], verbose=False)
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
