"""
Created on May 28, 2019

@author: Daniel Konecki
"""
import os
import sys
import unittest
import numpy as np
from time import sleep
from shutil import rmtree
from unittest import TestCase
from Bio.Seq import Seq
from Bio.SeqIO import write
from Bio.Blast import NCBIXML
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import Gapped
from Bio.Alphabet.IUPAC import IUPACProtein
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

from SupportingClasses.SeqAlignment import SeqAlignment
from SupportingClasses.EvolutionaryTraceAlphabet import FullIUPACProtein
from SupportingClasses.AlignmentDistanceCalculator import AlignmentDistanceCalculator
from SupportingClasses.DataSetGenerator import (DataSetGenerator, import_protein_list, download_pdb,
                                                parse_query_sequence, init_pdb_processing_pool, pdb_processing,
                                                blast_query_sequence, identity_filter,
                                                load_filtered_sequences, init_filter_sequences, filter_sequence,
                                                init_align_sequences, align_sequences)
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


class TestDataSetGeneratorLoadFilteredSequences(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ex_lin = 'Eukaryota;Metazoa;Chordata;Craniata;Vertebrata;Euteleostomi;Mammalia;Eutheria;Euarchontoglires;Glires;Rodentia;Hystricomorpha;Caviidae;Cavia'
        cls.proper_header1 = f'>2tesA HSP_identity={33} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={cls.ex_lin}\n'
        cls.seq_dummy = ('M' * 35) + '\n'

    def test_single_seq(self):
        fn = write_out_temp_fn(suffix='fasta', out_str=self.proper_header1 + self.seq_dummy)
        num_seq = load_filtered_sequences(protein_id='1tesA', pileup_fn=fn, min_fraction=0.7, min_identity=0.4,
                                          max_identity=0.98)
        self.assertEqual(num_seq, 1)
        os.remove(fn)

    def test_multiple_seq(self):
        proper_header2 = f'>3tesA HSP_identity={15} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={self.ex_lin}\n'
        fn = write_out_temp_fn(suffix='fasta', out_str=(self.proper_header1 + self.seq_dummy + proper_header2 +
                                                        self.seq_dummy))
        num_seq = load_filtered_sequences(protein_id='1tesA', pileup_fn=fn, min_fraction=0.7, min_identity=0.4,
                                          max_identity=0.98)
        self.assertEqual(num_seq, 2)
        os.remove(fn)

    def test_bad_fraction(self):
        bad_header = f'>2tesA HSP_identity={15} HSP_alignment_length={23} Fraction_length={0.65} HSP_taxonomy={self.ex_lin}\n'
        fn = write_out_temp_fn(suffix='fasta', out_str=bad_header + self.seq_dummy)
        with self.assertRaises(ValueError):
            load_filtered_sequences(protein_id='1tesA', pileup_fn=fn, min_fraction=0.7, min_identity=0.4,
                                    max_identity=0.98)
        os.remove(fn)

    def test_good_and_bad_fraction(self):
        bad_header = f'>2tesA HSP_identity={15} HSP_alignment_length={23} Fraction_length={0.65} HSP_taxonomy={self.ex_lin}\n'
        fn = write_out_temp_fn(suffix='fasta', out_str=self.proper_header1 + self.seq_dummy + bad_header +
                                                       self.seq_dummy)
        with self.assertRaises(ValueError):
            load_filtered_sequences(protein_id='1tesA', pileup_fn=fn, min_fraction=0.7, min_identity=0.4,
                                    max_identity=0.98)
        os.remove(fn)

    def test_bad_min_identity(self):
        bad_header = f'>2tesA HSP_identity={13} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={self.ex_lin}\n'
        fn = write_out_temp_fn(suffix='fasta', out_str=bad_header + self.seq_dummy)
        with self.assertRaises(ValueError):
            load_filtered_sequences(protein_id='1tesA', pileup_fn=fn, min_fraction=0.7, min_identity=0.4,
                                    max_identity=0.98)
        os.remove(fn)

    def test_good_and_bad_min_identity(self):
        bad_header = f'>2tesA HSP_identity={13} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={self.ex_lin}\n'
        fn = write_out_temp_fn(suffix='fasta', out_str=self.proper_header1 +  self.seq_dummy + bad_header +
                                                       self.seq_dummy)
        with self.assertRaises(ValueError):
            load_filtered_sequences(protein_id='1tesA', pileup_fn=fn, min_fraction=0.7, min_identity=0.4,
                                    max_identity=0.98)
        os.remove(fn)

    def test_bad_max_identity(self):
        bad_header = f'>2tesA HSP_identity={35} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={self.ex_lin}\n'
        fn = write_out_temp_fn(suffix='fasta', out_str=bad_header + self.seq_dummy)
        with self.assertRaises(ValueError):
            load_filtered_sequences(protein_id='1tesA', pileup_fn=fn, min_fraction=0.7, min_identity=0.4,
                                    max_identity=0.98)
        os.remove(fn)

    def test_good_and_bad_max_identity(self):
        bad_header = f'>2tesA HSP_identity={35} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={self.ex_lin}\n'
        fn = write_out_temp_fn(suffix='fasta', out_str=self.proper_header1 + self.seq_dummy + bad_header +
                                                       self.seq_dummy)
        with self.assertRaises(ValueError):
            load_filtered_sequences(protein_id='1tesA', pileup_fn=fn, min_fraction=0.7, min_identity=0.4,
                                    max_identity=0.98)
        os.remove(fn)

    def test_bad_description_format1(self):
        bad_header = f'>2tesA\n'
        fn = write_out_temp_fn(suffix='fasta', out_str=bad_header + self.seq_dummy)
        with self.assertRaises(AttributeError):
            load_filtered_sequences(protein_id='1tesA', pileup_fn=fn, min_fraction=0.7, min_identity=0.4,
                                    max_identity=0.98)
        os.remove(fn)

    def test_bad_description_format2(self):
        old_header = f'>2tesA HSP_identity={15} HSP_alignment_length={35} Fraction_length={1.0}\n'
        fn = write_out_temp_fn(suffix='fasta', out_str=old_header + self.seq_dummy)
        with self.assertRaises(AttributeError):
            load_filtered_sequences(protein_id='1tesA', pileup_fn=fn, min_fraction=0.7, min_identity=0.4,
                                    max_identity=0.98)
        os.remove(fn)

    def test_bad_description_format3(self):
        scrambled_header = f'>2tesA HSP_alignment_length={35} HSP_identity={15} HSP_taxonomy={self.ex_lin} Fraction_length={1.0}\n'
        fn = write_out_temp_fn(suffix='fasta', out_str=scrambled_header + self.seq_dummy)
        with self.assertRaises(AttributeError):
            load_filtered_sequences(protein_id='1tesA', pileup_fn=fn, min_fraction=0.7, min_identity=0.4,
                                    max_identity=0.98)
        os.remove(fn)


class TestDataSetGeneratorFilterSequence(TestCase):

    @classmethod
    def setUpClass(cls):
        # Note the examples below are not real entries, they were taken from an existing xml to maintain the correct
        # structure and were edited to create convenient examples.
        cls.original_seq = SeqRecord(Seq('DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK', alphabet=FullIUPACProtein()), id='4rex',
                                     description='Target Query: Chain: A: From UniProt Accession: P46937')
        cls.header = '<?xml version="1.0"?>\n'\
                     '<!DOCTYPE BlastOutput PUBLIC "-//NCBI//NCBI BlastOutput/EN" "http://www.ncbi.nlm.nih.gov/dtd/NCBI_BlastOutput.dtd">\n'\
                     '<BlastOutput>\n'\
                     '  <BlastOutput_program>blastp</BlastOutput_program>\n'\
                     '  <BlastOutput_version>BLASTP 2.9.0+</BlastOutput_version>\n'\
                     '  <BlastOutput_reference>Stephen F. Altschul, Thomas L. Madden, Alejandro A. Sch&amp;auml;ffer, Jinghui Zhang, Zheng Zhang, Webb Miller, and David J. Lipman (1997), &quot;Gapped BLAST and PSI-BLAST: a new generation of protein database search programs&quot;, Nucleic Acids Res. 25:3389-3402.</BlastOutput_reference>\n''' \
                     '  <BlastOutput_db>/media/daniel/ExtraDrive1/blast_databases/uniref90_05122020/custom_uniref90_05122020.fasta</BlastOutput_db>\n'\
                     '  <BlastOutput_query-ID>Query_1</BlastOutput_query-ID>\n'\
                     '  <BlastOutput_query-def>1yap</BlastOutput_query-def>\n'\
                     '  <BlastOutput_query-len>35</BlastOutput_query-len>\n'\
                     '  <BlastOutput_param>\n'\
                     '    <Parameters>\n'\
                     '      <Parameters_matrix>BLOSUM62</Parameters_matrix>\n'\
                     '      <Parameters_expect>0.05</Parameters_expect>\n'\
                     '      <Parameters_gap-open>11</Parameters_gap-open>\n'\
                     '      <Parameters_gap-extend>1</Parameters_gap-extend>\n'\
                     '      <Parameters_filter>F</Parameters_filter>\n'\
                     '    </Parameters>\n'\
                     '  </BlastOutput_param>\n'\
                     '<BlastOutput_iterations>\n'

        cls.footer = '</BlastOutput_iterations>\n'\
                     '</BlastOutput>\n'\
                     '\n'

        cls.iter_start = '<Iteration>\n'\
                         '  <Iteration_iter-num>1</Iteration_iter-num>\n'\
                         '  <Iteration_query-ID>Query_1</Iteration_query-ID>\n'\
                         '  <Iteration_query-def>1yap</Iteration_query-def>\n'\
                         '  <Iteration_query-len>35</Iteration_query-len>\n'\
                         '<Iteration_hits>\n'

        cls.iter_end = '</Iteration_hits>\n'\
                       '  <Iteration_stat>\n'\
                       '    <Statistics>\n'\
                       '      <Statistics_db-num>101795000</Statistics_db-num>\n'\
                       '      <Statistics_db-len>34384606400</Statistics_db-len>\n'\
                       '      <Statistics_hsp-len>9</Statistics_hsp-len>\n'\
                       '      <Statistics_eff-space>870179736400</Statistics_eff-space>\n'\
                       '      <Statistics_kappa>0.041</Statistics_kappa>\n'\
                       '      <Statistics_lambda>0.267</Statistics_lambda>\n'\
                       '      <Statistics_entropy>0.14</Statistics_entropy>\n'\
                       '    </Statistics>\n'\
                       '  </Iteration_stat>\n'\
                       '</Iteration>\n'

        cls.hit_start = '<Hit>\n'\
                        '  <Hit_num>1</Hit_num>\n'\
                        '  <Hit_id>UniRef90_A0A286XE94</Hit_id>\n'\
                        '  <Hit_def>Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO</Hit_def>\n'\
                        '  <Hit_accession>UniRef90_A0A286XE94</Hit_accession>\n'\
                        '  <Hit_len>381</Hit_len>\n'\
                        '  <Hit_hsps>\n'

        cls.hit_end = '  </Hit_hsps>\n'\
                      '</Hit>\n'

        cls.hsp_ex = '    <Hsp>\n' \
                     '      <Hsp_num>1</Hsp_num>\n' \
                     '      <Hsp_bit-score>77.0258</Hsp_bit-score>\n' \
                     '      <Hsp_score>188</Hsp_score>\n' \
                     '      <Hsp_evalue>3.71968e-18</Hsp_evalue>\n' \
                     '      <Hsp_query-from>1</Hsp_query-from>\n' \
                     '      <Hsp_query-to>35</Hsp_query-to>\n' \
                     '      <Hsp_hit-from>7</Hsp_hit-from>\n' \
                     '      <Hsp_hit-to>41</Hsp_hit-to>\n' \
                     '      <Hsp_query-frame>0</Hsp_query-frame>\n' \
                     '      <Hsp_hit-frame>0</Hsp_hit-frame>\n' \
                     '      <Hsp_identity>33</Hsp_identity>\n' \
                     '      <Hsp_positive>33</Hsp_positive>\n' \
                     '      <Hsp_gaps>0</Hsp_gaps>\n' \
                     '      <Hsp_align-len>35</Hsp_align-len>\n' \
                     '      <Hsp_qseq>DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_qseq>\n' \
                     '      <Hsp_hseq>DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>\n' \
                     '      <Hsp_midline>DVPLP GWE AKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>\n' \
                     '    </Hsp>\n'

    def evaluate_filter_sequence(self, blast_fn, expect_none, expected_desc=None, expected_seq=None):
        blast_handle = open(blast_fn, 'r')
        blast_iter = NCBIXML.parse(blast_handle)
        for blast_record in blast_iter:
            for alignment in blast_record.alignments:
                init_filter_sequences(query_seq=self.original_seq, e_value_threshold=0.05, min_fraction=0.7,
                                      min_identity=0.4, max_identity=0.98, alphabet=Gapped(IUPACProtein), verbose=False)
                filter_res = filter_sequence(alignment)
                if expect_none:
                    self.assertIsNone(filter_res)
                else:
                    self.assertEqual(filter_res.description, expected_desc)
                    self.assertEqual(filter_res.seq, expected_seq)
                break
            break
        blast_handle.close()

    def test_one_hit_one_passing_hsp(self):
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + self.hit_start + self.hsp_ex +
                                                      self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=False,
                                      expected_desc=f'Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO HSP_identity=33 HSP_alignment_length=35 Fraction_length=1.0 HSP_taxonomy=eukaryota;metazoa;chordata;craniata;vertebrata;euteleostomi;mammalia;eutheria;euarchontoglires;glires;rodentia;hystricomorpha;caviidae;cavia',
                                      expected_seq='DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK')
        os.remove(fn)

    def test_one_hit_two_passing_hsps_second_better(self):
        second_hsp = self.hsp_ex.replace('DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK', 'DVPLPAGWEQAKTSSGQRYFLNHIDQTTTWQDPRK')
        second_hsp = second_hsp.replace('<Hsp_identity>33</Hsp_identity>', '<Hsp_identity>34</Hsp_identity>')
        second_hsp = second_hsp.replace('<Hsp_positive>33</Hsp_positive>', '<Hsp_positive>34</Hsp_positive>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + self.hit_start + self.hsp_ex +
                                                      second_hsp + self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=False,
                                      expected_desc=f'Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO HSP_identity=34 HSP_alignment_length=35 Fraction_length=1.0 HSP_taxonomy=eukaryota;metazoa;chordata;craniata;vertebrata;euteleostomi;mammalia;eutheria;euarchontoglires;glires;rodentia;hystricomorpha;caviidae;cavia',
                                      expected_seq='DVPLPAGWEQAKTSSGQRYFLNHIDQTTTWQDPRK')
        os.remove(fn)

    def test_one_hit_one_failing_hsp_e_value(self):
        bad_hsp = self.hsp_ex.replace('<Hsp_evalue>3.71968e-18</Hsp_evalue>', '<Hsp_evalue>1.0</Hsp_evalue>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + self.hit_start + bad_hsp +
                                                      self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=True)
        os.remove(fn)

    def test_one_hit_one_passing_one_failing_hsp_e_value(self):
        bad_hsp = self.hsp_ex.replace('<Hsp_evalue>3.71968e-18</Hsp_evalue>', '<Hsp_evalue>1.0</Hsp_evalue>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + self.hit_start + bad_hsp +
                                                      self.hsp_ex + self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=False,
                                      expected_desc=f'Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO HSP_identity=33 HSP_alignment_length=35 Fraction_length=1.0 HSP_taxonomy=eukaryota;metazoa;chordata;craniata;vertebrata;euteleostomi;mammalia;eutheria;euarchontoglires;glires;rodentia;hystricomorpha;caviidae;cavia',
                                      expected_seq='DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK')
        os.remove(fn)

    def test_one_hit_one_failing_hsp_min_fraction(self):
        bad_hsp = self.hsp_ex.replace('<Hsp_query-from>1</Hsp_query-from>', '<Hsp_query-from>12</Hsp_query-from>')
        bad_hsp = bad_hsp.replace('<Hsp_hit-from>7</Hsp_hit-from>', '<Hsp_hit-from>18</Hsp_hit-from>')
        bad_hsp = bad_hsp.replace('<Hsp_identity>33</Hsp_identity>', '<Hsp_identity>23</Hsp_identity>')
        bad_hsp = bad_hsp.replace('<Hsp_positive>33</Hsp_positive>', '<Hsp_positive>23</Hsp_positive>')
        bad_hsp = bad_hsp.replace('<Hsp_align-len>35</Hsp_align-len>', '<Hsp_align-len>23</Hsp_align-len>')
        bad_hsp = bad_hsp.replace('<Hsp_qseq>DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_qseq>',
                                  '<Hsp_qseq>TSSGQRYFLNHIDQTTTWQDPRK</Hsp_qseq>')
        bad_hsp = bad_hsp.replace('<Hsp_hseq>DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>',
                                  '<Hsp_hseq>TSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>')
        bad_hsp = bad_hsp.replace('<Hsp_midline>DVPLP GWE AKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>',
                                  '<Hsp_midline>TSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + self.hit_start + bad_hsp +
                                                      self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=True)
        os.remove(fn)

    def test_one_hit_one_passing_one_failing_hsp_min_fraction(self):
        bad_hsp = self.hsp_ex.replace('<Hsp_query-from>1</Hsp_query-from>', '<Hsp_query-from>12</Hsp_query-from>')
        bad_hsp = bad_hsp.replace('<Hsp_hit-from>7</Hsp_hit-from>', '<Hsp_hit-from>18</Hsp_hit-from>')
        bad_hsp = bad_hsp.replace('<Hsp_identity>33</Hsp_identity>', '<Hsp_identity>23</Hsp_identity>')
        bad_hsp = bad_hsp.replace('<Hsp_positive>33</Hsp_positive>', '<Hsp_positive>23</Hsp_positive>')
        bad_hsp = bad_hsp.replace('<Hsp_align-len>35</Hsp_align-len>', '<Hsp_align-len>23</Hsp_align-len>')
        bad_hsp = bad_hsp.replace('<Hsp_qseq>DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_qseq>',
                                  '<Hsp_qseq>TSSGQRYFLNHIDQTTTWQDPRK</Hsp_qseq>')
        bad_hsp = bad_hsp.replace('<Hsp_hseq>DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>',
                                  '<Hsp_hseq>TSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>')
        bad_hsp = bad_hsp.replace('<Hsp_midline>DVPLP GWE AKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>',
                                  '<Hsp_midline>TSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + self.hit_start + bad_hsp +
                                                      self.hsp_ex + self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=False,
                                      expected_desc=f'Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO HSP_identity=33 HSP_alignment_length=35 Fraction_length=1.0 HSP_taxonomy=eukaryota;metazoa;chordata;craniata;vertebrata;euteleostomi;mammalia;eutheria;euarchontoglires;glires;rodentia;hystricomorpha;caviidae;cavia',
                                      expected_seq='DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK')
        os.remove(fn)

    def test_one_hit_one_failing_hsp_min_id(self):
        bad_hsp = self.hsp_ex.replace('<Hsp_identity>33</Hsp_identity>', '<Hsp_identity>13</Hsp_identity>')
        bad_hsp = bad_hsp.replace('<Hsp_positive>33</Hsp_positive>', '<Hsp_positive>13</Hsp_positive>')
        bad_hsp = bad_hsp.replace('<Hsp_align-len>35</Hsp_align-len>', '<Hsp_align-len>35</Hsp_align-len>')
        bad_hsp = bad_hsp.replace('<Hsp_qseq>DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_qseq>',
                                  '<Hsp_qseq>DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_qseq>')
        bad_hsp = bad_hsp.replace('<Hsp_hseq>DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>',
                                  '<Hsp_hseq>VDLPAPWGMEKASTGSRQFYNLHIDQTTTWQDPRK</Hsp_hseq>')
        bad_hsp = bad_hsp.replace('<Hsp_midline>DVPLP GWE AKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>',
                                  '<Hsp_midline>                      HIDQTTTWQDPRK</Hsp_midline>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + self.hit_start + bad_hsp +
                                                      self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=True)
        os.remove(fn)

    def test_one_hit_one_passing_one_failing_hsp_min_id(self):
        bad_hsp = self.hsp_ex.replace('<Hsp_identity>33</Hsp_identity>', '<Hsp_identity>13</Hsp_identity>')
        bad_hsp = bad_hsp.replace('<Hsp_positive>33</Hsp_positive>', '<Hsp_positive>13</Hsp_positive>')
        bad_hsp = bad_hsp.replace('<Hsp_hseq>DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>',
                                  '<Hsp_hseq>VDLPAPWGMEKASTGSRQFYNLHIDQTTTWQDPRK</Hsp_hseq>')
        bad_hsp = bad_hsp.replace('<Hsp_midline>DVPLP GWE AKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>',
                                  '<Hsp_midline>                      HIDQTTTWQDPRK</Hsp_midline>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + self.hit_start + bad_hsp +
                                                      self.hsp_ex + self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=False,
                                      expected_desc=f'Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO HSP_identity=33 HSP_alignment_length=35 Fraction_length=1.0 HSP_taxonomy=eukaryota;metazoa;chordata;craniata;vertebrata;euteleostomi;mammalia;eutheria;euarchontoglires;glires;rodentia;hystricomorpha;caviidae;cavia',
                                      expected_seq='DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK')
        os.remove(fn)

    def test_one_hit_one_failing_hsp_max_id(self):
        bad_hsp = self.hsp_ex.replace('<Hsp_identity>33</Hsp_identity>', '<Hsp_identity>35</Hsp_identity>')
        bad_hsp = bad_hsp.replace('<Hsp_positive>33</Hsp_positive>', '<Hsp_positive>35</Hsp_positive>')
        bad_hsp = bad_hsp.replace('<Hsp_hseq>DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>',
                                  '<Hsp_hseq>DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>')
        bad_hsp = bad_hsp.replace('<Hsp_midline>DVPLP GWE AKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>',
                                  '<Hsp_midline>DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + self.hit_start + bad_hsp +
                                                      self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=True)
        os.remove(fn)

    def test_one_hit_one_passing_one_failing_hsp_max_id(self):
        bad_hsp = self.hsp_ex.replace('<Hsp_identity>33</Hsp_identity>', '<Hsp_identity>35</Hsp_identity>')
        bad_hsp = bad_hsp.replace('<Hsp_positive>33</Hsp_positive>', '<Hsp_positive>35</Hsp_positive>')
        bad_hsp = bad_hsp.replace('<Hsp_hseq>DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>',
                                  '<Hsp_hseq>DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>')
        bad_hsp = bad_hsp.replace('<Hsp_midline>DVPLP GWE AKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>',
                                  '<Hsp_midline>DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + self.hit_start + bad_hsp +
                                                      self.hsp_ex + self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=False,
                                      expected_desc=f'Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO HSP_identity=33 HSP_alignment_length=35 Fraction_length=1.0 HSP_taxonomy=eukaryota;metazoa;chordata;craniata;vertebrata;euteleostomi;mammalia;eutheria;euarchontoglires;glires;rodentia;hystricomorpha;caviidae;cavia',
                                      expected_seq='DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK')
        os.remove(fn)

    def test_one_hit_one_failing_bad_sequence_alphabet(self):
        bad_hsp = self.hsp_ex.replace('<Hsp_hseq>DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>',
                                      '<Hsp_hseq>DVPLPBGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + self.hit_start + bad_hsp +
                                                      self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=True)
        os.remove(fn)

    def test_one_hit_one_passing_one_failing_bad_sequence_alphabet(self):
        bad_hsp = self.hsp_ex.replace('<Hsp_hseq>DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>',
                                      '<Hsp_hseq>DVPLPBGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + self.hit_start + bad_hsp +
                                                      self.hsp_ex + self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=False,
                                      expected_desc=f'Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO HSP_identity=33 HSP_alignment_length=35 Fraction_length=1.0 HSP_taxonomy=eukaryota;metazoa;chordata;craniata;vertebrata;euteleostomi;mammalia;eutheria;euarchontoglires;glires;rodentia;hystricomorpha;caviidae;cavia',
                                      expected_seq='DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK')
        os.remove(fn)

    def test_one_hit_one_failing_bad_description_artificial(self):
        bad_hit = self.hit_start.replace(
            '<Hit_def>Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO</Hit_def>',
            '<Hit_def>Artificial Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO</Hit_def>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + bad_hit + self.hsp_ex +
                                                      self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=True)
        os.remove(fn)

    def test_one_hit_one_failing_bad_description_fragment(self):
        bad_hit = self.hit_start.replace(
            '<Hit_def>Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO</Hit_def>',
            '<Hit_def>Yes associated protein 1 Fragment n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO</Hit_def>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + bad_hit + self.hsp_ex +
                                                      self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=True)
        os.remove(fn)

    def test_one_hit_one_failing_bad_description_low_quality(self):
        bad_hit = self.hit_start.replace(
            '<Hit_def>Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO</Hit_def>',
            '<Hit_def>Low Quality Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO</Hit_def>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + bad_hit + self.hsp_ex +
                                                      self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=True)
        os.remove(fn)

    def test_one_hit_one_failing_bad_description_partial(self):
        bad_hit = self.hit_start.replace(
            '<Hit_def>Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO</Hit_def>',
            '<Hit_def>Partial Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO</Hit_def>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + bad_hit + self.hsp_ex +
                                                      self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=True)
        os.remove(fn)

    def test_one_hit_one_failing_bad_description_synthetic(self):
        bad_hit = self.hit_start.replace(
            '<Hit_def>Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO</Hit_def>',
            '<Hit_def>Synthetic Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO</Hit_def>')
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + bad_hit + self.hsp_ex +
                                                      self.hit_end + self.iter_end + self.footer))
        self.evaluate_filter_sequence(blast_fn=fn, expect_none=True)
        os.remove(fn)


class TestDataSetGeneratorFilterBLASTResults(TestCase):

    @classmethod
    def setUpClass(cls):
        # Note the examples below are not real entries, they were taken from an existing xml to maintain the correct
        # structure and were edited to create convenient examples.
        cls.original_seq = SeqRecord(Seq('DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK', alphabet=FullIUPACProtein()), id='4rex',
                                     description='Target Query: Chain: A: From UniProt Accession: P46937')
        cls.header = '<?xml version="1.0"?>\n' \
                     '<!DOCTYPE BlastOutput PUBLIC "-//NCBI//NCBI BlastOutput/EN" "http://www.ncbi.nlm.nih.gov/dtd/NCBI_BlastOutput.dtd">\n' \
                     '<BlastOutput>\n' \
                     '  <BlastOutput_program>blastp</BlastOutput_program>\n' \
                     '  <BlastOutput_version>BLASTP 2.9.0+</BlastOutput_version>\n' \
                     '  <BlastOutput_reference>Stephen F. Altschul, Thomas L. Madden, Alejandro A. Sch&amp;auml;ffer, Jinghui Zhang, Zheng Zhang, Webb Miller, and David J. Lipman (1997), &quot;Gapped BLAST and PSI-BLAST: a new generation of protein database search programs&quot;, Nucleic Acids Res. 25:3389-3402.</BlastOutput_reference>\n''' \
                     '  <BlastOutput_db>/media/daniel/ExtraDrive1/blast_databases/uniref90_05122020/custom_uniref90_05122020.fasta</BlastOutput_db>\n' \
                     '  <BlastOutput_query-ID>Query_1</BlastOutput_query-ID>\n' \
                     '  <BlastOutput_query-def>1tesA</BlastOutput_query-def>\n' \
                     '  <BlastOutput_query-len>35</BlastOutput_query-len>\n' \
                     '  <BlastOutput_param>\n' \
                     '    <Parameters>\n' \
                     '      <Parameters_matrix>BLOSUM62</Parameters_matrix>\n' \
                     '      <Parameters_expect>0.05</Parameters_expect>\n' \
                     '      <Parameters_gap-open>11</Parameters_gap-open>\n' \
                     '      <Parameters_gap-extend>1</Parameters_gap-extend>\n' \
                     '      <Parameters_filter>F</Parameters_filter>\n' \
                     '    </Parameters>\n' \
                     '  </BlastOutput_param>\n' \
                     '<BlastOutput_iterations>\n'

        cls.footer = '</BlastOutput_iterations>\n' \
                     '</BlastOutput>\n' \
                     '\n'

        cls.iter_start = '<Iteration>\n' \
                         '  <Iteration_iter-num>1</Iteration_iter-num>\n' \
                         '  <Iteration_query-ID>Query_1</Iteration_query-ID>\n' \
                         '  <Iteration_query-def>1tesA</Iteration_query-def>\n' \
                         '  <Iteration_query-len>35</Iteration_query-len>\n' \
                         '<Iteration_hits>\n'

        cls.iter_end = '</Iteration_hits>\n' \
                       '  <Iteration_stat>\n' \
                       '    <Statistics>\n' \
                       '      <Statistics_db-num>101795000</Statistics_db-num>\n' \
                       '      <Statistics_db-len>34384606400</Statistics_db-len>\n' \
                       '      <Statistics_hsp-len>9</Statistics_hsp-len>\n' \
                       '      <Statistics_eff-space>870179736400</Statistics_eff-space>\n' \
                       '      <Statistics_kappa>0.041</Statistics_kappa>\n' \
                       '      <Statistics_lambda>0.267</Statistics_lambda>\n' \
                       '      <Statistics_entropy>0.14</Statistics_entropy>\n' \
                       '    </Statistics>\n' \
                       '  </Iteration_stat>\n' \
                       '</Iteration>\n'

        cls.hit_start = '<Hit>\n' \
                        '  <Hit_num>1</Hit_num>\n' \
                        '  <Hit_id>UniRef90_A0A286XE94</Hit_id>\n' \
                        '  <Hit_def>Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO</Hit_def>\n' \
                        '  <Hit_accession>UniRef90_A0A286XE94</Hit_accession>\n' \
                        '  <Hit_len>381</Hit_len>\n' \
                        '  <Hit_hsps>\n'

        cls.hit_end = '  </Hit_hsps>\n' \
                      '</Hit>\n'

        cls.hsp_ex = '    <Hsp>\n' \
                     '      <Hsp_num>1</Hsp_num>\n' \
                     '      <Hsp_bit-score>77.0258</Hsp_bit-score>\n' \
                     '      <Hsp_score>188</Hsp_score>\n' \
                     '      <Hsp_evalue>3.71968e-18</Hsp_evalue>\n' \
                     '      <Hsp_query-from>1</Hsp_query-from>\n' \
                     '      <Hsp_query-to>35</Hsp_query-to>\n' \
                     '      <Hsp_hit-from>7</Hsp_hit-from>\n' \
                     '      <Hsp_hit-to>41</Hsp_hit-to>\n' \
                     '      <Hsp_query-frame>0</Hsp_query-frame>\n' \
                     '      <Hsp_hit-frame>0</Hsp_hit-frame>\n' \
                     '      <Hsp_identity>33</Hsp_identity>\n' \
                     '      <Hsp_positive>33</Hsp_positive>\n' \
                     '      <Hsp_gaps>0</Hsp_gaps>\n' \
                     '      <Hsp_align-len>35</Hsp_align-len>\n' \
                     '      <Hsp_qseq>DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_qseq>\n' \
                     '      <Hsp_hseq>DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>\n' \
                     '      <Hsp_midline>DVPLP GWE AKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>\n' \
                     '    </Hsp>\n'
        #
        cls.second_hsp = cls.hsp_ex.replace('DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK', 'DVPLPAGWEQAKTSSGQRYFLNHIDQTTTWQDPRK')
        cls.second_hsp = cls.second_hsp.replace('<Hsp_identity>33</Hsp_identity>', '<Hsp_identity>34</Hsp_identity>')
        cls.second_hsp = cls.second_hsp.replace('<Hsp_positive>33</Hsp_positive>', '<Hsp_positive>34</Hsp_positive>')
        #
        cls.bad_e_val_hsp = cls.hsp_ex.replace('<Hsp_evalue>3.71968e-18</Hsp_evalue>', '<Hsp_evalue>1.0</Hsp_evalue>')
        #
        cls.bad_min_frac_hsp = cls.hsp_ex.replace('<Hsp_query-from>1</Hsp_query-from>',
                                                  '<Hsp_query-from>12</Hsp_query-from>')
        cls.bad_min_frac_hsp = cls.bad_min_frac_hsp.replace('<Hsp_hit-from>7</Hsp_hit-from>',
                                                            '<Hsp_hit-from>18</Hsp_hit-from>')
        cls.bad_min_frac_hsp = cls.bad_min_frac_hsp.replace('<Hsp_identity>33</Hsp_identity>',
                                                            '<Hsp_identity>23</Hsp_identity>')
        cls.bad_min_frac_hspp = cls.bad_min_frac_hsp.replace('<Hsp_positive>33</Hsp_positive>',
                                                             '<Hsp_positive>23</Hsp_positive>')
        cls.bad_min_frac_hsp = cls.bad_min_frac_hsp.replace('<Hsp_align-len>35</Hsp_align-len>',
                                                            '<Hsp_align-len>23</Hsp_align-len>')
        cls.bad_min_frac_hsp = cls.bad_min_frac_hsp.replace('<Hsp_qseq>DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_qseq>',
                                  '<Hsp_qseq>TSSGQRYFLNHIDQTTTWQDPRK</Hsp_qseq>')
        cls.bad_min_frac_hsp = cls.bad_min_frac_hsp.replace('<Hsp_hseq>DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>',
                                  '<Hsp_hseq>TSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>')
        cls.bad_min_frac_hsp = cls.bad_min_frac_hsp.replace('<Hsp_midline>DVPLP GWE AKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>',
                                  '<Hsp_midline>TSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>')
        #
        cls.bad_min_id_hsp = cls.hsp_ex.replace('<Hsp_identity>33</Hsp_identity>', '<Hsp_identity>13</Hsp_identity>')
        cls.bad_min_id_hsp = cls.bad_min_id_hsp.replace('<Hsp_positive>33</Hsp_positive>', '<Hsp_positive>13</Hsp_positive>')
        cls.bad_min_id_hsp = cls.bad_min_id_hsp.replace('<Hsp_align-len>35</Hsp_align-len>', '<Hsp_align-len>35</Hsp_align-len>')
        cls.bad_min_id_hsp = cls.bad_min_id_hsp.replace('<Hsp_qseq>DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_qseq>',
                                  '<Hsp_qseq>DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_qseq>')
        cls.bad_min_id_hsp = cls.bad_min_id_hsp.replace('<Hsp_hseq>DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>',
                                  '<Hsp_hseq>VDLPAPWGMEKASTGSRQFYNLHIDQTTTWQDPRK</Hsp_hseq>')
        cls.bad_min_id_hsp = cls.bad_min_id_hsp.replace('<Hsp_midline>DVPLP GWE AKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>',
                                  '<Hsp_midline>                      HIDQTTTWQDPRK</Hsp_midline>')
        #
        cls.bad_max_id_hsp = cls.hsp_ex.replace('<Hsp_identity>33</Hsp_identity>', '<Hsp_identity>35</Hsp_identity>')
        cls.bad_max_id_hsp = cls.bad_max_id_hsp.replace('<Hsp_positive>33</Hsp_positive>',
                                                        '<Hsp_positive>35</Hsp_positive>')
        cls.bad_max_id_hsp = cls.bad_max_id_hsp.replace('<Hsp_hseq>DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>',
                                                        '<Hsp_hseq>DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>')
        cls.bad_max_id_hsp = cls.bad_max_id_hsp.replace('<Hsp_midline>DVPLP GWE AKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>',
                                                        '<Hsp_midline>DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_midline>')
        #
        cls.bad_sequence_hsp = cls.hsp_ex.replace('<Hsp_hseq>DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>',
                                                  '<Hsp_hseq>DVPLPBGWEMAKTSSGQRYFLNHIDQTTTWQDPRK</Hsp_hseq>')
        #
        cls.bad_desc_hit = cls.hit_start.replace(
            '<Hit_def>Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO</Hit_def>',
            '<Hit_def>Artificial Yes associated protein 1 n=6 Tax=Eutheria TaxID=9347 RepID=A0A286XE94_CAVPO</Hit_def>')
        #
        cls.ex_lin = 'Eukaryota;Metazoa;Chordata;Craniata;Vertebrata;Euteleostomi;Mammalia;Eutheria;Euarchontoglires;Glires;Rodentia;Hystricomorpha;Caviidae;Cavia'
        cls.proper_header1 = f'>2tesA HSP_identity={33} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={cls.ex_lin}\n'
        cls.seq1 = 'DVPLPDGWEQAKTSSGQRYFLNHIDQTTTWQDPRK\n'
        cls.proper_header2 = f'>3tesA HSP_identity={15} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={cls.ex_lin}\n'
        cls.seq2 = 'DVPLPAGWEQAKTSSGQRYFLNHIDQTTTWQDPRK\n'

    def tearDown(self):
        rmtree('PDB')
        rmtree('Sequences')
        rmtree('BLAST')
        rmtree('Filtered_BLAST')
        rmtree('Alignments')
        rmtree('Filtered_Alignments')
        rmtree('Final_Alignments')

    def evaluate_filter_blast_results(self, dsg, expected_protein_data):
        self.assertEqual(len(dsg.protein_data), len(expected_protein_data))
        for p_id in dsg.protein_data:
            self.assertEqual(dsg.protein_data[p_id]['Filtered_BLAST'], expected_protein_data[p_id]['Filtered_BLAST'])
            self.assertEqual(dsg.protein_data[p_id]['Filter_Count'], expected_protein_data[p_id]['Filter_Count'])
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Filtered_BLAST']))

    def test_one_pid_xml_single_process(self):
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + self.hit_start + self.hsp_ex +
                                                      self.second_hsp + self.bad_e_val_hsp + self.bad_min_frac_hsp +
                                                      self.bad_min_id_hsp + self.bad_max_id_hsp + self.bad_sequence_hsp +
                                                      self.hit_end + self.bad_desc_hit + self.hsp_ex + self.hit_end +
                                                      self.iter_end + self.footer))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq}}
        dsg.filter_blast_results(['1tesA'], e_value_threshold=0.05, min_fraction=0.7, min_identity=0.4,
                                 max_identity=0.98, blast_fn=fn, processes=1, verbose=False)
        self.evaluate_filter_blast_results(dsg, {'1tesA': {'Filter_Count': 2,
                                                           'Filtered_BLAST': './Filtered_BLAST/1tesA.fasta'}})
        os.remove(fn)

    def test_one_pid_xml_multi_process(self):
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header + self.iter_start + self.hit_start + self.hsp_ex +
                                                      self.second_hsp + self.bad_e_val_hsp + self.bad_min_frac_hsp +
                                                      self.bad_min_id_hsp + self.bad_max_id_hsp + self.bad_sequence_hsp +
                                                      self.hit_end + self.bad_desc_hit + self.hsp_ex + self.hit_end +
                                                      self.iter_end + self.footer))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq}}
        dsg.filter_blast_results(['1tesA'], e_value_threshold=0.05, min_fraction=0.7, min_identity=0.4,
                                 max_identity=0.98, blast_fn=fn, processes=2, verbose=False)
        self.evaluate_filter_blast_results(dsg, {'1tesA': {'Filter_Count': 2,
                                                           'Filtered_BLAST': './Filtered_BLAST/1tesA.fasta'}})
        os.remove(fn)

    def test_one_pid_fasta_single_process(self):
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq}}
        with open('./Filtered_BLAST/1tesA.fasta', 'w') as handle:
            handle.write('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header1 + self.seq1)
        dsg.filter_blast_results(['1tesA'], e_value_threshold=0.05, min_fraction=0.7, min_identity=0.4,
                                 max_identity=0.98, blast_fn='./BLAST/All_Seqs.xml', processes=1, verbose=False)
        self.evaluate_filter_blast_results(dsg, {'1tesA': {'Filter_Count': 2,
                                                           'Filtered_BLAST': './Filtered_BLAST/1tesA.fasta'}})

    def test_one_pid_fasta_multi_process(self):
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq}}
        with open('./Filtered_BLAST/1tesA.fasta', 'w') as handle:
            handle.write('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header1 + self.seq1)
        dsg.filter_blast_results(['1tesA'], e_value_threshold=0.05, min_fraction=0.7, min_identity=0.4,
                                 max_identity=0.98, blast_fn='./BLAST/All_Seqs.xml', processes=2, verbose=False)
        self.evaluate_filter_blast_results(dsg, {'1tesA': {'Filter_Count': 2,
                                                           'Filtered_BLAST': './Filtered_BLAST/1tesA.fasta'}})

    def test_two_pids_xml_single_process(self):
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header +
                                                      self.iter_start + self.hit_start + self.hsp_ex +
                                                      self.second_hsp + self.bad_e_val_hsp + self.bad_min_frac_hsp +
                                                      self.bad_min_id_hsp + self.bad_max_id_hsp + self.bad_sequence_hsp +
                                                      self.hit_end + self.bad_desc_hit + self.hsp_ex + self.hit_end +
                                                      self.iter_end +
                                                      self.iter_start.replace('1tesA', '1estA') + self.hit_start +
                                                      self.hsp_ex + self.hit_end +
                                                      self.hit_start.replace('<Hit_num>1</Hit_num>', '<Hit_num>2</Hit_num>') +
                                                      self.second_hsp + self.bad_e_val_hsp + self.bad_min_frac_hsp +
                                                      self.bad_min_id_hsp + self.bad_max_id_hsp + self.bad_sequence_hsp +
                                                      self.hit_end + self.bad_desc_hit + self.hsp_ex + self.hit_end +
                                                      self.iter_end +
                                                      self.footer))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq}, '1estA': {'Sequence': self.original_seq}}
        dsg.filter_blast_results(['1tesA', '1estA'], e_value_threshold=0.05, min_fraction=0.7, min_identity=0.4,
                                 max_identity=0.98, blast_fn=fn, processes=1, verbose=False)
        self.evaluate_filter_blast_results(dsg, {'1tesA': {'Filter_Count': 2,
                                                           'Filtered_BLAST': './Filtered_BLAST/1tesA.fasta'},
                                                 '1estA': {'Filter_Count': 3,
                                                           'Filtered_BLAST': './Filtered_BLAST/1estA.fasta'}})
        os.remove(fn)

    def test_two_pids_xml_multi_process(self):
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header +
                                                      self.iter_start + self.hit_start + self.hsp_ex +
                                                      self.second_hsp + self.bad_e_val_hsp + self.bad_min_frac_hsp +
                                                      self.bad_min_id_hsp + self.bad_max_id_hsp + self.bad_sequence_hsp +
                                                      self.hit_end + self.bad_desc_hit + self.hsp_ex + self.hit_end +
                                                      self.iter_end +
                                                      self.iter_start.replace('1tesA', '1estA') + self.hit_start +
                                                      self.hsp_ex + self.hit_end +
                                                      self.hit_start.replace('<Hit_num>1</Hit_num>',
                                                                             '<Hit_num>2</Hit_num>') +
                                                      self.second_hsp + self.bad_e_val_hsp + self.bad_min_frac_hsp +
                                                      self.bad_min_id_hsp + self.bad_max_id_hsp + self.bad_sequence_hsp +
                                                      self.hit_end + self.bad_desc_hit + self.hsp_ex + self.hit_end +
                                                      self.iter_end +
                                                      self.footer))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq}, '1estA': {'Sequence': self.original_seq}}
        dsg.filter_blast_results(['1tesA', '1estA'], e_value_threshold=0.05, min_fraction=0.7, min_identity=0.4,
                                 max_identity=0.98, blast_fn=fn, processes=2, verbose=False)
        self.evaluate_filter_blast_results(dsg, {'1tesA': {'Filter_Count': 2,
                                                           'Filtered_BLAST': './Filtered_BLAST/1tesA.fasta'},
                                                 '1estA': {'Filter_Count': 3,
                                                           'Filtered_BLAST': './Filtered_BLAST/1estA.fasta'}})
        os.remove(fn)

    def test_two_pids_fasta_single_process(self):
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq}, '1estA': {'Sequence': self.original_seq}}
        with open('./Filtered_BLAST/1tesA.fasta', 'w') as handle:
            handle.write('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 + self.seq2)
        with open('./Filtered_BLAST/1estA.fasta', 'w') as handle:
            handle.write('>1estA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 + self.seq2 +
                         self.proper_header1 + self.seq1)
        dsg.filter_blast_results(['1tesA', '1estA'], e_value_threshold=0.05, min_fraction=0.7, min_identity=0.4,
                                 max_identity=0.98, blast_fn='./BLAST/All_Seqs.xml', processes=1, verbose=False)
        self.evaluate_filter_blast_results(dsg, {'1tesA': {'Filter_Count': 2,
                                                           'Filtered_BLAST': './Filtered_BLAST/1tesA.fasta'},
                                                 '1estA': {'Filter_Count': 3,
                                                           'Filtered_BLAST': './Filtered_BLAST/1estA.fasta'}})

    def test_two_pids_fasta_multi_process(self):
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq}, '1estA': {'Sequence': self.original_seq}}
        with open('./Filtered_BLAST/1tesA.fasta', 'w') as handle:
            handle.write('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 + self.seq2)
        with open('./Filtered_BLAST/1estA.fasta', 'w') as handle:
            handle.write('>1estA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 + self.seq2 +
                         self.proper_header1 + self.seq1)
        dsg.filter_blast_results(['1tesA', '1estA'], e_value_threshold=0.05, min_fraction=0.7, min_identity=0.4,
                                 max_identity=0.98, blast_fn='./BLAST/All_Seqs.xml', processes=2, verbose=False)
        self.evaluate_filter_blast_results(dsg, {'1tesA': {'Filter_Count': 2,
                                                           'Filtered_BLAST': './Filtered_BLAST/1tesA.fasta'},
                                                 '1estA': {'Filter_Count': 3,
                                                           'Filtered_BLAST': './Filtered_BLAST/1estA.fasta'}})

    def test_two_pids_xml_and_fasta_single_process(self):
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header +
                                                      self.iter_start + self.hit_start + self.hsp_ex +
                                                      self.second_hsp + self.bad_e_val_hsp + self.bad_min_frac_hsp +
                                                      self.bad_min_id_hsp + self.bad_max_id_hsp + self.bad_sequence_hsp +
                                                      self.hit_end + self.bad_desc_hit + self.hsp_ex + self.hit_end +
                                                      self.iter_end +
                                                      self.iter_start.replace('1tesA', '1estA') + self.hit_start +
                                                      self.hsp_ex + self.hit_end +
                                                      self.hit_start.replace('<Hit_num>1</Hit_num>',
                                                                             '<Hit_num>2</Hit_num>') +
                                                      self.second_hsp + self.bad_e_val_hsp + self.bad_min_frac_hsp +
                                                      self.bad_min_id_hsp + self.bad_max_id_hsp + self.bad_sequence_hsp +
                                                      self.hit_end + self.bad_desc_hit + self.hsp_ex + self.hit_end +
                                                      self.iter_end +
                                                      self.footer))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq}, '1estA': {'Sequence': self.original_seq}}
        with open('./Filtered_BLAST/1tesA.fasta', 'w') as handle:
            handle.write('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 + self.seq2)
        dsg.filter_blast_results(['1tesA', '1estA'], e_value_threshold=0.05, min_fraction=0.7, min_identity=0.4,
                                 max_identity=0.98, blast_fn=fn, processes=1, verbose=False)
        self.evaluate_filter_blast_results(dsg, {'1tesA': {'Filter_Count': 2,
                                                           'Filtered_BLAST': './Filtered_BLAST/1tesA.fasta'},
                                                 '1estA': {'Filter_Count': 3,
                                                           'Filtered_BLAST': './Filtered_BLAST/1estA.fasta'}})
        os.remove(fn)

    def test_two_pids_xml_and_fasta_multi_process(self):
        fn = write_out_temp_fn(suffix='xml', out_str=(self.header +
                                                      self.iter_start + self.hit_start + self.hsp_ex +
                                                      self.second_hsp + self.bad_e_val_hsp + self.bad_min_frac_hsp +
                                                      self.bad_min_id_hsp + self.bad_max_id_hsp + self.bad_sequence_hsp +
                                                      self.hit_end + self.bad_desc_hit + self.hsp_ex + self.hit_end +
                                                      self.iter_end +
                                                      self.iter_start.replace('1tesA', '1estA') + self.hit_start +
                                                      self.hsp_ex + self.hit_end +
                                                      self.hit_start.replace('<Hit_num>1</Hit_num>',
                                                                             '<Hit_num>2</Hit_num>') +
                                                      self.second_hsp + self.bad_e_val_hsp + self.bad_min_frac_hsp +
                                                      self.bad_min_id_hsp + self.bad_max_id_hsp + self.bad_sequence_hsp +
                                                      self.hit_end + self.bad_desc_hit + self.hsp_ex + self.hit_end +
                                                      self.iter_end +
                                                      self.footer))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq}, '1estA': {'Sequence': self.original_seq}}
        with open('./Filtered_BLAST/1tesA.fasta', 'w') as handle:
            handle.write('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 + self.seq2)
        dsg.filter_blast_results(['1tesA', '1estA'], e_value_threshold=0.05, min_fraction=0.7, min_identity=0.4,
                                 max_identity=0.98, blast_fn=fn, processes=2, verbose=False)
        self.evaluate_filter_blast_results(dsg, {'1tesA': {'Filter_Count': 2,
                                                           'Filtered_BLAST': './Filtered_BLAST/1tesA.fasta'},
                                                 '1estA': {'Filter_Count': 3,
                                                           'Filtered_BLAST': './Filtered_BLAST/1estA.fasta'}})
        os.remove(fn)


class TestDataSetGeneratorAlignSequences(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.original_seq = SeqRecord(Seq('DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK', alphabet=FullIUPACProtein()), id='4rex',
                                     description='Target Query: Chain: A: From UniProt Accession: P46937')
        cls.ex_lin = 'Eukaryota;Metazoa;Chordata;Craniata;Vertebrata;Euteleostomi;Mammalia;Eutheria;Euarchontoglires;Glires;Rodentia;Hystricomorpha;Caviidae;Cavia'
        cls.proper_header1 = f'>2tesA HSP_identity={33} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={cls.ex_lin}\n'
        cls.seq1 = 'DVPLP-GWE-AKTSSGQRYFLNHIDQTTTWQDPRK\n'
        cls.proper_header2 = f'>3tesA HSP_identity={15} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={cls.ex_lin}\n'
        cls.seq2 = 'DVPLPAGWE-AKTSSGQRYFLNHIDQTTTWQDPRK\n'

    def test_align_sequences_msf_and_fasta(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        init_align_sequences(alignment_path='./', msf=True, fasta=True, verbose=False)
        protein_id, msf_fn, fasta_fn = align_sequences(protein_id='1tesA', pileup_fn=fn)
        self.assertEqual(protein_id, '1tesA')
        self.assertIsNotNone(msf_fn)
        self.assertTrue(os.path.isfile(msf_fn))
        self.assertIsNotNone(fasta_fn)
        self.assertTrue(os.path.isfile(fasta_fn))
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')
        os.remove(msf_fn)
        os.remove(fasta_fn)

    def test_align_sequences_msf(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        init_align_sequences(alignment_path='./', msf=True, fasta=False, verbose=False)
        protein_id, msf_fn, fasta_fn = align_sequences(protein_id='1tesA', pileup_fn=fn)
        self.assertEqual(protein_id, '1tesA')
        self.assertIsNotNone(msf_fn)
        self.assertTrue(os.path.isfile(msf_fn))
        self.assertIsNone(fasta_fn)
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')
        os.remove(msf_fn)

    def test_align_sequences_fasta(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        init_align_sequences(alignment_path='./', msf=False, fasta=True, verbose=False)
        protein_id, msf_fn, fasta_fn = align_sequences(protein_id='1tesA', pileup_fn=fn)
        self.assertEqual(protein_id, '1tesA')
        self.assertIsNone(msf_fn)
        self.assertIsNotNone(fasta_fn)
        self.assertTrue(os.path.isfile(fasta_fn))
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')
        os.remove(fasta_fn)

    def test_align_sequences_bad_alignment_path(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        init_align_sequences(alignment_path='./Alignments', msf=True, fasta=True, verbose=False)
        protein_id, msf_fn, fasta_fn = align_sequences(protein_id='1tesA', pileup_fn=fn)
        self.assertEqual(protein_id, '1tesA')
        self.assertIsNotNone(msf_fn)
        self.assertTrue(os.path.isfile(msf_fn))
        self.assertIsNotNone(fasta_fn)
        self.assertTrue(os.path.isfile(fasta_fn))
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')
        rmtree('./Alignments')

    def test_align_sequences_no_alignment_path(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        init_align_sequences(alignment_path=None, msf=True, fasta=True, verbose=False)
        with self.assertRaises(TypeError):
            align_sequences(protein_id='1tesA', pileup_fn=fn)
        os.remove(fn)

    def test_no_protein_id(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        init_align_sequences(alignment_path='./', msf=True, fasta=True, verbose=False)
        with self.assertRaises(ValueError):
            align_sequences(protein_id=None, pileup_fn=fn)
        os.remove(fn)

    def test_no_pileup_fn(self):
        init_align_sequences(alignment_path='./', msf=True, fasta=True, verbose=False)
        with self.assertRaises(ValueError):
            align_sequences(protein_id='1tesA', pileup_fn=None)


class TestDataSetGeneratorAlignBLASTHits(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.original_seq = SeqRecord(Seq('DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK', alphabet=FullIUPACProtein()), id='4rex',
                                     description='Target Query: Chain: A: From UniProt Accession: P46937')
        cls.ex_lin = 'Eukaryota;Metazoa;Chordata;Craniata;Vertebrata;Euteleostomi;Mammalia;Eutheria;Euarchontoglires;Glires;Rodentia;Hystricomorpha;Caviidae;Cavia'
        cls.proper_header1 = f'>2tesA HSP_identity={33} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={cls.ex_lin}\n'
        cls.seq1 = 'DVPLP-GWE-AKTSSGQRYFLNHIDQTTTWQDPRK\n'
        cls.proper_header2 = f'>3tesA HSP_identity={15} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={cls.ex_lin}\n'
        cls.seq2 = 'DVPLPAGWE-AKTSSGQRYFLNHIDQTTTWQDPRK\n'

    def tearDown(self):
        rmtree('PDB')
        rmtree('Sequences')
        rmtree('BLAST')
        rmtree('Filtered_BLAST')
        rmtree('Alignments')
        rmtree('Filtered_Alignments')
        rmtree('Final_Alignments')

    def test_align_blast_hits_single_protein_single_process_msf_fasta(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq, 'Filtered_BLAST': fn}}
        dsg.align_blast_hits(unique_ids=['1tesA'], msf=True, fasta=True, processes=1)
        self.assertEqual(len(dsg.protein_data), 1)
        for p_id in dsg.protein_data:
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['MSF_Aln']))
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['FA_Aln']))
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')

    def test_align_blast_hits_single_protein_multi_process_msf_fasta(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq, 'Filtered_BLAST': fn}}
        dsg.align_blast_hits(unique_ids=['1tesA'], msf=True, fasta=True, processes=2)
        self.assertEqual(len(dsg.protein_data), 1)
        for p_id in dsg.protein_data:
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['MSF_Aln']))
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['FA_Aln']))
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')

    def test_align_blast_hits_multi_protein_single_process_msf_fasta(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        sleep(1)
        fn2 = write_out_temp_fn(suffix='fasta',
                                out_str=('>1estA\n' + str(self.original_seq.seq) + '\n' + self.proper_header1 +
                                         self.seq1))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq, 'Filtered_BLAST': fn},
                            '1estA': {'Sequence': self.original_seq, 'Filtered_BLAST': fn2}}
        dsg.align_blast_hits(unique_ids=['1tesA', '1estA'], msf=True, fasta=True, processes=1)
        self.assertEqual(len(dsg.protein_data), 2)
        for p_id in dsg.protein_data:
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['MSF_Aln']))
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['FA_Aln']))
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')
        os.remove(fn2)
        os.remove(os.path.splitext(fn2)[0] + '.dnd')

    def test_align_blast_hits_multi_protein_multi_process_msf_fasta(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        sleep(1)
        fn2 = write_out_temp_fn(suffix='fasta',
                                out_str=('>1estA\n' + str(self.original_seq.seq) + '\n' + self.proper_header1 +
                                         self.seq1))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq, 'Filtered_BLAST': fn},
                            '1estA': {'Sequence': self.original_seq, 'Filtered_BLAST': fn2}}
        dsg.align_blast_hits(unique_ids=['1tesA', '1estA'], msf=True, fasta=True, processes=2)
        self.assertEqual(len(dsg.protein_data), 2)
        for p_id in dsg.protein_data:
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['MSF_Aln']))
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['FA_Aln']))
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')
        os.remove(fn2)
        os.remove(os.path.splitext(fn2)[0] + '.dnd')

    def test_align_blast_hits_multi_protein_multi_process_msf(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        sleep(1)
        fn2 = write_out_temp_fn(suffix='fasta',
                                out_str=('>1estA\n' + str(self.original_seq.seq) + '\n' + self.proper_header1 +
                                         self.seq1))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq, 'Filtered_BLAST': fn},
                            '1estA': {'Sequence': self.original_seq, 'Filtered_BLAST': fn2}}
        dsg.align_blast_hits(unique_ids=['1tesA', '1estA'], msf=True, fasta=False, processes=2)
        self.assertEqual(len(dsg.protein_data), 2)
        for p_id in dsg.protein_data:
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['MSF_Aln']))
            self.assertIsNone(dsg.protein_data[p_id]['FA_Aln'])
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')
        os.remove(fn2)
        os.remove(os.path.splitext(fn2)[0] + '.dnd')

    def test_align_blast_hits_multi_protein_multi_process_fasta(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        sleep(1)
        fn2 = write_out_temp_fn(suffix='fasta',
                                out_str=('>1estA\n' + str(self.original_seq.seq) + '\n' + self.proper_header1 +
                                         self.seq1))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq, 'Filtered_BLAST': fn},
                            '1estA': {'Sequence': self.original_seq, 'Filtered_BLAST': fn2}}
        dsg.align_blast_hits(unique_ids=['1tesA', '1estA'], msf=False, fasta=True, processes=2)
        self.assertEqual(len(dsg.protein_data), 2)
        for p_id in dsg.protein_data:
            self.assertIsNone(dsg.protein_data[p_id]['MSF_Aln'])
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['FA_Aln']))
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')
        os.remove(fn2)
        os.remove(os.path.splitext(fn2)[0] + '.dnd')


class TestDataSetGeneratorIdentityFilter(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.original_seq = SeqRecord(Seq('DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK', alphabet=FullIUPACProtein()), id='4rex',
                                     description='Target Query: Chain: A: From UniProt Accession: P46937')
        cls.ex_lin = 'Eukaryota;Metazoa;Chordata;Craniata;Vertebrata;Euteleostomi;Mammalia;Eutheria;Euarchontoglires;Glires;Rodentia;Hystricomorpha;Caviidae;Cavia'
        cls.proper_header1 = f'>2tesA HSP_identity={33} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={cls.ex_lin}\n'
        cls.seq1 = 'DVPLP-GWE-AKTSSGQRYFLNHIDQTTTWQDPRK\n'
        cls.proper_header2 = f'>3tesA HSP_identity={15} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={cls.ex_lin}\n'
        cls.seq2 = 'DVPLPAGWE-AKTSSGQRYFLNHIDQTTTWQDPRK\n'

    def test_identity_filter_remove_no_sequences(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        aln = SeqAlignment(file_name=fn, query_id='1tesA')
        aln.import_alignment()
        calculator = AlignmentDistanceCalculator()
        dist_mat = np.triu(1.0 - np.array(calculator.get_distance(aln.alignment, processes=1)), k=1)
        target_fn = './identity_filter.fasta'
        count = identity_filter(protein_id='1tesA', alignment=aln, distance_matrix=dist_mat,
                                identity_filtered_fn=target_fn, max_identity=0.98)
        self.assertEqual(count, 3)
        self.assertTrue(os.path.isfile(target_fn))
        os.remove(fn)
        os.remove(target_fn)

    def test_identity_filter_remove_one_sequence(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1 +
                                        '>4tesA\n' + str(self.original_seq.seq)))
        aln = SeqAlignment(file_name=fn, query_id='1tesA')
        aln.import_alignment()
        calculator = AlignmentDistanceCalculator()
        dist_mat = np.triu(1.0 - np.array(calculator.get_distance(aln.alignment, processes=1)), k=1)
        target_fn = './identity_filter.fasta'
        count = identity_filter(protein_id='1tesA', alignment=aln, distance_matrix=dist_mat,
                                identity_filtered_fn=target_fn, max_identity=0.98)
        self.assertEqual(count, 3)
        self.assertTrue(os.path.isfile(target_fn))
        os.remove(fn)
        os.remove(target_fn)

    def test_identity_filter_remove_one_cluster(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + '>4tesA\n' + str(self.original_seq.seq) + '\n' +
                                        self.proper_header1 + self.seq1 + '>5tesA\n' + str(self.original_seq.seq) +
                                        '\n' + '>6tesA\n' + str(self.original_seq.seq) + '\n'))
        aln = SeqAlignment(file_name=fn, query_id='1tesA')
        aln.import_alignment()
        calculator = AlignmentDistanceCalculator()
        dist_mat = np.triu(1.0 - np.array(calculator.get_distance(aln.alignment, processes=1)), k=1)
        target_fn = './identity_filter.fasta'
        count = identity_filter(protein_id='1tesA', alignment=aln, distance_matrix=dist_mat,
                                identity_filtered_fn=target_fn, max_identity=0.98)
        self.assertEqual(count, 3)
        self.assertTrue(os.path.isfile(target_fn))
        os.remove(fn)
        os.remove(target_fn)

    def test_identity_filter_remove_two_clusters(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + '>4tesA\n' + str(self.original_seq.seq) + '\n' +
                                        self.proper_header1 + self.seq1 + '>5tesA\n' + str(self.original_seq.seq) +
                                        '\n' + '>6tesA\n' + self.seq2 + '>7tesA\n' +
                                        str(self.original_seq.seq) + '\n' + '>8tesA\n' + self.seq2 + '\n'))
        aln = SeqAlignment(file_name=fn, query_id='1tesA')
        aln.import_alignment()
        calculator = AlignmentDistanceCalculator()
        dist_mat = np.triu(1.0 - np.array(calculator.get_distance(aln.alignment, processes=1)), k=1)
        target_fn = './identity_filter.fasta'
        count = identity_filter(protein_id='1tesA', alignment=aln, distance_matrix=dist_mat,
                                identity_filtered_fn=target_fn, max_identity=0.98)
        self.assertEqual(count, 3)
        self.assertTrue(os.path.isfile(target_fn))
        os.remove(fn)
        os.remove(target_fn)


class TestDataSetGeneratorIdentityFilterAlignment(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.original_seq = SeqRecord(Seq('DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK', alphabet=FullIUPACProtein()), id='4rex',
                                     description='Target Query: Chain: A: From UniProt Accession: P46937')
        cls.ex_lin = 'Eukaryota;Metazoa;Chordata;Craniata;Vertebrata;Euteleostomi;Mammalia;Eutheria;Euarchontoglires;Glires;Rodentia;Hystricomorpha;Caviidae;Cavia'
        cls.proper_header1 = f'>2tesA HSP_identity={33} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={cls.ex_lin}\n'
        cls.seq1 = 'DVPLP-GWE-AKTSSGQRYFLNHIDQTTTWQDPRK\n'
        cls.proper_header2 = f'>3tesA HSP_identity={15} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={cls.ex_lin}\n'
        cls.seq2 = 'DVPLPAGWE-AKTSSGQRYFLNHIDQTTTWQDPRK\n'

    def tearDown(self):
        rmtree('PDB')
        rmtree('Sequences')
        rmtree('BLAST')
        rmtree('Filtered_BLAST')
        rmtree('Alignments')
        rmtree('Filtered_Alignments')
        rmtree('Final_Alignments')

    def test_remove_no_sequences_single_protein_single_process(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))

        expected_protein_data = {'1tesA': {'FA_Aln': fn, 'Final_Count': 3,
                                           'Filtered_Alignment': './Filtered_Alignments/1tesA.fasta'}}

        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'FA_Aln': fn}}
        dsg.identity_filter_alignment(unique_ids={'1tesA'}, max_identity=0.98, processes=1, verbose=False)
        self.assertEqual(len(dsg.protein_data), 1)
        for p_id in dsg.protein_data:
            self.assertEqual(dsg.protein_data[p_id]['FA_Aln'], expected_protein_data[p_id]['FA_Aln'])
            self.assertEqual(dsg.protein_data[p_id]['Final_Count'], expected_protein_data[p_id]['Final_Count'])
            self.assertEqual(dsg.protein_data[p_id]['Filtered_Alignment'],
                             expected_protein_data[p_id]['Filtered_Alignment'])
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Filtered_Alignment']))
        os.remove(fn)

    def test_remove_no_sequences_single_protein_multi_process(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))

        expected_protein_data = {'1tesA': {'FA_Aln': fn, 'Final_Count': 3,
                                           'Filtered_Alignment': './Filtered_Alignments/1tesA.fasta'}}

        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'FA_Aln': fn}}
        dsg.identity_filter_alignment(unique_ids={'1tesA'}, max_identity=0.98, processes=2, verbose=False)
        self.assertEqual(len(dsg.protein_data), 1)
        for p_id in dsg.protein_data:
            self.assertEqual(dsg.protein_data[p_id]['FA_Aln'], expected_protein_data[p_id]['FA_Aln'])
            self.assertEqual(dsg.protein_data[p_id]['Final_Count'], expected_protein_data[p_id]['Final_Count'])
            self.assertEqual(dsg.protein_data[p_id]['Filtered_Alignment'],
                             expected_protein_data[p_id]['Filtered_Alignment'])
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Filtered_Alignment']))
        os.remove(fn)

    def test_remove_one_sequence_single_protein_multi_process(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1 +
                                        '>4tesA\n' + str(self.original_seq.seq)))

        expected_protein_data = {'1tesA': {'FA_Aln': fn, 'Final_Count': 3,
                                           'Filtered_Alignment': './Filtered_Alignments/1tesA.fasta'}}

        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'FA_Aln': fn}}
        dsg.identity_filter_alignment(unique_ids={'1tesA'}, max_identity=0.98, processes=2, verbose=False)
        self.assertEqual(len(dsg.protein_data), 1)
        for p_id in dsg.protein_data:
            self.assertEqual(dsg.protein_data[p_id]['FA_Aln'], expected_protein_data[p_id]['FA_Aln'])
            self.assertEqual(dsg.protein_data[p_id]['Final_Count'], expected_protein_data[p_id]['Final_Count'])
            self.assertEqual(dsg.protein_data[p_id]['Filtered_Alignment'],
                             expected_protein_data[p_id]['Filtered_Alignment'])
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Filtered_Alignment']))
        os.remove(fn)

    def test_identity_filter_remove_remove_one_cluster_single_protein_multi_process(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + '>4tesA\n' + str(self.original_seq.seq) + '\n' +
                                        self.proper_header1 + self.seq1 + '>5tesA\n' + str(self.original_seq.seq) +
                                        '\n' + '>6tesA\n' + str(self.original_seq.seq) + '\n'))

        expected_protein_data = {'1tesA': {'FA_Aln': fn, 'Final_Count': 3,
                                           'Filtered_Alignment': './Filtered_Alignments/1tesA.fasta'}}

        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'FA_Aln': fn}}
        dsg.identity_filter_alignment(unique_ids={'1tesA'}, max_identity=0.98, processes=2, verbose=False)
        self.assertEqual(len(dsg.protein_data), 1)
        for p_id in dsg.protein_data:
            self.assertEqual(dsg.protein_data[p_id]['FA_Aln'], expected_protein_data[p_id]['FA_Aln'])
            self.assertEqual(dsg.protein_data[p_id]['Final_Count'], expected_protein_data[p_id]['Final_Count'])
            self.assertEqual(dsg.protein_data[p_id]['Filtered_Alignment'],
                             expected_protein_data[p_id]['Filtered_Alignment'])
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Filtered_Alignment']))
        os.remove(fn)

    def test_identity_filter_remove_two_clusters_single_protein_multi_process(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + '>4tesA\n' + str(self.original_seq.seq) + '\n' +
                                        self.proper_header1 + self.seq1 + '>5tesA\n' + str(self.original_seq.seq) +
                                        '\n' + '>6tesA\n' + self.seq2 + '>7tesA\n' +
                                        str(self.original_seq.seq) + '\n' + '>8tesA\n' + self.seq2 + '\n'))

        expected_protein_data = {'1tesA': {'FA_Aln': fn, 'Final_Count': 3,
                                           'Filtered_Alignment': './Filtered_Alignments/1tesA.fasta'}}

        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'FA_Aln': fn}}
        dsg.identity_filter_alignment(unique_ids={'1tesA'}, max_identity=0.98, processes=2, verbose=False)
        self.assertEqual(len(dsg.protein_data), 1)
        for p_id in dsg.protein_data:
            self.assertEqual(dsg.protein_data[p_id]['FA_Aln'], expected_protein_data[p_id]['FA_Aln'])
            self.assertEqual(dsg.protein_data[p_id]['Final_Count'], expected_protein_data[p_id]['Final_Count'])
            self.assertEqual(dsg.protein_data[p_id]['Filtered_Alignment'],
                             expected_protein_data[p_id]['Filtered_Alignment'])
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Filtered_Alignment']))
        os.remove(fn)

    def test_identity_filter_remove_two_clusters_multi_protein_multi_process(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        sleep(1)
        fn2 = write_out_temp_fn(suffix='fasta',
                                out_str=('>1tesB\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                         self.seq2 + self.proper_header1 + self.seq1 +
                                         '>4tesA\n' + str(self.original_seq.seq)))
        sleep(1)
        fn3 = write_out_temp_fn(suffix='fasta',
                                out_str=('>1tesC\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                         self.seq2 + '>4tesA\n' + str(self.original_seq.seq) + '\n' +
                                         self.proper_header1 + self.seq1 + '>5tesA\n' + str(self.original_seq.seq) +
                                         '\n' + '>6tesA\n' + str(self.original_seq.seq) + '\n'))
        sleep(1)
        fn4 = write_out_temp_fn(suffix='fasta',
                                out_str=('>1tesD\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                         self.seq2 + '>4tesA\n' + str(self.original_seq.seq) + '\n' +
                                         self.proper_header1 + self.seq1 + '>5tesA\n' + str(self.original_seq.seq) +
                                         '\n' + '>6tesA\n' + self.seq2 + '>7tesA\n' +
                                         str(self.original_seq.seq) + '\n' + '>8tesA\n' + self.seq2 + '\n'))

        expected_protein_data = {'1tesA': {'FA_Aln': fn, 'Final_Count': 3,
                                           'Filtered_Alignment': './Filtered_Alignments/1tesA.fasta'},
                                 '1tesB': {'FA_Aln': fn2, 'Final_Count': 3,
                                           'Filtered_Alignment': './Filtered_Alignments/1tesB.fasta'},
                                 '1tesC': {'FA_Aln': fn3, 'Final_Count': 3,
                                           'Filtered_Alignment': './Filtered_Alignments/1tesC.fasta'},
                                 '1tesD': {'FA_Aln': fn4, 'Final_Count': 3,
                                           'Filtered_Alignment': './Filtered_Alignments/1tesD.fasta'}}

        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'FA_Aln': fn}, '1tesB': {'FA_Aln': fn2}, '1tesC': {'FA_Aln': fn3},
                            '1tesD': {'FA_Aln': fn4}}
        dsg.identity_filter_alignment(unique_ids={'1tesA', '1tesB', '1tesC', '1tesD'}, max_identity=0.98, processes=2, verbose=False)
        self.assertEqual(len(dsg.protein_data), 4)
        for p_id in dsg.protein_data:
            self.assertEqual(dsg.protein_data[p_id]['FA_Aln'], expected_protein_data[p_id]['FA_Aln'])
            self.assertEqual(dsg.protein_data[p_id]['Final_Count'], expected_protein_data[p_id]['Final_Count'])
            self.assertEqual(dsg.protein_data[p_id]['Filtered_Alignment'],
                             expected_protein_data[p_id]['Filtered_Alignment'])
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Filtered_Alignment']))
        os.remove(fn)
        os.remove(fn2)
        os.remove(fn3)
        os.remove(fn4)


class TestDataSetGeneratorAlignIdentityFiltered(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.original_seq = SeqRecord(Seq('DVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK', alphabet=FullIUPACProtein()), id='4rex',
                                     description='Target Query: Chain: A: From UniProt Accession: P46937')
        cls.ex_lin = 'Eukaryota;Metazoa;Chordata;Craniata;Vertebrata;Euteleostomi;Mammalia;Eutheria;Euarchontoglires;Glires;Rodentia;Hystricomorpha;Caviidae;Cavia'
        cls.proper_header1 = f'>2tesA HSP_identity={33} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={cls.ex_lin}\n'
        cls.seq1 = 'DVPLPGWEAKTSSGQRYFLNHIDQTTTWQDPRK\n'
        cls.proper_header2 = f'>3tesA HSP_identity={15} HSP_alignment_length={35} Fraction_length={1.0} HSP_taxonomy={cls.ex_lin}\n'
        cls.seq2 = 'DVPLPAGWEAKTSSGQRYFLNHIDQTTTWQDPRK\n'

    def tearDown(self):
        rmtree('PDB')
        rmtree('Sequences')
        rmtree('BLAST')
        rmtree('Filtered_BLAST')
        rmtree('Alignments')
        rmtree('Filtered_Alignments')
        rmtree('Final_Alignments')

    def test_align_identity_filtered_single_protein_single_process_msf_fasta(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq, 'Filtered_Alignment': fn}}
        dsg.align_identity_filtered(unique_ids=['1tesA'], msf=True, fasta=True, processes=1)
        self.assertEqual(len(dsg.protein_data), 1)
        for p_id in dsg.protein_data:
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Final_MSF_Aln']))
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Final_FA_Aln']))
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')

    def test_align_identity_filtered_single_protein_multi_process_msf_fasta(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq, 'Filtered_Alignment': fn}}
        dsg.align_identity_filtered(unique_ids=['1tesA'], msf=True, fasta=True, processes=2)
        self.assertEqual(len(dsg.protein_data), 1)
        for p_id in dsg.protein_data:
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Final_MSF_Aln']))
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Final_FA_Aln']))
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')

    def test_align_identity_filtered_multi_protein_single_process_msf_fasta(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        sleep(1)
        fn2 = write_out_temp_fn(suffix='fasta',
                                out_str=('>1estA\n' + str(self.original_seq.seq) + '\n' + self.proper_header1 +
                                         self.seq1))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq, 'Filtered_Alignment': fn},
                            '1estA': {'Sequence': self.original_seq, 'Filtered_Alignment': fn2}}
        dsg.align_identity_filtered(unique_ids=['1tesA', '1estA'], msf=True, fasta=True, processes=1)
        self.assertEqual(len(dsg.protein_data), 2)
        for p_id in dsg.protein_data:
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Final_MSF_Aln']))
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Final_FA_Aln']))
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')
        os.remove(fn2)
        os.remove(os.path.splitext(fn2)[0] + '.dnd')

    def test_align_identity_filtered_multi_protein_multi_process_msf_fasta(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        sleep(1)
        fn2 = write_out_temp_fn(suffix='fasta',
                                out_str=('>1estA\n' + str(self.original_seq.seq) + '\n' + self.proper_header1 +
                                         self.seq1))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq, 'Filtered_Alignment': fn},
                            '1estA': {'Sequence': self.original_seq, 'Filtered_Alignment': fn2}}
        dsg.align_identity_filtered(unique_ids=['1tesA', '1estA'], msf=True, fasta=True, processes=2)
        self.assertEqual(len(dsg.protein_data), 2)
        for p_id in dsg.protein_data:
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Final_MSF_Aln']))
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Final_FA_Aln']))
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')
        os.remove(fn2)
        os.remove(os.path.splitext(fn2)[0] + '.dnd')

    def test_align_identity_filtered_multi_protein_multi_process_msf(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        sleep(1)
        fn2 = write_out_temp_fn(suffix='fasta',
                                out_str=('>1estA\n' + str(self.original_seq.seq) + '\n' + self.proper_header1 +
                                         self.seq1))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq, 'Filtered_Alignment': fn},
                            '1estA': {'Sequence': self.original_seq, 'Filtered_Alignment': fn2}}
        dsg.align_identity_filtered(unique_ids=['1tesA', '1estA'], msf=True, fasta=False, processes=2)
        self.assertEqual(len(dsg.protein_data), 2)
        for p_id in dsg.protein_data:
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Final_MSF_Aln']))
            self.assertIsNone(dsg.protein_data[p_id]['Final_FA_Aln'])
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')
        os.remove(fn2)
        os.remove(os.path.splitext(fn2)[0] + '.dnd')

    def test_align_identity_filtered_multi_protein_multi_process_fasta(self):
        fn = write_out_temp_fn(suffix='fasta',
                               out_str=('>1tesA\n' + str(self.original_seq.seq) + '\n' + self.proper_header2 +
                                        self.seq2 + self.proper_header1 + self.seq1))
        sleep(1)
        fn2 = write_out_temp_fn(suffix='fasta',
                                out_str=('>1estA\n' + str(self.original_seq.seq) + '\n' + self.proper_header1 +
                                         self.seq1))
        dsg = DataSetGenerator(input_path='./')
        dsg.protein_data = {'1tesA': {'Sequence': self.original_seq, 'Filtered_Alignment': fn},
                            '1estA': {'Sequence': self.original_seq, 'Filtered_Alignment': fn2}}
        dsg.align_identity_filtered(unique_ids=['1tesA', '1estA'], msf=False, fasta=True, processes=2)
        self.assertEqual(len(dsg.protein_data), 2)
        for p_id in dsg.protein_data:
            self.assertIsNone(dsg.protein_data[p_id]['Final_MSF_Aln'])
            self.assertTrue(os.path.isfile(dsg.protein_data[p_id]['Final_FA_Aln']))
        os.remove(fn)
        os.remove(os.path.splitext(fn)[0] + '.dnd')
        os.remove(fn2)
        os.remove(os.path.splitext(fn2)[0] + '.dnd')


class TestDataSetGeneratorBuildPDBAlignmentDataSet(TestCase):

    def setUp(self):
        os.mkdir('./Input')

    def tearDown(self):
        rmtree('./Input')

    def test_build_pdb_alignment_data_set_single_protein_single_process_pdb(self):
        os.mkdir('./Input/ProteinLists')
        with open('./Input/ProteinLists/test.txt', 'w') as handle:
            handle.write('4rexA\n')
        dsg = DataSetGenerator(input_path='./Input')
        dsg.build_pdb_alignment_dataset(protein_list_fn='test.txt', processes=1, max_target_seqs=2500,
                                        e_value_threshold=0.05, database='test.fasta', remote=False, min_fraction=0.7,
                                        min_identity=0.40, max_identity=0.98, msf=True, fasta=True, sources=['PDB'],
                                        verbose=False)
        expected_protein_data = {'4rexA': {'Accession': '4rex', 'Chain': 'A', 'PDB': '4rex',
                                           'PDB_FN': './Input/PDB/re/pdb4rex.ent', 'Length': 48,
                                           'Seq_Fasta': './Input/Sequences/4rexA.fasta',
                                           'Sequence': 'GAMGFEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLS',
                                           'BLAST': './Input/BLAST/test_All_Seqs.xml', 'BLAST_Hits': 12,
                                           'Filtered_BLAST': './Input/Filtered_BLAST/4rexA.fasta', 'Filter_Count': 7,
                                           'MSF_Aln': './Input/Alignments/4rexA.msf',
                                           'FA_Aln': './Input/Alignments/4rexA.fasta',
                                           'Filtered_Alignment': './Input/Filtered_Alignments/4rexA.fasta',
                                           'Final_Count': 6, 'Final_MSF_Aln': './Input/Final_Alignments/4rexA.msf',
                                           'Final_FA_Aln': './Input/Final_Alignments/4rexA.fasta'}}
        self.assertEqual(len(dsg.protein_data), 1)
        for p_id in dsg.protein_data:
            self.assertEqual(len(dsg.protein_data[p_id]), len(expected_protein_data[p_id]))
            for key in dsg.protein_data[p_id]:
                if key == 'Sequence':
                    self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_protein_data[p_id][key])
                else:
                    self.assertEqual(dsg.protein_data[p_id][key], expected_protein_data[p_id][key],
                                     f'{p_id} - {key} failed {dsg.protein_data[p_id][key]} vs {expected_protein_data[p_id][key]}')
                    if key in ['PDB_FN', 'Seq_Fasta', 'BLAST', 'Filtered_BLAST', 'MSF_Aln', 'FA_Aln',
                               'Filtered_Alignment', 'Final_MSF_Aln', 'Final_FA_Aln']:
                        self.assertTrue(os.path.isfile(dsg.protein_data[p_id][key]))

    def test_build_pdb_alignment_data_set_single_protein_multi_process_pdb(self):
        os.mkdir('./Input/ProteinLists', )
        with open('./Input/ProteinLists/test.txt', 'w') as handle:
            handle.write('4rexA\n')
        dsg = DataSetGenerator(input_path='./Input')
        dsg.build_pdb_alignment_dataset(protein_list_fn='test.txt', processes=2, max_target_seqs=2500,
                                        e_value_threshold=0.05, database='test.fasta', remote=False, min_fraction=0.7,
                                        min_identity=0.40, max_identity=0.98, msf=True, fasta=True, sources=['PDB'],
                                        verbose=False)
        expected_protein_data = {'4rexA': {'Accession': '4rex', 'Chain': 'A', 'PDB': '4rex',
                                           'PDB_FN': './Input/PDB/re/pdb4rex.ent', 'Length': 48,
                                           'Seq_Fasta': './Input/Sequences/4rexA.fasta',
                                           'Sequence': 'GAMGFEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLS',
                                           'BLAST': './Input/BLAST/test_All_Seqs.xml', 'BLAST_Hits': 12,
                                           'Filtered_BLAST': './Input/Filtered_BLAST/4rexA.fasta', 'Filter_Count': 7,
                                           'MSF_Aln': './Input/Alignments/4rexA.msf',
                                           'FA_Aln': './Input/Alignments/4rexA.fasta',
                                           'Filtered_Alignment': './Input/Filtered_Alignments/4rexA.fasta',
                                           'Final_Count': 6, 'Final_MSF_Aln': './Input/Final_Alignments/4rexA.msf',
                                           'Final_FA_Aln': './Input/Final_Alignments/4rexA.fasta'}}
        self.assertEqual(len(dsg.protein_data), 1)
        for p_id in dsg.protein_data:
            self.assertEqual(len(dsg.protein_data[p_id]), len(expected_protein_data[p_id]))
            for key in dsg.protein_data[p_id]:
                if key == 'Sequence':
                    self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_protein_data[p_id][key])
                else:
                    self.assertEqual(dsg.protein_data[p_id][key], expected_protein_data[p_id][key],
                                     f'{p_id} - {key} failed {dsg.protein_data[p_id][key]} vs {expected_protein_data[p_id][key]}')
                    if key in ['PDB_FN', 'Seq_Fasta', 'BLAST', 'Filtered_BLAST', 'MSF_Aln', 'FA_Aln',
                               'Filtered_Alignment', 'Final_MSF_Aln', 'Final_FA_Aln']:
                        self.assertTrue(os.path.isfile(dsg.protein_data[p_id][key]))

    def test_build_pdb_alignment_data_set_single_protein_multi_process_pdb_and_gb(self):
        os.mkdir('./Input/ProteinLists', )
        with open('./Input/ProteinLists/test.txt', 'w') as handle:
            handle.write('4rexA\n')
        dsg = DataSetGenerator(input_path='./Input')
        dsg.build_pdb_alignment_dataset(protein_list_fn='test.txt', processes=2, max_target_seqs=2500,
                                        e_value_threshold=0.05, database='test.fasta', remote=False, min_fraction=0.7,
                                        min_identity=0.40, max_identity=0.98, msf=True, fasta=True,
                                        sources=['GB', 'PDB'], verbose=False)
        expected_protein_data = {'4rexA': {'Accession': '4rex', 'Chain': 'A', 'PDB': '4rex',
                                           'PDB_FN': './Input/PDB/re/pdb4rex.ent', 'Length': 48,
                                           'Seq_Fasta': './Input/Sequences/4rexA.fasta',
                                           'Sequence': 'GAMGFEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLS',
                                           'BLAST': './Input/BLAST/test_All_Seqs.xml', 'BLAST_Hits': 12,
                                           'Filtered_BLAST': './Input/Filtered_BLAST/4rexA.fasta', 'Filter_Count': 7,
                                           'MSF_Aln': './Input/Alignments/4rexA.msf',
                                           'FA_Aln': './Input/Alignments/4rexA.fasta',
                                           'Filtered_Alignment': './Input/Filtered_Alignments/4rexA.fasta',
                                           'Final_Count': 6, 'Final_MSF_Aln': './Input/Final_Alignments/4rexA.msf',
                                           'Final_FA_Aln': './Input/Final_Alignments/4rexA.fasta'}}
        self.assertEqual(len(dsg.protein_data), 1)
        for p_id in dsg.protein_data:
            self.assertEqual(len(dsg.protein_data[p_id]), len(expected_protein_data[p_id]))
            for key in dsg.protein_data[p_id]:
                if key == 'Sequence':
                    self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_protein_data[p_id][key])
                else:
                    self.assertEqual(dsg.protein_data[p_id][key], expected_protein_data[p_id][key],
                                     f'{p_id} - {key} failed {dsg.protein_data[p_id][key]} vs {expected_protein_data[p_id][key]}')
                    if key in ['PDB_FN', 'Seq_Fasta', 'BLAST', 'Filtered_BLAST', 'MSF_Aln', 'FA_Aln',
                               'Filtered_Alignment', 'Final_MSF_Aln', 'Final_FA_Aln']:
                        self.assertTrue(os.path.isfile(dsg.protein_data[p_id][key]))

    def test_build_pdb_alignment_data_set_single_protein_multi_process_gb_only(self):
        os.mkdir('./Input/ProteinLists', )
        with open('./Input/ProteinLists/test.txt', 'w') as handle:
            handle.write('4rexA\n')
        dsg = DataSetGenerator(input_path='./Input')
        with self.assertRaises(ValueError):
            dsg.build_pdb_alignment_dataset(protein_list_fn='test.txt', processes=2, max_target_seqs=2500,
                                            e_value_threshold=0.05, database='test.fasta', remote=False, min_fraction=0.7,
                                            min_identity=0.40, max_identity=0.98, msf=True, fasta=True,
                                            sources=['GB'], verbose=False)

    def test_build_pdb_alignment_data_set_single_protein_multi_process_unp_and_pdb(self):
        os.mkdir('./Input/ProteinLists', )
        with open('./Input/ProteinLists/test.txt', 'w') as handle:
            handle.write('4rexA\n')
        dsg = DataSetGenerator(input_path='./Input')
        dsg.build_pdb_alignment_dataset(protein_list_fn='test.txt', processes=2, max_target_seqs=2500,
                                        e_value_threshold=0.05, database='test.fasta', remote=False, min_fraction=0.7,
                                        min_identity=0.40, max_identity=0.98, msf=True, fasta=True,
                                        sources=['UNP', 'PDB'], verbose=False)
        expected_protein_data = {'4rexA': {'Accession': ['P46937', 'YAP1_HUMAN'], 'Chain': 'A', 'PDB': '4rex',
                                           'PDB_FN': './Input/PDB/re/pdb4rex.ent', 'Length': 45,
                                           'Seq_Fasta': './Input/Sequences/4rexA.fasta',
                                           'Sequence': 'FEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLSQ',
                                           'BLAST': './Input/BLAST/test_All_Seqs.xml', 'BLAST_Hits': 12,
                                           'Filtered_BLAST': './Input/Filtered_BLAST/4rexA.fasta', 'Filter_Count': 8,
                                           'MSF_Aln': './Input/Alignments/4rexA.msf',
                                           'FA_Aln': './Input/Alignments/4rexA.fasta',
                                           'Filtered_Alignment': './Input/Filtered_Alignments/4rexA.fasta',
                                           'Final_Count': 7, 'Final_MSF_Aln': './Input/Final_Alignments/4rexA.msf',
                                           'Final_FA_Aln': './Input/Final_Alignments/4rexA.fasta'}}
        self.assertEqual(len(dsg.protein_data), 1)
        for p_id in dsg.protein_data:
            self.assertEqual(len(dsg.protein_data[p_id]), len(expected_protein_data[p_id]))
            for key in dsg.protein_data[p_id]:
                if key == 'Sequence':
                    self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_protein_data[p_id][key])
                elif key == 'Accession':
                    self.assertIn(dsg.protein_data[p_id][key], expected_protein_data[p_id][key])
                else:
                    self.assertEqual(dsg.protein_data[p_id][key], expected_protein_data[p_id][key],
                                     f'{p_id} - {key} failed {dsg.protein_data[p_id][key]} vs {expected_protein_data[p_id][key]}')
                    if key in ['PDB_FN', 'Seq_Fasta', 'BLAST', 'Filtered_BLAST', 'MSF_Aln', 'FA_Aln',
                               'Filtered_Alignment', 'Final_MSF_Aln', 'Final_FA_Aln']:
                        self.assertTrue(os.path.isfile(dsg.protein_data[p_id][key]))

    def test_build_pdb_alignment_data_set_single_protein_multi_process_unp_only(self):
        os.mkdir('./Input/ProteinLists', )
        with open('./Input/ProteinLists/test.txt', 'w') as handle:
            handle.write('4rexA\n')
        dsg = DataSetGenerator(input_path='./Input')
        dsg.build_pdb_alignment_dataset(protein_list_fn='test.txt', processes=2, max_target_seqs=2500,
                                        e_value_threshold=0.05, database='test.fasta', remote=False, min_fraction=0.7,
                                        min_identity=0.40, max_identity=0.98, msf=True, fasta=True,
                                        sources=['UNP'], verbose=False)
        expected_protein_data = {'4rexA': {'Accession': ['P46937', 'YAP1_HUMAN'], 'Chain': 'A', 'PDB': '4rex',
                                           'PDB_FN': './Input/PDB/re/pdb4rex.ent', 'Length': 45,
                                           'Seq_Fasta': './Input/Sequences/4rexA.fasta',
                                           'Sequence': 'FEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLSQ',
                                           'BLAST': './Input/BLAST/test_All_Seqs.xml', 'BLAST_Hits': 12,
                                           'Filtered_BLAST': './Input/Filtered_BLAST/4rexA.fasta', 'Filter_Count': 8,
                                           'MSF_Aln': './Input/Alignments/4rexA.msf',
                                           'FA_Aln': './Input/Alignments/4rexA.fasta',
                                           'Filtered_Alignment': './Input/Filtered_Alignments/4rexA.fasta',
                                           'Final_Count': 7, 'Final_MSF_Aln': './Input/Final_Alignments/4rexA.msf',
                                           'Final_FA_Aln': './Input/Final_Alignments/4rexA.fasta'}}
        self.assertEqual(len(dsg.protein_data), 1)
        for p_id in dsg.protein_data:
            self.assertEqual(len(dsg.protein_data[p_id]), len(expected_protein_data[p_id]))
            for key in dsg.protein_data[p_id]:
                if key == 'Sequence':
                    self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_protein_data[p_id][key])
                elif key == 'Accession':
                    self.assertIn(dsg.protein_data[p_id][key], expected_protein_data[p_id][key])
                else:
                    self.assertEqual(dsg.protein_data[p_id][key], expected_protein_data[p_id][key],
                                     f'{p_id} - {key} failed {dsg.protein_data[p_id][key]} vs {expected_protein_data[p_id][key]}')
                    if key in ['PDB_FN', 'Seq_Fasta', 'BLAST', 'Filtered_BLAST', 'MSF_Aln', 'FA_Aln',
                               'Filtered_Alignment', 'Final_MSF_Aln', 'Final_FA_Aln']:
                        self.assertTrue(os.path.isfile(dsg.protein_data[p_id][key]))

    def test_build_pdb_alignment_data_set_multi_protein_single_process_pdb(self):
        os.mkdir('./Input/ProteinLists')
        with open('./Input/ProteinLists/test.txt', 'w') as handle:
            handle.write('4rexA\n1zljC')
        dsg = DataSetGenerator(input_path='./Input')
        dsg.build_pdb_alignment_dataset(protein_list_fn='test.txt', processes=1, max_target_seqs=2500,
                                        e_value_threshold=0.05, database='test.fasta', remote=False, min_fraction=0.7,
                                        min_identity=0.40, max_identity=0.98, msf=True, fasta=True, sources=['PDB'],
                                        verbose=False)
        expected_protein_data = {'4rexA': {'Accession': '4rex', 'Chain': 'A', 'PDB': '4rex',
                                           'PDB_FN': './Input/PDB/re/pdb4rex.ent', 'Length': 48,
                                           'Seq_Fasta': './Input/Sequences/4rexA.fasta',
                                           'Sequence': 'GAMGFEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLS',
                                           'BLAST': './Input/BLAST/test_All_Seqs.xml', 'BLAST_Hits': 12,
                                           'Filtered_BLAST': './Input/Filtered_BLAST/4rexA.fasta', 'Filter_Count': 7,
                                           'MSF_Aln': './Input/Alignments/4rexA.msf',
                                           'FA_Aln': './Input/Alignments/4rexA.fasta',
                                           'Filtered_Alignment': './Input/Filtered_Alignments/4rexA.fasta',
                                           'Final_Count': 6, 'Final_MSF_Aln': './Input/Final_Alignments/4rexA.msf',
                                           'Final_FA_Aln': './Input/Final_Alignments/4rexA.fasta'},
                                 '1zljC': {'Accession': '1zlj', 'Chain': 'C', 'PDB': '1zlj',
                                           'PDB_FN': './Input/PDB/zl/pdb1zlj.ent', 'Length': 67,
                                           'Seq_Fasta': './Input/Sequences/1zljC.fasta',
                                           'Sequence': 'DPLSGLTDQERTLLGLLSEGLTNKQIADRFLAEKTVKNYVSRLLAKLGERRTQAAVFATELKRSRPP',
                                           'BLAST': './Input/BLAST/test_All_Seqs.xml', 'BLAST_Hits': 8,
                                           'Filtered_BLAST': './Input/Filtered_BLAST/1zljC.fasta', 'Filter_Count': 9,
                                           'MSF_Aln': './Input/Alignments/1zljC.msf',
                                           'FA_Aln': './Input/Alignments/1zljC.fasta',
                                           'Filtered_Alignment': './Input/Filtered_Alignments/1zljC.fasta',
                                           'Final_Count': 8, 'Final_MSF_Aln': './Input/Final_Alignments/1zljC.msf',
                                           'Final_FA_Aln': './Input/Final_Alignments/1zljC.fasta'}}
        self.assertEqual(len(dsg.protein_data), 2)
        for p_id in dsg.protein_data:
            self.assertEqual(len(dsg.protein_data[p_id]), len(expected_protein_data[p_id]))
            for key in dsg.protein_data[p_id]:
                if key == 'Sequence':
                    self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_protein_data[p_id][key])
                else:
                    self.assertEqual(dsg.protein_data[p_id][key], expected_protein_data[p_id][key],
                                     f'{p_id} - {key} failed {dsg.protein_data[p_id][key]} vs {expected_protein_data[p_id][key]}')
                    if key in ['PDB_FN', 'Seq_Fasta', 'BLAST', 'Filtered_BLAST', 'MSF_Aln', 'FA_Aln',
                               'Filtered_Alignment', 'Final_MSF_Aln', 'Final_FA_Aln']:
                        self.assertTrue(os.path.isfile(dsg.protein_data[p_id][key]))

    def test_build_pdb_alignment_data_set_multi_protein_multi_process_pdb(self):
        os.mkdir('./Input/ProteinLists', )
        with open('./Input/ProteinLists/test.txt', 'w') as handle:
            handle.write('4rexA\n1zljC\n')
        dsg = DataSetGenerator(input_path='./Input')
        dsg.build_pdb_alignment_dataset(protein_list_fn='test.txt', processes=2, max_target_seqs=2500,
                                        e_value_threshold=0.05, database='test.fasta', remote=False, min_fraction=0.7,
                                        min_identity=0.40, max_identity=0.98, msf=True, fasta=True, sources=['PDB'],
                                        verbose=False)
        expected_protein_data = {'4rexA': {'Accession': '4rex', 'Chain': 'A', 'PDB': '4rex',
                                           'PDB_FN': './Input/PDB/re/pdb4rex.ent', 'Length': 48,
                                           'Seq_Fasta': './Input/Sequences/4rexA.fasta',
                                           'Sequence': 'GAMGFEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLS',
                                           'BLAST': './Input/BLAST/test_All_Seqs.xml', 'BLAST_Hits': 12,
                                           'Filtered_BLAST': './Input/Filtered_BLAST/4rexA.fasta', 'Filter_Count': 7,
                                           'MSF_Aln': './Input/Alignments/4rexA.msf',
                                           'FA_Aln': './Input/Alignments/4rexA.fasta',
                                           'Filtered_Alignment': './Input/Filtered_Alignments/4rexA.fasta',
                                           'Final_Count': 6, 'Final_MSF_Aln': './Input/Final_Alignments/4rexA.msf',
                                           'Final_FA_Aln': './Input/Final_Alignments/4rexA.fasta'},
                                 '1zljC': {'Accession': '1zlj', 'Chain': 'C', 'PDB': '1zlj',
                                           'PDB_FN': './Input/PDB/zl/pdb1zlj.ent', 'Length': 67,
                                           'Seq_Fasta': './Input/Sequences/1zljC.fasta',
                                           'Sequence': 'DPLSGLTDQERTLLGLLSEGLTNKQIADRFLAEKTVKNYVSRLLAKLGERRTQAAVFATELKRSRPP',
                                           'BLAST': './Input/BLAST/test_All_Seqs.xml', 'BLAST_Hits': 8,
                                           'Filtered_BLAST': './Input/Filtered_BLAST/1zljC.fasta', 'Filter_Count': 9,
                                           'MSF_Aln': './Input/Alignments/1zljC.msf',
                                           'FA_Aln': './Input/Alignments/1zljC.fasta',
                                           'Filtered_Alignment': './Input/Filtered_Alignments/1zljC.fasta',
                                           'Final_Count': 8, 'Final_MSF_Aln': './Input/Final_Alignments/1zljC.msf',
                                           'Final_FA_Aln': './Input/Final_Alignments/1zljC.fasta'}}
        self.assertEqual(len(dsg.protein_data), 2)
        for p_id in dsg.protein_data:
            self.assertEqual(len(dsg.protein_data[p_id]), len(expected_protein_data[p_id]))
            for key in dsg.protein_data[p_id]:
                if key == 'Sequence':
                    self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_protein_data[p_id][key])
                else:
                    self.assertEqual(dsg.protein_data[p_id][key], expected_protein_data[p_id][key],
                                     f'{p_id} - {key} failed {dsg.protein_data[p_id][key]} vs {expected_protein_data[p_id][key]}')
                    if key in ['PDB_FN', 'Seq_Fasta', 'BLAST', 'Filtered_BLAST', 'MSF_Aln', 'FA_Aln',
                               'Filtered_Alignment', 'Final_MSF_Aln', 'Final_FA_Aln']:
                        self.assertTrue(os.path.isfile(dsg.protein_data[p_id][key]))

    def test_build_pdb_alignment_data_set_multi_protein_multi_process_pdb_and_gb(self):
        os.mkdir('./Input/ProteinLists', )
        with open('./Input/ProteinLists/test.txt', 'w') as handle:
            handle.write('4rexA\n1zljC\n')
        dsg = DataSetGenerator(input_path='./Input')
        dsg.build_pdb_alignment_dataset(protein_list_fn='test.txt', processes=2, max_target_seqs=2500,
                                        e_value_threshold=0.05, database='test.fasta', remote=False, min_fraction=0.7,
                                        min_identity=0.40, max_identity=0.98, msf=True, fasta=True,
                                        sources=['GB', 'PDB'], verbose=False)
        expected_protein_data = {'4rexA': {'Accession': '4rex', 'Chain': 'A', 'PDB': '4rex',
                                           'PDB_FN': './Input/PDB/re/pdb4rex.ent', 'Length': 48,
                                           'Seq_Fasta': './Input/Sequences/4rexA.fasta',
                                           'Sequence': 'GAMGFEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLS',
                                           'BLAST': './Input/BLAST/test_All_Seqs.xml', 'BLAST_Hits': 12,
                                           'Filtered_BLAST': './Input/Filtered_BLAST/4rexA.fasta', 'Filter_Count': 7,
                                           'MSF_Aln': './Input/Alignments/4rexA.msf',
                                           'FA_Aln': './Input/Alignments/4rexA.fasta',
                                           'Filtered_Alignment': './Input/Filtered_Alignments/4rexA.fasta',
                                           'Final_Count': 6, 'Final_MSF_Aln': './Input/Final_Alignments/4rexA.msf',
                                           'Final_FA_Aln': './Input/Final_Alignments/4rexA.fasta'},
                                 '1zljC': {'Accession': ['15610269', 'NP_217649'], 'Chain': 'C', 'PDB': '1zlj',
                                           'PDB_FN': './Input/PDB/zl/pdb1zlj.ent', 'Length': 74,
                                           'Seq_Fasta': './Input/Sequences/1zljC.fasta',
                                           'Sequence': 'QDPLSGLTDQERTLLGLLSEGLTNKQIADRMFLAEKTVKNYVSRLLAKLGMERRTQAAVFATELKRSRPPGDGP',
                                           'BLAST': './Input/BLAST/test_All_Seqs.xml', 'BLAST_Hits': 8,
                                           'Filtered_BLAST': './Input/Filtered_BLAST/1zljC.fasta', 'Filter_Count': 7,
                                           'MSF_Aln': './Input/Alignments/1zljC.msf',
                                           'FA_Aln': './Input/Alignments/1zljC.fasta',
                                           'Filtered_Alignment': './Input/Filtered_Alignments/1zljC.fasta',
                                           'Final_Count': 7, 'Final_MSF_Aln': './Input/Final_Alignments/1zljC.msf',
                                           'Final_FA_Aln': './Input/Final_Alignments/1zljC.fasta'}}
        self.assertEqual(len(dsg.protein_data), 2)
        for p_id in dsg.protein_data:
            self.assertEqual(len(dsg.protein_data[p_id]), len(expected_protein_data[p_id]))
            for key in dsg.protein_data[p_id]:
                if key == 'Sequence':
                    self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_protein_data[p_id][key])
                elif key == 'Accession' and isinstance(expected_protein_data[p_id][key], list):
                    self.assertIn(dsg.protein_data[p_id][key], expected_protein_data[p_id][key])
                else:
                    self.assertEqual(dsg.protein_data[p_id][key], expected_protein_data[p_id][key],
                                     f'{p_id} - {key} failed {dsg.protein_data[p_id][key]} vs {expected_protein_data[p_id][key]}')
                    if key in ['PDB_FN', 'Seq_Fasta', 'BLAST', 'Filtered_BLAST', 'MSF_Aln', 'FA_Aln',
                               'Filtered_Alignment', 'Final_MSF_Aln', 'Final_FA_Aln']:
                        self.assertTrue(os.path.isfile(dsg.protein_data[p_id][key]))

    def test_build_pdb_alignment_data_set_multi_protein_multi_process_gb_only(self):
        os.mkdir('./Input/ProteinLists', )
        with open('./Input/ProteinLists/test.txt', 'w') as handle:
            handle.write('4rexA\n1zljC\n')
        dsg = DataSetGenerator(input_path='./Input')
        dsg.build_pdb_alignment_dataset(protein_list_fn='test.txt', processes=2, max_target_seqs=2500,
                                        e_value_threshold=0.05, database='test.fasta', remote=False, min_fraction=0.7,
                                        min_identity=0.40, max_identity=0.98, msf=True, fasta=True,
                                        sources=['GB'], verbose=False)
        expected_protein_data = {'4rexA': {'Accession': None, 'Chain': 'A', 'PDB': '4rex',
                                           'PDB_FN': './Input/PDB/re/pdb4rex.ent', 'Sequence': None, 'Length': 0,
                                           'Seq_Fasta': None},
                                 '1zljC': {'Accession': ['15610269', 'NP_217649'], 'Chain': 'C', 'PDB': '1zlj',
                                           'PDB_FN': './Input/PDB/zl/pdb1zlj.ent', 'Length': 74,
                                           'Seq_Fasta': './Input/Sequences/1zljC.fasta',
                                           'Sequence': 'QDPLSGLTDQERTLLGLLSEGLTNKQIADRMFLAEKTVKNYVSRLLAKLGMERRTQAAVFATELKRSRPPGDGP',
                                           'BLAST': './Input/BLAST/test_All_Seqs.xml', 'BLAST_Hits': 8,
                                           'Filtered_BLAST': './Input/Filtered_BLAST/1zljC.fasta', 'Filter_Count': 7,
                                           'MSF_Aln': './Input/Alignments/1zljC.msf',
                                           'FA_Aln': './Input/Alignments/1zljC.fasta',
                                           'Filtered_Alignment': './Input/Filtered_Alignments/1zljC.fasta',
                                           'Final_Count': 7, 'Final_MSF_Aln': './Input/Final_Alignments/1zljC.msf',
                                           'Final_FA_Aln': './Input/Final_Alignments/1zljC.fasta'}}
        self.assertEqual(len(dsg.protein_data), 2)
        for p_id in dsg.protein_data:
            print(dsg.protein_data[p_id])
            self.assertEqual(len(dsg.protein_data[p_id]), len(expected_protein_data[p_id]))
            for key in dsg.protein_data[p_id]:
                if expected_protein_data[p_id][key] is None:
                    self.assertIsNone(dsg.protein_data[p_id][key])
                elif key == 'Sequence':
                    self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_protein_data[p_id][key])
                elif key == 'Accession' and isinstance(expected_protein_data[p_id][key], list):
                    self.assertIn(dsg.protein_data[p_id][key], expected_protein_data[p_id][key])
                else:
                    self.assertEqual(dsg.protein_data[p_id][key], expected_protein_data[p_id][key],
                                     f'{p_id} - {key} failed {dsg.protein_data[p_id][key]} vs {expected_protein_data[p_id][key]}')
                    if key in ['PDB_FN', 'Seq_Fasta', 'BLAST', 'Filtered_BLAST', 'MSF_Aln', 'FA_Aln',
                               'Filtered_Alignment', 'Final_MSF_Aln', 'Final_FA_Aln']:
                        self.assertTrue(os.path.isfile(dsg.protein_data[p_id][key]))

    def test_build_pdb_alignment_data_set_multi_protein_multi_process_unp_and_pdb(self):
        os.mkdir('./Input/ProteinLists', )
        with open('./Input/ProteinLists/test.txt', 'w') as handle:
            handle.write('4rexA\n1zljC\n')
        dsg = DataSetGenerator(input_path='./Input')
        dsg.build_pdb_alignment_dataset(protein_list_fn='test.txt', processes=2, max_target_seqs=2500,
                                        e_value_threshold=0.05, database='test.fasta', remote=False, min_fraction=0.7,
                                        min_identity=0.40, max_identity=0.98, msf=True, fasta=True,
                                        sources=['UNP', 'PDB'], verbose=False)
        expected_protein_data = {'4rexA': {'Accession': ['P46937', 'YAP1_HUMAN'], 'Chain': 'A', 'PDB': '4rex',
                                           'PDB_FN': './Input/PDB/re/pdb4rex.ent', 'Length': 45,
                                           'Seq_Fasta': './Input/Sequences/4rexA.fasta',
                                           'Sequence': 'FEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLSQ',
                                           'BLAST': './Input/BLAST/test_All_Seqs.xml', 'BLAST_Hits': 12,
                                           'Filtered_BLAST': './Input/Filtered_BLAST/4rexA.fasta', 'Filter_Count': 8,
                                           'MSF_Aln': './Input/Alignments/4rexA.msf',
                                           'FA_Aln': './Input/Alignments/4rexA.fasta',
                                           'Filtered_Alignment': './Input/Filtered_Alignments/4rexA.fasta',
                                           'Final_Count': 7, 'Final_MSF_Aln': './Input/Final_Alignments/4rexA.msf',
                                           'Final_FA_Aln': './Input/Final_Alignments/4rexA.fasta'},
                                 '1zljC': {'Accession': '1zlj', 'Chain': 'C', 'PDB': '1zlj',
                                           'PDB_FN': './Input/PDB/zl/pdb1zlj.ent', 'Length': 67,
                                           'Seq_Fasta': './Input/Sequences/1zljC.fasta',
                                           'Sequence': 'DPLSGLTDQERTLLGLLSEGLTNKQIADRFLAEKTVKNYVSRLLAKLGERRTQAAVFATELKRSRPP',
                                           'BLAST': './Input/BLAST/test_All_Seqs.xml', 'BLAST_Hits': 8,
                                           'Filtered_BLAST': './Input/Filtered_BLAST/1zljC.fasta', 'Filter_Count': 9,
                                           'MSF_Aln': './Input/Alignments/1zljC.msf',
                                           'FA_Aln': './Input/Alignments/1zljC.fasta',
                                           'Filtered_Alignment': './Input/Filtered_Alignments/1zljC.fasta',
                                           'Final_Count': 8, 'Final_MSF_Aln': './Input/Final_Alignments/1zljC.msf',
                                           'Final_FA_Aln': './Input/Final_Alignments/1zljC.fasta'}}
        self.assertEqual(len(dsg.protein_data), 2)
        for p_id in dsg.protein_data:
            self.assertEqual(len(dsg.protein_data[p_id]), len(expected_protein_data[p_id]))
            for key in dsg.protein_data[p_id]:
                if key == 'Sequence':
                    self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_protein_data[p_id][key])
                elif key == 'Accession':
                    self.assertIn(dsg.protein_data[p_id][key], expected_protein_data[p_id][key])
                else:
                    self.assertEqual(dsg.protein_data[p_id][key], expected_protein_data[p_id][key],
                                     f'{p_id} - {key} failed {dsg.protein_data[p_id][key]} vs {expected_protein_data[p_id][key]}')
                    if key in ['PDB_FN', 'Seq_Fasta', 'BLAST', 'Filtered_BLAST', 'MSF_Aln', 'FA_Aln',
                               'Filtered_Alignment', 'Final_MSF_Aln', 'Final_FA_Aln']:
                        self.assertTrue(os.path.isfile(dsg.protein_data[p_id][key]))

    def test_build_pdb_alignment_data_set_multi_protein_multi_process_unp_only(self):
        os.mkdir('./Input/ProteinLists', )
        with open('./Input/ProteinLists/test.txt', 'w') as handle:
            handle.write('4rexA\n')
        dsg = DataSetGenerator(input_path='./Input')
        dsg.build_pdb_alignment_dataset(protein_list_fn='test.txt', processes=2, max_target_seqs=2500,
                                        e_value_threshold=0.05, database='test.fasta', remote=False, min_fraction=0.7,
                                        min_identity=0.40, max_identity=0.98, msf=True, fasta=True,
                                        sources=['UNP'], verbose=False)
        expected_protein_data = {'4rexA': {'Accession': ['P46937', 'YAP1_HUMAN'], 'Chain': 'A', 'PDB': '4rex',
                                           'PDB_FN': './Input/PDB/re/pdb4rex.ent', 'Length': 45,
                                           'Seq_Fasta': './Input/Sequences/4rexA.fasta',
                                           'Sequence': 'FEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLSQ',
                                           'BLAST': './Input/BLAST/test_All_Seqs.xml', 'BLAST_Hits': 12,
                                           'Filtered_BLAST': './Input/Filtered_BLAST/4rexA.fasta', 'Filter_Count': 8,
                                           'MSF_Aln': './Input/Alignments/4rexA.msf',
                                           'FA_Aln': './Input/Alignments/4rexA.fasta',
                                           'Filtered_Alignment': './Input/Filtered_Alignments/4rexA.fasta',
                                           'Final_Count': 7, 'Final_MSF_Aln': './Input/Final_Alignments/4rexA.msf',
                                           'Final_FA_Aln': './Input/Final_Alignments/4rexA.fasta'},
                                 '1zljC': {'Accession': None, 'Chain': 'A', 'PDB': '1zlj',
                                           'PDB_FN': './Input/PDB/re/pdb1jl.ent.ent', 'Sequence': None, 'Length': 0,
                                           'Seq_Fasta': None},
                                 }
        self.assertEqual(len(dsg.protein_data), 1)
        for p_id in dsg.protein_data:
            self.assertEqual(len(dsg.protein_data[p_id]), len(expected_protein_data[p_id]))
            for key in dsg.protein_data[p_id]:
                if key == 'Sequence':
                    self.assertEqual(str(dsg.protein_data[p_id][key].seq), expected_protein_data[p_id][key])
                elif key == 'Accession':
                    self.assertIn(dsg.protein_data[p_id][key], expected_protein_data[p_id][key])
                else:
                    self.assertEqual(dsg.protein_data[p_id][key], expected_protein_data[p_id][key],
                                     f'{p_id} - {key} failed {dsg.protein_data[p_id][key]} vs {expected_protein_data[p_id][key]}')
                    if key in ['PDB_FN', 'Seq_Fasta', 'BLAST', 'Filtered_BLAST', 'MSF_Aln', 'FA_Aln',
                               'Filtered_Alignment', 'Final_MSF_Aln', 'Final_FA_Aln']:
                        self.assertTrue(os.path.isfile(dsg.protein_data[p_id][key]))


if __name__ == '__main__':
    unittest.main()
