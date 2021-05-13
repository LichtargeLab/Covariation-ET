import os
import sys
import unittest
from unittest import TestCase
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

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
from SupportingClasses.PDBReference import PDBReference
from SupportingClasses.EvolutionaryTraceAlphabet import FullIUPACProtein
from Evaluation.SequencePDBMap import SequencePDBMap
from Testing.test_Base import write_out_temp_fn, protein_seq1, protein_seq2, protein_seq3, protein_aln
from Testing.test_PDBReference import chain_a_pdb_partial2, chain_a_pdb_partial, chain_b_pdb, chain_b_pdb_partial

pro_str = f'>{protein_seq1.id}\n{protein_seq1.seq}\n>{protein_seq2.id}\n{protein_seq2.seq}\n'\
          f'>{protein_seq3.id}\n{protein_seq3.seq}'
pro_pdb1 = chain_a_pdb_partial2 + chain_a_pdb_partial
pro_pdb1_scramble = chain_a_pdb_partial + chain_a_pdb_partial2
pro_pdb2 = chain_b_pdb
pro_pdb2_scramble = chain_b_pdb + chain_b_pdb_partial
pro_pdb_full = pro_pdb1 + pro_pdb2
pro_pdb_full_scramble = pro_pdb1_scramble + pro_pdb2_scramble
aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
protein_aln1 = protein_aln.remove_gaps()
protein_aln1.file_name = aln_fn
protein_aln2 = SeqAlignment(query_id='seq2', file_name=aln_fn)
protein_aln2.import_alignment()
protein_aln2 = protein_aln2.remove_gaps()
protein_aln3 = SeqAlignment(query_id='seq3', file_name=aln_fn)
protein_aln3.import_alignment()
protein_aln3 = protein_aln3.remove_gaps()
os.remove(aln_fn)


class TestSequencePDBMapInit(TestCase):

    def evaluate_init(self, expected_query, expected_aln, expected_structure, expected_chain):
        mapper = SequencePDBMap(query=expected_query, query_alignment=expected_aln, query_structure=expected_structure,
                                chain=expected_chain)
        self.assertEqual(mapper.seq_aln, expected_aln)
        self.assertEqual(mapper.pdb_ref, expected_structure)
        if expected_chain:
            self.assertEqual(mapper.best_chain, expected_chain)
        else:
            self.assertIsNone(mapper.best_chain)
        self.assertIsNone(mapper.query_pdb_mapping)
        self.assertIsNone(mapper.pdb_query_mapping)

    def test_init_aln_file_pdb_file_chain(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        self.evaluate_init(expected_query='seq1', expected_aln=aln_fn, expected_structure=pdb_fn, expected_chain='A')
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_aln_obj_pdb_file_chain(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        self.evaluate_init(expected_query='seq1', expected_aln=aln_obj, expected_structure=pdb_fn, expected_chain='A')
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_aln_file_pdb_obj_chain(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        self.evaluate_init(expected_query='seq1', expected_aln=aln_fn, expected_structure=pdb_obj, expected_chain='A')
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_aln_file_pdb_file_no_chain(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        self.evaluate_init(expected_query='seq1', expected_aln=aln_fn, expected_structure=pdb_fn, expected_chain=None)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_failure_empty(self):
        with self.assertRaises(TypeError):
            SequencePDBMap()


class TestSequencePDBMapAlign(TestCase):

    def setUp(self):
        self.expected_mapping_A = {0: 0, 1: 1, 2: 2}
        self.expected_mapping_B = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        self.expected_mapping_mismatch = {0: 0, 1: 3, 2: 4}
        self.expected_pro_seq_2 = SeqRecord(id='seq2', seq=Seq('MTREE', alphabet=FullIUPACProtein()))
        self.expected_pro_seq_3 = SeqRecord(id='seq3', seq=Seq('MFREE', alphabet=FullIUPACProtein()))

    def evaluate_align(self, mapper, expected_aln, expected_struct, expected_chain, expected_mapping):
        reverse_expected_mapping = {v: k for k, v in expected_mapping.items()}
        self.assertEqual(mapper.seq_aln, expected_aln)
        self.assertEqual(mapper.pdb_ref, expected_struct)
        mapper.align()
        self.assertEqual(mapper.best_chain, expected_chain)
        self.assertEqual(mapper.query_pdb_mapping, expected_mapping)
        self.assertEqual(mapper.pdb_query_mapping, reverse_expected_mapping)
        mapper.best_chain = None
        mapper.align()
        self.assertEqual(mapper.best_chain, expected_chain)
        self.assertEqual(mapper.query_pdb_mapping, expected_mapping)
        self.assertEqual(mapper.pdb_query_mapping, reverse_expected_mapping)
        if type(expected_struct) is str:
            expected_struct = PDBReference(pdb_file=expected_struct)
            expected_struct.import_pdb(structure_id='1TES')

    def test_align_aln_file_pdb_file_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb_fn, chain='A')
        self.evaluate_align(mapper=mapper, expected_aln=aln_fn, expected_struct=pdb_fn, expected_chain='A',
                            expected_mapping=self.expected_mapping_A)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_file_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        mapper = SequencePDBMap(query='seq1', query_alignment=protein_aln1, query_structure=pdb_fn, chain='A')
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln1, expected_struct=pdb_fn, expected_chain='A',
                            expected_mapping=self.expected_mapping_A)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_file_pdb_obj_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb, chain='A')
        self.evaluate_align(mapper=mapper, expected_aln=aln_fn, expected_struct=pdb, expected_chain='A',
                            expected_mapping=self.expected_mapping_A)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_obj_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        mapper = SequencePDBMap(query='seq1', query_alignment=protein_aln1, query_structure=pdb, chain='A')
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln1, expected_struct=pdb, expected_chain='A',
                            expected_mapping=self.expected_mapping_A)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_obj_chain_not_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        mapper = SequencePDBMap(query='seq1', query_alignment=protein_aln1, query_structure=pdb, chain=None)
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln1, expected_struct=pdb, expected_chain='A',
                            expected_mapping=self.expected_mapping_A)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_file_scrambled_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1_scramble)
        mapper = SequencePDBMap(query='seq1', query_alignment=protein_aln1, query_structure=pdb_fn, chain='A')
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln1, expected_struct=pdb_fn, expected_chain='A',
                            expected_mapping=self.expected_mapping_A)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_file_scrambled_chain_not_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1_scramble)
        mapper = SequencePDBMap(query='seq1', query_alignment=protein_aln1, query_structure=pdb_fn, chain=None)
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln1, expected_struct=pdb_fn, expected_chain='A',
                            expected_mapping=self.expected_mapping_A)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_file_pdb_file_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb_fn, chain='B')
        self.evaluate_align(mapper=mapper, expected_aln=aln_fn, expected_struct=pdb_fn, expected_chain='B',
                            expected_mapping=self.expected_mapping_mismatch)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_file_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        mapper = SequencePDBMap(query='seq1', query_alignment=protein_aln1, query_structure=pdb_fn, chain='B')
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln1, expected_struct=pdb_fn, expected_chain='B',
                            expected_mapping=self.expected_mapping_mismatch)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_file_pdb_obj_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb, chain='B')
        self.evaluate_align(mapper=mapper, expected_aln=aln_fn, expected_struct=pdb, expected_chain='B',
                            expected_mapping=self.expected_mapping_mismatch)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_obj_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        mapper = SequencePDBMap(query='seq1', query_alignment=protein_aln1, query_structure=pdb, chain='B')
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln1, expected_struct=pdb, expected_chain='B',
                            expected_mapping=self.expected_mapping_mismatch)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_obj_chain_not_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        mapper = SequencePDBMap(query='seq1', query_alignment=protein_aln1, query_structure=pdb, chain=None)
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln1, expected_struct=pdb, expected_chain='B',
                            expected_mapping=self.expected_mapping_mismatch)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_file_scrambled_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        mapper = SequencePDBMap(query='seq1', query_alignment=protein_aln1, query_structure=pdb_fn, chain='B')
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln1, expected_struct=pdb_fn, expected_chain='B',
                            expected_mapping=self.expected_mapping_mismatch)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_file_scrambled_chain_not_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        mapper = SequencePDBMap(query='seq1', query_alignment=protein_aln1, query_structure=pdb_fn, chain=None)
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln1, expected_struct=pdb_fn, expected_chain='B',
                            expected_mapping=self.expected_mapping_mismatch)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_file_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        mapper = SequencePDBMap(query='seq2', query_alignment=aln_fn, query_structure=pdb_fn, chain='B')
        self.evaluate_align(mapper=mapper, expected_aln=aln_fn, expected_struct=pdb_fn, expected_chain='B',
                            expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_file_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        mapper = SequencePDBMap(query='seq2', query_alignment=protein_aln2, query_structure=pdb_fn, chain='B')
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln2, expected_struct=pdb_fn, expected_chain='B',
                            expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_file_pdb_obj_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        mapper = SequencePDBMap(query='seq2', query_alignment=aln_fn, query_structure=pdb, chain='B')
        self.evaluate_align(mapper=mapper, expected_aln=aln_fn, expected_struct=pdb, expected_chain='B',
                            expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_obj_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        mapper = SequencePDBMap(query='seq2', query_alignment=protein_aln2, query_structure=pdb, chain='B')
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln2, expected_struct=pdb, expected_chain='B',
                            expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_obj_chain_not_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        mapper = SequencePDBMap(query='seq2', query_alignment=protein_aln2, query_structure=pdb, chain=None)
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln2, expected_struct=pdb, expected_chain='B',
                            expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_file_scrambled_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        mapper = SequencePDBMap(query='seq2', query_alignment=protein_aln2, query_structure=pdb_fn, chain='B')
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln2, expected_struct=pdb_fn, expected_chain='B',
                            expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_file_scrambled_chain_not_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        mapper = SequencePDBMap(query='seq2', query_alignment=protein_aln2, query_structure=pdb_fn, chain=None)
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln2, expected_struct=pdb_fn, expected_chain='B',
                            expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_file_pdb_file_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        mapper = SequencePDBMap(query='seq3', query_alignment=aln_fn, query_structure=pdb_fn, chain='B')
        self.evaluate_align(mapper=mapper, expected_aln=aln_fn, expected_struct=pdb_fn, expected_chain='B',
                            expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_file_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        mapper = SequencePDBMap(query='seq3', query_alignment=protein_aln3, query_structure=pdb_fn, chain='B')
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln3, expected_struct=pdb_fn, expected_chain='B',
                            expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_file_pdb_obj_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        mapper = SequencePDBMap(query='seq3', query_alignment=aln_fn, query_structure=pdb, chain='B')
        self.evaluate_align(mapper=mapper, expected_aln=aln_fn, expected_struct=pdb, expected_chain='B',
                            expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_obj_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        mapper = SequencePDBMap(query='seq3', query_alignment=protein_aln3, query_structure=pdb, chain='B')
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln3, expected_struct=pdb, expected_chain='B',
                            expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_obj_chain_not_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        mapper = SequencePDBMap(query='seq3', query_alignment=protein_aln3, query_structure=pdb, chain=None)
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln3, expected_struct=pdb, expected_chain='B',
                            expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_align_aln_obj_pdb_file_scrambled_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full_scramble)
        mapper = SequencePDBMap(query='seq3', query_alignment=protein_aln3, query_structure=pdb_fn, chain='B')
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln3, expected_struct=pdb_fn, expected_chain='B',
                            expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_not_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full_scramble)
        mapper = SequencePDBMap(query='seq3', query_alignment=protein_aln3, query_structure=pdb_fn, chain=None)
        self.evaluate_align(mapper=mapper, expected_aln=protein_aln3, expected_struct=pdb_fn, expected_chain='B',
                            expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)


class TestSequencePDBMapIsAligned(TestCase):

    def test_not_aligned(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb_fn, chain=None)
        align_status = mapper.is_aligned()
        self.assertFalse(align_status)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_aligned(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb_fn, chain=None)
        mapper.align()
        align_status = mapper.is_aligned()
        self.assertTrue(align_status)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_best_chain_missing(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb_fn, chain=None)
        mapper.align()
        mapper.best_chain = None
        align_status = mapper.is_aligned()
        self.assertFalse(align_status)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_query_pdb_mapping_missing(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb_fn, chain=None)
        mapper.align()
        mapper.query_pdb_mapping = None
        align_status = mapper.is_aligned()
        self.assertFalse(align_status)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_pdb_query_mapping_missing(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb_fn, chain=None)
        mapper.align()
        mapper.pdb_query_mapping = None
        align_status = mapper.is_aligned()
        self.assertFalse(align_status)
        os.remove(aln_fn)
        os.remove(pdb_fn)


class TestSequencePDBMapMapSeqPositionToPDBRes(TestCase):

    def setUp(self):
        self.expected_mapping_A = {0: (1, 'M'), 1: (2, 'E'), 2: (3, 'T')}
        self.expected_mapping_B = {0: (1, 'M'), 1: (2, 'T'), 2: (3, 'R'), 3: (4, 'E'), 4: (5, 'E')}
        self.expected_mapping_mismatch = {0: (1, 'M'), 1: (4, 'E'), 2: (5, 'E')}

    def evaluate_map_seq_position_to_pdb_res(self, mapper, expected_mapping):
        for key in expected_mapping:
            pdb_res, pdb_char = mapper.map_seq_position_to_pdb_res(key)
            self.assertEqual(pdb_res, expected_mapping[key][0])
            self.assertEqual(pdb_char, expected_mapping[key][1])

    def test_first_alignment_and_pdb(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb_fn, chain='A')
        mapper.align()
        self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_A)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_first_alignment_and_scrambled_pdb(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1_scramble)
        mapper = SequencePDBMap(query='seq1', query_alignment=protein_aln1, query_structure=pdb_fn, chain='A')
        mapper.align()
        self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_A)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_first_alignment_second_pdb_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb_fn, chain='B')
        mapper.align()
        self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_mismatch)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_first_alignment_second_pdb_mismatch_scrambled(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        mapper = SequencePDBMap(query='seq1', query_alignment=protein_aln1, query_structure=pdb_fn, chain='B')
        mapper.align()
        self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_mismatch)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_second_alignment_and_pdb(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        mapper = SequencePDBMap(query='seq2', query_alignment=aln_fn, query_structure=pdb_fn, chain='B')
        mapper.align()
        self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_second_alignment_and_pdb_scrambled(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        mapper = SequencePDBMap(query='seq2', query_alignment=protein_aln2, query_structure=pdb_fn, chain='B')
        mapper.align()
        self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_third_alignment_full_pdb(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        mapper = SequencePDBMap(query='seq3', query_alignment=aln_fn, query_structure=pdb_fn, chain='B')
        mapper.align()
        self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fail_not_aligned(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb_fn, chain='A')
        with self.assertRaises(AttributeError):
            self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_A)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_first_alignment_and_pdb_bad_seq_pos(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb_fn, chain='A')
        mapper.align()
        pdb_res, pdb_char = mapper.map_seq_position_to_pdb_res(5)
        self.assertIsNone(pdb_res)
        self.assertIsNone(pdb_char)
        os.remove(aln_fn)
        os.remove(pdb_fn)


class TestSequencePDBMapMapPDBResToSeqPosition(TestCase):

    def setUp(self):
        self.expected_mapping_A = {1: (0, 'M'), 2: (1, 'E'), 3: (2, 'T')}
        self.expected_mapping_B = {1: (0, 'M'), 2: (1, 'T'), 3: (2, 'R'), 4: (3, 'E'), 5: (4, 'E')}
        self.expected_mapping_C = {1: (0, 'M'), 2: (1, 'F'), 3: (2, 'R'), 4: (3, 'E'), 5: (4, 'E')}
        self.expected_mapping_mismatch = {1: (0, 'M'), 4: (1, 'E'), 5: (2, 'T')}

    def evaluate_map_seq_position_to_pdb_res(self, mapper, expected_mapping):
        for key in expected_mapping:
            seq_pos, seq_char = mapper.map_pdb_res_to_seq_position(key)
            self.assertEqual(seq_pos, expected_mapping[key][0])
            self.assertEqual(seq_char, expected_mapping[key][1])

    def test_first_alignment_and_pdb(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb_fn, chain='A')
        mapper.align()
        self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_A)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_first_alignment_and_scrambled_pdb(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1_scramble)
        mapper = SequencePDBMap(query='seq1', query_alignment=protein_aln1, query_structure=pdb_fn, chain='A')
        mapper.align()
        self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_A)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_first_alignment_second_pdb_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb_fn, chain='B')
        mapper.align()
        self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_mismatch)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_first_alignment_second_pdb_mismatch_scrambled(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        mapper = SequencePDBMap(query='seq1', query_alignment=protein_aln1, query_structure=pdb_fn, chain='B')
        mapper.align()
        self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_mismatch)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_second_alignment_and_pdb(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        mapper = SequencePDBMap(query='seq2', query_alignment=aln_fn, query_structure=pdb_fn, chain='B')
        mapper.align()
        self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_second_alignment_and_pdb_scrambled(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        mapper = SequencePDBMap(query='seq2', query_alignment=protein_aln2, query_structure=pdb_fn, chain='B')
        mapper.align()
        self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_B)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_third_alignment_full_pdb(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        mapper = SequencePDBMap(query='seq3', query_alignment=aln_fn, query_structure=pdb_fn, chain='B')
        mapper.align()
        self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_C)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fail_not_aligned(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb_fn, chain='A')
        with self.assertRaises(AttributeError):
            self.evaluate_map_seq_position_to_pdb_res(mapper=mapper, expected_mapping=self.expected_mapping_A)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_first_alignment_and_pdb_bad_pdb_res(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        mapper = SequencePDBMap(query='seq1', query_alignment=aln_fn, query_structure=pdb_fn, chain='A')
        mapper.align()
        seq_pos, seq_char = mapper.map_pdb_res_to_seq_position(6)
        self.assertIsNone(seq_pos)
        self.assertIsNone(seq_char)
        os.remove(aln_fn)
        os.remove(pdb_fn)


if __name__ == '__main__':
    unittest.main()
