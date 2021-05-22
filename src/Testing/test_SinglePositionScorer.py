import os
import sys
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

from SupportingClasses.PDBReference import PDBReference
from SupportingClasses.SeqAlignment import SeqAlignment
from SupportingClasses.EvolutionaryTraceAlphabet import FullIUPACProtein
from Evaluation.SinglePositionScorer import SinglePositionScorer
from Evaluation.SequencePDBMap import SequencePDBMap
from Testing.test_Base import (protein_seq1, protein_seq2, protein_seq3, protein_aln, write_out_temp_fn)
from Testing.test_PDBReference import chain_a_pdb_partial2, chain_a_pdb_partial, chain_b_pdb, chain_b_pdb_partial

pro_str = f'>{protein_seq1.id}\n{protein_seq1.seq}\n>{protein_seq2.id}\n{protein_seq2.seq}\n>{protein_seq3.id}\n{protein_seq3.seq}'
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
pro_pdb1 = chain_a_pdb_partial2 + chain_a_pdb_partial
pro_pdb1_scramble = chain_a_pdb_partial + chain_a_pdb_partial2
pro_pdb2 = chain_b_pdb
pro_pdb2_scramble = chain_b_pdb + chain_b_pdb_partial
pro_pdb_full = pro_pdb1 + pro_pdb2
pro_pdb_full_scramble = pro_pdb1_scramble + pro_pdb2_scramble


class TestScorerInit(TestCase):

    def evaluate_init(self, expected_query, expected_aln, expected_structure, expected_cutoff, expected_chain):
        scorer = SinglePositionScorer(query=expected_query, seq_alignment=expected_aln,
                                      pdb_reference=expected_structure, chain=expected_chain)
        self.assertIsInstance(scorer.query_pdb_mapper, SequencePDBMap)
        self.assertFalse(scorer.query_pdb_mapper.is_aligned())
        self.assertIsNone(scorer.data)

    def test_init_aln_file_pdb_file_chain(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        self.evaluate_init(expected_query='seq1', expected_aln=aln_fn, expected_structure=pdb_fn, expected_cutoff=8.0,
                           expected_chain='A')
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_aln_obj_pdb_file_chain(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        self.evaluate_init(expected_query='seq1', expected_aln=aln_obj, expected_structure=pdb_fn, expected_cutoff=8.0,
                           expected_chain='A')
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_aln_file_pdb_obj_chain(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        self.evaluate_init(expected_query='seq1', expected_aln=aln_fn, expected_structure=pdb_obj, expected_cutoff=8.0,
                           expected_chain='A')
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_aln_file_pdb_file_no_chain(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        self.evaluate_init(expected_query='seq1', expected_aln=aln_fn, expected_structure=pdb_fn, expected_cutoff=8.0,
                           expected_chain=None)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_failure_empty(self):
        with self.assertRaises(TypeError):
            SinglePositionScorer()