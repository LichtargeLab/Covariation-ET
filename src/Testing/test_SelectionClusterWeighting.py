"""

"""
import os
import sys
import numpy as np
from unittest import TestCase
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
from Evaluation.SequencePDBMap import SequencePDBMap
from Evaluation.ContactScorer import ContactScorer
from Evaluation.SelectionClusterWeighting import SelectionClusterWeighting, compute_w_and_w2_ave_sub
from Testing.test_PDBReference import chain_a_pdb_partial2, chain_a_pdb_partial, chain_b_pdb, chain_b_pdb_partial
from Testing.test_Base import (protein_seq1, protein_seq2, protein_seq3, write_out_temp_fn)

pro_str = f'>{protein_seq1.id}\n{protein_seq1.seq}\n>{protein_seq2.id}\n{protein_seq2.seq}\n>{protein_seq3.id}\n{protein_seq3.seq}'
pro_pdb1 = chain_a_pdb_partial2 + chain_a_pdb_partial
CONTACT_DISTANCE2 = 16


def et_calcDist(atoms1, atoms2):
    """return smallest distance (squared) between two groups of atoms"""
    # (not distant by more than ~100 A)
    # mind2=CONTACT_DISTANCE2+100
    c1 = atoms1[0]  # atoms must not be empty
    c2 = atoms2[0]
    mind2 = (c1[0] - c2[0]) * (c1[0] - c2[0]) + \
            (c1[1] - c2[1]) * (c1[1] - c2[1]) + \
            (c1[2] - c2[2]) * (c1[2] - c2[2])
    for c1 in atoms1:
        for c2 in atoms2:
            d2 = (c1[0] - c2[0]) * (c1[0] - c2[0]) + \
                 (c1[1] - c2[1]) * (c1[1] - c2[1]) + \
                 (c1[2] - c2[2]) * (c1[2] - c2[2])
            if d2 < mind2:
                mind2 = d2
    return mind2  # Square of distance between most proximate atoms


class TestSCWInit(TestCase):

    def test_init_biased(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        aln_obj2 = aln_obj.remove_gaps()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_obj, pdb_reference=pdb_obj, cutoff=0.5, chain='A')
        scorer.fit()
        scorer.measure_distance()
        scw_obj = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                            biased=True)
        self.assertEqual(scw_obj.query_pdb_mapper.seq_aln.seq_length, aln_obj2.seq_length)
        self.assertEqual(scw_obj.query_pdb_mapper.pdb_ref, pdb_obj)
        self.assertEqual(scw_obj.query_pdb_mapper.best_chain, 'A')
        self.assertTrue(scw_obj.query_pdb_mapper.is_aligned())
        self.assertEqual(np.sum(np.abs(scw_obj.distances - scorer.distances)), 0)
        self.assertTrue(scw_obj.bias)
        self.assertIsNone(scw_obj.w_and_w2_ave_sub)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_unbiased(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        aln_obj2 = aln_obj.remove_gaps()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_obj, pdb_reference=pdb_obj, cutoff=0.5, chain='A')
        scorer.fit()
        scorer.measure_distance()
        scw_obj = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                            biased=False)
        self.assertEqual(scw_obj.query_pdb_mapper.seq_aln.seq_length, aln_obj2.seq_length)
        self.assertEqual(scw_obj.query_pdb_mapper.pdb_ref, pdb_obj)
        self.assertEqual(scw_obj.query_pdb_mapper.best_chain, 'A')
        self.assertTrue(scw_obj.query_pdb_mapper.is_aligned())
        self.assertEqual(np.sum(np.abs(scw_obj.distances - scorer.distances)), 0)
        self.assertFalse(scw_obj.bias)
        self.assertIsNone(scw_obj.w_and_w2_ave_sub)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_no_sequence_pdb_map(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_obj, pdb_reference=pdb_obj, cutoff=0.5, chain='A')
        scorer.fit()
        scorer.measure_distance()
        with self.assertRaises(AssertionError):
            SelectionClusterWeighting(seq_pdb_map=None, pdb_dists=scorer.distances, biased=True)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_bad_sequence_pdb_map(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        seq_pdb_map = SequencePDBMap(query='seq1', query_alignment=aln_obj, query_structure=pdb_obj, chain='A')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_obj, pdb_reference=pdb_obj, cutoff=0.5, chain='A')
        scorer.fit()
        scorer.measure_distance()
        with self.assertRaises(AssertionError):
            SelectionClusterWeighting(seq_pdb_map=seq_pdb_map, pdb_dists=scorer.distances, biased=True)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_fail_no_pdb_dist(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_obj, pdb_reference=pdb_obj, cutoff=0.5, chain='A')
        scorer.fit()
        scorer.measure_distance()
        with self.assertRaises(AssertionError):
            SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=None, biased=True)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_fail_no_bias(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_obj, pdb_reference=pdb_obj, cutoff=0.5, chain='A')
        scorer.fit()
        scorer.measure_distance()
        with self.assertRaises(AssertionError):
            SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances, biased=None)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_init_fail_bad_bias(self):
        aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        aln_obj = SeqAlignment(file_name=aln_fn, query_id='seq1')
        aln_obj.import_alignment()
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb_obj = PDBReference(pdb_file=pdb_fn)
        pdb_obj.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_obj, pdb_reference=pdb_obj, cutoff=0.5, chain='A')
        scorer.fit()
        scorer.measure_distance()
        with self.assertRaises(AssertionError):
            SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances, biased='biased')
        os.remove(aln_fn)
        os.remove(pdb_fn)
