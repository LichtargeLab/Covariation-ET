import os
import sys
import datetime
import unittest
import numpy as np
import pandas as pd
from pymol import cmd
from math import floor
from shutil import rmtree
from unittest import TestCase
from scipy.stats import rankdata
from itertools import combinations
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve

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

from EvolutionaryTrace import EvolutionaryTrace
from SupportingClasses.SeqAlignment import SeqAlignment
from SupportingClasses.PDBReference import PDBReference
from SupportingClasses.utils import compute_rank_and_coverage
from SupportingClasses.EvolutionaryTraceAlphabet import FullIUPACProtein
from Evaluation.ContactScorer import (ContactScorer, surface_plot, heatmap_plot, plot_z_scores,
                                      init_scw_z_score_selection, scw_z_score_selection)
from Evaluation.SequencePDBMap import SequencePDBMap
from Evaluation.SelectionClusterWeighting import SelectionClusterWeighting
from Testing.test_Base import (protein_seq1, protein_seq2, protein_seq3, dna_seq1, dna_seq2, dna_seq3,
                               write_out_temp_fn, protein_aln)
from Testing.test_PDBReference import chain_a_pdb_partial2, chain_a_pdb_partial, chain_b_pdb, chain_b_pdb_partial


pro_str = f'>{protein_seq1.id}\n{protein_seq1.seq}\n>{protein_seq2.id}\n{protein_seq2.seq}\n>{protein_seq3.id}\n{protein_seq3.seq}'
pro_str_long = f'>{protein_seq1.id}\n{protein_seq1.seq*10}\n>{protein_seq2.id}\n{protein_seq2.seq*10}\n>{protein_seq3.id}\n{protein_seq3.seq*10}'
dna_str = f'>{dna_seq1.id}\n{dna_seq1.seq}\n>{dna_seq2.id}\n{dna_seq2.seq}\n>{dna_seq3.id}\n{dna_seq3.seq}'
test_dir = os.path.join(os.getcwd(), 'TestCase')
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

chain_a_pdb_partial2 = 'ATOM      1  N  AMET A   1     152.897  26.590  66.235  1.00 57.82           N  \n'\
                       'ATOM      1  N  BMET A   1     150.123  25.456  70.789  1.00 57.82           N  \n'\
                       'ATOM      2  CA AMET A   1     153.488  26.592  67.584  1.00 57.79           C  \n'\
                       'ATOM      2  CA BMET A   1     155.123  30.456  65.789  1.00 57.79           C  \n'\
                       'ATOM      3  C  AMET A   1     153.657  28.043  68.066  1.00 57.26           C  \n'\
                       'ATOM      3  C  BMET A   1     155.123  25.456  70.789  1.00 57.26           C  \n'\
                       'ATOM      4  O  AMET A   1     153.977  28.924  67.266  1.00 57.37           O  \n'\
                       'ATOM      4  O  BMET A   1     150.123  30.456  70.789  1.00 57.37           O  \n'\
                       'ATOM      5  CB AMET A   1     154.843  25.881  67.544  1.00 58.16           C  \n'\
                       'ATOM      5  CB BMET A   1     155.123  25.456  65.789  1.00 58.16           C  \n'\
                       'ATOM      6  CG AMET A   1     155.689  25.983  68.820  1.00 59.67           C  \n'\
                       'ATOM      6  CG BMET A   1     160.123  20.456  75.789  1.00 59.67           C  \n'\
                       'ATOM      7  SD AMET A   1     157.418  25.517  68.551  1.00 62.17           S  \n'\
                       'ATOM      7  SD BMET A   1     165.123  35.456  55.789  1.00 62.17           S  \n'\
                       'ATOM      8  CE AMET A   1     158.062  26.956  67.686  1.00 61.68           C  \n'\
                       'ATOM      8  CE BMET A   1     175.123  50.456  75.789  1.00 61.68           C  \n'

pro_pdb_1_alt_locs = chain_a_pdb_partial2 + chain_a_pdb_partial
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


def identify_expected_scores_and_distances(scorer, scores, coverages, ranks, distances, category='Any', n=None, k=None,
                                           cutoff=8.0, threshold=0.5):
    seq_sep_ind = scorer.find_pairs_by_separation(category=category, mappable_only=True)
    converted_ind = list(zip(*seq_sep_ind))
    dist_ind = [(scorer.query_pdb_mapper.query_pdb_mapping[x[0]], scorer.query_pdb_mapper.query_pdb_mapping[x[1]])
                for x in seq_sep_ind]
    converted_dist_ind = list(zip(*dist_ind))
    if n and k:
        raise ValueError('Both n and k cannot be defined when identifying data for testing.')
    elif n is None and k is None:
        # n = len(converted_ind[0])
        n = scorer.query_pdb_mapper.seq_aln.seq_length
    elif k is not None:
        n = int(floor(scorer.query_pdb_mapper.seq_aln.seq_length / float(k)))
    else:
        pass
    scores_subset = scores[converted_ind]
    coverage_subset = coverages[converted_ind]
    ranks_subset = ranks[converted_ind]
    preds_subset = coverage_subset <= threshold
    distance_subset = distances[converted_dist_ind]
    contact_subset = distance_subset <= cutoff
    if len(converted_ind) == 0:
        df_final = pd.DataFrame({'Seq Pos 1': [], 'Seq Pos 2': [], 'Struct Pos 1': [], 'Struct Pos 2': [],
                                 'Score': [], 'Coverage': [], 'Rank': [], 'Predictions': [], 'Distance': [],
                                 'Contact': [], 'Top Predictions': []})
    else:
        df = pd.DataFrame({'Seq Pos 1': converted_ind[0], 'Seq Pos 2': converted_ind[1],
                           'Struct Pos 1': [scorer.query_pdb_mapper.pdb_ref.pdb_residue_list[scorer.query_pdb_mapper.best_chain][x]
                                            for x in converted_dist_ind[0]],
                           'Struct Pos 2': [scorer.query_pdb_mapper.pdb_ref.pdb_residue_list[scorer.query_pdb_mapper.best_chain][x]
                                            for x in converted_dist_ind[1]],
                           'Score': scores_subset, 'Coverage': coverage_subset, 'Rank': ranks_subset,
                           'Predictions': preds_subset, 'Distance': distance_subset, 'Contact': contact_subset})
        df_sorted = df.sort_values(by='Coverage')
        df_sorted['Top Predictions'] = rankdata(df_sorted['Coverage'], method='dense')
        n_index = df_sorted['Top Predictions'] <= n
        df_final = df_sorted.loc[n_index, :]
    return df_final


class TestContactScorerInit(TestCase):

    def evaluate_init(self, expected_query, expected_aln, expected_structure, expected_cutoff, expected_chain):
        scorer = ContactScorer(query=expected_query, seq_alignment=expected_aln, pdb_reference=expected_structure,
                               cutoff=expected_cutoff, chain=expected_chain)
        self.assertIsInstance(scorer.query_pdb_mapper, SequencePDBMap)
        self.assertFalse(scorer.query_pdb_mapper.is_aligned())
        self.assertIsNone(scorer._specific_mapping)
        self.assertIsNone(scorer.distances)
        self.assertIsNone(scorer.dist_type)
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
            ContactScorer()


class TestContactScorerStr(TestCase):

    def evaluate_str(self, expected_query, expected_aln, expected_struct, expected_cutoff, expected_chain,
                     expected_str):
        scorer = ContactScorer(query=expected_query, seq_alignment=expected_aln, pdb_reference=expected_struct,
                               cutoff=expected_cutoff, chain=expected_chain)
        scorer.fit()
        scorer_str = str(scorer)
        self.assertEqual(scorer_str, expected_str)

    def test_str_aln_file_pdb_file_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        expected_str = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            3, 1, 'A')
        self.evaluate_str(expected_query='seq1', expected_cutoff=8.0, expected_aln=aln_fn, expected_struct=pdb_fn,
                          expected_chain='A', expected_str=expected_str)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_str_aln_obj_pdb_obj_chain_not_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        expected_str = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            3, 1, 'A')
        self.evaluate_str(expected_query='seq1', expected_cutoff=8.0, expected_aln=protein_aln1, expected_struct=pdb,
                          expected_chain='A', expected_str=expected_str)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_str_aln_obj_pdb_file_scrambled_chain_not_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1_scramble)
        expected_str = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            3, 1, 'A')
        self.evaluate_str(expected_query='seq1', expected_cutoff=8.0, expected_aln=protein_aln1, expected_struct=pdb_fn,
                          expected_chain='A', expected_str=expected_str)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_str_aln_file_pdb_file_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        expected_str = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            3, 1, 'B')
        self.evaluate_str(expected_query='seq1', expected_cutoff=8.0, expected_aln=aln_fn, expected_struct=pdb_fn,
                          expected_chain='B', expected_str=expected_str)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_str_aln_obj_pdb_obj_chain_not_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        expected_str = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            3, 1, 'B')
        self.evaluate_str(expected_query='seq1', expected_cutoff=8.0, expected_aln=protein_aln1, expected_struct=pdb,
                          expected_chain='B', expected_str=expected_str)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_str_aln_obj_pdb_file_scrambled_chain_not_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        expected_str = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            3, 1, 'B')
        self.evaluate_str(expected_query='seq1', expected_cutoff=8.0, expected_aln=protein_aln1, expected_struct=pdb_fn,
                          expected_chain='B', expected_str=expected_str)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_str_aln_file_pdb_file_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        expected_str = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            5, 1, 'B')
        self.evaluate_str(expected_query='seq2', expected_cutoff=8.0, expected_aln=aln_fn, expected_struct=pdb_fn,
                          expected_chain='B', expected_str=expected_str)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_str_aln_obj_pdb_obj_chain_not_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        expected_str = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            5, 1, 'B')
        self.evaluate_str(expected_query='seq2', expected_cutoff=8.0, expected_aln=protein_aln2, expected_struct=pdb,
                          expected_chain='B', expected_str=expected_str)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_str_aln_obj_pdb_file_scrambled_chain_not_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        expected_str = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            5, 1, 'B')
        self.evaluate_str(expected_query='seq2', expected_cutoff=8.0, expected_aln=protein_aln2, expected_struct=pdb_fn,
                          expected_chain='B', expected_str=expected_str)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_str_aln_file_pdb_file_chain_specified_3(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        expected_str = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            5, 2, 'B')
        self.evaluate_str(expected_query='seq3', expected_cutoff=8.0, expected_aln=aln_fn, expected_struct=pdb_fn,
                          expected_chain='B', expected_str=expected_str)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_str_aln_obj_pdb_obj_chain_not_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        expected_str = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            5, 2, 'B')
        self.evaluate_str(expected_query='seq3', expected_cutoff=8.0, expected_aln=protein_aln3, expected_struct=pdb,
                          expected_chain='B', expected_str=expected_str)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_str_aln_obj_pdb_file_scrambled_chain_not_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full_scramble)
        expected_str = 'Query Sequence of Length: {}\nPDB with {} Chains\nBest Sequence Match to Chain: {}'.format(
            5, 2, 'B')
        self.evaluate_str(expected_query='seq3', expected_cutoff=8.0, expected_aln=protein_aln3, expected_struct=pdb_fn,
                          expected_chain='B', expected_str=expected_str)
        os.remove(aln_fn)
        os.remove(pdb_fn)


class TestContactScorerFit(TestCase):

    def setUp(self):
        self.expected_mapping_A = {0: 0, 1: 1, 2: 2}
        self.expected_mapping_B = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        self.expected_mapping_mismatch = {0: 0, 1: 3, 2: 4}
        self.expected_pro_seq_2 = SeqRecord(id='seq2', seq=Seq('MTREE', alphabet=FullIUPACProtein()))
        self.expected_pro_seq_3 = SeqRecord(id='seq3', seq=Seq('MFREE', alphabet=FullIUPACProtein()))

    def evaluate_fit(self, scorer, expected_struct, expected_chain, expected_seq):
        self.assertFalse(scorer.query_pdb_mapper.is_aligned())
        scorer.fit()
        self.assertTrue(scorer.query_pdb_mapper.is_aligned())
        self.assertIsNotNone(scorer.data)
        scorer.query_pdb_mapper.best_chain = None
        self.assertFalse(scorer.query_pdb_mapper.is_aligned())
        scorer.fit()
        self.assertTrue(scorer.query_pdb_mapper.is_aligned())
        self.assertIsNotNone(scorer.data)
        scorer.query_pdb_mapper.query_pdb_mapping = None
        self.assertFalse(scorer.query_pdb_mapper.is_aligned())
        scorer.fit()
        self.assertTrue(scorer.query_pdb_mapper.is_aligned())
        self.assertIsNotNone(scorer.data)
        scorer.query_pdb_mapper.pdb_query_mapping = None
        self.assertFalse(scorer.query_pdb_mapper.is_aligned())
        scorer.fit()
        self.assertTrue(scorer.query_pdb_mapper.is_aligned())
        self.assertIsNotNone(scorer.data)
        if type(expected_struct) is str:
            expected_struct = PDBReference(pdb_file=expected_struct)
            expected_struct.import_pdb(structure_id='1TES')
        for i in scorer.data.index:
            self.assertEqual(scorer.data.loc[i, 'Seq AA 1'], expected_seq.seq[scorer.data.loc[i, 'Seq Pos 1']])
            self.assertEqual(scorer.data.loc[i, 'Seq AA 2'], expected_seq.seq[scorer.data.loc[i, 'Seq Pos 2']])
            self.assertEqual(scorer.data.loc[i, 'Seq Separation'],
                             scorer.data.loc[i, 'Seq Pos 2'] - scorer.data.loc[i, 'Seq Pos 1'])
            if scorer.data.loc[i, 'Seq Separation'] < 6:
                self.assertEqual(scorer.data.loc[i, 'Seq Separation Category'], 'Neighbors')
            elif scorer.data.loc[i, 'Seq Separation'] < 12:
                self.assertEqual(scorer.data.loc[i, 'Seq Separation Category'], 'Short')
            elif scorer.data.loc[i, 'Seq Separation'] < 24:
                self.assertEqual(scorer.data.loc[i, 'Seq Separation Category'], 'Medium')
            else:
                self.assertEqual(scorer.data.loc[i, 'Seq Separation Category'], 'Long')
            if scorer.data.loc[i, 'Struct Pos 1'] == '-':
                self.assertFalse(scorer.data.loc[i, 'Seq AA 1'] in scorer.query_pdb_mapper.query_pdb_mapping)
                self.assertEqual(scorer.data.loc[i, 'Struct AA 1'], '-')
            else:
                mapped_struct_pos = scorer.query_pdb_mapper.query_pdb_mapping[scorer.data.loc[i, 'Seq Pos 1']]
                self.assertEqual(scorer.data.loc[i, 'Struct Pos 1'],
                                 expected_struct.pdb_residue_list[expected_chain][mapped_struct_pos])
                self.assertEqual(scorer.data.loc[i, 'Struct AA 1'],
                                 expected_struct.seq[expected_chain][mapped_struct_pos])
            if scorer.data.loc[i, 'Struct Pos 2'] == '-':
                self.assertFalse(scorer.data.loc[i, 'Seq AA 2'] in scorer.query_pdb_mapper.query_pdb_mapping)
                self.assertEqual(scorer.data.loc[i, 'Struct AA 2'], '-')
            else:
                mapped_struct_pos = scorer.query_pdb_mapper.query_pdb_mapping[scorer.data.loc[i, 'Seq Pos 2']]
                self.assertEqual(scorer.data.loc[i, 'Struct Pos 2'],
                                 expected_struct.pdb_residue_list[expected_chain][mapped_struct_pos])
                self.assertEqual(scorer.data.loc[i, 'Struct AA 2'],
                                 expected_struct.seq[expected_chain][mapped_struct_pos])

    def test_fit_aln_file_pdb_file_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        scorer = ContactScorer(query='seq1', seq_alignment=aln_fn, pdb_reference=pdb_fn, cutoff=8.0, chain='A')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='A', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, cutoff=8.0, chain='A')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='A', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_obj_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_fn, pdb_reference=pdb, cutoff=8.0, chain='A')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='A', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb, cutoff=8.0, chain='A')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='A', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_not_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb, cutoff=8.0, chain=None)
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='A', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1_scramble)
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, cutoff=8.0, chain='A')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='A', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_not_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1_scramble)
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, cutoff=8.0, chain=None)
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='A', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_file_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        scorer = ContactScorer(query='seq1', seq_alignment=aln_fn, pdb_reference=pdb_fn, cutoff=8.0, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, cutoff=8.0, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_obj_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=aln_fn, pdb_reference=pdb, cutoff=8.0, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb, cutoff=8.0, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_not_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb, cutoff=8.0, chain=None)
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, cutoff=8.0, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_not_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, cutoff=8.0, chain=None)
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_file_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        scorer = ContactScorer(query='seq2', seq_alignment=aln_fn, pdb_reference=pdb_fn, cutoff=8.0, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_2)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=pdb_fn, cutoff=8.0, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_2)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_obj_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq2', seq_alignment=aln_fn, pdb_reference=pdb, cutoff=8.0, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=self.expected_pro_seq_2)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=pdb, cutoff=8.0, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=self.expected_pro_seq_2)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_not_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=pdb, cutoff=8.0, chain=None)
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=self.expected_pro_seq_2)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=pdb_fn, cutoff=8.0, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_2)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_not_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=pdb_fn, cutoff=8.0, chain=None)
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_2)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_file_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        scorer = ContactScorer(query='seq3', seq_alignment=aln_fn, pdb_reference=pdb_fn, cutoff=8.0, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_3)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=pdb_fn, cutoff=8.0, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_3)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_obj_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq3', seq_alignment=aln_fn, pdb_reference=pdb, cutoff=8.0, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=self.expected_pro_seq_3)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=pdb, cutoff=8.0, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=self.expected_pro_seq_3)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_not_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=pdb, cutoff=8.0, chain=None)
        self.evaluate_fit(scorer=scorer,  expected_struct=pdb, expected_chain='B', expected_seq=self.expected_pro_seq_3)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full_scramble)
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=pdb_fn, cutoff=8.0, chain='B')
        self.evaluate_fit(scorer=scorer,  expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_3)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_not_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full_scramble)
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=pdb_fn, cutoff=8.0, chain=None)
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_3)
        os.remove(aln_fn)
        os.remove(pdb_fn)


class TestContactScorerGetCoords(TestCase):  # get_all, get_c_alpha, get_c_beta, get_coords

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        protein_aln1.write_out_alignment(aln_fn)
        cls.expected_chain_a_coords = {1: {'all': [np.array([152.897, 26.590, 66.235], dtype=np.float32),
                                                   np.array([153.488, 26.592, 67.584], dtype=np.float32),
                                                   np.array([153.657, 28.043, 68.066], dtype=np.float32),
                                                   np.array([153.977, 28.924, 67.266], dtype=np.float32),
                                                   np.array([154.843, 25.881, 67.544], dtype=np.float32),
                                                   np.array([155.689, 25.983, 68.820], dtype=np.float32),
                                                   np.array([157.418, 25.517, 68.551], dtype=np.float32),
                                                   np.array([158.062, 26.956, 67.686], dtype=np.float32)],
                                           'alpha': [np.array([153.488, 26.592, 67.584], dtype=np.float32)],
                                           'beta': [np.array([154.843, 25.881, 67.544], dtype=np.float32)]},
                                       2: {'all': [np.array([153.913, 21.571, 52.586], dtype=np.float32),
                                                   np.array([153.617, 20.553, 53.599], dtype=np.float32),
                                                   np.array([153.850, 21.067, 55.018], dtype=np.float32),
                                                   np.array([153.177, 20.639, 55.960], dtype=np.float32),
                                                   np.array([154.431, 19.277, 53.350], dtype=np.float32),
                                                   np.array([153.767, 18.286, 52.392], dtype=np.float32),
                                                   np.array([153.675, 18.807, 50.965], dtype=np.float32),
                                                   np.array([152.542, 19.056, 50.495], dtype=np.float32),
                                                   np.array([154.735, 18.978, 50.321], dtype=np.float32)],
                                           'alpha': [np.array([153.617, 20.553, 53.599], dtype=np.float32)],
                                           'beta': [np.array([154.431, 19.277, 53.350], dtype=np.float32)]},
                                       3: {'all': [np.array([24.499, 13.739, 37.648], dtype=np.float32),
                                                   np.array([24.278, 13.068, 38.914], dtype=np.float32),
                                                   np.array([22.973, 13.448, 39.580], dtype=np.float32),
                                                   np.array([22.188, 12.566, 39.933], dtype=np.float32),
                                                   np.array([25.405, 13.382, 39.929], dtype=np.float32),
                                                   np.array([25.336, 12.578, 41.212], dtype=np.float32),
                                                   np.array([26.603, 13.084, 39.227], dtype=np.float32)],
                                           'alpha': [np.array([24.278, 13.068, 38.914], dtype=np.float32)],
                                           'beta': [np.array([25.405, 13.382, 39.929], dtype=np.float32)]}}
        cls.expected_chain_a_alt_coords = {1: {'all': [np.array([152.897, 26.590, 66.235], dtype=np.float32),
                                                       np.array([150.123, 25.456, 70.789], dtype=np.float32),
                                                       np.array([153.488, 26.592, 67.584], dtype=np.float32),
                                                       np.array([155.123, 30.456, 65.789], dtype=np.float32),
                                                       np.array([153.657, 28.043, 68.066], dtype=np.float32),
                                                       np.array([155.123, 25.456, 70.789], dtype=np.float32),
                                                       np.array([153.977, 28.924, 67.266], dtype=np.float32),
                                                       np.array([150.123, 30.456, 70.789], dtype=np.float32),
                                                       np.array([154.843, 25.881, 67.544], dtype=np.float32),
                                                       np.array([155.123, 25.456, 65.789], dtype=np.float32),
                                                       np.array([155.689, 25.983, 68.820], dtype=np.float32),
                                                       np.array([160.123, 20.456, 75.789], dtype=np.float32),
                                                       np.array([157.418, 25.517, 68.551], dtype=np.float32),
                                                       np.array([165.123, 35.456, 55.789], dtype=np.float32),
                                                       np.array([158.062, 26.956, 67.686], dtype=np.float32),
                                                       np.array([175.123, 50.456, 75.789], dtype=np.float32)],
                                               'alpha': [np.array([153.488, 26.592, 67.584], dtype=np.float32),
                                                         np.array([155.123, 30.456, 65.789], dtype=np.float32)],
                                               'beta': [np.array([154.843, 25.881, 67.544], dtype=np.float32),
                                                        np.array([155.123, 25.456, 65.789], dtype=np.float32)]},
                                           2: {'all': [np.array([153.913, 21.571, 52.586], dtype=np.float32),
                                                       np.array([153.617, 20.553, 53.599], dtype=np.float32),
                                                       np.array([153.850, 21.067, 55.018], dtype=np.float32),
                                                       np.array([153.177, 20.639, 55.960], dtype=np.float32),
                                                       np.array([154.431, 19.277, 53.350], dtype=np.float32),
                                                       np.array([153.767, 18.286, 52.392], dtype=np.float32),
                                                       np.array([153.675, 18.807, 50.965], dtype=np.float32),
                                                       np.array([152.542, 19.056, 50.495], dtype=np.float32),
                                                       np.array([154.735, 18.978, 50.321], dtype=np.float32)],
                                               'alpha': [np.array([153.617, 20.553, 53.599], dtype=np.float32)],
                                               'beta': [np.array([154.431, 19.277, 53.350], dtype=np.float32)]},
                                           3: {'all': [np.array([24.499, 13.739, 37.648], dtype=np.float32),
                                                       np.array([24.278, 13.068, 38.914], dtype=np.float32),
                                                       np.array([22.973, 13.448, 39.580], dtype=np.float32),
                                                       np.array([22.188, 12.566, 39.933], dtype=np.float32),
                                                       np.array([25.405, 13.382, 39.929], dtype=np.float32),
                                                       np.array([25.336, 12.578, 41.212], dtype=np.float32),
                                                       np.array([26.603, 13.084, 39.227], dtype=np.float32)],
                                               'alpha': [np.array([24.278, 13.068, 38.914], dtype=np.float32)],
                                               'beta': [np.array([25.405, 13.382, 39.929], dtype=np.float32)]}}
        cls.expected_chain_b_coords = {1: {'all': [np.array([152.897, 26.590, 66.235], dtype=np.float32),
                                                   np.array([153.488, 26.592, 67.584], dtype=np.float32),
                                                   np.array([153.657, 28.043, 68.066], dtype=np.float32),
                                                   np.array([153.977, 28.924, 67.266], dtype=np.float32),
                                                   np.array([154.843, 25.881, 67.544], dtype=np.float32),
                                                   np.array([155.689, 25.983, 68.820], dtype=np.float32),
                                                   np.array([157.418, 25.517, 68.551], dtype=np.float32),
                                                   np.array([158.062, 26.956, 67.686], dtype=np.float32)],
                                           'alpha': [np.array([153.488, 26.592, 67.584], dtype=np.float32)],
                                           'beta': [np.array([154.843, 25.881, 67.544], dtype=np.float32)]},
                                       2: {'all': [np.array([24.499, 13.739, 37.648], dtype=np.float32),
                                                   np.array([24.278, 13.068, 38.914], dtype=np.float32),
                                                   np.array([22.973, 13.448, 39.580], dtype=np.float32),
                                                   np.array([22.188, 12.566, 39.933], dtype=np.float32),
                                                   np.array([25.405, 13.382, 39.929], dtype=np.float32),
                                                   np.array([25.336, 12.578, 41.212], dtype=np.float32),
                                                   np.array([26.603, 13.084, 39.227], dtype=np.float32)],
                                           'alpha': [np.array([24.278, 13.068, 38.914], dtype=np.float32)],
                                           'beta': [np.array([25.405, 13.382, 39.929], dtype=np.float32)]},
                                       3: {'all': [np.array([24.805, 9.537, 22.454], dtype=np.float32),
                                                   np.array([24.052, 8.386, 22.974], dtype=np.float32),
                                                   np.array([24.897, 7.502, 23.849], dtype=np.float32),
                                                   np.array([24.504, 7.220, 24.972], dtype=np.float32),
                                                   np.array([23.506, 7.549, 21.793], dtype=np.float32),
                                                   np.array([22.741, 6.293, 22.182], dtype=np.float32),
                                                   np.array([22.242, 5.559, 20.931], dtype=np.float32),
                                                   np.array([23.319, 5.176, 20.037], dtype=np.float32),
                                                   np.array([23.931, 3.984, 20.083], dtype=np.float32),
                                                   np.array([23.622, 3.034, 20.961], dtype=np.float32),
                                                   np.array([24.895, 3.751, 19.199], dtype=np.float32)],
                                           'alpha': [np.array([24.052, 8.386, 22.974], dtype=np.float32)],
                                           'beta': [np.array([23.506, 7.549, 21.793], dtype=np.float32)]},
                                       4: {'all': [np.array([163.913, 21.571, 52.586], dtype=np.float32),
                                                   np.array([163.617, 20.553, 53.599], dtype=np.float32),
                                                   np.array([163.850, 21.067, 55.018], dtype=np.float32),
                                                   np.array([163.177, 20.639, 55.960], dtype=np.float32),
                                                   np.array([164.431, 19.277, 53.350], dtype=np.float32),
                                                   np.array([163.767, 18.286, 52.392], dtype=np.float32),
                                                   np.array([163.675, 18.807, 50.965], dtype=np.float32),
                                                   np.array([162.542, 19.056, 50.495], dtype=np.float32),
                                                   np.array([164.735, 18.978, 50.321], dtype=np.float32)],
                                           'alpha': [np.array([163.617, 20.553, 53.599], dtype=np.float32)],
                                           'beta': [np.array([164.431, 19.277, 53.350], dtype=np.float32)]},
                                       5: {'all': [np.array([153.913, 31.571, 52.586], dtype=np.float32),
                                                   np.array([153.617, 30.553, 53.599], dtype=np.float32),
                                                   np.array([153.850, 31.067, 55.018], dtype=np.float32),
                                                   np.array([153.177, 30.639, 55.960], dtype=np.float32),
                                                   np.array([154.431, 29.277, 53.350], dtype=np.float32),
                                                   np.array([153.767, 28.286, 52.392], dtype=np.float32),
                                                   np.array([153.675, 28.807, 50.965], dtype=np.float32),
                                                   np.array([152.542, 29.056, 50.495], dtype=np.float32),
                                                   np.array([154.735, 28.978, 50.321], dtype=np.float32)],
                                           'alpha': [np.array([153.617, 30.553, 53.599], dtype=np.float32)],
                                           'beta': [np.array([154.431, 29.277, 53.350], dtype=np.float32)]}}

    def evaluate_get_coords(self, query, pdb, chain, method_name, method, expected_coords, opt_param=None):
        cmd.load(pdb.file_name, query)
        cmd.select('best_chain', f'{query} and chain {chain}')
        for res_num in pdb.pdb_residue_list[chain]:
            residue = pdb.structure[0][chain][res_num]
            if opt_param is None:
                returned_coords = method(residue)
            else:
                returned_coords = method(residue, method=opt_param)
            self.assertEqual(len(returned_coords), len(expected_coords[res_num][method_name]))
            for i in range(len(returned_coords)):
                self.assertFalse((returned_coords[i] - expected_coords[res_num][method_name][i]).any())
        cmd.delete(query)

    def test_get_all_coords_chain_A(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=8.0, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a, chain='A', method_name='all',
                                 method=scorer._get_all_coords, expected_coords=self.expected_chain_a_coords)

    def test_get_alpha_coords_chain_A(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=8.0, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a, chain='A', method_name='alpha',
                                 method=scorer._get_c_alpha_coords, expected_coords=self.expected_chain_a_coords)

    def test_get_beta_coords_chain_A(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=8.0, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a, chain='A', method_name='beta',
                                 method=scorer._get_c_beta_coords, expected_coords=self.expected_chain_a_coords)

    def test_get_coords_all_chain_A(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=8.0, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a, chain='A', method_name='all',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_a_coords,
                                 opt_param='Any')

    def test_get_coords_alpha_chain_A(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=8.0, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a, chain='A', method_name='alpha',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_a_coords,
                                 opt_param='CA')

    def test_get_coords_beta_chain_A(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=8.0, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a, chain='A', method_name='beta',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_a_coords,
                                 opt_param='CB')

    def test_get_all_coords_chain_A_altLoc(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=8.0, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a_alt, chain='A', method_name='all',
                                 method=scorer._get_all_coords, expected_coords=self.expected_chain_a_alt_coords)

    def test_get_alpha_coords_chain_A_altLoc(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=8.0, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a_alt, chain='A', method_name='alpha',
                                 method=scorer._get_c_alpha_coords, expected_coords=self.expected_chain_a_alt_coords)

    def test_get_beta_coords_chain_A_altLoc(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=8.0, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a_alt, chain='A', method_name='beta',
                                 method=scorer._get_c_beta_coords, expected_coords=self.expected_chain_a_alt_coords)

    def test_get_coords_all_chain_A_altLoc(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=8.0, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a_alt, chain='A', method_name='all',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_a_alt_coords,
                                 opt_param='Any')

    def test_get_coords_alpha_chain_A_altLoc(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=8.0, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a_alt, chain='A', method_name='alpha',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_a_alt_coords,
                                 opt_param='CA')

    def test_get_coords_beta_chain_A_altLoc(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=8.0, chain='A')
        scorer.fit()
        self.evaluate_get_coords(query='seq1', pdb=self.pdb_chain_a_alt, chain='A', method_name='beta',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_a_alt_coords,
                                 opt_param='CB')

    def test_get_all_coords_chain_B(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=8.0, chain='B')
        scorer.fit()
        self.evaluate_get_coords(query='seq2', pdb=self.pdb_chain_b, chain='B', method_name='all',
                                 method=scorer._get_all_coords, expected_coords=self.expected_chain_b_coords)

    def test_get_alpha_coords_chain_B(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=8.0, chain='B')
        scorer.fit()
        self.evaluate_get_coords(query='seq2', pdb=self.pdb_chain_b, chain='B', method_name='alpha',
                                 method=scorer._get_c_alpha_coords, expected_coords=self.expected_chain_b_coords)

    def test_get_beta_coords_chain_B(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=8.0, chain='B')
        scorer.fit()
        self.evaluate_get_coords(query='seq2', pdb=self.pdb_chain_b, chain='B', method_name='beta',
                                 method=scorer._get_c_beta_coords, expected_coords=self.expected_chain_b_coords)

    def test_get_coords_all_chain_B(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=8.0, chain='B')
        scorer.fit()
        self.evaluate_get_coords(query='seq2', pdb=self.pdb_chain_b, chain='B', method_name='all',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_b_coords,
                                 opt_param='Any')

    def test_get_coords_alpha_chain_B(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=8.0, chain='B')
        scorer.fit()
        self.evaluate_get_coords(query='seq2', pdb=self.pdb_chain_b, chain='B', method_name='alpha',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_b_coords,
                                 opt_param='CA')

    def test_get_coords_beta_chain_B(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=8.0, chain='B')
        scorer.fit()
        self.evaluate_get_coords(query='seq2', pdb=self.pdb_chain_b, chain='B', method_name='beta',
                                 method=scorer._get_coords, expected_coords=self.expected_chain_b_coords,
                                 opt_param='CB')


class TestContactScorerMeasureDistance(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        cls.aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str)
        protein_aln1.write_out_alignment(aln_fn)

    def evaluate_measure_distance(self, scorer, method):
        scorer.fit()
        scorer.measure_distance(method=method)
        self.assertEqual(scorer.dist_type, method)
        residue_coords = {}
        size1 = len(scorer.query_pdb_mapper.pdb_ref.seq[scorer.query_pdb_mapper.best_chain])
        dists = np.zeros((size1, size1))
        dists2 = np.zeros((size1, size1))
        counter = 0
        counter_map = {}
        cmd.load(scorer.query_pdb_mapper.pdb_ref.file_name, scorer.query_pdb_mapper.query)
        cmd.select('best_chain', f'{scorer.query_pdb_mapper.query} and chain {scorer.query_pdb_mapper.best_chain}')
        for res_num in scorer.query_pdb_mapper.pdb_ref.residue_pos[scorer.query_pdb_mapper.best_chain]:
            counter_map[counter] = res_num
            residue = scorer.query_pdb_mapper.pdb_ref.structure[0][scorer.query_pdb_mapper.best_chain][res_num]
            coords = scorer._get_coords(residue, method=method)
            residue_coords[counter] = coords
            for residue2 in residue_coords:
                if residue2 == counter:
                    continue
                else:
                    if method == 'Any':
                        dist = et_calcDist(coords, residue_coords[residue2])
                    else:
                        dist = np.inf
                        for i in range(len(coords)):
                            for j in range(len(residue_coords[residue2])):
                                curr_dist = np.abs(np.power(coords[i][0] - residue_coords[residue2][j][0], 2) +
                                                   np.power(coords[i][1] - residue_coords[residue2][j][1], 2) +
                                                   np.power(coords[i][2] - residue_coords[residue2][j][2], 2))
                                if curr_dist < dist:
                                    dist = curr_dist
                    dist2 = np.sqrt(dist)
                    dists[counter, residue2] = dist
                    dists[residue2, counter] = dist
                    dists2[counter, residue2] = dist2
                    dists2[residue2, counter] = dist2
                    self.assertLess(np.abs(dist - np.square(
                        scorer.data.loc[(scorer.data['Struct Pos 1'] == counter_map[residue2]) &
                                        (scorer.data['Struct Pos 2'] == res_num), 'Distance'].values[0])), 1E-2)
                    self.assertLess(np.abs(dist2 -
                                           scorer.data.loc[(scorer.data['Struct Pos 1'] == counter_map[residue2]) &
                                                           (scorer.data['Struct Pos 2'] == res_num),
                                                           'Distance'].values[0]), 1E-4)
            counter += 1
        cmd.delete(scorer.query_pdb_mapper.query)
        distance_diff = np.square(scorer.distances) - dists
        self.assertLess(np.max(distance_diff), 1E-2)
        adj_diff = ((np.square(scorer.distances)[np.nonzero(distance_diff)] < CONTACT_DISTANCE2) ^
                    (dists[np.nonzero(distance_diff)] < CONTACT_DISTANCE2))
        self.assertEqual(np.sum(adj_diff), 0)
        self.assertEqual(len(np.nonzero(adj_diff)[0]), 0)
        distance_diff2 = scorer.distances - dists2
        self.assertLess(np.max(distance_diff2), 1E-4)

    def test_measure_distance_method_Any_chain_A(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=8.0, chain='A')
        self.evaluate_measure_distance(scorer=scorer, method='Any')

    def test_measure_distance_method_CA_chain_A(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=8.0, chain='A')
        self.evaluate_measure_distance(scorer=scorer, method='CA')

    def test_measure_distance_method_CB_chain_A(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=8.0, chain='A')
        self.evaluate_measure_distance(scorer=scorer, method='CB')

    def test_measure_distance_method_Any_chain_A_altLoc(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=8.0, chain='A')
        self.evaluate_measure_distance(scorer=scorer, method='Any')

    def test_measure_distance_method_CA_chain_A_altLoc(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=8.0, chain='A')
        self.evaluate_measure_distance(scorer=scorer, method='CA')

    def test_measure_distance_method_CB_chain_A_altLoc(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=8.0, chain='A')
        self.evaluate_measure_distance(scorer=scorer, method='CB')

    def test_measure_distance_method_Any_chain_B(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=8.0, chain='B')
        self.evaluate_measure_distance(scorer=scorer, method='Any')

    def test_measure_distance_method_CA_chain_B(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=8.0, chain='B')
        self.evaluate_measure_distance(scorer=scorer, method='CA')

    def test_measure_distance_method_CB_chain_B(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=8.0, chain='B')
        self.evaluate_measure_distance(scorer=scorer, method='CB')


class TestContactScorerFindPairsBySeparation(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(cls.aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        cls.aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str_long)
        cls.aln1 = SeqAlignment(query_id='seq1', file_name=cls.aln_fn)
        cls.aln1.import_alignment()
        cls.aln1 = cls.aln1.remove_gaps()
        cls.aln2 = SeqAlignment(query_id='seq2', file_name=cls.aln_fn)
        cls.aln2.import_alignment()
        cls.aln2 = cls.aln2.remove_gaps()
        cls.aln3 = SeqAlignment(query_id='seq3', file_name=cls.aln_fn)
        cls.aln3.import_alignment()
        cls.aln3 = cls.aln3.remove_gaps()

    def evaluate_find_pairs_by_separation(self, scorer):
        scorer.fit()
        with self.assertRaises(ValueError):
            scorer.find_pairs_by_separation(category='Wide')
        expected1 = {'Any': [], 'Neighbors': [], 'Short': [], 'Medium': [], 'Long': []}
        for i in range(scorer.query_pdb_mapper.seq_aln.seq_length):
            for j in range(i + 1, scorer.query_pdb_mapper.seq_aln.seq_length):
                pair = (i, j)
                separation = j - i
                if (separation >= 1) and (separation < 6):
                    expected1['Neighbors'].append(pair)
                if (separation >= 6) and (separation < 12):
                    expected1['Short'].append(pair)
                if (separation >= 12) and (separation < 24):
                    expected1['Medium'].append(pair)
                if separation >= 24:
                    expected1['Long'].append(pair)
                expected1['Any'].append(pair)
        self.assertEqual(scorer.find_pairs_by_separation(category='Any'), expected1['Any'])
        self.assertEqual(scorer.find_pairs_by_separation(category='Neighbors'), expected1['Neighbors'])
        self.assertEqual(scorer.find_pairs_by_separation(category='Short'), expected1['Short'])
        self.assertEqual(scorer.find_pairs_by_separation(category='Medium'), expected1['Medium'])
        self.assertEqual(scorer.find_pairs_by_separation(category='Long'), expected1['Long'])

    def test_find_pairs_by_separation_seq1(self):
        scorer = ContactScorer(query='seq1', seq_alignment=self.aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=8.0, chain='A')
        self.evaluate_find_pairs_by_separation(scorer)

    def test_find_pairs_by_separation_seq1_alt_loc_pdb(self):
        scorer = ContactScorer(query='seq1', seq_alignment=self.aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=8.0, chain='A')
        self.evaluate_find_pairs_by_separation(scorer)

    def test_find_pairs_by_separation_seq2(self):
        scorer = ContactScorer(query='seq2', seq_alignment=self.aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=8.0, chain='B')
        self.evaluate_find_pairs_by_separation(scorer)

    def test_find_pairs_by_separation_seq3(self):
        scorer = ContactScorer(query='seq3', seq_alignment=self.aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=8.0, chain='B')
        self.evaluate_find_pairs_by_separation(scorer)


class TestContactScorerMapPredictionsToPDB(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        protein_aln1.write_out_alignment(aln_fn)

    def evaluate_map_prediction_to_pdb(self, scorer):
        scorer.fit()
        scorer.measure_distance(method='CB')
        scores = np.random.rand(scorer.query_pdb_mapper.seq_aln.seq_length, scorer.query_pdb_mapper.seq_aln.seq_length)
        scores[np.tril_indices(scorer.query_pdb_mapper.seq_aln.seq_length, 1)] = 0
        scores += scores.T
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 2, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        self.assertIsNotNone(scorer.data)
        self.assertTrue(pd.Series(['Rank', 'Score', 'Coverage', 'True Prediction']).isin(scorer.data.columns).all())
        for i in scorer.data.index:
            pos1 = scorer.data.loc[i, 'Seq Pos 1']
            pos2 = scorer.data.loc[i, 'Seq Pos 2']
            self.assertEqual(ranks[pos1, pos2], scorer.data.loc[i, 'Rank'])
            self.assertEqual(scores[pos1, pos2], scorer.data.loc[i, 'Score'])
            self.assertEqual(coverages[pos1, pos2], scorer.data.loc[i, 'Coverage'])
            if coverages[pos1, pos2] <= 0.5:
                self.assertEqual(scorer.data.loc[i, 'True Prediction'], 1)
            else:
                self.assertEqual(scorer.data.loc[i, 'True Prediction'], 0)

    def test_map_prediction_to_pdb_seq1(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=8.0, chain='A')
        self.evaluate_map_prediction_to_pdb(scorer=scorer)

    def test_map_prediction_to_pdb_seq1_alt_loc_pdb(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=8.0, chain='A')
        self.evaluate_map_prediction_to_pdb(scorer=scorer)

    def test_map_prediction_to_pdb_seq2(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=8.0, chain='B')
        self.evaluate_map_prediction_to_pdb(scorer=scorer)

    def test_map_prediction_to_pdb_seq3(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=8.0, chain='B')
        self.evaluate_map_prediction_to_pdb(scorer=scorer)


class TestContactScorerIdentifyRelevantData(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(cls.aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        cls.aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str_long)
        cls.aln1 = SeqAlignment(query_id='seq1', file_name=cls.aln_fn)
        cls.aln1.import_alignment()
        cls.aln1 = cls.aln1.remove_gaps()
        cls.aln2 = SeqAlignment(query_id='seq2', file_name=cls.aln_fn)
        cls.aln2.import_alignment()
        cls.aln2 = cls.aln2.remove_gaps()
        cls.aln3 = SeqAlignment(query_id='seq3', file_name=cls.aln_fn)
        cls.aln3.import_alignment()
        cls.aln3 = cls.aln3.remove_gaps()

    def evaluate_identify_relevant_data(self, scorer):
        scorer.fit()
        scorer.measure_distance(method='CB')
        scores = np.random.rand(scorer.query_pdb_mapper.seq_aln.seq_length, scorer.query_pdb_mapper.seq_aln.seq_length)
        scores[np.tril_indices(scorer.query_pdb_mapper.seq_aln.seq_length)] = 0
        scores += scores.T
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 2, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        with self.assertRaises(ValueError):
            scorer._identify_relevant_data(category='Any', n=10, k=10)
        for category in ['Any', 'Neighbors', 'Short', 'Medium', 'Long']:
            for n, k in [(None, None), (1, None), (2, None), (3, None), (None, 1), (None, 2), (None, 3)]:
                curr_subset = scorer._identify_relevant_data(category=category, n=n, k=k)
                expected_subset = identify_expected_scores_and_distances(
                    scorer, scores, coverages, ranks, scorer.distances, category, n=n, k=k, cutoff=scorer.cutoff)
                if len(curr_subset) == 0:
                    self.assertEqual(len(expected_subset), 0)
                else:
                    self.assertEqual(len(curr_subset), len(expected_subset))
                    seq_1_pos_diff = np.abs(curr_subset['Seq Pos 1'].values - expected_subset['Seq Pos 1'].values)
                    seq_1_pos_not_passing = seq_1_pos_diff > 0
                    self.assertFalse(seq_1_pos_not_passing.any())
                    seq_2_pos_diff = np.abs(curr_subset['Seq Pos 2'].values - expected_subset['Seq Pos 2'].values)
                    seq_2_pos_not_passing = seq_2_pos_diff > 0
                    self.assertFalse(seq_2_pos_not_passing.any())
                    struct_1_pos_diff = np.abs(curr_subset['Struct Pos 1'].values -
                                               expected_subset['Struct Pos 1'].values)
                    struct_1_not_passing = struct_1_pos_diff > 0
                    self.assertFalse(struct_1_not_passing.any())
                    struct_2_pos_diff = np.abs(curr_subset['Struct Pos 2'].values -
                                               expected_subset['Struct Pos 2'].values)
                    struct_2_not_passing = struct_2_pos_diff > 0
                    self.assertFalse(struct_2_not_passing.any())
                    if k and (n is None):
                        n = int(floor(scorer.query_pdb_mapper.seq_aln.seq_length / float(k)))
                    if n:
                        self.assertFalse((curr_subset['Rank'] - expected_subset['Rank']).any())
                        self.assertLessEqual(len(curr_subset['Rank'].unique()), n)
                        self.assertLessEqual(len(expected_subset['Rank'].unique()), n)
                        self.assertFalse((curr_subset['Score'] - expected_subset['Score']).any())
                        self.assertLessEqual(len(curr_subset['Score'].unique()), n)
                        self.assertLessEqual(len(expected_subset['Score'].unique()), n)
                        self.assertFalse((curr_subset['Coverage'] - expected_subset['Coverage']).any())
                        self.assertLessEqual(len(curr_subset['Coverage'].unique()), n)
                        self.assertLessEqual(len(expected_subset['Coverage'].unique()), n)
                    else:
                        self.assertEqual(len(curr_subset['Rank'].unique()), len(expected_subset['Rank'].unique()))
                        self.assertEqual(len(curr_subset['Score'].unique()), len(expected_subset['Score'].unique()))
                        self.assertEqual(len(curr_subset['Coverage'].unique()),
                                         len(expected_subset['Coverage'].unique()))
                    self.assertEqual(len(curr_subset['Distance'].unique()), len(expected_subset['Distance'].unique()))
                    self.assertEqual(len(curr_subset['Contact (within {}A cutoff)'.format(scorer.cutoff)].unique()),
                                     len(expected_subset['Contact'].unique()))
                    self.assertEqual(len(curr_subset['True Prediction'].unique()),
                                     len(expected_subset['Predictions'].unique()))
                    diff_ranks = np.abs(curr_subset['Rank'].values - expected_subset['Rank'].values)
                    not_passing_ranks = diff_ranks > 1E-12
                    self.assertFalse(not_passing_ranks.any())
                    diff_scores = np.abs(curr_subset['Score'].values - expected_subset['Score'].values)
                    not_passing_scores = diff_scores > 1E-12
                    self.assertFalse(not_passing_scores.any())
                    diff_coverages = np.abs(curr_subset['Coverage'].values - expected_subset['Coverage'].values)
                    not_passing_coverages = diff_coverages > 1E-12
                    self.assertFalse(not_passing_coverages.any())
                    diff_preds = curr_subset['True Prediction'].values ^ expected_subset['Predictions'].values
                    not_passing_preds = diff_preds > 1E-12
                    self.assertFalse(not_passing_preds.any())
                    diff_contacts = (curr_subset['Contact (within {}A cutoff)'.format(scorer.cutoff)].values ^
                                     expected_subset['Contact'].values)
                    not_passing_contacts = diff_contacts > 1E-12
                    self.assertFalse(not_passing_contacts.any())
                    diff_distances = np.abs(curr_subset['Distance'].values - expected_subset['Distance'].values)
                    not_passing_distances = diff_distances > 1E-12
                    self.assertFalse(not_passing_distances.any())

    def test_identify_relevant_data_seq1(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        self.evaluate_identify_relevant_data(scorer=scorer)

    def test_identify_relevant_data_seq1_alt_loc_pdb(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=14.0, chain='A')
        self.evaluate_identify_relevant_data(scorer=scorer)

    def test_identify_relevant_data_seq2(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_identify_relevant_data(scorer=scorer)

    def test_identify_relevant_data_seq3(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_identify_relevant_data(scorer=scorer)

    def evaluate_identify_relevant_data_lists(self, scorer):
        scorer.fit()
        scorer.measure_distance(method='CB')
        scores = np.random.rand(scorer.query_pdb_mapper.seq_aln.seq_length, scorer.query_pdb_mapper.seq_aln.seq_length)
        scores[np.tril_indices(scorer.query_pdb_mapper.seq_aln.seq_length)] = 0
        scores += scores.T
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 2, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        with self.assertRaises(ValueError):
            scorer._identify_relevant_data(category='Any', n=10, k=10)
        for category in [['Any', 'Neighbors'], ['Any', 'Neighbors', 'Short', 'Medium', 'Long'],
                         ['Neighbors', 'Short'], ['Short', 'Medium', 'Long']]:
            for n, k in [(None, None), (1, None), (2, None), (3, None), (None, 1), (None, 2), (None, 3)]:
                curr_subset = scorer._identify_relevant_data(category=category, n=n, k=k)
                curr_subset = curr_subset.sort_values(by=['Coverage', 'Seq Pos 1', 'Seq Pos 2'])
                expected_subset = None
                for cat in category:
                    if expected_subset is None:
                        expected_subset = identify_expected_scores_and_distances(
                            scorer, scores, coverages, ranks, scorer.distances, cat, n=n, k=k)
                    else:
                        expected_subset = expected_subset.append(identify_expected_scores_and_distances(
                            scorer, scores, coverages, ranks, scorer.distances, cat, n=n, k=k))
                expected_subset = expected_subset.sort_values(by=['Coverage', 'Seq Pos 1', 'Seq Pos 2'])
                expected_subset = expected_subset.drop_duplicates()
                self.assertEqual(len(curr_subset), len(expected_subset))
                self.assertFalse((curr_subset['Seq Pos 1'].values - expected_subset['Seq Pos 1'].values).any())
                self.assertFalse((curr_subset['Seq Pos 2'].values - expected_subset['Seq Pos 2'].values).any())
                self.assertFalse((curr_subset['Struct Pos 1'].values - expected_subset['Struct Pos 1'].values).any())
                self.assertFalse((curr_subset['Struct Pos 2'].values - expected_subset['Struct Pos 2'].values).any())
                self.assertFalse((curr_subset['Rank'] - expected_subset['Rank']).any())
                self.assertFalse((curr_subset['Score'] - expected_subset['Score']).any())
                self.assertFalse((curr_subset['Coverage'] - expected_subset['Coverage']).any())
                self.assertFalse((curr_subset['Distance'] - expected_subset['Distance']).any())
                self.assertFalse((curr_subset['Contact (within {}A cutoff)'.format(scorer.cutoff)] -
                                  expected_subset['Contact']).any())
                self.assertFalse((curr_subset['True Prediction'] - expected_subset['Predictions']).any())
                    # diff_ranks = np.abs(curr_subset['Rank'].values - expected_subset['Rank'].values)
                    # not_passing_ranks = diff_ranks > 1E-12
                    # self.assertFalse(not_passing_ranks.any())
                    # diff_scores = np.abs(curr_subset['Score'].values - expected_subset['Score'].values)
                    # not_passing_scores = diff_scores > 1E-12
                    # self.assertFalse(not_passing_scores.any())
                    # diff_coverages = np.abs(curr_subset['Coverage'].values - expected_subset['Coverage'].values)
                    # not_passing_coverages = diff_coverages > 1E-12
                    # self.assertFalse(not_passing_coverages.any())
                    # diff_preds = curr_subset['True Prediction'].values ^ expected_subset['Predictions'].values
                    # not_passing_preds = diff_preds > 1E-12
                    # self.assertFalse(not_passing_preds.any())
                    # diff_contacts = (curr_subset['Contact (within {}A cutoff)'.format(scorer.cutoff)].values ^
                    #                  expected_subset['Contact'].values)
                    # not_passing_contacts = diff_contacts > 1E-12
                    # self.assertFalse(not_passing_contacts.any())
                    # diff_distances = np.abs(curr_subset['Distance'].values - expected_subset['Distance'].values)
                    # not_passing_distances = diff_distances > 1E-12
                    # self.assertFalse(not_passing_distances.any())

    def test_identify_relevant_data_lists_seq1(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        self.evaluate_identify_relevant_data_lists(scorer=scorer)

    def test_identify_relevant_data_lists_seq1_alt_loc_pdb(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=14.0, chain='A')
        self.evaluate_identify_relevant_data_lists(scorer=scorer)

    def test_identify_relevant_data_lists_seq2(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_identify_relevant_data_lists(scorer=scorer)

    def test_identify_relevant_data_lists_seq3(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_identify_relevant_data_lists(scorer=scorer)


class TestContactScorerScoreAUC(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(cls.aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        cls.aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str_long)
        cls.aln1 = SeqAlignment(query_id='seq1', file_name=cls.aln_fn)
        cls.aln1.import_alignment()
        cls.aln1 = cls.aln1.remove_gaps()
        cls.aln2 = SeqAlignment(query_id='seq2', file_name=cls.aln_fn)
        cls.aln2.import_alignment()
        cls.aln2 = cls.aln2.remove_gaps()
        cls.aln3 = SeqAlignment(query_id='seq3', file_name=cls.aln_fn)
        cls.aln3.import_alignment()
        cls.aln3 = cls.aln3.remove_gaps()

    def evaluate_score_auc(self, scorer):
        scorer.fit()
        scorer.measure_distance(method='CB')
        scores = np.random.rand(scorer.query_pdb_mapper.seq_aln.seq_length, scorer.query_pdb_mapper.seq_aln.seq_length)
        scores[np.tril_indices(scorer.query_pdb_mapper.seq_aln.seq_length, 1)] = 0
        scores += scores.T
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 2, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        expected_df_a = identify_expected_scores_and_distances(scorer=scorer, scores=scores, coverages=coverages,
                                                               ranks=ranks, distances=scorer.distances, category='Any')
        fpr_expected1a, tpr_expected1a, _ = roc_curve(expected_df_a['Distance'] <= scorer.cutoff,
                                                      1.0 - expected_df_a['Coverage'], pos_label=True)
        auroc_expected1a = auc(fpr_expected1a, tpr_expected1a)
        tpr1a, fpr1a, auroc1a = scorer.score_auc(category='Any')
        self.assertEqual(np.sum(fpr_expected1a - fpr1a), 0)
        self.assertEqual(np.sum(tpr_expected1a - tpr1a), 0)
        self.assertEqual(auroc_expected1a, auroc1a)

    def test_score_auc_seq1(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        self.evaluate_score_auc(scorer=scorer)

    def test_score_auc_seq1_alt_loc_pdb(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=14.0, chain='A')
        self.evaluate_score_auc(scorer=scorer)

    def test_score_auc_seq2(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_auc(scorer=scorer)

    def test_score_auc_seq3(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_auc(scorer=scorer)


class TestContactScorerPlotAUC(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(cls.aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        cls.aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str_long)
        cls.aln1 = SeqAlignment(query_id='seq1', file_name=cls.aln_fn)
        cls.aln1.import_alignment()
        cls.aln1 = cls.aln1.remove_gaps()
        cls.aln2 = SeqAlignment(query_id='seq2', file_name=cls.aln_fn)
        cls.aln2.import_alignment()
        cls.aln2 = cls.aln2.remove_gaps()
        cls.aln3 = SeqAlignment(query_id='seq3', file_name=cls.aln_fn)
        cls.aln3.import_alignment()
        cls.aln3 = cls.aln3.remove_gaps()

    def evaluate_plot_auc(self, scorer, file_name, expected_file_name, output_dir):
        scorer.fit()
        scorer.measure_distance(method='CB')
        scores = np.random.rand(scorer.query_pdb_mapper.seq_aln.seq_length, scorer.query_pdb_mapper.seq_aln.seq_length)
        scores[np.tril_indices(scorer.query_pdb_mapper.seq_aln.seq_length, 1)] = 0
        scores += scores.T
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 2, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        tpr, fpr, auroc = scorer.score_auc(category='Any')
        scorer.plot_auc(auc_data=(tpr, fpr, auroc), title=None, file_name=file_name, output_dir=output_dir)
        self.assertTrue(os.path.isfile(expected_file_name))

    def test_no_fn_no_dir(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_roc.png'
        self.evaluate_plot_auc(scorer=scorer, file_name=None, expected_file_name=expected_file_name, output_dir=None)
        os.remove(expected_file_name)

    def test_fn_no_png_no_dir(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_roc.png'
        self.evaluate_plot_auc(scorer=scorer, file_name='seq3_Cutoff20.0A_roc', expected_file_name=expected_file_name,
                               output_dir=None)
        os.remove(expected_file_name)

    def test_fn_no_dir(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_roc.png'
        self.evaluate_plot_auc(scorer=scorer, file_name=expected_file_name, expected_file_name=expected_file_name,
                               output_dir=None)
        os.remove(expected_file_name)

    def test_no_fn_dir(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_roc.png'
        new_dir = './Plots/'
        os.makedirs(new_dir)
        self.evaluate_plot_auc(scorer=scorer, file_name=None, output_dir=new_dir,
                               expected_file_name=os.path.join(new_dir, expected_file_name))
        rmtree(new_dir)

    def test_fn_no_png_dir(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_roc.png'
        new_dir = './Plots/'
        os.makedirs(new_dir)
        self.evaluate_plot_auc(scorer=scorer, file_name='seq3_Cutoff20.0A_roc', output_dir=new_dir,
                               expected_file_name=os.path.join(new_dir, expected_file_name))
        rmtree(new_dir)

    def test_fn_dir(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_roc.png'
        new_dir = './Plots/'
        os.makedirs(new_dir)
        self.evaluate_plot_auc(scorer=scorer, file_name=expected_file_name, output_dir=new_dir,
                               expected_file_name=os.path.join(new_dir, expected_file_name))
        rmtree(new_dir)
        
        
class TestContactScorerScorePrecisionAndRecall(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(cls.aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        cls.aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str_long)
        cls.aln1 = SeqAlignment(query_id='seq1', file_name=cls.aln_fn)
        cls.aln1.import_alignment()
        cls.aln1 = cls.aln1.remove_gaps()
        cls.aln2 = SeqAlignment(query_id='seq2', file_name=cls.aln_fn)
        cls.aln2.import_alignment()
        cls.aln2 = cls.aln2.remove_gaps()
        cls.aln3 = SeqAlignment(query_id='seq3', file_name=cls.aln_fn)
        cls.aln3.import_alignment()
        cls.aln3 = cls.aln3.remove_gaps()

    def evaluate_score_precision_and_recall(self, scorer):
        scorer.fit()
        scorer.measure_distance(method='CB')
        scores = np.random.rand(scorer.query_pdb_mapper.seq_aln.seq_length, scorer.query_pdb_mapper.seq_aln.seq_length)
        scores[np.tril_indices(scorer.query_pdb_mapper.seq_aln.seq_length, 1)] = 0
        scores += scores.T
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 2, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        expected_df_a = identify_expected_scores_and_distances(scorer=scorer, scores=scores, coverages=coverages,
                                                               ranks=ranks, distances=scorer.distances, category='Any')
        precision_expected, recall_expected, _ = precision_recall_curve(expected_df_a['Distance'] <= scorer.cutoff,
                                                                        1.0 - expected_df_a['Coverage'], pos_label=True)
        recall_expected, precision_expected = zip(*sorted(zip(recall_expected, precision_expected)))
        auprc_expected = auc(recall_expected, precision_expected)
        precision, recall, auprc = scorer.score_precision_recall(category='Any')
        self.assertEqual(np.sum(precision_expected - precision), 0)
        self.assertEqual(np.sum(recall_expected - recall), 0)
        self.assertEqual(auprc_expected, auprc)

    def test_score_precision_recall_seq1(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        self.evaluate_score_precision_and_recall(scorer=scorer)

    def test_score_precision_recall_seq1_alt_loc_pdb(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=14.0, chain='A')
        self.evaluate_score_precision_and_recall(scorer=scorer)

    def test_score_precision_recall_seq2(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_precision_and_recall(scorer=scorer)

    def test_score_auc_seq3(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_precision_and_recall(scorer=scorer)


class TestContactScorerPlotAUPRC(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(cls.aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        cls.aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str_long)
        cls.aln1 = SeqAlignment(query_id='seq1', file_name=cls.aln_fn)
        cls.aln1.import_alignment()
        cls.aln1 = cls.aln1.remove_gaps()
        cls.aln2 = SeqAlignment(query_id='seq2', file_name=cls.aln_fn)
        cls.aln2.import_alignment()
        cls.aln2 = cls.aln2.remove_gaps()
        cls.aln3 = SeqAlignment(query_id='seq3', file_name=cls.aln_fn)
        cls.aln3.import_alignment()
        cls.aln3 = cls.aln3.remove_gaps()

    def evaluate_plot_auprc(self, scorer, file_name, expected_file_name, output_dir):
        scorer.fit()
        scorer.measure_distance(method='CB')
        scores = np.random.rand(scorer.query_pdb_mapper.seq_aln.seq_length, scorer.query_pdb_mapper.seq_aln.seq_length)
        scores[np.tril_indices(scorer.query_pdb_mapper.seq_aln.seq_length, 1)] = 0
        scores += scores.T
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 2, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        precision, recall, auprc = scorer.score_precision_recall(category='Any')
        scorer.plot_auprc(auprc_data=(precision, recall, auprc), title=None, file_name=file_name, output_dir=output_dir)
        self.assertTrue(os.path.isfile(expected_file_name))

    def test_no_fn_no_dir(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_auprc.png'
        self.evaluate_plot_auprc(scorer=scorer, file_name=None, expected_file_name=expected_file_name, output_dir=None)
        os.remove(expected_file_name)

    def test_fn_no_png_no_dir(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_auprc.png'
        self.evaluate_plot_auprc(scorer=scorer, file_name='seq3_Cutoff20.0A_auprc',
                                 expected_file_name=expected_file_name, output_dir=None)
        os.remove(expected_file_name)

    def test_fn_no_dir(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_auprc.png'
        self.evaluate_plot_auprc(scorer=scorer, file_name=expected_file_name, expected_file_name=expected_file_name,
                                 output_dir=None)
        os.remove(expected_file_name)

    def test_no_fn_dir(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_auprc.png'
        new_dir = './Plots/'
        os.makedirs(new_dir)
        self.evaluate_plot_auprc(scorer=scorer, file_name=None, output_dir=new_dir,
                                 expected_file_name=os.path.join(new_dir, expected_file_name))
        rmtree(new_dir)

    def test_fn_no_png_dir(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_auprc.png'
        new_dir = './Plots/'
        os.makedirs(new_dir)
        self.evaluate_plot_auprc(scorer=scorer, file_name='seq3_Cutoff20.0A_auprc', output_dir=new_dir,
                                 expected_file_name=os.path.join(new_dir, expected_file_name))
        rmtree(new_dir)

    def test_fn_dir(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        expected_file_name = 'seq3_Cutoff20.0A_auprc.png'
        new_dir = './Plots/'
        os.makedirs(new_dir)
        self.evaluate_plot_auprc(scorer=scorer, file_name=expected_file_name, output_dir=new_dir,
                                 expected_file_name=os.path.join(new_dir, expected_file_name))
        rmtree(new_dir)


class TestContactScorerScorePrecision(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(cls.aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        cls.aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str_long)
        cls.aln1 = SeqAlignment(query_id='seq1', file_name=cls.aln_fn)
        cls.aln1.import_alignment()
        cls.aln1 = cls.aln1.remove_gaps()
        cls.aln2 = SeqAlignment(query_id='seq2', file_name=cls.aln_fn)
        cls.aln2.import_alignment()
        cls.aln2 = cls.aln2.remove_gaps()
        cls.aln3 = SeqAlignment(query_id='seq3', file_name=cls.aln_fn)
        cls.aln3.import_alignment()
        cls.aln3 = cls.aln3.remove_gaps()

    def evaluate_score_precision(self, scorer, k=None, n=None):
        scorer.fit()
        scorer.measure_distance(method='CB')
        scores = np.random.rand(scorer.query_pdb_mapper.seq_aln.seq_length, scorer.query_pdb_mapper.seq_aln.seq_length)
        scores[np.tril_indices(scorer.query_pdb_mapper.seq_aln.seq_length, 1)] = 0
        scores += scores.T
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 2, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        expected_df_a = identify_expected_scores_and_distances(scorer=scorer, scores=scores, coverages=coverages, n=n,
                                                               k=k, ranks=ranks, distances=scorer.distances,
                                                               category='Any')
        precision_expected = precision_score(expected_df_a['Distance'] <= scorer.cutoff,
                                             expected_df_a['Coverage'] <= 0.5, pos_label=True)
        precision = scorer.score_precision(category='Any', n=n, k=k)
        self.assertEqual(np.sum(precision_expected - precision), 0)

    def test_score_precision_seq1(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        self.evaluate_score_precision(scorer=scorer)

    def test_score_precision_seq1_n(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        self.evaluate_score_precision(scorer=scorer, n=2)

    def test_score_precision_seq1_k(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        self.evaluate_score_precision(scorer=scorer, k=2)

    def test_score_precision_seq1_alt_loc_pdb(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=14.0, chain='A')
        self.evaluate_score_precision(scorer=scorer)

    def test_score_precision_seq1_alt_loc_pdb_n(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=14.0, chain='A')
        self.evaluate_score_precision(scorer=scorer, n=2)

    def test_score_precision_seq1_alt_loc_pdb_k(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=14.0, chain='A')
        self.evaluate_score_precision(scorer=scorer, k=2)

    def test_score_precision_seq2(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_precision(scorer=scorer)

    def test_score_precision_seq2_n(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_precision(scorer=scorer, n=2)

    def test_score_precision_seq2_k(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_precision(scorer=scorer, k=2)

    def test_score_precision_seq3(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_precision(scorer=scorer)

    def test_score_precision_seq3_n(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_precision(scorer=scorer, n=2)

    def test_score_precision_seq3_k(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_precision(scorer=scorer, k=2)


class TestContactScorerScoreRecall(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(cls.aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        cls.aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str_long)
        cls.aln1 = SeqAlignment(query_id='seq1', file_name=cls.aln_fn)
        cls.aln1.import_alignment()
        cls.aln1 = cls.aln1.remove_gaps()
        cls.aln2 = SeqAlignment(query_id='seq2', file_name=cls.aln_fn)
        cls.aln2.import_alignment()
        cls.aln2 = cls.aln2.remove_gaps()
        cls.aln3 = SeqAlignment(query_id='seq3', file_name=cls.aln_fn)
        cls.aln3.import_alignment()
        cls.aln3 = cls.aln3.remove_gaps()

    def evaluate_score_recall(self, scorer, k=None, n=None):
        scorer.fit()
        scorer.measure_distance(method='CB')
        scores = np.random.rand(scorer.query_pdb_mapper.seq_aln.seq_length, scorer.query_pdb_mapper.seq_aln.seq_length)
        scores[np.tril_indices(scorer.query_pdb_mapper.seq_aln.seq_length, 1)] = 0
        scores += scores.T
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 2, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        expected_df_a = identify_expected_scores_and_distances(scorer=scorer, scores=scores, coverages=coverages, n=n,
                                                               k=k, ranks=ranks, distances=scorer.distances,
                                                               category='Any')
        precision_expected = recall_score(expected_df_a['Distance'] <= scorer.cutoff,
                                          expected_df_a['Coverage'] <= 0.5, pos_label=True)
        precision = scorer.score_recall(category='Any', n=n, k=k)
        self.assertEqual(np.sum(precision_expected - precision), 0)

    def test_score_recall_seq1(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        self.evaluate_score_recall(scorer=scorer)

    def test_score_recall_seq1_n(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        self.evaluate_score_recall(scorer=scorer, n=2)

    def test_score_recall_seq1_k(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        self.evaluate_score_recall(scorer=scorer, k=2)

    def test_score_recall_seq1_alt_loc_pdb(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=14.0, chain='A')
        self.evaluate_score_recall(scorer=scorer)

    def test_score_recall_seq1_alt_loc_pdb_n(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=14.0, chain='A')
        self.evaluate_score_recall(scorer=scorer, n=2)

    def test_score_recall_seq1_alt_loc_pdb_k(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=14.0, chain='A')
        self.evaluate_score_recall(scorer=scorer, k=2)

    def test_score_recall_seq2(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_recall(scorer=scorer)

    def test_score_recall_seq2_n(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_recall(scorer=scorer, n=2)

    def test_score_recall_seq2_k(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_recall(scorer=scorer, k=2)

    def test_score_recall_seq3(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_recall(scorer=scorer)

    def test_score_recall_seq3_n(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_recall(scorer=scorer, n=2)

    def test_score_recall_seq3_k(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_recall(scorer=scorer, k=2)


class TestContactScorerScoreF1(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(cls.aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        cls.aln_fn = write_out_temp_fn(suffix='fasta', out_str=pro_str_long)
        cls.aln1 = SeqAlignment(query_id='seq1', file_name=cls.aln_fn)
        cls.aln1.import_alignment()
        cls.aln1 = cls.aln1.remove_gaps()
        cls.aln2 = SeqAlignment(query_id='seq2', file_name=cls.aln_fn)
        cls.aln2.import_alignment()
        cls.aln2 = cls.aln2.remove_gaps()
        cls.aln3 = SeqAlignment(query_id='seq3', file_name=cls.aln_fn)
        cls.aln3.import_alignment()
        cls.aln3 = cls.aln3.remove_gaps()

    def evaluate_score_f1(self, scorer, k=None, n=None):
        scorer.fit()
        scorer.measure_distance(method='CB')
        scores = np.random.rand(scorer.query_pdb_mapper.seq_aln.seq_length, scorer.query_pdb_mapper.seq_aln.seq_length)
        scores[np.tril_indices(scorer.query_pdb_mapper.seq_aln.seq_length, 1)] = 0
        scores += scores.T
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 2, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        expected_df_a = identify_expected_scores_and_distances(scorer=scorer, scores=scores, coverages=coverages, n=n,
                                                               k=k, ranks=ranks, distances=scorer.distances,
                                                               category='Any')
        precision_expected = f1_score(expected_df_a['Distance'] <= scorer.cutoff, expected_df_a['Coverage'] <= 0.5,
                                      pos_label=True)
        precision = scorer.score_f1(category='Any', n=n, k=k)
        self.assertEqual(np.sum(precision_expected - precision), 0)

    def test_score_f1_seq1(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        self.evaluate_score_f1(scorer=scorer)

    def test_score_f1_seq1_n(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        self.evaluate_score_f1(scorer=scorer, n=2)

    def test_score_f1_seq1_k(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        self.evaluate_score_f1(scorer=scorer, k=2)

    def test_score_f1_alt_loc_pdb(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=14.0, chain='A')
        self.evaluate_score_f1(scorer=scorer)

    def test_score_f1_seq1_alt_loc_pdb_n(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=14.0, chain='A')
        self.evaluate_score_f1(scorer=scorer, n=2)

    def test_score_f1_seq1_alt_loc_pdb_k(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=14.0, chain='A')
        self.evaluate_score_f1(scorer=scorer, k=2)

    def test_score_f1_seq2(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_f1(scorer=scorer)

    def test_score_f1_seq2_n(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_f1(scorer=scorer, n=2)

    def test_score_f1_k(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_f1(scorer=scorer, k=2)

    def test_score_f1_seq3(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_f1(scorer=scorer)

    def test_score_f1_seq3_n(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_f1(scorer=scorer, n=2)

    def test_score_f1_seq3_k(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        self.evaluate_score_f1(scorer=scorer, k=2)


class TestContactScorerWriteOutCovariationAndStructuralData(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        protein_aln1.write_out_alignment(aln_fn)

    def evaluate_write_out_covariation_and_structural_data(self, scorer, filename, output_dir, expected_path):
        self.assertFalse(os.path.isfile(expected_path))
        scorer.fit()
        scorer.measure_distance(method='CB')
        scores = np.random.rand(scorer.query_pdb_mapper.seq_aln.seq_length, scorer.query_pdb_mapper.seq_aln.seq_length)
        scores[np.tril_indices(scorer.query_pdb_mapper.seq_aln.seq_length, 1)] = 0
        scores += scores.T
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 2, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        scorer.write_out_covariation_and_structural_data(file_name=filename, output_dir=output_dir)
        self.assertTrue(os.path.isfile(expected_path))
        df = pd.read_csv(expected_path, sep='\t', header=0, index_col=None)
        self.assertEqual(len(df), len(scorer.data))
        self.assertTrue(all(df.columns == ['Pos1', '(AA1)', 'Pos2', '(AA2)', 'Raw_Score', 'Coverage_Score', 'PDB Pos1',
                                           '(PDB AA1)', 'PDB Pos2', '(PDB AA2)', 'Residue_Dist', 'Within_Threshold']))
        self.assertTrue(all(scorer.data['Seq Pos 1'] == df['Pos1']))
        self.assertTrue(all(scorer.data['Seq AA 1'] == df['(AA1)']))
        self.assertTrue(all(scorer.data['Seq Pos 2'] == df['Pos2']))
        self.assertTrue(all(scorer.data['Seq AA 2'] == df['(AA2)']))
        self.assertTrue(all(np.abs(scorer.data['Score'] - df['Raw_Score']) < 1E-6))
        self.assertTrue(all(np.abs(scorer.data['Coverage'] - df['Coverage_Score']) < 1E-6))
        self.assertTrue(all(scorer.data['Struct Pos 1'] == df['PDB Pos1']))
        self.assertTrue(all(scorer.data['Struct AA 1'] == df['(PDB AA1)']))
        self.assertTrue(all(scorer.data['Struct Pos 2'] == df['PDB Pos2']))
        self.assertTrue(all(scorer.data['Struct AA 2'] == df['(PDB AA2)']))
        self.assertTrue(all(np.abs(scorer.data['Distance'] - df['Residue_Dist']) < 1E-6))
        self.assertTrue(all(scorer.data[f'Contact (within {scorer.cutoff}A cutoff)'] == df['Within_Threshold']))

    def test_seq1_no_filename_no_dir(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=8.0, chain='A')
        expected_path = f"{datetime.datetime.now().strftime('%m_%d_%Y')}_{scorer.query_pdb_mapper.query}" \
                        ".Covariance_vs_Structure.tsv"
        self.evaluate_write_out_covariation_and_structural_data(scorer=scorer, filename=None, output_dir=None,
                                                                expected_path=expected_path)
        os.remove(expected_path)

    def test_seq1_filename_no_dir(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=8.0, chain='A')
        expected_path = "Covariance_vs_Structure.tsv"
        self.evaluate_write_out_covariation_and_structural_data(scorer=scorer, filename=expected_path, output_dir=None,
                                                                expected_path=expected_path)
        os.remove(expected_path)

    def test_seq1_no_filename_dir(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=8.0, chain='A')
        new_dir = 'Output'
        os.mkdir(new_dir)
        expected_path = f"{datetime.datetime.now().strftime('%m_%d_%Y')}_{scorer.query_pdb_mapper.query}" \
                        ".Covariance_vs_Structure.tsv"
        self.evaluate_write_out_covariation_and_structural_data(scorer=scorer, filename=None, output_dir=new_dir,
                                                                expected_path=os.path.join(new_dir, expected_path))
        rmtree(new_dir)

    def test_seq1_filename_dir(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=8.0, chain='A')
        new_dir = 'Output'
        os.mkdir(new_dir)
        expected_path = "Covariance_vs_Structure.tsv"
        self.evaluate_write_out_covariation_and_structural_data(scorer=scorer, filename=expected_path,
                                                                output_dir=new_dir,
                                                                expected_path=os.path.join(new_dir, expected_path))
        rmtree(new_dir)

    def test_seq1_alt_loc_pdb(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=8.0, chain='A')
        expected_path = f"{datetime.datetime.now().strftime('%m_%d_%Y')}_{scorer.query_pdb_mapper.query}" \
                        ".Covariance_vs_Structure.tsv"
        self.evaluate_write_out_covariation_and_structural_data(scorer=scorer, filename=None, output_dir=None,
                                                                expected_path=expected_path)
        os.remove(expected_path)

    def test_seq2(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=8.0, chain='B')
        expected_path = f"{datetime.datetime.now().strftime('%m_%d_%Y')}_{scorer.query_pdb_mapper.query}" \
                        ".Covariance_vs_Structure.tsv"
        self.evaluate_write_out_covariation_and_structural_data(scorer=scorer, filename=None, output_dir=None,
                                                                expected_path=expected_path)
        os.remove(expected_path)

    def test_seq3(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=8.0, chain='B')
        expected_path = f"{datetime.datetime.now().strftime('%m_%d_%Y')}_{scorer.query_pdb_mapper.query}" \
                        ".Covariance_vs_Structure.tsv"
        self.evaluate_write_out_covariation_and_structural_data(scorer=scorer, filename=None, output_dir=None,
                                                                expected_path=expected_path)
        os.remove(expected_path)


class TestHeatmapPlot(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scores = np.random.rand(50, 50)
        cls.scores[np.tril_indices(50, 1)] = 0
        cls.scores += cls.scores.T

    def evaluate_heatmap_plot(self, name, output_dir, expected_path):
        heatmap_plot(name=name, data_mat=self.scores, output_dir=output_dir)
        self.assertTrue(os.path.isfile(expected_path))
        os.remove(expected_path)

    def test_fail_no_data(self):
        with self.assertRaises(TypeError):
            heatmap_plot(name='Test', data_mat=None, output_dir=None)

    def test_fail_no_fn(self):
        with self.assertRaises(AttributeError):
            self.evaluate_heatmap_plot(name=None, output_dir=None, expected_path=None)

    def test_fn_no_dir(self):
        expected_path = 'First_Test.png'
        self.evaluate_heatmap_plot(name='First Test', output_dir=None, expected_path=expected_path)

    def test_fn_dir(self):
        expected_path = 'First_Test.png'
        new_dir = 'Output'
        os.makedirs(new_dir)
        self.evaluate_heatmap_plot(name='First Test', output_dir=new_dir,
                                   expected_path=os.path.join(new_dir, expected_path))
        rmtree(new_dir)


class TestSurfacePlot(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scores = np.random.rand(50, 50)
        cls.scores[np.tril_indices(50, 1)] = 0
        cls.scores += cls.scores.T

    def evaluate_surface_plot(self, name, output_dir, expected_path):
        surface_plot(name=name, data_mat=self.scores, output_dir=output_dir)
        self.assertTrue(os.path.isfile(expected_path))
        os.remove(expected_path)

    def test_fail_no_data(self):
        with self.assertRaises(TypeError):
            surface_plot(name='Test', data_mat=None, output_dir=None)

    def test_fail_no_fn(self):
        with self.assertRaises(AttributeError):
            self.evaluate_surface_plot(name=None, output_dir=None, expected_path=None)

    def test_fn_no_dir(self):
        expected_path = 'First_Test.png'
        self.evaluate_surface_plot(name='First Test', output_dir=None, expected_path=expected_path)

    def test_fn_dir(self):
        expected_path = 'First_Test.png'
        new_dir = 'Output'
        os.makedirs(new_dir)
        self.evaluate_surface_plot(name='First Test', output_dir=new_dir,
                                   expected_path=os.path.join(new_dir, expected_path))
        rmtree(new_dir)


class TestContactScorerEvaluatePredictions(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        protein_aln1.write_out_alignment(aln_fn)

    def evaluate_evaluate_predictions(self, scorer, plot, output_dir, processes):
        seq_len = scorer.query_pdb_mapper.seq_aln.seq_length
        scores = np.random.RandomState(1234567890).rand(seq_len, seq_len)
        scores[np.tril_indices(seq_len, 1)] = 0
        scores += scores.T
        ranks, coverages = compute_rank_and_coverage(seq_len, scores, 2, 'min')
        prev_b_w2_ave = None
        prev_u_w2_ave = None
        chain_len = len(scorer.query_pdb_mapper.pdb_ref.seq[scorer.query_pdb_mapper.best_chain])
        for v in range(1, 3 if chain_len <= 3 else 4):
            curr_stats, curr_b_w2_ave, curr_u_w2_ave = scorer.evaluate_predictions(
                verbosity=v, out_dir=output_dir, scores=scores, coverages=coverages, ranks=ranks, dist='CB',
                file_prefix='SCORER_TEST', biased_w2_ave=prev_b_w2_ave, unbiased_w2_ave=prev_u_w2_ave,
                processes=processes, threshold=0.5, plots=plot)
            curr_stats = pd.DataFrame(curr_stats)
            # Tests
            # Check that the correct data is in the dataframe according to the verbosity
            column_length = None
            for key in curr_stats:
                if column_length is None:
                    column_length = len(curr_stats[key])
                else:
                    self.assertEqual(len(curr_stats[key]), column_length)
            if v >= 1:
                self.assertTrue('Distance' in curr_stats)
                self.assertTrue('Sequence_Separation' in curr_stats)
                self.assertTrue('AUROC' in curr_stats)
                self.assertTrue('AUPRC' in curr_stats)
                for sep in ['Any', 'Neighbors', 'Short', 'Medium', 'Long']:
                    current_auroc = curr_stats.loc[curr_stats['Sequence_Separation'] == sep, 'AUROC'].unique()[0]
                    if current_auroc == 'N/A':
                        with self.assertRaises(IndexError):
                            scorer.score_auc(category=sep)
                    else:
                        _, _, expected_auroc = scorer.score_auc(category=sep)
                        if np.isnan(current_auroc):
                            self.assertTrue(np.isnan(expected_auroc))
                        else:
                            self.assertEqual(current_auroc, expected_auroc)
                    current_auprc = curr_stats.loc[curr_stats['Sequence_Separation'] == sep, 'AUPRC'].unique()[0]
                    if current_auroc == 'N/A':
                        with self.assertRaises(IndexError):
                            scorer.score_precision_recall(category=sep)
                    else:
                        _, _, expected_auprc = scorer.score_precision_recall(category=sep)
                        if np.isnan(current_auprc):
                            self.assertTrue(np.isnan(expected_auprc))
                        else:
                            self.assertEqual(current_auprc, expected_auprc)
                    if plot:
                        fn1 = os.path.join(output_dir, f'SCORER_TESTAUROC_Evaluation_Dist-CB_Separation-{sep}.png')
                        if current_auroc == 'N/A':
                            self.assertFalse(os.path.isfile(fn1))
                        else:
                            self.assertTrue(os.path.isfile(fn1))
                            os.remove(fn1)
                        fn2 = os.path.join(output_dir, f'SCORER_TESTAUPRC_Evaluation_Dist-CB_Separation-{sep}.png')
                        if current_auprc == 'N/A':
                            self.assertFalse(os.path.isfile(fn2))
                        else:
                            self.assertTrue(os.path.isfile(fn2))
                            os.remove(fn2)
                if v == 1:
                    self.assertTrue(curr_b_w2_ave is None)
                    self.assertTrue(curr_u_w2_ave is None)
            if v >= 2:
                self.assertTrue('Top K Predictions' in curr_stats)
                self.assertTrue('Precision' in curr_stats)
                self.assertTrue('Recall' in curr_stats)
                self.assertTrue('F1 Score' in curr_stats)
                for sep in ['Any', 'Neighbors', 'Short', 'Medium', 'Long']:
                    for k in range(1, 11):
                        label = 'L' if k == 1 else f'L/{k}'
                        self.assertTrue(any(curr_stats['Top K Predictions'] == label))
                        current_precision = curr_stats.loc[(curr_stats['Sequence_Separation'] == sep) &
                                                           (curr_stats['Top K Predictions'] == label),
                                                           'Precision'].unique()[0]
                        expected_precision = scorer.score_precision(k=k, category=sep)
                        self.assertEqual(current_precision, expected_precision)
                        current_recall = curr_stats.loc[(curr_stats['Sequence_Separation'] == sep) &
                                                        (curr_stats['Top K Predictions'] == label),
                                                        'Recall'].unique()[0]
                        expected_recall = scorer.score_recall(k=k, category=sep)
                        self.assertEqual(current_recall, expected_recall)
                        current_f1 = curr_stats.loc[(curr_stats['Sequence_Separation'] == sep) &
                                                    (curr_stats['Top K Predictions'] == label),
                                                    'F1 Score'].unique()[0]
                        expected_f1 = scorer.score_f1(k=k, category=sep)
                        self.assertEqual(current_f1, expected_f1)
                if v == 2:
                    self.assertTrue(curr_b_w2_ave is None)
                    self.assertTrue(curr_u_w2_ave is None)
            if v >= 3:
                self.assertTrue('Max Biased Z-Score' in curr_stats)
                self.assertTrue('AUC Biased Z-Score' in curr_stats)
                biased_df, biased_reusable, biased_auc_scw_z_curve = scorer.score_clustering_of_contact_predictions(
                    bias=True, file_path='test.tsv')
                os.remove('test.tsv')
                self.assertEqual(curr_stats['AUC Biased Z-Score'].unique()[0], biased_auc_scw_z_curve)
                if biased_df['Z-Score'].max() == 'NA':
                    self.assertTrue(np.isnan(curr_stats['Max Biased Z-Score'].unique()[0]))
                else:
                    self.assertEqual(curr_stats['Max Biased Z-Score'].unique()[0], biased_df['Z-Score'].max())
                self.assertEqual(curr_b_w2_ave, biased_reusable)
                self.assertTrue('Max Unbiased Z-Score' in curr_stats)
                self.assertTrue('AUC Unbiased Z-Score' in curr_stats)
                unbiased_df, unbiased_reusable, unbiased_auc_scw_z_curve = scorer.score_clustering_of_contact_predictions(
                    bias=False, file_path='test.tsv')
                os.remove('test.tsv')
                self.assertEqual(curr_stats['AUC Unbiased Z-Score'].unique()[0], unbiased_auc_scw_z_curve)
                if unbiased_df['Z-Score'].max() == 'NA':
                    self.assertTrue(np.isnan(curr_stats['Max Unbiased Z-Score'].unique()[0]))
                else:
                    self.assertEqual(curr_stats['Max Unbiased Z-Score'].unique()[0], unbiased_df['Z-Score'].max())
                self.assertEqual(curr_u_w2_ave, unbiased_reusable)
                fn4 = os.path.join(output_dir, 'SCORER_TEST' + 'Dist-CB_Biased_ZScores.tsv')
                self.assertTrue(os.path.isfile(fn4))
                os.remove(fn4)
                fn5 = os.path.join(output_dir, 'SCORER_TEST' + 'Dist-CB_Unbiased_ZScores.tsv')
                self.assertTrue(os.path.isfile(fn5))
                os.remove(fn5)
                if plot:
                    fn6 = os.path.join(output_dir, 'SCORER_TEST' + 'Dist-CB_Biased_ZScores.png')
                    self.assertTrue(os.path.isfile(fn6))
                    os.remove(fn6)
                    fn7 = os.path.join(output_dir, 'SCORER_TEST' + 'Dist-CB_Unbiased_ZScores.png')
                    self.assertTrue(os.path.isfile(fn7))
                    os.remove(fn7)
                self.assertTrue(curr_b_w2_ave is not None)
                self.assertTrue(curr_u_w2_ave is not None)
            # Update
            prev_b_w2_ave = curr_b_w2_ave
            prev_u_w2_ave = curr_u_w2_ave

    def test_seq1_no_plots_one_process(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        new_dir = 'Test_Dir'
        os.mkdir(new_dir)
        self.evaluate_evaluate_predictions(scorer=scorer, output_dir=new_dir, processes=1, plot=False)
        rmtree(new_dir)

    def test_seq1_plots_one_process(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        new_dir = 'Test_Dir'
        os.mkdir(new_dir)
        self.evaluate_evaluate_predictions(scorer=scorer, output_dir=new_dir, processes=1, plot=True)
        rmtree(new_dir)

    def test_seq1_plots_multi_process(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        new_dir = 'Test_Dir'
        os.mkdir(new_dir)
        self.evaluate_evaluate_predictions(scorer=scorer, output_dir=new_dir, processes=2, plot=True)
        rmtree(new_dir)

    def test_seq1_alt_loc_pdb(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=14.0, chain='A')
        new_dir = 'Test_Dir'
        os.mkdir(new_dir)
        self.evaluate_evaluate_predictions(scorer=scorer, output_dir=new_dir, processes=2, plot=True)
        rmtree(new_dir)

    def test_seq2(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        new_dir = 'Test_Dir'
        os.mkdir(new_dir)
        self.evaluate_evaluate_predictions(scorer=scorer, output_dir=new_dir, processes=2, plot=True)
        rmtree(new_dir)

    def test_seq3(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        new_dir = 'Test_Dir'
        os.mkdir(new_dir)
        self.evaluate_evaluate_predictions(scorer=scorer, output_dir=new_dir, processes=2, plot=True)
        rmtree(new_dir)


class TestContactScorerEvaluatePredictor(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        protein_aln1.write_out_alignment(aln_fn)

    def evaluate_evaluate_predictor(self, scorer, plot, output_dir, processes):
        seq_len = scorer.query_pdb_mapper.seq_aln.seq_length
        scores = np.random.RandomState(1234567890).rand(seq_len, seq_len)
        scores[np.tril_indices(seq_len, 1)] = 0
        scores += scores.T
        ranks, coverages = compute_rank_and_coverage(seq_len, scores, 2, 'min')
        predictor = EvolutionaryTrace(query=scorer.query_pdb_mapper.query, polymer_type='Protein',
                                      aln_file=scorer.query_pdb_mapper.seq_aln.file_name, et_distance=True,
                                      distance_model='blosum62', tree_building_method='et', tree_building_options={},
                                      ranks=None, position_type='pair', scoring_metric='mismatch_diversity',
                                      gap_correction=None, out_dir=output_dir, processors=1, low_memory=True,
                                      output_files={'original_aln', 'non_gap_aln', 'tree', 'scores'})
        predictor.rankings = ranks
        predictor.scores = scores
        predictor.coverages = coverages
        prev_b_w2_ave = None
        prev_u_w2_ave = None
        chain_len = len(scorer.query_pdb_mapper.pdb_ref.seq[scorer.query_pdb_mapper.best_chain])
        for v in range(1, 3 if chain_len <= 3 else 4):
            curr_stats, curr_b_w2_ave, curr_u_w2_ave = scorer.evaluate_predictor(
                predictor=predictor, verbosity=v, out_dir=output_dir, dist='CB', biased_w2_ave=prev_b_w2_ave,
                unbiased_w2_ave=prev_u_w2_ave, processes=processes, threshold=0.5, file_prefix='SCORER_TEST',
                plots=plot)
            expected_res_fn = os.path.join(output_dir, 'SCORER_TEST_Evaluation_Dist-CB.txt')
            self.assertTrue(os.path.isfile(expected_res_fn))
            expected_read_in_df = curr_stats.fillna('N/A')
            read_in_df = pd.read_csv(expected_res_fn, header=0, index_col=None, sep='\t')
            read_in_df.fillna('N/A', inplace=True)
            self.assertEqual(len(set(expected_read_in_df.columns).intersection(read_in_df.columns)), len(curr_stats.columns))
            for col in curr_stats.columns:
                self.assertTrue(all(expected_read_in_df[col] == read_in_df[col]))
            os.remove(expected_res_fn)
            # Tests
            # Check that the correct data is in the dataframe according to the verbosity
            column_length = None
            for key in curr_stats:
                if column_length is None:
                    column_length = len(curr_stats[key])
                else:
                    self.assertEqual(len(curr_stats[key]), column_length)
            if v >= 1:
                self.assertTrue('Distance' in curr_stats)
                self.assertTrue('Sequence_Separation' in curr_stats)
                self.assertTrue('AUROC' in curr_stats)
                self.assertTrue('AUPRC' in curr_stats)
                for sep in ['Any', 'Neighbors', 'Short', 'Medium', 'Long']:
                    current_auroc = curr_stats.loc[curr_stats['Sequence_Separation'] == sep, 'AUROC'].unique()[0]
                    if current_auroc == 'N/A':
                        with self.assertRaises(IndexError):
                            scorer.score_auc(category=sep)
                    else:
                        _, _, expected_auroc = scorer.score_auc(category=sep)
                        if np.isnan(current_auroc):
                            self.assertTrue(np.isnan(expected_auroc))
                        else:
                            self.assertEqual(current_auroc, expected_auroc)
                    current_auprc = curr_stats.loc[curr_stats['Sequence_Separation'] == sep, 'AUPRC'].unique()[0]
                    if current_auroc == 'N/A':
                        with self.assertRaises(IndexError):
                            scorer.score_precision_recall(category=sep)
                    else:
                        _, _, expected_auprc = scorer.score_precision_recall(category=sep)
                        if np.isnan(current_auprc):
                            self.assertTrue(np.isnan(expected_auprc))
                        else:
                            self.assertEqual(current_auprc, expected_auprc)
                    if plot:
                        fn1 = os.path.join(output_dir, f'SCORER_TESTAUROC_Evaluation_Dist-CB_Separation-{sep}.png')
                        if current_auroc == 'N/A':
                            self.assertFalse(os.path.isfile(fn1))
                        else:
                            self.assertTrue(os.path.isfile(fn1))
                            os.remove(fn1)
                        fn2 = os.path.join(output_dir, f'SCORER_TESTAUPRC_Evaluation_Dist-CB_Separation-{sep}.png')
                        if current_auprc == 'N/A':
                            self.assertFalse(os.path.isfile(fn2))
                        else:
                            self.assertTrue(os.path.isfile(fn2))
                            os.remove(fn2)
                if v == 1:
                    self.assertTrue(curr_b_w2_ave is None)
                    self.assertTrue(curr_u_w2_ave is None)
            if v >= 2:
                self.assertTrue('Top K Predictions' in curr_stats)
                self.assertTrue('Precision' in curr_stats)
                self.assertTrue('Recall' in curr_stats)
                self.assertTrue('F1 Score' in curr_stats)
                for sep in ['Any', 'Neighbors', 'Short', 'Medium', 'Long']:
                    for k in range(1, 11):
                        label = 'L' if k == 1 else f'L/{k}'
                        self.assertTrue(any(curr_stats['Top K Predictions'] == label))
                        current_precision = curr_stats.loc[(curr_stats['Sequence_Separation'] == sep) &
                                                           (curr_stats['Top K Predictions'] == label),
                                                           'Precision'].unique()[0]
                        expected_precision = scorer.score_precision(k=k, category=sep)
                        self.assertEqual(current_precision, expected_precision)
                        current_recall = curr_stats.loc[(curr_stats['Sequence_Separation'] == sep) &
                                                        (curr_stats['Top K Predictions'] == label),
                                                        'Recall'].unique()[0]
                        expected_recall = scorer.score_recall(k=k, category=sep)
                        self.assertEqual(current_recall, expected_recall)
                        current_f1 = curr_stats.loc[(curr_stats['Sequence_Separation'] == sep) &
                                                    (curr_stats['Top K Predictions'] == label),
                                                    'F1 Score'].unique()[0]
                        expected_f1 = scorer.score_f1(k=k, category=sep)
                        self.assertEqual(current_f1, expected_f1)
                if v == 2:
                    self.assertTrue(curr_b_w2_ave is None)
                    self.assertTrue(curr_u_w2_ave is None)
            if v >= 3:
                self.assertTrue('Max Biased Z-Score' in curr_stats)
                self.assertTrue('AUC Biased Z-Score' in curr_stats)
                biased_df, biased_reusable, biased_auc_scw_z_curve = scorer.score_clustering_of_contact_predictions(
                    bias=True, file_path='test.tsv')
                os.remove('test.tsv')
                self.assertEqual(curr_stats['AUC Biased Z-Score'].unique()[0], biased_auc_scw_z_curve)
                if biased_df['Z-Score'].max() == 'NA':
                    self.assertTrue(np.isnan(curr_stats['Max Biased Z-Score'].unique()[0]))
                else:
                    self.assertEqual(curr_stats['Max Biased Z-Score'].unique()[0], biased_df['Z-Score'].max())
                self.assertEqual(curr_b_w2_ave, biased_reusable)
                self.assertTrue('Max Unbiased Z-Score' in curr_stats)
                self.assertTrue('AUC Unbiased Z-Score' in curr_stats)
                unbiased_df, unbiased_reusable, unbiased_auc_scw_z_curve = scorer.score_clustering_of_contact_predictions(
                    bias=False, file_path='test.tsv')
                os.remove('test.tsv')
                self.assertEqual(curr_stats['AUC Unbiased Z-Score'].unique()[0], unbiased_auc_scw_z_curve)
                if unbiased_df['Z-Score'].max() == 'NA':
                    self.assertTrue(np.isnan(curr_stats['Max Unbiased Z-Score'].unique()[0]))
                else:
                    self.assertEqual(curr_stats['Max Unbiased Z-Score'].unique()[0], unbiased_df['Z-Score'].max())
                self.assertEqual(curr_u_w2_ave, unbiased_reusable)
                fn4 = os.path.join(output_dir, 'SCORER_TEST' + 'Dist-CB_Biased_ZScores.tsv')
                self.assertTrue(os.path.isfile(fn4))
                os.remove(fn4)
                fn5 = os.path.join(output_dir, 'SCORER_TEST' + 'Dist-CB_Unbiased_ZScores.tsv')
                self.assertTrue(os.path.isfile(fn5))
                os.remove(fn5)
                if plot:
                    fn6 = os.path.join(output_dir, 'SCORER_TEST' + 'Dist-CB_Biased_ZScores.png')
                    self.assertTrue(os.path.isfile(fn6))
                    os.remove(fn6)
                    fn7 = os.path.join(output_dir, 'SCORER_TEST' + 'Dist-CB_Unbiased_ZScores.png')
                    self.assertTrue(os.path.isfile(fn7))
                    os.remove(fn7)
                self.assertTrue(curr_b_w2_ave is not None)
                self.assertTrue(curr_u_w2_ave is not None)
            # Update
            prev_b_w2_ave = curr_b_w2_ave
            prev_u_w2_ave = curr_u_w2_ave

    def test_seq1_no_plots_one_process(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        new_dir = 'Test_Dir'
        os.mkdir(new_dir)
        self.evaluate_evaluate_predictor(scorer=scorer, output_dir=new_dir, processes=1, plot=False)
        rmtree(new_dir)

    def test_seq1_plots_one_process(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        new_dir = 'Test_Dir'
        os.mkdir(new_dir)
        self.evaluate_evaluate_predictor(scorer=scorer, output_dir=new_dir, processes=1, plot=True)
        rmtree(new_dir)

    def test_seq1_plots_multi_process(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                               cutoff=16.0, chain='A')
        new_dir = 'Test_Dir'
        os.mkdir(new_dir)
        self.evaluate_evaluate_predictor(scorer=scorer, output_dir=new_dir, processes=2, plot=True)
        rmtree(new_dir)

    def test_seq1_alt_loc_pdb(self):
        scorer = ContactScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                               cutoff=14.0, chain='A')
        new_dir = 'Test_Dir'
        os.mkdir(new_dir)
        self.evaluate_evaluate_predictor(scorer=scorer, output_dir=new_dir, processes=2, plot=True)
        rmtree(new_dir)

    def test_seq2(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        new_dir = 'Test_Dir'
        os.mkdir(new_dir)
        self.evaluate_evaluate_predictor(scorer=scorer, output_dir=new_dir, processes=2, plot=True)
        rmtree(new_dir)

    def test_seq3(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        new_dir = 'Test_Dir'
        os.mkdir(new_dir)
        self.evaluate_evaluate_predictor(scorer=scorer, output_dir=new_dir, processes=2, plot=True)
        rmtree(new_dir)


# class TestPlotZScores(TestCase):
# class TestContactScorerScoreClusteringOfContactPredictions(TestCase):


class TestSCWZScoreSelection(TestCase):

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.pdb_fn1)
        os.remove(cls.pdb_fn1b)
        os.remove(cls.pdb_fn2)
        os.remove(aln_fn)

    @classmethod
    def setUpClass(cls):
        cls.pdb_fn1 = write_out_temp_fn(out_str=pro_pdb1, suffix='1.pdb')
        cls.pdb_fn1b = write_out_temp_fn(out_str=pro_pdb_1_alt_locs, suffix='1b.pdb')
        cls.pdb_fn2 = write_out_temp_fn(out_str=pro_pdb2, suffix='2.pdb')
        cls.pdb_chain_a = PDBReference(pdb_file=cls.pdb_fn1)
        cls.pdb_chain_a.import_pdb(structure_id='1TES')
        cls.pdb_chain_a_alt = PDBReference(pdb_file=cls.pdb_fn1b)
        cls.pdb_chain_a_alt.import_pdb(structure_id='1TES')
        cls.pdb_chain_b = PDBReference(pdb_file=cls.pdb_fn2)
        cls.pdb_chain_b.import_pdb(structure_id='1TES')
        protein_aln1.write_out_alignment(aln_fn)

    def evaluate_scw_z_score_selection(self, scw_scorer):
        scw_scorer.compute_background_w_and_w2_ave(processes=2)
        chain_length = scw_scorer.query_pdb_mapper.pdb_ref.size[scw_scorer.query_pdb_mapper.best_chain]
        for i in range(chain_length):
            for comb in combinations(list(range(scw_scorer.query_pdb_mapper.seq_aln.seq_length)), i):
                expected_scw_z_score = scw_scorer.clustering_z_score(res_list=list(comb))
                init_scw_z_score_selection(scw_scorer)
                scw_z_score = scw_z_score_selection(res_list=list(comb))
                mappable = all([x in scw_scorer.query_pdb_mapper.query_pdb_mapping for x in comb])
                if mappable:
                    self.assertEqual(scw_z_score[1], expected_scw_z_score[1])
                    self.assertEqual(scw_z_score[2], expected_scw_z_score[2])
                    self.assertEqual(scw_z_score[3], expected_scw_z_score[3])
                    self.assertEqual(scw_z_score[4], expected_scw_z_score[4])
                    self.assertEqual(scw_z_score[5], expected_scw_z_score[5])
                    self.assertEqual(scw_z_score[6], expected_scw_z_score[6])
                    self.assertEqual(scw_z_score[7], expected_scw_z_score[7])
                    self.assertEqual(scw_z_score[8], expected_scw_z_score[8])
                    self.assertEqual(scw_z_score[9], expected_scw_z_score[9])
                    self.assertEqual(scw_z_score[10], expected_scw_z_score[10])
                else:
                    self.assertIsNone(scw_z_score[1])
                    self.assertIsNone(scw_z_score[2])
                    self.assertIsNone(scw_z_score[3])
                    self.assertIsNone(scw_z_score[4])
                    self.assertIsNone(scw_z_score[5])
                    self.assertEqual(scw_z_score[6], '-')
                    self.assertIsNone(scw_z_score[7])
                    self.assertIsNone(scw_z_score[8])
                    self.assertIsNone(scw_z_score[9])
                    self.assertIsNone(scw_z_score[10])
                    self.assertEqual(scw_z_score[11], len(comb))

    def test_seq2_no_bias(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=False)
        self.evaluate_scw_z_score_selection(scw_scorer=scw_scorer)

    def test_seq2_bias(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        self.evaluate_scw_z_score_selection(scw_scorer=scw_scorer)

    def test_seq3_no_bias(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=False)
        self.evaluate_scw_z_score_selection(scw_scorer=scw_scorer)

    def test_seq3_bias(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        self.evaluate_scw_z_score_selection(scw_scorer=scw_scorer)


if __name__ == '__main__':
    unittest.main()
