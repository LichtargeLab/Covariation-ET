import os
import sys
import math
import numpy as np
import pandas as pd
from math import floor
from shutil import rmtree
from unittest import TestCase
from scipy.stats import rankdata
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

from SupportingClasses.utils import compute_rank_and_coverage
from SupportingClasses.PDBReference import PDBReference
from SupportingClasses.SeqAlignment import SeqAlignment
from SupportingClasses.EvolutionaryTraceAlphabet import FullIUPACProtein
from Evaluation.SelectionClusterWeighting import SelectionClusterWeighting
from Evaluation.SinglePositionScorer import SinglePositionScorer
from Evaluation.SequencePDBMap import SequencePDBMap
from Testing.test_Scorer import et_calcDist, et_computeAdjacency, et_calcZScore
from Testing.test_Base import (protein_seq1, protein_seq2, protein_seq3, protein_aln, write_out_temp_fn)
from Testing.test_PDBReference import chain_a_pdb_partial2, chain_a_pdb_partial, chain_b_pdb, chain_b_pdb_partial

pro_str = f'>{protein_seq1.id}\n{protein_seq1.seq}\n>{protein_seq2.id}\n{protein_seq2.seq}\n>{protein_seq3.id}\n{protein_seq3.seq}'
pro_str_long = f'>{protein_seq1.id}\n{protein_seq1.seq*10}\n>{protein_seq2.id}\n{protein_seq2.seq*10}\n>{protein_seq3.id}\n{protein_seq3.seq*10}'
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
pro_pdb_1_alt_locs = chain_a_pdb_partial2 + chain_a_pdb_partial
pro_pdb2 = chain_b_pdb
pro_pdb2_scramble = chain_b_pdb + chain_b_pdb_partial
pro_pdb_full = pro_pdb1 + pro_pdb2
pro_pdb_full_scramble = pro_pdb1_scramble + pro_pdb2_scramble


def identify_expected_scores_and_distances(scorer, scores, coverages, ranks, n=None, k=None, coverage_cutoff=None,
                                           threshold=0.5):
    converted_ind = sorted(scorer.query_pdb_mapper.query_pdb_mapping.keys())
    if n and k and coverage_cutoff:
        raise ValueError('n, k, and coverage_cutoff cannot be defined when identifying data for testing.')
    elif n and k:
        raise ValueError('Both n and k cannot be defined when identifying data for testing.')
    elif n and coverage_cutoff:
        raise ValueError('Both n and coverage_cutoff cannot be defined when identifying data for testing.')
    elif k and coverage_cutoff:
        raise ValueError('Both k and coverage_cutoff cannot be defined when identifying data for testing.')
    elif k is not None:
        n = int(floor(scorer.query_pdb_mapper.seq_aln.seq_length / float(k)))
    elif coverage_cutoff is not None:
        rank_ind = []
        for ind in converted_ind:
            rank_ind.append((ranks[ind], ind))
        max_res_count = floor(scorer.query_pdb_mapper.pdb_ref.size[scorer.query_pdb_mapper.best_chain] *
                              coverage_cutoff)
        sorted_rank_ind = sorted(rank_ind)
        prev_rank = None
        curr_res = set()
        rank_pairs = 0
        n = 0
        all_res = set()
        for rank, ind in sorted_rank_ind:
            if (prev_rank is not None) and (prev_rank != rank):
                # Evaluate
                curr_res_count = len(all_res.union(curr_res))
                if curr_res_count > max_res_count:
                    break
                # Reset
                all_res |= curr_res
                curr_res = set()
                n += rank_pairs
                rank_pairs = 0
                prev_rank = rank
            curr_res.add(ind)
            rank_pairs += 1
            prev_rank = rank
        if (len(all_res) <= max_res_count) and (len(all_res.union(curr_res)) <= max_res_count):
            all_res |= curr_res
            n += 1
    elif n is None and k is None:
        try:
            n = len(converted_ind)
        except IndexError:
            n = 0
    else:
        pass
    scores_subset = scores[converted_ind]
    coverage_subset = coverages[converted_ind]
    ranks_subset = ranks[converted_ind]
    preds_subset = coverage_subset <= threshold
    if len(converted_ind) == 0:
        df_final = pd.DataFrame({'Seq Pos 1': [], 'Seq Pos 2': [], 'Struct Pos 1': [], 'Struct Pos 2': [],
                                 'Score': [], 'Coverage': [], 'Rank': [], 'Predictions': [], 'Distance': [],
                                 'Contact': [], 'Top Predictions': []})
    else:
        df = pd.DataFrame({'Seq Pos': converted_ind,
                           'Struct Pos': [scorer.query_pdb_mapper.pdb_ref.pdb_residue_list[scorer.query_pdb_mapper.best_chain][x]
                                          for x in converted_ind],
                           'Score': scores_subset, 'Coverage': coverage_subset, 'Rank': ranks_subset,
                           'Predictions': preds_subset})
        df_sorted = df.sort_values(by='Coverage')
        df_sorted['Top Predictions'] = rankdata(df_sorted['Coverage'], method='dense')
        n_index = df_sorted['Top Predictions'] <= n
        df_final = df_sorted.loc[n_index, :]
    return df_final


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


class TestSinglePositionScorerFit(TestCase):

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
            self.assertEqual(scorer.data.loc[i, 'Seq AA'], expected_seq.seq[scorer.data.loc[i, 'Seq Pos']])
            if scorer.data.loc[i, 'Struct Pos'] == '-':
                self.assertFalse(scorer.data.loc[i, 'Seq AA'] in scorer.query_pdb_mapper.query_pdb_mapping)
                self.assertEqual(scorer.data.loc[i, 'Struct AA'], '-')
            else:
                mapped_struct_pos = scorer.query_pdb_mapper.query_pdb_mapping[scorer.data.loc[i, 'Seq Pos']]
                self.assertEqual(scorer.data.loc[i, 'Struct Pos'],
                                 expected_struct.pdb_residue_list[expected_chain][mapped_struct_pos])
                self.assertEqual(scorer.data.loc[i, 'Struct AA'],
                                 expected_struct.seq[expected_chain][mapped_struct_pos])

    def test_fit_aln_file_pdb_file_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        scorer = SinglePositionScorer(query='seq1', seq_alignment=aln_fn, pdb_reference=pdb_fn, chain='A')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='A', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, chain='A')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='A', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_obj_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = SinglePositionScorer(query='seq1', seq_alignment=aln_fn, pdb_reference=pdb, chain='A')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='A', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb, chain='A')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='A', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_not_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb, chain=None)
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='A', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1_scramble)
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, chain='A')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='A', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_not_specified_1(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb1_scramble)
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, chain=None)
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='A', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_file_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        scorer = SinglePositionScorer(query='seq1', seq_alignment=aln_fn, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_obj_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = SinglePositionScorer(query='seq1', seq_alignment=aln_fn, pdb_reference=pdb, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_not_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb, chain=None)
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_not_specified_1_mismatch(self):
        protein_aln1.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=pdb_fn, chain=None)
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B', expected_seq=protein_seq1)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_file_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        scorer = SinglePositionScorer(query='seq2', seq_alignment=aln_fn, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_2)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_2)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_obj_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = SinglePositionScorer(query='seq2', seq_alignment=aln_fn, pdb_reference=pdb, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=self.expected_pro_seq_2)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=pdb, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=self.expected_pro_seq_2)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_not_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=pdb, chain=None)
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=self.expected_pro_seq_2)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_2)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_not_specified_2(self):
        protein_aln2.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb2_scramble)
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=pdb_fn, chain=None)
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_2)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_file_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        scorer = SinglePositionScorer(query='seq3', seq_alignment=aln_fn, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_3)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_3)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_file_pdb_obj_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = SinglePositionScorer(query='seq3', seq_alignment=aln_fn, pdb_reference=pdb, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=self.expected_pro_seq_3)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=pdb, chain='B')
        self.evaluate_fit(scorer=scorer, expected_struct=pdb, expected_chain='B', expected_seq=self.expected_pro_seq_3)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_obj_chain_not_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full)
        pdb = PDBReference(pdb_file=pdb_fn)
        pdb.import_pdb(structure_id='1TES')
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=pdb, chain=None)
        self.evaluate_fit(scorer=scorer,  expected_struct=pdb, expected_chain='B', expected_seq=self.expected_pro_seq_3)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full_scramble)
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=pdb_fn, chain='B')
        self.evaluate_fit(scorer=scorer,  expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_3)
        os.remove(aln_fn)
        os.remove(pdb_fn)

    def test_fit_aln_obj_pdb_file_scrambled_chain_not_specified_3(self):
        protein_aln3.write_out_alignment(aln_fn)
        pdb_fn = write_out_temp_fn(suffix='pdb', out_str=pro_pdb_full_scramble)
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=pdb_fn, chain=None)
        self.evaluate_fit(scorer=scorer, expected_struct=pdb_fn, expected_chain='B',
                          expected_seq=self.expected_pro_seq_3)
        os.remove(aln_fn)
        os.remove(pdb_fn)


class TestSinglePositionScorerMapPredictionsToPDB(TestCase):

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
        scores = np.random.rand(scorer.query_pdb_mapper.seq_aln.seq_length)
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 1, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        self.assertIsNotNone(scorer.data)
        self.assertTrue(pd.Series(['Rank', 'Score', 'Coverage', 'True Prediction']).isin(scorer.data.columns).all())
        for i in scorer.data.index:
            pos1 = scorer.data.loc[i, 'Seq Pos']
            self.assertEqual(ranks[pos1], scorer.data.loc[i, 'Rank'])
            self.assertEqual(scores[pos1], scorer.data.loc[i, 'Score'])
            self.assertEqual(coverages[pos1], scorer.data.loc[i, 'Coverage'])
            if coverages[pos1] <= 0.5:
                self.assertEqual(scorer.data.loc[i, 'True Prediction'], 1)
            else:
                self.assertEqual(scorer.data.loc[i, 'True Prediction'], 0)

    def test_map_prediction_to_pdb_seq1(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                                      chain='A')
        self.evaluate_map_prediction_to_pdb(scorer=scorer)

    def test_map_prediction_to_pdb_seq1_alt_loc_pdb(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                                      chain='A')
        self.evaluate_map_prediction_to_pdb(scorer=scorer)

    def test_map_prediction_to_pdb_seq2(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_map_prediction_to_pdb(scorer=scorer)

    def test_map_prediction_to_pdb_seq3(self):
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_map_prediction_to_pdb(scorer=scorer)


class TestSinglePositionScorerIdentifyRelevantData(TestCase):

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
        scores = np.random.rand(scorer.query_pdb_mapper.seq_aln.seq_length)
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 1, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        with self.assertRaises(ValueError):
            scorer._identify_relevant_data(n=10, k=10, coverage_cutoff=None)
        with self.assertRaises(ValueError):
            scorer._identify_relevant_data(n=None, k=10, coverage_cutoff=0.1)
        with self.assertRaises(ValueError):
            scorer._identify_relevant_data(n=10, k=None, coverage_cutoff=0.1)
        with self.assertRaises(ValueError):
            scorer._identify_relevant_data(n=10, k=10, coverage_cutoff=0.1)
        with self.assertRaises(AssertionError):
            scorer._identify_relevant_data(n=None, k=None, coverage_cutoff=10)
        for n, k, cc in [(None, None, None), (1, None, None), (2, None, None), (3, None, None), (None, 1, None),
                         (None, 2, None), (None, 3, None), (None, None, 0.34), (None, None, 0.67), (None, None, 1.0)]:
            curr_subset = scorer._identify_relevant_data(n=n, k=k, coverage_cutoff=cc)
            expected_subset = identify_expected_scores_and_distances(
                scorer, scores, coverages, ranks, n=n, k=k, coverage_cutoff=cc)
            if len(curr_subset) == 0:
                self.assertEqual(len(expected_subset), 0)
            else:
                self.assertEqual(len(curr_subset), len(expected_subset))
                seq_1_pos_diff = np.abs(curr_subset['Seq Pos'].values - expected_subset['Seq Pos'].values)
                seq_1_pos_not_passing = seq_1_pos_diff > 0
                self.assertFalse(seq_1_pos_not_passing.any())
                struct_1_pos_diff = np.abs(curr_subset['Struct Pos'].values -
                                           expected_subset['Struct Pos'].values)
                struct_1_not_passing = struct_1_pos_diff > 0
                self.assertFalse(struct_1_not_passing.any())
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

    def test_identify_relevant_data_seq1(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                                      chain='A')
        self.evaluate_identify_relevant_data(scorer=scorer)

    def test_identify_relevant_data_seq1_alt_loc_pdb(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a_alt,
                                      chain='A')
        self.evaluate_identify_relevant_data(scorer=scorer)

    def test_identify_relevant_data_seq2(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_identify_relevant_data(scorer=scorer)

    def test_identify_relevant_data_seq3(self):
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_identify_relevant_data(scorer=scorer)


class TestSinglePositionScorerScorePDBResidueIdentification(TestCase):

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

    def evaluate_score_pdb_residue_identification(self, scorer, scores, n, k, cov_cutoff, res_list, expected_X,
                                                  expected_n, expected_N, expected_M, expected_p_val):
        scorer.fit()
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 1, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        X, M, n, N, p_val = scorer.score_pdb_residue_identification(pdb_residues=res_list, n=n, k=k,
                                                                    coverage_cutoff=cov_cutoff)
        self.assertEqual(X, expected_X)
        self.assertEqual(M, expected_M)
        self.assertEqual(n, expected_n)
        self.assertEqual(N, expected_N)
        self.assertLess(abs(p_val - expected_p_val), 1E-6)

    def test_seq1_n(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                                      chain='A')
        self.evaluate_score_pdb_residue_identification(scorer=scorer, n=1, k=None, cov_cutoff=None, res_list=[1],
                                                       expected_X=1, expected_n=1, expected_N=1, expected_M=3,
                                                       expected_p_val=0.3333333, scores=np.array([0.1, 0.5, 0.9]))

    def test_seq1_k(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                                      chain='A')
        self.evaluate_score_pdb_residue_identification(scorer=scorer, n=None, k=3, cov_cutoff=None, res_list=[1],
                                                       expected_X=1, expected_n=1, expected_N=1, expected_M=3,
                                                       expected_p_val=0.3333333, scores=np.array([0.1, 0.5, 0.9]))

    def test_seq1_cov(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                                      chain='A')
        self.evaluate_score_pdb_residue_identification(scorer=scorer, n=None, k=None, cov_cutoff=0.67, res_list=[1],
                                                       expected_X=1, expected_n=1, expected_N=2, expected_M=3,
                                                       expected_p_val=0.6666666, scores=np.array([0.1, 0.5, 0.9]))

    def test_seq2_n(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_score_pdb_residue_identification(scorer=scorer, n=1, k=None, cov_cutoff=None, res_list=[1, 2],
                                                       expected_X=1, expected_n=2, expected_N=1, expected_M=5,
                                                       expected_p_val=0.4, scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))

    def test_seq2_k(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_score_pdb_residue_identification(scorer=scorer, n=None, k=5, cov_cutoff=None, res_list=[1, 2],
                                                       expected_X=1, expected_n=2, expected_N=1, expected_M=5,
                                                       expected_p_val=0.4, scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))

    def test_seq2_cov(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_score_pdb_residue_identification(scorer=scorer, n=None, k=None, cov_cutoff=0.4, res_list=[1, 2],
                                                       expected_X=1, expected_n=2, expected_N=2, expected_M=5,
                                                       expected_p_val=0.7, scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))

    def test_seq3_n(self):
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_score_pdb_residue_identification(scorer=scorer, n=2, k=None, cov_cutoff=None, res_list=[1, 2],
                                                       expected_X=1, expected_n=2, expected_N=2, expected_M=5,
                                                       expected_p_val=0.7, scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))

    def test_seq3_k(self):
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_score_pdb_residue_identification(scorer=scorer, n=None, k=2, cov_cutoff=None, res_list=[1, 2],
                                                       expected_X=1, expected_n=2, expected_N=2, expected_M=5,
                                                       expected_p_val=0.7, scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))

    def test_seq3_cov(self):
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_score_pdb_residue_identification(scorer=scorer, n=None, k=None, cov_cutoff=0.8, res_list=[1, 2],
                                                       expected_X=1, expected_n=2, expected_N=4, expected_M=5,
                                                       expected_p_val=1.0, scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))

    def test_fail_n_and_k(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                                      chain='A')
        with self.assertRaises(ValueError):
            self.evaluate_score_pdb_residue_identification(scorer=scorer, n=2, k=2, cov_cutoff=None, res_list=None,
                                                           expected_X=None, expected_n=None, expected_N=None,
                                                           expected_M=None, expected_p_val=None,
                                                           scores=np.array([0.1, 0.5, 0.9]))

    def test_fail_n_and_cov_cutoff(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                                      chain='A')
        with self.assertRaises(ValueError):
            self.evaluate_score_pdb_residue_identification(scorer=scorer, n=2, k=None, cov_cutoff=0.3, res_list=None,
                                                           expected_X=None, expected_n=None, expected_N=None,
                                                           expected_M=None, expected_p_val=None,
                                                           scores=np.array([0.1, 0.5, 0.9]))

    def test_fail_k_and_cov_cutoff(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                                      chain='A')
        with self.assertRaises(ValueError):
            self.evaluate_score_pdb_residue_identification(scorer=scorer, n=None, k=2, cov_cutoff=0.3, res_list=None,
                                                           expected_X=None, expected_n=None, expected_N=None,
                                                           expected_M=None, expected_p_val=None,
                                                           scores=np.array([0.1, 0.5, 0.9]))


class TestSinglePositionScorerRecoveryOfPDBResidues(TestCase):

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

    def evaluate_recovery_of_pdb_residues(self, scorer, scores, res_list, expected_tpr, expected_fpr, expected_auroc,
                                          expected_precision, expected_recall, expected_auprc):
        scorer.fit()
        scorer.measure_distance(method='CB')
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 1, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        tpr, fpr, auroc, precision, recall, auprc = scorer.recovery_of_pdb_residues(pdb_residues=res_list)
        np.testing.assert_equal(tpr, expected_tpr)
        np.testing.assert_equal(fpr, expected_fpr)
        np.testing.assert_equal(precision, expected_precision)
        np.testing.assert_equal(recall, expected_recall)
        if np.isnan(expected_auroc):
            self.assertTrue(np.isnan(auroc))
        else:
            self.assertEqual(auroc, expected_auroc)
        if np.isnan(expected_auprc):
            self.assertTrue(np.isnan(auprc))
        else:
            self.assertEqual(auprc, expected_auprc)
        print('FINAL AUROC: ', auroc)
        print('FINAL AUPRC: ', auprc)

    def test_seq1(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                                      chain='A')
        self.evaluate_recovery_of_pdb_residues(scorer=scorer, res_list=[1], expected_tpr=[0.0, 1.0, 1.0],
                                               expected_fpr=[0.0, 0.0, 1.0], expected_auroc=1.0,
                                               expected_precision=[1.0, 1.0], expected_recall=[0.0, 1.0],
                                               expected_auprc=1.0, scores=np.array([0.1, 0.5, 0.9]))

    def test_seq2(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_recovery_of_pdb_residues(scorer=scorer, res_list=[1, 2], expected_tpr=[0.0, 0.5, 0.5, 1.0],
                                               expected_fpr=[0.0, 0.0, 1.0, 1.0], expected_auroc=0.5,
                                               expected_precision=[1.0, 0.25, 1 / 3., 0.5, 1.0, 0.4],
                                               expected_recall=[0.0, 0.5, 0.5, 0.5, 0.5, 1.0], expected_auprc=0.6625,
                                               scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))

    def test_seq3(self):
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_recovery_of_pdb_residues(scorer=scorer, res_list=[1, 2], expected_tpr=[0.0, 0.5, 0.5, 1.0],
                                               expected_fpr=[0.0, 0.0, 1.0, 1.0], expected_auroc=0.5,
                                               expected_precision=[1.0, 0.25, 1 / 3., 0.5, 1.0, 0.4],
                                               expected_recall=[0.0, 0.5, 0.5, 0.5, 0.5, 1.0], expected_auprc=0.6625,
                                               scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))

    def test_no_residues(self):
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_recovery_of_pdb_residues(scorer=scorer, res_list=[], expected_tpr=[np.nan, np.nan, np.nan],
                                               expected_fpr=[0.0, 0.2, 1.0], expected_auroc=np.nan,
                                               expected_precision=[0.0, 1.0], expected_recall=[np.nan, 0.0],
                                               expected_auprc=np.nan, scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))

    def test_fail_bad_residues(self):
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        with self.assertRaises(TypeError):
            self.evaluate_recovery_of_pdb_residues(scorer=scorer, res_list=None, expected_tpr=[np.nan, np.nan, np.nan],
                                                   expected_fpr=[0.0, 0.2, 1.0], expected_auroc=np.nan,
                                                   expected_precision=[0.0, 1.0], expected_recall=[np.nan, 0.0],
                                                   expected_auprc=np.nan, scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))


class TestSinglePositionScorerSelectAndColorResidues(TestCase):

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

    def evaluate_select_and_color_residues(self, scorer, scores, n, k, cov_cutoff):
        scorer.fit()
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 1, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        new_dir = 'Testing_Dir'
        os.mkdir(new_dir)
        scorer.select_and_color_residues(out_dir=new_dir, n=n, k=k, residue_coverage=cov_cutoff,
                                         fn=None)
        self.assertTrue(os.path.isfile(os.path.join(
            new_dir, f'{scorer.query_pdb_mapper.query}_Coverage_Chain_{scorer.query_pdb_mapper.best_chain}_colored_residues.pse')))
        self.assertTrue(os.path.isfile(os.path.join(
            new_dir, f'{scorer.query_pdb_mapper.query}_Coverage_Chain_{scorer.query_pdb_mapper.best_chain}_colored_residues_all_pymol_commands.txt')))
        scorer.select_and_color_residues(out_dir=new_dir, n=n, k=k, residue_coverage=cov_cutoff,
                                         fn='Test')
        self.assertTrue(os.path.isfile(os.path.join(new_dir, 'Test.pse')))
        self.assertTrue(os.path.isfile(os.path.join(new_dir, 'Test_all_pymol_commands.txt')))
        rmtree(new_dir)

    def test_seq1_n(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                                      chain='A')
        self.evaluate_select_and_color_residues(scorer=scorer, n=1, k=None, cov_cutoff=None,
                                                scores=np.array([0.1, 0.5, 0.9]))

    def test_seq1_k(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                                      chain='A')
        self.evaluate_select_and_color_residues(scorer=scorer, n=None, k=3, cov_cutoff=None,
                                                scores=np.array([0.1, 0.5, 0.9]))

    def test_seq1_cov(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                                      chain='A')
        self.evaluate_select_and_color_residues(scorer=scorer, n=None, k=None, cov_cutoff=0.67,
                                                scores=np.array([0.1, 0.5, 0.9]))

    def test_seq2_n(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_select_and_color_residues(scorer=scorer, n=1, k=None, cov_cutoff=None,
                                                scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))

    def test_seq2_k(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_select_and_color_residues(scorer=scorer, n=None, k=5, cov_cutoff=None,
                                                scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))

    def test_seq2_cov(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_select_and_color_residues(scorer=scorer, n=None, k=None, cov_cutoff=0.4,
                                                scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))

    def test_seq3_n(self):
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_select_and_color_residues(scorer=scorer, n=2, k=None, cov_cutoff=None,
                                                scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))

    def test_seq3_k(self):
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_select_and_color_residues(scorer=scorer, n=None, k=2, cov_cutoff=None,
                                                scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))

    def test_seq3_cov(self):
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        self.evaluate_select_and_color_residues(scorer=scorer, n=None, k=None, cov_cutoff=0.8,
                                                scores=np.array([0.1, 0.5, 0.3, 0.4, 0.2]))

    def test_fail_n_and_k(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                                      chain='A')
        with self.assertRaises(ValueError):
            self.evaluate_select_and_color_residues(scorer=scorer, n=2, k=2, cov_cutoff=None,
                                                    scores=np.array([0.1, 0.5, 0.9]))
        rmtree('Testing_Dir')

    def test_fail_n_and_cov_cutoff(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                                      chain='A')
        with self.assertRaises(ValueError):
            self.evaluate_select_and_color_residues(scorer=scorer, n=2, k=None, cov_cutoff=0.3,
                                                    scores=np.array([0.1, 0.5, 0.9]))
        rmtree('Testing_Dir')

    def test_fail_k_and_cov_cutoff(self):
        scorer = SinglePositionScorer(query='seq1', seq_alignment=protein_aln1, pdb_reference=self.pdb_chain_a,
                                      chain='A')
        with self.assertRaises(ValueError):
            self.evaluate_select_and_color_residues(scorer=scorer, n=None, k=2, cov_cutoff=0.3,
                                                    scores=np.array([0.1, 0.5, 0.9]))
        rmtree('Testing_Dir')


class TestSinglePositionScorerScoreClusteringOfImportantResidues(TestCase):

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

    def evaluate_score_clustering_of_important_residues(self, scorer, bias, processes, scw_scorer):
        # Initialize scorer and scores
        scorer.fit()
        scorer.measure_distance(method='Any')
        scores = np.random.RandomState(1234567890).rand(scorer.query_pdb_mapper.seq_aln.seq_length)
        ranks, coverages = compute_rank_and_coverage(scorer.query_pdb_mapper.seq_aln.seq_length, scores, 1, 'min')
        scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
        # Calculate Z-scores for the structure
        expected_fn = os.path.join('TEST_z_score.tsv')
        zscore_df, _, _ = scorer.score_clustering_of_important_residues(biased=bias, file_path=expected_fn,
                                                                        scw_scorer=scw_scorer, processes=processes)
        print(zscore_df[['Res', 'Coverage', 'Z-Score']])
        # Check that the scoring file was written out to the expected file.
        self.assertTrue(os.path.isfile(expected_fn))
        os.remove(expected_fn)
        # Generate data for calculating expected values
        recip_map = {v: k for k, v in scorer.query_pdb_mapper.query_pdb_mapping.items()}
        struc_seq_map = {k: i for i, k in
                         enumerate(scorer.query_pdb_mapper.pdb_ref.pdb_residue_list[scorer.query_pdb_mapper.best_chain])}
        final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
        A, res_atoms = et_computeAdjacency(scorer.query_pdb_mapper.pdb_ref.structure[0][scorer.query_pdb_mapper.best_chain],
                                           mapping=final_map)
        # Iterate over the returned data frame row by row and test whether the results are correct
        visited_scorable_residues = set()
        prev_len = 0
        prev_stats = None
        prev_composed_w2_ave = None
        prev_composed_sigma = None
        prev_composed_z_score = None
        zscore_df[['Res']] = zscore_df[['Res']].astype(dtype=np.int64)
        zscore_df[['Importance_Score', 'Coverage', 'W', 'W_Ave', 'W2_Ave', 'Sigma', 'Num_Residues']].replace(
            [None, '-', 'NA'], np.nan, inplace=True)
        zscore_df[['Importance_Score', 'Coverage', 'W', 'W_Ave', 'W2_Ave', 'Sigma']] = zscore_df[
            ['Importance_Score', 'Coverage', 'W', 'W_Ave', 'W2_Ave', 'Sigma']].astype(dtype=np.float64)
        curr_res = set()
        prev_coverage = None
        for ind in zscore_df.index:
            res = zscore_df.loc[ind, 'Res']
            cov = zscore_df.loc[ind, 'Coverage']
            # Check for ties
            if (prev_coverage is None) or (prev_coverage == cov):
                curr_res.add(res)
            else:
                sub_df = zscore_df.loc[zscore_df['Res'].isin(curr_res), :]
                curr_covs = sub_df['Coverage'].unique()
                self.assertTrue(len(curr_covs), 1)
                if curr_covs[0] != '-':
                    visited_scorable_residues |= curr_res
                    if len(visited_scorable_residues) > prev_len:
                        curr_stats = et_calcZScore(reslist=sorted(visited_scorable_residues),
                                                   L=len(scorer.query_pdb_mapper.pdb_ref.seq[scorer.query_pdb_mapper.best_chain]),
                                                   A=A, bias=1 if bias else 0)
                        expected_composed_w2_ave = ((curr_stats[2] * curr_stats[10]['Case1']) +
                                                    (curr_stats[3] * curr_stats[10]['Case2']) +
                                                    (curr_stats[4] * curr_stats[10]['Case3']))
                        expected_composed_sigma = math.sqrt(expected_composed_w2_ave - curr_stats[7] * curr_stats[7])
                        if expected_composed_sigma == 0.0:
                            expected_composed_z_score = 'NA'
                        else:
                            expected_composed_z_score = (curr_stats[6] - curr_stats[7]) / expected_composed_sigma
                        prev_len = len(visited_scorable_residues)
                        prev_stats = curr_stats
                        prev_composed_w2_ave = expected_composed_w2_ave
                        prev_composed_sigma = expected_composed_sigma
                        prev_composed_z_score = expected_composed_z_score
                    else:
                        curr_stats = prev_stats
                        expected_composed_w2_ave = prev_composed_w2_ave
                        expected_composed_sigma = prev_composed_sigma
                        expected_composed_z_score = prev_composed_z_score
                    error_message = f'\nW: {zscore_df.loc[ind, "W"]}\nExpected W: {curr_stats[6]}\n' \
                                    f'W Ave: {zscore_df.loc[ind, "W_Ave"]}\nExpected W Ave: {curr_stats[7]}\n' \
                                    f'W2 Ave: {zscore_df.loc[ind, "W2_Ave"]}\nExpected W2 Ave: {curr_stats[8]}\n' \
                                    f'Composed Expected W2 Ave: {expected_composed_w2_ave}\n' \
                                    f'Sigma: {zscore_df.loc[ind, "Sigma"]}\nExpected Sigma: {curr_stats[9]}\n' \
                                    f'Composed Expected Sigma: {expected_composed_sigma}\n' \
                                    f'Z-Score: {zscore_df.loc[ind, "Z-Score"]}\nExpected Z-Score: {curr_stats[5]}\n' \
                                    f'Composed Expected Z-Score: {expected_composed_z_score}'
                    self.assertEqual(zscore_df.loc[ind, 'Num_Residues'], len(visited_scorable_residues))
                    self.assertLessEqual(np.abs(zscore_df.loc[ind, 'W'] - curr_stats[6]), 1E-16, error_message)
                    self.assertLessEqual(np.abs(zscore_df.loc[ind, 'W_Ave'] - curr_stats[7]), 1E-16, error_message)
                    self.assertLessEqual(np.abs(zscore_df.loc[ind, 'W2_Ave'] - expected_composed_w2_ave), 1E-16,
                                         error_message)
                    self.assertLessEqual(np.abs(zscore_df.loc[ind, 'W2_Ave'] - curr_stats[8]), 1E-2, error_message)
                    self.assertLessEqual(np.abs(zscore_df.loc[ind, 'Sigma'] - expected_composed_sigma), 1E-16,
                                         error_message)
                    self.assertLessEqual(np.abs(zscore_df.loc[ind, 'Sigma'] - curr_stats[9]), 1E-5, error_message)
                    if expected_composed_sigma == 0.0:
                        self.assertEqual(zscore_df.loc[ind, 'Z-Score'], expected_composed_z_score)
                        self.assertEqual(zscore_df.loc[ind, 'Z-Score'], curr_stats[5])
                    else:
                        self.assertLessEqual(np.abs(zscore_df.loc[ind, 'Z-Score'] - expected_composed_z_score), 1E-16,
                                             error_message)
                        self.assertLessEqual(np.abs(zscore_df.loc[ind, 'Z-Score'] - curr_stats[5]), 1E-5, error_message)
                else:
                    self.assertEqual(zscore_df.loc[ind, 'Z-Score'], '-')
                    self.assertTrue(np.isnan(zscore_df.loc[ind, 'W']))
                    self.assertTrue(np.isnan(zscore_df.loc[ind, 'W_Ave']))
                    self.assertTrue(np.isnan(zscore_df.loc[ind, 'W2_Ave']))
                    self.assertTrue(np.isnan(zscore_df.loc[ind, 'Sigma']))
                    self.assertIsNone(zscore_df.loc[ind, 'Num_Residues'])
            self.assertEqual(zscore_df.loc[ind, 'Importance_Score'], scores[res])

    def test_seq2_no_bias_single_process_no_scw(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        self.evaluate_score_clustering_of_important_residues(scorer=scorer, bias=False, processes=1, scw_scorer=None)

    def test_seq2_no_bias_single_process(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=False)
        scw_scorer.compute_background_w_and_w2_ave(processes=2)
        self.evaluate_score_clustering_of_important_residues(scorer=scorer, bias=False, processes=1,
                                                             scw_scorer=scw_scorer)

    def test_seq2_no_bias_multi_process(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=False)
        scw_scorer.compute_background_w_and_w2_ave(processes=2)
        self.evaluate_score_clustering_of_important_residues(scorer=scorer, bias=False, processes=2,
                                                             scw_scorer=scw_scorer)

    def test_seq2_bias_single_process_no_scw(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        self.evaluate_score_clustering_of_important_residues(scorer=scorer, bias=True, processes=1, scw_scorer=None)

    def test_seq2_bias_single_process(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        scw_scorer.compute_background_w_and_w2_ave(processes=2)
        self.evaluate_score_clustering_of_important_residues(scorer=scorer, bias=True, processes=1,
                                                             scw_scorer=scw_scorer)

    def test_seq2_bias_multi_process(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        scw_scorer.compute_background_w_and_w2_ave(processes=2)
        self.evaluate_score_clustering_of_important_residues(scorer=scorer, bias=True, processes=2,
                                                             scw_scorer=scw_scorer)

    def test_seq3_no_bias(self):
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=False)
        scw_scorer.compute_background_w_and_w2_ave(processes=2)
        self.evaluate_score_clustering_of_important_residues(scorer=scorer, bias=False, processes=2,
                                                             scw_scorer=scw_scorer)

    def test_seq3_bias(self):
        scorer = SinglePositionScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        scw_scorer.compute_background_w_and_w2_ave(processes=2)
        self.evaluate_score_clustering_of_important_residues(scorer=scorer, bias=True, processes=2,
                                                             scw_scorer=scw_scorer)

    def test_fail_bias_and_scw_mismatched(self):
        scorer = SinglePositionScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                                      chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        scw_scorer.compute_background_w_and_w2_ave(processes=2)
        with self.assertRaises(AssertionError):
            self.evaluate_score_clustering_of_important_residues(scorer=scorer, bias=False, processes=2,
                                                                 scw_scorer=scw_scorer)