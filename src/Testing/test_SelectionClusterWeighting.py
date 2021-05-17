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
from Evaluation.SelectionClusterWeighting import (SelectionClusterWeighting, init_compute_w_and_w2_ave_sub,
                                                  compute_w_and_w2_ave_sub)
from Testing.test_Base import (protein_seq1, protein_seq2, protein_seq3, write_out_temp_fn, protein_aln)
from Testing.test_contactScorer import (et_calcDist, CONTACT_DISTANCE2, pro_str, pro_pdb1, pro_pdb_1_alt_locs,
                                        pro_pdb1_scramble, pro_pdb2, pro_pdb2_scramble, pro_pdb_full,
                                        pro_pdb_full_scramble, aln_fn, protein_aln1, protein_aln2, protein_aln3)


def et_computeAdjacency(chain, mapping):
    """Compute the pairs of contacting residues
    A(i,j) implemented as a hash of hash of residue numbers"""
    three2one = {
        "ALA": 'A',
        "ARG": 'R',
        "ASN": 'N',
        "ASP": 'D',
        "CYS": 'C',
        "GLN": 'Q',
        "GLU": 'E',
        "GLY": 'G',
        "HIS": 'H',
        "ILE": 'I',
        "LEU": 'L',
        "LYS": 'K',
        "MET": 'M',
        "PHE": 'F',
        "PRO": 'P',
        "SER": 'S',
        "THR": 'T',
        "TRP": 'W',
        "TYR": 'Y',
        "VAL": 'V',
        "A": "A",
        "G": "G",
        "T": "T",
        "U": "U",
        "C": "C", }

    ResAtoms = {}
    print(chain)
    for residue in chain:
        try:
            aa = three2one[residue.get_resname()]
        except KeyError:
            continue
        # resi = residue.get_id()[1]
        resi = mapping[residue.get_id()[1]]
        for atom in residue:
            try:
                # ResAtoms[resi - 1].append(atom.coord)
                ResAtoms[resi].append(atom.coord)
            except KeyError:
                # ResAtoms[resi - 1] = [atom.coord]
                ResAtoms[resi] = [atom.coord]
    A = {}
    for resi in ResAtoms.keys():
        for resj in ResAtoms.keys():
            if resi < resj:
                curr_dist = et_calcDist(ResAtoms[resi], ResAtoms[resj])
                if curr_dist < CONTACT_DISTANCE2:
                    try:
                        A[resi][resj] = 1
                    except KeyError:
                        A[resi] = {resj: 1}
    return A, ResAtoms


def et_calc_w2_sub_problems(A, bias=1):
    """Calculate w2_ave components for calculation z-score (z_S) for residue selection reslist=[1,2,...]
    z_S = (w-<w>_S)/sigma_S
    The steps are:
    1. Calculate Selection Clustering Weight (SCW) 'w'
    2. Calculate mean SCW (<w>_S) in the ensemble of random
    selections of len(reslist) residues
    3. Calculate mean square SCW (<w^2>_S) and standard deviation (sigma_S)
    Reference: Mihalek, Res, Yao, Lichtarge (2003)

    reslist - a list of int's of protein residue numbers, e.g. ET residues
    L - length of protein
    A - the adjacency matrix implemented as a dictionary. The first key is related to the second key by resi<resj.
    bias - option to calculate with bias or nobias (j-i factor)"""
    part1 = 0.0
    part2 = 0.0
    part3 = 0.0
    if bias == 1:
        for resi, neighborsj in A.items():
            for resj in neighborsj:
                for resk, neighborsl in A.items():
                    for resl in neighborsl:
                        if (resi == resk and resj == resl) or \
                                (resi == resl and resj == resk):
                            part1 += (resj - resi) * (resl - resk)
                        elif (resi == resk) or (resj == resl) or \
                                (resi == resl) or (resj == resk):
                            part2 += (resj - resi) * (resl - resk)
                        else:
                            part3 += (resj - resi) * (resl - resk)
    elif bias == 0:
        for resi, neighborsj in A.items():
            for resj in neighborsj:
                for resk, neighborsl in A.items():
                    for resl in neighborsl:
                        if (resi == resk and resj == resl) or \
                                (resi == resl and resj == resk):
                            part1 += 1
                        elif (resi == resk) or (resj == resl) or \
                                (resi == resl) or (resj == resk):
                            part2 += 1
                        else:
                            part3 += 1
    return part1, part2, part3


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
        self.assertTrue(scw_obj.biased)
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
        self.assertFalse(scw_obj.biased)
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


class TestComputeWAndW2Ave(TestCase):

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

    def evaluate_compute_w_and_w2_ave(self, distances, pdb_residues, biased):
        for i in range(len(pdb_residues)):
            curr_w_ave_pre = 0
            curr_case1 = 0
            curr_case2 = 0
            curr_case3 = 0
            for j in range(i + 1, len(pdb_residues)):
                distance = distances[i, j]
                if distance >= 4.0:
                    continue
                if biased:
                    bias_term = abs(pdb_residues[i] - pdb_residues[j])
                else:
                    bias_term = 1
                curr_w_ave_pre += bias_term
                for k in range(len(pdb_residues)):
                    for l in range(k + 1, len(pdb_residues)):
                        if biased:
                            bias_term2 = abs(pdb_residues[k] - pdb_residues[l])
                        else:
                            bias_term2 = 1
                        final_bias = bias_term * bias_term2
                        pair_overlap = len({i, j}.intersection({k, l}))
                        if pair_overlap == 2:
                            curr_case1 += final_bias
                        elif pair_overlap == 1:
                            curr_case2 += final_bias
                        elif pair_overlap == 0:
                            curr_case3 += final_bias
                        else:
                            raise ValueError('Impossible pair')
            init_compute_w_and_w2_ave_sub(dists=distances, structure_res_num=pdb_residues, bias_bool=biased)
            cases = compute_w_and_w2_ave_sub(res_i=i)
            self.assertEqual(cases['w_ave_pre'], curr_w_ave_pre)
            self.assertEqual(cases['Case1'], curr_case1)
            self.assertEqual(cases['Case2'], curr_case2)
            self.assertEqual(cases['Case3'], curr_case3)

    def test_seq2_no_bias(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        self.evaluate_compute_w_and_w2_ave(distances=scorer.distances, biased=False,
                                           pdb_residues=scorer.query_pdb_mapper.pdb_ref.residue_pos[scorer.query_pdb_mapper.best_chain])

    def test_seq2_bias(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        self.evaluate_compute_w_and_w2_ave(distances=scorer.distances, biased=True,
                                           pdb_residues=scorer.query_pdb_mapper.pdb_ref.residue_pos[scorer.query_pdb_mapper.best_chain])

    def test_seq3_no_bias(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        self.evaluate_compute_w_and_w2_ave(distances=scorer.distances, biased=False,
                                           pdb_residues=scorer.query_pdb_mapper.pdb_ref.residue_pos[scorer.query_pdb_mapper.best_chain])

    def test_seq3_bias(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        self.evaluate_compute_w_and_w2_ave(distances=scorer.distances, biased=True,
                                           pdb_residues=scorer.query_pdb_mapper.pdb_ref.residue_pos[scorer.query_pdb_mapper.best_chain])


class TestComputeBackgroundWAndW2Ave(TestCase):

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

    def evaluate_compute_background_w_and_w2_ave(self, scw_scorer, processes):
        scw_scorer.compute_background_w_and_w2_ave(processes=processes)
        best_chain = None
        print(scw_scorer.query_pdb_mapper.pdb_ref.structure)
        for model in scw_scorer.query_pdb_mapper.pdb_ref.structure:
            print(model)
            for chain in model:
                print(chain)
                if chain.id == scw_scorer.query_pdb_mapper.best_chain:
                    best_chain = chain
                    break
        if best_chain is None:
            raise ValueError('Best Chain Never Initialized')
        adj, res_atoms = et_computeAdjacency(chain=best_chain,
                                             mapping={res: i for i, res in enumerate(scw_scorer.query_pdb_mapper.pdb_ref.pdb_residue_list[scw_scorer.query_pdb_mapper.best_chain])})
        case1, case2, case3 = et_calc_w2_sub_problems(adj, bias=1 if scw_scorer.biased else 0)
        self.assertEqual(scw_scorer.w_and_w2_ave_sub['Case1'], case1)
        self.assertEqual(scw_scorer.w_and_w2_ave_sub['Case2'], case2)
        self.assertEqual(scw_scorer.w_and_w2_ave_sub['Case3'], case3)

    def test_seq2_no_bias_single_process(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=False)
        self.evaluate_compute_background_w_and_w2_ave(scw_scorer=scw_scorer, processes=1)

    def test_seq2_no_bias_multi_process(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=False)
        self.evaluate_compute_background_w_and_w2_ave(scw_scorer=scw_scorer, processes=2)

    def test_seq2_bias_single_process(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        self.evaluate_compute_background_w_and_w2_ave(scw_scorer=scw_scorer, processes=1)

    def test_seq2_bias_multi_process(self):
        scorer = ContactScorer(query='seq2', seq_alignment=protein_aln2, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        self.evaluate_compute_background_w_and_w2_ave(scw_scorer=scw_scorer, processes=2)

    def test_seq3_no_bias(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=False)
        self.evaluate_compute_background_w_and_w2_ave(scw_scorer=scw_scorer, processes=2)

    def test_seq3_bias(self):
        scorer = ContactScorer(query='seq3', seq_alignment=protein_aln3, pdb_reference=self.pdb_chain_b,
                               cutoff=20.0, chain='B')
        scorer.fit()
        scorer.measure_distance('Any')
        scw_scorer = SelectionClusterWeighting(seq_pdb_map=scorer.query_pdb_mapper, pdb_dists=scorer.distances,
                                               biased=True)
        self.evaluate_compute_background_w_and_w2_ave(scw_scorer=scw_scorer, processes=2)
# class TestContactScorer(TestBase):
#
#
#     @staticmethod
#     def _et_calcZScore(reslist, L, A, bias=1):
#         """Calculate z-score (z_S) for residue selection reslist=[1,2,...]
#         z_S = (w-<w>_S)/sigma_S
#         The steps are:
#         1. Calculate Selection Clustering Weight (SCW) 'w'
#         2. Calculate mean SCW (<w>_S) in the ensemble of random
#         selections of len(reslist) residues
#         3. Calculate mean square SCW (<w^2>_S) and standard deviation (sigma_S)
#         Reference: Mihalek, Res, Yao, Lichtarge (2003)
#
#         reslist - a list of int's of protein residue numbers, e.g. ET residues
#         L - length of protein
#         A - the adjacency matrix implemented as a dictionary. The first key is related to the second key by resi<resj.
#         bias - option to calculate with bias or nobias (j-i factor)"""
#         w = 0
#         if bias == 1:
#             for resi in reslist:
#                 for resj in reslist:
#                     if resi < resj:
#                         try:
#                             Aij = A[resi][resj]  # A(i,j)==1
#                             w += (resj - resi)
#                         except KeyError:
#                             pass
#         elif bias == 0:
#             for resi in reslist:
#                 for resj in reslist:
#                     if resi < resj:
#                         try:
#                             Aij = A[resi][resj]  # A(i,j)==1
#                             w += 1
#                         except KeyError:
#                             pass
#         M = len(reslist)
#         pi1 = M * (M - 1.0) / (L * (L - 1.0))
#         pi2 = pi1 * (M - 2.0) / (L - 2.0)
#         pi3 = pi2 * (M - 3.0) / (L - 3.0)
#         w_ave = 0
#         w2_ave = 0
#         cases = {'Case1': 0, 'Case2': 0, 'Case3': 0}
#         if bias == 1:
#             for resi, neighborsj in A.items():
#                 for resj in neighborsj:
#                     w_ave += (resj - resi)
#                     for resk, neighborsl in A.items():
#                         for resl in neighborsl:
#                             if (resi == resk and resj == resl) or \
#                                     (resi == resl and resj == resk):
#                                 w2_ave += pi1 * (resj - resi) * (resl - resk)
#                                 cases['Case1'] += (resj - resi) * (resl - resk)
#                             elif (resi == resk) or (resj == resl) or \
#                                     (resi == resl) or (resj == resk):
#                                 w2_ave += pi2 * (resj - resi) * (resl - resk)
#                                 cases['Case2'] += (resj - resi) * (resl - resk)
#                             else:
#                                 w2_ave += pi3 * (resj - resi) * (resl - resk)
#                                 cases['Case3'] += (resj - resi) * (resl - resk)
#         elif bias == 0:
#             for resi, neighborsj in A.items():
#                 w_ave += len(neighborsj)
#                 for resj in neighborsj:
#                     for resk, neighborsl in A.items():
#                         for resl in neighborsl:
#                             if (resi == resk and resj == resl) or \
#                                     (resi == resl and resj == resk):
#                                 w2_ave += pi1
#                                 cases['Case1'] += 1
#                             elif (resi == resk) or (resj == resl) or \
#                                     (resi == resl) or (resj == resk):
#                                 w2_ave += pi2
#                                 cases['Case2'] += 1
#                             else:
#                                 w2_ave += pi3
#                                 cases['Case3'] += 1
#         w_ave = w_ave * pi1
#         # print('EXPECTED M: ', M)
#         # print('EXPECTED L: ', L)
#         # print('EXPECTED W: ', w)
#         # print('EXPECTED RES LIST: ', sorted(reslist))
#         # print('EXPECTED W_AVE: ', w_ave)
#         # print('EXPECTED W_AVE^2: ', (w_ave * w_ave))
#         # print('EXPECTED W^2_AVE: ', w2_ave)
#         # print('EXPECTED DIFF: ', w2_ave - w_ave * w_ave)
#         # print('EXPECTED DIFF2: ', w2_ave - (w_ave * w_ave))
#         sigma = math.sqrt(w2_ave - w_ave * w_ave)
#         if sigma == 0:
#             return M, L, pi1, pi2, pi3, 'NA', w, w_ave, w2_ave, sigma, cases
#         return M, L, pi1, pi2, pi3, (w - w_ave) / sigma, w, w_ave, w2_ave, sigma, cases
#
#     def all_z_scores(self, mapping, special_mapping, A, L, bias, res_i, res_j, scores):
#         data = {'Res_i': res_i, 'Res_j': res_j, 'Covariance_Score': scores, 'Z-Score': [], 'W': [], 'W_Ave': [],
#                 'W2_Ave': [], 'Sigma': [], 'Num_Residues': []}
#         res_list = []
#         res_set = set()
#         prev_size = 0
#         prev_score = None
#         for i in range(len(scores)):
#             # curr_i = res_i[i] + 1
#             curr_i = res_i[i]
#             # curr_j = res_j[i] + 1
#             curr_j = res_j[i]
#             if (curr_i not in A) or (curr_j not in A):
#                 score_data = (None, None, None, None, None, None, '-', None, None, None, None, None)
#             else:
#                 if curr_i not in res_set:
#                     res_list.append(curr_i)
#                     res_set.add(curr_i)
#                 if curr_j not in res_set:
#                     res_list.append(curr_j)
#                     res_set.add(curr_j)
#                 if len(res_set) == prev_size:
#                     score_data = prev_score
#                 else:
#                     # score_data = self._et_calcZScore(reslist=[special_mapping[res] for res in res_list], L=L, A=A, bias=bias)
#                     score_data = self._et_calcZScore(reslist=res_list, L=L, A=A, bias=bias)
#             data['Z-Score'].append(score_data[5])
#             data['W'].append(score_data[6])
#             data['W_Ave'].append(score_data[7])
#             data['W2_Ave'].append(score_data[8])
#             data['Sigma'].append(score_data[9])
#             data['Num_Residues'].append(len(res_list))
#             prev_size = len(res_set)
#             prev_score = score_data
#         return pd.DataFrame(data)
#
#     def evaluate_compute_w2_ave_sub(self, scorer):
#         scorer.fit()
#         scorer.measure_distance(method='Any')
#         recip_map = {v: k for k, v in scorer.query_pdb_mapping.items()}
#         struc_seq_map = {k: i for i, k in enumerate(scorer.query_structure.pdb_residue_list[scorer.best_chain])}
#         final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
#         expected_adjacency, res_atoms = self._et_computeAdjacency(
#             scorer.query_structure.structure[0][scorer.best_chain], mapping=final_map)
#         # Test biased SCW Z-Score component
#         init_compute_w2_ave_sub(dists=scorer.distances, bias_bool=True)
#         cases_biased = {}
#         for i in range(scorer.distances.shape[0]):
#             curr_cases = compute_w2_ave_sub(i)
#             for k in curr_cases:
#                 if k not in cases_biased:
#                     cases_biased[k] = 0
#                 cases_biased[k] += curr_cases[k]
#         expected_w2_biased = self._et_calc_w2_sub_problems(A=expected_adjacency, bias=1)
#         self.assertEqual(cases_biased['Case1'], expected_w2_biased[0])
#         self.assertEqual(cases_biased['Case2'], expected_w2_biased[1])
#         self.assertEqual(cases_biased['Case3'], expected_w2_biased[2])
#         # Test biased SCW Z-Score component
#         init_compute_w2_ave_sub(dists=scorer.distances, bias_bool=False)
#         cases_unbiased = {}
#         for i in range(scorer.distances.shape[0]):
#             curr_cases = compute_w2_ave_sub(i)
#             for k in curr_cases:
#                 if k not in cases_unbiased:
#                     cases_unbiased[k] = 0
#                 cases_unbiased[k] += curr_cases[k]
#         expected_w2_unbiased = self._et_calc_w2_sub_problems(A=expected_adjacency, bias=0)
#         self.assertEqual(cases_unbiased['Case1'], expected_w2_unbiased[0])
#         self.assertEqual(cases_unbiased['Case2'], expected_w2_unbiased[1])
#         self.assertEqual(cases_unbiased['Case3'], expected_w2_unbiased[2])
#
#     def test_16a_compute_w2_ave_sub(self):
#         self.evaluate_compute_w2_ave_sub(scorer=self.scorer1)
#
#     def test_16b_compute_w2_ave_sub(self):
#         self.evaluate_compute_w2_ave_sub(scorer=self.scorer2)
#
#     def evaluate_clustering_z_scores(self, scorer):
#         scorer.fit()
#         scorer.measure_distance(method='Any')
#         recip_map = {v: k for k, v in scorer.query_pdb_mapping.items()}
#         struc_seq_map = {k: i for i, k in enumerate(scorer.query_structure.pdb_residue_list[scorer.best_chain])}
#         final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
#         expected_adjacency, res_atoms = self._et_computeAdjacency(
#             scorer.query_structure.structure[0][scorer.best_chain], mapping=final_map)
#         residue_list = list(scorer.query_pdb_mapping.keys())
#         shuffle(residue_list)
#         # Test unbiased SCW Z-Score computation
#         init_compute_w2_ave_sub(dists=scorer.distances, bias_bool=False)
#         cases_unbiased = {}
#         for i in range(scorer.distances.shape[0]):
#             curr_cases = compute_w2_ave_sub(i)
#             for k in curr_cases:
#                 if k not in cases_unbiased:
#                     cases_unbiased[k] = 0
#                 cases_unbiased[k] += curr_cases[k]
#         init_clustering_z_score(bias_bool=False, w2_ave_sub_dict=cases_unbiased, curr_pdb=scorer.query_structure,
#                                 map_to_structure=scorer.query_pdb_mapping, residue_dists=scorer.distances,
#                                 best_chain=scorer.best_chain)
#         for i in range(len(residue_list)):
#             curr_residues = residue_list[:(i + 1)]
#             a, m, l, pi1, pi2, pi3, z_score, w, w_ave, w2_ave, sigma, num_residues = clustering_z_score(curr_residues)
#             em, el, epi1, epi2, epi3, e_z_score, e_w, e_w_ave, e_w2_ave, e_sigma, e_cases = self._et_calcZScore(
#                 reslist=curr_residues, L=len(scorer.query_structure.seq[scorer.best_chain]),
#                 A=expected_adjacency, bias=False)
#             for res_i in expected_adjacency:
#                 for res_j in expected_adjacency[res_i]:
#                     self.assertEqual(a[scorer.query_pdb_mapping[res_i], scorer.query_pdb_mapping[res_j]],
#                                      expected_adjacency[res_i][res_j])
#                     a[scorer.query_pdb_mapping[res_i], scorer.query_pdb_mapping[res_j]] = 0
#             self.assertEqual(m, em)
#             self.assertEqual(l, el)
#             self.assertLess(np.abs(pi1 - epi1), 1E-16)
#             self.assertLess(np.abs(pi2 - epi2), 1E-16)
#             self.assertLess(np.abs(pi3 - epi3), 1E-16)
#             self.assertEqual(num_residues, len(curr_residues))
#             self.assertLess(np.abs(w - e_w), 1E-16, '{} vs {}'.format(w, e_w))
#             self.assertLess(np.abs(w_ave - e_w_ave), 1E-16, '{} vs {}'.format(w_ave, e_w_ave))
#             for case in e_cases:
#                 self.assertEqual(cases_unbiased[case], e_cases[case])
#             self.assertLess(np.abs(w2_ave - e_w2_ave), 1E-4, '{} vs {}'.format(w2_ave, e_w2_ave))
#             composed_w2_ave = ((pi1 * cases_unbiased['Case1']) + (pi2 * cases_unbiased['Case2']) +
#                                (pi3 * cases_unbiased['Case3']))
#             expected_composed_w2_ave = ((epi1 * e_cases['Case1']) + (epi2 * e_cases['Case2']) +
#                                         (epi3 * e_cases['Case3']))
#             self.assertLess(np.abs(composed_w2_ave - expected_composed_w2_ave), 1E-16)
#             self.assertLess(np.abs(sigma - e_sigma), 1E-5, '{} vs {}'.format(sigma, e_sigma))
#             expected_composed_sigma = math.sqrt(expected_composed_w2_ave - e_w_ave * e_w_ave)
#             self.assertLess(np.abs(sigma - expected_composed_sigma), 1E-16)
#             if isinstance(z_score, str):
#                 self.assertTrue(isinstance(e_z_score, str))
#                 self.assertEqual(z_score, e_z_score, '{} vs {}'.format(z_score, e_z_score))
#             else:
#                 if z_score < 0:
#                     self.assertTrue(e_z_score < 0)
#                 else:
#                     self.assertFalse(e_z_score < 0)
#                 self.assertLess(np.abs(z_score - e_z_score), 1E-6, '{} vs {}'.format(z_score, e_z_score))
#                 expected_composed_z_score = (e_w - e_w_ave) / expected_composed_sigma
#                 self.assertLess(np.abs(z_score - expected_composed_z_score), 1E-16)
#         # Test biased SCW Z-Score computation
#         init_compute_w2_ave_sub(dists=scorer.distances, bias_bool=True)
#         cases_biased = {}
#         for i in range(scorer.distances.shape[0]):
#             curr_cases = compute_w2_ave_sub(i)
#             for k in curr_cases:
#                 if k not in cases_biased:
#                     cases_biased[k] = 0
#                 cases_biased[k] += curr_cases[k]
#         init_clustering_z_score(bias_bool=True, w2_ave_sub_dict=cases_biased, curr_pdb=scorer.query_structure,
#                                 map_to_structure=scorer.query_pdb_mapping, residue_dists=scorer.distances,
#                                 best_chain=scorer.best_chain)
#         for i in range(len(residue_list)):
#             curr_residues = residue_list[:(i + 1)]
#             a, m, l, pi1, pi2, pi3, z_score, w, w_ave, w2_ave, sigma, num_residues = clustering_z_score(curr_residues)
#             em, el, epi1, epi2, epi3, e_z_score, e_w, e_w_ave, e_w2_ave, e_sigma, e_cases = self._et_calcZScore(
#                 reslist=curr_residues, L=len(scorer.query_structure.seq[scorer.best_chain]),
#                 A=expected_adjacency, bias=True)
#             for res_i in expected_adjacency:
#                 for res_j in expected_adjacency[res_i]:
#                     self.assertEqual(a[scorer.query_pdb_mapping[res_i], scorer.query_pdb_mapping[res_j]],
#                                      expected_adjacency[res_i][res_j])
#                     a[scorer.query_pdb_mapping[res_i], scorer.query_pdb_mapping[res_j]] = 0
#             self.assertEqual(m, em)
#             self.assertEqual(l, el)
#             self.assertLess(np.abs(pi1 - epi1), 1E-16)
#             self.assertLess(np.abs(pi2 - epi2), 1E-16)
#             self.assertLess(np.abs(pi3 - epi3), 1E-16)
#             self.assertEqual(num_residues, len(curr_residues))
#             self.assertLess(np.abs(w - e_w), 1E-16, '{} vs {}'.format(w, e_w))
#             self.assertLess(np.abs(w_ave - e_w_ave), 1E-16, '{} vs {}'.format(w_ave, e_w_ave))
#             for case in e_cases:
#                 self.assertEqual(cases_biased[case], e_cases[case])
#             self.assertLess(np.abs(w2_ave - e_w2_ave), 1E-2, '{} vs {}'.format(w2_ave, e_w2_ave))
#             composed_w2_ave = ((pi1 * cases_biased['Case1']) + (pi2 * cases_biased['Case2']) +
#                                (pi3 * cases_biased['Case3']))
#             expected_composed_w2_ave = ((epi1 * e_cases['Case1']) + (epi2 * e_cases['Case2']) +
#                                         (epi3 * e_cases['Case3']))
#             self.assertLess(np.abs(composed_w2_ave - expected_composed_w2_ave), 1E-16)
#             self.assertLess(np.abs(sigma - e_sigma), 1E-4, '{} vs {}'.format(sigma, e_sigma))
#             expected_composed_sigma = math.sqrt(expected_composed_w2_ave - e_w_ave * e_w_ave)
#             self.assertLess(np.abs(sigma - expected_composed_sigma), 1E-16)
#             if isinstance(z_score, str):
#                 self.assertTrue(isinstance(e_z_score, str))
#                 self.assertEqual(z_score, e_z_score, '{} vs {}'.format(z_score, e_z_score))
#             else:
#                 if z_score < 0:
#                     self.assertTrue(e_z_score < 0)
#                 else:
#                     self.assertFalse(e_z_score < 0)
#                 self.assertLess(np.abs(z_score - e_z_score), 1E-4, '{} vs {}'.format(z_score, e_z_score))
#                 expected_composed_z_score = (e_w - e_w_ave) / expected_composed_sigma
#                 self.assertLess(np.abs(z_score - expected_composed_z_score), 1E-16)
#
#     def test_17a_clustering_z_scores(self):
#         self.evaluate_clustering_z_scores(scorer=self.scorer1)
#
#     def test_17b_clustering_z_scores(self):
#         self.evaluate_clustering_z_scores(scorer=self.scorer2)
#
#     def evaluate_score_clustering_of_contact_predictions(self, scorer, seq_len, bias):
#         # Initialize scorer and scores
#         scorer.fit()
#         scorer.measure_distance(method='Any')
#         scores = np.random.RandomState(1234567890).rand(scorer.query_alignment.seq_length,
#                                                         scorer.query_alignment.seq_length)
#         scores[np.tril_indices(scorer.query_alignment.seq_length, 1)] = 0
#         scores += scores.T
#         ranks, coverages = compute_rank_and_coverage(seq_len, scores, 2, 'min')
#         scorer.map_predictions_to_pdb(ranks=ranks, predictions=scores, coverages=coverages, threshold=0.5)
#         # Calculate Z-scores for the structure
#         start1 = time()
#         output_fn_1b = os.path.join(self.testing_dir, 'z_score1b.tsv')
#         zscore_df, _, _ = scorer.score_clustering_of_contact_predictions(bias=bias, file_path=output_fn_1b,
#                                                                             w2_ave_sub=None)
#         end1 = time()
#         print('Time for ContactScorer to compute SCW: {}'.format((end1 - start1) / 60.0))
#         # Check that the scoring file was written out to the expected file.
#         self.assertTrue(os.path.isfile(output_fn_1b))
#         os.remove(output_fn_1b)
#         # Generate data for calculating expected values
#         recip_map = {v: k for k, v in scorer.query_pdb_mapping.items()}
#         struc_seq_map = {k: i for i, k in enumerate(scorer.query_structure.pdb_residue_list[scorer.best_chain])}
#         final_map = {k: recip_map[v] for k, v in struc_seq_map.items()}
#         A, res_atoms = self._et_computeAdjacency(scorer.query_structure.structure[0][scorer.best_chain],
#                                                  mapping=final_map)
#         # Iterate over the returned data frame row by row and test whether the results are correct
#         visited_scorable_residues = set()
#         prev_len = 0
#         prev_stats = None
#         prev_composed_w2_ave = None
#         prev_composed_sigma = None
#         prev_composed_z_score = None
#         zscore_df[['Res_i', 'Res_j']] = zscore_df[['Res_i', 'Res_j']].astype(dtype=np.int64)
#         zscore_df[['Covariance_Score', 'W', 'W_Ave', 'W2_Ave', 'Sigma', 'Num_Residues']].replace([None, '-', 'NA'],
#                                                                                                  np.nan, inplace=True)
#         zscore_df[['Covariance_Score', 'W', 'W_Ave', 'W2_Ave', 'Sigma']] = zscore_df[
#             ['Covariance_Score', 'W', 'W_Ave', 'W2_Ave', 'Sigma']].astype(dtype=np.float64)
#         # print(zscore_df.dtypes)
#         for ind in zscore_df.index:
#             print('{}:{}'.format(ind, np.max(zscore_df.index)))
#             res_i = zscore_df.loc[ind, 'Res_i']
#             res_j = zscore_df.loc[ind, 'Res_j']
#             if (res_i in scorer.query_pdb_mapping) and (res_j in scorer.query_pdb_mapping):
#                 visited_scorable_residues.add(res_i)
#                 visited_scorable_residues.add(res_j)
#                 if len(visited_scorable_residues) > prev_len:
#                     curr_stats = self._et_calcZScore(
#                         reslist=sorted(visited_scorable_residues),
#                         L=len(scorer.query_structure.seq[scorer.best_chain]), A=A, bias=bias)
#                     expected_composed_w2_ave = ((curr_stats[2] * curr_stats[10]['Case1']) +
#                                                 (curr_stats[3] * curr_stats[10]['Case2']) +
#                                                 (curr_stats[4] * curr_stats[10]['Case3']))
#                     expected_composed_sigma = math.sqrt(expected_composed_w2_ave - curr_stats[7] * curr_stats[7])
#                     if expected_composed_sigma == 0.0:
#                         expected_composed_z_score = 'NA'
#                     else:
#                         expected_composed_z_score = (curr_stats[6] - curr_stats[7]) / expected_composed_sigma
#                     prev_len = len(visited_scorable_residues)
#                     prev_stats = curr_stats
#                     prev_composed_w2_ave = expected_composed_w2_ave
#                     prev_composed_sigma = expected_composed_sigma
#                     prev_composed_z_score = expected_composed_z_score
#                 else:
#                     curr_stats = prev_stats
#                     expected_composed_w2_ave = prev_composed_w2_ave
#                     expected_composed_sigma = prev_composed_sigma
#                     expected_composed_z_score = prev_composed_z_score
#                 error_message = '\nW: {}\nExpected W: {}\nW Ave: {}\nExpected W Ave: {}\nW2 Ave: {}\nExpected W2 Ave: '\
#                                 '{}\nComposed Expected W2 Ave: {}\nSigma: {}\nExpected Sigma: {}\nComposed Expected '\
#                                 'Sigma: {}\nZ-Score: {}\nExpected Z-Score: {}\nComposed Expected Z-Score: {}'.format(
#                     zscore_df.loc[ind, 'W'], curr_stats[6], zscore_df.loc[ind, 'W_Ave'], curr_stats[7],
#                     zscore_df.loc[ind, 'W2_Ave'], curr_stats[8], expected_composed_w2_ave,
#                     zscore_df.loc[ind, 'Sigma'], curr_stats[9], expected_composed_sigma,
#                     zscore_df.loc[ind, 'Z-Score'], curr_stats[5], expected_composed_z_score)
#                 self.assertEqual(zscore_df.loc[ind, 'Num_Residues'], len(visited_scorable_residues))
#                 self.assertLessEqual(np.abs(zscore_df.loc[ind, 'W'] - curr_stats[6]), 1E-16, error_message)
#                 self.assertLessEqual(np.abs(zscore_df.loc[ind, 'W_Ave'] - curr_stats[7]), 1E-16, error_message)
#                 self.assertLessEqual(np.abs(zscore_df.loc[ind, 'W2_Ave'] - expected_composed_w2_ave), 1E-16,
#                                      error_message)
#                 self.assertLessEqual(np.abs(zscore_df.loc[ind, 'W2_Ave'] - curr_stats[8]), 1E-2, error_message)
#                 self.assertLessEqual(np.abs(zscore_df.loc[ind, 'Sigma'] - expected_composed_sigma), 1E-16,
#                                      error_message)
#                 self.assertLessEqual(np.abs(zscore_df.loc[ind, 'Sigma'] - curr_stats[9]), 1E-5, error_message)
#                 if expected_composed_sigma == 0.0:
#                     self.assertEqual(zscore_df.loc[ind, 'Z-Score'], expected_composed_z_score)
#                     self.assertEqual(zscore_df.loc[ind, 'Z-Score'], curr_stats[5])
#                 else:
#                     self.assertLessEqual(np.abs(zscore_df.loc[ind, 'Z-Score'] - expected_composed_z_score), 1E-16,
#                                          error_message)
#                     self.assertLessEqual(np.abs(zscore_df.loc[ind, 'Z-Score'] - curr_stats[5]), 1E-5, error_message)
#             else:
#                 self.assertEqual(zscore_df.loc[ind, 'Z-Score'], '-')
#                 self.assertTrue(np.isnan(zscore_df.loc[ind, 'W']))
#                 self.assertTrue(np.isnan(zscore_df.loc[ind, 'W_Ave']))
#                 self.assertTrue(np.isnan(zscore_df.loc[ind, 'W2_Ave']))
#                 self.assertTrue(np.isnan(zscore_df.loc[ind, 'Sigma']))
#                 self.assertIsNone(zscore_df.loc[ind, 'Num_Residues'])
#             self.assertEqual(zscore_df.loc[ind, 'Covariance_Score'], coverages[res_i, res_j])
#
#     def test_18a_score_clustering_of_contact_predictions(self):
#         self.evaluate_score_clustering_of_contact_predictions(scorer=self.scorer1, seq_len=self.seq_len1, bias=True)
#
#     def test_18b_score_clustering_of_contact_predictions(self):
#         self.evaluate_score_clustering_of_contact_predictions(scorer=self.scorer1, seq_len=self.seq_len1, bias=False)
#
#     def test_18c_score_clustering_of_contact_predictions(self):
#         self.evaluate_score_clustering_of_contact_predictions(scorer=self.scorer2, seq_len=self.seq_len2, bias=True)
#
#     def test_18d_score_clustering_of_contact_predictions(self):
#         self.evaluate_score_clustering_of_contact_predictions(scorer=self.scorer2, seq_len=self.seq_len2, bias=False)