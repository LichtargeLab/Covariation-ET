import os
import datetime
import numpy as np
from time import time
from shutil import rmtree
from unittest import TestCase
from scipy.stats import rankdata
from Bio.Align import MultipleSeqAlignment
from sklearn.metrics import mutual_info_score
from ETMIPC import (ETMIPC, pool_init_sub_aln, generate_sub_alignment, pool_init_score, mip_score,
                    pool_init_calculate_branch_score, calculate_branch_score, pool_init_calculate_score_and_coverage,
                    calculate_score_and_coverage, pool_init_write_score, write_score, single_matrix_filename,
                    exists_single_matrix, save_single_matrix, load_single_matrix)
from SeqAlignment import SeqAlignment


class TestETMIPC(TestCase):

    def conservative_mip(self, alignment2Num):
        overallMMI = 0.0
        seq_length = alignment2Num.shape[1]
        # generate a MI matrix for each cluster
        MI_matrix = np.zeros([seq_length, seq_length])
        MMI = np.zeros([seq_length, 1])  # Vector of 1 column
        APC_matrix = np.zeros([seq_length, seq_length])
        MIP_matrix = np.zeros([seq_length, seq_length])
        for i in range(0, seq_length):
            MMI[i][0] = 0.0
            column_i = []
            column_j = []
            for j in range(0, seq_length):
                if i >= j:
                    continue
                column_i = [int(item[i]) for item in alignment2Num]
                column_j = [int(item[j]) for item in alignment2Num]
                MI_matrix[i][j] = mutual_info_score(
                    column_i, column_j, contingency=None)
                # AW: divides by individual entropies to normalize.
                MI_matrix[j][i] = MI_matrix[i][j]

        for i in range(0, seq_length):  # this is where we can do i, j by running a second loop
            for j in range(0, seq_length):
                if i != j:
                    MMI[i][0] += MI_matrix[i][j]
                    if i > j:
                        overallMMI += MI_matrix[i][j]
            MMI[i][0] = MMI[i][0] / (seq_length - 1)

        overallMMI = 2.0 * (overallMMI / (seq_length - 1)) / seq_length
        ####--------------------------------------------#####
        # Calculating APC
        ####--------------------------------------------#####
        for i in range(0, seq_length):
            for j in range(0, seq_length):
                if i == j:
                    continue
                APC_matrix[i][j] = (MMI[i][0] * MMI[j][0]) / overallMMI

        for i in range(0, seq_length):
            for j in range(0, seq_length):
                if i == j:
                    continue
                MIP_matrix[i][j] = MI_matrix[i][j] - APC_matrix[i][j]
        return MIP_matrix

    def calculate_coverage(self, mat):
        coverage = np.zeros(mat.shape)
        test_mat = np.triu(mat)
        mask = np.triu(np.ones(mat.shape), k=1)
        normalization = ((mat.shape[0] ** 2 - mat.shape[0]) / 2.0)
        for i in range(mat.shape[0]):
            for j in range(i + 1, mat.shape[0]):
                bool_mat = (test_mat[i, j] >= test_mat) * 1.0
                corrected_mat = bool_mat * mask
                compute_coverage2 = (((np.sum(corrected_mat) - 1) * 100) / normalization)
                coverage[i, j] = coverage[j, i] = compute_coverage2
        return coverage

    def test___init__(self):
        with self.assertRaises(TypeError):
            ETMIPC()
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        self.assertEqual(etmipc1.alignment, '../Test/1c17A.fa')
        self.assertIsNone(etmipc1.tree_depth)
        self.assertIsNone(etmipc1.nongap_counts)
        self.assertIsNone(etmipc1.unique_clusters)
        self.assertIsNone(etmipc1.cluster_scores)
        self.assertIsNone(etmipc1.branch_scores)
        self.assertIsNone(etmipc1.scores)
        self.assertIsNone(etmipc1.coverage)
        self.assertIsNone(etmipc1.time)
        self.assertIsNone(etmipc1.processes)
        self.assertIsNone(etmipc1.low_mem)
        self.assertIsNone(etmipc1.output_dir)
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        self.assertEqual(etmipc2.alignment, '../Test/1h1vA.fa')
        self.assertIsNone(etmipc2.tree_depth)
        self.assertIsNone(etmipc2.nongap_counts)
        self.assertIsNone(etmipc2.unique_clusters)
        self.assertIsNone(etmipc2.cluster_scores)
        self.assertIsNone(etmipc2.branch_scores)
        self.assertIsNone(etmipc2.scores)
        self.assertIsNone(etmipc2.coverage)
        self.assertIsNone(etmipc2.time)
        self.assertIsNone(etmipc2.processes)
        self.assertIsNone(etmipc2.low_mem)
        self.assertIsNone(etmipc2.output_dir)

    def test_get_sub_alignment(self):
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = os.path.abspath('../Test/')
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 6
        etmipc1._generate_sub_alignments()
        for tree_position in etmipc1.unique_clusters:
            self.assertEqual(etmipc1.get_sub_alignment(branch=tree_position[0], cluster=tree_position[1]),
                             etmipc1.unique_clusters[tree_position]['sub_alignment'])
            for tree_position2 in etmipc1.unique_clusters[tree_position]['tree_positions']:
                self.assertEqual(etmipc1.get_sub_alignment(branch=tree_position[0], cluster=tree_position[1]),
                                 etmipc1.get_sub_alignment(branch=tree_position2[0], cluster=tree_position2[1]))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = os.path.abspath('../Test/')
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 6
        etmipc2._generate_sub_alignments()
        for tree_position in etmipc2.unique_clusters:
            self.assertEqual(etmipc2.get_sub_alignment(branch=tree_position[0], cluster=tree_position[1]),
                             etmipc2.unique_clusters[tree_position]['sub_alignment'])
            for tree_position2 in etmipc2.unique_clusters[tree_position]['tree_positions']:
                self.assertEqual(etmipc2.get_sub_alignment(branch=tree_position[0], cluster=tree_position[1]),
                                 etmipc2.get_sub_alignment(branch=tree_position2[0], cluster=tree_position2[1]))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))

    # def test__get_c_level_matrices(self):

    def test_get_nongap_counts(self):
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        out_dir = os.path.abspath('../Test/')
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = out_dir
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 1
        etmipc1.low_mem = False
        etmipc1.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
        total_dict_non_gap_counts = etmipc1.get_nongap_counts()
        for branch in etmipc1.tree_depth:
            dict_non_gap_counts = etmipc1.get_nongap_counts(branch=branch, three_dim=False)
            three_d_non_gap_counts = etmipc1.get_nongap_counts(branch=branch, three_dim=True)
            for cluster in range(branch):
                curr_nongap_counts = etmipc1.get_nongap_counts(branch=branch, cluster=cluster)
                self.assertEqual(np.sum(curr_nongap_counts -
                                 etmipc1.unique_clusters[etmipc1.cluster_mapping[(branch, cluster)]]['nongap_counts']), 0)
                self.assertEqual(np.sum(curr_nongap_counts - total_dict_non_gap_counts[branch][cluster]), 0)
                self.assertEqual(np.sum(curr_nongap_counts - dict_non_gap_counts[cluster]), 0)
                self.assertEqual(np.sum(curr_nongap_counts - three_d_non_gap_counts[cluster, :, :]), 0)
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = out_dir
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 6
        etmipc2.low_mem = True
        etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
        total_dict_non_gap_counts = etmipc2.get_nongap_counts()
        for branch in etmipc2.tree_depth:
            dict_non_gap_counts = etmipc2.get_nongap_counts(branch=branch, three_dim=False)
            three_d_non_gap_counts = etmipc2.get_nongap_counts(branch=branch, three_dim=True)
            for cluster in range(branch):
                curr_nongap_counts = etmipc2.get_nongap_counts(branch=branch, cluster=cluster)
                self.assertEqual(np.sum(curr_nongap_counts - np.load(etmipc2.nongap_counts[branch][cluster])['mat']), 0)
                self.assertEqual(np.sum(curr_nongap_counts - total_dict_non_gap_counts[branch][cluster]), 0)
                self.assertEqual(np.sum(curr_nongap_counts - dict_non_gap_counts[cluster]), 0)
                self.assertEqual(np.sum(curr_nongap_counts - three_d_non_gap_counts[cluster, :, :]), 0)
        for k in etmipc2.tree_depth:
            rmtree(os.path.join(out_dir, str(k)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))

    def test_get_cluster_scores(self):
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        out_dir = os.path.abspath('../Test/')
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = out_dir
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 1
        etmipc1.low_mem = False
        etmipc1.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
        total_dict_cluster_scores = etmipc1.get_cluster_scores()
        for branch in etmipc1.tree_depth:
            dict_cluster_scores = etmipc1.get_cluster_scores(branch=branch, three_dim=False)
            three_d_cluster_scores = etmipc1.get_cluster_scores(branch=branch, three_dim=True)
            for cluster in range(branch):
                curr_cluster_scores = etmipc1.get_cluster_scores(branch=branch, cluster=cluster)
                self.assertEqual(np.sum(curr_cluster_scores -
                                        etmipc1.unique_clusters[etmipc1.cluster_mapping[(branch, cluster)]][
                                            'cluster_scores']), 0)
                self.assertEqual(np.sum(curr_cluster_scores - total_dict_cluster_scores[branch][cluster]), 0)
                self.assertEqual(np.sum(curr_cluster_scores - dict_cluster_scores[cluster]), 0)
                self.assertEqual(np.sum(curr_cluster_scores - three_d_cluster_scores[cluster, :, :]), 0)
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = out_dir
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 6
        etmipc2.low_mem = True
        etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
        total_dict_cluster_scores = etmipc2.get_cluster_scores()
        for branch in etmipc2.tree_depth:
            dict_cluster_scores = etmipc2.get_cluster_scores(branch=branch, three_dim=False)
            three_d_cluster_scores = etmipc2.get_cluster_scores(branch=branch, three_dim=True)
            for cluster in range(branch):
                curr_cluster_scores = etmipc2.get_cluster_scores(branch=branch, cluster=cluster)
                self.assertEqual(np.sum(curr_cluster_scores - np.load(etmipc2.cluster_scores[branch][cluster])['mat']), 0)
                self.assertEqual(np.sum(curr_cluster_scores - total_dict_cluster_scores[branch][cluster]), 0)
                self.assertEqual(np.sum(curr_cluster_scores - dict_cluster_scores[cluster]), 0)
                self.assertEqual(np.sum(curr_cluster_scores - three_d_cluster_scores[cluster, :, :]), 0)
        for k in etmipc2.tree_depth:
            rmtree(os.path.join(out_dir, str(k)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))

    def test_get_branch_scores(self):
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        out_dir = os.path.abspath('../Test/')
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = out_dir
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 1
        etmipc1.low_mem = False
        etmipc1.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
        etmipc1.calculate_branch_scores(combine_clusters='sum')
        total_dict_branch_scores = etmipc1.get_branch_scores()
        three_d_branch_scores = etmipc1.get_branch_scores(three_dim=True)
        for branch in etmipc1.tree_depth:
            curr_branch_scores = etmipc1.get_branch_scores(branch=branch, three_dim=False)
            self.assertEqual(np.sum(curr_branch_scores - etmipc1.branch_scores[branch]), 0)
            self.assertEqual(np.sum(curr_branch_scores - total_dict_branch_scores[branch]), 0)
            self.assertEqual(np.sum(curr_branch_scores - three_d_branch_scores[etmipc1.tree_depth.index(branch), :, :]),
                             0)
        os.remove(os.path.join(out_dir, 'alignment.pkl'))
        os.remove(os.path.join(out_dir, 'ungapped_alignment.pkl'))
        os.remove(os.path.join(out_dir, 'UngappedAlignment.fa'))
        os.remove(os.path.join(out_dir, 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = out_dir
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 6
        etmipc2.low_mem = True
        etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
        etmipc2.calculate_branch_scores(combine_clusters='evidence_weighted')
        total_dict_branch_scores = etmipc2.get_branch_scores()
        three_d_branch_scores = etmipc2.get_branch_scores(three_dim=True)
        for branch in etmipc2.tree_depth:
            curr_branch_scores = etmipc2.get_branch_scores(branch=branch)
            self.assertEqual(np.sum(curr_branch_scores - np.load(etmipc2.branch_scores[branch])['mat']), 0)
            self.assertEqual(np.sum(curr_branch_scores - total_dict_branch_scores[branch]), 0)
            self.assertEqual(np.sum(curr_branch_scores - three_d_branch_scores[etmipc1.tree_depth.index(branch), :, :]),
                             0)
            rmtree(os.path.join(out_dir, str(branch)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))

    def test_get_scores_and_coverage(self):
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        out_dir = os.path.abspath('../Test/')
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = out_dir
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 1
        etmipc1.low_mem = False
        etmipc1.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
        etmipc1.calculate_branch_scores(combine_clusters='sum')
        etmipc1.calculate_final_scores(combine_branches='sum')
        total_dict_scores = etmipc1.get_scores()
        total_dict_coverages = etmipc1.get_coverage()
        three_d_scores = etmipc1.get_scores(three_dim=True)
        three_d_coverages = etmipc1.get_coverage(three_dim=True)
        for branch in etmipc1.tree_depth:
            curr_scores = etmipc1.get_scores(branch=branch, three_dim=False)
            curr_coverage = etmipc1.get_coverage(branch=branch, three_dim=False)
            self.assertEqual(np.sum(curr_scores - etmipc1.scores[branch]), 0)
            self.assertEqual(np.sum(curr_coverage - etmipc1.coverage[branch]), 0)
            self.assertEqual(np.sum(curr_scores - total_dict_scores[branch]), 0)
            self.assertEqual(np.sum(curr_coverage - total_dict_coverages[branch]), 0)
            self.assertEqual(np.sum(curr_scores - three_d_scores[etmipc1.tree_depth.index(branch), :, :]), 0)
            self.assertEqual(np.sum(curr_coverage - three_d_coverages[etmipc1.tree_depth.index(branch), :, :]), 0)
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = out_dir
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 6
        etmipc2.low_mem = True
        etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
        etmipc2.calculate_branch_scores(combine_clusters='evidence_weighted')
        etmipc2.calculate_final_scores(combine_branches='average')
        total_dict_scores = etmipc2.get_scores()
        total_dict_coverage = etmipc2.get_coverage()
        three_d_scores = etmipc2.get_scores(three_dim=True)
        three_d_coverage = etmipc2.get_coverage(three_dim=True)
        for branch in etmipc2.tree_depth:
            curr_scores = etmipc2.get_scores(branch=branch)
            curr_coverage = etmipc2.get_coverage(branch=branch)
            self.assertEqual(np.sum(curr_scores - np.load(etmipc2.scores[branch])['mat']), 0)
            self.assertEqual(np.sum(curr_coverage - np.load(etmipc2.coverage[branch])['mat']), 0)
            self.assertEqual(np.sum(curr_scores - total_dict_scores[branch]), 0)
            self.assertEqual(np.sum(curr_coverage - total_dict_coverage[branch]), 0)
            self.assertEqual(np.sum(curr_scores - three_d_scores[etmipc1.tree_depth.index(branch), :, :]), 0)
            self.assertEqual(np.sum(curr_coverage - three_d_coverage[etmipc1.tree_depth.index(branch), :, :]), 0)
            rmtree(os.path.join(out_dir, str(branch)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))

    def test_import_alignment(self):
        def check_tree(seq_dict, curr, prev=None):
            if prev is None:
                return True
            c_prev = 0
            c_curr = 0
            while (c_prev != (prev - 1)) and (c_curr != (curr - 1)):
                if not seq_dict[curr][c_curr].issubset(seq_dict[prev][c_prev]):
                    c_prev += 1
                if not seq_dict[curr][c_curr].issubset(seq_dict[prev][c_prev]):
                    return False
                c_curr += 1
            return True

        with self.assertRaises(TypeError):
            ETMIPC()
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = os.path.abspath('../Test/')
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        self.assertEqual(etmipc1.tree_depth, [1, 2, 3, 4])
        self.assertEqual(len(etmipc1.unique_clusters), 7)
        self.assertEqual(len(etmipc1.cluster_mapping), 10)
        self.assertEqual(len(set(etmipc1.cluster_mapping.values())), 7)
        tree_points = set()
        for unique in etmipc1.unique_clusters:
            tree_points = tree_points.union(etmipc1.unique_clusters[unique]['tree_positions'])
        self.assertEqual(len(tree_points), 10)
        self.assertIsInstance(etmipc1.alignment, SeqAlignment)
        self.assertFalse(etmipc1.alignment.file_name.startswith('..'), 'Filename set to absolute path.')
        self.assertTrue(etmipc1.alignment.file_name.endswith('1c17A.fa'), 'Filename set to absolute path.')
        self.assertEqual(etmipc1.alignment.query_id, '>query_1c17A', 'Query ID properly changed per lab protocol.')
        self.assertNotEqual(etmipc1.alignment.query_id, '1c17A', 'Query ID properly changed per lab protocol.')
        self.assertIsInstance(etmipc1.alignment.alignment, MultipleSeqAlignment)
        self.assertEqual(etmipc1.alignment.seq_order, ['Q3J6M6-1', 'A4BNZ9-1', 'Q0EZP2-1', 'Q31DL5-1', 'D3LWU0-1',
                                                       'A0P4Z7-1', 'B8AR76-1', 'G2J8E3-1', 'A4STP8-1', 'C5WC20-1',
                                                       'Q8D3J8-1', 'Q89B44-1', 'B8D8G8-1', 'Q494C8-1', 'Q7VQW1-1',
                                                       'Q1LTU9-1', 'A0KQY3-1', 'B2VCA9-1', 'query_1c17A', 'G9EBA7-1',
                                                       'H0J1A3-1', 'A4B9C5-1', 'H3NVB3-1', 'B5JT14-1', 'G9ZC44-1',
                                                       'I8TEE1-1', 'Q6FFK5-1', 'Q8DDH3-1', 'S3EH80-1', 'Q0I5W8-1',
                                                       'A3N2U9-1', 'Q9CKW5-1', 'Q5QZI1-1', 'R7UBD3-1', 'W6LXX8-1',
                                                       'H5SE71-1', 'A1SBU5-1', 'Q48BG0-1', 'S2KJX1-1', 'F7RWD9-1',
                                                       'R9PGI3-1', 'X7E8G8-1', 'B0TQF9-1', 'R8AS80-1', 'K1Z6P6-1',
                                                       'K2CTJ5-1', 'K2D5F5-1', 'A8PQE9-1', 'K2AG65-1'],
                         'seq_order imported correctly.')
        self.assertEqual(etmipc1.alignment.query_sequence, 'MENLNMDLLYMAAAVMMGLAAIGAAIGIGILGGKFLEGAARQPDLIPLLRTQFFIVMGL'
                                                           'VDAIPMIAVGLGLYVMFAVA',
                         'Query sequence correctly identified.')
        self.assertEqual(etmipc1.alignment.seq_length, 79, 'seq_length is correctly determined.')
        self.assertEqual(etmipc1.alignment.size, 49, 'size is correctly determined.')
        # Compute distance matrix manually
        aln_obj1_num_mat = etmipc1.alignment._alignment_to_num(aa_dict=aa_dict)
        value_matrix = np.zeros([etmipc1.alignment.size, etmipc1.alignment.size])
        for i in range(etmipc1.alignment.size):
            check = aln_obj1_num_mat - aln_obj1_num_mat[i]
            value_matrix[i] = np.sum(check == 0, axis=1)
        value_matrix /= etmipc1.alignment.seq_length
        value_matrix = 1 - value_matrix
        self.assertEqual(0, np.sum(etmipc1.alignment.distance_matrix[range(etmipc1.alignment.size),
                                                                     range(etmipc1.alignment.size)]))
        self.assertEqual(0, np.sum(value_matrix - etmipc1.alignment.distance_matrix))
        self.assertEqual(set(etmipc1.alignment.seq_order), set(etmipc1.alignment.tree_order))
        self.assertNotEqual(etmipc1.alignment.seq_order, etmipc1.alignment.tree_order)
        self.assertTrue(check_tree(etmipc1.alignment.sequence_assignments, curr=1))
        for k in range(2, 5):
            self.assertTrue(check_tree(etmipc1.alignment.sequence_assignments, curr=k, prev=k - 1))
        self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl')))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl')))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa')))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'X.npz')))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = os.path.abspath('../Test/')
        etmipc2.import_alignment(query='1h1vA')
        self.assertEqual(etmipc2.tree_depth, [1, 2, 3, 4])
        self.assertEqual(len(etmipc2.unique_clusters), 7)
        tree_points = set()
        for unique in etmipc2.unique_clusters:
            tree_points = tree_points.union(etmipc2.unique_clusters[unique]['tree_positions'])
        self.assertEqual(len(etmipc2.cluster_mapping), 10)
        self.assertEqual(len(set(etmipc2.cluster_mapping.values())), 7)
        self.assertEqual(len(tree_points), 10)
        self.assertIsInstance(etmipc2.alignment, SeqAlignment)
        self.assertFalse(etmipc2.alignment.file_name.startswith('..'), 'Filename set to absolute path.')
        self.assertFalse(etmipc2.alignment.file_name.endswith('1h1bA.fa'), 'Filename set to absolute path.')
        self.assertEqual(etmipc2.alignment.query_id, '>query_1h1vA', 'Query ID properly changed per lab protocol.')
        self.assertNotEqual(etmipc2.alignment.query_id, '1h1vA', 'Query ID properly changed per lab protocol.')
        self.assertIsInstance(etmipc2.alignment.alignment, MultipleSeqAlignment)
        self.assertEqual(etmipc2.alignment.seq_order, ['W4Z7Q7-1', 'K3WHZ1-1', 'A0A067CNU4-1', 'W4FYX5-1', 'D0LWX4-1',
                                                       'R7UQM5-1', 'F0ZSV3-1', 'L7FJ78-1', 'D2UYT9-1', 'G0U265-1',
                                                       'B6KS69-1', 'D2VZ68-1', 'J9ITS7-1', 'D2VH92-1', 'W4YXM8-1',
                                                       'A0E1S8-1', 'L7FL22-1', 'F4PSD7-1', 'D2VNZ3-1', 'F4PVY9-1',
                                                       'D2VUN6-1', 'F6VIH9-1', 'Q9D9J3-1', 'G3TNP4-1', 'G3UJI6-1',
                                                       'H0X7C3-1', 'H0XHD2-1', 'H0XQE9-1', 'H0XJZ2-1', 'H0X8C7-1',
                                                       'H0XXS0-1', 'F6RBR7-1', 'U3KP83-1', 'G1SHW3-1', 'L8Y648-1',
                                                       'G3SNC6-1', 'Q8TDY3-1', 'Q9D9L5-1', 'L5KGB5-1', 'L8YCU9-1',
                                                       'M3WIX4-1', 'F1RJB0-1', 'Q2TA43-1', 'M3Z8W5-1', 'F7DVW3-1',
                                                       'S9WE56-1', 'E2QWP5-1', 'F6WN44-1', 'F7FG86-1', 'Q8TDG2-1',
                                                       'L8GXM8-1', 'D2W078-1', 'F4Q8W5-1', 'W4Z0L8-1', 'W4Z7J1-1',
                                                       'D2VKG9-1', 'I1C368-1', 'H2ZCW0-1', 'L8HDP5-1', 'X6N716-1',
                                                       'D2V8A6-1', 'D3AYM2-1', 'J9ER42-1', 'D2UYR7-1', 'B7P258-1',
                                                       'B0DKC7-1', 'F7EG46-1', 'D3B0W3-1', 'D2VZ15-1', 'R7UXI8-1',
                                                       'E3N7U2-1', 'D3BBZ6-1', 'D3BG26-1', 'X6M6K0-1', 'M2W7W2-1',
                                                       'B5DKX0-1', 'Q54L54-1', 'P51775-1', 'V6LIA4-1', 'F6Y1F4-1',
                                                       'D2VUK3-1', 'Q17C85-1', 'I1G8G1-1', 'I1GHS9-1', 'F4PGJ9-1',
                                                       'A8PB10-1', 'R7URM9-1', 'X6MVH5-1', 'F0ZR57-1', 'Q54PQ2-1',
                                                       'D3B8T6-1', 'F4PYF0-1', 'Q9XZB9-1', 'D2V0P1-1', 'F4Q8W4-1',
                                                       'C3ZM16-1', 'F2E730-1', 'Q9BIG6-1', 'Q95UN9-1', 'P12715-1',
                                                       'P53468-1', 'D3B1Y2-1', 'X6LZP1-1', 'I1GHT0-1', 'Q55CU2-1',
                                                       'Q54HE9-1', 'D2VU98-1', 'W4XW06-1', 'A8YGN0-1', 'E1CFD8-1',
                                                       'T1J8A7-1', 'F5AMM3-1', 'Q76L02-1', 'B6RB19-1', 'B5DJ22-1',
                                                       'D3AW82-1', 'Q76L01-1', 'F2UCJ6-1', 'A8Y226-1', 'T1J9I3-1',
                                                       'C3ZLQ0-1', 'C3ZMR7-2', 'D3BSY8-1', 'F4PJ19-1', 'E3MAX5-1',
                                                       'K1RBG6-1', 'F4PJ20-1', 'W4Z0B2-1', 'Q5QEI8-1', 'K1RVE3-2',
                                                       'R1F3T3-1', 'F2UCJ7-1', 'G0PAI7-1', 'C3XYD5-1', 'H2KRT9-1',
                                                       'G3RV12-1', 'Q8SWN8-1', 'R0M8Z2-1', 'T0MEN7-1', 'L2GXI8-1',
                                                       'Q76L00-1', 'V2Y4Y3-1', 'S7W9W5-1', 'F6ZHA7-1', 'B7XHF2-1',
                                                       'L2GLN4-1', 'L8GEM2-1', 'L8GYN8-1', 'K1RVE3-1', 'R7U3X6-1',
                                                       'K1R557-1', 'J9A0F1-1', 'M3Y8P7-1', 'Q54HF0-1', 'F7GD92-1',
                                                       'F0Z7P3-1', 'F2UCK1-1', 'J0D8Q2-1', 'K1RHJ4-1', 'K1QXE8-1',
                                                       'G3SAF6-1', 'K1QMP4-1', 'W2T202-1', 'C3ZMR7-1', 'C3ZTX5-1',
                                                       'Q54HF1-1', 'X6N7F5-1', 'X6NZM5-1', 'Q2U7A3-1', 'C1GG58-1',
                                                       'Q9P4D1-1', 'N1RMQ6-1', 'H1UXN5-1', 'Q4WDH2-1', 'A0A017S1P3-1',
                                                       'P53455-1', 'U7PLB6-1', 'N1RL78-1', 'Q554S6-1', 'V4BV77-1',
                                                       'S9W2X5-1', 'K1RA57-1', 'F0Z7P2-1', 'I3EP48-1', 'P07828-1',
                                                       'K1RU04-1', 'G4TBF6-1', 'Q9Y896-1', 'B4G2W4-1', 'U4U6N7-1',
                                                       'C3ZMR5-1', 'C3ZMR4-1', 'A8E073-1', 'S9XQ78-1', 'A8PB07-1',
                                                       'D6WJW7-1', 'V2WIQ9-1', 'V2Y4Z9-1', 'B0DKF0-1', 'F6R0S4-1',
                                                       'G3TAI8-1', 'G1S9C9-1', 'G7MTS6-1', 'L8WKN6-1', 'W4YKQ1-1',
                                                       'M2PKY1-1', 'F6SC57-1', 'Q93132-1', 'A7SJK6-1', 'W5JXB3-1',
                                                       'G3VC64-1', 'Q58DT9-1', 'W5PKC6-1', 'F4PJ18-1', 'R9P1L9-1',
                                                       'A0A067QDX1-1', 'E9H3B3-1', 'W2SXZ4-1', 'S9XWJ6-1', 'A8WS37-1',
                                                       'Q9UVF3-1', 'S9XSM0-1', 'P53483-1', 'P11426-1', 'Q8WPD5-1',
                                                       'K1QHY0-1', 'B4G5W8-1', 'L0R5C4-1', 'G3RQT5-1', 'Q6S8J3-1',
                                                       'D3B1V2-1', 'G3QF51-1', 'S9WUG6-1', 'F2DZG9-1', 'A5A6N1-1',
                                                       'H2KPA2-1', 'Q03341-1', 'W6U4L6-1', 'G7Y3S9-1', 'L8WKN1-1',
                                                       'Q9UVX4-1', 'G7E609-1', 'H2Z8N7-1', 'G7YDY9-1', 'W5J4N5-1',
                                                       'H3F5B4-1', 'W4Y157-1', 'P83967-1', 'G7YCA9-1', 'F1L3U5-1',
                                                       'H9J7T4-1', 'F7GR39-1', 'P60709-1', 'L9L479-1', 'P63261-1',
                                                       'S5M4F4-1', 'H2KR85-1', 'K7A690-1', 'P62736-1', 'F6PLA9-1',
                                                       'P63267-1', 'S7MYN0-1', 'query_1h1vA', 'Q5T8M7-1', 'P68133-1',
                                                       'H2R815-1', 'P68032-1', 'G6DIM9-1', 'H9IXU9-1', 'F7FMI7-1',
                                                       'Q6AY16-1', 'Q8CG27-1', 'M3X2X7-1', 'F7FGP2-1', 'Q8TC94-1',
                                                       'S7MH42-1', 'H0XP60-1', 'E2RML1-1', 'Q2T9W4-1', 'L5K8R3-1',
                                                       'F1SA46-1', 'T0MIL1-1', 'L5KV92-1', 'E1B7X2-1', 'G3UNF4-1',
                                                       'G1U4K9-1', 'H0Y1X4-1', 'I3LJA4-1', 'F6XB70-1', 'L5LE08-1',
                                                       'Q9QY84-1', 'Q4R6Q3-1', 'Q9Y615-1', 'L9KP75-1', 'S9Y9E0-1',
                                                       'E2R497-1', 'Q32KZ2-1', 'F1SP29-1', 'E2R4A0-1', 'F6XB87-1',
                                                       'L5LG13-1', 'G1TA65-1', 'Q9QY83-1', 'F6RBJ9-1', 'G3TN84-1',
                                                       'Q32L91-1', 'Q95JK8-1', 'Q9Y614-1', 'D2V3Y5-1', 'Q3M0X2-1',
                                                       'G0QIW4-1', 'I7LW62-1', 'A2FH22-1', 'G0UYV8-1', 'Q387T4-1',
                                                       'G0U4W2-1', 'K4DXF6-1', 'J9JB44-1', 'K8EMB1-1', 'A0A061RHG9-1',
                                                       'D8MBI2-1', 'L1JCH3-1', 'S0AWX7-1', 'S0B0L7-1', 'A8ISF0-1',
                                                       'D8UFJ7-1', 'I0Z7H8-1', 'A4SBH2-1', 'Q00VM6-1', 'F0WFS9-1',
                                                       'H3G9P8-1', 'K3W8L9-1', 'A0A024T9P2-1', 'A0A067C9P1-1',
                                                       'D7FV64-1', 'F0XXQ8-1', 'V7CT22-1', 'W1PF76-1', 'M1BSY7-1',
                                                       'B8BA93-1', 'Q6Z256-1', 'V4K5E9-1', 'V4U305-1', 'M4E8Y7-1',
                                                       'Q9LSD6-1', 'A9RMT8-1', 'D8S0E5-1', 'D8SB34-1', 'E4YJF0-1',
                                                       'E5S2L6-1', 'D2V9N2-1', 'D2VJG9-1', 'R1EGT7-1', 'A0A058ZB61-1',
                                                       'P53487-1', 'F4PQD5-1', 'O96621-1', 'B4G8J6-1', 'U5H6J3-1',
                                                       'M7NWC2-1', 'B6JZD3-1', 'Q9UUJ1-1', 'R4XCF6-1', 'G1X8M6-1',
                                                       'S8C0I0-1', 'D5GHR3-1', 'M2LXB0-1', 'K9GB28-1', 'Q5BFK7-1',
                                                       'A1CPC5-1', 'B6H4Z8-1', 'K3VCA8-1', 'L7JRP4-1', 'F7W7D5-1',
                                                       'S3CPY6-1', 'A0A060T738-1', 'Q6C1K7-1', 'Q759G0-1', 'I2H5H9-1',
                                                       'P32381-1', 'F2QWC6-1', 'A3LYA7-1', 'G3AXQ5-1', 'I1BX49-1',
                                                       'I4Y6F7-1', 'G4TJB5-1', 'G7E9P6-1', 'M7WPT6-1', 'A8PW20-1',
                                                       'M5EB81-1', 'A0A066VYB8-1', 'V5EVF4-1', 'J5R510-1', 'E3JXI2-1',
                                                       'F4RKV7-1', 'L8WPT5-1', 'M5BPW3-1', 'M5GC69-1', 'A0A067MV70-1',
                                                       'V2XB33-1', 'W4K143-1', 'D8Q243-1', 'A0A067TPG6-1', 'R7SGD9-1',
                                                       'A0A060SBN4-1', 'B0CTC7-1', 'A8N9D5-1', 'G4VBW8-1', 'T1KX76-1',
                                                       'H3EWD5-1', 'U1P378-1', 'P53489-1', 'U6NJ96-1', 'F2UIC6-1',
                                                       'A9V6V2-1', 'F4PDS1-1', 'E9C5M3-1', 'H2Z5X4-1', 'V4A503-1',
                                                       'T1FP77-1', 'B3RJY8-1', 'R7TR89-1', 'T2MFE1-1', 'F6QD48-1',
                                                       'I1FLA9-1', 'F7DAH7-1', 'A7S7W1-1', 'G3SJX1-1', 'S7NN93-1',
                                                       'P61160-1', 'W5QFG6-1', 'C4A0H4-1', 'K1QFS4-1', 'A0A023NL46-1',
                                                       'J9K2X7-1', 'E9G618-1', 'B4R5U8-1', 'W8AY62-1', 'P45888-1',
                                                       'H9IS56-1', 'R4WT27-1', 'A0A023BA72-1', 'R7Q985-1', 'Q3M0X7-1',
                                                       'Q3M0X9-1', 'G0R3U5-1', 'Q24C02-1', 'A0A058Z6W9-1', 'W1Q741-1',
                                                       'J9F9Y8-1', 'F2QNY4-1', 'O94630-1', 'B5Y5B2-1', 'B8C9E2-1',
                                                       'K0RM55-1', 'B6A9I8-1', 'Q5CRL6-1', 'D2VQU2-1', 'W7K197-1',
                                                       'D8LPW8-1', 'W7U6E6-1', 'A0A024U2D6-1', 'F0WRQ9-1', 'D0MTM6-1',
                                                       'H3G8F5-1', 'C4Y9V7-1', 'A5DQ76-1', 'G3B2V8-1', 'G8Y7R5-1',
                                                       'C5M8G8-1', 'M3IJA9-1', 'G3APJ3-1', 'B5RTY7-1', 'A3LVF8-1',
                                                       'A5E2K2-1', 'H8X0P7-1', 'A0A061AJ95-1', 'Q54I79-1', 'D3B962-1',
                                                       'F4QEP2-1', 'E9C4W5-1', 'B6JVI6-1', 'F2UJ75-1', 'A9V5L4-1',
                                                       'W6MH92-1', 'B9PQT4-1', 'U6KLL3-1', 'Q6C6Y2-1', 'A0A060T2S9-1',
                                                       'A8Q3A0-1', 'M5EAS4-1', 'U1HFI3-1', 'K9GI51-1', 'V5G412-1',
                                                       'H6C2R9-1','W2RYH4-1', 'Q2GY63-1', 'F9WW78-1', 'M2N0J1-1',
                                                       'B2VYQ0-1', 'E5ADV7-1', 'K2RBZ3-1', 'R7Z3L8-1', 'D4ATY3-1',
                                                       'D5GPJ5-1', 'G1XP43-1', 'U7Q578-1', 'L8G2I3-1', 'N1JC81-1',
                                                       'W3X8Q2-1', 'L7JGL5-1', 'B2B7W3-1', 'P38673-1', 'C9S6W6-1',
                                                       'N1RLV5-1', 'E9E4X9-1', 'A0A063BT17-1', 'F4NW25-1', 'U5HBM6-1',
                                                       'G7E6C4-1', 'P42023-1', 'A0A066VBJ4-1', 'U9UKT3-1', 'S2JIV6-1',
                                                       'A0A061HA90-1', 'V5EB15-1', 'Q4PE63-1', 'R9P4N6-1', 'R9AE54-1',
                                                       'M5G835-1', 'Q5KPW4-1', 'G4TCU4-1', 'A8N1N8-1', 'J4HWB6-1',
                                                       'L8WVM7-1', 'E4X3K5-1', 'L8HDV5-1', 'I1GBC4-1', 'H3FGG2-1',
                                                       'U1NUA4-1', 'W2T4S4-1', 'E5SG12-1', 'J9K0J3-1', 'W6V0K4-1',
                                                       'G4LW08-1', 'H2KNF2-1', 'F6TVZ8-1', 'H2Y6D4-1', 'H2Y6D5-1',
                                                       'T1JS59-1', 'C3ZIN7-1', 'T1FN22-1', 'E9HMX4-1', 'P45889-1',
                                                       'W5JLB7-1', 'E0VYK5-1', 'B7QDE8-1', 'G6D508-1', 'Q1HQC8-1',
                                                       'C1BUD7-1', 'E2A6V0-1', 'T1ISG9-1', 'W4XDB1-1', 'G3N132-1',
                                                       'F6ZWP5-1', 'P42025-1', 'L8Y915-1', 'R4GMT0-1', 'L5MBG6-1',
                                                       'L8Y993-1', 'P61163-1', 'V4AQ12-1', 'A7RID7-1', 'K1Q811-1',
                                                       'R7UT93-1', 'J7S8W4-1', 'G8BXK9-1', 'C5DZM6-1', 'S6EDH7-1',
                                                       'G0W6S0-1', 'G0VA67-1', 'H2AWA6-1', 'I2GXU8-1', 'J8PMY8-1',
                                                       'P38696-1', 'J8QFR4-1', 'C5DCU7-1', 'Q6CM53-1', 'G8JUI0-1',
                                                       'R9XKT5-1', 'C1EDX6-1', 'D8R5L0-1', 'Q84M92-1', 'W1NX64-1',
                                                       'A0A067J8R0-1', 'M1BYV2-1', 'A0A061IV08-1', 'Q4D1Q8-1',
                                                       'M2XCM8-1', 'D2VAP1-1', 'F0ZKF6-1', 'S0B5B4-1', 'C4M3T9-1',
                                                       'L0AVA8-1', 'J4C338-1', 'Q4N681-1', 'A7AQ08-1', 'A0A061CYT9-1',
                                                       'C1L3T0-1', 'D2VSL4-1', 'A2DBN6-1', 'G7YTM5-2', 'Q55DS6-1',
                                                       'X1WIR2-1', 'V3Z137-1', 'V4B6R7-1', 'D2VWT0-1', 'D2VC45-1',
                                                       'E4X651-1', 'C3ZLM1-1', 'J9J079-1', 'A0A067MHL9-1',
                                                       'A0A067MRJ5-1', 'A2ECQ8-1', 'D2W474-1', 'S9XKB1-1', 'L9LB89-1',
                                                       'L5KNB5-1', 'J9JH95-1', 'F6YZV6-1', 'G3W1M0-1', 'Q6BG22-1',
                                                       'K7E6E7-1', 'G3VN48-1', 'Q8BXF8-1', 'H0X7T9-1', 'G1P2S5-1',
                                                       'G3TKM5-1', 'Q9BYD9-1', 'M3Z068-1', 'Q2TBH3-1', 'G1SJB1-1',
                                                       'I3LTR7-1', 'L5KP75-1', 'F6VG50-1', 'X6NLU7-1', 'D2VIR3-1',
                                                       'D2V3F0-1', 'R7T5B0-1', 'D2V093-1', 'D2VE55-1', 'D2VEC2-1',
                                                       'A0A061J480-1', 'Q4CV94-1', 'A2E502-1', 'M1V7D1-1', 'D2VGH4-1',
                                                       'A2DKQ5-1', 'E4XMR2-1', 'Q3M0W9-1', 'G7YTM5-1', 'G4VL77-1',
                                                       'Q5DGJ8-1', 'D2V393-1', 'D2VGH8-1', 'Q28WQ3-1', 'B4MRY2-1',
                                                       'B4JWP6-1', 'B4KSP4-1', 'B3MDI2-1', 'P45891-1', 'D2VJ60-1',
                                                       'Q3M0X5-1', 'D2V0N9-1', 'G0R1H2-1', 'I7LWH9-1', 'P20360-1',
                                                       'D8TK19-1', 'O24426-1', 'A0A024VSB6-1', 'W6UF88-1',
                                                       'A0A023BCQ8-1', 'K7UYX2-1', 'W1NVU4-1', 'V4TZQ9-1', 'B9RR79-1',
                                                       'D7TCC4-1', 'U5GD97-1', 'A0A061GL34-1', 'Q6BG21-1', 'A0E660-1',
                                                       'D9N2U7-1', 'D9N2U8-1', 'A0A022R5U7-1', 'A0A059B3W3-1',
                                                       'D7LI17-1', 'P93738-1', 'R7QEX9-1', 'M2XRF6-1', 'Q3KSZ3-1',
                                                       'Q948X5-1', 'P53499-1', 'B5MEK7-1', 'B5MEK8-1', 'D8LXQ7-1',
                                                       'A0A022RAT7-1', 'A0A022R911-1', 'I2FHA6-1', 'I2FHE0-1',
                                                       'W7X4V0-1', 'I7M741-1', 'A0A023B3J4-1', 'D8M7Z9-1', 'S9VIR3-1',
                                                       'W6KV94-1', 'P45520-1', 'P53477-1', 'F2DIT4-1', 'M2XZ77-1',
                                                       'D8LXR4-1', 'D8M0Z4-1', 'D7LI18-1', 'R0HGB1-1', 'A0A059F032-1',
                                                       'D2V8I2-1', 'I1TEC2-1', 'G0R1C5-1', 'P10992-1', 'Q4YU79-1',
                                                       'V4LXM5-1', 'O65204-1', 'Q4W4T1-1', 'Q76IH7-1', 'Q8RYC2-1',
                                                       'R0HFC0-1', 'P53500-1', 'M2W5E2-1', 'X6N6X6-1', 'M0S644-1',
                                                       'Q6F5I1-1', 'B8LQ86-1', 'A4S825-1', 'Q9SWF3-1', 'P23344-1',
                                                       'J3M7K7-1', 'W4ZTI7-1', 'A2WXB0-1', 'O65314-1', 'M4E0Q9-1',
                                                       'Q96292-1', 'Q2QLI5-1', 'W4ZPJ3-1', 'K4AQE7-1', 'A0A059ATR0-1',
                                                       'M4CEE0-1', 'M0S1H8-1', 'M8CE78-1', 'M4CLU1-1', 'M0RLH0-1',
                                                       'K4AL27-1', 'P53496-1', 'B4F989-1', 'M4CXY4-1', 'I1QXY9-1',
                                                       'M0S856-1', 'M0TIQ0-1', 'D2VEW0-1', 'D2VE41-1', 'D2VUS6-1',
                                                       'P27131-1', 'D2VS98-1', 'Q9NJV4-1', 'Q2HX31-1', 'B7G878-1',
                                                       'K0SCC3-1', 'A7AX51-1', 'I7IRD5-1', 'Q8I4X0-1', 'W7JQC5-1',
                                                       'C5K5X4-1', 'P26183-1', 'P22132-1', 'P22131-1', 'C5KIZ2-1',
                                                       'C5LNQ3-1'], 'seq_order imported correctly.')
        self.assertEqual(etmipc2.alignment.query_sequence, 'TTALVCDNGSGLVKAGFAGDDAPRAVFPSIVGRPRHMVGMGQKDSYVGDEAQSKRGILT'
                                                           'LKYPIEHGIITNWDDMEKIWHHTFYNELRVAPEEHPTLLTEAPLNPKANREKMTQIMFE'
                                                           'TFNVPAMYVAIQAVLSLYASGRTTGIVLDSGDGVTHNVPIYEGYALPHAIMRLDLAGRD'
                                                           'LTDYLMKILTERGYSFVTTAEREIVRDIKEKLCYVALDFENEMATAASSSSLEKSYELP'
                                                           'DGQVITIGNERFRCPETLFQPSFIGMESAGIHETTYNSIMKCDIDIRKDLYANNVMSGG'
                                                           'TTMYPGIADRMQKEITALAPSTMKIKIIAPPERKYSVWIGGSILASLSTFQQMWITKQE'
                                                           'YDEAGPSIVHRKCF', 'Query sequence correctly identified.')
        self.assertEqual(etmipc2.alignment.seq_length, 368, 'seq_length is correctly determined.')
        self.assertEqual(etmipc2.alignment.size, 785, 'size is correctly determined.')
        # Compute distance matrix manually
        aln_obj2_num_mat = etmipc2.alignment._alignment_to_num(aa_dict=aa_dict)
        value_matrix = np.zeros([etmipc2.alignment.size, etmipc2.alignment.size])
        for i in range(etmipc2.alignment.size):
            check = aln_obj2_num_mat - aln_obj2_num_mat[i]
            value_matrix[i] = np.sum(check == 0, axis=1)
        value_matrix /= etmipc2.alignment.seq_length
        value_matrix = 1 - value_matrix
        self.assertEqual(0, np.sum(etmipc2.alignment.distance_matrix[range(etmipc2.alignment.size),
                                                                     range(etmipc2.alignment.size)]))
        self.assertEqual(0, np.sum(value_matrix - etmipc2.alignment.distance_matrix))
        self.assertEqual(set(etmipc2.alignment.seq_order), set(etmipc2.alignment.tree_order))
        self.assertNotEqual(etmipc2.alignment.seq_order, etmipc2.alignment.tree_order)
        self.assertTrue(check_tree(etmipc2.alignment.sequence_assignments, curr=1))
        for k in range(2, 5):
            self.assertTrue(check_tree(etmipc2.alignment.sequence_assignments, curr=k, prev=k - 1))
        self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl')))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl')))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa')))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        self.assertTrue(os.path.isfile(os.path.join(os.path.abspath('../Test/'), 'X.npz')))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))

    def test__generate_sub_alignment(self):
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = os.path.abspath('../Test/')
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        pool_init_sub_aln(etmipc1.alignment, etmipc1.unique_clusters)
        for tree_position in etmipc1.unique_clusters:
            k, c = tree_position
            _, sub_aln, time = generate_sub_alignment(tree_position)
            self.assertEqual(etmipc1.alignment.file_name, sub_aln.file_name)
            self.assertEqual(etmipc1.alignment.query_id, sub_aln.query_id)
            self.assertEqual(etmipc1.alignment.query_sequence, sub_aln.query_sequence)
            self.assertIsNone(sub_aln.distance_matrix)
            self.assertIsNone(sub_aln.sequence_assignments)
            self.assertEqual(sub_aln.size, len(etmipc1.alignment.sequence_assignments[k][c]))
            self.assertEqual(sub_aln.seq_order, [x for x in etmipc1.alignment.seq_order
                                                 if x in etmipc1.alignment.sequence_assignments[k][c]])
            self.assertEqual(sub_aln.tree_order, [x for x in etmipc1.alignment.tree_order
                                                  if (x in etmipc1.alignment.sequence_assignments[k][c])])
            self.assertGreater(time, 0)
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = os.path.abspath('../Test/')
        etmipc2.import_alignment(query='1h1vA')
        pool_init_sub_aln(etmipc2.alignment, etmipc2.unique_clusters)
        for tree_position in etmipc2.unique_clusters:
            k, c = tree_position
            _, sub_aln, time = generate_sub_alignment(tree_position)
            self.assertEqual(etmipc2.alignment.file_name, sub_aln.file_name)
            self.assertEqual(etmipc2.alignment.query_id, sub_aln.query_id)
            self.assertEqual(etmipc2.alignment.query_sequence, sub_aln.query_sequence)
            self.assertIsNone(sub_aln.distance_matrix)
            self.assertIsNone(sub_aln.sequence_assignments)
            self.assertEqual(sub_aln.size, len(etmipc2.alignment.sequence_assignments[k][c]))
            self.assertEqual(sub_aln.seq_order, [x for x in etmipc2.alignment.seq_order
                                                 if x in etmipc2.alignment.sequence_assignments[k][c]])
            self.assertEqual(sub_aln.tree_order, [x for x in etmipc2.alignment.tree_order
                                                  if (x in etmipc2.alignment.sequence_assignments[k][c])])
            self.assertGreater(time, 0)
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))

    def test__generate_sub_alignments_single_process(self):
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = os.path.abspath('../Test/')
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 1
        etmipc1._generate_sub_alignments()
        for tree_position in etmipc1.unique_clusters:
            k, c = tree_position
            self.assertGreater(etmipc1.unique_clusters[tree_position]['time'], 0)
            sub_aln = etmipc1.unique_clusters[tree_position]['sub_alignment']
            self.assertEqual(etmipc1.alignment.file_name, sub_aln.file_name)
            self.assertEqual(etmipc1.alignment.query_id, sub_aln.query_id)
            self.assertEqual(etmipc1.alignment.query_sequence, sub_aln.query_sequence)
            self.assertIsNone(sub_aln.distance_matrix)
            self.assertIsNone(sub_aln.sequence_assignments)
            self.assertEqual(sub_aln.size, len(etmipc1.alignment.sequence_assignments[k][c]))
            self.assertEqual(sub_aln.seq_order, [x for x in etmipc1.alignment.seq_order
                                                 if x in etmipc1.alignment.sequence_assignments[k][c]])
            self.assertEqual(sub_aln.tree_order, [x for x in etmipc1.alignment.tree_order
                                                  if (x in etmipc1.alignment.sequence_assignments[k][c])])
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = os.path.abspath('../Test/')
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 1
        etmipc2._generate_sub_alignments()
        for tree_position in etmipc2.unique_clusters:
            k, c = tree_position
            self.assertGreater(etmipc2.unique_clusters[tree_position]['time'], 0)
            sub_aln = etmipc2.unique_clusters[tree_position]['sub_alignment']
            self.assertEqual(etmipc2.alignment.file_name, sub_aln.file_name)
            self.assertEqual(etmipc2.alignment.query_id, sub_aln.query_id)
            self.assertEqual(etmipc2.alignment.query_sequence, sub_aln.query_sequence)
            self.assertIsNone(sub_aln.distance_matrix)
            self.assertIsNone(sub_aln.sequence_assignments)
            self.assertEqual(sub_aln.size, len(etmipc2.alignment.sequence_assignments[k][c]))
            self.assertEqual(sub_aln.seq_order, [x for x in etmipc2.alignment.seq_order
                                                 if x in etmipc2.alignment.sequence_assignments[k][c]])
            self.assertEqual(sub_aln.tree_order, [x for x in etmipc2.alignment.tree_order
                                                  if (x in etmipc2.alignment.sequence_assignments[k][c])])
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))

    def test__generate_sub_alignments_multi_process(self):
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = os.path.abspath('../Test/')
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 6
        etmipc1._generate_sub_alignments()
        for tree_position in etmipc1.unique_clusters:
            k, c = tree_position
            self.assertGreater(etmipc1.unique_clusters[tree_position]['time'], 0)
            sub_aln = etmipc1.unique_clusters[tree_position]['sub_alignment']
            self.assertEqual(etmipc1.alignment.file_name, sub_aln.file_name)
            self.assertEqual(etmipc1.alignment.query_id, sub_aln.query_id)
            self.assertEqual(etmipc1.alignment.query_sequence, sub_aln.query_sequence)
            self.assertIsNone(sub_aln.distance_matrix)
            self.assertIsNone(sub_aln.sequence_assignments)
            self.assertEqual(sub_aln.size, len(etmipc1.alignment.sequence_assignments[k][c]))
            self.assertEqual(sub_aln.seq_order, [x for x in etmipc1.alignment.seq_order
                                                 if x in etmipc1.alignment.sequence_assignments[k][c]])
            self.assertEqual(sub_aln.tree_order, [x for x in etmipc1.alignment.tree_order
                                                  if (x in etmipc1.alignment.sequence_assignments[k][c])])
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = os.path.abspath('../Test/')
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 6
        etmipc2._generate_sub_alignments()
        for tree_position in etmipc2.unique_clusters:
            k, c = tree_position
            self.assertGreater(etmipc2.unique_clusters[tree_position]['time'], 0)
            sub_aln = etmipc2.unique_clusters[tree_position]['sub_alignment']
            self.assertEqual(etmipc2.alignment.file_name, sub_aln.file_name)
            self.assertEqual(etmipc2.alignment.query_id, sub_aln.query_id)
            self.assertEqual(etmipc2.alignment.query_sequence, sub_aln.query_sequence)
            self.assertIsNone(sub_aln.distance_matrix)
            self.assertIsNone(sub_aln.sequence_assignments)
            self.assertEqual(sub_aln.size, len(etmipc2.alignment.sequence_assignments[k][c]))
            self.assertEqual(sub_aln.seq_order, [x for x in etmipc2.alignment.seq_order
                                                 if x in etmipc2.alignment.sequence_assignments[k][c]])
            self.assertEqual(sub_aln.tree_order, [x for x in etmipc2.alignment.tree_order
                                                  if (x in etmipc2.alignment.sequence_assignments[k][c])])
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))

    def test__score_clusters(self):
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        out_dir = os.path.abspath('../Test/')
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = out_dir
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 1
        etmipc1.low_mem = False
        etmipc1._generate_sub_alignments()
        etmipc1._score_clusters(evidence=False, aa_dict=aa_dict)
        for tree_position in etmipc1.unique_clusters:
            self.assertIsNotNone(etmipc1.unique_clusters[tree_position]['nongap_counts'])
            self.assertEqual(np.sum(etmipc1.unique_clusters[tree_position]['nongap_counts']), 0)
            c_mip = self.conservative_mip(
                etmipc1.unique_clusters[tree_position]['sub_alignment']._alignment_to_num(aa_dict))
            self.assertLess(np.sum(etmipc1.unique_clusters[tree_position]['cluster_scores'] - c_mip), 1e-10)
            self.assertGreater(etmipc1.unique_clusters[tree_position]['time'], 0)
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = out_dir
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 6
        etmipc2.low_mem = True
        etmipc2._generate_sub_alignments()
        etmipc2._score_clusters(evidence=True, aa_dict=aa_dict)
        for tree_position in etmipc2.unique_clusters:
            self.assertIsNotNone(etmipc2.unique_clusters[tree_position]['nongap_counts'])
            self.assertEqual(etmipc2.unique_clusters[tree_position]['nongap_counts'],
                             single_matrix_filename('Nongap_counts_C{}'.format(tree_position[1]),
                                                    branch=tree_position[0], out_dir=out_dir)[1])
            self.assertGreater(np.sum(load_single_matrix('Nongap_counts_C{}'.format(tree_position[1]),
                                                         branch=tree_position[0], out_dir=out_dir)), 0)
            c_mip = self.conservative_mip(
                etmipc2.unique_clusters[tree_position]['sub_alignment']._alignment_to_num(aa_dict))
            self.assertEqual(etmipc2.unique_clusters[tree_position]['cluster_scores'],
                             single_matrix_filename(name='Raw_C{}'.format(tree_position[1]), branch=tree_position[0],
                                                out_dir=out_dir)[1])
            self.assertLess(np.sum(load_single_matrix(name='Raw_C{}'.format(tree_position[1]), branch=tree_position[0],
                                                      out_dir=out_dir) - c_mip), 1e-9)
            self.assertGreater(etmipc2.unique_clusters[tree_position]['time'], 0)
        for k in etmipc2.tree_depth:
            rmtree(os.path.join(out_dir, str(k)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))

    def test_single_matrix_filename(self):
        out_dir = os.path.abspath('../Test/')
        parent_dir, fn = single_matrix_filename(name='Dummy', branch=1, out_dir=out_dir)
        self.assertEqual(os.path.join(out_dir, str(1)), parent_dir)
        self.assertEqual(os.path.join(out_dir, str(1), 'K1_Dummy.npz'), fn)

    def test_exists_single_matrix(self):
        out_dir = os.path.abspath('../Test/')
        self.assertFalse(exists_single_matrix(name='Dummy', branch=1, out_dir=out_dir))
        os.mkdir(os.path.join(out_dir, str(1)))
        dummy_fn = os.path.join(out_dir, str(1), 'K1_Dummy.npz')
        dummy_handle = open(dummy_fn, 'wb')
        dummy_handle.write('Testing')
        dummy_handle.close()
        rmtree(os.path.join(out_dir, str(1)))

    def test_save_single_matrix(self):
        out_dir = os.path.abspath('../Test/')
        dummy = np.random.rand(10,10)
        save_single_matrix(mat=dummy, name='Dummy', branch=1, out_dir=out_dir)
        _, fn = single_matrix_filename(name='Dummy', branch=1, out_dir=out_dir)
        self.assertTrue(os.path.isfile(fn))
        dummy_prime = np.load(fn)['mat']
        self.assertEqual(np.sum(dummy - dummy_prime), 0)
        rmtree(os.path.join(out_dir, str(1)))

    def test_load_single_matrix(self):
        out_dir = os.path.abspath('../Test/')
        dummy = np.random.rand(10, 10)
        self.assertIsNone(load_single_matrix(name='Dummy', branch=1, out_dir=out_dir))
        save_single_matrix(mat=dummy, name='Dummy', branch=1, out_dir=out_dir)
        _, fn = single_matrix_filename(name='Dummy', branch=1, out_dir=out_dir)
        self.assertTrue(os.path.isfile(fn))
        dummy_prime = load_single_matrix(name='Dummy', branch=1, out_dir=out_dir)
        self.assertEqual(np.sum(dummy - dummy_prime), 0)
        rmtree(os.path.join(out_dir, str(1)))

    # Could not properly test this method, not sure how to check the global variables in another module like this
    # explicitly, will try to figure it out later. For now the next tests will evaluate if this works or not by proxy.
    # def test_pool_init_sub_aln(self):
    #     etmipc1 = ETMIPC('../Test/1c17A.fa')
    #     etmipc1.tree_depth = (2, 5)
    #     etmipc1.output_dir = os.path.abspath('../Test/')
    #     etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
    #     pool_init_sub_aln(etmipc1.alignment, etmipc1.unique_clusters)
    #     self.assertTrue('assignment_dict' in globals())
    #     self.assertTrue('full_aln' in globals())
    #     self.assertIs(globals()['assignment_dict'], etmipc1.unique_clusters)
    #     self.assertIs(globals()['full_aln'], etmipc1.alignment)
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
    #     del globals()['assignment_dict']
    #     del globals()['full_aln']
    #     etmipc2 = ETMIPC('../Test/1h1vA.fa')
    #     etmipc2.tree_depth = (2, 5)
    #     etmipc2.output_dir = os.path.abspath('../Test/')
    #     etmipc2.import_alignment(query='1h1vA')
    #     pool_init_sub_aln(etmipc2.alignment, etmipc2.unique_clusters)
    #     self.assertTrue('assignment_dict' in globals())
    #     self.assertTrue('full_aln' in globals())
    #     self.assertIs(globals()['assignment_dict'], etmipc2.unique_clusters)
    #     self.assertIs(globals()['full_aln'], etmipc2.alignment)
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
    #     del globals()['assignment_dict']
    #     del globals()['full_aln']

    # Could not properly test this method, not sure how to check the global variables in another module like this
    # explicitly, will try to figure it out later. For now the next tests will evaluate if this works or not by proxy.
    # def test_pool_init_score(self):
    #     aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    #                '-']
    #     aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
    #     out_dir = os.path.abspath('../Test/')
    #     etmipc1 = ETMIPC('../Test/1c17A.fa')
    #     etmipc1.tree_depth = (2, 5)
    #     etmipc1.output_dir = os.path.abspath('../Test/')
    #     etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
    #     etmipc1.processes = 6
    #     etmipc1._generate_sub_alignments()
    #     pool_init_score(evidence=True, cluster_dict=etmipc1.unique_clusters, amino_acid_mapping=aa_dict,
    #                     out_dir=out_dir, low_mem=True)
    #     pool_init_score(evidence=False, cluster_dict=etmipc1.unique_clusters, amino_acid_mapping=aa_dict,
    #                     out_dir=out_dir, low_mem=False)
    #     self.assertTrue('assignment_dict' in globals())
    #     self.assertTrue('full_aln' in globals())
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
    #     # del(full_aln)
    #     # del(assignment_dict)
    #     etmipc2 = ETMIPC('../Test/1h1vA.fa')
    #     etmipc2.tree_depth = (2, 5)
    #     etmipc2.output_dir = os.path.abspath('../Test/')
    #     etmipc2.import_alignment(query='1h1vA')
    #     etmipc2.processes = 6
    #     etmipc2._generate_sub_alignments()
    #     pool_init_score(evidence=True, cluster_dict=etmipc2.unique_clusters, amino_acid_mapping=aa_dict,
    #                     out_dir=out_dir, low_mem=True)
    #     pool_init_score(evidence=False, cluster_dict=etmipc2.unique_clusters, amino_acid_mapping=aa_dict,
    #                     out_dir=out_dir, low_mem=False)
    #     self.assertTrue('assignment_dict' in globals())
    #     self.assertTrue('full_aln' in globals())
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
    #     # del (full_aln)
    #     # del (assignment_dict)

    def test_mip_score(self):
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        out_dir = os.path.abspath('../Test/')
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = os.path.abspath('../Test/')
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 6
        etmipc1._generate_sub_alignments()
        pool_init_score(evidence=True, cluster_dict=etmipc1.unique_clusters, amino_acid_mapping=aa_dict,
                        out_dir=out_dir, low_mem=True)
        etmipc1_mip_res1 = mip_score((1, 0))
        print(etmipc1_mip_res1)
        etmipc1_conservateive_mip = self.conservative_mip(etmipc1.unique_clusters[(1,0)]['sub_alignment']._alignment_to_num(aa_dict))

        self.assertEqual(etmipc1_mip_res1[0], (1,0))
        self.assertEqual(etmipc1_mip_res1[1], single_matrix_filename(name='Raw_C0', branch=1, out_dir=out_dir)[1])
        self.assertLess(np.sum(load_single_matrix(name='Raw_C0', branch=1, out_dir=out_dir) - etmipc1_conservateive_mip),
                        1e-10)
        self.assertEqual(etmipc1_mip_res1[2], single_matrix_filename(name='Nongap_counts_C0', branch=1, out_dir=out_dir)[1])
        self.assertGreater(np.sum(load_single_matrix(name='Nongap_counts_C0', branch=1, out_dir=out_dir)), 0)
        self.assertGreater(etmipc1_mip_res1[3], 0)
        pool_init_score(evidence=False, cluster_dict=etmipc1.unique_clusters, amino_acid_mapping=aa_dict,
                        out_dir=out_dir, low_mem=False)
        etmipc1_mip_res2 = mip_score((1, 0))
        self.assertEqual(etmipc1_mip_res2[0], (1, 0))
        self.assertLess(np.sum(etmipc1_mip_res2[1] - etmipc1_conservateive_mip), 1e-10)
        self.assertEqual(np.sum(etmipc1_mip_res2[2]), 0)
        self.assertGreater(etmipc1_mip_res2[3], 0)
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))
        rmtree(os.path.join(out_dir, '1'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = os.path.abspath('../Test/')
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 6
        etmipc2._generate_sub_alignments()
        pool_init_score(evidence=True, cluster_dict=etmipc2.unique_clusters, amino_acid_mapping=aa_dict,
                        out_dir=out_dir, low_mem=True)
        etmipc2_mip_res1 = mip_score((1, 0))
        etmipc2_conservateive_mip = self.conservative_mip(etmipc2.unique_clusters[(1, 0)]['sub_alignment']._alignment_to_num(aa_dict))
        self.assertEqual(etmipc2_mip_res1[0], (1, 0))
        self.assertEqual(etmipc2_mip_res1[1], single_matrix_filename(name='Raw_C0', branch=1, out_dir=out_dir)[1])
        self.assertLess(np.sum(load_single_matrix(name='Raw_C0', branch=1, out_dir=out_dir) - etmipc2_conservateive_mip),
                        1e-10)
        self.assertEqual(etmipc2_mip_res1[2], single_matrix_filename(name='Nongap_counts_C0', branch=1, out_dir=out_dir)[1])
        self.assertGreater(np.sum(load_single_matrix(name='Nongap_counts_C0', branch=1, out_dir=out_dir)), 0)
        self.assertGreater(etmipc2_mip_res1[3], 0)
        pool_init_score(evidence=False, cluster_dict=etmipc2.unique_clusters, amino_acid_mapping=aa_dict,
                        out_dir=out_dir, low_mem=False)
        etmipc2_mip_res2 = mip_score((1, 0))
        self.assertEqual(etmipc2_mip_res2[0], (1, 0))
        self.assertLess(np.sum(etmipc2_mip_res2[1] - etmipc2_conservateive_mip), 1e-10)
        self.assertEqual(np.sum(etmipc2_mip_res2[2]), 0)
        self.assertGreater(etmipc2_mip_res2[3], 0)
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))
        rmtree(os.path.join(out_dir, '1'))

    def test_calculate_cluster_scores(self):
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        out_dir = os.path.abspath('../Test/')
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = out_dir
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 1
        etmipc1.low_mem = False
        etmipc1.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
        for tree_position in etmipc1.unique_clusters:
            k, c = tree_position
            self.assertGreater(etmipc1.unique_clusters[tree_position]['time'], 0)
            sub_aln = etmipc1.unique_clusters[tree_position]['sub_alignment']
            self.assertEqual(etmipc1.alignment.file_name, sub_aln.file_name)
            self.assertEqual(etmipc1.alignment.query_id, sub_aln.query_id)
            self.assertEqual(etmipc1.alignment.query_sequence, sub_aln.query_sequence)
            self.assertIsNone(sub_aln.distance_matrix)
            self.assertIsNone(sub_aln.sequence_assignments)
            self.assertEqual(sub_aln.size, len(etmipc1.alignment.sequence_assignments[k][c]))
            self.assertEqual(sub_aln.seq_order, [x for x in etmipc1.alignment.seq_order
                                                 if x in etmipc1.alignment.sequence_assignments[k][c]])
            self.assertEqual(sub_aln.tree_order, [x for x in etmipc1.alignment.tree_order
                                                  if (x in etmipc1.alignment.sequence_assignments[k][c])])
            self.assertIsNotNone(etmipc1.unique_clusters[tree_position]['nongap_counts'])
            self.assertEqual(np.sum(etmipc1.unique_clusters[tree_position]['nongap_counts']), 0)
            self.assertEqual(np.sum(etmipc1.nongap_counts[tree_position[0]][tree_position[1]]), 0)
            c_mip = self.conservative_mip(
                etmipc1.unique_clusters[tree_position]['sub_alignment']._alignment_to_num(aa_dict))
            self.assertLess(np.sum(etmipc1.unique_clusters[tree_position]['cluster_scores'] - c_mip), 1e-10)
            self.assertLess(np.sum(etmipc1.cluster_scores[tree_position[0]][tree_position[1]] - c_mip), 1e-10)
            self.assertGreater(etmipc1.unique_clusters[tree_position]['time'], 0)
            self.assertGreater(etmipc1.times[k], 0)
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = out_dir
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 6
        etmipc2.low_mem = True
        etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
        for tree_position in etmipc2.unique_clusters:
            k, c = tree_position
            self.assertGreater(etmipc2.unique_clusters[tree_position]['time'], 0)
            sub_aln = etmipc2.unique_clusters[tree_position]['sub_alignment']
            self.assertEqual(etmipc2.alignment.file_name, sub_aln.file_name)
            self.assertEqual(etmipc2.alignment.query_id, sub_aln.query_id)
            self.assertEqual(etmipc2.alignment.query_sequence, sub_aln.query_sequence)
            self.assertIsNone(sub_aln.distance_matrix)
            self.assertIsNone(sub_aln.sequence_assignments)
            self.assertEqual(sub_aln.size, len(etmipc2.alignment.sequence_assignments[k][c]))
            self.assertEqual(sub_aln.seq_order, [x for x in etmipc2.alignment.seq_order
                                                 if x in etmipc2.alignment.sequence_assignments[k][c]])
            self.assertEqual(sub_aln.tree_order, [x for x in etmipc2.alignment.tree_order
                                                  if (x in etmipc2.alignment.sequence_assignments[k][c])])
            self.assertIsNotNone(etmipc2.unique_clusters[tree_position]['nongap_counts'])
            self.assertEqual(etmipc2.unique_clusters[tree_position]['nongap_counts'],
                             single_matrix_filename('Nongap_counts_C{}'.format(tree_position[1]),
                                                    branch=tree_position[0], out_dir=out_dir)[1])
            self.assertGreater(np.sum(load_single_matrix('Nongap_counts_C{}'.format(tree_position[1]),
                                                         branch=tree_position[0], out_dir=out_dir)), 0)
            self.assertGreater(np.sum(np.load(etmipc2.nongap_counts[tree_position[0]][tree_position[1]])['mat']), 0)
            c_mip = self.conservative_mip(
                etmipc2.unique_clusters[tree_position]['sub_alignment']._alignment_to_num(aa_dict))
            self.assertEqual(etmipc2.unique_clusters[tree_position]['cluster_scores'],
                             single_matrix_filename(name='Raw_C{}'.format(tree_position[1]), branch=tree_position[0],
                                                    out_dir=out_dir)[1])
            self.assertLess(np.sum(load_single_matrix(name='Raw_C{}'.format(tree_position[1]), branch=tree_position[0],
                                                      out_dir=out_dir) - c_mip), 1e-9)
            self.assertLess(np.sum(np.load(etmipc2.cluster_scores[tree_position[0]][tree_position[1]])['mat'] - c_mip), 1e-9)
            self.assertGreater(etmipc2.unique_clusters[tree_position]['time'], 0)
            self.assertGreater(etmipc2.times[k], 0)
        for k in etmipc2.tree_depth:
            rmtree(os.path.join(out_dir, str(k)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))

    # Could not properly test this method, not sure how to check the global variables in another module like this
    # explicitly, will try to figure it out later. For now the next tests will evaluate if this works or not by proxy.
    # def test_init_calculate_branch_score(self):
    #     aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    #                '-']
    #     aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
    #     out_dir = os.path.abspath('../Test/')
    #     etmipc1 = ETMIPC('../Test/1c17A.fa')
    #     etmipc1.tree_depth = (2, 5)
    #     etmipc1.output_dir = os.path.abspath('../Test/')
    #     etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
    #     etmipc1.processes = 6
    #     etmipc1.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
    #     pool_init_calculate_branch_score(curr_instance=etmipc1, combine_clusters='sum')
    #     self.assertTrue('instance' in globals())
    #     self.assertTrue('combination_method' in globals())
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
    #     # del(full_aln)
    #     # del(assignment_dict)
    #     etmipc2 = ETMIPC('../Test/1h1vA.fa')
    #     etmipc2.tree_depth = (2, 5)
    #     etmipc2.output_dir = os.path.abspath('../Test/')
    #     etmipc2.import_alignment(query='1h1vA')
    #     etmipc2.processes = 6
    #     etmipc2.low_mem = True
    #     etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
    #     pool_init_calculate_branch_score(curr_instance=etmipc2, combine_clusters='evidence_weighted')
    #     self.assertTrue('instance' in globals())
    #     self.assertTrue('combination_method' in globals())
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
    #     # del (full_aln)
    #     # del (assignment_dict)

    def test_calculate_branch_score(self):
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        out_dir = os.path.abspath('../Test/')
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = out_dir
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 1
        etmipc1.low_mem = False
        etmipc1.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
        pool_init_calculate_branch_score(curr_instance=etmipc1, combine_clusters='sum')
        etmipc1_branch_res1 = calculate_branch_score(2)
        self.assertEqual(etmipc1_branch_res1[0], 2)
        self.assertGreater(etmipc1_branch_res1[2], 0)
        scores1 = etmipc1.get_cluster_scores(branch=2, cluster=0, three_dim=False) + \
                  etmipc1.get_cluster_scores(branch=2, cluster=1, three_dim=False)
        self.assertLess(np.sum(etmipc1_branch_res1[1] - scores1), 1e-10)
        pool_init_calculate_branch_score(curr_instance=etmipc1, combine_clusters='average')
        etmipc1_branch_res2 = calculate_branch_score(2)
        self.assertEqual(etmipc1_branch_res2[0], 2)
        self.assertGreater(etmipc1_branch_res2[2], 0)
        scores2 = scores1 / 2.0
        self.assertLess(np.sum(etmipc1_branch_res2[1] - scores2), 1e-10)
        pool_init_calculate_branch_score(curr_instance=etmipc1, combine_clusters='size_weighted')
        etmipc1_branch_res3 = calculate_branch_score(2)
        self.assertEqual(etmipc1_branch_res3[0], 2)
        self.assertGreater(etmipc1_branch_res3[2], 0)
        scores3 = (etmipc1.get_sub_alignment(branch=2, cluster=0).size *
                           etmipc1.get_cluster_scores(branch=2, cluster=0, three_dim=False)) + \
                  (etmipc1.get_sub_alignment(branch=2, cluster=1).size *
                            etmipc1.get_cluster_scores(branch=2, cluster=1, three_dim=False))
        scores3 /= float(etmipc1.alignment.size)
        self.assertLess(np.sum(etmipc1_branch_res3[1] - scores3), 1e-10)
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = out_dir
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 6
        etmipc2.low_mem = True
        etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
        pool_init_calculate_branch_score(curr_instance=etmipc2, combine_clusters='evidence_weighted')
        etmipc2_branch_res1 = calculate_branch_score(2)
        self.assertEqual(etmipc2_branch_res1[0], 2)
        self.assertGreater(etmipc2_branch_res1[2], 0)
        scores_evidence = (etmipc2.get_nongap_counts(branch=2, cluster=0) *
                                   etmipc2.get_cluster_scores(branch=2, cluster=0, three_dim=False)) + \
                          (etmipc2.get_nongap_counts(branch=2, cluster=1) *
                                    etmipc2.get_cluster_scores(branch=2, cluster=1, three_dim=False))
        scores_evidence[np.isnan(scores_evidence)] = 0.0
        scores4 = scores_evidence / etmipc2.get_nongap_counts(branch=1, cluster=0)
        scores4[np.isnan(scores4)] = 0.0
        print(etmipc2_branch_res1[1])
        self.assertLess(np.sum(np.load(etmipc2_branch_res1[1])['mat'] - scores4), 1e-10)
        pool_init_calculate_branch_score(curr_instance=etmipc2, combine_clusters='evidence_vs_size')
        etmipc2_branch_res2 = calculate_branch_score(2)
        self.assertEqual(etmipc2_branch_res2[0], 2)
        self.assertGreater(etmipc2_branch_res2[2], 0)
        scores5 = scores_evidence / float(etmipc2.alignment.size)
        scores5[np.isnan(scores5)] = 0.0
        self.assertLess(np.sum(np.load(etmipc2_branch_res2[1])['mat'] - scores5), 1e-10)
        for k in etmipc2.tree_depth:
            rmtree(os.path.join(out_dir, str(k)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))

    def test_calculate_branch_scores(self):
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        out_dir = os.path.abspath('../Test/')
        etmipc1_a = ETMIPC('../Test/1c17A.fa')
        etmipc1_a.tree_depth = (2, 5)
        etmipc1_a.output_dir = out_dir
        etmipc1_a.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1_a.processes = 1
        etmipc1_a.low_mem = False
        etmipc1_a.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
        etmipc1_a.calculate_branch_scores(combine_clusters='sum')
        for branch in etmipc1_a.tree_depth:
            scores = np.zeros((etmipc1_a.alignment.seq_length, etmipc1_a.alignment.seq_length))
            for cluster in range(branch):
                scores += etmipc1_a.get_cluster_scores(branch=branch, cluster=cluster, three_dim=False)
            self.assertLess(np.sum(etmipc1_a.branch_scores[branch] - scores), 1e-10)
            self.assertGreater(etmipc1_a.times[branch], 0)
        etmipc1_b = ETMIPC('../Test/1c17A.fa')
        etmipc1_b.tree_depth = (2, 5)
        etmipc1_b.output_dir = out_dir
        etmipc1_b.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1_b.processes = 1
        etmipc1_b.low_mem = False
        etmipc1_b.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
        etmipc1_b.calculate_branch_scores(combine_clusters='average')
        for branch in etmipc1_b.tree_depth:
            scores = np.zeros((etmipc1_b.alignment.seq_length, etmipc1_b.alignment.seq_length))
            for cluster in range(branch):
                scores += etmipc1_b.get_cluster_scores(branch=branch, cluster=cluster, three_dim=False)
            scores /= float(branch)
            self.assertLess(np.sum(etmipc1_b.branch_scores[branch] - scores), 1e-10)
            self.assertGreater(etmipc1_b.times[branch], 0)
        etmipc1_c = ETMIPC('../Test/1c17A.fa')
        etmipc1_c.tree_depth = (2, 5)
        etmipc1_c.output_dir = out_dir
        etmipc1_c.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1_c.processes = 1
        etmipc1_c.low_mem = False
        etmipc1_c.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
        etmipc1_c.calculate_branch_scores(combine_clusters='size_weighted')
        for branch in etmipc1_c.tree_depth:
            scores = np.zeros((etmipc1_c.alignment.seq_length, etmipc1_c.alignment.seq_length))
            for cluster in range(branch):
                scores += etmipc1_c.get_cluster_scores(branch=branch, cluster=cluster, three_dim=False) * \
                          etmipc1_c.get_sub_alignment(branch=branch, cluster=cluster).size
            scores /= float(branch)
            self.assertLess(np.sum(etmipc1_c.branch_scores[branch] - scores), 1e-10)
            self.assertGreater(etmipc1_c.times[branch], 0)
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))
        etmipc2_a = ETMIPC('../Test/1h1vA.fa')
        etmipc2_a.tree_depth = (2, 5)
        etmipc2_a.output_dir = out_dir
        etmipc2_a.import_alignment(query='1h1vA')
        etmipc2_a.processes = 6
        etmipc2_a.low_mem = True
        etmipc2_a.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
        etmipc2_a.calculate_branch_scores(combine_clusters='evidence_weighted')
        for branch in etmipc2_a.tree_depth:
            scores = np.zeros((etmipc2_a.alignment.seq_length, etmipc2_a.alignment.seq_length))
            for cluster in range(branch):
                scores += etmipc2_a.get_cluster_scores(branch=branch, cluster=cluster, three_dim=False) * \
                          etmipc2_a.get_nongap_counts(branch=branch, cluster=cluster)
            scores[np.isnan(scores)] = 0.0
            scores /= etmipc2_a.get_nongap_counts(branch=1, cluster=0)
            scores[np.isnan(scores)] = 0.0
            print(np.load(etmipc2_a.branch_scores[branch])['mat'])
            print(scores)
            self.assertLess(np.sum(np.load(etmipc2_a.branch_scores[branch])['mat'] - scores), 1e-10)
            self.assertGreater(etmipc2_a.times[branch], 0)
            os.remove(os.path.join(os.path.abspath('../Test/'), str(branch), 'K{}_Result.npz'.format(branch)))
        etmipc2_b = ETMIPC('../Test/1h1vA.fa')
        etmipc2_b.tree_depth = (2, 5)
        etmipc2_b.output_dir = out_dir
        etmipc2_b.import_alignment(query='1h1vA')
        etmipc2_b.processes = 6
        etmipc2_b.low_mem = True
        etmipc2_b.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
        etmipc2_b.calculate_branch_scores(combine_clusters='evidence_vs_size')
        for branch in etmipc2_b.tree_depth:
            scores = np.zeros((etmipc2_b.alignment.seq_length, etmipc2_b.alignment.seq_length))
            for cluster in range(branch):
                scores += etmipc2_b.get_cluster_scores(branch=branch, cluster=cluster, three_dim=False) * \
                          etmipc2_b.get_nongap_counts(branch=branch, cluster=cluster)
            scores[np.isnan(scores)] = 0.0
            scores /= float(etmipc2_b.alignment.size)
            scores[np.isnan(scores)] = 0.0
            self.assertLess(np.sum(np.load(etmipc2_b.branch_scores[branch])['mat'] - scores), 1e-10)
            self.assertGreater(etmipc2_b.times[branch], 0)
            os.remove(os.path.join(os.path.abspath('../Test/'), str(branch), 'K{}_Result.npz'.format(branch)))
        for branch in etmipc2_b.tree_depth:
            rmtree(os.path.join(out_dir, str(branch)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))

    # Could not properly test this method, not sure how to check the global variables in another module like this
    # explicitly, will try to figure it out later. For now the next tests will evaluate if this works or not by proxy.
    # def test_init_calculate_scores_and_coverage(self):
    #     aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    #                '-']
    #     aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
    #     out_dir = os.path.abspath('../Test/')
    #     etmipc1 = ETMIPC('../Test/1c17A.fa')
    #     etmipc1.tree_depth = (2, 5)
    #     etmipc1.output_dir = os.path.abspath('../Test/')
    #     etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
    #     etmipc1.processes = 6
    #     etmipc1.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
    #     etmipc1.calculate_branch_scores(combine_clusters='sum')
    #     pool_init_calculate_scores_and_coverage(curr_instance=etmipc1, combine_branches='sum')
    #     self.assertTrue('instance' in globals())
    #     self.assertTrue('combination_method' in globals())
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
    #     etmipc2 = ETMIPC('../Test/1h1vA.fa')
    #     etmipc2.tree_depth = (2, 5)
    #     etmipc2.output_dir = os.path.abspath('../Test/')
    #     etmipc2.import_alignment(query='1h1vA')
    #     etmipc2.processes = 6
    #     etmipc2.low_mem = True
    #     etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
    #     etmipc2.calculate_branch_scores(combine_clusters='evidence_weigthed')
    #     pool_init_calculate_scores_and_coverage(curr_instance=etmipc2, combine_clusters='evidence_weighted')
    #     self.assertTrue('instance' in globals())
    #     self.assertTrue('combination_method' in globals())
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
    #     for k in etmipc2.tree_depth:
    #         rmtree(os.path.join(out_dir, str(k)))
    #     # del (full_aln)
    #     # del (assignment_dict)

    def test_calculate_score_and_coverage(self):
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        out_dir = os.path.abspath('../Test/')
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = out_dir
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 1
        etmipc1.low_mem = False
        etmipc1.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
        etmipc1.calculate_branch_scores(combine_clusters='sum')
        pool_init_calculate_score_and_coverage(curr_instance=etmipc1, combine_branches='sum')
        etmipc1_res1 = calculate_score_and_coverage(2)
        self.assertEqual(etmipc1_res1[0], 2)
        self.assertGreater(etmipc1_res1[3], 0)
        scores1 = etmipc1.get_branch_scores(branch=1, three_dim=False) + \
                  etmipc1.get_branch_scores(branch=2, three_dim=False)
        coverage1 = self.calculate_coverage(scores1)
        self.assertLess(np.sum(etmipc1_res1[1] - scores1), 1e-10)
        self.assertLess(np.sum(etmipc1_res1[2] - coverage1), 1e-10)
        self.assertEqual(np.sum(rankdata(etmipc1_res1[1][np.triu_indices(n=etmipc1.alignment.seq_length, k=1)]) -
                                rankdata(etmipc1_res1[2][np.triu_indices(n=etmipc1.alignment.seq_length, k=1)])), 0)
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = out_dir
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 6
        etmipc2.low_mem = True
        etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
        etmipc2.calculate_branch_scores(combine_clusters='evidence_weighted')
        pool_init_calculate_score_and_coverage(curr_instance=etmipc2, combine_branches='evidence_weighted')
        etmipc2_res1 = calculate_score_and_coverage(2)
        self.assertEqual(etmipc2_res1[0], 2)
        self.assertGreater(etmipc2_res1[3], 0)
        scores2 = etmipc2.get_branch_scores(branch=1, three_dim=False) + \
                  etmipc2.get_branch_scores(branch=2, three_dim=False)
        scores2 /= 2.0
        coverage2 = self.calculate_coverage(scores2)
        self.assertLess(np.sum(np.load(etmipc2_res1[1])['mat'] - scores2), 1e-10)
        self.assertLess(np.sum(np.load(etmipc2_res1[2])['mat'] - coverage2), 1e-10)
        self.assertEqual(np.sum(rankdata(np.load(etmipc2_res1[1])['mat'][np.triu_indices(n=etmipc2.alignment.seq_length, k=1)], 'dense') -
                         rankdata(np.load(etmipc2_res1[2])['mat'][np.triu_indices(n=etmipc2.alignment.seq_length, k=1)], 'dense')), 0)
        for k in etmipc2.tree_depth:
            rmtree(os.path.join(out_dir, str(k)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))

    def test_calculate_final_scores(self):
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        out_dir = os.path.abspath('../Test/')
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = out_dir
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 1
        etmipc1.low_mem = False
        etmipc1.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
        etmipc1.calculate_branch_scores(combine_clusters='sum')
        etmipc1.calculate_final_scores(combine_branches='sum')
        for i, branch in enumerate(etmipc1.tree_depth):
            self.assertGreater(etmipc1.times[branch], 0)
            scores1 = np.zeros((etmipc1.alignment.seq_length, etmipc1.alignment.seq_length))
            for j in range(i):
                scores1 += etmipc1.get_branch_scores(branch=etmipc1.tree_depth[j], three_dim=False)
            coverage1 = self.calculate_coverage(scores1)
            self.assertLess(np.sum(etmipc1.scores[branch] - scores1), 1e-10)
            self.assertLess(np.sum(etmipc1.coverage[branch] - coverage1), 1e-10)
            self.assertEqual(np.sum(rankdata(etmipc1.scores[branch][np.triu_indices(n=etmipc1.alignment.seq_length, k=1)]) -
                                    rankdata(etmipc1.coverage[branch][np.triu_indices(n=etmipc1.alignment.seq_length, k=1)])), 0)
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = out_dir
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 6
        etmipc2.low_mem = True
        etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
        etmipc2.calculate_branch_scores(combine_clusters='evidence_weighted')
        etmipc2.calculate_final_scores(combine_branches='average')
        for i, branch in enumerate(etmipc2.tree_depth):
            self.assertGreater(etmipc2.times[branch], 0)
            scores2 = np.zeros((etmipc2.alignment.seq_length, etmipc2.alignment.seq_length))
            for j in range(i):
                scores2 += etmipc2.get_branch_scores(branch=etmipc2.tree_depth[j], three_dim=False)
            coverage2 = self.calculate_coverage(scores2)
            self.assertLess(np.sum(np.load(etmipc2.scores[branch])['mat'] - scores2), 1e-10)
            self.assertLess(np.sum(np.load(etmipc2.coverage[branch])['mat'] - coverage2), 1e-10)
            self.assertEqual(np.sum(rankdata(np.load(etmipc2.scores[branch])['mat'][np.triu_indices(n=etmipc2.alignment.seq_length, k=1)]) -
                                    rankdata(np.load(etmipc2.coverage[branch])['mat'][np.triu_indices(n=etmipc2.alignment.seq_length, k=1)])), 0)
        for branch in etmipc2.tree_depth:
            rmtree(os.path.join(out_dir, str(branch)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))

    # Could not properly test this method, not sure how to check the global variables in another module like this
    # explicitly, will try to figure it out later. For now the next tests will evaluate if this works or not by proxy.
    # def test_pool_init_write_score(self):
    #
    #     aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    #                '-']
    #     aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
    #     out_dir = os.path.abspath('../Test/')
    #     etmipc1 = ETMIPC('../Test/1c17A.fa')
    #     etmipc1.tree_depth = (2, 5)
    #     etmipc1.output_dir = os.path.abspath('../Test/')
    #     etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
    #     etmipc1.processes = 6
    #     etmipc1.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
    #     etmipc1.calculate_branch_scores(combine_clusters='sum')
    #     etmipc1.calculate_final_scores(combine_branches='sum')
    #     pool_init_write_score(curr_instance=etmipc1, curr_date=str(datetime.date.today()))
    #     self.assertTrue('instance' in globals())
    #     self.assertTrue('today' in globals())
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
    #     etmipc2 = ETMIPC('../Test/1h1vA.fa')
    #     etmipc2.tree_depth = (2, 5)
    #     etmipc2.output_dir = os.path.abspath('../Test/')
    #     etmipc2.import_alignment(query='1h1vA')
    #     etmipc2.processes = 6
    #     etmipc2.low_mem = True
    #     etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
    #     etmipc2.calculate_branch_scores(combine_clusters='evidence_weigthed')
    #     etmipc2.calculate_final_scores(combine_branches='average')
    #     pool_init_write_score(curr_instance=etmipc2, curr_date=str(datetime.date.today()))
    #     self.assertTrue('instance' in globals())
    #     self.assertTrue('today' in globals())
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
    #     os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
    #     for k in etmipc2.tree_depth:
    #         rmtree(os.path.join(out_dir, str(k)))
    #     # del (full_aln)
    #     # del (assignment_dict)

    def test_write_score(self):
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        out_dir = os.path.abspath('../Test/')
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = out_dir
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 1
        etmipc1.low_mem = False
        etmipc1.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
        etmipc1.calculate_branch_scores(combine_clusters='sum')
        etmipc1.calculate_final_scores(combine_branches='sum')
        pool_init_write_score(curr_instance=etmipc1, curr_date=str(datetime.date.today()))
        branch1, time1 = write_score(branch=2)
        self.assertEqual(branch1, 2)
        self.assertGreater(time1, 0)
        self.assertTrue(os.path.isdir(os.path.join(etmipc1.output_dir, str(2))))
        self.assertTrue(os.path.isfile(os.path.join(etmipc1.output_dir, str(2),
                                                    "{}_{}_{}.all_scores.txt".format(str(datetime.date.today()),
                                                                                     etmipc1.alignment.query_id.split('_')[1], 2))))
        rmtree(os.path.join(etmipc1.output_dir, str(2)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = out_dir
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 6
        etmipc2.low_mem = True
        etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
        etmipc2.calculate_branch_scores(combine_clusters='evidence_weighted')
        etmipc2.calculate_final_scores(combine_branches='average')
        pool_init_write_score(curr_instance=etmipc2, curr_date=str(datetime.date.today()))
        branch2, time2 = write_score(branch=2)
        self.assertEqual(branch2, 2)
        self.assertGreater(time2, 0)
        self.assertTrue(os.path.isdir(os.path.join(etmipc2.output_dir, str(2))))
        self.assertTrue(os.path.isfile(os.path.join(etmipc2.output_dir, str(2),
                                                    "{}_{}_{}.all_scores.txt".format(str(datetime.date.today()),
                                                                                     etmipc2.alignment.query_id.split(
                                                                                         '_')[1], 2))))
        for branch in etmipc2.tree_depth:
            rmtree(os.path.join(out_dir, str(branch)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))

    def test_write_out_scores(self):
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        out_dir = os.path.abspath('../Test/')
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = out_dir
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 1
        etmipc1.low_mem = False
        etmipc1.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
        etmipc1.calculate_branch_scores(combine_clusters='sum')
        etmipc1.calculate_final_scores(combine_branches='sum')
        etmipc1.write_out_scores(today=str(datetime.date.today()))
        for branch1 in etmipc1.tree_depth:
            self.assertTrue(os.path.isdir(os.path.join(etmipc1.output_dir, str(branch1))))
            self.assertTrue(os.path.isfile(os.path.join(
                etmipc1.output_dir, str(branch1),
                "{}_{}_{}.all_scores.txt".format(str(datetime.date.today()), etmipc1.alignment.query_id.split('_')[1],
                                                 branch1))))
            rmtree(os.path.join(etmipc1.output_dir, str(branch1)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = out_dir
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 6
        etmipc2.low_mem = True
        etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
        etmipc2.calculate_branch_scores(combine_clusters='evidence_weighted')
        etmipc2.calculate_final_scores(combine_branches='average')
        etmipc2.write_out_scores(today=str(datetime.date.today()))
        for branch2 in etmipc2.tree_depth:
            self.assertTrue(os.path.isdir(os.path.join(etmipc2.output_dir, str(branch2))))
            self.assertTrue(os.path.isfile(
                os.path.join(etmipc2.output_dir, str(branch2),
                             "{}_{}_{}.all_scores.txt".format(str(datetime.date.today()),
                                                              etmipc2.alignment.query_id.split('_')[1], branch2))))
            rmtree(os.path.join(out_dir, str(branch2)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))

    def test_clear_intermediate_files(self):
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        out_dir = os.path.abspath('../Test/')
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        etmipc1.tree_depth = (2, 5)
        etmipc1.output_dir = out_dir
        etmipc1.import_alignment(query='1c17A', ignore_alignment_size=True)
        etmipc1.processes = 1
        etmipc1.low_mem = False
        etmipc1.calculate_cluster_scores(evidence=False, aa_dict=aa_dict)
        etmipc1.clear_intermediate_files()
        etmipc1.calculate_branch_scores(combine_clusters='sum')
        etmipc1.clear_intermediate_files()
        etmipc1.calculate_final_scores(combine_branches='sum')
        etmipc1.clear_intermediate_files()
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))
        etmipc2 = ETMIPC('../Test/1h1vA.fa')
        etmipc2.tree_depth = (2, 5)
        etmipc2.output_dir = out_dir
        etmipc2.import_alignment(query='1h1vA')
        etmipc2.processes = 6
        etmipc2.low_mem = True
        etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
        for tree_pos in etmipc2.unique_clusters:
            self.assertTrue(os.path.isfile(etmipc2.unique_clusters[tree_pos]['cluster_scores']))
            self.assertTrue(os.path.isfile(etmipc2.unique_clusters[tree_pos]['nongap_counts']))
        etmipc2.clear_intermediate_files()
        for tree_pos in etmipc2.unique_clusters:
            self.assertFalse(os.path.isfile(etmipc2.unique_clusters[tree_pos]['cluster_scores']))
            self.assertFalse(os.path.isfile(etmipc2.unique_clusters[tree_pos]['nongap_counts']))
        etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
        etmipc2.calculate_branch_scores(combine_clusters='evidence_weighted')
        for branch in etmipc2.tree_depth:
            self.assertTrue(os.path.isfile(etmipc2.branch_scores[branch]))
        etmipc2.clear_intermediate_files()
        for branch in etmipc2.tree_depth:
            self.assertFalse(os.path.isfile(etmipc2.branch_scores[branch]))
        etmipc2.calculate_cluster_scores(evidence=True, aa_dict=aa_dict)
        etmipc2.calculate_branch_scores(combine_clusters='evidence_weighted')
        etmipc2.calculate_final_scores(combine_branches='average')
        for branch in etmipc2.tree_depth:
            self.assertTrue(os.path.isfile(etmipc2.scores[branch]))
            self.assertTrue(os.path.isfile(etmipc2.coverage[branch]))
        etmipc2.clear_intermediate_files()
        for branch in etmipc2.tree_depth:
            self.assertFalse(os.path.isfile(etmipc2.scores[branch]))
            self.assertFalse(os.path.isfile(etmipc2.coverage[branch]))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        rmtree(os.path.join(out_dir, 'joblib'))

    def test_calculate_scores(self):
        aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   '-']
        aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
        out_dir = os.path.abspath('../Test/')
        etmipc1 = ETMIPC('../Test/1c17A.fa')
        start1 = time()
        time1 = etmipc1.calculate_scores(curr_date=str(datetime.date.today()), query='1c17A', tree_depth=(2, 5),
                                         out_dir=out_dir, processes=1, ignore_alignment_size=True,
                                         clustering='agglomerative', clustering_args={'affinity': 'euclidean',
                                                                                      'linkage': 'ward'},
                                         aa_mapping=aa_dict, combine_clusters='sum',
                                         combine_branches='sum', del_intermediate=False, low_mem=False)
        end1 = time()
        print(time1)
        print(end1 - start1)
        self.assertLessEqual(time1, end1 - start1)
        for branch1 in etmipc1.tree_depth:
            self.assertTrue(os.path.isdir(os.path.join(etmipc1.output_dir, str(branch1))))
            self.assertTrue(os.path.isfile(os.path.join(
                etmipc1.output_dir, str(branch1),
                "{}_{}_{}.all_scores.txt".format(str(datetime.date.today()), etmipc1.alignment.query_id.split('_')[1],
                                                 branch1))))
        # Repeat test on serialized data.
        etmipc2 = ETMIPC('../Test/1c17A.fa')
        start2 = time()
        time2 = etmipc2.calculate_scores(curr_date=str(datetime.date.today()), query='1c17A', tree_depth=(2, 5),
                                         out_dir=out_dir, processes=1, ignore_alignment_size=True,
                                         clustering='agglomerative', clustering_args={'affinity': 'euclidean',
                                                                                      'linkage': 'ward'},
                                         aa_mapping=aa_dict, combine_clusters='sum',
                                         combine_branches='sum', del_intermediate=False, low_mem=False)
        end2 = time()
        self.assertGreaterEqual(time2, end2 - start2)
        for branch2 in etmipc2.tree_depth:
            self.assertTrue(os.path.isdir(os.path.join(etmipc1.output_dir, str(branch2))))
            self.assertTrue(os.path.isfile(os.path.join(
                etmipc1.output_dir, str(branch2),
                "{}_{}_{}.all_scores.txt".format(str(datetime.date.today()), etmipc2.alignment.query_id.split('_')[1],
                                                 branch2))))
        for branch2 in etmipc1.tree_depth:
            rmtree(os.path.join(etmipc1.output_dir, str(branch2)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        os.remove(os.path.join(os.path.abspath('../Test/'), '{}_cET-MIp.npz'.format('1c17A')))
        os.remove(os.path.join(os.path.abspath('../Test/'), '{}_cET-MIp.pkl'.format('1c17A')))
        rmtree(os.path.join(out_dir, 'joblib'))
        etmipc3 = ETMIPC('../Test/1h1vA.fa')
        start3 = time()
        time3 = etmipc3.calculate_scores(curr_date=str(datetime.date.today()), query='1h1vA', tree_depth=(2, 5),
                                         out_dir=out_dir, processes=6, ignore_alignment_size=False,
                                         clustering='agglomerative', clustering_args={'affinity': 'euclidean',
                                                                                      'linkage': 'ward'},
                                         aa_mapping=aa_dict, combine_clusters='evidence_weighted',
                                         combine_branches='average', del_intermediate=False, low_mem=True)
        end3 = time()
        self.assertLessEqual(time3, (end3 - start3))
        for branch3 in etmipc3.tree_depth:
            self.assertTrue(os.path.isdir(os.path.join(etmipc3.output_dir, str(branch3))))
            self.assertTrue(os.path.isfile(
                os.path.join(etmipc3.output_dir, str(branch3),
                             "{}_{}_{}.all_scores.txt".format(str(datetime.date.today()),
                                                              etmipc3.alignment.query_id.split('_')[1], branch3))))
        # Repeat test on serialized data.
        etmipc4 = ETMIPC('../Test/1h1vA.fa')
        start4 = time()
        time4 = etmipc4.calculate_scores(curr_date=str(datetime.date.today()), query='1h1vA', tree_depth=(2, 5),
                                         out_dir=out_dir, processes=6, ignore_alignment_size=False,
                                         clustering='agglomerative', clustering_args={'affinity': 'euclidean',
                                                                                      'linkage': 'ward'},
                                         aa_mapping=aa_dict, combine_clusters='evidence_weighted',
                                         combine_branches='average', del_intermediate=False, low_mem=True)
        end4 = time()
        self.assertGreaterEqual(time4, (end4 - start4))
        for branch4 in etmipc4.tree_depth:
            self.assertTrue(os.path.isdir(os.path.join(etmipc4.output_dir, str(branch4))))
            self.assertTrue(os.path.isfile(
                os.path.join(etmipc4.output_dir, str(branch4),
                             "{}_{}_{}.all_scores.txt".format(str(datetime.date.today()),
                                                              etmipc4.alignment.query_id.split('_')[1], branch4))))
        for branch4 in etmipc4.tree_depth:
            rmtree(os.path.join(out_dir, str(branch4)))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'ungapped_alignment.pkl'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'UngappedAlignment.fa'))
        os.remove(os.path.join(os.path.abspath('../Test/'), 'X.npz'))
        os.remove(os.path.join(os.path.abspath('../Test/'), '{}_cET-MIp.npz'.format('1h1vA')))
        os.remove(os.path.join(os.path.abspath('../Test/'), '{}_cET-MIp.pkl'.format('1h1vA')))
        rmtree(os.path.join(out_dir, 'joblib'))
