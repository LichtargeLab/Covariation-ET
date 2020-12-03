"""
Created on Sep 17, 2018

@author: dmkonecki
"""
import os
import pickle
import numpy as np
import pandas as pd
from time import time
from scipy.stats import rankdata
from subprocess import Popen, PIPE
from Bio.Phylo.TreeConstruction import DistanceMatrix
from Bio.Align.Applications import ClustalwCommandline
from dotenv import find_dotenv, load_dotenv
from Predictor import Predictor
from PhylogeneticTree import PhylogeneticTree
from AlignmentDistanceCalculator import convert_array_to_distance_matrix
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)


class ETMIPWrapper(Predictor):
    """
    This class is intended as a wrapper for the original ET-MIp binary created by Angela Wilson for the publication:
        Sung Y-M, Wilkins AD, Rodriguez GJ, Wensel TG, Lichtarge O. Intramolecular allosteric communication in dopamine
        D2 receptor revealed by evolutionary amino acid covariation. Proceedings of the National Academy of Sciences of
        the United States of America. 2016;113(13):3539-3544. doi:10.1073/pnas.1516579113.
    It requires the use of a .env file at the project level, which describes the location of the WETC binary as well as
    the ClustalW alignment binary.

    This wrapper makes it possible to convert alignments (in fa format) to msf format, perform covariance analysis using
    the ET-MIp method on an msf formatted alignment, and to import the covariance raw and coverage scores from the
    analysis.

    This class inherits from the Predictor class which means it shares the attributes and initializer of that class,
    as well as implementing the calculate_scores method.

    Attributes:
        out_dir (str): The path where results of this analysis should be written to.
        query (str): The sequence identifier for the sequence being analyzed.
        original_aln (SeqAlignment): A SeqAlignment object representing the alignment originally passed in.
        original_aln_fn (str): The path to the alignment to analyze.
        non_gapped_aln (SeqAlignment): SeqAlignment object representing the original alignment with all columns which
        are gaps in the query sequence removed.
        non_gapped_aln_fn (str): Path to where the non-gapped alignment is written.
        msf_aln_fn (str): The path to the location at which the msf formatted alignment to be analyzed is located.
        method (str): 'WETC' for all instances of this class, because it describes how the importance/covariances scores
        are being computed.
        scores (np.array): The raw scores calculated for each single or paired position in the provided alignment by
        wetc.
        coverages (np.array): The percentage of scores at or better than the score for this single or paired position
        (i.e. the percentile rank).
        rankings (np.array): The rank (lowest being best, highest being worst) of each single or paired position in the
        provided alignment as determined from the calculated scores.
        time (float): The time (in seconds) required to complete the computation of importance/covariance scores by
        wetc.
        tree - (PhylogeneticTree): The tree structure computed from the distance matrix during the ET computation.
        rank_group_assignments - (dict): First level of the dictionary maps a rank to another dictionary. The second
        level of the dictionary maps a group value to another dictionary. This third level of the dictionary maps the
        key 'node' to the node which is the root of the group at the given rank, 'terminals' to a list of node names for
        the leaf/terminal nodes which are ancestors of the root node, and 'descendants' to a list of nodes which are
        descendants of 'node' from the closest assigned rank (.i.e. the nodes children, at the lowest rank this will be
        None).
        rank_scores - (numpy.array): A square matrix whose length on either axis is the query sequence length (without
        gaps). The contents of the matrix are scores from specific ranks in the prediction computation.
        rho - (numpy.array): A square matrix whose length on either axis is the query sequence length (without
        gaps). The contents of the matrix are rho scores from the EvolutionaryTrace analysis.
        entropy - (dict): A dictionary where the key is a rank in the tree (int) and the values is a square matrix whose
        length on either axis is the query sequence length (without gaps). The contents of the matrix are entropy scores
        at that rank/level in the tree.
    """

    def __init__(self, query, aln_file, out_dir='.', polymer_type='Protein'):
        """
        __init__

        The initialization function for the ETMIPWrapper class which draws on its parent (Predictor)
        initialization.

        Arguments:
            query (str): The sequence identifier for the sequence being analyzed.
            aln_file (str): The path to the alignment to analyze, the file is expected to be in fasta format.
            out_dir (str): The path where results of this analysis should be written to. If no path is provided the
            default will be to write results to the current working directory.
            polymer_type (str): What kind of sequence information is being analyzed (.i.e. Protein or DNA).
        """
        super().__init__(query, aln_file, polymer_type, out_dir)
        self.msf_aln_fn = None
        self.distance_matrix = None
        self.tree = None
        self.rank_group_assignments = None
        self.rho = None
        self.rank_scores = None
        self.entropy = None
        self.method = 'WETC'

    def convert_alignment(self):
        """
        Convert Alignment

        This method converts the alignment currently associated with the ETMIPWrapper and uses the clustalw tool to
        convert the initial alignment (assumed to be in fasta format) to an .msf formatted alignment in the out_dir. The
        file path to this .msf alignment is stored in the msf_path variable. This function is dependent on the .env file
        at the project level containing the 'CLUSTALW_PATH' variable describing the path to the clustalw binary.
        """
        clustalw_path = os.environ.get('CLUSTALW_PATH')
        new_file_name = os.path.join(self.out_dir, 'Non-Gapped_Alignment.msf')
        if not os.path.isfile(new_file_name):
            c_line = ClustalwCommandline(clustalw_path, infile=self.non_gapped_aln_fn, convert=True,
                                         outfile=new_file_name, output='GCG')
            c_line()
        self.msf_aln_fn = new_file_name

    def import_rank_scores(self, file_name_format='etc_out.rank_{}.tsv', rank_type='id'):
        """
        Import Rank Scores

        This function imports intermediate ranks files (ending in .tsv) to the self.rank_scores attribute.

        Args:
            file_name_format (str): The format for the rank file name, a placeholder is left for the rank type.
            rank_type (str): The rank type ('identity', 'weak', etc.) for which to import scores.
        Return:
            np.array: Rank scores for each position in the analyzed alignment.
        """
        if not os.path.isdir(self.out_dir):
            raise ValueError('Provided directory does not exist: {}!'.format(self.out_dir))
        file_path1 = os.path.join(self.out_dir, file_name_format.format(rank_type))
        if not os.path.isfile(file_path1):
            raise ValueError('Provided directory does not contain expected rank file!')
        rank_df = pd.read_csv(file_path1, sep='\t', header=0, index_col=0)
        self.rank_scores = rank_df['Rank'].values

    def import_entropy_rank_sores(self, file_name_format='etc_out.rank_{}_entropy.tsv', rank_type='plain'):
        """
        Import Entropy Rank Scores

        This function imports intermediate ranks files (ending in .tsv) to the self.rho and self.entropy attributes.

        Args:
            file_name_format (str): The format for the rank file name, a placeholder is left for the rank type.
            rank_type (str): The rank type ('plain', etc.) for which to import scores.
        Return:
            np.array: Rank scores for each position in the analyzed alignment.
        """
        file_path1 = os.path.join(self.out_dir, file_name_format.format(rank_type))
        if not os.path.isfile(file_path1):
            raise ValueError('Provided directory does not contain expected rank file!')
        rank_df = pd.read_csv(file_path1, sep='\t', header=0, index_col=0)
        self.rho = rank_df['Rho'].values
        self.entropy = {}
        for i in range(1, self.non_gapped_aln.size):
            self.entropy[i] = rank_df['Rank {} Entropy'.format(i)].values

    def import_distance_matrices(self, prefix='etc_out'):
        """
        Import Distance Matrices

        This method looks for the files containing the alignment distances and identity distances computed by the ET-MIp
        code base. Not all versions of that code base produce the necessary files, if this is the case an exception will
        be raised.

        Args:
            prefix (str): The file prefix to prepend to the distance files (WETC -o option).
        Returns:
            pd.DataFrame: Values computed by ET for the alignment (scoring matrix based) distance between sequences.
            pd.DataFrame: Values computed by ET for the identity distance between sequences.
            pd.DataFrame: Intermediate values used to compute the distances in the prior DataFrames. These values
            include the identity counts generated by the two different methods as well as the sequence lengths used for
            normalization.
        """
        file_path1 = os.path.join(self.out_dir, '{}.aln_dist.tsv'.format(prefix))
        file_path2 = os.path.join(self.out_dir, '{}.id_dist.tsv'.format(prefix))
        file_path3 = os.path.join(self.out_dir, '{}.debug.tsv'.format(prefix))
        if not os.path.isfile(file_path1) or not os.path.isfile(file_path2) or not os.path.isfile(file_path3):
            raise ValueError('Provided directory does not contain expected distance files!\n{}\n{}\n{}'.format(
                file_path1, file_path2, file_path3))
        aln_dist_df = pd.read_csv(file_path1, sep='\t', header=0, index_col=0)
        id_dist_df = pd.read_csv(file_path2, sep='\t', header=0, index_col=0)
        intermediate_df = pd.read_csv(file_path3, sep='\t', header=0, index_col=False, comment='%')
        self.distance_matrix = convert_array_to_distance_matrix(aln_dist_df.values, list(aln_dist_df.columns))
        return aln_dist_df, id_dist_df, intermediate_df

    def import_phylogenetic_tree(self, prefix='etc_out'):
        """
        Import Phylogenetic Tree

        This function imports the nhx tree produced by ETC as well as its distance matrix.

        Args:
            prefix (str): The file prefix to prepend to the nhx tree file (WETC -o option).
        """
        file_path1 = os.path.join(self.out_dir, '{}.nhx'.format(prefix))
        if not os.path.isfile(file_path1):
            raise ValueError('Provided directory does not contain expected distance files!')
        tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': file_path1})
        if self.distance_matrix is None:
            print('DM IS NONE')
            try:
                print('IMPORTING DM')
                self.import_distance_matrices(prefix=prefix)
            except ValueError:
                print('FAILED TO IMPORT DM')
                self.distance_matrix = DistanceMatrix(names=self.non_gapped_aln.seq_order)
        tree.construct_tree(dm=self.distance_matrix)
        self.tree = tree

    def import_assignments(self, prefix='etc_out'):
        """
        Import Assignments

        This function imports the rank and group assignments for all nodes/sequences in the alignment and therefore in
        the tree attribute. Not all versions of that code base produce the necessary files, if this is the case an
        exception will be raised.

        Args:
            prefix (str): The file prefix to prepend to the nhx tree file (WETC -o option).
        """
        file_path1 = os.path.join(self.out_dir, '{}.group.tsv'.format(prefix))
        if not os.path.isfile(file_path1):
            raise ValueError('Provided directory does not contain expected distance files!')
        if self.tree is None:
            self.import_phylogenetic_tree(prefix=prefix)
        from time import time
        start = time()
        full_df = pd.read_csv(file_path1, sep='\t', comment='%', header=None, index_col=None,
                              names=range(1 + self.non_gapped_aln.size))
        table_names = ['Mapping', 'Nodes']
        tables = full_df[0].isin(['Rank']).cumsum()
        table_dict = {table_names[k - 1]: t.iloc[:] for k, t in full_df.groupby(tables)}
        rank_group_assignments = {}
        inter1 = time()
        print('Reading in the csv took {} min'.format((inter1 - start) / 60.0))
        rank_group_mapping = table_dict['Mapping']
        rank_group_mapping.rename(columns={i: label for i, label in enumerate(rank_group_mapping.iloc[0])},
                                  inplace=True)
        rank_group_mapping.drop(index=0, inplace=True)
        rank_group_mapping = rank_group_mapping.astype(int)
        rank_group_mapping.set_index('Rank', inplace=True)
        for rank in rank_group_mapping.index:
            if rank not in rank_group_assignments:
                rank_group_assignments[rank] = {}
            for terminal in rank_group_mapping.columns:
                group = rank_group_mapping.at[rank, terminal]
                if group not in rank_group_assignments[rank]:
                    rank_group_assignments[rank][group] = {'node': None, 'terminals': []}
                rank_group_assignments[rank][group]['terminals'].append(terminal)
        inter2 = time()
        print('Importing table 1 took {} min'.format((inter2 - inter1) / 60.0))
        node_mapping = {}
        for node in self.tree.traverse_top_down():
            if node.is_terminal():
                node_mapping[(self.non_gapped_aln.seq_order.index(node.name) + 1) * -1] = node
            else:
                node_mapping[int(node.name.strip('Inner'))] = node
        inter3 = time()
        print('Mapping nodes took {} min'.format((inter3 - inter2) / 60.0))
        rank_group_nodes = table_dict['Nodes']
        rank_group_nodes.dropna(axis=1, how='all', inplace=True)
        rank_group_nodes.rename(columns={i: label for i, label in enumerate(rank_group_nodes.iloc[0])}, inplace=True)
        rank_group_nodes.drop(index=rank_group_nodes.index[0], inplace=True)
        rank_group_nodes = rank_group_nodes.astype(int)
        rank_group_nodes.set_index(['Rank', 'Group'], inplace=True)
        for rank in rank_group_nodes.index.get_level_values('Rank'):
            for group in rank_group_nodes.index.get_level_values('Group'):
                node_index = int(rank_group_nodes.at[(rank, group), 'Root_Node'])
                if node_index == 0:
                    break
                if rank not in rank_group_assignments:
                    raise ValueError('Rank: {} in node table but not mapping! Group: {} Node: {}'.format(rank, group,
                                                                                                         node_index))
                node = node_mapping[node_index]
                rank_group_assignments[rank][group]['node'] = node
                if node.is_terminal():
                    rank_group_assignments[rank][group]['descendants'] = None
                else:
                    rank_group_assignments[rank][group]['descendants'] = node.clades
        end = time()
        print('Importing table 2 took {} min'.format((end - inter3) / 60.0))
        self.rank_group_assignments = rank_group_assignments

    def import_intermediate_covariance_scores(self, prefix='etc_out'):
        """
        Import Intermediate Covariance Scores

        This function imports the intermediate covariance scores computed for ranks and groups while computing
        covariance with ET. Not all versions of that code base produce the necessary files, if this is the case an
        exception will be raised.

        Args:
            prefix (str): The file prefix to prepend to the nhx tree file (WETC -o option).
        """
        file_path1 = os.path.join(self.out_dir, '{}.all_rank_and_group_mip.tsv'.format(prefix))
        if not os.path.isfile(file_path1):
            print('File not found, searching for serialized files.')
            # raise ValueError('Provided directory does not contain expected intermediate file!')
        loaded = True
        #
        from src.SupportingClasses.Trace import save_numpy_array, check_numpy_array
        intermediate_mi_arrays = {}
        intermediate_amii_arrays = {}
        intermediate_amij_arrays = {}
        intermediate_ami_arrays = {}
        intermediate_mip_arrays = {}
        for rank in self.rank_group_assignments:
            intermediate_mi_arrays[rank] = {}
            intermediate_amii_arrays[rank] = {}
            intermediate_amij_arrays[rank] = {}
            intermediate_ami_arrays[rank] = {}
            intermediate_mip_arrays[rank] = {}
            for group in self.rank_group_assignments[rank]:
                check1, intermediate_mi_arrays[rank][group] = check_numpy_array(
                    out_dir=self.out_dir, node_name='R{}G{}'.format(rank, group), pos_type='pair', score_type='group',
                    metric='WETC_MI', low_memory=True)
                check2, intermediate_amii_arrays[rank][group] = check_numpy_array(
                    out_dir=self.out_dir, node_name='R{}G{}'.format(rank, group), pos_type='pair', score_type='group',
                    metric='WETC_AMIi', low_memory=True)
                check3, intermediate_amij_arrays[rank][group] = check_numpy_array(
                    out_dir=self.out_dir, node_name='R{}G{}'.format(rank, group), pos_type='pair', score_type='group',
                    metric='WETC_AMIj', low_memory=True)
                check4, intermediate_ami_arrays[rank][group] = check_numpy_array(
                    out_dir=self.out_dir, node_name='R{}G{}'.format(rank, group), pos_type='pair', score_type='group',
                    metric='WETC_AMI', low_memory=True)
                check5, intermediate_mip_arrays[rank][group] = check_numpy_array(
                    out_dir=self.out_dir, node_name='R{}G{}'.format(rank, group), pos_type='pair', score_type='group',
                    metric='WETC_MIp', low_memory=True)
                final_check = check1 and check2 and check3 and check4 and check5
                loaded &= final_check
        #
        if not loaded and not os.path.isfile(file_path1):
            raise ValueError('Provided directory does not contain expected intermediate file or serialized files!')
        if not loaded:
            intermediate_mip_rank_df = pd.read_csv(file_path1, sep='\t', header=0, index_col=None)
            if self.tree is None:
                self.import_phylogenetic_tree(prefix=prefix)
            if self.rank_group_assignments is None:
                self.rank_group_assignments = self.tree.assign_group_rank()
            # size = len(str(self.non_gapped_aln.query_sequence).replace('-', ''))
            size = self.non_gapped_aln.seq_length

            # pos_mapping = {i: char for i, char in enumerate(self.non_gapped_aln.query_sequence)}

            # pos_counter = 0
            # for i in range(self.non_gapped_aln.seq_length):
            #     if self.non_gapped_aln.query_sequence[i] != '-':
            #         pos_mapping[i] = pos_counter
            #         pos_counter += 1
            print('NON GAP SEQUENCE LENGTH: {}'.format(size))
            # print('NON GAP POSITION MAPPING SIZE: {}'.format(len(pos_mapping)))
            # print(pos_mapping)
            for rank in self.rank_group_assignments:
                for group in self.rank_group_assignments[rank]:
                    intermediate_mi_arrays[rank][group] = np.zeros((size, size))
                    intermediate_amii_arrays[rank][group] = np.zeros((size, size))
                    intermediate_amij_arrays[rank][group] = np.zeros((size, size))
                    intermediate_ami_arrays[rank][group] = np.zeros((size, size))
                    intermediate_mip_arrays[rank][group] = np.zeros((size, size))
            for ind in intermediate_mip_rank_df.index:
                row = intermediate_mip_rank_df.loc[ind, :]
                i = int(row['i'] - 1)
                j = int(row['j'] - 1)
                rank = row['rank']
                group = row['group']
                # Import MI values
                try:
                    group_mi = float(row['group_MI'])
                    # print(f'GROUP_MI: {group_mi}')
                    # print(type(intermediate_mi_arrays))
                    # print(type(intermediate_mi_arrays[rank]))
                    # print(type(intermediate_mi_arrays[rank][group]))
                    # print(intermediate_mi_arrays[rank][group].shape)
                    # print(f'{i} - {j}')
                    intermediate_mi_arrays[rank][group][i, j] = group_mi
                    # if (i in pos_mapping) and (j in pos_mapping):
                    #     intermediate_mi_arrays[rank][group][pos_mapping[i], pos_mapping[j]] = group_mi
                    # else:
                    #     print('Skipped MIp position has value: {}\ti: {}\tj: {}\tchar_i: {}\tchar_j: {}'.format(
                    #         group_mi, i, j, self.non_gapped_aln.query_sequence[i],
                    #         self.non_gapped_aln.query_sequence[j]))
                except ValueError:
                    pass
                # Import APC Values
                # Import E[MIi] values
                try:
                    group_amii = float(row['E[MIi]'])
                    intermediate_amii_arrays[rank][group][i, j] = group_amii
                    # if (i in pos_mapping) and (j in pos_mapping):
                    #     intermediate_amii_arrays[rank][group][pos_mapping[i], pos_mapping[j]] = group_amii
                    # else:
                    #     print('Skipped AMIi position has value: {}\ti: {}\tj: {}\tchar_i: {}\tchar_j: {}'.format(
                    #         group_amii, i, j, self.non_gapped_aln.query_sequence[i],
                    #         self.non_gapped_aln.query_sequence[j]))
                except ValueError:
                    pass
                # Import E[MIj] values
                try:
                    group_amij = float(row['E[MIj]'])
                    intermediate_amij_arrays[rank][group][i, j] = group_amij
                    # if (i in pos_mapping) and (j in pos_mapping):
                    #     intermediate_amij_arrays[rank][group][pos_mapping[i], pos_mapping[j]] = group_amij
                    # else:
                    #     print('Skipped AMIj position has value: {}\ti: {}\tj: {}\tchar_i: {}\tchar_j: {}'.format(
                    #         group_amij, i, j, self.non_gapped_aln.query_sequence[i],
                    #         self.non_gapped_aln.query_sequence[j]))
                except ValueError:
                    pass
                # Import E[MI] values
                try:
                    group_ami = float(row['E[MI]'])
                    intermediate_ami_arrays[rank][group][i, j] = group_ami
                    # if (i in pos_mapping) and (j in pos_mapping):
                    #     intermediate_ami_arrays[rank][group][pos_mapping[i], pos_mapping[j]] = group_ami
                    # else:
                    #     print('Skipped AMI position has value: {}\ti: {}\tj: {}\tchar_i: {}\tchar_j: {}'.format(
                    #         group_ami, i, j, self.non_gapped_aln.query_sequence[i],
                    #         self.non_gapped_aln.query_sequence[j]))
                except ValueError:
                    pass
                # Import MIp values
                try:
                    group_mip = float(row['group_MIp'])
                    intermediate_mip_arrays[rank][group][i, j] = group_mip
                    # if (i in pos_mapping) and (j in pos_mapping):
                    #     intermediate_mip_arrays[rank][group][pos_mapping[i], pos_mapping[j]] = group_mip
                    # else:
                    #     print('Skipped MIp position has value: {}\ti: {}\tj: {}\tchar_i: {}\tchar_j: {}'.format(
                    #         group_mip, i, j, self.non_gapped_aln.query_sequence[i],
                    #         self.non_gapped_aln.query_sequence[j]))
                except ValueError:
                    pass
            for rank in self.rank_group_assignments:
                for group in self.rank_group_assignments[rank]:
                    intermediate_mi_arrays[rank][group] = save_numpy_array(
                        mat=intermediate_mi_arrays[rank][group], out_dir=self.out_dir,
                        node_name='R{}G{}'.format(rank, group), pos_type='pair', score_type='group', metric='WETC_MI',
                        low_memory=True)
                    intermediate_amii_arrays[rank][group] = save_numpy_array(
                        mat=intermediate_amii_arrays[rank][group], out_dir=self.out_dir,
                        node_name='R{}G{}'.format(rank, group), pos_type='pair', score_type='group', metric='WETC_AMIi',
                        low_memory=True)
                    intermediate_amij_arrays[rank][group] = save_numpy_array(
                        mat=intermediate_amij_arrays[rank][group], out_dir=self.out_dir,
                        node_name='R{}G{}'.format(rank, group), pos_type='pair', score_type='group', metric='WETC_AMIj',
                        low_memory=True)
                    intermediate_ami_arrays[rank][group] = save_numpy_array(
                        mat=intermediate_ami_arrays[rank][group], out_dir=self.out_dir,
                        node_name='R{}G{}'.format(rank, group), pos_type='pair', score_type='group', metric='WETC_AMI',
                        low_memory=True)
                    intermediate_mip_arrays[rank][group] = save_numpy_array(
                        mat=intermediate_mip_arrays[rank][group], out_dir=self.out_dir,
                        node_name='R{}G{}'.format(rank, group), pos_type='pair', score_type='group', metric='WETC_MIp',
                        low_memory=True)
        return (intermediate_mi_arrays, intermediate_amii_arrays, intermediate_amij_arrays, intermediate_ami_arrays,
                intermediate_mip_arrays)

    def import_covariance_scores(self, prefix='etc_out'):
        """
        Import Covariance Scores

        This method looks for the etc_out.tree_mip_sorted file in the directory where the ET-MIp scores were calculated
        and written to file. This file is then imported using Pandas and is then used to fill in the scores and
        coverage matrices.

        Args:
            prefix (str): The file prefix to prepend to the covariance score files (WETC -o option).
        Raises:
            ValueError: If the directory does not exist, or the expected file is not found in that directory.
        """
        file_path = os.path.join(self.out_dir, '{}.tree_mip_sorted'.format(prefix))
        if not os.path.isfile(file_path):
            raise ValueError('Provided directory does not contain expected covariance file!')
        data = pd.read_csv(file_path, comment='%', sep='\s+', names=['Sort', 'Res_i', 'i(AA)', 'Res_j', 'j(AA)',
                                                                     'Raw_Scores', 'Coverage_Scores', 'Interface',
                                                                     'Contact', 'Number', 'Average_Contact'])
        size = len(str(self.non_gapped_aln.query_sequence).replace('-', ''))
        self.scores = np.zeros((size, size))
        self.coverages = np.zeros((size, size))
        self.rankings = np.zeros((size, size))
        for ind in data.index:
            i = data.loc[ind, 'Res_i'] - 1
            j = data.loc[ind, 'Res_j'] - 1
            if i > j:
                str1 = 'Switching i: {} and j: {}'.format(i, j)
                j, i = i, j
                print(str1 + ' to i: {} and j:{}'.format(i, j))
            self.scores[i, j] = data.loc[ind, 'Raw_Scores']
            self.coverages[i, j] = data.loc[ind, 'Coverage_Scores']
            self.rankings[i, j] = data.loc[ind, 'Sort']

    def import_et_ranks(self, method, prefix='etc_out'):
        """
        Import ET ranks

        This method looks for the etc_out.ranks file in the directory where the ET scores were calculated and written to
        file. This file is then imported using Pandas and is then used to fill in the scores and coverage matrices.

        Args:
            method (str): Which method (rvET or intET) was used to generate the scores.
            prefix (str): The file prefix to prepend to the rank files (WETC -o option).
        Raises:
            ValueError: If the directory does not exist, or the expected file is not found in that directory.
        """
        file_path = os.path.join(self.out_dir, '{}.ranks'.format(prefix))
        if not os.path.isfile(file_path):
            raise ValueError('Provided directory does not contain expected covariance file!')
        columns = ["alignment#", "residue#", "type", "rank", "variability", "characters"]
        if method == 'rvET':
            columns.append('rho')
        columns.append('coverage')
        data = pd.read_csv(file_path, comment='%', sep='\s+', index_col=None, names=columns)
        size = len(str(self.non_gapped_aln.query_sequence).replace('-', ''))
        self.scores = np.zeros(size)
        self.coverages = np.zeros(size)
        for ind in data.index:
            i = data.loc[ind, 'residue#'] - 1
            self.scores[i] = data.loc[ind, 'rank']
            self.coverages[i] = data.loc[ind, 'coverage']
        self.rankings = rankdata(self.scores, method='dense')

    def import_scores(self, method, prefix='etc_out'):
        """
        Import Scores

        This method acts as a switch statement for importing scores and coverage data from ET predictions. Single
        position ET predictions (intET or rvET) are written to file in a slightly different format than the paired
        position predictions (ET-MIp), this method calls the proper import statement for the method specified.

        Args:
            method (str): Which method was used to generate results, current expected inputs are intET, rvET, or ET-MIp.
            prefix (str): The file prefix to prepend to the rank files (WETC -o option).
        """

        if method == 'ET-MIp':
            self.import_covariance_scores(prefix=prefix)
        elif method == 'intET' or method == 'rvET':
            self.import_et_ranks(method=method, prefix=prefix)
        else:
            raise ValueError('import_scores does not support method: {}'.format(method))

    def calculate_scores(self, method='ET-MIp', delete_files=True):
        """
        Calculated Scores

        This method uses the wetc binary (with options to run ET-MIp) to compute single position importance or
        covariance scores on an msf formatted multiple sequence alignment. The code requires a .env at the project level
        which has a variable 'WEC_PATH' that describes the location of the WETC binary. The method makes use of
        import_phylogenetic_tree, import_assignments, and import_scores() to load the data produced by the run (some of
        these data are only available for certain version of the binary, so if they cannot be imported placeholders are
        used or the python implementation is used to fill in the missing values).

        Args:
            method (str): Which method (intET, rvET, or ET-MIp) to use to generate scores.
            delete_files (boolean): If True all of the files written out by calling this method will be deleted after
            importing the relevant data, if False all files will be left in the specified out_dir.
        Raises:
            ValueError: If the directory does not exist, or if the file expected to be created by this method is not
            found in that directory.
        """
        prefix = '{}_{}'.format('etc_out', method)
        serialized_path1 = os.path.join(self.out_dir, '{}.npz'.format(method))
        serialized_path2 = os.path.join(self.out_dir, '{}.pkl'.format(method))
        if os.path.isfile(serialized_path1) and os.path.isfile(serialized_path2):
            loaded_data = np.load(serialized_path1)
            self.scores = loaded_data['scores']
            self.rankings = loaded_data['ranks']
            self.coverages = loaded_data['coverages']
            self.time = loaded_data['time']
            with open(serialized_path2, 'rb') as handle:
                self.distance_matrix, self.tree, self.rank_group_assignments = pickle.load(handle)
        else:
            self.convert_alignment()
            binary_path = os.environ.get('WETC_PATH')
            start = time()
            current_dir = os.getcwd()
            os.chdir(self.out_dir)
            # Call binary
            options = [binary_path, '-p', self.msf_aln_fn, '-x', self.non_gapped_aln.query_id, '-o', prefix]
            if method == 'ET-MIp':
                options.append('-allpairs')
            elif method == 'intET':
                options.append('-int')
            elif method == 'rvET':
                options.append('-realval')
            else:
                raise ValueError('ETMIPWrapper not implemented to calculate {} score type.'.format(method))
            p = Popen(options, stdout=PIPE, stderr=PIPE)
            # Retrieve communications from binary call
            out, error = p.communicate()
            end = time()
            self.time = end - start
            print('Output:')
            print(out)
            print('Error:')
            print(error)
            os.chdir(current_dir)
            self.import_phylogenetic_tree(prefix=prefix)
            try:
                self.import_assignments(prefix=prefix)
            except ValueError:
                self.rank_group_assignments = self.tree.assign_group_rank()
            self.import_scores(prefix=prefix, method=method)
            if delete_files:
                self.remove_ouptut(prefix=prefix)
            np.savez(serialized_path1, time=self.time, scores=self.scores, ranks=self.rankings,
                     coverages=self.coverages)
            with open(serialized_path2, 'wb') as handle:
                pickle.dump((self.distance_matrix, self.tree, self.rank_group_assignments), handle,
                            pickle.HIGHEST_PROTOCOL)
        print(self.time)
        return self.time

    def remove_output(self, prefix='etc_out'):
        """
        Remove Output

        This method will take the directory where ET-MIp output has been written and remove all of the files which are
        generated by the code.

        Args:
            prefix (str): The file prefix to prepend to the rank files (WETC -o option).
        Raises:
            ValueError: If the directory does not exist.
        """
        suffixes = ['aa_freqs', 'allpair_ranks', 'allpair_ranks_sorted', 'auc', 'average_ranks_sorted',
                    'covariation_matrix', 'entro.heatmap', 'entroMI.heatmap', 'mip_sorted', 'MI_sorted', 'nhx',
                    'pairs_allpair_ranks_sorted', 'pairs_average_ranks_sorted', 'pairs_mip_sorted',
                    'pairs_tree_mip_sorted', 'pairs_MI_sorted', 'pairs_tre_mip_sorted', 'pss.nhx', 'rank_matrix',
                    'ranks', 'ranks_sorted', 'rv.heatmap', 'rvMI.heatmap', 'tree_mip_matrix', 'tree_mip_sorted',
                    'tree_mip_top40_matrix']
        if not os.path.isdir(self.out_dir):
            raise ValueError('Provided directory does not exist: {}!'.format(self.out_dir))
        for suffix in suffixes:
            curr_path = os.path.join(self.out_dir, '{}.{}'.format(prefix, suffix))
            if os.path.isfile(curr_path):
                os.remove(curr_path)
