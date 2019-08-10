"""
Created on Sep 17, 2018

@author: dmkonecki
"""
import os
import pickle
import numpy as np
import pandas as pd
from time import time
from subprocess import Popen, PIPE
from Bio.Phylo.TreeConstruction import DistanceMatrix
from Bio.Align.Applications import ClustalwCommandline
from dotenv import find_dotenv, load_dotenv
from PhylogeneticTree import PhylogeneticTree
from AlignmentDistanceCalculator import convert_array_to_distance_matrix
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)


class ETMIPWrapper(object):
    """
    This class is intended as a wrapper for the original ET-MIp binary created by Angela Wilson for the publication:
        Sung Y-M, Wilkins AD, Rodriguez GJ, Wensel TG, Lichtarge O. Intramolecular allosteric communication in dopamine
        D2 receptor revealed by evolutionary amino acid covariation. Proceedings of the National Academy of Sciences of
        the United States of America. 2016;113(13):3539-3544. doi:10.1073/pnas.1516579113.
    It requires the use of a .env file at the project level, which describes the location of the WETC binary as well as
    the muscle alignment binary.

    This wrapper makes it possible to convert alignments (in fa format) to msf format, perform covariance analysis using
    the ET-MIp method on an msf formatted alignment, and to import the covariance raw and coverage scores from the
    analysis.

    Attributes:
        alignment (SeqAlignment): This variable holds a seq alignment object. This is mainly used to provide the path
        the to the alignment used, but is also used to determine the length of the query sequence. Because of this it is
        advised that the import_alignment() and remove_gaps() methods be run before the SeqAlignment instance is passed
        to ETMIPWrapper.
        msf_path (str): The file path to the location at which the msf formatted alignment to be analyzed is located. If
        the SeqAlignment object was created using an msf formatted alignments its filename variable will be the same as
        msf_path, other wise a different path will be found here.
        scores (numpy.array): A square matrix whose length on either axis is the query sequence length (without gaps).
        This matrix contains the raw covariance scores computed by the ET-MIp method.
        coverage (numpy.array): A square matrix whose length on either axis is the query sequence length (without gaps).
        This matrix contains the processed coverage covariance scores computed by the ET-MIp method.
        distance_matrix
        tree
        rank_group_assignments
        rank_scores
        rho
        entropy
        time (float): The number of seconds it took for the ET-MIp code to load the msf formatted file and calculate
        covariance scores.
    """

    def __init__(self, alignment):
        """
        This method instantiates an instance of the ETMIPWrapper class.

        Args:
            alignment (SeqAlignment): This is mainly used to provide the path the to the alignment used, but is also
            used to determine the length of the query sequence. Because of this it is advised that the
            import_alignment() method be run before the SeqAlignment instance is passed to ETMIPWrapper.
        """
        self.alignment = alignment
        self.msf_path = None
        self.scores = None
        self.coverage = None
        self.distance_matrix = None
        self.tree = None
        self.rank_group_assignments = None
        self.rank_scores = None
        self.rho = None
        self.entropy = None
        self.time = None

    def check_alignment(self, target_dir=None):
        """
        Check Alignment

        This method checks whether the alignment currently associated with the ETMIPWrapper object ends with '.msf'. If
        it does not the clustalw tool is used to convert the initial alignment (assumed to be in .fa format) to an .msf
        formatted alignment at the same location as the .fa alignment. The file path to this .msf alignment is stored in
        the msf_path variable. This function is dependent on the .env file at the project level containing the
        'CLUSTALW_PATH' variable describing the path to the muscle binary.

        Args:
            target_dir (str): The path to a directory where the msf alignment should be written if the passed in
            alignment is not an msf alignment.
        """
        if not self.alignment.file_name.endswith('.msf'):
            clustalw_path = os.environ.get('CLUSTALW_PATH')
            if target_dir is None:
                target_dir = os.path.dirname(self.alignment.file_name)
            old_file_name = os.path.basename(self.alignment.file_name)
            new_file_name = os.path.join(target_dir, '{}.msf'.format(old_file_name.split('.')[0]))
            if not os.path.isfile(new_file_name):
                c_line = ClustalwCommandline(clustalw_path, infile=self.alignment.file_name, convert=True,
                                             outfile=new_file_name, output='GCG')
                c_line()
            self.msf_path = new_file_name
        else:
            self.msf_path = self.alignment.file_name

    def import_rank_sores(self, out_dir, file_name_format='etc_out.rank_{}.tsv', rank_type='id'):
        """
        Import Rank Scores

        This function imports intermediate ranks files (ending in .tsv) to the self.rank_scores attribute.

        Args:
            out_dir (str): The path to the directory where the ET-MIp scores have been written.
            file_name_format (str): The format for the rank file name, a placeholder is left for the rank type.
            rank_type (str): The rank type ('identity', 'weak', etc.) for which to import scores.
        Return:
            np.array: Rank scores for each position in the analyzed alignment.
        """
        if not os.path.isdir(out_dir):
            raise ValueError('Provided directory does not exist: {}!'.format(out_dir))
        file_path1 = os.path.join(out_dir, file_name_format.format(rank_type))
        if not os.path.isfile(file_path1):
            raise ValueError('Provided directory does not contain expected rank file!')
        rank_df = pd.read_csv(file_path1, sep='\t', header=0, index_col=0)
        self.rank_scores = rank_df['Rank'].values

    def import_entropy_rank_sores(self, out_dir, file_name_format='etc_out.rank_{}_entropy.tsv', rank_type='plain'):
        """
        Import Entropy Rank Scores

        This function imports intermediate ranks files (ending in .tsv) to the self.rho and self.entropy attributes.

        Args:
            out_dir (str): The path to the directory where the ET-MIp scores have been written.
            file_name_format (str): The format for the rank file name, a placeholder is left for the rank type.
            rank_type (str): The rank type ('plain', etc.) for which to import scores.
        Return:
            np.array: Rank scores for each position in the analyzed alignment.
        """
        if not os.path.isdir(out_dir):
            raise ValueError('Provided directory does not exist: {}!'.format(out_dir))
        file_path1 = os.path.join(out_dir, file_name_format.format(rank_type))
        if not os.path.isfile(file_path1):
            raise ValueError('Provided directory does not contain expected rank file!')
        rank_df = pd.read_csv(file_path1, sep='\t', header=0, index_col=0)
        self.rho = rank_df['Rho'].values
        self.entropy = {}
        for i in range(1, self.alignment.size):
            self.entropy[i] = rank_df['Rank {} Entropy'.format(i)].values

    def import_distance_matrices(self, out_dir, prefix='etc_out'):
        """
        Import Distance Matrices

        This method looks for the files containing the alignment distances and identity distances computed by the ET-MIp
        code base. Not all versions of that code base produce the necessary files, if this is the case an exception will
        be raised.

        Args:
            out_dir (str): The path to the directory where the ET-MIp scores have been written.
        Returns:
            pd.DataFrame: Values computed by ET-MIp for the alignment (scoring matrix based) distance between sequences.
            pd.DataFrame: Values computed by ET-MIp for the identity distance between sequences.
            pd.DataFrame: Intermediate values used to compute the distances in the prior DataFrames. These values
            include the identity counts generated by the two different methods as well as the sequence lengths used for
            normalization.
        """
        if not os.path.isdir(out_dir):
            raise ValueError('Provided directory does not exist: {}!'.format(out_dir))
        file_path1 = os.path.join(out_dir, '{}.aln_dist.tsv'.format(prefix))
        file_path2 = os.path.join(out_dir, '{}.id_dist.tsv'.format(prefix))
        file_path3 = os.path.join(out_dir, '{}.debug.tsv'.format(prefix))
        if not os.path.isfile(file_path1) or not os.path.isfile(file_path2) or not os.path.isfile(file_path3):
            raise ValueError('Provided directory does not contain expected distance files!')
        aln_dist_df = pd.read_csv(file_path1, sep='\t', header=0, index_col=0)
        id_dist_df = pd.read_csv(file_path2, sep='\t', header=0, index_col=0)
        intermediate_df = pd.read_csv(file_path3, sep='\t', header=0, index_col=False, comment='%')
        array_data = np.asarray(aln_dist_df,dtype=float)
        self.distance_matrix = convert_array_to_distance_matrix(array_data, list(aln_dist_df.columns))
        return aln_dist_df, id_dist_df, intermediate_df

    def import_phylogenetic_tree(self, out_dir, file_name='etc_out.nhx'):
        """
        Import Phylogenetic Tree

        This function imports the nhx tree produced by ETC as well as its distance matrix.

        Args:
            out_dir (str): The path to the directory where the ETC tree has been written.
            file_name (str): The name of the nhx tree to import.
        """
        if not os.path.isdir(out_dir):
            raise ValueError('Provided directory does not exist: {}!'.format(out_dir))
        file_path1 = os.path.join(out_dir, file_name)
        if not os.path.isfile(file_path1):
            raise ValueError('Provided directory does not contain expected distance files!')
        tree = PhylogeneticTree(tree_building_method='custom', tree_building_args={'tree_path': file_path1})
        if self.distance_matrix is None:
            try:
                self.import_distance_matrices(out_dir=out_dir)
            except ValueError:
                self.distance_matrix = DistanceMatrix(names=self.alignment.seq_order)
        tree.construct_tree(dm=self.distance_matrix)
        self.tree = tree

    def import_assignments(self, out_dir):
        """
        Import Assignments
        Args:
            out_dir (str): The path to the directory where the ETC group and rank assignments have been written.
        """
        if not os.path.isdir(out_dir):
            raise ValueError('Provided directory does not exist: {}!'.format(out_dir))
        file_path1 = os.path.join(out_dir, 'etc_out.group.tsv')
        if not os.path.isfile(file_path1):
            raise ValueError('Provided directory does not contain expected distance files!')
        from time import time
        start = time()
        full_df = pd.read_csv(file_path1, sep='\t', comment='%', header=None, index_col=None,
                              names=range(1 + self.alignment.size))
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
                node_mapping[(self.alignment.seq_order.index(node.name) + 1) * -1] = node
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
        # print(rank_group_nodes)
        # raise ValueError('just checking')
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

    def import_intermediate_covariance_scores(self, prefix, out_dir):
        if not os.path.isdir(out_dir):
            raise ValueError('Provided directory does not exist: {}!'.format(out_dir))
        file_path1 = os.path.join(out_dir, '{}.all_rank_and_group_mip.tsv'.format(prefix))
        if not os.path.isfile(file_path1):
            raise ValueError('Provided directory does not contain expected intermediate file!')
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
                    out_dir=out_dir, node_name='R{}G{}'.format(rank, group), pos_type='pair', score_type='group',
                    metric='WETC_MI', low_memory=True)
                check2, intermediate_amii_arrays[rank][group] = check_numpy_array(
                    out_dir=out_dir, node_name='R{}G{}'.format(rank, group), pos_type='pair', score_type='group',
                    metric='WETC_AMIi', low_memory=True)
                check3, intermediate_amij_arrays[rank][group] = check_numpy_array(
                    out_dir=out_dir, node_name='R{}G{}'.format(rank, group), pos_type='pair', score_type='group',
                    metric='WETC_AMIj', low_memory=True)
                check4, intermediate_ami_arrays[rank][group] = check_numpy_array(
                    out_dir=out_dir, node_name='R{}G{}'.format(rank, group), pos_type='pair', score_type='group',
                    metric='WETC_AMI', low_memory=True)
                check5, intermediate_mip_arrays[rank][group] = check_numpy_array(
                    out_dir=out_dir, node_name='R{}G{}'.format(rank, group), pos_type='pair', score_type='group',
                    metric='WETC_MIp', low_memory=True)
                final_check = check1 and check2 and check3 and check4 and check5
                loaded &= final_check
        #
        if not loaded:
            intermediate_mip_rank_df = pd.read_csv(file_path1, sep='\t', header=0, index_col=None)
            if self.tree is None:
                self.import_phylogenetic_tree(out_dir=out_dir, file_name='{}.nhx'.format(prefix))
            if self.rank_group_assignments is None:
                self.rank_group_assignments = self.tree.assign_group_rank()
            size = len(str(self.alignment.query_sequence).replace('-', ''))
            pos_mapping = {}
            pos_counter = 0
            for i in range(self.alignment.seq_length):
                if self.alignment.query_sequence[i] != '-':
                    pos_mapping[i] = pos_counter
                    pos_counter += 1
            print('NON GAP SEQUENCE LENGTH: {}'.format(size))
            print('NON GAP POSITION MAPPING SIZE: {}'.format(len(pos_mapping)))
            print(pos_mapping)
            for rank in self.rank_group_assignments:
                for group in self.rank_group_assignments[rank]:
                    intermediate_mi_arrays[rank][group] = np.zeros((size, size))
                    intermediate_amii_arrays[rank][group] = np.zeros((size, size))
                    intermediate_amij_arrays[rank][group] = np.zeros((size, size))
                    intermediate_ami_arrays[rank][group] = np.zeros((size, size))
                    intermediate_mip_arrays[rank][group] = np.zeros((size, size))
            for ind in intermediate_mip_rank_df.index:
                row = intermediate_mip_rank_df.loc[ind, :]
                i = row['i'] - 1
                j = row['j'] - 1
                rank = row['rank']
                group = row['group']
                # Import MI values
                try:
                    group_mi = float(row['group_MI'])
                    if (i in pos_mapping) and (j in pos_mapping):
                        intermediate_mi_arrays[rank][group][pos_mapping[i], pos_mapping[j]] = group_mi
                    else:
                        print('Skipped MIp position has value: {}\ti: {}\tj: {}\tchar_i: {}\tchar_j: {}'.format(
                            group_mi, i, j, self.alignment.query_sequence[i], self.alignment.query_sequence[j]))
                except ValueError:
                    pass
                # Import APC Values
                # Import E[MIi] values
                try:
                    group_amii = float(row['E[MIi]'])
                    if (i in pos_mapping) and (j in pos_mapping):
                        intermediate_amii_arrays[rank][group][pos_mapping[i], pos_mapping[j]] = group_amii
                    else:
                        print('Skipped AMIi position has value: {}\ti: {}\tj: {}\tchar_i: {}\tchar_j: {}'.format(
                            group_amii, i, j, self.alignment.query_sequence[i], self.alignment.query_sequence[j]))
                except ValueError:
                    pass
                # Import E[MIj] values
                try:
                    group_amij = float(row['E[MIj]'])
                    if (i in pos_mapping) and (j in pos_mapping):
                        intermediate_amij_arrays[rank][group][pos_mapping[i], pos_mapping[j]] = group_amij
                    else:
                        print('Skipped AMIj position has value: {}\ti: {}\tj: {}\tchar_i: {}\tchar_j: {}'.format(
                            group_amij, i, j, self.alignment.query_sequence[i], self.alignment.query_sequence[j]))
                except ValueError:
                    pass
                # Import E[MI] values
                try:
                    group_ami = float(row['E[MI]'])
                    if (i in pos_mapping) and (j in pos_mapping):
                        intermediate_ami_arrays[rank][group][pos_mapping[i], pos_mapping[j]] = group_ami
                    else:
                        print('Skipped AMI position has value: {}\ti: {}\tj: {}\tchar_i: {}\tchar_j: {}'.format(
                            group_ami, i, j, self.alignment.query_sequence[i], self.alignment.query_sequence[j]))
                except ValueError:
                    pass
                # Import MIp values
                try:
                    group_mip = float(row['group_MIp'])
                    if (i in pos_mapping) and (j in pos_mapping):
                        intermediate_mip_arrays[rank][group][pos_mapping[i], pos_mapping[j]] = group_mip
                    else:
                        print('Skipped MIp position has value: {}\ti: {}\tj: {}\tchar_i: {}\tchar_j: {}'.format(
                            group_mip, i, j, self.alignment.query_sequence[i], self.alignment.query_sequence[j]))
                except ValueError:
                    pass
            for rank in self.rank_group_assignments:
                for group in self.rank_group_assignments[rank]:
                    intermediate_mi_arrays[rank][group] = save_numpy_array(
                        mat=intermediate_mi_arrays[rank][group], out_dir=out_dir, node_name='R{}G{}'.format(rank, group),
                        pos_type='pair', score_type='group', metric='WETC_MI', low_memory=True)
                    intermediate_amii_arrays[rank][group] = save_numpy_array(
                        mat=intermediate_amii_arrays[rank][group], out_dir=out_dir, node_name='R{}G{}'.format(rank, group),
                        pos_type='pair', score_type='group', metric='WETC_AMIi', low_memory=True)
                    intermediate_amij_arrays[rank][group] = save_numpy_array(
                        mat=intermediate_amij_arrays[rank][group], out_dir=out_dir, node_name='R{}G{}'.format(rank, group),
                        pos_type='pair', score_type='group', metric='WETC_AMIj', low_memory=True)
                    intermediate_ami_arrays[rank][group] = save_numpy_array(
                        mat=intermediate_ami_arrays[rank][group], out_dir=out_dir, node_name='R{}G{}'.format(rank, group),
                        pos_type='pair', score_type='group', metric='WETC_AMI', low_memory=True)
                    intermediate_mip_arrays[rank][group] = save_numpy_array(
                        mat=intermediate_mip_arrays[rank][group], out_dir=out_dir, node_name='R{}G{}'.format(rank, group),
                        pos_type='pair', score_type='group', metric='WETC_MIp', low_memory=True)
        return (intermediate_mi_arrays, intermediate_amii_arrays, intermediate_amij_arrays, intermediate_ami_arrays,
                intermediate_mip_arrays)

    def import_covariance_scores(self, prefix, out_dir):
        """
        Import Covariance Scores

        This method looks for the etc_out.tree_mip_sorted file in the directory where the ET-MIp scores were calculated
        and written to file. This file is then imported using Pandas and is then used to fill in the scores and
        coverage matrices.

        Args:
            out_dir (str): The path to the directory where the ET-MIp scores have been written.
        Raises:
            ValueError: If the directory does not exist, or the expected file is not found in that directory.
        """
        if not os.path.isdir(out_dir):
            raise ValueError('Provided directory does not exist: {}!'.format(out_dir))
        file_path = os.path.join(out_dir, '{}.tree_mip_sorted'.format(prefix))
        if not os.path.isfile(file_path):
            raise ValueError('Provided directory does not contain expected covariance file!')
        data = pd.read_csv(file_path, comment='%', sep='\s+', names=['Sort', 'Res_i', 'i(AA)', 'Res_j', 'j(AA)',
                                                                     'Raw_Scores', 'Coverage_Scores', 'Interface',
                                                                     'Contact', 'Number', 'Average_Contact'])
        size = len(str(self.alignment.query_sequence).replace('-', ''))
        self.scores = np.zeros((size, size))
        self.coverage = np.zeros((size, size))
        for ind in data.index:
            i = data.loc[ind, 'Res_i'] - 1
            j = data.loc[ind, 'Res_j'] - 1
            if i > j:
                str1 = 'Switching i: {} and j: {}'.format(i, j)
                j, i = i, j
                print(str1 + ' to i: {} and j:{}'.format(i, j))
            # self.scores[i, j] = self.scores[j, i] = data.loc[ind, 'Raw_Scores']
            self.scores[i, j] = data.loc[ind, 'Raw_Scores']
            # self.coverage[i, j] = self.coverage[j, i] = data.loc[ind, 'Coverage_Scores']
            self.coverage[i, j] = data.loc[ind, 'Coverage_Scores']

    def import_et_ranks(self, method, prefix, out_dir):
        """
        Import ET ranks

        This method looks for the etc_out.tree_mip_sorted file in the directory where the ET-MIp scores were calculated
        and written to file. This file is then imported using Pandas and is then used to fill in the scores and
        coverage matrices.

        Args:
            out_dir (str): The path to the directory where the ET-MIp scores have been written.
        Raises:
            ValueError: If the directory does not exist, or the expected file is not found in that directory.
        """
        if not os.path.isdir(out_dir):
            raise ValueError('Provided directory does not exist: {}!'.format(out_dir))
        file_path = os.path.join(out_dir, '{}.ranks'.format(prefix))
        if not os.path.isfile(file_path):
            raise ValueError('Provided directory does not contain expected covariance file!')
        columns = ["alignment#", "residue#", "type", "rank", "variability", "characters"]
        if method == 'rvET':
            columns.append('rho')
        columns.append('coverage')
        data = pd.read_csv(file_path, comment='%', sep='\s+', index_col=None, names=columns)
        size = len(str(self.alignment.query_sequence).replace('-', ''))
        self.scores = np.zeros(size)
        for ind in data.index:
            i = data.loc[ind, 'residue#'] - 1
            self.scores[i] = data.loc[ind, 'rank']

    def import_scores(self, method, prefix, out_dir):
        if method == 'ET-MIp':
            self.import_covariance_scores(prefix=prefix, out_dir=out_dir)
        elif method == 'intET' or method == 'rvET':
            self.import_et_ranks(method=method, prefix=prefix, out_dir=out_dir)
        else:
            raise ValueError('import_scores does not support method: {}'.format(method))

    def calculate_scores(self, out_dir, method='ET-MIp', delete_files=True):
        """
        Calculated ET-MIp Scores

        This method uses the ET-MIp binary to compute covariance scores on an msf formatted multiple sequence alignment.
        The code requires a .env at the project level which has a variable 'WEC_PATH' that describes the location of the
        WETC binary. The method makes use of import_covariance_scores() to load the data produced by the run.

        Args:
            out_dir (str): The path to the directory where the ET-MIp scores should be written.
            delete_files (boolean): If True all of the files written out by calling this method will be deleted after
            importing the relevant data, if False all files will be left in the specified out_dir.
        Raises:
            ValueError: If the directory does not exist, or if the file expected to be created by this method is not
            found in that directory.
        """
        prefix = '{}_{}'.format('etc_out', method)
        serialized_path1 = os.path.join(out_dir, '{}.npz'.format(method))
        serialized_path2 = os.path.join(out_dir, '{}.pkl'.format(method))
        if os.path.isfile(serialized_path1) and os.path.isfile(serialized_path2):
            loaded_data = np.load(serialized_path1)
            self.scores = loaded_data['scores']
            self.coverage = loaded_data['coverage']
            self.time = loaded_data['time']
            with open(serialized_path2, 'rb') as handle:
                self.tree, self.rank_group_assignments = pickle.load(handle)
        else:
            self.check_alignment(target_dir=out_dir)
            binary_path = os.environ.get('WETC_PATH')
            start = time()
            current_dir = os.getcwd()
            os.chdir(out_dir)
            # Call binary
            options = [binary_path, '-p', self.msf_path, '-x', self.alignment.query_id, '-o', prefix]
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
            self.import_phylogenetic_tree(file_name='{}.nhx'.format(prefix), out_dir=out_dir)
            self.rank_group_assignments = self.tree.assign_group_rank()
            self.import_scores(prefix=prefix, out_dir=out_dir, method=method)
            if delete_files:
                self.remove_ouptut(out_dir=out_dir)
            np.savez(serialized_path1, time=self.time, scores=self.scores, coverage=self.coverage)
            with open(serialized_path2, 'wb') as handle:
                pickle.dump((self.tree, self.rank_group_assignments), handle, pickle.HIGHEST_PROTOCOL)
        print(self.time)
        return self.time

    @staticmethod
    def remove_ouptut(out_dir):
        """
        Remove Output

        This method will take the directory where ET-MIp output has been written and remove all of the files which are
        generated by the code.

        Args:
            out_dir (str): The path to the directory where the ET-MIp scores have been written.
        Raises:
            ValueError: If the directory does not exist.
        """
        prefix = 'etc_out'
        suffixes = ['aa_freqs', 'allpair_ranks', 'allpair_ranks_sorted', 'auc', 'average_ranks_sorted',
                    'covariation_matrix', 'entro.heatmap', 'entroMI.heatmap', 'mip_sorted', 'MI_sorted', 'nhx',
                    'pairs_allpair_ranks_sorted', 'pairs_average_ranks_sorted', 'pairs_mip_sorted',
                    'pairs_tree_mip_sorted', 'pairs_MI_sorted', 'pairs_tre_mip_sorted', 'pss.nhx', 'rank_matrix',
                    'ranks', 'ranks_sorted', 'rv.heatmap', 'rvMI.heatmap', 'tree_mip_matrix', 'tree_mip_sorted',
                    'tree_mip_top40_matrix']
        if not os.path.isdir(out_dir):
            raise ValueError('Provided directory does not exist: {}!'.format(out_dir))
        for suffix in suffixes:
            curr_path = os.path.join(out_dir, '{}.{}'.format(prefix, suffix))
            if os.path.isfile(curr_path):
                os.remove(curr_path)
