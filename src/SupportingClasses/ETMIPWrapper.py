"""
Created on Sep 17, 2018

@author: dmkonecki
"""
import os
import numpy as np
import pandas as pd
from time import time
from subprocess import Popen, PIPE
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

    def import_covariance_scores(self, out_dir):
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
        file_path = os.path.join(out_dir, 'etc_out.tree_mip_sorted')
        if not os.path.isfile(file_path):
            raise ValueError('Provided directory does not contain expected covariance file!')
        data = pd.read_csv(file_path, comment='%', sep='\s+', names=['Sort', 'Res_i', 'i(AA)', 'Res_j', 'j(AA)',
                                                                     'Raw_Scores', 'Coverage_Scores', 'Interface',
                                                                     'Contact', 'Number', 'Average_Contact'])
        self.scores = np.zeros((self.alignment.seq_length, self.alignment.seq_length))
        self.coverage = np.zeros((self.alignment.seq_length, self.alignment.seq_length))
        for ind in data.index:
            i = data.loc[ind, 'Res_i'] - 1
            j = data.loc[ind, 'Res_j'] - 1
            self.scores[i, j] = self.scores[j, i] = data.loc[ind, 'Raw_Scores']
            self.coverage[i, j] = self.coverage[j, i] = data.loc[ind, 'Coverage_Scores']

    def import_distance_matrices(self, out_dir):
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
        file_path1 = os.path.join(out_dir, 'etc_out.aln_dist.tsv')
        file_path2 = os.path.join(out_dir, 'etc_out.id_dist.tsv')
        file_path3 = os.path.join(out_dir, 'etc_out.debug.tsv')
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
            self.import_distance_matrices(out_dir=out_dir)
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
        end = time()
        print('Importing table 2 took {} min'.format((end - inter3) / 60.0))
        self.rank_group_assignments = rank_group_assignments

    def calculate_scores(self, out_dir, delete_files=True):
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
        serialized_path = os.path.join(out_dir, 'ET-MIp.npz')
        if os.path.isfile(serialized_path):
            loaded_data = np.load(serialized_path)
            self.scores = loaded_data['scores']
            self.coverage = loaded_data['coverage']
            self.time = loaded_data['time']
        else:
            self.check_alignment(target_dir=out_dir)
            binary_path = os.environ.get('WETC_PATH')
            start = time()
            current_dir = os.getcwd()
            os.chdir(out_dir)
            # Call binary
            p = Popen([binary_path, '-p', self.msf_path, '-x', self.alignment.query_id, '-allpairs'], stdout=PIPE,
                      stderr=PIPE)
            # Retrieve communications from binary call
            out, error = p.communicate()
            end = time()
            self.time = end - start
            print('Output:')
            print(out)
            print('Error:')
            print(error)
            os.chdir(current_dir)
            self.import_covariance_scores(out_dir=out_dir)
            if delete_files:
                self.remove_ouptut(out_dir=out_dir)
            np.savez(serialized_path, time=self.time, scores=self.scores, coverage=self.coverage)
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