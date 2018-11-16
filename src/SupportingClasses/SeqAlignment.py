"""
Created on Aug 17, 2017

@author: daniel
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from seaborn import heatmap, clustermap
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator
from sklearn.cluster import AgglomerativeClustering
from shutil import rmtree
import cPickle as pickle
from time import time
import pandas as pd
import numpy as np
import os


class SeqAlignment(object):
    """
    classdocs
    """

    def __init__(self, file_name, query_id):
        """
        Constructor

        Initiates an instance of the SeqAlignment class which stores the
        following data:

        file_name: str
            The file path to the file from which the alignment can be parsed.
        query_id: str
            The provided query_id prepended with >query_, which should be the
            identifier for query sequence in the alignment file.
        alignment: dict
            A dictionary mapping sequence IDs with their sequences as parsed
            from the alignment file.
        seq_order: list
            List of sequence ids in the order in which they were parsed from the
            alignment file.
        query_sequence: str
            The sequence matching the sequence identifier give by the query_id
            variable.
        seq_length: int
            The length of the query sequence.
        size: int
            The number of sequences in the alignment represented by this object.
        distance_matrix: np.array
            A matrix with the identity scores between sequences in the
            alignment.
        tree_order: list
            A list of query IDs ordered as they should be for sequence writing
            to file. This should reflect some purposeful ordering of the
            sequences such as sequence clustering/phylogeny.
        """
        if file_name.startswith('..'):
            file_name = os.path.abspath(os.path.join(os.getcwd(), file_name))
        self.file_name = file_name
        self.query_id = '>query_' + query_id
        self.alignment = None
        self.seq_order = None
        self.query_sequence = None
        self.seq_length = None
        self.size = None
        self.distance_matrix = None
        self.tree_order = None
        self.sequence_assignments = None

    def import_alignment(self, save_file=None, verbose=False):
        """
        Import alignments:

        This method imports the alignments into the class and forces all
        non-amino acids to take on the standard gap character "-".  This
        updates the alignment, seq_order, query_sequence, seq_length, and size
        class variables.

        Args:
            save_file (str): Path to file in which the desired alignment was stored previously.
        """
        if verbose:
            start = time()
        if (save_file is not None) and (os.path.exists(save_file)):
            alignment, seq_order, query_sequence = pickle.load(open(save_file, 'rb'))
        else:
            alignment = AlignIO.read(self.file_name, 'fasta')
            seq_order = []
            for record in alignment:
                seq_order.append(record.id)
                if record.id == self.query_id[1:]:
                    query_sequence = record.seq
            if save_file is not None:
                pickle.dump((alignment, seq_order, query_sequence), open(save_file, 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            end = time()
            print('Importing alignment took {} min'.format((end - start) / 60.0))
        self.alignment = alignment
        self.seq_order = seq_order
        self.query_sequence = query_sequence
        self.seq_length = len(self.query_sequence)
        self.size = len(self.alignment)

    def write_out_alignment(self, file_name):
        """
        This method writes out the alignment in the standard fa format.  Any
        sequence which is longer than 60 positions will be split over multiple
        lines with 60 characters per line.

        Args:
            file_name (str): Path to file where the alignment should be written.
        """
        start = time()
        AlignIO.write(self.alignment, file_name, "fasta")
        end = time()
        print('Writing out alignment took {} min'.format((end - start) / 60.0))

    def _subset_columns(self, indices_to_keep):
        """
        Subset Columns

        This is private method meant to subset an alignment to a specified set of positions, using the Bio.Align slicing
        syntax.

        Args:
            indices_to_keep (list, numpy.array, or iterable in sorted order): The indices to keep when creating a new
            alignment.
        Returns:
            Bio.Align. A new alignment which is a subset of self.alignment, specified by the passed in list or array.

        Example Usage:
        >>> self._subset_columns(query_ungapped_ind)
        """
        new_alignment = None
        start = None
        for i in range(self.seq_length):
            if start is None:
                if i in indices_to_keep:
                    start = i
            else:
                if i not in indices_to_keep:
                    if new_alignment is None:
                        new_alignment = self.alignment[:, start:i]
                    else:
                        new_alignment += self.alignment[:, start:i]
                    start = None
        if start is not None:
            if new_alignment is None:
                new_alignment = self.alignment[:, start:]
            else:
                new_alignment += self.alignment[:, start:]
        return new_alignment

    def remove_gaps(self, save_file=None):
        """
        Remove Gaps

        Removes all gaps from the query sequence and removes characters at the
        corresponding positions in all other sequences. This method updates the
        class variables alignment, query_sequence, and seq_length.

        Args:
            save_file (str): Path to a file where the alignment with gaps in the query sequence removed was stored
            previously.
        """
        start = time()
        if (save_file is not None) and os.path.exists(save_file):
            new_alignment = pickle.load(open(save_file, 'rb'))
        else:
            query_arr = np.array(list(self.query_sequence))
            query_ungapped_ind = np.where(query_arr != '-')[0]
            if len(query_ungapped_ind) > 0:
                new_alignment = self._subset_columns(query_ungapped_ind)
            else:
                pass
            if save_file is not None:
                pickle.dump(new_alignment, open(save_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        self.alignment = new_alignment
        self.query_sequence = self.alignment[self.seq_order.index(self.query_id[1:])].seq
        self.seq_length = len(self.query_sequence)
        end = time()
        print('Removing gaps took {} min'.format((end - start) / 60.0))

    def compute_distance_matrix(self, save_file=None):
        """
        Distance matrix

        Computes the sequence identity distance between a set of sequences and
        returns a matrix of the pairwise distances.  This method updates the
        distance_matrix class variable.

        Args:
            save_file (str): The path for an .npz file containing distances between sequences in the alignment (leave
            out the .npz as it will be added automatically.
        """
        start = time()
        if (save_file is not None) and os.path.exists(save_file + '.npz'):
            value_matrix = np.load(save_file + '.npz')['X']
        else:
            calculator = DistanceCalculator('identity')
            value_matrix = np.array(calculator.get_distance(self.alignment))
            if save_file is not None:
                np.savez(save_file, X=value_matrix)
        end = time()
        print('Computing the distance matrix took {} min'.format((end - start) / 60.0))
        self.distance_matrix = value_matrix

    def compute_effective_alignment_size(self, identity_threshold=0.62):
        """
        Compute Effective Alignment Size

        This method uses the distance_matrix variable (containing sequence identities) to compute the effective size of
        the current alignment. The equation (given below) and default threshold (62% identity) are taken from
        PMID:29047157.
            Meff = SUM_(i=0)^(N) of 1/n_i
            where n_i are the number of sequences sequence identity >= the identity threshold
        Args:
            identity_threshold (float): The threshold for what is considered an identical (non-unique) sequence.
        Returns:
            float. The effective alignment size of the current alignment (must be <= SeqAlignment.size)
        """
        if self.distance_matrix is None:
            self.compute_distance_matrix()
        meets_threshold = (1 - self.distance_matrix) >= identity_threshold
        meets_threshold[range(meets_threshold.shape[0]), range(meets_threshold.shape[1])] = True
        n_i = np.sum(meets_threshold, axis=1)
        rec_n_i = 1.0 / n_i
        effective_alignment_size = np.sum(rec_n_i)
        if effective_alignment_size > self.size:
            raise ValueError('Effective alignment size is greater than the original alignment size.')
        return effective_alignment_size

    # def agg_clustering(self, n_cluster, cache_dir):
    #     """
    #     Agglomerative clustering
    #
    #     Performs agglomerative clustering on a matrix of pairwise distances
    #     between sequences in the alignment being analyzed.
    #
    #     Args:
    #         n_cluster (int): The number of clusters to separate sequences into.
    #         cache_dir (str): The path to the directory where the clustering model can be stored for access later when
    #         identifying different numbers of clusters.
    #     Returns:
    #         dict. A dictionary with cluster number as the key and a list of sequences in the specified cluster as a
    #         value.
    #         set. A unique sorted set of the cluster values.
    #     """
    #     start = time()
    #     affinity = 'euclidean'
    #     linkage = 'ward'
    #     model = AgglomerativeClustering(affinity=affinity, linkage=linkage,
    #                                     n_clusters=n_cluster, memory=cache_dir,
    #                                     compute_full_tree=True)
    #     model.fit(self.distance_matrix)
    #     # unique and sorted list of cluster ids e.g. for n_clusters=2, g=[0,1]
    #     cluster_list = model.labels_.tolist()
    #     ################################################################################################################
    #     #       Mapping Clusters to Sequences
    #     ################################################################################################################
    #     cluster_labels = set(cluster_list)
    #     cluster_dict = {}
    #     cluster_ordering = {}
    #     for i in range(len(cluster_list)):
    #         seq_id = self.seq_order[i]
    #         index = self.tree_order.index(seq_id)
    #         key = cluster_list[i]
    #         if key not in cluster_dict:
    #             cluster_dict[key] = []
    #             cluster_ordering[key] = index
    #         if index < cluster_ordering[key]:
    #             cluster_ordering[key] = index
    #         cluster_dict[key].append(self.seq_order[i])
    #     sorted_cluster_labels = sorted(cluster_ordering, key=lambda k: cluster_ordering[k])
    #     cluster_dict2 = {}
    #     for i in range(len(cluster_labels)):
    #         cluster_dict2[i] = cluster_dict[sorted_cluster_labels[i]]
    #     end = time()
    #     print('Performing agglomerative clustering took {} min'.format((end - start) / 60.0))
    #     return cluster_dict2, cluster_labels

    def _agglomerative_clustering(self, n_cluster, cache_dir=None, affinity='euclidean', linkage='ward'):
        """
        Agglomerative clustering

        Performs agglomerative clustering on a matrix of pairwise distances
        between sequences in the alignment being analyzed.

        Args:
            n_cluster (int): The number of clusters to separate sequences into.
            cache_dir (str): The path to the directory where the clustering model can be stored for access later when
            identifying different numbers of clusters.
        Returns:
            dict. A dictionary with cluster number as the key and a list of sequences in the specified cluster as a
            value.
            set. A unique sorted set of the cluster values.
        """
        if self.distance_matrix is None:
            self.compute_distance_matrix()
        model = AgglomerativeClustering(affinity=affinity, linkage=linkage, n_clusters=n_cluster, memory=cache_dir,
                                        compute_full_tree=True)
        model.fit(self.distance_matrix)
        # unique and sorted list of cluster ids e.g. for n_clusters=2, g=[0,1]
        cluster_list = model.labels_.tolist()
        return cluster_list

    # def random_assignment(self, n_cluster):
    #     """
    #     random_assignment
    #
    #     Randomly assigns sequence IDs to groups totaling the specified
    #     nClusters. Sequences are split as evenly as possible with all clusters
    #     getting the same number of sequences if self.size % nClusters = 0 and
    #     with self.size % nClusters groups getting one additional sequence
    #     otherwise.
    #
    #     Args:
    #         nClusters (int): The number of clusters to produce by random assignment.
    #     Returns:
    #         dict. A dictionary mapping a cluster labels (0 to nClusters -1) to a list of sequence IDs assigned to that
    #         cluster.
    #         set. The set of labels used for clustering (0 to nClusters -1).
    #     """
    #     start = time()
    #     cluster_sizes = np.ones(n_cluster, dtype=np.int64)
    #     min_size = self.size / n_cluster
    #     cluster_sizes *= min_size
    #     larger_clusters = self.size % n_cluster
    #     if larger_clusters > 0:
    #         cluster_sizes[(-1 * larger_clusters):] += 1
    #     cluster_list = [0] * self.size
    #     choices = range(self.size)
    #     for i in range(n_cluster):
    #         curr_assignment = np.random.choice(choices, size=cluster_sizes[i])
    #         for j in curr_assignment:
    #             cluster_list[j] = i
    #         choices = list(set(choices) - set(curr_assignment))
    #     cluster_labels = set(cluster_list)
    #     cluster_dict = {}
    #     cluster_ordering = {}
    #     for i in range(len(cluster_list)):
    #         seq_id = self.seq_order[i]
    #         index = self.tree_order.index(seq_id)
    #         key = cluster_list[i]
    #         if key not in cluster_dict:
    #             cluster_dict[key] = []
    #             cluster_ordering[key] = index
    #         if index < cluster_ordering[key]:
    #             cluster_ordering[key] = index
    #         cluster_dict[key].append(self.seq_order[i])
    #     sorted_cluster_labels = sorted(cluster_ordering, key=lambda k: cluster_ordering[k])
    #     cluster_dict2 = {}
    #     for i in range(len(cluster_labels)):
    #         cluster_dict2[i] = cluster_dict[sorted_cluster_labels[i]]
    #     end = time()
    #     print('Performing agglomerative clustering took {} min'.format((end - start) / 60.0))
    #     return cluster_dict2, cluster_labels

    def _random_assignment(self, n_cluster, cache_dir=None):
        """
        random_assignment

        Randomly assigns sequence IDs to groups totaling the specified
        nClusters. Sequences are split as evenly as possible with all clusters
        getting the same number of sequences if self.size % nClusters = 0 and
        with self.size % nClusters groups getting one additional sequence
        otherwise.

        Args:
            nClusters (int): The number of clusters to produce by random assignment.
        Returns:
            dict. A dictionary mapping a cluster labels (0 to nClusters -1) to a list of sequence IDs assigned to that
            cluster.
            set. The set of labels used for clustering (0 to nClusters -1).
        """
        if cache_dir is not None:
            save_dir = os.path.join(cache_dir, 'joblib')
        else:
            save_dir = None
        if save_dir is not None:
            save_file = os.path.join(save_dir, 'K_{}.pkl'.format(n_cluster))
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            else:
                if os.path.isfile(save_file):
                    return pickle.load(save_file)
        cluster_sizes = np.ones(n_cluster, dtype=np.int64)
        min_size = self.size // n_cluster
        cluster_sizes *= min_size
        larger_clusters = self.size % n_cluster
        if larger_clusters > 0:
            cluster_sizes[(-1 * larger_clusters):] += 1
        cluster_list = [0] * self.size
        choices = range(self.size)
        for i in range(n_cluster):
            curr_assignment = np.random.choice(choices, size=cluster_sizes[i])
            for j in curr_assignment:
                cluster_list[j] = i
            choices = list(set(choices) - set(curr_assignment))
        if save_dir is not None:
            pickle.dump(cluster_list, save_file, pickle.HIGHEST_PROTOCOL)
        return cluster_list

    @staticmethod
    def _re_label_clusters(prev, curr):
        if len(prev) != len(curr):
            raise ValueError('Cluster labels do not match in length: {} vs {}.'.format(len(prev), len(curr)))
        prev_labels = sorted(set(prev))
        # print('Prev Labels')
        # print(prev_labels)
        curr_labels = set()
        prev_to_curr = {}
        for i in range(len(prev)):
            prev_c = prev[i]
            curr_c = curr[i]
            if prev_c not in prev_to_curr:
                prev_to_curr[prev_c] = {'clusters': [], 'indices': []}
            if curr_c not in curr_labels:
                curr_labels.add(curr_c)
                prev_to_curr[prev_c]['clusters'].append(curr_c)
                prev_to_curr[prev_c]['indices'].append(i)
        # print('Prev_To_Curr')
        # print(prev_to_curr)
        curr_to_new = {}
        counter = 0
        for c in prev_labels:
            for curr_c in zip(*sorted(zip(prev_to_curr[c]['clusters'], prev_to_curr[c]['indices']),
                                      key=lambda x: x[1]))[0]:
                curr_to_new[curr_c] = counter
                counter += 1
        # print('Curr_To_New')
        # print(curr_to_new)
        new_labels = [curr_to_new[c] for c in curr]
        return new_labels

    # def set_tree_ordering(self, t_order=None):
    #     """
    #     Determine the ordering of the sequences from the full clustering tree
    #     used when separating the alignment into sub-clusters.
    #
    #     Args:
    #         t_order (list): An ordered list of sequence IDs which contains at least the sequence IDs represented in this
    #         SeqAlignment.
    #     """
    #     if t_order is not None:
    #         self.tree_order = [x for x in t_order if x in self.seq_order]
    #     elif self.tree_order is None:
    #         df = pd.DataFrame(self.alignment_matrix,
    #                           columns=list(self.query_sequence),
    #                           index=self.tree_order)
    #         hm = clustermap(df, method='ward', metric='euclidean',
    #                         z_score=None, standard_scale=None, row_cluster=True,
    #                         col_cluster=False, cmap='jet')
    #         re_indexing = hm.dendrogram_row.reordered_ind
    #         plt.clf()
    #         self.tree_order = [self.seq_order[i] for i in re_indexing]
    #     else:
    #         pass

    def set_tree_ordering(self, tree_depth=None, clustering='agglomerative', cache_dir=None, clustering_args={}):
        """
        Determine the ordering of the sequences from the full clustering tree
        used when separating the alignment into sub-clusters.

        Args:
            t_order (list): An ordered list of sequence IDs which contains at least the sequence IDs represented in this
            SeqAlignment.
        """
        method_dict = {'agglomerative': self._agglomerative_clustering, 'random': self._random_assignment}
        curr_order = [0] * self.size
        sequence_assignments = {1: {0: set(self.seq_order)}}
        if tree_depth is None:
            tree_depth = range(1, self.size + 1)
        elif isinstance(tree_depth, tuple):
            if len(tree_depth) != 2:
                raise ValueError('If a tuple is provided for tree_depth, two values must be specified.')
            tree_depth = [1] + range(tree_depth[0], tree_depth[1])
        elif isinstance(tree_depth, list):
            if tree_depth[0] != 1:
                tree_depth = [1] + tree_depth
        else:
            raise ValueError('tree_depth must be None, a tuple, or a list.')
        if cache_dir is None:
            remove_dir = True
            cache_dir = os.getcwd()
        else:
            remove_dir = False
        for k in tree_depth:
            cluster_list = method_dict[clustering](n_cluster=k, cache_dir=cache_dir, *clustering_args)
            new_clusters = self._re_label_clusters(curr_order, cluster_list)
            curr_order = new_clusters
            sequence_assignments[k] = {}
            for i, c in enumerate(curr_order):
                if c not in sequence_assignments[k]:
                    sequence_assignments[k][c] = set()
                sequence_assignments[k][c].add(self.seq_order[i])
        self.sequence_assignments = sequence_assignments
        self.tree_order = list(zip(*sorted(zip(self.seq_order, curr_order), key=lambda x: x[1]))[0])
        if remove_dir:
            rmtree(os.path.join(cache_dir, 'joblib'))

    def visualize_tree(self, out_dir=None):
        if out_dir is None:
            out_dir = os.getcwd()
        if self.sequence_assignments is None:
            self.set_tree_ordering()
        check = {'SeqID': self.tree_order, 1: [0] * self.size}
        for k in self.sequence_assignments:
            curr_order = []
            for i in range(self.size):
                for c in self.sequence_assignments[k]:
                    if self.tree_order[i] in self.sequence_assignments[k][c]:
                        curr_order.append(c)
            check[k] = curr_order
        df = pd.DataFrame(check).set_index('SeqID').sort_values(by=self.size)[range(1, self.size + 1)]
        df.to_csv(os.path.join(out_dir, '{}_Sequence_Assignment.csv'.format(self.query_id[1:])), sep='\t', header=True,
                  index=True)
        heatmap(df, cmap='tab10', square=True)
        plt.savefig(os.path.join(out_dir, '{}_Sequence_Assignment.eps'.format(self.query_id[1:])))
        plt.close()
        return df

    def generate_sub_alignment(self, sequence_ids):
        """
        Initializes a new alignment which is a subset of the current alignment.

        This method creates a new alignment which contains only sequences
        relating to a set of provided sequence ids.

        Args:
            sequence_ids (list): A list of strings which are sequence identifiers for sequences in the current
            alignment. Other sequence ids will be skipped.
        Returns:
            SeqAlignment. A new SeqAlignment object containing the same file_name, query_id, seq_length, and query
            sequence.  The seq_order will be updated to only those passed in ids which are also in the current
            alignment, preserving their ordering from the current SeqAlignment object. The alignment will contain only
            the subset of sequences represented by ids which are present in the new seq_order.  The size is set to the
            length of the new seq_order.
        """
        new_alignment = SeqAlignment(self.file_name, self.query_id.split('_')[1])
        new_alignment.query_id = self.query_id
        new_alignment.query_sequence = self.query_sequence
        new_alignment.seq_length = self.seq_length
        sub_records = []
        sub_seq_order = []
        if self.tree_order:
            sub_tree_order = []
        else:
            sub_tree_order = None
        for i in range(self.size):
            if self.alignment[i].id in sequence_ids:
                sub_records.append(self.alignment[i])
                sub_seq_order.append(self.alignment[i].id)
            if (sub_tree_order is not None) and (self.tree_order[i] in sequence_ids):
                sub_tree_order.append(self.tree_order[i])
        # sub_records = [rec for rec in self.alignment if rec.id in sequence_ids]
        # new_alignment.seq_order = [x.id for x in sub_records]
        new_alignment.alignment = MultipleSeqAlignment(sub_records)
        new_alignment.seq_order = sub_seq_order
        new_alignment.tree_order = sub_tree_order
        new_alignment.size = len(new_alignment.seq_order)
        # new_alignment.tree_order = [x for x in self.tree_order if x in sequence_ids]
        return new_alignment

    def get_branch_cluster(self, k, c):
        if self.sequence_assignments is None:
            self.set_tree_ordering()
        cluster_seq_ids = [s for s in self.tree_order if s in self.sequence_assignments[k][c]]
        return self.generate_sub_alignment(sequence_ids=cluster_seq_ids)

    def generate_positional_sub_alignment(self, i, j):
        new_alignment = SeqAlignment(self.file_name, self.query_id.split('_')[1])
        new_alignment.query_id = self.query_id
        new_alignment.query_sequence = self.query_sequence[i] + self.query_sequence[j]
        new_alignment.seq_length = 2
        new_alignment.seq_order = self.seq_order
        new_alignment.alignment = self._subset_columns(indices_to_keep=[i, j])
        new_alignment.size = self.size
        new_alignment.tree_order = self.tree_order
        return new_alignment

    def determine_usable_positions(self, ratio):
        """
        Determine which positions in the alignment can be used for analysis.

        Args:
            ratio (float): The maximum percentage of sequences which can have a gap at a specific position before it can
            no longer be used for analysis.
        Returns:
            numpy ndarray. The positions for which this alignment meets the specified ratio.
            numpy ndarray. The number of sequences which do not have gaps at each position in the sequence alignment.
        """
        per_column = np.array([self.alignment[:, x].count('-') for x in range(self.seq_length)], dtype=float)
        percent_gaps = per_column / self.size
        usable_positions = np.where(percent_gaps <= ratio)[0]
        evidence = (np.ones(self.seq_length) * self.size) - per_column
        return usable_positions, evidence

    def identify_comparable_sequences(self, pos1, pos2):
        """
        For two specified sequence positions identify the sequences which are
        not gaps in either and return them.

        Args:
            pos1 (int): First position to check in the sequence alignment.
            pos2 (int): Second position to check in the sequence alignment.
        Returns:
            np.array. The column for position 1 which was specified, where the amino acids are not gaps in position 1 or
            position 2.
            np.array. The column for position 2 which was specified, where the amino acids are not gaps in position 1 or
            position 2.
            np.array. The array of indices for which the positions were not gapped.  This corresponds to the sequences
            where there were no gaps in the alignment at those positions.
            int. Number of comparable positions, this will be less than or equal to the SeqAlignment.size variable.
        """
        column_i = np.array(list(self.alignment[:, pos1]), dtype=str)
        indices1 = (column_i != '-') * 1
        column_j = np.array(list(self.alignment[:, pos2]), dtype=str)
        indices2 = (column_j != '-') * 1
        check = np.where((indices1 + indices2) == 2)[0]
        return column_i[check], column_j[check], check, check.shape[0]

    def _alignment_to_num(self, aa_dict):
        """
        Alignment to num

        Converts an alignment dictionary to a numerical representation.  This
        method updates the alignment_matrix class variable.

        Args:
            aa_dict (dict): Dictionary mapping characters which can appear in the alignment to digits for
            representation.
        """
        alignment_to_num = np.zeros((self.size, self.seq_length))
        for i in range(self.size):
            for j in range(self.seq_length):
                curr_seq = self.alignment[i]
                alignment_to_num[i, j] = aa_dict[curr_seq[j]]
        return alignment_to_num


    def heatmap_plot(self, name, out_dir=None, aa_dict=None, save=True):
        """
        Heatmap Plot

        This method creates a heatmap using the Seaborn plotting package. The
        data used can come from the summary_matrices or coverage data.

        Args:
            name (str): Name used as the title of the plot and the filename for the saved figure.
            out_dir (str): Path to directory where the heatmap image file should be saved. If None (default) then the
            image will be stored in the current working directory.
        """
        start = time()
        df = pd.DataFrame(self._alignment_to_num(aa_dict), index=self.seq_order,
                          columns=list(self.query_sequence))
        df = df.loc[self.tree_order]
        if aa_dict:
            cmap = matplotlib.cm.get_cmap('jet', len(aa_dict))
        else:
            cmap = 'jet'
        # hm = heatmap(data=df, cmap='jet', center=10.0, vmin=0.0, vmax=20.0,
        hm = heatmap(data=df, cmap=cmap, center=10.0, vmin=0.0, vmax=20.0, cbar=True, square=False)
        hm.set_yticklabels(hm.get_yticklabels(), fontsize=8, rotation=0)
        hm.set_xticklabels(hm.get_xticklabels(), fontsize=6, rotation=0)
        plt.title(name)
        if save:
            file_name = name.replace(' ', '_') + '.pdf'
            if out_dir:
                file_name = os.path.join(out_dir, file_name)
            plt.savefig(file_name)
            plt.clf()
        plt.show()
        end = time()
        print('Plotting alignment took {} min'.format((end - start) / 60.0))
        return hm