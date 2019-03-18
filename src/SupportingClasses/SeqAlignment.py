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
from Bio.Phylo import read, write
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from sklearn.cluster import AgglomerativeClustering
from shutil import rmtree
import cPickle as pickle
from time import time
import pandas as pd
import numpy as np
import os


class SeqAlignment(object):
    """
    This class is meant to represent the type of alignment which is usually used within our lab. The sequence of
    interest is represented by an ID that begins with ">query_" followed by the chosen identifier for that sequence.

    Attributes:
        file_name (str): The path to the file from which the alignment can be parsed.
        query_id (str): A sequence identifier prepended with ">query_", which should be the identifier for query
        sequence in the alignment file.
        alignment (Bio.Align.MultipleSeqAlignment): A biopython representation for a multiple sequence alignment and its
        sequences.
        seq_order (list): List of sequence ids in the order in which they were parsed from the alignment file.
        query_sequence (str): The sequence matching the sequence identifier given by the query_id attribute.
        seq_length (int): The length of the query sequence.
        size (int): The number of sequences in the alignment represented by this object.
        distance_matrix (np.array): A matrix with the identity scores between sequences in the alignment.
        tree_order (list): A list of sequence IDs ordered as they would be in the leaves of a phylogenetic tree, or some
        other purposeful ordering, which should be observed when writing the alignment to file.
        sequence_assignments (dict): An attribute used to track the assignment of sequences to clusters or branches at
        different levels of a phylogenetic/clustering tree. The key for the first level of the dictionary corresponds to
        the level of the tree, while the key in the second level corresponds to the specific branch cluster, and the
        value corresponds to the sequence ID.
    """

    def __init__(self, file_name, query_id):
        """
        __init__

        Initiates an instance of the SeqAlignment class represents a multiple sequence alignment such that common
        actions taken on an alignment in the lab are simple to perform.

        Args:
            file_name (str or path): The path to the file from which the alignment can be parsed. If a relative path is
                used (i.e. the ".." prefix), python's path library will be used to attempt to define the full path.
            query_id (str): The sequence identifier of interest. When stored by the class ">query_" will be prepended
                because it is assumed that alignments used within the lab highlight the sequence of interest in this
                way.
        """
        if file_name.startswith('..'):
            file_name = os.path.abspath(file_name)
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

        This method imports the alignments using the AlignIO.read method expecting the 'fasta' format. It then updates
        the alignment, seq_order, query_sequence, seq_length, and size class class attributes.

        Args:
            save_file (str, optional): Path to file in which the desired alignment was should be stored, or was stored
            previously. If the alignment was previously imported and stored at this location it will be loaded via
            pickle instead of reprocessing the the file in the file_name attribute.
            verbose (bool, optional): Whether or not to print the time spent while importing the alignment or not.
        """
        if verbose:
            start = time()
        if (save_file is not None) and (os.path.exists(save_file)):
            alignment, seq_order, query_sequence = pickle.load(open(save_file, 'rb'))
        else:
            alignment = AlignIO.read(self.file_name, 'fasta')
            seq_order = []
            query_sequence = None
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
        This method writes out the alignment in the standard fa format.  Any sequence which is longer than 60 positions
        will be split over multiple lines with 60 characters per line.

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
            Bio.Align.MultipleSeqAlignment: A new alignment which is a subset of self.alignment, specified by the passed
            in list or array.

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

        Removes all gaps from the query sequence and removes characters at the corresponding positions in all other
        sequences. This method updates the class variables alignment, query_sequence, and seq_length.

        Args:
            save_file (str): Path to a file where the alignment with gaps in the query sequence removed should be stored
            or was stored previously. If the updated alignment was stored previously it will be loaded from the
            specified save_file instead of processing the current alignment.
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
                new_alignment = self.alignment
            if save_file is not None:
                pickle.dump(new_alignment, open(save_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        self.alignment = new_alignment
        self.query_sequence = self.alignment[self.seq_order.index(self.query_id[1:])].seq
        self.seq_length = len(self.query_sequence)
        end = time()
        print('Removing gaps took {} min'.format((end - start) / 60.0))

    def compute_distance_matrix(self, model, save_dir=None):
        """
        Distance matrix

        Computes the sequence identity distance between a set of sequences and returns a matrix of the pairwise
        distances.  This method updates the distance_matrix class variable.

        Args:
            model (str): The type of distance matrix which was computed. Current options include the 'identity' model
            and Bio.Phylo.TreeConstruction.DistanceCalculator.protein_models.
            save_dir (str): The path to a directory wheren a .npz file containing distances between sequences in the
            alignment can be saved. The file created will be <model>.npz.
        """
        start = time()
        if save_dir is None:
            save_dir = os.getcwd()
        save_file = os.path.join(save_dir, model + '.pkl')
        if os.path.exists(save_file):
            with open(save_file, 'rb') as save_handle:
                value_matrix = pickle.load(save_handle)
        else:
            calculator = DistanceCalculator(model=model)
            value_matrix = calculator.get_distance(self.alignment)
            with open(save_file, 'wb') as save_handle:
                pickle.dump(value_matrix, save_handle, protocol=pickle.HIGHEST_PROTOCOL)
        end = time()
        print('Computing the distance matrix took {} min'.format((end - start) / 60.0))
        self.distance_matrix = value_matrix

    def compute_effective_alignment_size(self, identity_threshold=0.62, save_dir=None):
        """
        Compute Effective Alignment Size

        This method uses the distance_matrix variable (containing sequence identities) to compute the effective size of
        the current alignment. The equation (given below) and default threshold (62% identity) are taken from
        PMID:29047157.
            Meff = SUM_(i=0)^(N) of 1/n_i
            where n_i are the number of sequences sequence identity >= the identity threshold
        Args:
            identity_threshold (float): The threshold for what is considered an identical (non-unique) sequence.
            save_dir (str): The path to a directory wheren a .npz file containing distances between sequences in the
            alignment can be saved. The file created will be <model>.npz.
        Returns:
            float: The effective alignment size of the current alignment (must be <= SeqAlignment.size)
        """
        distance_matrix = None
        save_file = None
        if save_dir:
            save_file = os.path.join(save_dir, 'identity.pkl')
            if os.path.isfile(save_file):
                with open(save_file, 'rb') as save_handle:
                    distance_matrix = pickle.load(save_handle)
        if distance_matrix is None:
            calculator = DistanceCalculator(model='identity')
            distance_matrix = calculator.get_distance(self.alignment)
            if save_file:
                with open(save_file, 'wb') as save_handle:
                    pickle.dump(distance_matrix, save_handle, pickle.HIGHEST_PROTOCOL)
        distance_matrix = np.array(distance_matrix)
        meets_threshold = (1 - distance_matrix) >= identity_threshold
        meets_threshold[range(meets_threshold.shape[0]), range(meets_threshold.shape[1])] = True
        n_i = np.sum(meets_threshold, axis=1)
        rec_n_i = 1.0 / n_i
        effective_alignment_size = np.sum(rec_n_i)
        if effective_alignment_size > self.size:
            raise ValueError('Effective alignment size is greater than the original alignment size.')
        return effective_alignment_size

    def _agglomerative_clustering(self, n_cluster, cache_dir=None, affinity='euclidean', linkage='ward', model=None):
        """
        Agglomerative clustering

        Performs agglomerative clustering on a matrix of pairwise distances between sequences in the alignment being
        analyzed.

        Args:
            n_cluster (int): The number of clusters to separate sequences into.
            cache_dir (str): The path to the directory where the clustering model can be stored for access later when
            identifying different numbers of clusters.
            affinity (str): The affinity/distance calculation method to use when operating on the distance values for
            clustering. Further details can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering.
            The options are:
                euclidean (default)
                l1
                l2
                manhattan
                cosin
                precomputed
            linkage (str): The linkage algorithm to use when building the agglomerative clustering tree structure.
            Further details can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering.
            The options are:
                ward (default)
                complete
                average
                single
            model (str): Only required if compute_distance_matrix has not been performed yet. The type of distance
            matrix which was computed. Current options include the 'identity' model and
            Bio.Phylo.TreeConstruction.DistanceCalculator.protein_models.
        Returns:
            list: The cluster assignments for each sequence in the alignment.
        """
        if self.distance_matrix is None:
            if model is None:
                raise ValueError('Agglomerative clustering failed because distance_matrix is None and no model was '
                                 'specified for computing distance.')
            else:
                self.compute_distance_matrix(model=model, save_dir=cache_dir)
        ml_model = AgglomerativeClustering(affinity=affinity, linkage=linkage, n_clusters=n_cluster, memory=cache_dir,
                                           compute_full_tree=True)
        ml_model.fit(np.array(self.distance_matrix))
        # unique and sorted list of cluster ids e.g. for n_clusters=2, g=[0,1]
        cluster_list = ml_model.labels_.tolist()
        return cluster_list

    def __traverse_base_tree(self, tree):
        """
        Traverse Base Tree

        This method generates a dictionary assigning sequences to clusters at each branching level (from 1 the root, to
        n the number of sequences) of the provided tree. This is useful for methods where a rooted Bio.Phylo.BaseTree
        object is used.

        Args:
            tree (Bio.Phylo.BaseTree): The tree to traverse and represent as a dictionary.
        Returns:
            dict: A nested dictionray where the first layer has keys corresponding to branching level and the second
            level has keys corresponding to specific branches within that level and the values are a list of the
            sequence identifiers in that branch.
        """

        def node_cmp(x, y):
            """
            Node Comparison

            This method is provided such that lists of nodes are sorted properly for the intended behavior of the UPGMA
            tree traversal.

            Args:
                x (Bio.Phylo.BaseTree.Clade): Left object in comparison.
                y (Bio.Phylo.BaseTree.Clade): Right object for comparison.
            Returns:
                int: Old comparator behavior, i.e. 1 x comes before y and -1 if y comes before x. If the two Clades are
                equal according to this comparison then 0 is returned.
            """
            if x.is_terminal() and not y.is_terminal():
                return -1
            elif not x.is_terminal() and y.is_terminal():
                return 1
            else:
                if x.branch_length < y.branch_length:
                    return -1
                elif x.branch_length > y.branch_length:
                    return 1
                else:
                    return 0

        # Travers the tree
        assignment_dict = {}
        lookup = {s: i for i, s in enumerate(self.seq_order)}
        nodes_to_process = [tree.root]
        unique_clusters = {}
        current_cluster = []
        k = 0
        while len(nodes_to_process) > 0:
            assignment_dict[k] = {}
            for node in nodes_to_process:
                current_cluster.append(node)
            current_cluster.sort(cmp=node_cmp)
            cluster = 0
            for node in current_cluster:
                if node.name in unique_clusters:
                    terminals = unique_clusters[node.name]
                else:
                    terminals = [lookup[x.name] for x in node.get_terminals()]
                    unique_clusters[node.name] = terminals
                assignment_dict[k][cluster] = terminals
                cluster += 1
            k += 1
            nearest_node = current_cluster.pop()
            if not nearest_node.is_terminal():
                nodes_to_process = nearest_node.clades
            else:
                nodes_to_process = []
        return assignment_dict

    def _upgma_tree(self, n_cluster, cache_dir=None, model=None):
        """
        UPGMA Tree

        Constructs a UPGMA tree from the current alignment using a specified model (or a precomputed distance matrix).
         This tree is then traveresed and stored such that the 'clusters' at a given n_clusters can be provided as
         requested.

        Args:
            n_cluster (int): The number of clusters to separate sequences into.
            cache_dir (str): The path to the directory where the tree and clusters identified by its traversal can be
            stored for access later when identifying different numbers of clusters.
            model (str): Only required if compute_distance_matrix has not been performed yet. The type of distance
            matrix which was computed. Current options include the 'identity' model and
            Bio.Phylo.TreeConstruction.DistanceCalculator.protein_models.
        Returns:
            list: The cluster assignments for each sequence in the alignment.
        """
        if cache_dir is None:
            cache_dir = os.getcwd()
        fn = 'serialized_aln_upgma.pkl'
        if os.path.isfile(os.path.join(cache_dir, fn)):
            with open(os.path.join(cache_dir, fn), 'rb') as fn_handle:
                assignment_dict = pickle.load(fn_handle)
        else:
            # Compute distance matrix
            if self.distance_matrix is None:
                if model is None:
                    raise ValueError('Agglomerative clustering failed because distance_matrix is None and no model was '
                                     'specified for computing distance.')
                else:
                    self.compute_distance_matrix(model=model, save_dir=cache_dir)
            # Create upgma tree
            constructor = DistanceTreeConstructor()
            upgma_tree = constructor.upgma(distance_matrix=self.distance_matrix)
            write(upgma_tree, os.path.join(cache_dir, fn.split('.')[0] + '.tre'), 'newick')
            # Travers the tree
            assignment_dict = self.__traverse_base_tree(tree=upgma_tree)
            with open(os.path.join(cache_dir, fn), 'wb') as fn_handle:
                pickle.dump(assignment_dict, fn_handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Emit clustering
        cluster_labels = np.zeros(self.size)
        for i in range(n_cluster):
            cluster_labels[list(assignment_dict[n_cluster - 1][i])] = i
        return list(cluster_labels)

    def _custom_tree(self, n_cluster, tree_path, tree_name, cache_dir=None):
        """
        Custom Tree

        Reads in a previously generated tree from a specified file. This tree must be written in the 'newick' format and
        must be rooted. This tree is then traveresed and stored such that the 'clusters' at a given n_clusters can be
        provided as requested.

        Args:
            n_cluster (int): The number of clusters to separate sequences into.
            cache_dir (str): The path to the directory where the tree and clusters identified by its traversal can be
            stored for access later when identifying different numbers of clusters.
            tree_path (str/path): The path to a file where the desired tree has been written in 'newick' format.
            tree_name (str): A meaningful name for the tree which is being imported, this will be used in the filename
            for the serialized data collected from the tree.
        Returns:
            list: The cluster assignments for each sequence in the alignment.
        """
        if cache_dir is None:
            cache_dir = os.getcwd()
        fn = 'serialized_{}_tree.pkl'.format(tree_name)
        if os.path.isfile(os.path.join(cache_dir, fn)):
            with open(os.path.join(cache_dir, fn), 'rb') as fn_handle:
                assignment_dict = pickle.load(fn_handle)
        else:

            # Read in custom tree
            custom_tree = read(file=tree_path, format='newick')
            # Travers the tree
            assignment_dict = self.__traverse_base_tree(tree=custom_tree)
            with open(os.path.join(cache_dir, fn), 'wb') as fn_handle:
                pickle.dump(assignment_dict, fn_handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Emit clustering
        cluster_labels = np.zeros(self.size)
        for i in range(n_cluster):
            cluster_labels[list(assignment_dict[n_cluster - 1][i])] = i
        return list(cluster_labels)

    def _random_assignment(self, n_cluster, cache_dir=None):
        """
        random_assignment

        Randomly assigns sequence IDs to groups totaling the specified n_clusters. Sequences are split as evenly as
        possible with all clusters getting the same number of sequences if self.size % n_clusters = 0 and with
        self.size % n_clusters groups getting one additional sequence otherwise.

        Args:
            n_cluster (int): The number of clusters to produce by random assignment.
            cache_dir (str): The path to the directory where the clustering model can be stored for access later when
            identifying different numbers of clusters.
        Returns:
            list: The cluster assignments for each sequence in the alignment.
        """
        if cache_dir is not None:
            save_dir = os.path.join(cache_dir, 'joblib')
            save_file = os.path.join(save_dir, 'K_{}.pkl'.format(n_cluster))
        else:
            save_dir = None
            save_file = None
        if save_dir is not None:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            else:
                if os.path.isfile(save_file):
                    with open(save_file, 'rb') as save_handle:
                        return pickle.load(save_handle)
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
            with open(save_file, 'wb') as save_handle:
                pickle.dump(cluster_list, save_handle, pickle.HIGHEST_PROTOCOL)
        return cluster_list

    @staticmethod
    def _re_label_clusters(prev, curr):
        """
        Relabel Clusters

        This method takes in a two sets of cluster labels and ensures that the new one (curr) aggrees in its ordering
        with the previous one (prev). This makes for easier tracking of matching clusters when using methods which do
        not have stable cluster labels even if the clusters themselves are stable.

        Args:
            prev (list): A list of cluster labels.
            curr (list): A list of cluster labels which may need to change, must have the same length as prev.
        Returns:
            list: A new list of cluster labels based on the passed in list (curr). Cluster assignment does not change,
            i.e. the same elements are together in clusters, but the labels change to represent the labels of those
            clusters in the previous (prev) set of labels.
        """
        if len(prev) != len(curr):
            raise ValueError('Cluster labels do not match in length: {} vs {}.'.format(len(prev), len(curr)))
        curr_labels = set()
        prev_to_curr = {}
        for i in range(len(prev)):
            prev_c = prev[i]
            curr_c = curr[i]
            if (prev_c not in prev_to_curr) and (curr_c not in curr_labels):
                prev_to_curr[prev_c] = {'clusters': [], 'indices': []}
            if curr_c not in curr_labels:
                curr_labels.add(curr_c)
                prev_to_curr[prev_c]['clusters'].append(curr_c)
                prev_to_curr[prev_c]['indices'].append(i)
        curr_to_new = {}
        counter = 0
        prev_labels = sorted(prev_to_curr.keys())
        for c in prev_labels:
            for curr_c in zip(*sorted(zip(prev_to_curr[c]['clusters'], prev_to_curr[c]['indices']),
                                      key=lambda x: x[1]))[0]:
                curr_to_new[curr_c] = counter
                counter += 1
        new_labels = [curr_to_new[c] for c in curr]
        return new_labels

    def set_tree_ordering(self, tree_depth=None, cache_dir=None, clustering_args={}, clustering='agglomerative'):
        """
        Determine the ordering of the sequences from the full clustering tree
        used when separating the alignment into sub-clusters.

        Args:
            tree_depth (None, tuple, or list): The levels of the phylogenetic tree to consider when analyzing this
            alignment, which determines the attributes sequence_assignments and tree_ordering. The following options are
            available:
                None: All branches from the top of the tree (1) to the leaves (size) will be analyzed.
                tuple: If a tuple is provided with two ints these will be taken as a range, the top of the tree (1), and
                all branches between the first and second (non-inclusive) integer will be analyzed.
                list: All branches listed will be analyzed, as well as the top of the tree (1) even if not listed.
            cache_dir (str): The path to the directory where the clustering model can be stored for access later when
            identifying different numbers of clusters.
            clustering_args (dict): Additional arguments needed by various clustering/tree building algorithms. If no
            other arguments are needed (as is the case when using 'random' or default settings for 'agglomerative') the
            dictionary can be left empty.
            clustering (str): The type of clustering/tree building to use. Current options are:
                agglomerative
                upgma
                random
        Return:
            list: The explicit list of tree levels analyzed, as described above in the tree_depth Args section.
        """
        method_dict = {'agglomerative': self._agglomerative_clustering, 'upgma': self._upgma_tree,
                       'random': self._random_assignment}
        curr_order = [0] * self.size
        sequence_assignments = {1: {0: set(self.seq_order)}}
        if tree_depth is None:
            tree_depth = range(1, self.size + 1)
        elif isinstance(tree_depth, tuple):
            if len(tree_depth) != 2:
                raise ValueError('If a tuple is provided for tree_depth, two values must be specified.')
            tree_depth = list(range(tree_depth[0], tree_depth[1]))
        elif isinstance(tree_depth, list):
            pass
        else:
            raise ValueError('tree_depth must be None, a tuple, or a list.')
        if tree_depth[0] != 1:
            tree_depth = [1] + tree_depth
        if cache_dir is None:
            remove_dir = True
            cache_dir = os.getcwd()
        else:
            remove_dir = False
        for k in tree_depth:
            cluster_list = method_dict[clustering](n_cluster=k, cache_dir=cache_dir, **clustering_args)
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
        return tree_depth

    def visualize_tree(self, out_dir=None):
        """
        Visualize Tree

        This method takes the sequence_assignments attribute and visualizes them as a heatmap so that the way clusters
        change throughout the tree can be easily seen.

        Args:
            out_dir (str): The location to which the tree visualization should be saved.
        Returns:
            pd.Dataframe: The data used for generating the heatmap, with sequence IDs as the index and tree level/branch
            as the columns.
        """
        if out_dir is None:
            out_dir = os.getcwd()
        if self.sequence_assignments is None:
            raise ValueError('SeqAlignment.sequence_assignments not initialized, run set_tree_ordering prior to this '
                             'method being run.')
        check = {'SeqID': self.tree_order, 1: [0] * self.size}
        for k in self.sequence_assignments:
            curr_order = []
            for i in range(self.size):
                for c in self.sequence_assignments[k]:
                    if self.tree_order[i] in self.sequence_assignments[k][c]:
                        curr_order.append(c)
            check[k] = curr_order
        branches = sorted(self.sequence_assignments.keys())
        df = pd.DataFrame(check).set_index('SeqID').sort_values(by=branches[::-1])[branches]
        df.to_csv(os.path.join(out_dir, '{}_Sequence_Assignment.csv'.format(self.query_id[1:])), sep='\t', header=True,
                  index=True)
        heatmap(df, cmap='tab10', square=True)
        plt.savefig(os.path.join(out_dir, '{}_Sequence_Assignment.eps'.format(self.query_id[1:])))
        plt.close()
        return df

    def generate_sub_alignment(self, sequence_ids):
        """
        Initializes a new alignment which is a subset of the current alignment.

        This method creates a new alignment which contains only sequences relating to a set of provided sequence ids.

        Args:
            sequence_ids (list): A list of strings which are sequence identifiers for sequences in the current
            alignment. Other sequence ids will be skipped.
        Returns:
            SeqAlignment: A new SeqAlignment object containing the same file_name, query_id, seq_length, and query
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
        new_alignment.alignment = MultipleSeqAlignment(sub_records)
        new_alignment.seq_order = sub_seq_order
        new_alignment.tree_order = sub_tree_order
        new_alignment.size = len(new_alignment.seq_order)
        return new_alignment

    def get_branch_cluster(self, k, c):
        """
        Get Branch Cluster

        Thus method generates a sub alignment based on a specific level/branch of the tree being analyzed and a specific
        node/cluster within that level/branch.

        Args:
            k (int): The branching level (root <= k <= leaves where root = 1, leaves = size) to which the desired
            set of sequences belongs.
            c (int): The cluster/node in the specified branching level (1 <= c <= k) to which the desired set of
            sequences belongs.
        Returns:
            SeqAlignment: A new SeqAlignment object which is a subset of the current SeqAlignment corresponding to the
            sequences in a specific branching level and cluster of the phylogenetic tree based on this multiple sequence
            alignment.
        """
        if self.sequence_assignments is None:
            self.set_tree_ordering()
        cluster_seq_ids = [s for s in self.tree_order if s in self.sequence_assignments[k][c]]
        return self.generate_sub_alignment(sequence_ids=cluster_seq_ids)

    def generate_positional_sub_alignment(self, i, j):
        """
        Generate Positional Sub Alignment

        This method generates an alignment with only two specified columns, meant to enable the interrogation of
        covariance scores.

        Args:
            i (int): The first position to consider when making a sub alignment of two specific positions.
            j (int): The first position to consider when making a sub alignment of two specific positions.
        Returns:
            SeqAlignment: A new subalignment containing all sequences from the current SeqAlignment object but with
            only the two sequence positions (columns) specified.
        """
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
        Determine Usable Positions

        Determine which positions in the alignment can be used for analysis if number of gaps is being considered.

        Args:
            ratio (float): The maximum percentage of sequences which can have a gap at a specific position before it can
            no longer be used for analysis.
        Returns:
            numpy ndarray: The positions for which this alignment meets the specified ratio.
            numpy ndarray: The number of sequences which do not have gaps at each position in the sequence alignment.
        """
        per_column = np.array([self.alignment[:, x].count('-') for x in range(self.seq_length)], dtype=float)
        percent_gaps = per_column / self.size
        usable_positions = np.where(percent_gaps <= ratio)[0]
        evidence = (np.ones(self.seq_length) * self.size) - per_column
        return usable_positions, evidence

    def identify_comparable_sequences(self, pos1, pos2):
        """
        For two specified sequence positions identify the sequences which are not gaps in either and return them.

        Args:
            pos1 (int): First position to check in the sequence alignment.
            pos2 (int): Second position to check in the sequence alignment.
        Returns:
            np.array: The column for position 1 which was specified, where the amino acids are not gaps in position 1 or
            position 2.
            np.array: The column for position 2 which was specified, where the amino acids are not gaps in position 1 or
            position 2.
            np.array: The array of indices for which the positions were not gapped.  This corresponds to the sequences
            where there were no gaps in the alignment at those positions.
            int: Number of comparable positions, this will be less than or equal to the SeqAlignment.size variable.
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

        Converts an Bio.Align.MultipleSeqAlignment object into a numerical matrix representation.

        Args:
            aa_dict (dict): Dictionary mapping characters which can appear in the alignment to digits for
            representation.
        Returns:
            np.array: Array with dimensions seq_length by size where the values are integers representing amino acids
            and gaps from the current alignment.
        """
        alignment_to_num = np.zeros((self.size, self.seq_length))
        for i in range(self.size):
            for j in range(self.seq_length):
                curr_seq = self.alignment[i]
                alignment_to_num[i, j] = aa_dict[curr_seq[j]]
        return alignment_to_num

    def heatmap_plot(self, name, aa_dict, out_dir=None, save=True):
        """
        Heatmap Plot

        This method creates a heatmap of the alignment so it can be easily visualized. A numerical representation of the
        amino acids is used so that cells can be colored differently for each amino acid. The ordering along the y-axis
        reflects the tree_order attribute, while the ordering along the x-axis represents the sequence positions from
        0 to seq_length.

        Args:
            name (str): Name used as the title of the plot and the filename for the saved figure (spaces will be
            replaced by underscores when saving the plot).
            aa_dict (dict): Dictionary mapping characters which can appear in the alignment to digits for
            representation.
            out_dir (str): Path to directory where the heatmap image file should be saved. If None (default) then the
            image will be stored in the current working directory.
            save (bool): Whether or not to save the plot to file.
        Returns:
            pd.Dataframe: The data used to generate the heatmap.
            matplotlib.Axes: The plotting object created when generating the heatmap.
        """
        start = time()
        df = pd.DataFrame(self._alignment_to_num(aa_dict), index=self.seq_order,
                          columns=['{}:{}'.format(x, aa) for x, aa in enumerate(self.query_sequence)])
        if self.tree_order:
            df = df.loc[self.tree_order]
        if aa_dict:
            cmap = matplotlib.cm.get_cmap('jet', len(aa_dict))
        else:
            cmap = 'jet'
        hm = heatmap(data=df, cmap=cmap, center=10.0, vmin=0.0, vmax=20.0, cbar=True, square=False)
        hm.set_yticklabels(hm.get_yticklabels(), fontsize=8, rotation=0)
        hm.set_xticklabels(hm.get_xticklabels(), fontsize=6, rotation=0)
        plt.title(name)
        if save:
            file_name = name.replace(' ', '_') + '.eps'
            if out_dir:
                file_name = os.path.join(out_dir, file_name)
            plt.savefig(file_name)
            plt.clf()
        plt.show()
        end = time()
        print('Plotting alignment took {} min'.format((end - start) / 60.0))
        return df, hm
