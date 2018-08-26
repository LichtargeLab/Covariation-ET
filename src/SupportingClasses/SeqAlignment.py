"""
Created on Aug 17, 2017

@author: daniel
"""
from sklearn.cluster import AgglomerativeClustering
import cPickle as pickle
from time import time
import pandas as pd
import numpy as np
import os
import re
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from seaborn import heatmap, clustermap


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
        alignment_dict: dict
            A dictionary mapping sequence IDs with their sequences as parsed
            from the alignment file.
        seq_order: list
            List of sequence ids in the order in which they were parsed from the
            alignment file.
        query_sequence: str
            The sequence matching the sequence identifier give by the query_id
            variable.
        alignment_matrix: np.array
            A numerical representation of alignment, every amino acid has been
            assigned a numerical representation as has the gap symbol.  All
            rows are different sequences as described in seq_order, while each
            column in the matrix is a position in the sequence.
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
        self.file_name = file_name
        self.query_id = '>query_' + query_id
        self.alignment_dict = None
        self.seq_order = None
        self.query_sequence = None
        self.alignment_matrix = None
        self.seq_length = None
        self.size = None
        self.distance_matrix = None
        self.tree_order = None

    def import_alignment(self, save_file=None):
        """
        Import alignments:

        This method imports the alignments into the class and forces all
        non-amino acids to take on the standard gap character "-".  This
        updates the alignment_dict, seq_order, query_sequence, seq_length, and size
        class variables.

        Parameters:
        -----------
        save_file: str
            Path to file in which the desired alignment was stored previously.
        """
        start = time()
        if (save_file is not None) and (os.path.exists(save_file)):
            alignment, seq_order = pickle.load(open(save_file, 'rb'))
        else:
            fa_file = open(self.file_name, 'rb')
            alignment = {}
            seq_order = []
            for line in fa_file:
                if line.startswith(">"):
                    key = line.rstrip()
                    alignment[key] = ''
                    seq_order.append(key)
                else:
                    alignment[key] += re.sub(r'[^a-zA-Z]', '-', line.rstrip())
            fa_file.close()
            if save_file is not None:
                pickle.dump((alignment, seq_order), open(save_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        end = time()
        print('Importing alignment took {} min'.format((end - start) / 60.0))
        self.alignment_dict = alignment
        self.seq_order = seq_order
        self.query_sequence = self.alignment_dict[self.query_id]
        self.seq_length = len(self.query_sequence)
        self.size = len(self.alignment_dict)

    def write_out_alignment(self, file_name):
        """
        This method writes out the alignment in the standard fa format.  Any
        sequence which is longer than 60 positions will be split over multiple
        lines with 60 characters per line.

        Parameters:
        file_name: str
            Path to file where the alignment should be written.
        """
        start = time()
        out_file = open(file_name, 'wb')
        for seq_id in self.seq_order:
            if seq_id in self.alignment_dict:
                out_file.write(seq_id + '\n')
                seq_len = len(self.alignment_dict[seq_id])
                breaks = seq_len / 60
                if (seq_len % 60) != 0:
                    breaks += 1
                for i in range(breaks):
                    start_pos = 0 + i * 60
                    end_pos = 60 + i * 60
                    out_file.write(
                        self.alignment_dict[seq_id][start_pos: end_pos] + '\n')
            else:
                pass
        out_file.close()
        end = time()
        print('Writing out alignment took {} min'.format((end - start) / 60.0))

    def heatmap_plot(self, name, out_dir=None):
        """
        Heatmap Plot

        This method creates a heatmap using the Seaborn plotting package. The
        data used can come from the summary_matrices or coverage data.

        Parameters:
        -----------
        name : str
            Name used as the title of the plot and the filename for the saved
            figure.
        out_dir : str
            Path to directory where the heatmap image file should be saved. If
            None (default) then the image will be stored in the current working
            directory.
        """
        start = time()
        df = pd.DataFrame(self.alignment_matrix, index=self.seq_order,
                          columns=list(self.query_sequence))
        df = df.loc[self.tree_order]
        hm = heatmap(data=df, cmap='jet', center=10.0, vmin=0.0, vmax=20.0,
                     cbar=True, square=False)
        hm.set_yticklabels(hm.get_yticklabels(), fontsize=8, rotation=0)
        hm.set_xticklabels(hm.get_xticklabels(), fontsize=6, rotation=0)
        plt.title(name)
        file_name = name.replace(' ', '_') + '.pdf'
        if out_dir:
            file_name = os.path.join(out_dir, file_name)
        plt.savefig(file_name)
        plt.clf()
        end = time()
        print('Plotting alignment took {} min'.format((end - start) / 60.0))

    def remove_gaps(self, save_file=None):
        """
        Remove Gaps

        Removes all gaps from the query sequence and removes characters at the
        corresponding positions in all other sequences. This method updates the
        class variables alignment_dict, query_sequence, and seq_length.

        Parameters:
        -----------
        save_file: str
            Path to a file where the alignment with gaps in the query sequence
            removed was stored previously.
        """
        start = time()
        if (save_file is not None) and os.path.exists(save_file):
            new_alignment = pickle.load(open(save_file, 'rb'))
        else:
            query_arr = np.array(list(self.query_sequence))
            query_ungapped_ind = np.where(query_arr != '-')[0]
            if len(query_ungapped_ind) > 0:
                new_alignment = {}
                for key, value in self.alignment_dict.iteritems():
                    curr_arr = np.array(list(value))[query_ungapped_ind]
                    new_alignment[key] = curr_arr.tostring()
            else:
                pass
            if save_file is not None:
                pickle.dump(new_alignment, open(save_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        self.alignment_dict = new_alignment
        self.query_sequence = self.alignment_dict[self.query_id]
        self.seq_length = len(self.query_sequence)
        end = time()
        print('Removing gaps took {} min'.format((end - start) / 60.0))

    def alignment_to_num(self, aa_dict):
        """
        Alignment to num

        Converts an alignment dictionary to a numerical representation.  This
        method updates the alignment_matrix class variable.

        Parameters:
        -----------
        aa_dict: dict
            Dictionary mapping characters which can appear in the alignment to
            digits for representation.
        """
        start = time()
        alignment_to_num = np.zeros((self.size, self.seq_length))
        for i in range(self.size):
            for j in range(self.seq_length):
                curr_seq = self.alignment_dict[self.seq_order[i]]
                alignment_to_num[i, j] = aa_dict[curr_seq[j]]
        self.alignment_matrix = alignment_to_num
        end = time()
        print('Converting alignment took {} min'.format((end - start) / 60.0))

    def compute_distance_matrix(self, save_file=None):
        """
        Distance matrix

        Computes the sequence identity distance between a set of sequences and
        returns a matrix of the pairwise distances.  This method updates the
        distance_matrix class variable.

        Parameters:
        -----------
        save_file: str
            The path for an .npz file containing distances between sequences in
            the alignment (leave out the .npz as it will be added automatically.
        """
        start = time()
        if (save_file is not None) and os.path.exists(save_file + '.npz'):
            value_matrix = np.load(save_file + '.npz')['X']
        else:
            value_matrix = np.zeros([self.size, self.size])
            for i in range(self.size):
                check = self.alignment_matrix - self.alignment_matrix[i]
                value_matrix[i] = np.sum(check == 0, axis=1)
            #                 value_matrix[i] = np.sum(check != 0, axis=1)
            value_matrix[np.arange(self.size), np.arange(self.size)] = 0
            value_matrix /= self.seq_length
            if save_file is not None:
                np.savez(save_file, X=value_matrix)
        end = time()
        print('Computing the distance matrix took {} min'.format((end - start) / 60.0))
        self.distance_matrix = value_matrix

    def set_tree_ordering(self, t_order=None):
        """
        Determine the ordering of the sequences from the full clustering tree
        used when separating the alignment into sub-clusters.

        t_order : list
            An ordered list of sequence IDs which contains at least the
            sequence IDs represented in this SeqAlignment.
        """
        if t_order is not None:
            self.tree_order = [x for x in t_order if x in self.seq_order]
        elif self.tree_order is None:
            df = pd.DataFrame(self.alignment_matrix,
                              columns=list(self.query_sequence),
                              index=self.tree_order)
            hm = clustermap(df, method='ward', metric='euclidean',
                            z_score=None, standard_scale=None, row_cluster=True,
                            col_cluster=False, cmap='jet')
            re_indexing = hm.dendrogram_row.reordered_ind
            plt.clf()
            self.tree_order = [self.seq_order[i] for i in re_indexing]
        else:
            pass

    def agg_clustering(self, n_cluster, cache_dir):
        """
        Agglomerative clustering

        Performs agglomerative clustering on a matrix of pairwise distances
        between sequences in the alignment being analyzed.

        Parameters:
        -----------
        n_cluster: int
            The number of clusters to separate sequences into.
        cache_dir : str
            The path to the directory where the clustering model can be stored
            for access later when identifying different numbers of clusters.
        Returns:
        --------
        dict
            A dictionary with cluster number as the key and a list of sequences in
            the specified cluster as a value.
        set
            A unique sorted set of the cluster values.
        """
        start = time()
        affinity = 'euclidean'
        linkage = 'ward'
        model = AgglomerativeClustering(affinity=affinity, linkage=linkage,
                                        n_clusters=n_cluster, memory=cache_dir,
                                        compute_full_tree=True)
        model.fit(self.distance_matrix)
        # unique and sorted list of cluster ids e.g. for n_clusters=2, g=[0,1]
        cluster_list = model.labels_.tolist()
        ################################################################################################################
        #       Mapping Clusters to Sequences
        ################################################################################################################
        cluster_labels = set(cluster_list)
        cluster_dict = {}
        cluster_ordering = {}
        for i in range(len(cluster_list)):
            seq_id = self.seq_order[i]
            index = self.tree_order.index(seq_id)
            key = cluster_list[i]
            if key not in cluster_dict:
                cluster_dict[key] = []
                cluster_ordering[key] = index
            if index < cluster_ordering[key]:
                cluster_ordering[key] = index
            cluster_dict[key].append(self.seq_order[i])
        sorted_cluster_labels = sorted(cluster_ordering, key=lambda k: cluster_ordering[k])
        cluster_dict2 = {}
        for i in range(len(cluster_labels)):
            cluster_dict2[i] = cluster_dict[sorted_cluster_labels[i]]
        end = time()
        print('Performing agglomerative clustering took {} min'.format((end - start) / 60.0))
        return cluster_dict2, cluster_labels

    def random_assignment(self, n_cluster):
        """
        random_assignment

        Randomly assigns sequence IDs to groups totaling the specified
        nClusters. Sequences are split as evenly as possible with all clusters
        getting the same number of sequences if self.size % nClusters = 0 and
        with self.size % nClusters groups getting one additional sequence
        otherwise.

        Parameters:
        -----------
        nClusters : int
            The number of clusters to produce by random assignment.
        Returns:
        --------
        dict
            A dictionary mapping a cluster labels (0 to nClusters -1) to a list
            of sequence IDs assigned to that cluster.
        set
            The set of labels used for clustering (0 to nClusters -1).
        """
        start = time()
        cluster_sizes = np.ones(n_cluster, dtype=np.int64)
        min_size = self.size / n_cluster
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
        cluster_labels = set(cluster_list)
        cluster_dict = {}
        cluster_ordering = {}
        for i in range(len(cluster_list)):
            seq_id = self.seq_order[i]
            index = self.tree_order.index(seq_id)
            key = cluster_list[i]
            if key not in cluster_dict:
                cluster_dict[key] = []
                cluster_ordering[key] = index
            if index < cluster_ordering[key]:
                cluster_ordering[key] = index
            cluster_dict[key].append(self.seq_order[i])
        sorted_cluster_labels = sorted(cluster_ordering, key=lambda k: cluster_ordering[k])
        cluster_dict2 = {}
        for i in range(len(cluster_labels)):
            cluster_dict2[i] = cluster_dict[sorted_cluster_labels[i]]
        end = time()
        print('Performing agglomerative clustering took {} min'.format((end - start) / 60.0))
        return cluster_dict2, cluster_labels

    def generate_sub_alignment(self, sequence_ids):
        """
        Initializes a new alignment which is a subset of the current alignment.

        This method creates a new alignment which contains only sequences
        relating to a set of provided sequence ids.

        Parameters:
        -----------
        sequence_ids: list
            A list of strings which are sequence identifiers for sequences in
            the current alignment.  Other sequence ids will be skipped.

        Returns:
        --------
        SeqAlignment
            A new SeqAlignment object containing the same file_name, query_id,
            seq_length, and query sequence.  The seq_order will be updated to
            only those passed in ids which are also in the current alignment,
            preserving their ordering from the current SeqAlignment object.
            The alignment_dict will contain only the subset of sequences
            represented by ids which are present in the new seq_order.  The size
            is set to the length of the new seq_order.
        """
        start = time()
        new_alignment = SeqAlignment(self.file_name, self.query_id.split('_')[1])
        new_alignment.query_id = self.query_id
        new_alignment.query_sequence = self.query_sequence
        new_alignment.seq_length = self.seq_length
        new_alignment.seq_order = [x for x in self.seq_order if x in sequence_ids]
        new_alignment.alignment_dict = {x: self.alignment_dict[x] for x in new_alignment.seq_order}
        new_alignment.size = len(new_alignment.seq_order)
        new_alignment.tree_order = [x for x in self.tree_order if x in sequence_ids]
        end = time()
        print('Generating sub-alignment took {} min'.format((end - start) / 60.0))
        return new_alignment

    def determine_usable_positions(self, ratio):
        """
        Determine which positions in the alignment can be used for analysis.

        Parameters:
        -----------
        ratio: float
            The maximum percentage of sequences which can have a gap at a
            specific position before it can no longer be used for analysis.

        Returns:
        --------
        numpy ndarray:
            The positions for which this alignment meets the specified ratio.
        numpy ndarray:
            The number of sequences which do not have gaps at each position in
            the sequence alignment.
        """
        gaps = (self.alignment_matrix == 21) * 1.0
        per_column = np.sum(gaps, axis=0)
        percent_gaps = per_column / self.alignment_matrix.shape[0]
        usable_positions = np.where(percent_gaps <= ratio)[0]
        evidence = (np.ones(self.seq_length) * self.size) - per_column
        return usable_positions, evidence

    def identify_comparable_sequences(self, pos1, pos2):
        """
        For two specified sequence positions identify the sequences which are
        not gaps in either and return them.

        Parameters:
        -----------
        pos1: int
            First position to check in the sequence alignment.
        pos2: int
            Second position to check in the sequence alignment.

        Returns:
        --------
        np.array
            The column for position 1 which was specified, where the amino acids
            are not gaps in position 1 or position 2.
        np.array
            The column for position 2 which was specified, where the amino acids
            are not gaps in position 1 or position 2.
        np.array
            The array of indices for which the positions were not gapped.  This
            corresponds to the sequences where there were no gaps in the
            alignment at those positions.
        int
            Number of comparable positions, this will be less than or equal to
            the SeqAlignment.size variable.
        """
        column_i = self.alignment_matrix[:, pos1]
        indices1 = (column_i != 20.0) * 1
        column_j = self.alignment_matrix[:, pos2]
        indices2 = (column_j != 20.0) * 1
        check = np.where((indices1 + indices2) == 2)[0]
        return column_i[check], column_j[check], check, check.shape[0]
