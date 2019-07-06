"""
Created on Aug 17, 2017

@author: daniel
"""
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from seaborn import heatmap, clustermap
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.Alphabet import Gapped
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from scipy.stats import zscore
import cPickle as pickle
from time import time
import pandas as pd
import numpy as np
import os
from utils import build_mapping
from utils import convert_seq_to_numeric
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from EvolutionaryTraceAlphabet import FullIUPACProtein, FullIUPACDNA


class SeqAlignment(object):
    """
    This class is meant to represent the type of alignment which is usually used within our lab. The sequence of
    interest is represented by an ID passed in by the user.

    Attributes:
        file_name (str): The path to the file from which the alignment can be parsed.
        query_id (str): A sequence identifier , which should be the identifier for query sequence in the alignment file.
        alignment (Bio.Align.MultipleSeqAlignment): A biopython representation for a multiple sequence alignment and its
        sequences.
        seq_order (list): List of sequence ids in the order in which they were parsed from the alignment file.
        query_sequence (Bio.Seq.Seq): The sequence matching the sequence identifier given by the query_id attribute.
        seq_length (int): The length of the query sequence (including gaps and ambiguity characters).
        size (int): The number of sequences in the alignment represented by this object.
        marked (list): List of boolean values tracking whether a sequence has been marked or not (this is generally used
        to track which sequences should be skipped during an analysis (for instance in the zoom version of Evolutionary
        Trace).
        polymer_type (str): Whether the represented alignment is a 'Protein' or 'DNA' alignment (these are currently the
        only two options.
        alphabet (EvolutionaryTraceAlphabet): The protein or DNA alphabet to use for this alignment (used by default if
        calculating sequence distances).
    """

    def __init__(self, file_name, query_id, polymer_type='Protein'):
        """
        __init__

        Initiates an instance of the SeqAlignment class represents a multiple sequence alignment such that common
        actions taken on an alignment in the lab are simple to perform.

        Args:
            file_name (str or path): The path to the file from which the alignment can be parsed. If a relative path is
                used (i.e. the ".." prefix), python's path library will be used to attempt to define the full path.
            query_id (str): The sequence identifier of interest.
            polymer_type (str): The type of polymer this alignment represents. Expected values are 'Protein' or 'DNA'.
        """
        if file_name.startswith('..'):
            file_name = os.path.abspath(file_name)
        self.file_name = file_name
        self.query_id = query_id
        self.alignment = None
        self.seq_order = None
        self.query_sequence = None
        self.seq_length = None
        self.size = None
        self.marked = None
        if polymer_type not in {'Protein', 'DNA'}:
            raise ValueError("Expected values for polymer_type are 'Protein' and 'DNA'.")
        self.polymer_type = polymer_type
        if self.polymer_type == 'Protein':
            self.alphabet = FullIUPACProtein()
        else:
            self.alphabet = FullIUPACDNA()

    def import_alignment(self, save_file=None, verbose=False):
        """
        Import alignments:

        This method imports the alignments using the AlignIO.read method expecting the 'fasta' format. It then updates
        the alignment, seq_order, query_sequence, seq_length, size, and marked class attributes.

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
            with open(self.file_name, 'rb') as file_handle:
                alignment = AlignIO.read(file_handle, format='fasta', alphabet=self.alphabet)
            seq_order = []
            query_sequence = None
            for record in alignment:
                seq_order.append(record.id)
                if record.id == self.query_id:
                    query_sequence = record.seq
            if query_sequence is None:
                raise ValueError('Query sequence was not found upon alignment import, check query_id or alignment file')
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
        self.marked = [False] * self.size

    def write_out_alignment(self, file_name):
        """
        This method writes out the alignment in the standard fa format.  Any sequence which is longer than 60 positions
        will be split over multiple lines with 60 characters per line.

        Args:
            file_name (str): Path to file where the alignment should be written.
        """
        if self.alignment is None:
            raise TypeError('Alignment must be Bio.Align.MultipleSequenceALignment not None.')
        if os.path.exists(file_name):
            return
        start = time()
        with open(file_name, 'wb') as file_handle:
            AlignIO.write(self.alignment, handle=file_handle, format="fasta")
        end = time()
        print('Writing out alignment took {} min'.format((end - start) / 60.0))

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
            length of the new seq_order. The marked attribute for all sequences in the sub alignment will be transferred
            from the current alignment.
        """
        new_alignment = SeqAlignment(self.file_name, self.query_id)
        new_alignment.query_id = deepcopy(self.query_id)
        new_alignment.query_sequence = deepcopy(self.query_sequence)
        new_alignment.seq_length = deepcopy(self.seq_length)
        new_alignment.polymer_type = deepcopy(self.polymer_type)
        new_alignment.alphabet = deepcopy(self.alphabet)
        sub_records = []
        sub_seq_order = []
        sub_marked = []
        indices = []
        for i in range(self.size):
            if self.alignment[i].id in sequence_ids:
                indices.append(i)
                sub_records.append(deepcopy(self.alignment[i]))
                sub_seq_order.append(deepcopy(self.alignment[i].id))
                sub_marked.append(self.marked[i])
        new_alignment.alignment = MultipleSeqAlignment(sub_records)
        new_alignment.seq_order = sub_seq_order
        new_alignment.size = len(new_alignment.seq_order)
        new_alignment.marked = sub_marked
        return new_alignment

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
            new_aln = pickle.load(open(save_file, 'rb'))
        else:
            query_arr = np.array(list(self.query_sequence))
            query_ungapped_ind = np.where(query_arr != '-')[0]
            if len(query_ungapped_ind) > 0:
                new_aln = self._subset_columns(query_ungapped_ind)
            else:
                new_aln = self.alignment
            if save_file is not None:
                pickle.dump(new_aln, open(save_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        new_alignment = SeqAlignment(self.file_name, self.query_id)
        new_alignment.query_id = deepcopy(self.query_id)
        new_alignment.alignment = new_aln
        new_alignment.seq_order = deepcopy(self.seq_order)
        new_alignment.query_sequence = new_aln[self.seq_order.index(self.query_id)].seq
        new_alignment.seq_length = len(new_alignment.query_sequence)
        new_alignment.size = deepcopy(self.size)
        new_alignment.polymer_type = deepcopy(self.polymer_type)
        new_alignment.marked = deepcopy(self.marked)
        new_alignment.alphabet = deepcopy(self.alphabet)
        end = time()
        print('Removing gaps took {} min'.format((end - start) / 60.0))
        return new_alignment

    def remove_bad_sequences(self):
        """
        Remove Bad Sequences

        This function checks each sequence in the alignment and keeps it if all characters are in the specified alphabet
        (or a gap).

        Return:
            SeqAlignment: A new SeqAlignment object which is a subset of this instance with all sequences which do not
            obey the specified alphabet removed.
        """
        start = time()
        valid_chars = set(Gapped(self.alphabet).letters)
        to_keep = []
        for i in range(self.size):
            if all(char in valid_chars for char in self.alignment[i].seq):
                to_keep.append(self.seq_order[i])
        new_alignment = self.generate_sub_alignment(sequence_ids=to_keep)
        end = time()
        print('Removing sequences that do not fit the alphabet took {} min'.format((end - start) / 60.00))
        return new_alignment

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
        new_alignment = SeqAlignment(self.file_name, self.query_id)
        new_alignment.query_id = deepcopy(self.query_id)
        new_alignment.query_sequence = SeqRecord(id=self.query_id,
                                                 seq=Seq(self.query_sequence[i] + self.query_sequence[j]))
        new_alignment.seq_length = 2
        new_alignment.seq_order = deepcopy(self.seq_order)
        new_alignment.alignment = self._subset_columns(indices_to_keep=[i, j])
        new_alignment.size = deepcopy(self.size)
        new_alignment.marked = deepcopy(self.marked)
        return new_alignment

    def compute_effective_alignment_size(self, identity_threshold=0.62, distance_matrix=None):  # ,save_dir=None):
        """
        Compute Effective Alignment Size

        This method uses the distance_matrix variable (containing sequence identities) to compute the effective size of
        the current alignment. The equation (given below) and default threshold (62% identity) are taken from
        PMID:29047157.
            Meff = SUM_(i=0)^(N) of 1/n_i
            where n_i are the number of sequences sequence identity >= the identity threshold
        Args:
            identity_threshold (float): The threshold for what is considered an identical (non-unique) sequence.
            distance_matrix (Bio.Phylo.TreeConstruction.DistanceMatrix): A precomputed identity distance matrix for this
            alignment.
            save_dir (str): The path to a directory wheren a .npz file containing distances between sequences in the
            alignment can be saved. The file created will be <model>.npz.
        Returns:
            float: The effective alignment size of the current alignment (must be <= SeqAlignment.size)
        """
        if distance_matrix is None:
            calculator = AlignmentDistanceCalculator(protein=(self.polymer_type == 'Protein'))
            distance_matrix = calculator.get_distance(self.alignment)
        distance_matrix = np.array(distance_matrix)
        meets_threshold = (1 - distance_matrix) >= identity_threshold
        meets_threshold[range(meets_threshold.shape[0]), range(meets_threshold.shape[1])] = True
        n_i = np.sum(meets_threshold, axis=1)
        rec_n_i = 1.0 / n_i
        effective_alignment_size = np.sum(rec_n_i)
        if effective_alignment_size > self.size:
            raise ValueError('Effective alignment size is greater than the original alignment size.')
        return effective_alignment_size

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

    def consensus_sequence(self, method='majority'):
        """
        Consensus Sequence

        This method builds a consensus sequence from the alignment using one of several methods.
            'majority' = To use the member of the alphabet which covers the majority of sequences at that position. If
            multiple characters have the same, highest, count the first one in the alphabet is used.

        Args:
            method (str): Which method to use for consensus sequence construction. The only current option is 'majority'
            though there are other possible options which may be added in the future to reflect the previous ETC tool.
        Returns:
            Bio.Seq.SeqRecord: A sequence record containing the consensus sequence for this alignment.
        """
        if method != 'majority':
            raise ValueError("consensus_sequence has not been implemented for method '{}'!".format(method))
        alpha_size, gap_chars, mapping = build_mapping(alphabet=self.alphabet)
        reverse_mapping = {i: k for k, i in mapping.items() if i < alpha_size}
        if '-' in gap_chars:
            gap_char = '-'
        elif '.' in gap_chars:
            gap_char = '.'
        elif '*' in gap_chars:
            gap_char = '*'
        else:
            gap_char = None
        if gap_char:
            reverse_mapping[alpha_size] = gap_char  # Map all gap positions back to the first gap character.
            reverse_mapping[alpha_size + 1] = gap_char  # Map all skip letter positions back to the first gap character.
        numeric_aln = self._alignment_to_num(mapping)
        consensus_seq = []
        for i in range(self.seq_length):
            unique_chars, counts = np.unique(numeric_aln[:, i], return_counts=True)
            highest_count = np.max(counts)
            positions_majority = np.where(counts == highest_count)
            consensus_seq.append(reverse_mapping[unique_chars[positions_majority[0][0]]])
        consensus_record = SeqRecord(id='Consensus Sequence', seq=Seq(''.join(consensus_seq), alphabet=self.alphabet))
        return consensus_record

    def _alignment_to_num(self, mapping):
        """
        Alignment to num

        Converts an Bio.Align.MultipleSeqAlignment object into a numerical matrix representation.

        Args:
            mapping (dict): Dictionary mapping characters which can appear in the alignment to digits for
            representation.
        Returns:
            np.array: Array with dimensions seq_length by size where the values are integers representing amino acids
            and gaps from the current alignment.
        """
        numeric_reps = []
        for seq_record in self.alignment:
            numeric_reps.append(convert_seq_to_numeric(seq_record.seq, mapping))
        alignment_to_num = np.stack(numeric_reps)
        return alignment_to_num

    def _gap_z_score_check(self, z_score_cutoff, num_aln, gap_num):
        """
        Gap Z Score Check

        This function computes the z-score for the gap count of each sequence. This is intended for use when pruning an
        alignment or tree based on sequence gap content.

        Args:
            z_score_cutoff (int/float): The z-score below which to pass a sequence.
            num_aln (numpy.array): A numerical (array) representation of a sequence alignment.
            gap_num (int): The number to which a gap character maps in the num_aln.
        Returns:
            numpy.array: A 1D array with one position for each sequence in the alignment, True if the sequence has a gap
            count whose z-score passes the set cutoff and False otherwise.
        """
        gap_check = num_aln == gap_num
        gap_count = np.sum(gap_check, axis=1)
        gap_z_scores = zscore(gap_count)
        passing_z_scores = gap_z_scores < z_score_cutoff
        return passing_z_scores

    def _gap_percentile_check(self, percentile_cutoff, num_aln, gap_num, mapping):
        """
        Gap Percentile Check

        This function computes a consensus sequence for the provided alignment and then checks the percentage of
        positions where the gap content of each sequence differs from that of the consensus. If the percentage is higher
        than the provided cutoff the sequence does not get marked as passing.

        Args:
            percentile_cutoff (float): The percentage (expressed as a decimal; e.g. 15% = 0.15), of gaps differeing from
            the consensus, above which a sequence does not pass the filter.
            num_aln (numpy.array): A numerical (array) representation of a sequence alignment.
            gap_num (int): The number to which a gap character maps in the num_aln.
            mapping (dict): Dictionary mapping a character to a number corresponding to its position in the alphabet
            and/or in the scoring/substitution matrix.
        Returns:
            numpy.array: A 1D array with one position for each sequence in the alignment, True if the sequence has a gap
            difference from the consensus whose percentile passes the set cutoff and False otherwise.
        """
        consensus_seq = self.consensus_sequence()
        numeric_consensus = convert_seq_to_numeric(seq=consensus_seq, mapping=mapping)
        gap_consensus = numeric_consensus == gap_num
        gap_aln = num_aln == gap_num
        gap_disagreement = np.logical_xor(gap_consensus, gap_aln)
        disgreement_count = np.sum(gap_disagreement, axis=1)
        disagreement_fraction = disgreement_count / float(self.size)
        passing_fractions = disagreement_fraction < percentile_cutoff
        return passing_fractions

    def gap_evaluation(self, size_cutoff=15, z_score_cutoff=20, percentile_cutoff=0.15):
        """
        Gap Evaluation

        This method evaluates each sequence in the alignment and determines if it passes a gap filter. If the alignment
        is larger than size_cutoff a z-score is used to evaluate gap content and identify and sequences which are
        heavily gapped (as specified by the z_score_cutoff). Otherwise, a consensus sequence is computed and the
        difference in gapped positions between each sequence in the alignment and the consensus sequence is calculated
        with sequences passing if the difference is less than the percentile_cutoff. This function is intended for
        filtering alignments and/or phylogenetic trees.

        Args:
            size_cutoff (int): The number of sequences above which to use a z-score when evaluating gap outliers and
            below which to use a percentage of the consensus sequence.
            z_score_cutoff (int/float): The z-score below which to pass a sequence.
            percentile_cutoff (float): The percentage (expressed as a decimal; e.g. 15% = 0.15), of gaps differeing from
            the consensus, above which a sequence does not pass the filter.
        Return:
            list: A list of the sequence IDs which pass the gap cut off used for this alignment.
            list: A list of sequences which do not pass the gap cut off used for this alignment.
        """
        alpha_size, gap_chars, mapping = build_mapping(alphabet=self.alphabet)
        numeric_aln = self._alignment_to_num(mapping)
        gap_number = self.alphabet.size
        if self.size > size_cutoff:
            passing_seqs = self._gap_z_score_check(z_score_cutoff=z_score_cutoff, num_aln=numeric_aln,
                                                   gap_num=gap_number)
        else:
            passing_seqs = self._gap_percentile_check(percentile_cutoff=percentile_cutoff, num_aln=numeric_aln,
                                                      gap_num=gap_number, mapping=mapping)
        seqs_to_keep, seqs_to_remove = [], []
        for i in range(self.size):
            if passing_seqs[i]:
                seqs_to_keep.append(self.seq_order[i])
            else:
                seqs_to_remove.append(self.seq_order[i])
        return seqs_to_keep, seqs_to_remove

    def heatmap_plot(self, name, out_dir=None, save=True):
        """
        Heatmap Plot

        This method creates a heatmap of the alignment so it can be easily visualized. A numerical representation of the
        amino acids is used so that cells can be colored differently for each amino acid. The ordering along the y-axis
        reflects the tree_order attribute, while the ordering along the x-axis represents the sequence positions from
        0 to seq_length.

        Args:
            name (str): Name used as the title of the plot and the filename for the saved figure (spaces will be
            replaced by underscores when saving the plot).
            out_dir (str): Path to directory where the heatmap image file should be saved. If None (default) then the
            image will be stored in the current working directory.
            save (bool): Whether or not to save the plot to file.
        Returns:
            pd.Dataframe: The data used to generate the heatmap.
            matplotlib.Axes: The plotting object created when generating the heatmap.
        """
        if save:
            file_name = name.replace(' ', '_') + '.eps'
            if out_dir:
                file_name = os.path.join(out_dir, file_name)
            if os.path.exists(file_name):
                return
        else:
            file_name = None
        start = time()
        _, _, mapping = build_mapping(alphabet=self.alphabet)
        df = pd.DataFrame(self._alignment_to_num(mapping=mapping), index=self.seq_order,
                          columns=['{}:{}'.format(x, aa) for x, aa in enumerate(self.query_sequence)])
        cmap = matplotlib.cm.get_cmap('jet', len(mapping))
        hm = heatmap(data=df, cmap=cmap, center=10.0, vmin=0.0, vmax=20.0, cbar=True, square=False)
        hm.set_yticklabels(hm.get_yticklabels(), fontsize=8, rotation=0)
        hm.set_xticklabels(hm.get_xticklabels(), fontsize=6, rotation=0)
        plt.title(name)
        if save:
            plt.savefig(file_name)
            plt.clf()
        plt.show()
        end = time()
        print('Plotting alignment took {} min'.format((end - start) / 60.0))
        return df, hm

    # def set_tree_ordering(self, tree_depth=None, cache_dir=None, clustering_args={}, clustering='agglomerative'):
    #     """
    #     Determine the ordering of the sequences from the full clustering tree
    #     used when separating the alignment into sub-clusters.
    #
    #     Args:
    #         tree_depth (None, tuple, or list): The levels of the phylogenetic tree to consider when analyzing this
    #         alignment, which determines the attributes sequence_assignments and tree_ordering. The following options are
    #         available:
    #             None: All branches from the top of the tree (1) to the leaves (size) will be analyzed.
    #             tuple: If a tuple is provided with two ints these will be taken as a range, the top of the tree (1), and
    #             all branches between the first and second (non-inclusive) integer will be analyzed.
    #             list: All branches listed will be analyzed, as well as the top of the tree (1) even if not listed.
    #         cache_dir (str): The path to the directory where the clustering model can be stored for access later when
    #         identifying different numbers of clusters.
    #         clustering_args (dict): Additional arguments needed by various clustering/tree building algorithms. If no
    #         other arguments are needed (as is the case when using 'random' or default settings for 'agglomerative') the
    #         dictionary can be left empty.
    #         clustering (str): The type of clustering/tree building to use. Current options are:
    #             agglomerative
    #             upgma
    #             random
    #             custom
    #     Return:
    #         list: The explicit list of tree levels analyzed, as described above in the tree_depth Args section.
    #     """
    #     method_dict = {'agglomerative': self._agglomerative_clustering, 'upgma': self._upgma_tree,
    #                    'random': self._random_assignment, 'custom': self._custom_tree}
    #     curr_order = [0] * self.size
    #     sequence_assignments = {1: {0: set(self.seq_order)}}
    #     if tree_depth is None:
    #         tree_depth = range(1, self.size + 1)
    #     elif isinstance(tree_depth, tuple):
    #         if len(tree_depth) != 2:
    #             raise ValueError('If a tuple is provided for tree_depth, two values must be specified.')
    #         tree_depth = list(range(tree_depth[0], tree_depth[1]))
    #     elif isinstance(tree_depth, list):
    #         pass
    #     else:
    #         raise ValueError('tree_depth must be None, a tuple, or a list.')
    #     if tree_depth[0] != 1:
    #         tree_depth = [1] + tree_depth
    #     if cache_dir is None:
    #         remove_dir = True
    #         cache_dir = os.getcwd()
    #     else:
    #         remove_dir = False
    #     for k in tree_depth:
    #         cluster_list = method_dict[clustering](n_cluster=k, cache_dir=cache_dir, **clustering_args)
    #         new_clusters = self._re_label_clusters(curr_order, cluster_list)
    #         curr_order = new_clusters
    #         sequence_assignments[k] = {}
    #         for i, c in enumerate(curr_order):
    #             if c not in sequence_assignments[k]:
    #                 sequence_assignments[k][c] = set()
    #             sequence_assignments[k][c].add(self.seq_order[i])
    #     self.sequence_assignments = sequence_assignments
    #     self.tree_order = list(zip(*sorted(zip(self.seq_order, curr_order), key=lambda x: x[1]))[0])
    #     if remove_dir:
    #         joblib_dir = os.path.join(cache_dir, 'joblib')
    #         if os.path.isdir(joblib_dir):
    #             rmtree(joblib_dir)
    #     return tree_depth

    # def _random_assignment(self, n_cluster, cache_dir=None):
    #     """
    #     random_assignment
    #
    #     Randomly assigns sequence IDs to groups totaling the specified n_clusters. Sequences are split as evenly as
    #     possible with all clusters getting the same number of sequences if self.size % n_clusters = 0 and with
    #     self.size % n_clusters groups getting one additional sequence otherwise.
    #
    #     Args:
    #         n_cluster (int): The number of clusters to produce by random assignment.
    #         cache_dir (str): The path to the directory where the clustering model can be stored for access later when
    #         identifying different numbers of clusters.
    #     Returns:
    #         list: The cluster assignments for each sequence in the alignment.
    #     """
    #     if cache_dir is not None:
    #         save_dir = os.path.join(cache_dir, 'joblib')
    #         save_file = os.path.join(save_dir, 'K_{}.pkl'.format(n_cluster))
    #     else:
    #         save_dir = None
    #         save_file = None
    #     if save_dir is not None:
    #         if not os.path.isdir(save_dir):
    #             os.mkdir(save_dir)
    #         else:
    #             if os.path.isfile(save_file):
    #                 with open(save_file, 'rb') as save_handle:
    #                     return pickle.load(save_handle)
    #     cluster_sizes = np.ones(n_cluster, dtype=np.int64)
    #     min_size = self.size // n_cluster
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
    #     if save_dir is not None:
    #         with open(save_file, 'wb') as save_handle:
    #             pickle.dump(cluster_list, save_handle, pickle.HIGHEST_PROTOCOL)
    #     return cluster_list
    #
    # @staticmethod
    # def _re_label_clusters(prev, curr):
    #     """
    #     Relabel Clusters
    #
    #     This method takes in a two sets of cluster labels and ensures that the new one (curr) aggrees in its ordering
    #     with the previous one (prev). This makes for easier tracking of matching clusters when using methods which do
    #     not have stable cluster labels even if the clusters themselves are stable.
    #
    #     Args:
    #         prev (list): A list of cluster labels.
    #         curr (list): A list of cluster labels which may need to change, must have the same length as prev.
    #     Returns:
    #         list: A new list of cluster labels based on the passed in list (curr). Cluster assignment does not change,
    #         i.e. the same elements are together in clusters, but the labels change to represent the labels of those
    #         clusters in the previous (prev) set of labels.
    #     """
    #     if len(prev) != len(curr):
    #         raise ValueError('Cluster labels do not match in length: {} vs {}.'.format(len(prev), len(curr)))
    #     curr_labels = set()
    #     prev_to_curr = {}
    #     for i in range(len(prev)):
    #         prev_c = prev[i]
    #         curr_c = curr[i]
    #         if (prev_c not in prev_to_curr) and (curr_c not in curr_labels):
    #             prev_to_curr[prev_c] = {'clusters': [], 'indices': []}
    #         if curr_c not in curr_labels:
    #             curr_labels.add(curr_c)
    #             prev_to_curr[prev_c]['clusters'].append(curr_c)
    #             prev_to_curr[prev_c]['indices'].append(i)
    #     curr_to_new = {}
    #     counter = 0
    #     prev_labels = sorted(prev_to_curr.keys())
    #     for c in prev_labels:
    #         for curr_c in zip(*sorted(zip(prev_to_curr[c]['clusters'], prev_to_curr[c]['indices']),
    #                                   key=lambda x: x[1]))[0]:
    #             curr_to_new[curr_c] = counter
    #             counter += 1
    #     new_labels = [curr_to_new[c] for c in curr]
    #     return new_labels
    #
    # def visualize_tree(self, out_dir=None):
    #     """
    #     Visualize Tree
    #
    #     This method takes the sequence_assignments attribute and visualizes them as a heatmap so that the way clusters
    #     change throughout the tree can be easily seen.
    #
    #     Args:
    #         out_dir (str): The location to which the tree visualization should be saved.
    #     Returns:
    #         pd.Dataframe: The data used for generating the heatmap, with sequence IDs as the index and tree level/branch
    #         as the columns.
    #     """
    #     if out_dir is None:
    #         out_dir = os.getcwd()
    #     if self.sequence_assignments is None:
    #         raise ValueError('SeqAlignment.sequence_assignments not initialized, run set_tree_ordering prior to this '
    #                          'method being run.')
    #     check = {'SeqID': self.tree_order, 1: [0] * self.size}
    #     for k in self.sequence_assignments:
    #         curr_order = []
    #         for i in range(self.size):
    #             for c in self.sequence_assignments[k]:
    #                 if self.tree_order[i] in self.sequence_assignments[k][c]:
    #                     curr_order.append(c)
    #         check[k] = curr_order
    #     branches = sorted(self.sequence_assignments.keys())
    #     df = pd.DataFrame(check).set_index('SeqID').sort_values(by=branches[::-1])[branches]
    #     df.to_csv(os.path.join(out_dir, '{}_Sequence_Assignment.csv'.format(self.query_id)), sep='\t', header=True,
    #               index=True)
    #     heatmap(df, cmap='tab10', square=True)
    #     plt.savefig(os.path.join(out_dir, '{}_Sequence_Assignment.eps'.format(self.query_id)))
    #     plt.close()
    #     return df
    #
    # def get_branch_cluster(self, k, c):
    #     """
    #     Get Branch Cluster
    #
    #     Thus method generates a sub alignment based on a specific level/branch of the tree being analyzed and a specific
    #     node/cluster within that level/branch.
    #
    #     Args:
    #         k (int): The branching level (root <= k <= leaves where root = 1, leaves = size) to which the desired
    #         set of sequences belongs.
    #         c (int): The cluster/node in the specified branching level (1 <= c <= k) to which the desired set of
    #         sequences belongs.
    #     Returns:
    #         SeqAlignment: A new SeqAlignment object which is a subset of the current SeqAlignment corresponding to the
    #         sequences in a specific branching level and cluster of the phylogenetic tree based on this multiple sequence
    #         alignment.
    #     """
    #     if self.sequence_assignments is None:
    #         self.set_tree_ordering()
    #     cluster_seq_ids = [s for s in self.tree_order if s in self.sequence_assignments[k][c]]
    #     return self.generate_sub_alignment(sequence_ids=cluster_seq_ids)
