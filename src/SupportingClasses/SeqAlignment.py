"""
Created on Aug 17, 2017

@author: daniel
"""
import os
import pickle
import numpy as np
import pandas as pd
from time import time
from copy import deepcopy
from scipy.stats import zscore
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.Alphabet import Gapped
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from seaborn import heatmap
from FrequencyTable import FrequencyTable
from utils import build_mapping, convert_seq_to_numeric
from AlignmentDistanceCalculator import AlignmentDistanceCalculator
from EvolutionaryTraceAlphabet import FullIUPACProtein, FullIUPACDNA, MultiPositionAlphabet


class SeqAlignment(object):
    """
    This class represents an alignment as expected by Evolutionary Trace methods. The sequence of interest is
    represented by an ID passed in by the user.

    Attributes:
        file_name (str): The path to the file from which the alignment can be parsed.
        query_id (str): A sequence identifier, which should be the identifier for query sequence in the alignment file.
        alignment (Bio.Align.MultipleSeqAlignment): A biopython representation of a multiple sequence alignment and its
        sequences.
        seq_order (list): List of sequence identifiers in the order in which they were parsed from the alignment file.
        query_sequence (Bio.Seq.Seq): The sequence matching the sequence identifier given by the query_id attribute.
        seq_length (int): The length of the query sequence (including gaps and ambiguity characters).
        size (int): The number of sequences in the alignment represented by this object.
        marked (list): List of boolean values tracking whether a sequence has been marked or not (this is generally used
        to track which sequences should be skipped during an analysis, for instance in the zoom version of Evolutionary
        Trace).
        polymer_type (str): Whether the represented alignment is a 'Protein' or 'DNA' alignment (these are currently the
        only two options).
        alphabet (EvolutionaryTraceAlphabet): The protein or DNA alphabet to use for this alignment (used by default if
        calculating sequence distances).
    """

    def __init__(self, file_name, query_id, polymer_type='Protein'):
        """
        __init__

        Initiates an instance of the SeqAlignment class represents a multiple sequence alignment such that common
        actions taken on an alignment during Evolutionary Trace analyses are simple to perform.

        Args:
            file_name (str or path): The path to the file from which the alignment can be parsed. If a relative path is
                used (i.e. the ".." prefix), python's path library will be used to attempt to define the full path.
            query_id (str): The sequence identifier of interest.
            polymer_type (str): The type of polymer this alignment represents. Expected values are 'Protein' or 'DNA'.
        """
        self.file_name = os.path.abspath(file_name)
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
        Import Alignment

        This method imports the alignment using the AlignIO.read method expecting the 'fasta' format. It then updates
        the alignment, seq_order, query_sequence, seq_length, size, and marked class attributes.

        Args:
            save_file (str, optional): Path to file in which the desired alignment should be stored, or was stored
            previously. If the alignment was previously imported and stored at this location it will be loaded via
            pickle instead of reprocessing the the file in the file_name attribute.
            verbose (bool, optional): Whether or not to print the time spent while importing the alignment or not.
        """
        start = time()
        if (save_file is not None) and (os.path.exists(save_file)):
            with open(save_file, 'rb') as handle:
                alignment, seq_order, query_sequence = pickle.load(handle)
        else:
            with open(self.file_name, 'r') as file_handle:
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
                with open(save_file, 'wb') as save_handle:
                    pickle.dump((alignment, seq_order, query_sequence), save_handle, protocol=pickle.HIGHEST_PROTOCOL)
        end = time()
        if verbose:
            print('Importing alignment took {} min'.format((end - start) / 60.0))
        self.alignment = alignment
        self.seq_order = seq_order
        self.query_sequence = query_sequence
        self.seq_length = len(self.query_sequence)
        self.size = len(self.alignment)
        self.marked = [False] * self.size

    def write_out_alignment(self, file_name):
        """
        Write Out Alignment

        This method writes out the alignment in the standard fasta format.  Any sequence which is longer than 60
        positions will be split over multiple lines with 60 characters per line.

        Args:
            file_name (str): Path to file where the alignment should be written.
        """
        if self.alignment is None:
            raise TypeError('Alignment must be Bio.Align.MultipleSequenceAlignment not None.')
        if os.path.exists(file_name):
            return
        with open(file_name, 'w') as file_handle:
            AlignIO.write(self.alignment, handle=file_handle, format="fasta")

    def generate_sub_alignment(self, sequence_ids):
        """
        Generate Sub Alignment

        Initializes a new alignment which is a subset of the current alignment.

        Args:
            sequence_ids (list): A list of strings which are sequence identifiers for sequences in the current
            alignment. Other sequence ids will be skipped.
        Returns:
            SeqAlignment: A new SeqAlignment object containing the same file_name, query_id, seq_length, and query
            sequence.  The seq_order will be updated to only the passed in identifiers which are also in the current
            alignment, preserving their ordering from the current SeqAlignment object. The alignment will contain only
            the subset of sequences represented by identifiers which are present in the new seq_order.  The size is set
            to the length of the new seq_order. The marked attribute for all sequences in the sub alignment will be
            transferred from the current alignment.
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
        sequences. Instead of updating the current object, a new object is returned with updated alignment and
        query_sequence, and seq_length fields.

        Args:
            save_file (str): Path to a file where the alignment with gaps in the query sequence removed should be stored
            or was stored previously. If the updated alignment was stored previously it will be loaded from the
            specified save_file instead of processing the current alignment.
        Returns:
            SeqAlignmentq: A new alignment which is a subset of the current alignment, with only columns in the
            alignment which were not gaps in the query sequence.
        """
        start = time()
        if (save_file is not None) and os.path.exists(save_file):
            new_aln = pickle.load(open(save_file, 'r'))
        else:
            query_arr = np.array(list(self.query_sequence))
            query_ungapped_ind = np.where(query_arr != '-')[0]
            if len(query_ungapped_ind) > 0:
                new_aln = self._subset_columns(query_ungapped_ind)
            else:
                new_aln = self.alignment
            if save_file is not None:
                pickle.dump(new_aln, open(save_file, 'w'), protocol=pickle.HIGHEST_PROTOCOL)
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

    def generate_positional_sub_alignment(self, positions):
        """
        Generate Positional Sub Alignment

        This method generates an alignment with only the specified columns, meant to enable the interrogation of
        positional importance and covariance scores.

        Args:
            positions (list): A list of integers where each element of the list is a position in the sequence alignment
            which should be kept in the sub-alignment.
        Returns:
            SeqAlignment: A new sub-alignment containing all sequences from the current SeqAlignment object but with
            only the sequence positions (columns) specified.
        """
        new_alignment = SeqAlignment(self.file_name, self.query_id)
        new_alignment.query_id = deepcopy(self.query_id)
        new_alignment.query_sequence = Seq(''.join([self.query_sequence[i] for i in positions]))
        new_alignment.seq_length = len(positions)
        new_alignment.seq_order = deepcopy(self.seq_order)
        new_alignment.alignment = self._subset_columns(indices_to_keep=positions)
        new_alignment.size = deepcopy(self.size)
        new_alignment.marked = deepcopy(self.marked)
        return new_alignment

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

    def compute_effective_alignment_size(self, identity_threshold=0.62, distance_matrix=None, processes=1):
        """
        Compute Effective Alignment Size

        This method uses the distance_matrix variable (containing sequence identities) to compute the effective size of
        the current alignment. The equation (given below) and default threshold (62% identity) are taken from
        PMID:29047157.
            Meff = SUM_(i=0)^(N) of 1/n_i
            where n_i are the number of sequences where sequence identity >= the identity threshold
        Args:
            identity_threshold (float): The threshold for what is considered an identical (non-unique) sequence.
            distance_matrix (Bio.Phylo.TreeConstruction.DistanceMatrix): A precomputed identity distance matrix for this
            alignment.
            processes (int): The number of processes which can be used to compute the identity distance for determining
            effective alignment size.
        Returns:
            float: The effective alignment size of the current alignment (must be <= SeqAlignment.size)
        """
        if distance_matrix is None:
            calculator = AlignmentDistanceCalculator(protein=(self.polymer_type == 'Protein'))
            distance_matrix = calculator.get_distance(self.alignment, processes=processes)
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

        This method builds a consensus sequence from the alignment.

        Args:
            method (str): Which method to use for consensus sequence construction. The only current option is 'majority'
            though there are other possible options which may be added in the future to reflect the previous ETC tool.
                'majority' = To use the member of the alphabet which covers the majority of sequences at that position.
                If multiple characters have the same, highest, count the first one in the alphabet is used.
        Returns:
            Bio.Seq.SeqRecord: A sequence record containing the consensus sequence for this alignment.
        """
        if method != 'majority':
            raise ValueError("consensus_sequence has not been implemented for method '{}'!".format(method))
        alpha_size, gap_chars, mapping, reverse_mapping = build_mapping(alphabet=self.alphabet)
        if '-' in gap_chars:
            gap_char = '-'
        elif '.' in gap_chars:
            gap_char = '.'
        elif '*' in gap_chars:
            gap_char = '*'
        else:
            gap_char = None
        if gap_char:
            reverse_mapping = np.hstack([reverse_mapping, np.array([gap_char, gap_char])])
        numeric_aln = self._alignment_to_num(mapping)
        consensus_seq = []
        for i in range(self.seq_length):
            unique_chars, counts = np.unique(numeric_aln[:, i], return_counts=True)
            highest_count = np.max(counts)
            positions_majority = np.where(counts == highest_count)
            consensus_seq.append(reverse_mapping[unique_chars[positions_majority[0][0]]])
        consensus_record = SeqRecord(id='Consensus Sequence', seq=Seq(''.join(consensus_seq), alphabet=self.alphabet))
        return consensus_record

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
        if (z_score_cutoff is None) or (num_aln is None) or (gap_num is None):
            raise ValueError('All parameters must be specified, None is not a valid input.')
        gap_check = num_aln == gap_num
        gap_count = np.sum(gap_check, axis=1)
        gap_z_scores = zscore(gap_count)
        if np.isnan(gap_z_scores).any():
            passing_z_scores = np.array([True] * self.size)
        else:
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
            percentile_cutoff (float): The percentage (expressed as a decimal; e.g. 15% = 0.15), of gaps differing from
            the consensus, above which a sequence does not pass the filter.
        Return:
            list: A list of the sequence IDs which pass the gap cut off used for this alignment.
            list: A list of sequences which do not pass the gap cut off used for this alignment.
        """
        alpha_size, gap_chars, mapping, _ = build_mapping(alphabet=self.alphabet)
        numeric_aln = self._alignment_to_num(mapping)
        gap_number = mapping[list(gap_chars)[0]]
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

    def heatmap_plot(self, name, out_dir=None, save=True, ax=None):
        """
        Heatmap Plot

        This method creates a heatmap of the alignment so it can be easily visualized. A numerical representation of the
        amino acids is used so that cells can be colored differently for each amino acid. The ordering along the y-axis
        reflects the seq_order attribute, while the ordering along the x-axis represents the sequence positions from
        0 to seq_length.

        Args:
            name (str): Name used as the title of the plot and the filename for the saved figure (spaces will be
            replaced by underscores when saving the plot).
            out_dir (str): Path to directory where the heatmap image file should be saved. If None (default) then the
            image will be stored in the current working directory.
            save (bool): Whether or not to save the plot to file.
            ax (matplotlib.axes.Axes): Optional argument to provide the sub-plot to which the heatmap_plot should be
            drawn.
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

        if ax is None:
            # start by specifying cell size and margins
            cellsize = 0.25  # inch
            marg_top = 0.7
            marg_bottom = 0.7
            marg_left = 0.7
            marg_right = 0.7
            # number of cells along width
            cells_in_row = self.seq_length
            # determine figure width
            figwidth = cellsize * cells_in_row + marg_left + marg_right
            # number of cells along height
            cells_in_column = self.size
            # calculate figure height in inches
            figheight = cellsize * cells_in_column + marg_top + marg_bottom
            # set figure size
            fig = plt.figure(figsize=(figwidth, figheight))
            # adjust margins (relative numbers) according to absolute values
            fig.subplots_adjust(bottom=marg_bottom / figheight, top=1. - marg_top / figheight,
                                left=marg_left / figwidth, right=1. - marg_right / figwidth)

        _, _, mapping, _ = build_mapping(alphabet=self.alphabet)
        df = pd.DataFrame(self._alignment_to_num(mapping=mapping), index=self.seq_order,
                          columns=['{}:{}'.format(x, aa) for x, aa in enumerate(self.query_sequence)])
        cmap = matplotlib.cm.get_cmap('jet', len(mapping))
        hm = heatmap(data=df, cmap=cmap, center=10.0, vmin=0.0, vmax=20.0, cbar=False, square=True,
                     annot=np.array([list(seq_rec) for seq_rec in self.alignment]), fmt='', ax=ax)
        hm.set_yticklabels(hm.get_yticklabels(), fontsize=8, rotation=0)
        hm.set_xticklabels(hm.get_xticklabels(), fontsize=8, rotation=0)
        hm.tick_params(left=False, bottom=False)
        hm.set_title(name)
        if save:
            plt.savefig(file_name, bbox_inches='tight')
            plt.clf()
        plt.show()
        end = time()
        print('Plotting alignment took {} min'.format((end - start) / 60.0))
        return df, hm

    def characterize_positions(self, single=True, pair=True, single_size=None, single_mapping=None, single_reverse=None,
                               pair_size=None, pair_mapping=None, pair_reverse=None):
        """
        Characterize Positions

        This method is meant to characterize the nucleic/amino acids at all positions in the aligned sequence across all
        sequences in the current alignment. This can be used by other methods to determine with a position is conserved
        in a given alignment, or compute more complex characterizations like positional entropy etc. This method can
        characterize single positions as well as pairs of positions and this can be done at the same time or separately.

        Args:
            single (bool): Whether to characterize the nucleic/amino acid counts for single positions in the alignment.
            pair (bool): Whether to characterize the a nucleic/amino acid counts for pairs of positions in the
            alignment.
            single_size (int): Size of the single letter alphabet to use when instantiating a FrequencyTable
            single_mapping (dict): Dictionary mapping single letter alphabet to numerical positions.
            single_reverse (np.array): Array mapping numerical positions back to the single letter alphabet.
            pair_size (int): Size of the pair of letters alphabet to use when instantiating a FrequencyTable.
            pair_mapping (dict): Dictionary mapping pairs of letters in an alphabet to numerical positions.
            pair_reverse (np.array): Array mapping numerical positions back to the pairs of letters alphabet.
        Returns:
            FrequencyTable/None: The characterization of single position nucleic/amino acid counts if requested.
            FrequencyTable/None: The characterization of pairs of positions and their nucleic/amino acid counts if
            requested.
        """
        pos_specific = None
        if single:
            if (single_size is None) or (single_mapping is None) or (single_reverse is None):
                single_size_size, _, single_mapping, single_reverse = build_mapping(alphabet=Gapped(self.alphabet))
            pos_specific = FrequencyTable(alphabet_size=single_size, mapping=single_mapping,
                                          reverse_mapping=single_reverse, seq_len=self.seq_length, pos_size=1)
        pair_specific = None
        if pair:
            if (pair_size is None) or (pair_mapping is None) or (pair_reverse is None):
                pair_size, _, pair_mapping, pair_reverse = build_mapping(
                    alphabet=MultiPositionAlphabet(alphabet=Gapped(self.alphabet), size=2))
            pair_specific = FrequencyTable(alphabet_size=pair_size, mapping=pair_mapping, reverse_mapping=pair_reverse,
                                           seq_len=self.seq_length, pos_size=2)
        # Iterate over all sequences
        for s in range(self.size):
            if single:
                pos_specific.characterize_sequence(seq=self.alignment[s].seq)
            if pair:
                pair_specific.characterize_sequence(seq=self.alignment[s].seq)
        if single:
            pos_specific.finalize_table()
        if pair:
            pair_specific.finalize_table()
        return pos_specific, pair_specific

    def characterize_positions2(self, single=True, pair=True, single_letter_size=None, single_letter_mapping=None,
                                single_letter_reverse=None, pair_letter_size=None, pair_letter_mapping=None,
                                pair_letter_reverse=None, single_to_pair=None):
        """
        Characterize Positions

        This method is meant to characterize the nucleic/amino acids at all positions in the aligned sequence across all
        sequences in the current alignment. This can be used by other methods to determine with a position is conserved
        in a given alignment, or compute more complex characterizations like positional entropy etc. This method can
        characterize single positions as well as pairs of positions and this can be done at the same time or separately.
        This method is slower in all cases when analyzing an alignment of size 1 than characterize_positions, however it
        may be faster if there are many sequences and the sequence length is large.

        Args:
            single (bool): Whether to characterize the nucleic/amino acid counts for single positions in the alignment.
            pair (bool): Whether to characterize the a nucleic/amino acid counts for pairs of positions in the
            alignment.
            single_letter_size (int): Size of the single letter alphabet to use when instantiating a FrequencyTable
            single_letter_mapping (dict): Dictionary mapping single letter alphabet to numerical positions.
            single_letter_reverse (np.array): Array mapping numerical positions back to the single letter alphabet.
            pair_letter_size (int): Size of the pair of letters alphabet to use when instantiating a FrequencyTable.
            pair_letter_mapping (dict): Dictionary mapping pairs of letters in an alphabet to numerical positions.
            pair_letter_reverse (np.array): Array mapping numerical positions back to the pairs of letters alphabet.
            single_to_pair (dict): A dictionary mapping tuples of integers to a single int. The tuple of integers should
            consist of the position of the first character in a pair of letters to its numerical position and the
            position of the second character in a pair of letters to its numerical position (single_letter_mapping).
            The value that this tuple maps to should be the integer value that a pair of letters maps to
            (pair_letter_mapping).
        Returns:
            FrequencyTable/None: The characterization of single position nucleic/amino acid counts if requested.
            FrequencyTable/None: The characterization of pairs of positions and their nucleic/amino acid counts if
            requested.
        """
        if (single_letter_mapping is None) or (single_letter_size is None) or (single_letter_reverse is None):
            single_letter_size, _, single_letter_mapping, single_letter_reverse = build_mapping(
                alphabet=Gapped(self.alphabet))
        num_aln = self._alignment_to_num(mapping=single_letter_mapping)
        pos_specific = None
        if single:
            pos_specific = FrequencyTable(alphabet_size=single_letter_size, mapping=single_letter_mapping,
                                          reverse_mapping=single_letter_reverse, seq_len=self.seq_length, pos_size=1)
            pos_specific.characterize_alignment(num_aln=num_aln, single_to_pair=single_to_pair)
        pair_specific = None
        if pair:
            if (pair_letter_mapping is None) or (pair_letter_size is None) or (pair_letter_reverse is None):
                pair_letter_size, _, pair_letter_mapping, pair_letter_reverse = build_mapping(
                    alphabet=MultiPositionAlphabet(alphabet=Gapped(self.alphabet), size=2))
            pair_specific = FrequencyTable(alphabet_size=pair_letter_size, mapping=pair_letter_mapping,
                                           reverse_mapping=pair_letter_reverse, seq_len=self.seq_length, pos_size=2)
            if single_to_pair is None:
                single_to_pair = {}
                for char in pair_letter_mapping:
                    key = (single_letter_mapping[char[0]], single_letter_mapping[char[1]])
                    single_to_pair[key] = pair_letter_mapping[char]
            pair_specific.characterize_alignment(num_aln=num_aln, single_to_pair=single_to_pair)
        return pos_specific, pair_specific
