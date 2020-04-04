"""
Created on Mar 4, 2020

@author: daniel
"""
from time import time
import numpy as np
from copy import deepcopy


class MatchMismatchTable(object):
    """
    This class is meant to characterize an alignment not by considering the counts of the nucleic/amino acids as they
    occur in a single column of the alignment, but all possible pairs of transitions observed between the nucleic/amino
    acids of a column of the alignment. This can be done for a single position or for higher combinations of positions.
    In the case of single positions a match occurs only when two characters are equal, and a mismatch occurs otherwise.
    For pairs of positions (or larger groupings) a match occurs if the compared group of residues is equal or if all
    residues in the compared residues differ. A mismatch for two or more positions is when a change occurs in a
    non-concerted fashion (i.e. some but not all positions differ between the compared groups of nucleic/amino acids).
    This behavior has been chosen to accurately capture phenomena like residue covariation.

    Attributes:
        seq_len (int): The length of the sequence being passed in (should agree with num_aln.shape[1]).
        pos_size (int): The size of positions being compared (1 for single positions, 2 for pairs of positions).
        num_aln (np.array): A two dimensional array of a sequence alignment where each position represents a
        nucleic/amino acid and the number corresponds to the single_mapping of the nucleic/amino acid at that position.
        __depth (int): The number of sequences in the alignment (should agree with num_aln.shape[0]).
        single_alphabet_size (int): The number of characters present in the single character alphabet for the alignment.
        single_mapping (dict): A mapping from character to integer representing the position of each nucleic/amino acid
        in an alphabet to its position in that alphabet (and also in a FrequencyTable).
        single_reverse_mapping (np.array): An array mapping from integer to character which reverses the nucleic/amino
        acid mapping from single_mapping.
        larger_alphabet_size (int): The number of characters present in the larger (pair, quad, etc.) character alphabet
        for the alignment.
        larger_mapping (dict): A mapping from character to integer representing the position of each grouping of
        nucleic/amino acids in the larger alphabet to its position in that alphabet (and also in a FrequencyTable).
        larger_reverse_mapping (np.array): An array mapping from integer to character which reverses the nucleic/amino
        acid mapping from larger_mapping.
        single_to_larger_mapping (dict): A dictionary mapping tuples of integers to an integer, where the tuple consists
        of the single alphabet position of each element of grouping of nucleic/amino acids in the larger alphabet, and
        the value integer is the position of the grouping according to larger_mapping (e.g. if A has position 1 in the
        single letter alphabet and AA has position 1 in the larger alphabet, then one entry will be (1,1) to 1 for
        ('A', 'A') maps to 'AA').
        match_mismatch_tables (dict): An integer to np.array mapping where the integer is the position in the sequence,
        and the array contains the upper triangle evaluation of matches (+1) and mismatches (-1) for that column of the
        alignment.
    """

    def __init__(self, seq_len, num_aln, single_alphabet_size, single_mapping, single_reverse_mapping,
                 larger_alphabet_size, larger_alphabet_mapping, larger_alphabet_reverse_mapping,
                 single_to_larger_mapping, pos_size=1):
        """
        __init__

        Initialization method for MatchMismatchTable.

        Arguments:
            seq_len (int): The length of the sequence being passed in (should agree with num_aln.shape[1]).
            num_aln (np.array): A two dimensional array of a sequence alignment where each position represents a
            nucleic/amino acid and the number corresponds to the single_mapping of the nucleic/amino acid at that
            position.
            single_alphabet_size (int): The number of characters present in the single character alphabet for the
            alignment.
            single_mapping (dict): A mapping from character to integer representing the position of each nucleic/amino
            acid in an alphabet to its position in that alphabet (and also in a FrequencyTable).
            single_reverse_mapping (np.array): An array mapping from integer to character which reverses the
            nucleic/amino acid mapping from single_mapping.
            larger_alphabet_size (int): The number of characters present in the larger (pair, quad, etc.) character
            alphabet for the alignment.
            larger_alphabet_mapping (dict): A mapping from character to integer representing the position of each
            grouping of nucleic/amino acids in the larger alphabet to its position in that alphabet (and also in a
            FrequencyTable).
            larger_alphabet_reverse_mapping (np.array): An array mapping from integer to character which reverses the
            nucleic/amino acid mapping from larger_mapping.
            single_to_larger_mapping (dict): A dictionary mapping tuples of integers to an integer, where the tuple
            consists of the single alphabet position of each element of grouping of nucleic/amino acids in the larger
            alphabet, and the value integer is the position of the grouping according to larger_mapping (e.g. if A has
            position 1 in the single letter alphabet and AA has position 1 in the larger alphabet, then one entry will
            be (1,1) to 1 for ('A', 'A') maps to 'AA').
            pos_size (int): The size of positions being compared (1 for single positions, 2 for pairs of positions).
        """
        self.seq_len = seq_len
        self.pos_size = pos_size
        self.num_aln = num_aln
        self.__depth = num_aln.shape[0]
        self.single_alphabet_size = single_alphabet_size
        self.single_mapping = single_mapping
        self.single_reverse_mapping = single_reverse_mapping
        self.larger_alphabet_size = larger_alphabet_size
        self.larger_mapping = larger_alphabet_mapping
        self.larger_reverse_mapping = larger_alphabet_reverse_mapping
        self.single_to_larger_mapping = single_to_larger_mapping
        self.match_mismatch_tables = None

    def identify_matches_mismatches(self):
        """
        Identify Matches Mismatches

        The method populates the match_mismatch_tables attribute by creating an array of matches and mismatches for each
        position and associating it with its 1-D numerical position in the match_mismatch_tables dictionary.
        """
        start = time()
        pos_table_dict = {}
        for i in range(self.seq_len):
            if i not in pos_table_dict:
                unique_chars = np.unique(self.num_aln[:, i])
                mm_table = np.zeros((self.__depth, self.__depth))
                upper_mask = np.triu(np.ones((self.__depth, self.__depth)), k=1)
                for char in unique_chars:
                    occurrences = self.num_aln[:, i] == char
                    char_mask = np.zeros((self.__depth, self.__depth))
                    char_mask[occurrences, :] = 1.0
                    final_mask = upper_mask * char_mask
                    matches = np.outer(occurrences, occurrences)
                    mm_table += final_mask * matches
                    mismatches = 1 - matches
                    mm_table -= final_mask * mismatches
                pos_table_dict[i] = mm_table
        self.match_mismatch_tables = pos_table_dict
        end = time()
        print('It took {} seconds to identify matches and mismatches.'.format(end - start))

    def get_status_and_character(self, pos, seq_ind1, seq_ind2):
        """
        Get Status and Character

        This method returns the character observed at a given position between two sequences and the 'match' or
        'mismatch' status. If the position size being considered is one the character will consist of two nucleic/amino
        acids since each character from the two sequences is returned, if position size is two then four characters will
        be returned (two for each sequence).

        Arguments:
            pos (int/tuple): The position of the alignment being considered. An integer is expected if position size is
            one, a tuple is expected if position size is two or greater.
            seq_ind1 (int): The index of the first sequence to consider when determining the match/mismatch of a
            transition.
            seq_ind2 (int): The index of the second sequence to consider when determining the match/mismatch of a
            transition.
        Returns:
            str: 'match' if the position is the same of a concerted change in the two sequences, and 'mismatch'
            otherwise.
            str: The (two, four, or greater length) character observed at a given position between the two specified
            sequences.
        """
        if seq_ind1 >= seq_ind2:
            raise ValueError('Matches and mismatches are defined only for the upper triangle of sequence comparisons, '
                             'please provide sequence indices such that seq_ind1 < seq_ind2.')
        if isinstance(pos, int):
            char_tup = (self.num_aln[seq_ind1, pos], self.num_aln[seq_ind2, pos])
            status = self.match_mismatch_tables[pos][seq_ind1, seq_ind2] == 1
        elif isinstance(pos, tuple):
            char_tup = ([], [])
            status = 0
            for x in range(len(pos)):
                char_tup[0].append(self.num_aln[seq_ind1, pos[x]])
                char_tup[1].append(self.num_aln[seq_ind2, pos[x]])
                status += self.match_mismatch_tables[pos[x]][seq_ind1, seq_ind2]
            char_tup = tuple(char_tup[0] + char_tup[1])
            status = np.abs(status) == len(pos)
        else:
            return ValueError('Received a position with type other than int or tuple.')
        char = self.larger_reverse_mapping[self.single_to_larger_mapping[char_tup]]
        ret_status = 'match' if status else 'mismatch'
        return ret_status, char

    def get_depth(self):
        """
        Get Depth

        This function returns the depth of the MatchMismatchTable, i.e. the number of sequences in the characterized
        alignment.

        Return:
             int: The number of sequences in the characterized alignment.
        """
        return deepcopy(self.__depth)

    def _get_characters_and_statuses_single_pos(self, pos, indices1, indices2):
        """
        Get Characters And Statuses Single Position

        This function returns all characters and their match/mismatch status for a given position at a specified set of
        indices. These indices indicate pairs of sequences, and therefore each entry in the return should be equivalent
        to calling get_status_and_character for a position and two sequences.

        Arguments:
            pos (int): A position in the sequence being characterized (bounds are between 0 and the sequence length).
            indices1 (list/np.array): A set of sequence indices indicating the first sequence in each comparison for
            which the character and match/mismatch status should be returned. This value should always be lower than the
            corresponding sequence index provided in indices2 since only the upper triangle is used in characterization.
            indices2 (list/np.array): A set of sequence indices indicating the second sequence in each comparison for
            which the character and match/mismatch status should be returned. This value should always be greater than
            the corresponding sequence index provided in indices1 since only the upper triangle is used in
            characterization.
        Returns:
             np.array: The single letter alphabet numerical representation for the characters in the alignment at this
             position for the sequences specified in index1.
             np.array: The single letter alphabet numerical representation for the characters in the alignment at this
             position for the sequences specified in index2.
             np.array: The match/mismatch values for all comparisons between sequences in index1 and index2 at the
             specified position. The possible values are 1 (for a match), -1 (for a mismatch), and 0 (if a value was
             returned from the diagonal or lower triangle because index1 and index2 were not defined as intended).
        """
        s1_chars = self.num_aln[indices1, pos]
        s2_chars = self.num_aln[indices2, pos]
        return s1_chars[np.newaxis].T, s2_chars[np.newaxis].T, self.match_mismatch_tables[pos][indices1, indices2]

    def _get_characters_and_statuses_multi_pos(self, pos, indices1, indices2):
        """
        Get Characters And Statuses Multi Position

        This function returns all characters and their match/mismatch status for a given (pair or larger) position at a
        specified set of indices. These indices indicate pairs of sequences, and therefore each entry in the return
        should be equivalent to calling get_status_and_character for a position and two sequences.

        Arguments:
            pos (tuple): A position defined as two or more single positions in the sequence being characterized (bounds
            are between 0 and the sequence length for each specific position).
            indices1 (list/np.array): A set of sequence indices indicating the first sequence in each comparison for
            which the character and match/mismatch status should be returned. This value should always be lower than the
            corresponding sequence index provided in indices2 since only the upper triangle is used in characterization.
            indices2 (list/np.array): A set of sequence indices indicating the second sequence in each comparison for
            which the character and match/mismatch status should be returned. This value should always be greater than
            the corresponding sequence index provided in indices1 since only the upper triangle is used in
            characterization.
        Returns:
             np.array: The numerical representation (determined by the size of the position and alphabet used) for the
             characters in the alignment at this position for the sequences specified in index1.
             np.array: The single letter alphabet numerical representation for the characters in the alignment at this
             position for the sequences specified in index2.
             np.array: The match/mismatch values for all comparisons between sequences in index1 and index2 at the
             specified position. The returned values are the sum of single position match/mismatch scores and so may
             range from -x to x where x is the size of a position (e.g. 2 for pairs), therefore a score of x represents
             all specific positions being matches and a score of -x represents all specific positions being a mismatch.
        """
        cumulative_s1_chars = []
        cumulative_s2_chars = []
        cumulative_status = None
        for i in range(len(pos)):
            curr_s1, curr_s2, curr_status = self._get_characters_and_statuses_single_pos(pos=pos[i], indices1=indices1,
                                                                                         indices2=indices2)
            cumulative_s1_chars.append(curr_s1)
            cumulative_s2_chars.append(curr_s2)
            if cumulative_status is None:
                cumulative_status = curr_status
            else:
                cumulative_status += curr_status
        s1_chars = np.hstack(cumulative_s1_chars)
        s2_chars = np.hstack(cumulative_s2_chars)
        return s1_chars, s2_chars, cumulative_status

    def get_upper_triangle(self, pos, indices):
        """
        Get Upper Triangle

        This function returns the characters and match/mismatch statuses for a subset of the upper triangle of sequence
        comparisons for a specified position. This is intended for use during the characterization step of a trace.

        Arguments:
            pos (int/tuple): A position defined as one (int) or two or more single positions (tuple) in the sequence
            being characterized (bounds are between 0 and the sequence length for each specific position).
            indices (list/np.array): A set of sequence indices indicating the members of the upper triangle for which
            character and match/mismatch status should be returned.
        Returns:
             np.array: The numerical representation of the characters for the sequences compared in the specified upper
             triangle. If the position specified was a single position the returned alphabet will correspond to pairs
             (since two sequences were compared and thus two characters were observed), if the position was a pair or
             larger then the returned character will map back to an alphabet two times the size of the input position
             (in this case all character contributions from sequence 1 are added to the character before all
             contributions from sequence 2).
             np.array: The match/mismatch status of all sequence comparisons performed in the subset of the upper
             triangle. If the position had size 1, then the match/mismatch string will correspond directly to whether
             the observed characters in the compared sequences were the same or not. If the position has a size larger
             than 1, the match status may correspond to all single positions matching (conservation) or all single
             positions mismatching (concerted change, i.e. covariation).
        """
        all_indices = np.triu_indices(n=self.__depth, k=1)
        mask = np.in1d(all_indices[0], indices) & np.in1d(all_indices[1], indices)
        s1_ind = all_indices[0][mask]
        s2_ind = all_indices[1][mask]
        if isinstance(pos, int):
            s1_chars, s2_chars, status = self._get_characters_and_statuses_single_pos(pos=pos, indices1=s1_ind,
                                                                                      indices2=s2_ind)
            pos_len = 1
        elif isinstance(pos, tuple):
            s1_chars, s2_chars, status = self._get_characters_and_statuses_multi_pos(pos=pos, indices1=s1_ind,
                                                                                     indices2=s2_ind)
            pos_len = len(pos)
            status = np.abs(status)
        else:
            raise ValueError('Pos has type other than the expected int or tuple.')
        combined_char = np.hstack([s1_chars, s2_chars])
        final_chars = np.array([self.larger_reverse_mapping[self.single_to_larger_mapping[char_tup]]
                                for char_tup in map(tuple, combined_char)])
        status_con = np.array(['mismatch', 'match'])
        final_status = status_con[(status == pos_len) * 1]
        return final_chars, final_status

    def get_upper_rectangle(self, pos, indices1, indices2):
        """
        Get Upper Rectangle

        This function returns the characters and match/mismatch statuses for a subset of the upper triangle of sequence
        comparisons for a specified position. This is intended for use during the characterization step of a trace.

        Arguments:
            pos (int/tuple): A position defined as one (int) or two or more single positions (tuple) in the sequence
            being characterized (bounds are between 0 and the sequence length for each specific position).
            indices1 (list/np.array): A set of sequence indices indicating the members of the rectangle in the upper
            triangle for which character and match/mismatch status should be returned. This should not overlap with
            indices2, though this behavior is not enforced, the expectation is that indices will belong to sequences in
            two different sub-alignments of the larger alignment characterized by this MatchMismatchTable.
            indices2 (list/np.array): A set of sequence indices indicating the members of the rectangle in the upper
            triangle for which character and match/mismatch status should be returned. This should not overlap with
            indices2, though this behavior is not enforced, the expectation is that indices will belong to sequences in
            two different sub-alignments of the larger alignment characterized by this MatchMismatchTable.
        Returns:
             np.array: The numerical representation of the characters for the sequences compared in the specified upper
             triangle. If the position specified was a single position the returned alphabet will correspond to pairs
             (since two sequences were compared and thus two characters were observed), if the position was a pair or
             larger then the returned character will map back to an alphabet two times the size of the input position
             (in this case all character contributions from sequence 1 are added to the character before all
             contributions from sequence 2).
             np.array: The match/mismatch status of all sequence comparisons performed in the subset of the upper
             triangle. If the position had size 1, then the match/mismatch string will correspond directly to whether
             the observed characters in the compared sequences were the same or not. If the position has a size larger
             than 1, the match status may correspond to all single positions matching (conservation) or all single
             positions mismatching (concerted change, i.e. covariation).
        """
        indices = np.triu_indices(n=self.__depth, k=1)
        mask1 = np.in1d(indices[0], indices1) & np.in1d(indices[1], indices2)
        mask2 = np.in1d(indices[0], indices2) & np.in1d(indices[1], indices1)
        mask = mask1 | mask2
        s1_ind = indices[0][mask]
        s2_ind = indices[1][mask]
        if isinstance(pos, int):
            s1_chars, s2_chars, status = self._get_characters_and_statuses_single_pos(pos=pos, indices1=s1_ind,
                                                                                      indices2=s2_ind)
            pos_len = 1
        elif isinstance(pos, tuple):
            s1_chars, s2_chars, status = self._get_characters_and_statuses_multi_pos(pos=pos, indices1=s1_ind,
                                                                                     indices2=s2_ind)
            pos_len = len(pos)
            status = np.abs(status)
        else:
            raise ValueError('Pos has type other than the expected int or tuple.')
        combined_char = np.hstack([s1_chars, s2_chars])
        final_chars = np.array([self.larger_reverse_mapping[self.single_to_larger_mapping[char_tup]]
                                for char_tup in map(tuple, combined_char)])
        status_con = np.array(['mismatch', 'match'])
        final_status = status_con[(status == pos_len) * 1]
        return final_chars, final_status
