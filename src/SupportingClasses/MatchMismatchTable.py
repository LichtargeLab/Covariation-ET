"""
Created on Mar 4, 2020

@author: daniel
"""
from time import time
import numpy as np
from copy import deepcopy
from FrequencyTable import FrequencyTable


class MatchMismatchTable(object):
    """
    This class is meant to characterize an alignment not by considering the counts of the nucleic/amino acids as they
    occur in a single column of the alignment, but all possible pairs of transitions observed between the nucleic/amino
    acids of a column of the alignment. This can be done for a single position or for higher comibinations of positions.
    In the case of single positions a match occurs only when two characters are equal, and a mismatch occurs otherwise.
    For pairs of positions (or larger groupings) a match occurs if the compared group of residues is equal or if all
    residues in the compared residues differ. A mismatch for two or more positions is when a change occurs in a
    non-concerted fashion (i.e. some but not all positions differ between the compared groups of nucleic/amino acids).
    This behavior has been chosen to accurately capture phenomena like residue covariation.

    Attributes:
        seq_len (int): The length of the sequence being passed in (should agree with num_aln.shape[1]).
        pos_size (int): The size of positions being compared (1 for single positions, 2 for pairs of positions)
        num_aln (np.array): A two dimensional array of a sequence alignment where each position represents a
        nucleic/amino acid and the number corresponds to the single_mapping of the nucleic/amino acid at that position.
        __depth (int): The number of sequences in the alignment (should agree with num_aln.shape[0]).
        single_alphabet_size (int): The number of characters present in the single character alphabet for the alignment.
        single_mapping (dict): A mapping from character to integer representing the position of each nucleic/amino acid
        in an alphabet to its position in that alphabet (and also in a FrequencyTable).
        single_reverse_mapping (dict): A mapping from integer to character which reverses the nucleic/amino acid mapping
        from single_mapping.
        larger_alphabet_size (int): The number of characters present in the larger (pair, quad, etc.) character alphabet
        for the alignment.
        larger_mapping (dict): A mapping from character to integer representing the position of each grouping of
        nucleic/amino acids in the larger alphabet to its position in that alphabet (and also in a FrequencyTable).
        larger_reverse_mapping (dict): A mapping from integer to character which reverses the nucleic/amino acid mapping
        from larger_mapping.
        single_to_larger_mapping (dict): A dictionary mapping tuples of integers to an integer, where the tuple consists
        of the single alphabet position of each element of grouping of nucleic/amino acids in the larger alphabet, and
        the value integer is the position of the grouping according to larger_mapping (e.g. if A has position 1 in the
        single letter alphabet and AA has position 1 in the larger alphabet, then one entry will be (1,1) to 1 for
        ('A', 'A') maps to 'AA').
        match_mismatch_tables (dict): An integer to np.array mapping where the integer is the position in the sequence,
        and the array contains the upper triangle evaluation of matches (+1) and mismatches (-1) for that column of the
        alignment.
        match_freq_table (FrequencyTable): A data structure tallying the specific matches observed at each position in
        alignment.
        mismatch_freq_table (FrequencyTable): A data structure tallying the specific mismatches observed at each
        position in the alignment.
    """

    def __init__(self, seq_len, num_aln, single_alphabet_size, single_mapping, single_reverse_mapping,
                 larger_alphabet_size, larger_alphabet_mapping, larger_alphabet_reverse_mapping,
                 single_to_larger_mapping, pos_size=1):
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
        self.match_freq_table = None
        self.mismatch_freq_table = None

    def identify_matches_mismatches(self):
        start = time()
        pos_table_dict = {}
        for i in range(self.seq_len):
            if i not in pos_table_dict:
                unique_chars = np.unique(self.num_aln[:, i])
                mm_table = np.zeros((self.__depth, self.__depth))
                upper_mask = np.triu(np.ones((self.__depth, self.__depth)), k=1)
                for char in unique_chars:
                    occurences = self.num_aln[:, i] == char
                    char_mask = np.zeros((self.__depth, self.__depth))
                    char_mask[occurences, :] = 1.0
                    final_mask = upper_mask * char_mask
                    matches = np.outer(occurences, occurences)
                    mm_table += final_mask * matches
                    mismatches = 1 - matches
                    mm_table -= final_mask * mismatches
                pos_table_dict[i] = mm_table
        self.match_mismatch_tables = pos_table_dict
        end = time()
        print('It took {} seconds to identify matches and mismatches.'.format(end - start))

    # def _characterize_single_mm(self, match_table, mismatch_table):
    #     for pos in self.match_mismatch_tables:
    #         matches = self.match_mismatch_tables[pos] > 0
    #         match_ind = np.nonzero(matches)
    #         # match_counts = {}
    #         for i in range(len(match_ind[0])):
    #             pair = (self.num_aln[match_ind[0][i], pos], self.num_aln[match_ind[1][i], pos])
    #             char = self.larger_reverse_mapping[self.single_to_larger_mapping[pair]]
    #             match_table._increment_count(pos=(pos1, pos2), char=char)
    #         #     if char not in match_counts:
    #         #         match_counts[char] = 0
    #         #     match_counts[char] += 1
    #         # for char in match_counts:
    #         #     match_table._increment_count(pos=pos, char=char, amount=match_counts[char])
    #         mismatches = self.match_mismatch_tables[pos] < 0
    #         mismatch_ind = np.nonzero(mismatches)
    #         # mismatch_counts = {}
    #         for i in range(len(mismatch_ind[0])):
    #             pair = (self.num_aln[mismatch_ind[0][i], pos], self.num_aln[mismatch_ind[1][i], pos])
    #             char = self.larger_reverse_mapping[self.single_to_larger_mapping[pair]]
    #             mismatch_table._increment_count(pos=(pos1, pos2), char=char)
    #         #     if char not in mismatch_counts:
    #         #         mismatch_counts[char] = 0
    #         #     mismatch_counts[char] += 1
    #         # for char in mismatch_counts:
    #         #     mismatch_table._increment_count(pos=pos, char=char, amount=mismatch_counts[char])
    #
    # def _characterize_pair_mm(self, match_table, mismatch_table):
    #     for pos1 in range(self.seq_len):
    #         # print('POS: ', pos1)
    #         for pos2 in range(pos1 + 1, self.seq_len):
    #             # print('Pos: {}, {}'.format(pos1, pos2))
    #             matches1 = self.match_mismatch_tables[pos1] > 0
    #             matches2 = self.match_mismatch_tables[pos2] > 0
    #             mismatches1 = self.match_mismatch_tables[pos1] < 0
    #             mismatches2 = self.match_mismatch_tables[pos2] < 0
    #             matches = (matches1 * matches2) + (mismatches1 * mismatches2)
    #             match_ind = np.nonzero(matches)
    #             # match_counts = {}
    #             for i in range(len(match_ind[0])):
    #                 quad = (self.num_aln[match_ind[0][i], pos1], self.num_aln[match_ind[1][i], pos1],
    #                         self.num_aln[match_ind[0][i], pos2], self.num_aln[match_ind[1][i], pos2])
    #                 char = self.larger_reverse_mapping[self.single_to_larger_mapping[quad]]
    #                 match_table._increment_count(pos=(pos1, pos2), char=char)
    #             #     if char not in match_counts:
    #             #         match_counts[char] = 0
    #             #     match_counts[char] += 1
    #             # for char in match_counts:
    #             #     match_table._increment_count(pos=(pos1, pos2), char=char, amount=match_counts[char])
    #             mismatches = np.triu((1 - matches), k=1)
    #             mismatch_ind = np.nonzero(mismatches)
    #             # mismatch_counts = {}
    #             for i in range(len(mismatch_ind[0])):
    #                 quad = (self.num_aln[mismatch_ind[0][i], pos1], self.num_aln[mismatch_ind[1][i], pos1],
    #                         self.num_aln[mismatch_ind[0][i], pos2], self.num_aln[mismatch_ind[1][i], pos2])
    #                 char = self.larger_reverse_mapping[self.single_to_larger_mapping[quad]]
    #                 mismatch_table._increment_count(pos=(pos1, pos2), char=char)
    #             #     if char not in mismatch_counts:
    #             #         mismatch_counts[char] = 0
    #             #     mismatch_counts[char] += 1
    #             # for char in mismatch_counts:
    #             #     mismatch_table._increment_count(pos=(pos1, pos2), char=char, amount=mismatch_counts[char])
    #
    # def characterize_matches_mismatches(self):
    #     start = time()
    #     # The FrequencyTable checks that the alphabet matches the provided position size by looking whether the first
    #     # element of the mapping dictionary has the same size as the position since we are looking at something twice
    #     # the size of the position (either pairs for single positions or quadruples for pairs of positions) this causes
    #     # an error. Therefore I introduced this hack, to use a dummy dictionary to get through proper initialization,
    #     # which does expose a vulnerability in the FrequencyTable class (the check could be more stringent) but it
    #     # allows for this more flexible behavior.
    #     dummy_dict = {('{0}' * self.pos_size).format(next(iter(self.single_mapping))): 0}
    #     match_table = FrequencyTable(alphabet_size=self.larger_alphabet_size, mapping=dummy_dict,
    #                                  reverse_mapping=self.larger_reverse_mapping, seq_len=self.seq_len,
    #                                  pos_size=self.pos_size)
    #     # Completes the hack just described, providing the correct mapping table to replace the dummy table provided.
    #     match_table.mapping = self.larger_mapping
    #     mismatch_table = FrequencyTable(alphabet_size=self.larger_alphabet_size, mapping=dummy_dict,
    #                                     reverse_mapping=self.larger_reverse_mapping, seq_len=self.seq_len,
    #                                     pos_size=self.pos_size)
    #     # Completes the hack just described, providing the correct mapping table to replace the dummy table provided.
    #     mismatch_table.mapping = self.larger_mapping
    #     if self.pos_size == 1:
    #         self._characterize_single_mm(match_table, mismatch_table)
    #     elif self.pos_size == 2:
    #         self._characterize_pair_mm(match_table, mismatch_table)
    #     else:
    #         raise ValueError('MatchMismatchTable.characterize_matches_mismtaches is only implemented for pos_size 1 or '
    #                          '2 at this time!')
    #     upper_triangle_count = ((self.seq_len**2) - self.seq_len) / 2.0
    #     match_table.set_depth(int(upper_triangle_count))
    #     mismatch_table.set_depth(int(upper_triangle_count))
    #     match_table.finalize_table()
    #     mismatch_table.finalize_table()
    #     self.match_freq_table = match_table
    #     self.mismatch_freq_table = mismatch_table
    #     end = time()
    #     print('It took {} seconds to characterize matches and mismatches.'.format(end - start))

    def get_status_and_character(self, pos, seq_ind1, seq_ind2):
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
        return deepcopy(self.__depth)

    # def subset_table(self, indices):
    #     sub_table = MatchMismatchTable(seq_len=self.seq_len, num_aln=self.num_aln[indices, :],
    #                                    single_alphabet_size=self.single_alphabet_size,
    #                                    single_mapping=self.single_mapping,
    #                                    single_reverse_mapping=self.single_reverse_mapping,
    #                                    larger_alphabet_size=self.larger_alphabet_size,
    #                                    larger_alphabet_mapping=self.larger_mapping,
    #                                    larger_alphabet_reverse_mapping=self.larger_reverse_mapping,
    #                                    single_to_larger_mapping=self.single_to_larger_mapping, pos_size=self.pos_size)
    #     sub_table.__depth = self.__depth
    #     if self.match_mismatch_tables:
    #         sub_table.match_mismatch_tables = {pos: self.match_mismatch_tables[pos][indices, :][:, indices]
    #                                            for pos in self.match_mismatch_tables}
    #     if self.match_freq_table and self.mismatch_freq_table:
    #         sub_table.characterize_matches_mismatches()