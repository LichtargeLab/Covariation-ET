"""
Created on May 23, 2019

@author: Daniel Konecki
"""
import os
import numpy as np
import pandas as pd
from time import time
from copy import deepcopy
from Bio import Alphabet
from scipy.sparse import lil_matrix, save_npz
from utils import build_mapping


class FrequencyTable(object):
    """
    This class represents the position or pair specific nucleic or amino acid counts for a given alignment.

    Attributes:
        alphabet (Bio.Alphabet.Alphabet/Bio.Alphabet.Gapped): The un/gapped alphabet for which this frequency table is
        valid.
        mapping (dict): A dictionary mapping the alphabet of the alignment being characterized to numerical positions.
        reverse_mapping (dict): A dictionary mapping positions back to alphabet characterized.
        position_size (int): How big a "position" is, i.e. if the frequency table measures single positions this should
        be 1, if it measures pairs of positions this should be 2, etc.
        num_pos (int): The number of positions being characterized.
        __position_table (scipy.sparse.lil_matrix/csc_matrix): A structure storing the position specific counts for
        amino acids found in the alignment.
        __frequencies (bool): Whether or not the frequencies for this table have been computed yet or not.
        __depth (int):
    """

    # def __init__(self, alphabet, mapping, seq_len, pos_size=1):
    #     """
    #     Initialization for a FrequencyTable object.
    #
    #     Args:
    #         alphabet (Bio.Alphabet.Alphabet/Bio.Alphabet.Gapped): The alphabet for which the frequency counts tracked by
    #         this table are valid.
    #         seq_len (int): The length of the sequences in the alignment characterized by this FrequencyTable.
    #         pos_size (int): The size of a position in the alignment to be characterized (single positions = 1, pairs of
    #         positions = 2, etc.).
    #     """
    #     if alphabet.size != pos_size:
    #         raise ValueError('Alphabet size must be equal to pos_size!')
    #     # self.alphabet = alphabet
    #     self.mapping = mapping
    #     # This will have strange effects for the gap characters not included in the actual alphabet but they should also
    #     # not matter here...
    #     self.reverse_mapping = {value: key for key, value in mapping.items()}
    #     self.position_size = pos_size
    #     self.sequence_length = seq_len
    #     self.__position_table = {}
    #     self.__depth = 0
    #     self.__frequencies = False

    def __init__(self, alphabet_size, mapping, seq_len, pos_size=1):
        """
        Initialization for a FrequencyTable object.

        Args:
            alphabet (Bio.Alphabet.Alphabet/Bio.Alphabet.Gapped): The alphabet for which the frequency counts tracked by
            this table are valid.
            seq_len (int): The length of the sequences in the alignment characterized by this FrequencyTable.
            pos_size (int): The size of a position in the alignment to be characterized (single positions = 1, pairs of
            positions = 2, etc.).
        """
        if len(list(mapping.keys())[0]) != pos_size:
            raise ValueError('Alphabet size must be equal to pos_size!')
        self.mapping = mapping
        # This will have strange effects for the gap characters not included in the actual alphabet but they should also
        # not matter here...
        self.reverse_mapping = {value: key for key, value in mapping.items() if value < alphabet_size}
        self.position_size = pos_size
        self.sequence_length = seq_len
        if pos_size == 1:
            self.num_pos = seq_len
        elif pos_size == 2:
            self.num_pos = int(np.sum(range(seq_len + 1)))
        else:
            raise ValueError('FrequencyTable not implemented to handle pos_size: {}'.format(pos_size))
        self.__position_table = lil_matrix((self.num_pos, alphabet_size))
        self.__depth = 0

    # def _add_position(self, pos):
    #     """
    #     Add Position
    #
    #     This function adds a given position to the __position_table.
    #
    #     Args
    #         pos (int/tuple): A sequence position from the alignment.
    #     """
    #     if pos not in self.__position_table:
    #         if self.position_size == 1:
    #             if not isinstance(pos, int):
    #                 raise TypeError('Position does not match size specification: {} X {}'.format(self.position_size,
    #                                                                                              type(pos)))
    #         elif self.position_size > 1:
    #             if not isinstance(pos, tuple) or len(pos) != self.position_size:
    #                 raise TypeError('Position does not match size specification: {} X {}'.format(self.position_size,
    #                                                                                              type(pos)))
    #         else:
    #             pass
    #         self.__position_table[pos] = {}

    # def _add_pos_char(self, pos, char):
    #     """
    #     Add Pos Char
    #
    #     Add a character to the table for a specific position.
    #
    #     Args:
    #         pos (int/tuple): A sequence position from the alignment.
    #         char (char/str): The character or string to add for the specified position.
    #     """
    #     if char not in self.alphabet.letters:
    #         raise ValueError('The character {} is not in the specified alphabet: {}'.format(char,
    #                                                                                         self.alphabet.letters))
    #     self._add_position(pos)
    #     if char not in self.__position_table[pos]:
    #         self.__position_table[pos][char] = {'count': 0}

    def __convert_pos(self, pos):
        if self.position_size == 1 and not isinstance(pos, int):
            raise TypeError('Positions for FrequencyTable with position_size==1 must be integers')
        if (self.position_size > 1) and not isinstance(pos, tuple) and (len(pos) != self.position_size):
            raise TypeError('Positions for FrequencyTable with position_size>1 must have length == position_size')
        if self.position_size == 1:
            final = pos
        elif self.position_size == 2:
            i_factor = np.sum([self.sequence_length - x for x in range(pos[0])])
            j_factor = pos[1] - pos[0]
            final = i_factor + j_factor
        else:
            raise ValueError('Position conversion not implemented for position sizes other than 1 or 2.')
        return int(final)

    # def _increment_count(self, pos, char):
    #     """
    #     Increment Count
    #
    #     This function increments the count of a character or string for the specified position. It also updates the
    #     frequencies attribute so that stale frequencies cannot be accidentally used.
    #
    #     Args:
    #         pos (int/tuple): A sequence position from the alignment.
    #         char (char/str): The character or string to add for the specified position.
    #     """
    #     self._add_pos_char(pos, char)
    #     self.__position_table[pos][char]['count'] += 1
    #     self.__frequencies = False

    def _increment_count(self, pos, char, amount=1):
        position = self.__convert_pos(pos=pos)
        char_pos = self.mapping[char]
        self.__position_table[position, char_pos] += 1

    def characterize_sequence(self, seq):
        # Iterate over all positions
        for i in range(self.sequence_length):
            # If single is specified, track the amino acid for this sequence and position
            if self.position_size == 1:
                self._increment_count(pos=i, char=seq[i])
            # If pair is not specified continue to the next position
            if self.position_size != 2:
                continue
            # If pair is specified iterate over all positions up to the current one (filling in upper triangle,
            # including the diagonal)
            for j in range(i, self.sequence_length):
                # Track the pair of amino acids for the positions i,j
                self._increment_count(pos=(i, j), char='{}{}'.format(seq[i], seq[j]))
        self.__depth += 1

    def finalize_table(self):
        self.__position_table = self.__position_table.tocsc()

    def get_table(self):
        """
        Get Table

        Returns the dictionary storing the position specific counts for the characters present in the alignment.

        Returns:
            dict: A nested dictionary where the first level describes positions in the MSA and maps to a second set of
            dictionaries where the key is the character from the alphabet of interest mapping to the count of that
            character at that position.
        """
        table = deepcopy(self.__position_table)
        return table

    def get_depth(self):
        """
        Get Depth

        Returns the maximum number of observations for any position.

        Returns:
             int: The maximum number of observations found for any position in the FrequencyTable
        """
        depth = deepcopy(self.__depth)
        return depth

    # def get_positions(self):
    #     """
    #     Get Positions
    #
    #     Provides the positions tracked in this frequency table.
    #
    #     Returns:
    #         list: The positions tracked in this frequency table.
    #     """
    #     positions = list(sorted(self.__position_table.keys()))
    #     return positions

    def get_positions(self):
        """
        Get Positions

        Provides the positions tracked in this frequency table.

        Returns:
            list: The positions tracked in this frequency table.
        """
        if self.position_size == 1:
            positions = list(range(self.sequence_length))
        elif self.position_size == 2:
            positions = []
            for i in range(self.sequence_length):
                for j in range(i, self.sequence_length):
                    positions.append((i, j))
        else:
            raise ValueError('Get positions not implemented for position sizes other than 1 or 2.')
        return positions

    # def get_chars(self, pos):
    #     """
    #     Get Chars
    #
    #     Returns the characters from the alphabet of the MSA (and gaps) present at a given position.
    #
    #     Args:
    #         pos (int/tuple): A sequence position from the alignment.
    #     Returns:
    #         list: All characters present at the specified position in the alignment.
    #     """
    #     characters = list(self.__position_table[pos].keys())
    #     return characters

    def get_chars(self, pos):
        """
        Get Chars

        Returns the characters from the alphabet of the MSA (and gaps) present at a given position.

        Args:
            pos (int/tuple): A sequence position from the alignment.
        Returns:
            list: All characters present at the specified position in the alignment.
        """
        position = self.__convert_pos(pos=pos)
        character_positions = np.nonzero(self.__position_table[position, :])
        characters = [self.reverse_mapping[char_pos] for char_pos in character_positions[1]]
        return characters

    # def get_count(self, pos, char):
    #     """
    #     Get Count
    #
    #     Returns the count for a character at a specific position, if the character is not present at that position 0 is
    #     returned.
    #
    #     Args:
    #         pos (int/tuple): A sequence position from the alignment.
    #         char (char/str): The character or string to look for at the specified position.
    #     Returns:
    #         int: The count of the specified character at the specified position.
    #     """
    #     if (pos in self.__position_table) and (char in self.__position_table[pos]):
    #         count = self.__position_table[pos][char]['count']
    #     else:
    #         count = 0
    #     return count

    def get_count(self, pos, char):
        """
        Get Count

        Returns the count for a character at a specific position, if the character is not present at that position 0 is
        returned.

        Args:
            pos (int/tuple): A sequence position from the alignment.
            char (char/str): The character or string to look for at the specified position.
        Returns:
            int: The count of the specified character at the specified position.
        """
        position = self.__convert_pos(pos=pos)
        char_pos = self.mapping[char]
        count = self.__position_table[position, char_pos]
        return count

    # def get_count_array(self, pos):
    #     """
    #     Get Count Array
    #
    #     Returns an array containing the counts for all characters at a specified position, the order for the counts is
    #     the same as the order of the characters returned by get_chars().
    #
    #     Args:
    #         pos (int/tuple): A sequence position from the alignment.
    #     Returns:
    #         np.array: An array of the counts for characters at a given position.
    #     """
    #     if pos in self.__position_table:
    #         arr = np.array([self.__position_table[pos][char]['count'] for char in self.get_chars(pos)],
    #                        dtype=np.dtype(int))
    #     else:
    #         arr = None
    #     return arr

    def get_count_array(self, pos):
        """
        Get Count Array

        Returns an array containing the counts for all characters at a specified position, the order for the counts is
        the same as the order of the characters returned by get_chars().

        Args:
            pos (int/tuple): A sequence position from the alignment.
        Returns:
            np.array: An array of the counts for characters at a given position.
        """
        position = self.__convert_pos(pos=pos)
        full_column = self.__position_table[position, :]
        indices = np.nonzero(full_column)
        arr = full_column.toarray()[indices].reshape(-1)
        return arr

    # def get_count_matrix(self):
    #     """
    #     Get Count Matrix
    #
    #     Returns a matrix of counts where axis=0 represents characters from the alphabet and axis=1 represents positions
    #     in the alignment.
    #
    #     Returns:
    #         np.array: An nXm array where n is the length of the alphabet used by the alignment (plus the gap character)
    #         and m is the length of the sequences in the alignment. Each position in the matrix specifies the count of a
    #         character at a given position.
    #     """
    #     if len(self.__position_table) == 0:
    #         mat = None
    #     else:
    #         alpha_size, gap_chars, mapping = build_mapping(alphabet=self.alphabet)
    #         mat = np.zeros((len(self.__position_table), alpha_size))
    #         positions = self.get_positions()
    #         for i in range(len(positions)):
    #             pos = positions[i]
    #             for char in self.__position_table[pos]:
    #                 j = mapping[char]
    #                 mat[i, j] = self.get_count(pos=pos, char=char)
    #     return mat

    def get_count_matrix(self):
        """
        Get Count Matrix

        Returns a matrix of counts where axis=0 represents characters from the alphabet and axis=1 represents positions
        in the alignment.

        Returns:
            np.array: An nXm array where n is the length of the alphabet used by the alignment (plus the gap character)
            and m is the length of the sequences in the alignment. Each position in the matrix specifies the count of a
            character at a given position.
        """
        mat = self.__position_table.toarray()
        return mat

    # def compute_frequencies(self):
    #     """
    #     Compute Frequencies
    #
    #     This function uses the counts for each position and the depth tracked by the instance to compute frequencies for
    #     each character observed.
    #     """
    #     if not self.__frequencies:
    #         for pos in self.__position_table:
    #             for char in self.__position_table[pos]:
    #                 self.__position_table[pos][char]['frequency'] = (float(self.__position_table[pos][char]['count']) /
    #                                                                  self.__depth)
    #         self.__frequencies = True

    # def get_frequency(self, pos, char):
    #     """
    #     Get Frequency
    #
    #     Returns the frequency for a character at a specific position, if the character is not present at that position 0
    #     is returned.
    #
    #     Args:
    #         pos (int/tuple): A sequence position from the alignment.
    #         char (char/str): The character or string to look for the at specified position.
    #     Returns:
    #         float: The frequency of the specified character at the specified position.
    #     """
    #     if not self.__frequencies:
    #         raise RuntimeError('Frequencies have not been computed, please call compute_frequencies()')
    #     if (pos in self.__position_table) and (char in self.__position_table[pos]):
    #         freq = self.__position_table[pos][char]['frequency']
    #     else:
    #         freq = 0.0
    #     return freq

    def get_frequency(self, pos, char):
        """
        Get Frequency

        Returns the frequency for a character at a specific position, if the character is not present at that position 0
        is returned.

        Args:
            pos (int/tuple): A sequence position from the alignment.
            char (char/str): The character or string to look for the at specified position.
        Returns:
            float: The frequency of the specified character at the specified position.
        """
        count = self.get_count(pos=pos, char=char)
        freq = count / float(self.__depth)
        return freq

    # def get_frequency_array(self, pos):
    #     """
    #     Get Frequency Array
    #
    #     Returns an array containing the frequencies for all characters at a specified position, the order for the
    #     frequencies is the same as the order of the characters returned by get_chars().
    #
    #     Args:
    #         pos (int/tuple): A sequence position from the alignment.
    #     Returns:
    #         np.array: An array of the frequencies for characters at a given position.
    #     """
    #     if not self.__frequencies:
    #         raise RuntimeError('Frequencies have not been computed, please call compute_frequencies()')
    #     if pos in self.__position_table:
    #         arr = np.array([self.__position_table[pos][char]['frequency'] for char in self.get_chars(pos)],
    #                        dtype=np.dtype(float))
    #     else:
    #         arr = None
    #     return arr

    def get_frequency_array(self, pos):
        """
        Get Frequency Array

        Returns an array containing the frequencies for all characters at a specified position, the order for the
        frequencies is the same as the order of the characters returned by get_chars().

        Args:
            pos (int/tuple): A sequence position from the alignment.
        Returns:
            np.array: An array of the frequencies for characters at a given position.
        """
        counts = self.get_count_array(pos=pos)
        arr = counts / float(self.__depth)
        return arr

    # def get_frequency_matrix(self):
    #     """
    #     Get Frequency Matrix
    #
    #     Returns a matrix of frequencies where axis=0 represents characters from the alphabet and axis=1 represents
    #     positions in the alignment.
    #
    #     Returns:
    #         np.array: An nXm array where n is the length of the alphabet used by the alignment (plus the gap character)
    #         and m is the length of the sequences in the alignment. Each position in the matrix specifies the
    #         frequency of a character at a given position.
    #     """
    #     if not self.__frequencies:
    #         raise RuntimeError('Frequencies have not been computed, please call compute_frequencies()')
    #     if len(self.__position_table) == 0:
    #         mat = None
    #     else:
    #         alpha_size, gap_chars, mapping = build_mapping(alphabet=self.alphabet)
    #         mat = np.zeros((len(self.__position_table), alpha_size))
    #         positions = self.get_positions()
    #         for i in range(len(positions)):
    #             pos = positions[i]
    #             for char in self.__position_table[pos]:
    #                 j = mapping[char]
    #                 mat[i, j] = self.get_frequency(pos=pos, char=char)
    #     return mat

    def get_frequency_matrix(self):
        """
        Get Frequency Matrix

        Returns a matrix of frequencies where axis=0 represents characters from the alphabet and axis=1 represents
        positions in the alignment.

        Returns:
            np.array: An nXm array where n is the length of the alphabet used by the alignment (plus the gap character)
            and m is the length of the sequences in the alignment. Each position in the matrix specifies the
            frequency of a character at a given position.
        """
        count_matrix = self.get_count_matrix()
        mat = count_matrix / float(self.__depth)
        return mat

    # def to_csv(self, file_path):
    #     """
    #     To CSV
    #
    #     This method writes the current FrequencyTable instance to a tab delimited file capturing the variability,
    #     characters, counts, and frequencies for every position.
    #
    #     Args:
    #         file_path (str): The full path to where the data should be written.
    #     """
    #     start = time()
    #     if not os.path.isfile(file_path):
    #         columns = ['Position', 'Variability', 'Characters', 'Counts', 'Frequencies']
    #         out_dict = {c: [] for c in columns}
    #         for position in self.get_positions():
    #             out_dict['Position'].append(position)
    #             chars = self.get_chars(pos=position)
    #             out_dict['Variability'].append(len(chars))
    #             out_dict['Characters'].append(','.join(chars))
    #             counts = self.get_count_array(pos=position)
    #             out_dict['Counts'].append(','.join([str(x) for x in counts]))
    #             try:
    #                 freqs = self.get_frequency_array(pos=position)
    #             except RuntimeError:
    #                 freqs = counts / float(self.__depth)
    #             out_dict['Frequencies'].append(','.join([str(x) for x in freqs]))
    #         df = pd.DataFrame(out_dict)
    #         df.to_csv(file_path, sep='\t', columns=columns, header=True, index=False)
    #     end = time()
    #     print('Writing FrequencyTable to file took {} min'.format((end - start) / 60.0))

    def to_csv(self, file_path):
        """
        To CSV

        This method writes the current FrequencyTable instance to a tab delimited file capturing the variability,
        characters, counts, and frequencies for every position.

        Args:
            file_path (str): The full path to where the data should be written.
        """
        start = time()
        if not os.path.isfile(file_path):
            columns = ['Position', 'Variability', 'Characters', 'Counts', 'Frequencies']
            out_dict = {c: [] for c in columns}
            positions = self.get_positions()
            for i in range(self.num_pos):
                position = positions[i]
                out_dict['Position'].append(position)
                chars = self.get_chars(pos=position)
                out_dict['Variability'].append(len(chars))
                out_dict['Characters'].append(','.join(chars))
                counts = self.get_count_array(pos=position)
                out_dict['Counts'].append(','.join([str(int(x)) for x in counts]))
                freqs = self.get_frequency_array(pos=position)
                out_dict['Frequencies'].append(','.join([str(x) for x in freqs]))
            df = pd.DataFrame(out_dict)
            df.to_csv(file_path, sep='\t', columns=columns, header=True, index=False)
        end = time()
        print('Writing FrequencyTable to file took {} min'.format((end - start) / 60.0))

    # def load_csv(self, file_path):
    #     """
    #     Load CSV
    #
    #     This method uses a csv written by the to_csv method to populate the __position_table and __depth attributes of a
    #     FrequencyTable instance.
    #
    #     Args:
    #         file_path (str): The path to the file written by to_csv from which the FrequencyTable data should be loaded.
    #     """
    #     start = time()
    #     if not os.path.isfile(file_path):
    #         raise ValueError('The provided path does not exist.')
    #     header = None
    #     indices = None
    #     max_depth = None
    #     with open(file_path, 'rb') as file_handle:
    #         for line in file_handle:
    #             elements = line.strip().split('\t')
    #             if header is None:
    #                 header = elements
    #                 indices = {col: i for i, col in enumerate(header)}
    #                 continue
    #             pos_str = elements[indices['Position']]
    #             try:
    #                 pos = int(pos_str)
    #                 if pos > (self.sequence_length - 1):
    #                     raise RuntimeError('Imported file does not match sequence position {} exceeds sequence length'.format(self.sequence_length))
    #             except ValueError:
    #                 pos = tuple([int(x) for x in pos_str.lstrip('(').rstrip(')').split(',')])
    #                 if pos[0] > (self.sequence_length - 1) or pos[1] > (self.sequence_length - 1):
    #                     raise RuntimeError('Imported file does not match sequence position {} exceeds sequence length'.format(self.sequence_length))
    #             self.__position_table[pos] = {}
    #             chars = elements[indices['Characters']].split(',')
    #             counts = [int(x) for x in elements[indices['Counts']].split(',')]
    #             frequencies = [float(x) for x in elements[indices['Frequencies']].split(',')]
    #             if len(chars) != len(counts) or len(counts) != len(frequencies):
    #                 raise ValueError('Frequency Table written to file incorrectly the length of Characters, Counts, and Frequencies does not match for position: {}'.format(pos))
    #             for i in range(len(chars)):
    #                 self.__position_table[pos][chars[i]] = {'count': counts[i], 'frequency': frequencies[i]}
    #                 curr_depth = np.sum(counts)
    #                 if max_depth is None:
    #                     max_depth = curr_depth
    #                 else:
    #                     if curr_depth != max_depth:
    #                         raise RuntimeError('Depth at position {} does not match the depth from previous positions {} vs {}'.format(pos, max_depth, curr_depth))
    #     self.__depth = max_depth
    #     self.__frequencies = True
    #     end = time()
    #     print('Loading FrequencyTable from file took {} min'.format((end - start) / 60.0))

    def load_csv(self, file_path):
        """
        Load CSV

        This method uses a csv written by the to_csv method to populate the __position_table and __depth attributes of a
        FrequencyTable instance.

        Args:
            file_path (str): The path to the file written by to_csv from which the FrequencyTable data should be loaded.
        """
        start = time()
        if not os.path.isfile(file_path):
            raise ValueError('The provided path does not exist.')
        header = None
        indices = None
        max_depth = None
        with open(file_path, 'rb') as file_handle:
            for line in file_handle:
                elements = line.strip().split('\t')
                if header is None:
                    header = elements
                    indices = {col: i for i, col in enumerate(header)}
                    continue
                pos_str = elements[indices['Position']]
                try:
                    pos = int(pos_str)
                    if pos > (self.sequence_length - 1):
                        raise RuntimeError('Imported file does not match sequence position {} exceeds sequence length'.format(self.sequence_length))
                except ValueError:
                    pos = tuple([int(x) for x in pos_str.lstrip('(').rstrip(')').split(',')])
                    if pos[0] > (self.sequence_length - 1) or pos[1] > (self.sequence_length - 1):
                        raise RuntimeError('Imported file does not match sequence position {} exceeds sequence length'.format(self.sequence_length))
                position = self.__convert_pos(pos=pos)
                chars = elements[indices['Characters']].split(',')
                counts = [int(x) for x in elements[indices['Counts']].split(',')]
                if len(chars) != len(counts):
                    raise ValueError('Frequency Table written to file incorrectly the length of Characters, Counts, and Frequencies does not match for position: {}'.format(pos))
                for i in range(len(chars)):
                    char_pos = self.mapping[chars[i]]
                    self.__position_table[position, char_pos] = counts[i]
                    curr_depth = np.sum(counts)
                    if max_depth is None:
                        max_depth = curr_depth
                    else:
                        if curr_depth != max_depth:
                            raise RuntimeError('Depth at position {} does not match the depth from previous positions {} vs {}'.format(pos, max_depth, curr_depth))
        self.__depth = max_depth
        end = time()
        print('Loading FrequencyTable from file took {} min'.format((end - start) / 60.0))

    # def __add__(self, other):
    #     """
    #     Overloads the + operator, combining the information from two FrequencyTables. The intention of this behavior is
    #     that during the trace FrequencyTables can be joined as nodes in the phylogenetic tree are joined, such that more
    #     expensive calculations can be avoided. If the frequencies for either table have been calculated before, these
    #     are not included when combining the two tables.
    #
    #     Args:
    #         other (FrequencyTable): Another instance of the FrequencyTable class which should be combined with this one.
    #     Returns:
    #         FrequencyTable: A new instance of the FrequencyTable class with the combined data of the two provided
    #         instances.
    #     """
    #     if not isinstance(other, FrequencyTable):
    #         raise ValueError('FrequencyTable can only be combined with another FrequencyTable instance.')
    #     if self.position_size != other.position_size:
    #         raise ValueError('FrequencyTables must have the same position size to be joined.')
    #     if self.sequence_length != other.sequence_length:
    #         raise ValueError('FrequencyTables must have the same sequence length to be joined.')
    #     # Determine the alphabet from the two FrequencyTables
    #     merged_alpha = Alphabet._consensus_alphabet([self.alphabet, other.alphabet])
    #     # Copy current table as starting point
    #     merged_table = deepcopy(self.__position_table)
    #     # Add any positions/characters in the other table which were not in the current table, and combine any that were
    #     # in both
    #     for pos in other.get_positions():
    #         if pos not in merged_table:
    #             merged_table[pos] = other.__position_table[pos]
    #             continue
    #         for char in other.get_chars(pos=pos):
    #             if char not in merged_table[pos]:
    #                 merged_table[pos][char] = {'count': other.get_count(pos=pos, char=char)}
    #             else:
    #                 merged_table[pos][char]['count'] += other.get_count(pos=pos, char=char)
    #     # If frequencies had been computed, remove them from the new instance of the table
    #     for pos in merged_table:
    #         for char in merged_table[pos]:
    #             if 'frequency' in merged_table[pos][char]:
    #                 del (merged_table[pos][char]['frequency'])
    #     new_table = FrequencyTable(alphabet=merged_alpha, seq_len=self.sequence_length, pos_size=self.position_size)
    #     new_table.__position_table = merged_table
    #     axis = self.position_size - 1
    #     new_table.__depth = self.__depth + other.__depth
    #     return new_table

    def __add__(self, other):
        """
        Overloads the + operator, combining the information from two FrequencyTables. The intention of this behavior is
        that during the trace FrequencyTables can be joined as nodes in the phylogenetic tree are joined, such that more
        expensive calculations can be avoided. If the frequencies for either table have been calculated before, these
        are not included when combining the two tables.

        Args:
            other (FrequencyTable): Another instance of the FrequencyTable class which should be combined with this one.
        Returns:
            FrequencyTable: A new instance of the FrequencyTable class with the combined data of the two provided
            instances.
        """
        if not isinstance(other, FrequencyTable):
            raise ValueError('FrequencyTable can only be combined with another FrequencyTable instance.')
        if self.position_size != other.position_size:
            raise ValueError('FrequencyTables must have the same position size to be joined.')
        if self.num_pos != other.num_pos:
            raise ValueError('FrequencyTables must have the same number of positions to be joined')
        if self.sequence_length != other.sequence_length:
            raise ValueError('FrequencyTables must have the same sequence length to be joined.')
        if self.mapping != other.mapping:
            raise ValueError('FrequencyTables must have the same alphabet character mapping to be joined.')
        if self.reverse_mapping != other.reverse_mapping:
            raise ValueError('FrequencyTables must have the same alphabet character mapping to be joined.')
        new_table = FrequencyTable(alphabet_size=len(self.reverse_mapping), mapping=self.mapping, seq_len=self.sequence_length,
                                   pos_size=self.position_size)
        new_table.__position_table = self.__position_table + other.__position_table
        new_table.__depth = self.__depth + other.__depth
        return new_table
