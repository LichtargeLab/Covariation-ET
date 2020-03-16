"""
Created on May 23, 2019

@author: Daniel Konecki
"""
import os
import numpy as np
import pandas as pd
from time import time
from copy import deepcopy
from scipy.sparse import lil_matrix, csc_matrix


class FrequencyTable(object):
    """
    This class represents the position or pair specific nucleic or amino acid counts for a given alignment.

    Attributes:
        mapping (dict): A dictionary mapping the alphabet of the alignment being characterized to numerical positions.
        reverse_mapping (dict): A dictionary mapping positions back to alphabet characterized.
        position_size (int): How big a "position" is, i.e. if the frequency table measures single positions this should
        be 1, if it measures pairs of positions this should be 2, etc.
        num_pos (int): The number of positions being characterized.
        __position_table (dict/csc_matrix): A structure storing the position specific counts for nucleic/amino acids
        found in the alignment. This attribute begins as a dictionary holding all of the values needed to initialize a
        csc_matrix (including empty lists for values and i and j positions). When the seqeunce/alignment
        characterization has been concluded this can be converted to a csc_matrix using the finalize_table() method.
        __depth (int): The number of possible observations per positions (the normalization count for calculating
        frequency).
    """

    def __init__(self, alphabet_size, mapping, reverse_mapping, seq_len, pos_size=1):
        """
        Initialization for a FrequencyTable object.

        Args:
            alphabet_size (int): The number of characters in the alphabet.
            mapping (dict): A mapping from alphabet characters to numerical representations (can be generated by
            utils.build_mapping).
            reverse_mapping (dict): A mapping from numerical representations back to alphabet characters.
            seq_len (int): The length of the sequences in the alignment characterized by this FrequencyTable.
            pos_size (int): The size of a position in the alignment to be characterized (single positions = 1, pairs of
            positions = 2, etc.).
        """
        if len(list(mapping.keys())[0]) != pos_size:
            raise ValueError('Alphabet size must be equal to pos_size!')
        self.mapping = mapping
        # Issues with gap characters addressed by keeping only character mappings within alphabet_size.
        self.reverse_mapping = reverse_mapping
        self.position_size = pos_size
        self.sequence_length = seq_len
        if pos_size == 1:
            self.num_pos = seq_len
        elif pos_size == 2:
            self.num_pos = int(np.sum(range(seq_len + 1)))
        else:
            raise ValueError('FrequencyTable not implemented to handle pos_size: {}'.format(pos_size))
        # Elements of a csc_matrix: values, i, j, shape
        self.__position_table = {'values': [], 'i': [], 'j': [], 'shape': (self.num_pos, alphabet_size)}
        self.__depth = 0

    def __convert_pos(self, pos):
        """
        Convert Position

        This method takes a position (1 or 2 dimensional) and maps it to the 1 dimensional position on the
        __positional_table that it corresponds to.

        Args:
            pos (int/tuple): The position of interest in the alignment being characterized.
        Return:
            int: The 1 dimensional position that the passed in position maps to on the __positional_table.
        """
        if self.position_size == 1 and not isinstance(pos, (int, np.integer)):
            raise TypeError('Positions for FrequencyTable with position_size==1 must be integers')
        # if (self.position_size > 1) and not isinstance(pos, tuple) and (len(pos) != self.position_size):
        if (self.position_size > 1) and ((not isinstance(pos, tuple)) or (len(pos) != self.position_size)):
            raise TypeError('Positions for FrequencyTable with position_size > 1 must have length == position_size')
        if self.position_size == 1:
            final = pos
        elif self.position_size == 2:
            i_factor = np.sum([self.sequence_length - x for x in range(pos[0])])
            j_factor = pos[1] - pos[0]
            final = i_factor + j_factor
        else:
            raise ValueError('Position conversion not implemented for position sizes other than 1 or 2.')
        return int(final)

    def _increment_count(self, pos, char, amount=1):
        """
        Increment Count

        This method updates the position and character by the specified amount, keeping track of the occurrence of of
        alphabet characters at each position in an alignment.

        Args:
            pos (int/tuple): The position in the alignment to update (will be mapped to __positional_table by
            __convert_pos.
            char (str): The character in the alignment's alphabet to update at the specified position.
            amount (int): The number of occurrences of the alphabet character observed at that specified position.
        """
        position = self.__convert_pos(pos=pos)
        char_pos = self.mapping[char]
        self.__position_table['values'].append(amount)
        self.__position_table['i'].append(position)
        self.__position_table['j'].append(char_pos)

    def characterize_alignment(self, num_aln, single_to_pair=None):
        """
        Characterize Alignment

        Characterize an entire alignment. This iterates over all positions (single positions in the the
        alignment if position_size==1 and pairs of positions in the alignment if position_size==2) and updates the
        character count found at that position in the sequence in __positional_table. At the end of this call __depth is
        update to the size of the alignment.

        Args:
            num_aln (np.array): Array representing an alignment with dimensions sequence_length by alignment size where
            the values are integers representing nucleic/amino acids and gaps from the desired alignment.
            single_to_pair (dict): A dictionary mapping tuples of integers to a single int. The tuple of integers should
            consist of the position of the first character in a pair of letters to its numerical position and the
            position of the second character in a pair of letters to its numerical position. The value that this tuple
            maps to should be the integer value that a pair of letters maps to.
        """
        if self.position_size == 2 and single_to_pair is None:
            raise ValueError('Mapping from single to pair letter alphabet must be provided if position_size == 2')
        # Iterate over all positions
        for i in range(self.sequence_length):
            # If single is specified, track the amino acid for this sequence and position
            if self.position_size == 1:
                # Find each unique character in the column and its count in that column
                char_pos, counts = np.unique(num_aln[:, i], axis=0, return_counts=True)
                # Update the observed characters with their counts
                # self.__position_table[i, char_pos.reshape(-1)] = counts
                self.__position_table['values'] += list(counts)
                self.__position_table['i'] += [i] * len(counts)
                self.__position_table['j'] += list(char_pos.reshape(-1))
            # If pair is not specified continue to the next position
            if self.position_size != 2:
                continue
            # If pair is specified iterate over all positions up to the current one (filling in upper triangle,
            # including the diagonal)
            for j in range(i, self.sequence_length):
                # Track the pair of amino acids for the positions i,j
                position = self.__convert_pos(pos=(i, j))
                # Find each unique pair of characters in two columns and their count
                char_pos, counts = np.unique(num_aln[:, [i, j]], axis=0, return_counts=True)
                # Map the individual character alphabet observations to the pair alphabet positions
                char_pos = [single_to_pair[tuple(pos)] for pos in char_pos]
                # Update the observed pairs of characters for the pair of columns using the counts.
                self.__position_table['values'] += list(counts)
                self.__position_table['i'] += [position] * len(counts)
                self.__position_table['j'] += list(char_pos)
        # Update the depth to the number of sequences in the characterized alignment
        self.__depth = num_aln.shape[0]
        self.finalize_table()

    def characterize_sequence(self, seq):
        """
        Characterize Sequence

        Characterize a single sequence from an alignment. This iterates over all positions (single positions in the the
        alignment if position_size==1 and pairs of positions in the alignment if position_size==2) and updates the
        character count found at that position in the sequence in __positional_table. Each call to this function updates
        __depth by 1.

        Args:
            seq (Bio.Seq.Seq): The sequence from an alignment to characterize.
        """
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
        """
        Finalize Table

        When all sequences from an alignment have been characterized the table is saved from a scipy.sparse.lil_matrix
        to a scipy.sparse.csc_matrix (since the table will most often be accessed by column). This also ensures proper
        behavior from other functions such as get_count_array() and get_frequency_array().
        """
        self.__position_table = csc_matrix((self.__position_table['values'],
                                            (self.__position_table['i'], self.__position_table['j'])),
                                           shape=self.__position_table['shape'])

    def set_depth(self, depth):
        """
        Set Depth

        This function is intended to update the depth attribute if the existing methods do not suffice.

        Arguments
            depth (int): The number of observations for all positions (normalization factor when turning count into
            frequency).
        """
        self.__depth = deepcopy(depth)

    def get_table(self):
        """
        Get Table

        Returns the matrix storing the position specific counts for the characters present in the alignment.

        Returns:
            scipy.sparse.lil_matrix/csc_matrix: A sparse matrix where one axis represents positions in the MSA
            and the other axis represents the characters from the alphabet of interest mapping. Each cell stores the
            count of that character at that position.
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

    def get_count(self, pos, char):
        """
        Get Count

        Returns the count for a character at a specific position, if the character is not present at that position 0 is
        returned.

        Args:
            pos (int/tuple): A sequence position from the alignment.
            char (char/str): The character or string (when position_size > 1) to look for at the specified position.
        Returns:
            int: The count of the specified character at the specified position.
        """
        position = self.__convert_pos(pos=pos)
        char_pos = self.mapping[char]
        count = self.__position_table[position, char_pos]
        return count

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

    def to_csv(self, file_path):
        """
        To CSV

        This method writes the current FrequencyTable instance to a tab delimited file capturing the variability,
        characters, counts, and frequencies for every position.

        Args:
            file_path (str): The full path to where the data should be written.
        """
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
        with open(file_path, 'r') as file_handle:
            for line in file_handle:
                elements = line.strip().split('\t')
                if header is None:
                    header = elements
                    indices = {col: i for i, col in enumerate(header)}
                    continue
                pos_str = elements[indices['Position']]
                try:
                    pos = int(pos_str)
                    # if pos > (self.sequence_length - 1):
                    if pos > (self.num_pos - 1):
                        raise RuntimeError('Imported file does not match sequence position {} exceeds sequence '
                                           'length'.format(self.sequence_length))
                    position = pos
                except ValueError:
                    pos = tuple([int(x) for x in pos_str.lstrip('(').rstrip(')').split(',')])
                    if pos[0] > (self.sequence_length - 1) or pos[1] > (self.sequence_length - 1):
                        raise RuntimeError('Imported file does not match sequence position {} exceeds sequence '
                                           'length'.format(self.sequence_length))
                    position = self.__convert_pos(pos=pos)
                # position = self.__convert_pos(pos=pos)
                chars = elements[indices['Characters']].split(',')
                counts = [int(x) for x in elements[indices['Counts']].split(',')]
                if len(chars) != len(counts):
                    raise ValueError('Frequency Table written to file incorrectly the length of Characters, Counts, and'
                                     ' Frequencies does not match for position: {}'.format(pos))
                for i in range(len(chars)):
                    try:
                        char_pos = self.mapping[chars[i]]
                    except KeyError as e:
                        print(self.mapping)
                        raise e
                    # self.__position_table[position, char_pos] = counts[i]
                    self.__position_table['values'].append(counts[i])
                    self.__position_table['i'].append(position)
                    self.__position_table['j'].append(char_pos)
                    curr_depth = np.sum(counts)
                    if max_depth is None:
                        max_depth = curr_depth
                    else:
                        if curr_depth != max_depth:
                            raise RuntimeError('Depth at position {} does not match the depth from previous positions '
                                               '{} vs {}'.format(pos, max_depth, curr_depth))
        self.finalize_table()
        self.__depth = max_depth
        end = time()
        print('Loading FrequencyTable from file took {} min'.format((end - start) / 60.0))

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
        if isinstance(self.__position_table, dict) or isinstance(other.__position_table,dict):
            raise AttributeError('Before combining FrequencyTable objects please call finalize_table().')
        dummy_dict = {('{0}' * self.position_size).format('A'): 0}
        new_table = FrequencyTable(alphabet_size=len(self.reverse_mapping), mapping=dummy_dict,
                                   reverse_mapping=self.reverse_mapping, seq_len=self.sequence_length,
                                   pos_size=self.position_size)
        new_table.mapping = self.mapping
        new_table.__position_table = self.__position_table + other.__position_table
        new_table.__depth = self.__depth + other.__depth
        return new_table
