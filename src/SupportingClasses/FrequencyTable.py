"""
Created on May 23, 2019

@author: Daniel Konecki
"""
import os
import numpy as np
import pandas as pd
from time import time
from copy import deepcopy
from scipy.sparse import csc_matrix


class FrequencyTable(object):
    """
    This class represents the position or pair specific nucleic or amino acid counts for a given alignment.

    Attributes:
        mapping (dict): A dictionary mapping the alphabet of the alignment being characterized to numerical positions.
        reverse_mapping (np.array): An array mapping positions back to alphabet characterized.
        position_size (int): How big a "position" is, i.e. if the frequency table measures single positions this should
        be 1, if it measures pairs of positions this should be 2, etc.
        num_pos (int): The number of positions being characterized.
        __position_table (dict/csc_matrix): A structure storing the position specific counts for nucleic/amino acids
        found in the alignment. This attribute begins as a dictionary holding all of the values needed to initialize a
        csc_matrix (including empty lists for values and i and j positions). When the sequence/alignment
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
        check_char = list(mapping.keys())[0]
        if reverse_mapping[mapping[check_char]] != check_char:
            raise ValueError('Mapping and reverse mapping do not agree!')
        if len(mapping) < alphabet_size or len(reverse_mapping) != alphabet_size:
            raise ValueError('Mapping ({}) and reverse mapping ({}) must match alphabet size ({})!'.format(
                len(mapping), len(reverse_mapping), alphabet_size))
        if (len(check_char) % pos_size) != 0:
            raise ValueError('Alphabet size must be equal to, or a multiple of, pos_size!')
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

    def _convert_pos(self, pos):
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
        if (self.position_size > 1) and ((not isinstance(pos, tuple)) or (len(pos) != self.position_size)):
            raise TypeError('Positions for FrequencyTable with position_size > 1 must have length == position_size')
        if (self.position_size == 1 and pos < 0) or (self.position_size > 1 and (np.array(pos) < 0).any()):
            raise ValueError('Position specified is out of bounds, the value(s) are less than 0.')
        elif ((self.position_size == 1 and pos >= self.num_pos) or
              (self.position_size > 1 and (np.array(pos) >= self.sequence_length).any())):
            raise ValueError('Position specified is out of bounds, the value(s) are greater than the table size.')
        else:
            pass
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

        This method updates the position and character by the specified amount, keeping track of the occurrence of
        alphabet characters at each position in an alignment.

        Args:
            pos (int/tuple): The position in the alignment to update (will be mapped to __positional_table by
            _convert_pos.
            char (str): The character in the alignment's alphabet to update at the specified position.
            amount (int): The number of occurrences of the alphabet character observed at that specified position.
        """
        if isinstance(self.__position_table, csc_matrix):
            raise AttributeError('FrequencyTable has already been finalized and cannot be updated.')
        if amount <= 0:
            raise ValueError('Amounts passed to increment must be positive values.')
        position = self._convert_pos(pos=pos)
        char_pos = self.mapping[char]
        self.__position_table['values'].append(amount)
        self.__position_table['i'].append(position)
        self.__position_table['j'].append(char_pos)

    def finalize_table(self):
        """
        Finalize Table

        When all sequences from an alignment have been characterized the table is saved from a dictionary of values
        to a scipy.sparse.csc_matrix (since the table will most often be accessed by column). This also ensures proper
        behavior from other functions such as get_count_array() and get_frequency_array().
        """
        self.__position_table = csc_matrix((self.__position_table['values'],
                                            (self.__position_table['i'], self.__position_table['j'])),
                                           shape=self.__position_table['shape'], dtype=np.int32)

    def set_depth(self, depth):
        """
        Set Depth

        This function is intended to update the depth attribute if the existing methods do not suffice (i.e. if
        characterize_alignment or characterize_sequence are not used.).

        Arguments
            depth (int): The number of observations for all positions (normalization factor when turning count into
            frequency).
        """
        if depth is None:
            raise ValueError('Depth cannot be None, please provide a value >= 0.')
        if depth < 0:
            raise ValueError('Depth cannot be negative, please provide a value >= 0.')
        self.__depth = deepcopy(depth)

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
        if seq is None:
            raise ValueError('seq must not be None.')
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
            single_to_pair (np.array): An array mapping single letter numerical representations (axes 0 and 1) to a
            numerical representations of pairs of residues (value).
        """
        if num_aln is None:
            raise ValueError('Numeric representation of an alignment must be provided as input.')
        if self.position_size == 2 and single_to_pair is None:
            raise ValueError('Mapping from single to pair letter alphabet must be provided if position_size == 2')
        # Iterate over all positions
        for i in range(self.sequence_length):
            # If single is specified, track the amino acid for this sequence and position
            if self.position_size == 1:
                # Add all characters observed at this position to the frequency table (this is inefficient in terms of
                # space but reduces the time required to identify and count individual characters.
                self.__position_table['values'] += [1] * num_aln.shape[0]
                self.__position_table['i'] += [i] * num_aln.shape[0]
                self.__position_table['j'] += list(num_aln[:, i])
            # If pair is not specified continue to the next position
            if self.position_size != 2:
                continue
            # If pair is specified iterate over all positions up to the current one (filling in upper triangle,
            # including the diagonal)
            for j in range(i, self.sequence_length):
                # Track the pair of amino acids for the positions i,j
                position = self._convert_pos(pos=(i, j))
                # Add all characters observed at this position to the frequency table (this is inefficient in terms of
                # space but reduces the time required to identify and count individual characters.
                self.__position_table['values'] += [1] * num_aln.shape[0]
                self.__position_table['i'] += [position] * num_aln.shape[0]
                self.__position_table['j'] += list(single_to_pair[num_aln[:, i], num_aln[:, j]])
        # Update the depth to the number of sequences in the characterized alignment
        self.__depth = num_aln.shape[0]
        self.finalize_table()

    def characterize_alignment_mm(self, num_aln, comparison, mismatch_mask, single_to_pair=None, indexes1=None,
                                  indexes2=None):
        """
        Characterize Alignment

        Characterize an entire alignment for matches (invariance or covariation) and mismatches (variation). This
        iterates over all positions (single positions in the the alignment if position_size==1 and pairs of positions in
        the alignment if position_size==2) and tracks the character count found at that position. When character counts
        have been identified all observed transitions (comparisons between one sequence and all other sequences) are
        determined and the count for each unique type of transition is stored in __position_table. The transition counts
        are split into two FrequencyTable objects, the matches (invariance and covariation counts) are kept in the table
        from which characterize_alignment_mm was called, while a new FrequencyTable object is created for the mismatch
        (variance) transitions. Both FrequencyTable objects are finalized (see finalize_table) and the __depth for both
        FrequencyTable objects is set to the size for the upper triangle of sequence comparisons
        (i.e. (num_aln.shape[0] * (num_aln.shape[0] - 1)) / 2 or 1 if there is only one sequence in the alignment).

        Args:
            num_aln (np.array): Array representing an alignment with dimensions sequence_length by alignment size where
            the values are integers representing nucleic/amino acids and gaps from the desired alignment.
            single_to_pair (np.array, dtype=np.int32): An array mapping single letter numerical representations
            (axes 0 and 1) to a numerical representations of pairs of residues (value). If position_size == 1 this
            argument should be set to None (default).
            comparison (np.array, dtype=np.int32): An array mapping the alphabet used for the positions_size of this
            FrequencyTable, mapped to the alphabet that has characters twice as large (if position_size == 1, this is
            the same as the description for single_to_pair, if position_size == 2 this is the mapping from the alphabet
            of pairs to the alphabet of quadruples.
            mismatch_mask (np.array, dtype=np.bool_): An array identifying which positions in the alphabet (the values
            in the provided alphabet mapping) correspond to mismatches (variance events). This will be used to separate
            the counts into match and mismatch tables.
            indexes1 (np.array, dtype=np.int32): If only a partial comparison is needed the indexes for the rectangle
            being characterized can be provided. indexes1 should have the the indexes for the sequences of interest for
            one side of the comparison rectangle in sorted order. It should have a lower minimum than indexes2.
            indexes2 (np.array, dtype=np.int32): If only a partial comparison is needed the indexes for the rectangle
            being characterized can be provided. indexes2 should have the the indexes for the sequences of interest for
            one side of the comparison rectangle in sorted order. It should have a higher minimum than indexes2.
        Return:
            FrequencyTable: A new FrequencyTable containing counts for all variance transitions observed in the provided
            alignment. The FrequencyTable matches the current FrequencyTable for all properties except the
            __position_table.
        """
        if num_aln is None:
            raise ValueError('Numeric representation of an alignment must be provided as input.')
        if comparison is None:
            raise ValueError('Mapping from single characters to transition/comparison characters is required.')
        if mismatch_mask is None:
            raise ValueError('An array indicating which characters are mismatch comparisons/transitions is required.')
        if self.position_size == 2 and single_to_pair is None:
            raise ValueError('Mapping from single to pair letter alphabet must be provided if position_size == 2')
        if (indexes1 is not None) and (indexes2 is None):
            raise ValueError('If indexes1 is provided indexes2 must also be provided.')
        if (indexes2 is not None) and (indexes1 is None):
            raise ValueError('If indexes2 is provided indexes1 must also be provided.')
        if (indexes1 is not None) and (indexes2 is not None):
            if np.min(indexes2) < np.min(indexes1):
                raise ValueError('indexes1 must come before indexes2.')
        # Iterate over all positions
        for i, pos in enumerate(self.get_positions()):
            # If single is specified, track the amino acid for this sequence and position
            if self.position_size == 1:
                curr_pos = num_aln[:, i]
            elif self.position_size == 2:
                curr_pos = single_to_pair[num_aln[:, pos[0]], num_aln[:, pos[1]]]
            else:
                raise ValueError(f'characterize_alignment_mm is not compatible with position_size {self.position_size}')
            # Characterize the transitions/comparisons from one sequence to each other sequence (unique).
            full_pos = []
            if indexes1 is None:
                indexes1 = range(num_aln.shape[0] - 1)
            for j in indexes1:
                if indexes2 is None:
                    comp_pos = curr_pos[j + 1:]
                else:
                    comp_pos = curr_pos[indexes2]
                curr_comp = comparison[curr_pos[j], comp_pos]
                full_pos += curr_comp.tolist()
            unique_chars, unique_counts = np.unique(full_pos, return_counts=True)
            self.__position_table['values'] += unique_counts.tolist()
            self.__position_table['i'] += [i] * unique_chars.shape[0]
            self.__position_table['j'] += unique_chars.tolist()
        # Update the depth to the number of sequences in the characterized alignment
        # The depth needs to be the number of possible matches mismatches when comparing all elements of a column or
        # pair of columns to one another (i.e. the upper triangle of the column vs. column matrix). For the smallest
        # sub-alignments the size of the sub-alignment is 1 and therefore there are no values in the upper triangle
        # (because the matrix of sequence comparisons is only one element large, which is also the diagonal of that
        # matrix and therefore not counted). Setting the depth to 0 causes divide by zero issues when calculating
        # frequencies so the depth is being arbitrarily set to 1 here. This is incorrect, but all counts should be 0 so
        # the resulting frequencies should be calculated correctly.
        self.__depth = 1 if num_aln.shape[0] == 1 else (num_aln.shape[0] * (num_aln.shape[0] - 1)) / 2
        self.finalize_table()
        # Split the tables in two, keep matches (invariant or covarying transitions/comparisons in the current table),
        # create a new table for mismatches (variant transitions/comparisons).
        mismatch_ft = FrequencyTable(alphabet_size=self.__position_table.shape[1], mapping=self.mapping,
                                     reverse_mapping=self.reverse_mapping, seq_len=self.sequence_length,
                                     pos_size=self.position_size)
        mismatch_ft.set_depth(depth=self.__depth)
        mismatch_ft.finalize_table()
        mismatch_ft.__position_table = csc_matrix(self.__position_table.multiply(mismatch_mask), dtype=np.int32)
        match_mask = np.ones(mismatch_mask.shape, dtype=np.bool_) ^ mismatch_mask
        self.__position_table = csc_matrix(self.__position_table.multiply(match_mask), dtype=np.int32)
        return mismatch_ft

    def get_table(self):
        """
        Get Table

        Returns the matrix storing the position specific counts for the characters present in the alignment.

        Returns:
            dict/scipy.sparse.csc_matrix: A dictionary containing the values needed to populate a
            scipy.sparse.csc_matrix. If the table has already been finazed, a sparse matrix where one axis represents
            positions in the MSA and the other axis represents the characters from the alphabet of interest mapping.
            Each cell stores the count of that character at that position.
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
            list: The positions tracked in this frequency table. If the position size is 1, then the list contains
            integers describing each position, if it is 2, the list contains tuples describing pairs of positions.
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
        if not isinstance(self.__position_table, csc_matrix):
            raise AttributeError('Finalize table before calling get_count.')
        if self.__depth == 0:
            raise AttributeError(f'No updates have been made to the FrequencyTable, depth: {self.__depth}')
        position = self._convert_pos(pos=pos)
        character_positions = self.__position_table[position, :].nonzero()
        characters = self.reverse_mapping[character_positions[1]]
        return list(characters)

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
        if not isinstance(self.__position_table, csc_matrix):
            raise AttributeError('Finalize table before calling get_count.')
        if self.__depth == 0:
            raise AttributeError(f'No updates have been made to the FrequencyTable, depth: {self.__depth}')
        position = self._convert_pos(pos=pos)
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
        if not isinstance(self.__position_table, csc_matrix):
            raise AttributeError('Finalize table before calling get_count_array.')
        if self.__depth == 0:
            raise AttributeError(f'No updates have been made to the FrequencyTable, depth: {self.__depth}')
        position = self._convert_pos(pos=pos)
        full_column = self.__position_table[position, :]
        indices = full_column.nonzero()
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
        if not isinstance(self.__position_table, csc_matrix):
            raise AttributeError('Finalize table before calling get_count_matrix.')
        if self.__depth == 0:
            raise AttributeError(f'No updates have been made to the FrequencyTable, depth: {self.__depth}')
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
        if not isinstance(self.__position_table, csc_matrix):
            raise AttributeError('Finalize table before calling get_count_matrix.')
        if self.__depth == 0:
            raise AttributeError(f'No updates have been made to the FrequencyTable, depth: {self.__depth}')
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

    def load_csv(self, file_path, intended_depth=None):
        """
        Load CSV

        This method uses a csv written by the to_csv method to populate the __position_table and __depth attributes of a
        FrequencyTable instance.

        Args:
            file_path (str): The path to the file written by to_csv from which the FrequencyTable data should be loaded.
            intended_depth (int): The depth of the table being imported. This can be used for tables where the counts of
            each position may not be equal (as in the match/mismatch tables which split observations at each position
            across two tables). Not required for tables where observations for each position have a consistent count.
        """
        if isinstance(self.__position_table, csc_matrix):
            raise AttributeError('The table has already been finalized, loading data would overwrite the table.')
        if self.__depth != 0:
            raise AttributeError(f'The FrequencyTable has already been updated, depth: {self.__depth}, '
                                 'loading would overwrite the table.')
        start = time()
        if not os.path.isfile(file_path):
            raise ValueError('The provided path does not exist.')
        header = None
        indices = None
        max_depth = None
        with open(file_path, 'r') as file_handle:
            for line in file_handle:
                elements = line.rstrip('\n').split('\t')
                if header is None:
                    header = elements
                    indices = {col: i for i, col in enumerate(header)}
                    continue
                pos_str = elements[indices['Position']]
                try:
                    pos = int(pos_str)
                    if pos > (self.num_pos - 1):
                        raise RuntimeError('Imported file does not match sequence position {} exceeds sequence '
                                           'length'.format(self.sequence_length))
                    position = pos
                except ValueError:
                    pos = tuple([int(x) for x in pos_str.lstrip('(').rstrip(')').split(',')])
                    if pos[0] > (self.sequence_length - 1) or pos[1] > (self.sequence_length - 1):
                        raise RuntimeError('Imported file does not match sequence position {} exceeds sequence '
                                           'length'.format(self.sequence_length))
                    position = self._convert_pos(pos=pos)
                chars = [] if elements[indices['Characters']] == '' else elements[indices['Characters']].split(',')
                counts = [] if elements[indices['Counts']] == '' else [int(x) for x in elements[indices['Counts']].split(',')]
                if len(chars) != len(counts):
                    raise ValueError('Frequency Table written to file incorrectly the length of Characters, Counts, and'
                                     ' Frequencies does not match for position: {}'.format(pos))
                for i in range(len(chars)):
                    try:
                        char_pos = self.mapping[chars[i]]
                    except KeyError as e:
                        print(self.mapping)
                        raise e
                    self.__position_table['values'].append(counts[i])
                    self.__position_table['i'].append(position)
                    self.__position_table['j'].append(char_pos)
                    curr_depth = np.sum(counts)
                    if (max_depth is None) and (intended_depth is None):
                        max_depth = curr_depth
                    if (max_depth is None) and (intended_depth is not None):
                        pass
                    else:
                        if curr_depth != max_depth:
                            raise RuntimeError('Depth at position {} does not match the depth from previous positions '
                                               '{} vs {}'.format(pos, max_depth, curr_depth))
        if (max_depth is None) and (intended_depth is None):
            max_depth = 1
        elif (max_depth is None) and (intended_depth is not None):
            max_depth = intended_depth
        else:
            assert max_depth is not None
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
        if not (self.reverse_mapping == other.reverse_mapping).all():
            raise ValueError('FrequencyTables must have the same alphabet character mapping to be joined.')
        if isinstance(self.__position_table, dict) or isinstance(other.__position_table, dict):
            raise AttributeError('Before combining FrequencyTable objects please call finalize_table().')
        if (self.__depth == 0) or (other.__depth == 0):
            raise AttributeError('At least one of the FrequencyTables being combined has not been updated.')
        new_table = FrequencyTable(alphabet_size=len(self.reverse_mapping), mapping=self.mapping,
                                   reverse_mapping=self.reverse_mapping, seq_len=self.sequence_length,
                                   pos_size=self.position_size)
        new_table.mapping = self.mapping
        new_table.__position_table = self.__position_table + other.__position_table
        new_table.__depth = self.__depth + other.__depth
        return new_table
