"""
Created on May 23, 2019

@author: Daniel Konecki
"""
import numpy as np
from copy import deepcopy
from Bio import Alphabet
from utils import build_mapping


class FrequencyTable(object):
    """
    This class represents the position or pair specific nucleic or amino acid counts for a given alignment.

    Attributes:
        alphabet (Bio.Alphabet.Alphabet/Bio.Alphabet.Gapped): The un/gapped alphabet for which this frequency table is
        valid.
        position_size (int): How big a "position" is, i.e. if the frequency table measures single positions this should
        be 1, if it measures pairs of positions this should be 2, etc.
        __position_table (dict): A structure storing the position specific counts for amino acids found in the
        alignment.
        frequencies (bool): Whether or not the frequencies for this table have been computed yet or not.
    """

    def __init__(self, alphabet, pos_size=1):
        """
        Initialization for a FrequencyTable object.

        Args:
            alphabet (Bio.Alphabet.Alphabet): The alphabet for which the frequency counts tracked by this table are
            valid.
        """
        if alphabet.size != pos_size:
            raise ValueError('Alphabet size must be equal to pos_size!')
        self.alphabet = alphabet
        self.position_size = pos_size
        self.__position_table = {}
        self.frequencies = False

    def _add_position(self, pos):
        """
        Add Position

        This function adds a given position to the __position_table.

        Args
            pos (int/tuple): A sequence position from the alignment.
        """
        if pos not in self.__position_table:
            if self.position_size == 1:
                if not isinstance(pos, int):
                    raise TypeError('Position does not match size specification: {} X {}'.format(self.position_size,
                                                                                                 type(pos)))
            elif self.position_size > 1:
                if not isinstance(pos, tuple) or len(pos) != self.position_size:
                    raise TypeError('Position does not match size specification: {} X {}'.format(self.position_size,
                                                                                                 type(pos)))
            else:
                pass
            self.__position_table[pos] = {}

    def _add_pos_char(self, pos, char):
        """
        Add Pos Char

        Add a character to the table for a specific position.

        Args:
            pos (int/tuple): A sequence position from the alignment.
            char (char/str): The character or string to add for the specified position.
        """
        if char not in self.alphabet.letters:
            raise ValueError('The character {} is not in the specified alphabet: {}'.format(char,
                                                                                            self.alphabet.letters))
        self._add_position(pos)
        if char not in self.__position_table[pos]:
            self.__position_table[pos][char] = {'count' : 0}

    def increment_count(self, pos, char):
        """
        Increment Count

        This function increments the count of a character or string for the specified position. It also updates the
        frequencies attribute so that stale frequencies cannot be accidentally used.

        Args:
            pos (int/tuple): A sequence position from the alignment.
            char (char/str): The character or string to add for the specified position.
        """
        self._add_pos_char(pos, char)
        self.__position_table[pos][char]['count'] += 1
        self.frequencies = False

    def get_table(self):
        """
        Get Table

        Returns the dictionary storing the position specific counts for the characters present in the alignment.

        Returns:
            dict: A nested dictionary where the first level describes positions in the MSA and maps to a second set of
            dictionaries where the key is the character from the alphabet of interest mapping to the count of that
            character at that position.
        """
        return deepcopy(self.__position_table)

    def get_positions(self):
        """
        Get Positions

        Provides the positions tracked in this frequency table.

        Returns:
            list: The positions tracked in this frequency table.
        """
        return list(sorted(self.__position_table.keys()))

    def get_chars(self, pos):
        """
        Get Chars

        Returns the characters from the alphabet of the MSA (and gaps) present at a given position.

        Args:
            pos (int/tuple): A sequence position from the alignment.
        Returns:
            list: All characters present at the specified position in the alignment.
        """
        return list(self.__position_table[pos].keys())

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
        if (pos in self.__position_table) and (char in self.__position_table[pos]):
            return self.__position_table[pos][char]['count']
        else:
            return 0

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
        if pos in self.__position_table:
            return np.array([self.__position_table[pos][char]['count'] for char in self.get_chars(pos)],
                            dtype=np.dtype(int))
        else:
            return None

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
        if len(self.__position_table) == 0:
            return None
        alpha_size, gap_chars, mapping = build_mapping(alphabet=self.alphabet)
        mat = np.zeros((len(self.__position_table), alpha_size))
        positions = self.get_positions()
        for i in range(len(positions)):
            pos = positions[i]
            for char in self.__position_table[pos]:
                j = mapping[char]
                mat[i, j] = self.get_count(pos=pos, char=char)
        return mat

    def _find_normalization(self):
        """
        Find Normalization

        This function looks for the largest number of sequences characterized for any given position by computing the
        cumulative sum over all characters observed at each position.

        Returns:
            int: The greatest number of observations made for any position in the alignment.
        """
        count_mat = self.get_count_matrix()
        if count_mat is None:
            return 0
        cumulative_counts = np.sum(count_mat, axis=0)
        max_count = int(np.max(cumulative_counts))
        return max_count

    def compute_frequencies(self, normalization=None):
        """
        Compute Frequencies

        This function uses the counts for each position to compute frequencies for each character observed. If a
        normalization value is provided it will be used as the denominator of the frequency calculation. If no
        normalization value is provided the _find_normalization will be called which will determine the max number of
        observations for any column and will be used as the denominator.

        Args:
            normalization (int): The size of the alignment which is characterized by this FrequencyTable. If None,
            _find_normalization will be used to determine the normalization value.
        """
        if normalization is None:
            normalization = self._find_normalization()
        for pos in self.__position_table:
            for char in self.__position_table[pos]:
                self.__position_table[pos][char]['frequency'] = (float(self.__position_table[pos][char]['count']) /
                                                                 normalization)
        self.frequencies = True

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
        if not self.frequencies:
            raise RuntimeError('Frequencies have not been computed, please call compute_frequencies()')
        if (pos in self.__position_table) and (char in self.__position_table[pos]):
            return self.__position_table[pos][char]['frequency']
        else:
            return 0.0

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
        if not self.frequencies:
            raise RuntimeError('Frequencies have not been computed, please call compute_frequencies()')
        if pos in self.__position_table:
            return np.array([self.__position_table[pos][char]['frequency'] for char in self.get_chars(pos)],
                            dtype=np.dtype(float))
        else:
            return None

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
        if not self.frequencies:
            raise RuntimeError('Frequencies have not been computed, please call compute_frequencies()')
        if len(self.__position_table) == 0:
            return None
        alpha_size, gap_chars, mapping = build_mapping(alphabet=self.alphabet)
        mat = np.zeros((len(self.__position_table), alpha_size))
        positions = self.get_positions()
        for i in range(len(positions)):
            pos = positions[i]
            for char in self.__position_table[pos]:
                j = mapping[char]
                mat[i, j] = self.get_frequency(pos=pos, char=char)
        return mat

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
        # Determine the alphabet from the two FrequencyTables
        merged_alpha = Alphabet._consensus_alphabet([self.alphabet, other.alphabet])
        # Copy current table as starting point
        merged_table = deepcopy(self.__position_table)
        # Add any positions/characters in the other table which were not in the current table, and combine any that were
        # in both
        for pos in other.get_positions():
            if pos not in merged_table:
                merged_table[pos] = other.__position_table[pos]
                continue
            for char in other.get_chars(pos=pos):
                if char not in merged_table[pos]:
                    merged_table[pos][char] = {'count': other.get_count(pos=pos, char=char)}
                else:
                    merged_table[pos][char]['count'] += other.get_count(pos=pos, char=char)
        # If frequencies had been computed, remove them from the new instance of the table
        for pos in merged_table:
            for char in merged_table[pos]:
                if 'frequency' in merged_table[pos][char]:
                    del (merged_table[pos][char]['frequency'])
        new_table = FrequencyTable(alphabet=merged_alpha, pos_size=self.position_size)
        new_table.__position_table = merged_table
        return new_table
