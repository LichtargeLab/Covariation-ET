"""
Created on Aug 17, 2017

@author: daniel
"""
import os
import re
import pickle
import numpy as np
import pandas as pd
from pymol import cmd
from time import time, sleep
from urllib.error import HTTPError
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex, hex2color, LinearSegmentedColormap
from Bio import Entrez
from Bio.SeqIO import parse, read
# This tool no longer works correctly so I have written my own regular expression based parser for amino acid sequences
# from Swiss/Uniprot.
# from Bio.SwissProt import read as sp_read
from Bio.ExPASy import get_sprot_raw
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
from dotenv import find_dotenv, load_dotenv
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)


class PDBReference(object):
    """
    This class contains the data for a single PDB entry which can be loaded from a specified .pdb file. Each instance is
    meant to serve as a reference for sequence based analyses.

    Attributes:
        file_name (str): The file name or path to the desired PDB file.
        structure (Bio.PDB.Structure.Structure): The Structure object parsed in from the PDB file, all other data in
        this class can be parsed out of this object but additional class attributes are generated (described below) to
        make these easier to access.
        chains (set): The chains which are present in this proteins structure.
        seq (dict): Sequence of the structure parsed in from the PDB file. For each chain in the structure (dict key)
        one sequence is stored (dict value).
        pdb_residue_list (dict): A sorted list of residue numbers (dict value) from the PDB file stored for each chain
        (dict key) in the structure.
        residue_pos (dict): A dictionary mapping chain identifier to another dictionary that maps residue number to the
        name of the residue (amino acid) at that position.
        size (dict): The length (dict value) of each amino acid chain (dict key) defining this structure.
        external_seq (dict): A multi-level dictionary where the first level key is the source ('UNP' for Swiss/Uniprot
        or 'GB' for GenBank), and the second level key is the chain identifier while its value is a tuple where the
        first position is the accession identifier and the second value is the amino acid sequence.
    """

    def __init__(self, pdb_file):
        """
        __init__

        Initiates an instance of the PDBReference class which stores structural data for a structure reference.

        Args:
            pdb_file (str): Path to the pdb file being represented by this instance.
        """
        if pdb_file is None:
            raise AttributeError('PDB File cannot be None!')
        else:
            pdb_file = os.path.abspath(pdb_file)
            if not os.path.isfile(pdb_file):
                raise AttributeError(f'PDB File path not valid: {pdb_file}')
        self.file_name = pdb_file
        self.structure = None
        self.chains = None
        self.seq = None
        self.pdb_residue_list = None
        self.residue_pos = None
        self.size = None
        self.external_seq = None

    def import_pdb(self, structure_id, save_file=None):
        """
        Import PDB

        This method imports a PDB file's information generating all data described in the Attribute list. This is
        achieved using the Bio.PDB package.

        Args:
            structure_id (str): The name of the query which the structure represents.
            save_file (str): The file path to a previously stored PDB file data structure.
        """
        start = time()
        if (save_file is not None) and os.path.exists(save_file):
            with open(save_file, 'rb') as handle:
                structure, seq, chains, pdb_residue_list, residue_pos = pickle.load(handle)
        else:
            # parser = PDBParser(PERMISSIVE=0)  # strict
            parser = PDBParser(PERMISSIVE=1)  # corrective
            structure = parser.get_structure(structure_id, self.file_name)
            seq = {}
            chains = set([])
            pdb_residue_list = {}
            residue_pos = {}
            for model in structure:
                for chain in model:
                    chains.add(chain.id)
                    pdb_residue_list[chain.id] = []
                    seq[chain.id] = ''
                    residue_pos[chain.id] = {}
                    for residue in chain:
                        if is_aa(residue.get_resname(), standard=True) and not residue.id[0].startswith('H_'):
                            res_name = three_to_one(residue.get_resname())
                            res_num = residue.get_id()[1]
                            residue_pos[chain.id][res_num] = res_name
                    for curr_res in sorted(residue_pos[chain.id]):
                        pdb_residue_list[chain.id].append(curr_res)
                        seq[chain.id] += residue_pos[chain.id][curr_res]
            if save_file is not None:
                with open(save_file, 'wb') as handle:
                    pickle.dump((structure, seq, chains, pdb_residue_list, residue_pos), handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
        self.structure = structure
        self.chains = chains
        self.seq = seq
        self.pdb_residue_list = pdb_residue_list
        self.residue_pos = residue_pos
        self.size = {chain: len(seq[chain]) for chain in self.chains}
        end = time()
        print('Importing the PDB file took {} min'.format((end - start) / 60.0))

    @staticmethod
    def _parse_uniprot_handle(handle):
        """
        Parse Uniprot Handle

        This method parses an amino acid sequence from a Swiss/Uniprot reference page.

        Argument:
            handle (IO Text Stream): A handle to a Swiss/Uniprot online reference for a given accession value.
        Return:
            str: The amino acid sequence retrieved from that reference.
        """
        data = handle.read()
        pattern = re.compile(r'.*SQ\s+SEQUENCE\s+\d+\s+AA;\s+\d+\s+MW;\s+[A-Z|\d]+\s+[A-Z|\d]+;\s+([A-Z|\s]+)\s+\/\/')
        seq = pattern.search(data).group(1)
        final_seq = re.sub(r'\s+', r'', seq)
        return final_seq

    @staticmethod
    def _retrieve_uniprot_seq(db_acc, db_id, seq_start, seq_end, max_tries=10):
        """
        Retrieve Uniprot Sequence

        This method iterates over a list of Swiss/Uniprot accession ids and returns the first amino acid sequence which
        it can successfully parse from the corresponding records.

        Argument:
            db_acc (str): The accession name for a Swiss/Uniprot sequence.
            db_id (str): The identifier for a Swiss/Uniprot sequence.
            seq_start (int): The first position in the sequence which is represented by the PDB file (the first index
            returned will be seq_start - 1 or 0 if None is provided).
            seq_end (int): The last position in the sequence which is represented by the PDB file (the last index
            returned will be seq_end - 1 or the full length of the sequence if None is provided).
            max_tries (int): The number of times attempts should be made to retrieve a sequence, due to inconsistent
            behavior from the target server.
        Return:
            str: The accession identifier to which the returned sequence belongs.
            str: The sequence of the first accession identifier that is successfully parsed.
        """
        if seq_start is None:
            seq_start = 0
        else:
            seq_start = seq_start - 1
        accessions = []
        if db_acc:
            accessions.append(db_acc)
        if db_id:
            accessions.append(db_id)
        record = None
        accession = None
        for accession in accessions:
            tries = 0
            while (tries < max_tries) and (record is None):
                if tries > 1:
                    sleep(1)
                tries += 1
                try:
                    handle = get_sprot_raw(accession)
                except HTTPError:
                    print('HTTPError on: {}'.format(accession))
                    continue
                try:
                    record = PDBReference._parse_uniprot_handle(handle=handle)
                    if seq_end is None:
                        seq_end = len(record)
                    record = record[seq_start: seq_end]
                except AttributeError:
                    print('No data returned for accession: {} with handle: {}'.format(accession, handle))
                    continue
                if record:
                    break
        # If no sequence could be successfully parsed, reset the accession id.
        if record is None:
            accession = None
        return accession, record

    @staticmethod
    def _retrieve_genbank_seq(db_acc, db_id, seq_start, seq_end, max_tries=10):
        """
        Retrieve GenBank Sequence

        This method iterates over a list of GenBank accession ids and returns the first amino acid sequence which
        it can successfully parse from the corresponding records.

        Argument:
            db_acc (str): The accession name for a GenBank sequence.
            db_id (str): The identifier for a GenBank sequence.
            seq_start (int): The first position in the sequence which is represented by the PDB file.
            seq_end (int): The last position in the sequence which is represented by the PDB file.
            max_tries (int): The number of times attempts should be made to retrieve a sequence, due to inconsistent
            behavior from the target server.
        Return:
            str: The accession identifier to which the returned sequence belongs.
            str: The sequence of the first accession identifier that is successfully parsed.
        """
        accessions = []
        if db_acc:
            accessions.append(db_acc)
        if db_id:
            accessions.append(db_id)
        record = None
        accession = None
        Entrez.email = os.environ.get('EMAIL')
        if seq_start is None:
            seq_start = 0
        else:
            seq_start = seq_start - 1
        for accession in accessions:
            tries = 0
            while (tries < max_tries) and (record is None):
                if tries > 1:
                    sleep(1)
                tries += 1
                try:
                    handle = Entrez.efetch(db='protein', rettype='fasta', retmode='text', id=accession)
                except IOError:
                    continue
                except AttributeError:
                    continue
                try:
                    record = str(read(handle, format='fasta').seq)
                    if seq_end is None:
                        seq_end = len(record)
                    record = record[seq_start: seq_end]
                except ValueError:
                    continue
                if record:
                    break
        if record is None:
            accession = None
        return accession, record

    @staticmethod
    def _parse_external_sequence_accessions(pdb_fn):
        """
        Parse External Sequence Accessions

        This function parses the PDB file again looking for external sequence accession identifiers. At the moment only
        Swiss/Uniprot and GenBank accessions are identified.

        Argument:
            pdb_fn (str/path): The path to the PDB file to parse.
        Return:
            dict: A two tiered dictionary where the first level key is the source, and the second level dictionary has
            chain identifiers as the keys and a list of identifiers as the values.
        """
        external_accessions = {}
        with open(pdb_fn, 'r') as pdb_handle:
            for line in pdb_handle:
                if line.startswith('DBREF'):
                    curr_dbref = dbref_parse(dbref_line=line)
                    if curr_dbref['db'] == 'PDB':
                        continue
                    elif curr_dbref['db'] not in external_accessions:
                        external_accessions[curr_dbref['db']] = {}
                    else:
                        pass
                    if curr_dbref['chain_id'] not in external_accessions[curr_dbref['db']]:
                        external_accessions[curr_dbref['db']][curr_dbref['chain_id']] = []
                    external_accessions[curr_dbref['db']][curr_dbref['chain_id']].append(
                        (curr_dbref['db_acc'], curr_dbref['db_id'], curr_dbref['db_seq_begin'],
                         curr_dbref['db_seq_end']))
        return external_accessions

    def _parse_external_sequences(self):
        """
        Parse External Sequences

        This method parses external sequences from the provided PDB file using any Swiss/Uniprot or GenBank identifiers
        found in the file.

        Return:
            dict: A two level dictionary where the first level keys are the source (i.e. whether the sequence comes from
            UNP, Swiss/Uniprot, or GB, GenBank) and the second level keys are chain identifiers which map to a set of
            tuples with the first identifier which could be parsed and its corresponding amino acid sequence.
        """
        retrieval_methods = {'UNP': self._retrieve_uniprot_seq, 'GB': self._retrieve_genbank_seq}
        external_accessions = self._parse_external_sequence_accessions(pdb_fn=self.file_name)
        external_seqs = {}
        for source in external_accessions:
            if source not in retrieval_methods:
                continue
            external_seqs[source] = {}
            for chain in external_accessions[source]:
                if len(external_accessions[source][chain]) > 1:
                    prev_acc = None
                    prev_id = None
                    prev_start = None
                    prev_end = None
                    for entry in sorted(external_accessions[source][chain]):
                        if prev_acc is None:
                            prev_acc = entry[0]
                            prev_id = entry[1]
                            prev_start = entry[2]
                            prev_end = entry[3]
                        elif (prev_acc == entry[0]) and (prev_id == entry[1]):
                            if prev_start > entry[2]:
                                prev_start = entry[2]
                            if prev_end < entry[3]:
                                prev_end = entry[3]
                        else:
                            raise ValueError(f'Multiple references for the same chain from different accessions: '
                                             f'{prev_acc} and {entry[0]}')
                    external_accessions[source][chain] = [(prev_acc, prev_id, prev_start, prev_end)]
                external_seqs[source][chain] = retrieval_methods[source](*external_accessions[source][chain][0])
        return external_seqs

    def get_sequence(self, chain, source='PDB'):
        """
        Get Sequence

        This method returns the sequence for a given chain from the specified source if available, if not available,
        None is returned.

        Arguments:
            chain (str): The chain identifier for which the sequence should be returned.
            source (str): The name of the source to use to retrieve the chain sequence from. If PDB is specified then
            the amino acid sequence of the chain in the structure is returned. If UNP or GB are specified then the
            sequence for the appropriate Swiss/Uniprot or GenBank (respectively) accession identifiers are returned.
        Return:
             str/None: The identifier for the given chain from the specified source if one could be parsed, or None
             otherwise.
             str/None: The sequence for the given chain from the specified source if available, or None otherwise.
        """
        sequence = None
        identifier = None
        if source == 'PDB':
            if self.seq is None:
                raise AttributeError('Source PDB cannot be accessed if import_pdb has not been called.')
            sequence = self.seq[chain]
            identifier = self.structure.id
        elif source in {'UNP', 'GB'}:
            if self.external_seq is None:
                self.external_seq = self._parse_external_sequences()
            if source in self.external_seq:
                if chain in self.external_seq[source]:
                    identifier, sequence = self.external_seq[source][chain]
        else:
            raise ValueError('Expected sources are PDB, UNP, or GB.')
        return identifier, sequence

    def color_structure(self, chain_id, data, data_type, data_direction, coloring_threshold, color_map, out_dir):
        """
        Color Structure

        This function takes data and colors the specified chain for this the current instance's structure according to
        that data and the direction, threshold, and color map provided.

        Args:
            chain_id (str): The chain in the structure to color.
            data_type (str): The name of the data to use from the provided pandas.DataFrame.
            data (pandas.DataFrame): A dataframe containing at least a column named 'RESIDUE_Index' and with the name
            provided in the data_type parameter. Entries in the 'RESIDUE_Index' column should be the numerical indices
            for the residues in the structure. Entries in the column corresponding to data_type, should be normalized
            between 0 and 1.
            data_direction (str): Expected values are 'min' or 'max'. If 'min' is specified then the normal version of
            the specified color map is used, if 'max' is specified then the reverse version of the specified color map
            is used. This parameter also affects how the coloring_threshold is used.
            coloring_threshold (float): The cutoff value for coloring residues. If data_direction is 'min' then any
            value greater than the provided threshold is not colored, if data_direction is 'max' then any value less
            than the provided threshold is not colored.
            color_map (str): A string specifying the color map to use. If 'ET' is specified the prismatic colormap used
            in the PyETViewer is used. Any other value which is provided should be a valid matplotlib colormap name as
            described here: https://matplotlib.org/stable/gallery/color/colormap_reference.html
            out_dir (str): The path to the directory where the .pse file and .txt file with all commands for this
            coloring should be saved.
        Returns:
            str: The path to the created pse file.
            str: The path to the text file containing all commands used to generate the pse file.
            list: The residues in the structure which have been colored by this method using the provided data.
        """
        if color_map == 'ET':
            color_list = ["ff0000", "ff0c00", "ff1800", "ff2400", "ff3000", "ff3d00", "ff4900", "ff5500", "ff6100",
                          "ff6e00", "ff7a00", "ff8600", "ff9200", "ff9f00", "ffab00", "ffb700", "ffc300", "ffd000",
                          "ffdc00", "ffe800", "fff400", "fcff00", "f0ff00", "e4ff00", "d8ff00", "cbff00", "bfff00",
                          "b3ff00", "a7ff00", "9bff00", "8eff00", "82ff00", "76ff00", "6aff00", "5dff00", "51ff00",
                          "45ff00", "39ff00", "2cff00", "20ff00", "14ff00", "08ff00", "00ff04", "00ff10", "00ff1c",
                          "00ff28", "00ff35", "00ff41", "00ff4d", "00ff59", "00ff66", "00ff72", "00ff7e", "00ff8a",
                          "00ff96", "00ffa3", "00ffaf", "00ffbb", "00ffc7", "00ffd4", "00ffe0", "00ffec", "00fff8",
                          "00f8ff", "00ecff", "00e0ff", "00d4ff", "00c7ff", "00bbff", "00afff", "00a3ff", "0096ff",
                          "008aff", "007eff", "0072ff", "0066ff", "0059ff", "004dff", "0041ff", "0035ff", "0028ff",
                          "001cff", "0010ff", "0004ff", "0800ff", "1400ff", "2000ff", "2c00ff", "3900ff", "4500ff",
                          "5100ff", "5d00ff", "6a00ff", "7600ff", "8200ff", "8e00ff", "9b00ff", "a700ff", "b300ff",
                          "bf00ff"]
            converted_color_list = [hex2color('#' + x) for x in color_list]
            cmap_f = LinearSegmentedColormap.from_list('ET_Color_Map', converted_color_list, N=len(color_list))
            cmap_r = LinearSegmentedColormap.from_list('Reverse_ET_Color_Map', converted_color_list[::-1],
                                                       N=len(color_list))
        else:
            cmap_f = get_cmap(color_map)
            cmap_r = get_cmap(color_map + '_r')
        full_selection_list = []
        all_commands = []
        if data_direction == 'min':
            cmap = cmap_f
        elif data_direction == 'max':
            cmap = cmap_r
        else:
            raise ValueError('Bad value provided for data_direction, expected "min" or "max".')
        curr_name = f"{self.structure.id}_{data_type.replace(' ', '_')}"
        cmd.load(self.file_name, curr_name)
        full_selection_list.append(curr_name)
        all_commands.append(f'load {self.file_name}, {curr_name}')
        cmd.color('white', curr_name)
        all_commands.append(f'color white, {curr_name}')
        if chain_id not in self.chains:
            raise ValueError('Provided chain_id is not in the current structure.')
        curr_chain = f'Chain_{chain_id}'
        cmd.select(curr_chain, f'{curr_name} and chain {chain_id}')
        full_selection_list.append(curr_chain)
        all_commands.append(f'select {curr_chain}, {curr_name} and chain {chain_id}')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Bad data provided, expected pandas.DataFrame')
        unique_values = data[data_type].unique()
        if (coloring_threshold < 0.0) or (coloring_threshold > 1.0):
            raise ValueError('coloring threshold outside of range from 0.0 to 1.0')
        colored_residues = []
        for value in sorted(unique_values, reverse=(False if data_direction == 'min' else True)):
            if ((value > coloring_threshold) and (data_direction == 'min')) or\
                    ((value < coloring_threshold) and (data_direction == 'max')) or (value is None) or np.isnan(value):
                continue
            residues = sorted(data.loc[data[data_type] == value, 'RESIDUE_Index'].unique())
            res_selection_label = f'{round(value, 5)}_residues'
            cmd.select(res_selection_label, f"{curr_chain} and resi {'+'.join([str(x) for x in residues])}")
            all_commands.append(f"select {res_selection_label}, {curr_chain} and resi {'+'.join([str(x) for x in residues])}")
            cmd.color(f'0x{to_hex(cmap(value)).upper()[1:]}', res_selection_label)
            all_commands.append(f'color 0x{to_hex(cmap(value)).upper()[1:]}, {res_selection_label}')
            full_selection_list.append(res_selection_label)
            colored_residues += residues
        if len(colored_residues) > 0:
            cmd.select('colored', f"{curr_chain} and resi {'+'.join([str(x) for x in colored_residues])}")
            all_commands.append(
                f"select colored, {curr_chain} and resi {'+'.join([str(x) for x in colored_residues])}")
            cmd.select('not_colored', f"{curr_chain} and (not colored)")
            all_commands.append(f"select not_colored, {curr_chain} and (not colored)")
            full_selection_list = full_selection_list[:2] + ['colored', 'not_colored'] + full_selection_list[2:]
        pse_path = os.path.join(out_dir, f'{curr_name}_{curr_chain}_threshold_{coloring_threshold}.pse')
        cmd.save(pse_path, ' or '.join(full_selection_list), -1, 'pse')
        all_commands.append(f"save {os.path.join(out_dir, f'{curr_name}_{curr_chain}_{coloring_threshold}.pse')},"
                            ' or '.join(full_selection_list) + ' , -1, pse')
        cmd.delete('all')
        all_commands.append('delete all')
        commands_path = os.path.join(out_dir, f'{curr_name}_{curr_chain}_threshold_{coloring_threshold}_all_pymol_commands.txt')
        with open(commands_path, 'w') as handle:
            for line in all_commands:
                handle.write(line + '\n')
        return pse_path, commands_path, sorted(colored_residues)

    def display_pairs(self, chain_id, data, pair_col, res_col1, res_col2, data_direction, color_map, out_dir, fn=None):
        """
        Color Structure

        This function takes data and colors the specified chain for this the current instance's structure according to
        that data and the direction, threshold, and color map provided.

        Args:
            chain_id (str): The chain in the structure to color.
            pair_col (str): The name of the column to use for the pair data from the provided pandas.DataFrame.
            res_col1 (str): The name of the column to use for residue specific data for the first residue in a pair
            (i.e. the residue in column 'RESIDUE_Index_1') from the provided pandas.DataFrame.
            res_col2 (str): The name of the column to use for residue specific data for the second residue in a pair
            (i.e. the residue in column 'RESIDUE_Index_2') from the provided pandas.DataFrame.
            data (pandas.DataFrame): A dataframe containing at least a column named 'RESIDUE_Index_1', 'RESIDUE_Index_2'
            and columns with the names provided in the pair_col and residue_col parameters. Entries in the
            'RESIDUE_Index_1' and 'RESIDUE_Index_2' columns should be the numerical indices for the residues in the
            structure. Entries in the column corresponding to pair_col and residue_col, should be normalized
            between 0 and 1.
            data_direction (str): Expected values are 'min' or 'max'. If 'min' is specified then the normal version of
            the specified color map is used, if 'max' is specified then the reverse version of the specified color map
            is used. This parameter also affects how the coloring_threshold is used.
            color_map (str): A string specifying the color map to use. If 'ET' is specified the prismatic colormap used
            in the PyETViewer is used. Any other value which is provided should be a valid matplotlib colormap name as
            described here: https://matplotlib.org/stable/gallery/color/colormap_reference.html
            out_dir (str): The path to the directory where the .pse file and .txt file with all commands for this
            coloring should be saved.
            fn (str): A string specifying the filename for the pse and commands files being written out which will have
            the format:
                f{fn}.pse
                f{fn}_all_pymol_commands.txt
            If None a default will be used with the format:
                f'{curr_name}_{curr_chain}_displayed_pairs.pse
                f'{curr_name}_{curr_chain}_displayed_paris_all_pymol_commands.txt
        Returns:
            str: The path to the created pse file.
            str: The path to the text file containing all commands used to generate the pse file.
            list: The residues in the structure which have been colored by this method using the provided data.
        """
        if color_map == 'ET':
            color_list = ["ff0000", "ff0c00", "ff1800", "ff2400", "ff3000", "ff3d00", "ff4900", "ff5500", "ff6100",
                          "ff6e00", "ff7a00", "ff8600", "ff9200", "ff9f00", "ffab00", "ffb700", "ffc300", "ffd000",
                          "ffdc00", "ffe800", "fff400", "fcff00", "f0ff00", "e4ff00", "d8ff00", "cbff00", "bfff00",
                          "b3ff00", "a7ff00", "9bff00", "8eff00", "82ff00", "76ff00", "6aff00", "5dff00", "51ff00",
                          "45ff00", "39ff00", "2cff00", "20ff00", "14ff00", "08ff00", "00ff04", "00ff10", "00ff1c",
                          "00ff28", "00ff35", "00ff41", "00ff4d", "00ff59", "00ff66", "00ff72", "00ff7e", "00ff8a",
                          "00ff96", "00ffa3", "00ffaf", "00ffbb", "00ffc7", "00ffd4", "00ffe0", "00ffec", "00fff8",
                          "00f8ff", "00ecff", "00e0ff", "00d4ff", "00c7ff", "00bbff", "00afff", "00a3ff", "0096ff",
                          "008aff", "007eff", "0072ff", "0066ff", "0059ff", "004dff", "0041ff", "0035ff", "0028ff",
                          "001cff", "0010ff", "0004ff", "0800ff", "1400ff", "2000ff", "2c00ff", "3900ff", "4500ff",
                          "5100ff", "5d00ff", "6a00ff", "7600ff", "8200ff", "8e00ff", "9b00ff", "a700ff", "b300ff",
                          "bf00ff"]
            converted_color_list = [hex2color('#' + x) for x in color_list]
            cmap_f = LinearSegmentedColormap.from_list('ET_Color_Map', converted_color_list, N=len(color_list))
            cmap_r = LinearSegmentedColormap.from_list('Reverse_ET_Color_Map', converted_color_list[::-1],
                                                       N=len(color_list))
        else:
            cmap_f = get_cmap(color_map)
            cmap_r = get_cmap(color_map + '_r')
        full_selection_list = []
        all_commands = []
        if data_direction == 'min':
            cmap = cmap_f
        elif data_direction == 'max':
            cmap = cmap_r
        else:
            raise ValueError('Bad value provided for data_direction, expected "min" or "max".')
        curr_name = f"{self.structure.id}_{pair_col.replace(' ', '_')}_pairs"
        cmd.load(self.file_name, curr_name)
        full_selection_list.append(curr_name)
        all_commands.append(f'load {self.file_name}, {curr_name}')
        cmd.color('white', curr_name)
        all_commands.append(f'color white, {curr_name}')
        if chain_id not in self.chains:
            raise ValueError('Provided chain_id is not in the current structure.')
        curr_chain = f'Chain_{chain_id}'
        cmd.select(curr_chain, f'{curr_name} and chain {chain_id}')
        full_selection_list.append(curr_chain)
        all_commands.append(f'select {curr_chain}, {curr_name} and chain {chain_id}')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Bad data provided, expected pandas.DataFrame')
        unique_res_values = data[pair_col].unique()
        colored_residues = []
        colored_pairs = []
        for value in sorted(unique_res_values, reverse=(False if data_direction == 'min' else True)):
            pairs = data.loc[data[pair_col] == value, ['RESIDUE_Index_1', 'RESIDUE_Index_2', res_col1, res_col2]]
            for _, pair in pairs.iterrows():
                if (pair[res_col1] is None or pair[res_col2] is None) or\
                        (np.isnan(pair[res_col1]) or np.isnan(pair[res_col2])):
                    continue
                if pair['RESIDUE_Index_1'] not in colored_residues:
                    cmd.select('curr_res', f"{curr_chain} and resi {pair['RESIDUE_Index_1']}")
                    all_commands.append(f"select curr_res, {curr_chain} and resi {pair['RESIDUE_Index_1']}")
                    cmd.color(f'0x{to_hex(cmap(value)).upper()[1:]}', 'curr_res')
                    all_commands.append(f'color 0x{to_hex(cmap(value)).upper()[1:]}, curr_res')
                    cmd.delete('curr_res')
                    all_commands.append('delete curr_res')
                    colored_residues.append(pair['RESIDUE_Index_1'])
                if pair['RESIDUE_Index_2'] not in colored_residues:
                    cmd.select('curr_res', f"{curr_chain} and resi {pair['RESIDUE_Index_2']}")
                    all_commands.append(f"select curr_res, {curr_chain} and resi {pair['RESIDUE_Index_2']}")
                    cmd.color(f'0x{to_hex(cmap(value)).upper()[1:]}', 'curr_res')
                    all_commands.append(f'color 0x{to_hex(cmap(value)).upper()[1:]}, curr_res')
                    cmd.delete('curr_res')
                    all_commands.append('delete curr_res')
                    colored_residues.append(pair['RESIDUE_Index_2'])
                if data_direction == 'min':
                    worse_score = max(pair[res_col1], pair[res_col2])
                else:
                    worse_score = min(pair[res_col1], pair[res_col2])
                pair_label = f"{pair['RESIDUE_Index_1']}_{pair['RESIDUE_Index_1']}_pair"
                cmd.distance(pair_label, f"{curr_chain} and resi {pair['RESIDUE_Index_1']} and name CA",
                             f"{curr_chain} and resi {pair['RESIDUE_Index_2']} and name CA")
                all_commands.append(f"distance {pair_label}, {curr_chain} and resi {pair['RESIDUE_Index_1']} and name"
                                    f" CA, {curr_chain} and resi {pair['RESIDUE_Index_2']} and name CA")
                cmd.hide(representation='labels', selection=pair_label)
                all_commands.append(f'hide labels, {pair_label}')
                cmd.set('dash_gap', 0.0, selection=pair_label)
                all_commands.append(f'set dash_gap, 0.0, {pair_label}')
                cmd.set('dash_radius', 0.1, selection=pair_label)
                all_commands.append(f'set dash_radius, 0.1, {pair_label}')
                cmd.set('dash_color', f'0x{to_hex(cmap(worse_score)).upper()[1:]}', selection=pair_label)
                all_commands.append(f'set dash_color, 0x{to_hex(cmap(worse_score)).upper()[1:]}, {pair_label}')
                colored_pairs.append((pair['RESIDUE_Index_1'], pair['RESIDUE_Index_2']))
        if len(colored_residues) > 0:
            cmd.select('colored', f"{curr_chain} and resi {'+'.join([str(x) for x in colored_residues])}")
            all_commands.append(
                f"select colored, {curr_chain} and resi {'+'.join([str(x) for x in colored_residues])}")
            cmd.select('not_colored', f"{curr_chain} and (not colored)")
            all_commands.append(f"select not_colored, {curr_chain} and (not colored)")
            full_selection_list = full_selection_list[:2] + ['colored', 'not_colored'] + full_selection_list[2:]
        if fn is None:
            pse_path = os.path.join(out_dir, f'{curr_name}_{curr_chain}_displayed_pairs.pse')
            commands_path = os.path.join(out_dir,
                                         f'{curr_name}_{curr_chain}_displayed_pairs_all_pymol_commands.txt')
        else:
            pse_path = os.path.join(out_dir, fn + '.pse')
            commands_path = os.path.join(out_dir, fn + '_all_pymol_commands.txt')
        cmd.save(pse_path, ' or '.join(full_selection_list), -1, 'pse')
        all_commands.append(f"save {pse_path}," ' or '.join(full_selection_list) + ' , -1, pse')
        cmd.delete('all')
        all_commands.append('delete all')

        with open(commands_path, 'w') as handle:
            for line in all_commands:
                handle.write(line + '\n')
        return pse_path, commands_path, sorted(colored_residues), sorted(colored_pairs)


def dbref_parse(dbref_line):
    """
    DBREF Parse

    This function parses values out of a DBREF entry line in a PDB file if it follows the conventions for DBREF standard
    format version 3.3 as described at https://www.wwpdb.org/documentation/file-format-content/format33/sect3.html

    Args:
        dbref_line (str): Line starting with "DBREF" which follows the standard format described in above.
    Return:
        dict: A dictionary returning each of the elements described in the DBREF standard format description with the
        following key names: rec_name, id_code, chain_id, seq_begin, ins_begin, seq_end, ins_end, db, db_acc, db_id,
        db_seq_begin, db_ins_begin, db_seq_end, db_ins_end. The values for keys seq_begin, seq_end, db_seq_begin, and
        deb_seq_end are returned as ints all others are returned as strs.
    """
    try:
        dbref_entry = {'rec_name': dbref_line[0:6].lstrip().rstrip(), 'id_code': dbref_line[7:11].lstrip().rstrip(),
                       'chain_id': dbref_line[12], 'seq_begin': int(dbref_line[14:18].lstrip().rstrip()),
                       'ins_begin': dbref_line[18].lstrip().rstrip(),
                       'seq_end': int(dbref_line[20:24].lstrip().rstrip()),
                       'ins_end': dbref_line[24].lstrip().rstrip(), 'db': dbref_line[26:32].lstrip().rstrip(),
                       'db_acc': dbref_line[33:41].lstrip().rstrip(), 'db_id': dbref_line[42:54].lstrip().rstrip(),
                       'db_seq_begin': int(dbref_line[55:60].lstrip().rstrip()),
                       'db_ins_begin': dbref_line[60].lstrip().rstrip(),
                       'db_seq_end': int(dbref_line[62:67].lstrip().rstrip()),
                       'db_ins_end': dbref_line[67].lstrip().rstrip()}
    except ValueError:
        raise ValueError('Provided DBREF line does not follow the expected format!'
                         'Only the standard format is currently supported.')
    except IndexError:
        raise ValueError('Provided DBREF line does not follow the expected format!'
                         'Only the standard format is currently supported.')
    return dbref_entry
