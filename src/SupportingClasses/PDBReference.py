"""
Created on Aug 17, 2017

@author: daniel
"""
import os
import re
import pickle
from time import time
from urllib.error import HTTPError
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
    def _retrieve_uniprot_seq(db_acc, db_id, seq_start, seq_end):
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
        Return:
            str: The accession identifier to which the returned sequence belongs.
            str: The sequence of the first accession identifier that is successfully parsed.
        """
        if seq_start is None:
            seq_start = 0
        else:
            seq_start = seq_start - 1
        accessions = [db_acc, db_id]
        record = None
        accession = None
        for accession in accessions:
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
    def _retrieve_genbank_seq(db_acc, db_id, seq_start, seq_end):
        """
        Retrieve GenBank Sequence

        This method iterates over a list of GenBank accession ids and returns the first amino acid sequence which
        it can successfully parse from the corresponding records.

        Argument:
            db_acc (str): The accession name for a GenBank sequence.
            db_id (str): The identifier for a GenBank sequence.
            seq_start (int): The first position in the sequence which is represented by the PDB file.
            seq_end (int): The last position in the sequence which is represented by the PDB file.
        Return:
            str: The accession identifier to which the returned sequence belongs.
            str: The sequence of the first accession identifier that is successfully parsed.
        """
        accessions = [db_acc, db_id]
        record = None
        accession = None
        Entrez.email = os.environ.get('EMAIL')
        for accession in accessions:
            try:
                handle = Entrez.efetch(db='protein', rettype='fasta', retmode='text', id=accession)
            except IOError:
                continue
            try:
                record = str(read(handle, format='fasta').seq)[seq_start - 1: seq_end]
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
        dbrefs = []
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
                external_seqs[source][chain] = retrieval_methods[source](accessions=external_accessions[source][chain])
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
