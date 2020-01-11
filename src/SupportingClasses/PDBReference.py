"""
Created on Aug 17, 2017

@author: daniel
"""
import os
import pickle
from time import time
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.Polypeptide import is_aa


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
    """

    def __init__(self, pdb_file):
        """
        __init__

        Initiates an instance of the PDBReference class which stores structural data for a structure reference.

        Args:
            pdb_file (str): Path to the pdb file being represented by this instance.
        """
        if pdb_file.startswith('..'):
            pdb_file = os.path.abspath(os.path.join(os.getcwd(), pdb_file))
        self.file_name = pdb_file
        self.structure = None
        self.chains = None
        self.seq = None
        self.pdb_residue_list = None
        self.residue_pos = None
        self.size = None

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
            structure, seq, chains, pdb_residue_list, residue_pos = pickle.load(open(save_file, 'r'))
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
                            seq[chain.id] += res_name
                            res_num = residue.get_id()[1]
                            residue_pos[chain.id][res_num] = res_name
                            pdb_residue_list[chain.id].append(res_num)
            if save_file is not None:
                pickle.dump((structure, seq, chains, pdb_residue_list, residue_pos), open(save_file, 'w'),
                            protocol=pickle.HIGHEST_PROTOCOL)
        self.structure = structure
        self.chains = chains
        self.seq = seq
        self.pdb_residue_list = pdb_residue_list
        self.residue_pos = residue_pos
        self.size = {chain: len(seq[chain]) for chain in self.chains}
        end = time()
        print('Importing the PDB file took {} min'.format((end - start) / 60.0))
