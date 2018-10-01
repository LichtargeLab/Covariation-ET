"""
Created on Aug 17, 2017

@author: daniel
"""
import cPickle as pickle
import os
from time import time

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.Polypeptide import is_aa


class PDBReference(object):
    """
    classdocs
    """

    def __init__(self, pdb_file):
        """
        Constructor

        Initiates an instance of the PDBReference class which stores the
        following data:

        file_name: str
            The file name or path to the desired PDB file.
        residue_3d : dict
            A dictionary mapping a residue number to its spatial position in 3D.
        pdb_residue_list : list
            A sorted list of residue numbers from the PDB file.
        residue_pos : dict
            A dictionary mapping residue number to the name of the residue at that
            position.
        seq:
            Sequence of the structure parsed in from the PDB file.
        query_pdb_mapping : dict
            A structure mapping the index of the positions in the fasta sequence
            which align to positions in the PDB sequence based on a local alignment
            with no mismatches allowed.
        residue_dists : list
            List of minimum distances between residues, sorted by the ordering
            of residues in pdb_residue_list.
        chains : set
            The chains which are present in this proteins structure.
        size : int
            The length of the amino acid chain defining this structure.
        """
        if pdb_file.startswith('..'):
            pdb_file = os.path.abspath(os.path.join(os.getcwd(), pdb_file))
        self.file_name = pdb_file
        self.pdb_residue_list = None
        self.residue_pos = None
        self.seq = None
        self.query_pdb_mapping = None
        self.residue_dists = None
        self.chains = None
        self.size = 0
        self.structure = None
        self.best_chain = None

    def import_pdb(self, structure_id, save_file=None):
        """
        import_pdb

        This method imports a PDB files information generating a list of lists.
        Each list contains the Amino Acid 3-letter abbreviation, residue number,
        x, y, and z coordinate. This method updates the following class
        variables: residue_3d, pdb_residue_list, residue_pos, and seq.

        Parameters:
        -----------
        save_file: str
            The file path to a previously stored PDB file data structure.
        """
        start = time()
        if (save_file is not None) and os.path.exists(save_file):
            structure, seq, chains, pdb_residue_list, residue_pos = pickle.load(open(save_file, 'rb'))
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
                        if is_aa(residue.get_resname(), standard=True):
                            res_name = three_to_one(residue.get_resname())
                        else:
                            res_name = 'X'
                        seq[chain.id] += res_name
                        res_num = residue.get_id()[1]
                        residue_pos[chain.id][res_num] = res_name
                        pdb_residue_list[chain.id].append(res_num)
            print(seq)
            if save_file is not None:
                pickle.dump((structure, seq, chains, pdb_residue_list, residue_pos), open(save_file, 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)
        self.structure = structure
        self.chains = chains
        self.seq = seq
        self.pdb_residue_list = pdb_residue_list
        self.residue_pos = residue_pos
        self.size = {chain: len(seq[chain]) for chain in self.chains}
        end = time()
        print('Importing the PDB file took {} min'.format((end - start) / 60.0))
