"""
Created on May 23, 2019

@author: Daniel Konecki
"""
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser
from Bio.Alphabet.IUPAC import ExtendedIUPACProtein
from Bio.PDB.Polypeptide import three_to_one, is_aa
from dotenv import find_dotenv, load_dotenv
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)


class DataSetGenerator(object):

    def __init__(self, protein_list):
        self.input_path = os.path.join(os.environ.get('PROJECT_PATH'), 'Input')
        self.file_name = protein_list
        self.protein_data = self._import_protein_list()

    def _import_protein_list(self):
        protein_list_fn = os.path.join(self.input_path, 'ProteinLists', self.file_name)
        protein_list = {}
        with open(protein_list_fn, mode='rb') as protein_list_handle:
            for line in protein_list_handle:
                protein_list[line.strip()] = {}
        return protein_list

    def _download_pdbs(self):
        pdb_path = os.path.join(self.input_path, 'PDB')
        if not os.path.isdir(pdb_path):
            os.makedirs(pdb_path)
        pdb_list = PDBList(server='ftp://ftp.wwpdb.org', pdb=pdb_path)
        for pdb_id in self.protein_data:
            pdb_file = pdb_list.retrieve_pdb_file(pdb_code=pdb_id, file_format='pdb')
            self.protein_data[pdb_id]['PDB_Path'] = pdb_file

    def _parse_query_sequences(self):
        preotin_fasta_
        parser = PDBParser(PERMISSIVE=1)  # corrective
        for pdb_id in self.protein_data:
            print(pdb_id)
            structure = parser.get_structure(pdb_id, self.protein_data[pdb_id]['PDB_Path'])
            model = structure[0]
            chain = model['A']
            sequence = []
            for residue in chain:
                if is_aa(residue.get_resname(), standard=True):
                    res_name = three_to_one(residue.get_resname())
                    sequence.append(res_name)
            sequence = ''.join(sequence)
            self.protein_data[pdb_id]['Query_Sequence'] = Seq(sequence, alphabet=ExtendedIUPACProtein)
            self.protein_data[pdb_id]['Sequence_Length'] = len(sequence)


    def blast_query_sequences(self):
        raise NotImplemented('Not yet implemented')