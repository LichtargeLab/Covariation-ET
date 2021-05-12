"""
Created on May 11, 2021

@author: Daniel Konecki
"""
from time import time
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo
from SupportingClasses.SeqAlignment import SeqAlignment
from SupportingClasses.PDBReference import PDBReference


class SequencePDBMap(object):
    """
    SequencePDBMap

    This class holds the mapping between a given alignment (and its query sequence) and a provided PBD structure. The
    alignment and mapping are performed using the same parameters used in the PyETViewer for consistency.

    Attributes:
        query (str): The name of the query sequence/structure.
        query_alignment (str/SeqAlignment): Either a SeqAlignment object where the alignment has already been
        imported or the path to a fasta alignment which can be imported to a SeqAlignment object. The query sequence
        of this alignment is what will be mapped to the provided structure.
        query_structure (str/PDBReference): Either a PDBReference object where the structure has already been
        imported or the path to a PDB formatted file which can be imported to a PDBReference object. The query structure
        is what the provided alignment (query sequence) will be mapped to.
        best_chain (str): The chain of the query structure to which the query alignment/sequence was aligned and mapped.
        query_pdb_mapping (dict): A mapping from the index of the query sequence to the index of the pdb chain's
        sequence for those positions which match according to a pairwise global alignment.
        pdb_query_mapping (dict): A mapping from the index of the pdb chain's sequence to the index of the query
        sequence for those positions which match according to a pairwise global alignment.
    """

    def __init__(self, query, query_alignment, query_structure, chain=None):
        """
        __init__

        This initializes a new instance of the SequencePDBMap.

        Args:
            query (str): The sequence identifier for the sequence of interest in the alignment.
            query_alignment (str/SeqAlignment): Either a SeqAlignment object where the alignment has already been
            imported or the path to a fasta alignment which can be imported to a SeqAlignment object.
            query_structure (str/PDBReference): Either a PDBReference object where the structure has already been
            imported or the path to a PDB formatted file which can be imported to a PDBReference object.
            chain (str): The chain to match the query sequence to, if None is provided the align() function will attempt
            to find the best matching chain.
        """
        self.query = query
        self.seq_aln = query_alignment
        self.pdb_ref = query_structure
        self.best_chain = chain
        self.query_pdb_mapping = None
        self.pdb_query_mapping = None

    def align(self):
        """
        Align

        This function aligns the provided query sequence and the provided reference PDB structure. The alignment is
        performed the same way it is performed in the PyETViewer module, using gap_open = -10 and gap_extend = -0.5. If
        the desired chain was not specified at initialization this function will attempt to find the closes match by
        aligning all chains with the query sequence and choosing the one which scores the best (if multiple chains score
        the same the first chain, alphabetically, is used). This function updates the query_pdb_mapping and
        pdb_query_mapping attributes.
        """
        gap_open = -10
        gap_extend = -0.5
        if (self.best_chain is None) or (self.query_pdb_mapping is None):
            start = time()
            if self.seq_aln is None:
                raise ValueError('Scorer cannot be fit, because no alignment was provided.')
            else:
                if not isinstance(self.seq_aln, SeqAlignment):
                    self.seq_aln = SeqAlignment(file_name=self.seq_aln, query_id=self.query)
                self.seq_aln.import_alignment()
                self.seq_aln = self.seq_aln.remove_gaps()
            if self.pdb_ref is None:
                raise ValueError('Scorer cannot be fit, because no PDB was provided.')
            else:
                if not isinstance(self.pdb_ref, PDBReference):
                    self.pdb_ref = PDBReference(pdb_file=self.pdb_ref)
                self.pdb_ref.import_pdb(structure_id=self.query)
            if self.best_chain is None:
                best_chain = None
                best_alignment = None
                for ch in self.pdb_ref.seq:
                    curr_align = pairwise2.align.globalds(self.seq_aln.query_sequence, self.pdb_ref.seq[ch],
                                                          MatrixInfo.blosum62, gap_open, gap_extend)
                    if (best_alignment is None) or (best_alignment[0][2] < curr_align[0][2]):
                        best_alignment = curr_align
                        best_chain = ch
                self.best_chain = best_chain
            else:
                best_alignment = pairwise2.align.globalds(self.seq_aln.query_sequence,
                                                          self.pdb_ref.seq[self.best_chain], MatrixInfo.blosum62,
                                                          gap_open, gap_extend)
            print(pairwise2.format_alignment(*best_alignment[0]))
            if (self.query_pdb_mapping is None) or (self.pdb_query_mapping is None):
                f_counter = 0
                p_counter = 0
                f_to_p_map = {}
                p_to_f_map = {}
                for i in range(len(best_alignment[0][0])):
                    if (best_alignment[0][0][i] != '-') and (best_alignment[0][1][i] != '-'):
                        f_to_p_map[f_counter] = p_counter
                        p_to_f_map[p_counter] = f_counter
                    if best_alignment[0][0][i] != '-':
                        f_counter += 1
                    if best_alignment[0][1][i] != '-':
                        p_counter += 1
                self.query_pdb_mapping = f_to_p_map
                self.pdb_query_mapping = p_to_f_map
            end = time()
            print('Mapping query sequence and pdb took {} min'.format((end - start) / 60.0))

    def map_seq_position_to_pdb_res(self, seq_pos):
        """
        Map Sequence Position to PDB Residue

        This function returns the PDB residue which corresponds with the provided sequence position.

        Args:
            seq_pos (int): The position of interest in the sequence.
        Return:
            int: The corresponding residue in the PDB structure.
        """
        if self.query_pdb_mapping is None:
            raise AttributeError('Alignment must be performed to be able to map.')
        return self.pdb_ref.pdb_residue_list[self.best_chain][self.query_pdb_mapping[seq_pos]]

    def map_pdb_res_to_seq_position(self, pdb_res):
        """
        Map PDB Residue to Sequence Position

        This function returns the position in the query sequence which corresponds to the given position provided in the
        PDB structure.

        Args:
            pdb_res (int): Position from the reference PDB structure.
        Return:
            int: Position in the query sequence which corresponds to the provided PDB residue.
        """
        if self.pdb_query_mapping is None:
            raise AttributeError('Alignment must be performed to be able to map.')
        index = self.pdb_ref.pdb_residue_list[self.best_chain].index(pdb_res)
        return self.pdb_query_mapping[index]
