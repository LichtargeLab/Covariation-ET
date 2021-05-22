"""
Created on May 22, 2021

@author: Daniel Konecki
"""
from time import time
from Evaluation.SequencePDBMap import SequencePDBMap


class Scorer(object):
    """
    This class is intended as the base for any scorer meant to perform evaluations on predictions of single position
    residue importance or paired position covariation predictions. It establishes a minimum set of attributes and the
    corresponding initializer as well as a method signature for the only other function all Predictor sub-classes must
    have, namely fit.

    Attributes:
        query_pdb_mapper (SequencePDBMap): A mapping from the index of the query sequence to the index of the
        pdb chain's sequence for those positions which match according to a pairwise global alignment and vice versa.
        data (panda.DataFrame): A data frame containing a mapping of the sequence, structure, and predictions as well as
        other fields which may be specific to different types of Scorer objects.
    """

    def __init__(self, query, seq_alignment, pdb_reference, chain=None):
        """
        __init__

        This function initializes a new Scorer, it accepts paths to an alignment and a PDB file to be used in the
        assessment of single residue importance or contact predictions for a given query. The PDB chain to use for this
        analysis may also be specified.

        Args:
            query (str): The name of the query sequence/structure.
            seq_alignment (str/path, SeqAlignment): Path to the alignment being evaluated in this contact scoring
            prediction task or an already initialized SeqAlignment object.
            pdb_reference (str/path, PDBReference): The path to a PDB structure file, or an already initialized
            PDBReference object.
            chain (str): Which chain in the PDB structure to use for comparison and evaluation. If left blank the best
            chain will be identified by aligning the query sequence from seq_alignment against the chains in
            pdb_reference and the closest match will be selected.
        """
        self.query_pdb_mapper = SequencePDBMap(query=query, query_alignment=seq_alignment,
                                               query_structure=pdb_reference, chain=chain)
        self.data = None

    def fit(self):
        """
        Fit

        This function maps sequence positions between the query sequence from the alignment and residues in PDB file.
        If there are multiple chains for the structure in the PDB, and no chain was specified upon initialization, the
        one which matches the query sequence best (highest global alignment score) is used and recorded in the
        seq_pdb_mapper variable. This method updates the seq_pdb_mapper attribute.
        """
        if not self.query_pdb_mapper.is_aligned():
            start = time()
            self.query_pdb_mapper.align()
            end = time()
            print('Mapping query sequence and pdb took {} min'.format((end - start) / 60.0))
