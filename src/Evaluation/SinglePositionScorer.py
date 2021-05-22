"""
Created on May 22, 2021

@author: dmkonecki
"""
from Evaluation.Scorer import Scorer


class SinglePositionScorer(Scorer):
    """
    SinglePositionScorer

    This class is meant to abstract the process of scoring a set of residue importance predictions.  This is being done
    for two main reasons. First, residue importance predictions are being made with several different methods and so
    their scoring should be performed consistently by another object or function. Second, this class will support
    several scoring methods such as the hypergoemtric test scoring of overlap with key sites as well as methods internal
    to the lab like the Selection Cluster Weighting Z-score derived from prior ET work.

    Attributes:
        query_pdb_mapper (SequencePDBMap): A mapping from the index of the query sequence to the index of the
        pdb chain's sequence for those positions which match according to a pairwise global alignment and vice versa.
        data (panda.DataFrame): A data frame containing a mapping of the sequence, structure, and predictions as well as
        labels to indicate which sequence separation category and top prediction category specific scores belong to.
        This is used for most evaluation methods.
    """

    def __init__(self, query, seq_alignment, pdb_reference, chain=None):
        """
        __init__

        This function initializes a new SinglePositionScorer, it accepts paths to an alignment and if available a pdb
        file to be used in the assessment of residue importance predictions for a given query.

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
        super().__init__(query, seq_alignment, pdb_reference, chain)