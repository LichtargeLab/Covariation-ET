"""
Created on February 5, 2020

@author: Daniel Konecki
"""
import os
from SeqAlignment import SeqAlignment


class Predictor(object):
    """
    This class is intended as the base for any predictor meant to perform single position residue importance predictions
    or paired position covariation predictions. It establishes a minimum set of attributes and the corresponding
    initializer as well as a method signature for the only other function all Predictor sub-classes should have, namely
    calculate_scores.

    Attributes:
        out_dir (str): The path where results of this analysis should be written to.
        query (str): The sequence identifier for the sequence being analyzed.
        original_aln (SeqAlignment): A SeqAlignment object representing the alignment originally passed in.
        original_aln_fn (str): The path to the alignment to analyze.
        non_gapped_aln (SeqAlignment): SeqAlignment object representing the original alignment with all columns which
        are gaps in the query sequence removed.
        non_gapped_aln_fn (str): Path to where the non-gapped alignment is written.
        scores (np.array): The raw scores calculated for each single or paired position in the provided alignment.
        coverages (np.array): The percentage of scores at or better than the score for this single or paired position
        (i.e. the percentile rank).
        rankings (np.array): The rank (lowest being best, highest being worst) of each single or paired position in the
        provided alignment as determined from the calculated scores.
    """
    def __init__(self, query, aln_file, out_dir='.'):
        """
        __init__

        The base initialization function for all Predictor sub-classes.

        Arguments:
            query (str): The sequence identifier for the sequence being analyzed.
            aln_file (str): The path to the alignment to analyze, the file is expected to be in fasta format.
            out_dir (str): The path where results of this analysis should be written to. If no path is provided the
            default will be to write results to the current working directory.
        """
        out_dir = os.path.abspath(out_dir)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.query = query
        self.original_aln = SeqAlignment(query_id=query, file_name=aln_file)
        self.original_aln.import_alignment()
        self.original_aln_fn = os.path.join(out_dir, 'Original_Alignment.fa')
        self.original_aln.write_out_alignment(self.original_aln_fn)
        self.non_gapped_aln = self.original_aln.remove_gaps()
        self.non_gapped_aln_fn = os.path.join(out_dir, 'Non-Gapped_Alignment.fa')
        self.non_gapped_aln.write_out_alignment(self.non_gapped_aln_fn)
        self.method = 'Base'
        self.scores = None
        self.coverages = None
        self.rankings = None
        self.time = None

    def calculate_scores(self):
        """
        Calculate Scores

        This method should be used to have a specific predictor perform its calculations and generate its predictions
        for a given alignment. This method should update the scores, coverages, rankings, and times attributes.

        Returns:
            float: The time in seconds it took to calculate scores using a given predictor, this includes time to import
            the scores (if written to file and not passed back by a method) and to compute rankings/coverage for a given
            prediction.
        """
        raise NotImplementedError('Calculate scores needs to be implemented for this Predictor class!')
