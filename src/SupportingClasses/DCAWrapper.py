"""
Created on Sep 20, 2018

@author: dmkonecki
"""
import os
import numpy as np
import pandas as pd
from time import time
from subprocess import Popen, PIPE
from dotenv import find_dotenv, load_dotenv
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)
from SupportingClasses.Predictor import Predictor
from SupportingClasses.utils import compute_rank_and_coverage


class DCAWrapper(Predictor):
    """
    This class is intended as a wrapper for the DCA julia code distributed through github for the following citation:
        https://github.com/carlobaldassi/GaussDCA.jl
        "Fast and accurate multivariate Gaussian modeling of protein families: Predicting residue contacts and protein-
        interaction partners" by Carlo Baldassi, Marco Zamparo, Christoph Feinauer, Andrea Procaccini, Riccardo
        Zecchina, Martin Weigt and Andrea Pagnani, (2014) PLoS ONE 9(3): e92721. doi:10.1371/journal.pone.0092721

    This wrapper makes it possible to  perform covariance analysis using the DCA method on a fasta formatted alignment
    and to import the covariance scores from the analysis.

    This class inherits from the Predictor class which means it shares the attributes and initializer of that class,
    as well as implementing the calculate_scores method.

    Attributes:
        out_dir (str): The path where results of this analysis should be written to.
        query (str): The sequence identifier for the sequence being analyzed.
        original_aln (SeqAlignment): A SeqAlignment object representing the alignment originally passed in.
        original_aln_fn (str): The path to the alignment to analyze.
        non_gapped_aln (SeqAlignment): SeqAlignment object representing the original alignment with all columns which
        are gaps in the query sequence removed.
        non_gapped_aln_fn (str): Path to where the non-gapped alignment is written.
        method (str): 'DCA' for all instances of this class, since the attribute describes how the covariance scores
        were computed.
        scores (np.array): A square matrix whose length on either axis is the query sequence length (without gaps). This
        matrix contains the covariance scores computed by the DCA method.
        coverages (np.array): The percentage of scores at or better than the score for this pair of positions (i.e. the
        percentile rank).
        rankings (np.array): The rank (lowest being best, highest being worst) of each pair of positions in the
        provided alignment as determined from the calculated scores.
        time (float): The number of seconds it took for the DCA code to load the alignment file and calculate
        covariance scores.
    """

    def __init__(self, query, aln_file, out_dir='.', polymer_type='Protein'):
        """
        __init__

        The initialization function for the DCAWrapper class which draws on its parent (Predictor) initialization.

        Arguments:
            query (str): The sequence identifier for the sequence being analyzed.
            aln_file (str): The path to the alignment to analyze, the file is expected to be in fasta format.
            out_dir (str): The path where results of this analysis should be written to. If no path is provided the
            default will be to write results to the current working directory.
            polymer_type (str): What kind of sequence information is being analyzed (.i.e. Protein or DNA).
        """
        super().__init__(query, aln_file, polymer_type, out_dir)
        self.method = 'DCA'

    def import_covariance_scores(self, out_path):
        """
        Import Covariance Scores

        This method looks for the specified file where the DCA scores were written. This file is then imported using
        Pandas and is then used to fill in the scores matrix.

        Args:
            out_path (str): The path to the file where the DCA scores have been written.
        Raises:
            ValueError: If the directory does not exist, or the expected file is not found in that directory.
        """
        if not os.path.isfile(out_path):
            raise ValueError('Provided file does not exist: {}!'.format(out_path))

        print('OUTPATH:')
        print(out_path)
        data = pd.read_csv(out_path, header=None, sep='\s+', names=['Res_i', 'Res_j', 'Scores'])
        self.scores = np.zeros((self.non_gapped_aln.seq_length, self.non_gapped_aln.seq_length))
        for ind in data.index:
            i = data.loc[ind, 'Res_i'] - 1
            j = data.loc[ind, 'Res_j'] - 1
            self.scores[i, j] = self.scores[j, i] = data.loc[ind, 'Scores']

    def calculate_scores(self, delete_file=True):
        """
        Calculate DCA Scores

        This method uses the DCA julia code to compute covariance scores on a fasta formatted multiple sequence
        alignment. The code requires a .env at the project level which has a variable 'PROJECT_PATH' that describes the
        location of the project so that the julia code to run DCA can be called. The method makes use of
        import_covariance_scores() to load the data produced by the run.

        Args:
            delete_file (boolean): If True the file written out by calling this method will be deleted after importing
            the relevant data, if False the file will be left at the specified out_path.
        Returns:
            float. Time in seconds.
        Raises:
            ValueError: If the file does not exist, or if the file expected to be created by this method is not
            found in that directory.
        """
        serialized_path = os.path.join(self.out_dir, 'DCA.npz')
        if os.path.isfile(serialized_path):
            loaded_data = np.load(serialized_path)
            self.scores = loaded_data['scores']
            self.coverages = loaded_data['coverages']
            self.rankings = loaded_data['rankings']
            self.time = loaded_data['time']
        else:
            julia_path = os.path.join(os.environ.get('PROJECT_PATH'), 'src', 'SupportingClasses', 'cmd_line_GausDCA.jl')
            out_path = os.path.join(self.out_dir, 'DCA_predictions.tsv')
            start = time()
            # Call julia code
            p = Popen(['julia', julia_path, self.non_gapped_aln_fn, out_path], stdout=PIPE, stderr=PIPE)
            # Retrieve communications from julia call
            out, error = p.communicate()
            end = time()
            self.time = end - start
            print('Output:')
            print(out.decode())
            print('Error:')
            print(error.decode())
            self.import_covariance_scores(out_path=out_path)
            self.rankings, self.coverages = compute_rank_and_coverage(seq_length=self.non_gapped_aln.seq_length,
                                                                      scores=self.scores, pos_size=2, rank_type='max')
            if delete_file:
                os.remove(out_path)
            np.savez(serialized_path, scores=self.scores, coverages=self.coverages, rankings=self.rankings,
                     time=self.time)
        print(self.time)
        return self.time
