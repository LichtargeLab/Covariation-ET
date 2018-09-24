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
# dotenv_path = find_dotenv(raise_error_if_not_found=True)
# print(dotenv_path)
# load_dotenv(dotenv_path)


class DCAWrapper(object):
    """
    This class is intended as a wrapper for the DCA julia code distributed through github for the following citation:
        (GITHUB)
        (REFERENCE)

    This wrapper makes it possible to  perform covariance analysis using the DCA method on a fasta formatted alignment
    and to import the covariance scores from the analysis.
    """

    def __init__(self, alignment):
        """
        This method instantiates an instance of the DCAWrapper class. It takes in an alignment and instantiates the
        following variables:
            alignment (SeqAlignment): This variable holds a seq alignment object. This is mainly used to provide the
            path the to the alignment used, but is also used to determine the length of the query sequence. Because of
            this it is advised that the import_alignment() and remove_gaps() methods be run before the SeqAlignment
            instance is passed to ETMIPWrapper.
            dca_scores (numpy.array): A square matrix whose length on either axis is the query sequence length (without
            gaps). This matrix contains the covariance scores computed by the DCA method.
            time (float): The number of seconds it took for the DCA code to load the alignment file and calculate
            covariance scores.

        Args:
            alignment (SeqAlignment): This variable holds a seq alignment object. This is mainly used to provide the
            path the to the alignment used, but is also used to determine the length of the query sequence. Because of
            this it is advised that the import_alignment() and remove_gaps() methods be run before the SeqAlignment
            instance is passed to ETMIPWrapper.
        """
        self.alignment = alignment
        self.dca_scores = None
        self.time = None

    def import_covariance_scores(self, out_path):
        """
        Import Covaraince Scores

        This method looks for the specified file where the DCA scores were written. This file is then imported using
        Pandas and is then used to fill in the dca_scores matrix.

        Args:
            out_path (str): The path to the file where the DCA scores have been written.
        Raises:
            ValueError: If the directory does not exist, or the expected file is not found in that directory.
        """
        if not os.path.isfile(out_path):
            raise ValueError('Provided file does not exist: {}!'.format(out_path))
        data = pd.read_csv(out_path, header=None, sep='\s+', names=['Res_i', 'Res_j' , 'Scores'])
        self.dca_scores = np.zeros((self.alignment.seq_length, self.alignment.seq_length))
        for ind in data.index:
            i = data.loc[ind, 'Res_i'] - 1
            j = data.loc[ind, 'Res_j'] - 1
            self.dca_scores[i, j] = self.dca_scores[j, i] = data.loc[ind, 'Scores']

    def calculate_dca_scores(self, out_path, delete_file=True):
        """
        Calculate DCA Scores

        This method uses the DCA julia code to compute covariance scores on a fasta formatted multiple sequence
        alignment. The code requires a .env at the project level which has a variable 'PROJECT_PATH' that describes the
        location of the project so that the julia code to run DCA can be called.. The method makes use of
        import_covariance_scores() to load the data produced by the run.

        Args:
            out_path (str): The path to the file where the DCA scores should be written.
            delete_file (boolean): If True the file written out by calling this method will be deleted after importing
            the relevant data, if False the file will be left at the specified out_path.
        Returns:
            float. Time in seconds.
        Raises:
            ValueError: If the file does not exist, or if the file expected to be created by this method is not
            found in that directory.
        """
        julia_path = os.path.join(os.environ.get('PROJECT_PATH'), 'src', 'SupportingClasses', 'cmd_line_GausDCA.jl')
        start = time()
        # Call julia code
        p = Popen(['julia', julia_path, self.alignment.file_name, out_path], stdout=PIPE, stderr=PIPE)
        # Retrieve communications from julia call
        out, error = p.communicate()
        end = time()
        self.time = end - start
        print('Output:')
        print(out)
        print('Error:')
        print(error)
        print(self.time)
        self.import_covariance_scores(out_path=out_path)
        if delete_file:
            os.remove(out_path)
        return self.time
