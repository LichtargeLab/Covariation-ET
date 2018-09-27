"""
Created on Sep 17, 2018

@author: dmkonecki
"""
import os
import numpy as np
import pandas as pd
from time import time
from subprocess import Popen, PIPE
from Bio.Align.Applications import MuscleCommandline
from dotenv import find_dotenv, load_dotenv
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)
# dotenv_path = find_dotenv(raise_error_if_not_found=True)
# print(dotenv_path)
# load_dotenv(dotenv_path)


class ETMIPWrapper(object):
    """
    This class is intended as a wrapper for the original ET-MIp binary created by Angela Wilson for the publication:
        Sung Y-M, Wilkins AD, Rodriguez GJ, Wensel TG, Lichtarge O. Intramolecular allosteric communication in dopamine
        D2 receptor revealed by evolutionary amino acid covariation. Proceedings of the National Academy of Sciences of
        the United States of America. 2016;113(13):3539-3544. doi:10.1073/pnas.1516579113.
    It requires the use of a .env file at the project level, which describes the location of the WETC binary as well as
    the muscle alignment binary.

    This wrapper makes it possible to convert alignments (in fa format) to msf format, perform covariance analysis using
    the ET-MIp method on an msf formatted alignment, and to import the covariance raw and coverage scores from the
    analysis.
    """

    def __init__(self, alignment):
        """
        This method instantiates an instance of the ETMIPWrapper class. It takes in an alignment and instantiates the
        following variables:
            alignment (SeqAlignment): This variable holds a seq alignment object. This is mainly used to provide the
            path the to the alignment used, but is also used to determine the length of the query sequence. Because of
            this it is advised that the import_alignment() and remove_gaps() methods be run before the SeqAlignment
            instance is passed to ETMIPWrapper.
            msf_path (str): The file path to the location at which the msf formatted alignment to be analyzed is
            located. If the SeqAlignment object was created using an msf formatted alignments its filename variable will
            be the same as msf_path, other wise a different path will be found here.
            raw_scores (numpy.array): A square matrix whose length on either axis is the query sequence length (without
            gaps). This matrix contains the raw covariance scores computed by the ET-MIp method.
            coverage_scores (numpy.array): A square matrix whose length on either axis is the query sequence length
            (without gaps). This matrix contains the processed coverage covariance scores computed by the ET-MIp method.
            time (float): The number of seconds it took for the ET-MIp code to load the msf formatted file and calculate
            covariance scores.

        Args:
            alignment (SeqAlignment): This variable holds a seq alignment object. This is mainly used to provide the
            path the to the alignment used, but is also used to determine the length of the query sequence. Because of
            this it is advised that the import_alignment() and remove_gaps() methods be run before the SeqAlignment
            instance is passed to ETMIPWrapper.
        """
        self.alignment = alignment
        self.msf_path = None
        self.scores = None
        # self.raw_scores = None
        self.coverage_scores = None
        self.time = None

    def check_alignment(self):
        """
        Check Alignment

        This method checks whether the alignment currently associated with the ETMIPWrapper object ends with '.msf'. If
        it does not the muscle tool is used to convert the initial alignment (assumed to be in .fa format) to an .msf
        formatted alignment at the same location as the .fa alignment. The file path to this .msf alignment is stored in
        the msf_path variable. This function is dependent on the .env file at the project level containing the
        'MUSCLE_PATH' variable describing the path to the muscle binary.
        """
        if not self.alignment.file_name.endswith('.msf'):
            muscle_path = os.environ.get('MUSCLE_PATH')
            target_dir = os.path.dirname(self.alignment.file_name)
            old_file_name = os.path.basename(self.alignment.file_name)
            new_file_name = os.path.join(target_dir, '{}.msf'.format(old_file_name.split('.')[0]))
            if not os.path.isfile(new_file_name):
                c_line = MuscleCommandline(muscle_path, input=self.alignment.file_name, out=new_file_name, msf=True)
                c_line()
            self.msf_path = new_file_name
        else:
            self.msf_path = self.alignment.file_name

    def import_covariance_scores(self, out_dir):
        """
        Import Covaraince Scores

        This method looks for the etc_out.tree_mip_sorted file in the directory where the ET-MIp scores were calculated
        and written to file. This file is then imported using Pandas and is then used to fill in the scores and
        coverage_scores matrices.

        Args:
            out_dir (str): The path to the directory where the ET-MIp scores have been written.
        Raises:
            ValueError: If the directory does not exist, or the expected file is not found in that directory.
        """
        if not os.path.isdir(out_dir):
            raise ValueError('Provided directory does not exist: {}!'.format(out_dir))
        file_path = os.path.join(out_dir, 'etc_out.tree_mip_sorted')
        if not os.path.isfile(file_path):
            raise ValueError('Provided directory does not contain expected covariance file!')
        data = pd.read_csv(file_path, comment='%', sep='\s+', names=['Sort', 'Res_i', 'i(AA)', 'Res_j', 'j(AA)',
                                                                     'Raw_Scores', 'Coverage_Scores', 'Interface',
                                                                     'Contact', 'Number', 'Average_Contact'])
        self.scores = np.zeros((self.alignment.seq_length, self.alignment.seq_length))
        self.coverage_scores = np.zeros((self.alignment.seq_length, self.alignment.seq_length))
        for ind in data.index:
            i = data.loc[ind, 'Res_i'] - 1
            j = data.loc[ind, 'Res_j'] - 1
            self.scores[i, j] = self.scores[j, i] = data.loc[ind, 'Raw_Scores']
            self.coverage_scores[i, j] = self.coverage_scores[j, i] = data.loc[ind, 'Coverage_Scores']

    def calculate_scores(self, out_dir, delete_files=True):
        """
        Calculated ET-MIp Scores

        This method uses the ET-MIp binary to compute covariance scores on an msf formatted multiple sequence alignment.
        The code requires a .env at the project level which has a variable 'WEC_PATH' that describes the location of the
        WETC binary. The method makes use of import_covariance_scores() to load the data produced by the run.

        Args:
            out_dir (str): The path to the directory where the ET-MIp scores should be written.
            delete_files (boolean): If True all of the files written out by calling this method will be deleted after
            importing the relevant data, if False all files will be left in the specified out_dir.
        Raises:
            ValueError: If the directory does not exist, or if the file expected to be created by this method is not
            found in that directory.
        """
        # from IPython import embed
        # embed()
        self.check_alignment()
        # embed()
        # exit()
        binary_path = os.environ.get('WETC_PATH')
        start = time()
        current_dir = os.getcwd()
        os.chdir(out_dir)
        # Call binary
        p = Popen([binary_path, '-p', self.msf_path, '-x', self.alignment.query_id[1:], '-allpairs'], stdout=PIPE,
                  stderr=PIPE)
        # Retrieve communications from binary call
        out, error = p.communicate()
        end = time()
        self.time = end - start
        print('Output:')
        print(out)
        print('Error:')
        print(error)
        print(self.time)
        os.chdir(current_dir)
        self.import_covariance_scores(out_dir=out_dir)
        if delete_files:
            self.remove_ouptut(out_dir=out_dir)

    @staticmethod
    def remove_ouptut(out_dir):
        """
        Remove Output

        This method will take the directory where ET-MIp output has been written and remove all of the files which are
        generated by the code.

        Args:
            out_dir (str): The path to the directory where the ET-MIp scores have been written.
        Raises:
            ValueError: If the directory does not exist.
        """
        prefix = 'etc_out'
        suffixes = ['aa_freqs', 'allpair_ranks', 'allpair_ranks_sorted', 'auc', 'average_ranks_sorted',
                    'covariation_matrix', 'entro.heatmap', 'entroMI.heatmap', 'mip_sorted', 'MI_sorted', 'nhx',
                    'pairs_allpair_ranks_sorted', 'pairs_average_ranks_sorted', 'pairs_mip_sorted',
                    'pairs_tree_mip_sorted', 'pairs_MI_sorted', 'pairs_tre_mip_sorted', 'pss.nhx', 'rank_matrix',
                    'ranks', 'ranks_sorted', 'rv.heatmap', 'rvMI.heatmap', 'tree_mip_matrix', 'tree_mip_sorted',
                    'tree_mip_top40_matrix']
        if not os.path.isdir(out_dir):
            raise ValueError('Provided directory does not exist: {}!'.format(out_dir))
        for suffix in suffixes:
            curr_path = os.path.join(out_dir, '{}.{}'.format(prefix, suffix))
            if os.path.isfile(curr_path):
                os.remove(curr_path)