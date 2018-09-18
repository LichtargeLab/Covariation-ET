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
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

class ETMIPWrapper(object):

    def __init__(self, alignment):
        self.alignment = alignment
        self.msf_path = None
        self.raw_scores = None
        self.coverage_scores = None
        self.time = None

    def check_alignment(self):
        if not self.alignment.file_name.endswith('.msf'):
            muscle_path = os.environ.get('MUSCLE_PATH')
            target_dir = os.path.dirname(self.alignment.file_name)
            old_file_name = os.path.basename(self.alignment.file_name)
            new_file_name = os.path.join(target_dir, '{}.msf'.format(old_file_name.split('.')[0]))
            c_line = MuscleCommandline(muscle_path, input=self.alignment.file_name, output=new_file_name, msf=True)
            c_line()
            self.msf_path = new_file_name
        else:
            self.msf_path = self.alignment.file_name

    def import_covariance_scores(self, out_dir):
        if not os.path.exists(out_dir):
            raise ValueError('Provided directory does not exist: {}!'.format(out_dir))
        file_path = os.path.join(out_dir, 'etc_out.tree_mip_matrix')
        if not os.path.exists(file_path):
            raise ValueError('Provided directory does not contain expected covariance file!')
        data = pd.read_csv(file_path, comment='%', sep='\s+', names=['Sort', 'Res_i', 'i(AA)', 'Res_j', 'j(AA)',
                                                                     'Raw_Scores', 'Coverage_Scores', 'Interface',
                                                                     'Contact', 'Number', 'Average_Contact'])
        self.raw_scores = self.coverage_scores = np.zeros((self.alignment.seq_length, self.alignment.seq_length))
        for ind in data.index:
            i = data.loc[ind, 'Res_i'] - 1
            j = data.loc[ind. 'Res_j'] - 1
            self.raw_scores[i, j] = self.raw_scores[j, i] = data.loc[ind, 'Raw_Scores']
            self.coverage_scores[i, j] = self.coverage_scores[j, i] = data.loc[ind, 'Coverage_Scores']


    def calculate_mip_scores(self, out_dir):
        binary_path = os.environ.get('WETC_PATH')
        start = time()
        current_dir = os.getcwd()
        os.chdir(out_dir)
        # Call binary
        p = Popen([binary_path, '-p', self.msf_path, '-x', self.alignment.query_id[1:], '-allpairs'], stdout=PIPE,
                  stderr=PIPE)
        # Retrieve communications from binary call
        out, error = p.communicate()
        print('Output:')
        print(out)
        print('Error:')
        print(error)
        end = time()
        os.chdir(current_dir)
        self.time = end - start
        self.import_covariance_scores(out_dir=out_dir)