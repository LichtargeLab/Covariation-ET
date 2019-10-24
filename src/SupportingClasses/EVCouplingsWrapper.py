"""
Created on Oct 23, 2019

@author: dmkonecki
"""
import os
import numpy as np
import pandas as pd
from time import time
from Bio.SeqIO import write
from evcouplings.utils import read_config_file, write_config_file
from evcouplings.utils.pipeline import execute
from dotenv import find_dotenv, load_dotenv
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)


class EVCouplingsWrapper(object):

    def __init__(self, alignment, protocol):
        self.protocol = protocol
        self.alignment = alignment.remove_gaps()
        self.scores = None
        self.probability = None
        self.time = None

    def import_covariance_scores(self, out_path):
        if not os.path.isfile(out_path):
            raise ValueError('Provided file does not exist: {}!'.format(out_path))
        data = pd.read_csv(out_path, sep=',')
        self.scores = np.zeros((self.alignment.seq_length, self.alignment.seq_length))
        self.probability = np.zeros((self.alignment.seq_length, self.alignment.seq_length))
        for ind in data.index:
            i = data.loc[ind, 'i'] - 1
            j = data.loc[ind, 'j'] - 1
            self.scores[i, j] = self.scores[j, i] = data.loc[ind, 'cn']
            self.probability[i, j] = self.scores[j, i] = data.loc[ind, 'probability']

    def configure_run(self, out_dir, cores):
        sample_config = os.environ.get('EVCOUPLINGS_CONFIG')
        config = read_config_file(sample_config)
        # Globals
        config["global"]["prefix"] = out_dir + ('/' if not out_dir.endswith('/') else '')
        config["global"]["sequence_id"] = self.alignment.query_id
        seq_fn = os.path.join(out_dir, '{}_seq.fasta'.format(self.alignment.query_id))
        with open(seq_fn, 'w') as seq_handle:
            seq = self.alignment.alignment[self.alignment.seq_order.index(self.alignment.query_id)]
            write(seq, seq_handle, 'fasta')
        config["global"]["sequence_file"] = seq_fn
        config["global"]["region"] = None
        config["global"]["theta"] = 0.8
        config["global"]["cpu"] = cores
        # Pipeline
        config["pipeline"] = 'protein_monomer'
        # Batch
        config["batch"] = None
        # Management
        config["management"]["database_uri"] = None
        config["management"]["job_name"] = None
        config["management"]["archive"] = None
        # Environment
        config["environment"]["engine"] = 'local'
        config["environment"]["queue"] = None
        config["environment"]["cores"] = cores
        config["environment"]["memory"] = None
        config["environment"]["time"] = None
        config["environment"]["configuration"] = None
        # Tools
        config["tools"]["plmc"] = os.environ.get('PLMC')
        config["tools"]["jackhmmer"] = os.environ.get('JACKHMMER')
        config["tools"]["hmmbuild"] = os.environ.get('HMMBUILD')
        config["tools"]["hmmsearch"] = os.environ.get('HMMSEARCH')
        config["tools"]["hhfilter"] = None
        config["tools"]["psipred"] = None
        config["tools"]["cns"] = None
        config["tools"]["maxcluster"] = None
        # Databases
        config["databases"]["uniprot"] = os.environ.get('UNIPROT')
        config["databases"]["uniref100"] = os.environ.get('UNIREF100')
        config["databases"]["uniref90"] = os.environ.get('UNIREF90')
        config["databases"]["sequence_download_url"] = None
        config["databases"]["pdb_mmtf_dir"] = None
        config["databases"]["sifts_mapping_table"] = os.environ.get('SIFTS_MAPPING_TABLE')
        config["databases"]["sifts_sequence_db"] = os.environ.get('SIFTS_SEQUENCE_DB')
        # Stages
        config["stages"] = ['align', 'couplings']
        # Align - only using the
        config["align"]["protocol"] = 'existing'
        aln_fn = os.path.join(out_dir, '{}_aln.fasta'.format(self.alignment.query_id))
        self.alignment.write_out_alignment(aln_fn)
        config["align"]["input_alignment"] = aln_fn
        config["align"]["first_index"] = 1
        config["align"]["compute_num_effective_seqs"] = False
        config["align"]["seqid_filter"] = None
        config["align"]["minimum_sequence_coverage"] = 0
        config["align"]["minimum_column_coverage"] = 0
        config["align"]["extract_annotation"] = False
        # Couplings
        if self.protocol == 'standard':
            config["couplings"]["protocol"] = 'standard'
            config["couplings"]["iterations"] = 100
            config["couplings"]["lambda_J"] = 0.01
            config["couplings"]["lambda_J_times_Lq"] = True
            config["couplings"]["lambda_h"] = 0.01
            config["couplings"]["lambda_group"] = None
            config["couplings"]["scale_clusters"] = None
        elif self.protocol == 'mean_field':
            config["couplings"]["protocol"] = 'mean_field'
            config["couplings"]["pseudo_count"] = 0.5
        else:
            raise AttributeError('Unrecognized EVCouplings Protocol')
        if isinstance(self.alignment.alphabet.letters, str):
            alphabet = self.alignment.alphabet.letters
        elif isinstance(self.alignment.alphabet.letters, list):
            alphabet = ''.join(self.alignment.alphabet.letters)
        else:
            raise TypeError('Alphabet cannot be properly processed when letters are type: {}'.format(
                self.alignment.alphabet.letters))
        final_alphabet = '-' + alphabet.replace('-', '')
        config["couplings"]["alphabet"] = final_alphabet
        config["couplings"]["ignore_gaps"] = False
        config["couplings"]["reuse_ecs"] = True
        config["couplings"]["min_sequence_distance"] = 0
        # Compare - skipping
        # Mutate - skipping
        # Fold - skipping
        write_config_file(os.path.join(out_dir, "{}_config.txt".format(self.alignment.query_id)), config)
        return config

    def calculate_scores(self, out_dir, cores=1, delete_file=True):
        serialized_path = os.path.join(out_dir, 'EVCouplings.npz')
        if os.path.isfile(serialized_path):
            loaded_data = np.load(serialized_path)
            self.scores = loaded_data['scores']
            self.time = loaded_data['time']
        else:
            config = self.configure_run(out_dir=out_dir, cores=cores)
            out_path = os.path.join(out_dir, 'couplings', '_CouplingScores.csv')
            start = time()
            # Call EVCouplings pipeline
            execute(**config)
            end = time()
            self.time = end - start
            self.import_covariance_scores(out_path=out_path)
            if delete_file:
                os.remove(out_path)
            np.savez(serialized_path, scores=self.scores, time=self.time)
        print(self.time)
        return self.time
