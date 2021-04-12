"""
Created on Oct 23, 2019

@author: dmkonecki
"""
import os
import numpy as np
import pandas as pd
from time import time
from shutil import rmtree
from Bio.SeqIO import write
from evcouplings.utils import read_config_file, write_config_file
from evcouplings.utils.pipeline import execute
from dotenv import find_dotenv, load_dotenv
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)
from SupportingClasses.Predictor import Predictor
from SupportingClasses.utils import compute_rank_and_coverage


class EVCouplingsWrapper(Predictor):
    """
    This class is intended as a wrapper for the EVcouplings python code distributed through github for the following
    citation:
        https://github.com/debbiemarkslab/EVcouplings
        Marks D. S., Colwell, L. J., Sheridan, R., Hopf, T.A., Pagnani, A., Zecchina, R., Sander, C. Protein 3D
        structure computed from evolutionary sequence variation. PLOS ONE 6(12), e28766 (2011)
        Hopf T. A., Colwell, L. J., Sheridan, R., Rost, B., Sander, C., Marks, D. S. Three-dimensional structures of
        membrane proteins from genomic sequencing. Cell 149, 1607-1621 (2012)
        Marks, D. S., Hopf, T. A., Sander, C. Protein structure prediction from sequence variation. Nature Biotechnology
        30, 1072–1080 (2012)
        Hopf, T. A., Schärfe, C. P. I., Rodrigues, J. P. G. L. M., Green, A. G., Kohlbacher, O., Sander, C., Bonvin,
        A. M. J. J., Marks, D. S. Sequence co-evolution gives 3D contacts and structures of protein complexes. eLife Sep
        25;3 (2014)
        Hopf, T. A., Ingraham, J. B., Poelwijk, F.J., Schärfe, C.P.I., Springer, M., Sander, C., & Marks, D. S. (2017).
        Mutation effects predicted from sequence co-variation. Nature Biotechnology 35, 128–135 doi:10.1038/nbt.3769

    This wrapper makes it possible to  perform covariance analysis using the EV Couplings method on a fasta formatted
    alignment and to import the covariance scores from the analysis.

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
        method (str): 'EVCouplings' for all instances of this class, since this attribute describes how the covariance
        scores were computed.
        protocol (str): Which evcouplings pipeline protocol to use ('standard' or 'mean_field' expected).
        scores (np.array): The raw scores calculated for pair of position by EVCouplings.
        probability (np.array): The probability based on the raw score that two positions are covarying.
        coverages (np.array): The percentage of scores at or better than the score for this pair of position (i.e. the
        percentile rank).
        rankings (np.array): The rank (lowest being best, highest being worst) of each pair of positions in the provided
        alignment as determined from the calculated scores.
        time (float): The time (in seconds) required to complete the computation of covariance scores by EVCouplings.
    """

    def __init__(self, query, aln_file, protocol, out_dir='.'):
        """
        __init__

        The initialization function for the EVCouplingsWrapper class which draws on its parent (Predictor)
        initialization.

        Arguments:
            query (str): The sequence identifier for the sequence being analyzed.
            aln_file (str): The path to the alignment to analyze, the file is expected to be in fasta format.
            protocol (str): Which method ('standard' or 'mean_field') to apply when performing the EVCouplings
            computation.
            polymer_type (str): What kind of sequence information is being analyzed (.i.e. Protein or DNA).
            out_dir (str): The path where results of this analysis should be written to. If no path is provided the
            default will be to write results to the current working directory.
        """
        super().__init__(query, aln_file, 'Protein', out_dir)
        self.method = 'EVCouplings'
        if protocol != 'standard':
            raise ValueError("Currently only the 'standard' protocol is supported by this wrapper class.")
        self.protocol = protocol
        self.probability = None

    def configure_run(self, cores):
        """
        Configure Run

        This method reads in a template configuration file for monomer pipeline usage and then updates all relevant
        fields. This includes 'global', 'pipeline', 'batch', 'management', 'environment', 'tools', 'databases',
        'stages', 'align', and 'couplings' fields. Only the 'align' and 'couplings' stages are configured. The 'align'
        stage is set to use a pre-generated alignment file and all filters are set to be completely permissive so that
        the files are not edited at all before being used in the pipeline. The 'couplings' stage is configured for the
        specified protocol with default parameters found in the example configuration file at the time this wrapper was
        written.

        Args:
            cores (int): The number of processors available to perform this analysis.
        Returns:
            dict: Configurations data structure created by evcouplings package.
        """
        sample_config = os.environ.get('EVCOUPLINGS_CONFIG')
        config = read_config_file(sample_config)
        # Globals
        config["global"]["prefix"] = self.out_dir + ('/' if not self.out_dir.endswith('/') else '')
        config["global"]["sequence_id"] = self.non_gapped_aln.query_id
        seq_fn = os.path.join(self.out_dir, '{}_seq.fasta'.format(self.non_gapped_aln.query_id))
        with open(seq_fn, 'w') as seq_handle:
            seq = self.non_gapped_aln.alignment[self.non_gapped_aln.seq_order.index(self.non_gapped_aln.query_id)]
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
        # aln_fn = os.path.join(out_dir, '{}_aln.fasta'.format(self.alignment.query_id))
        # self.alignment.write_out_alignment(aln_fn)
        config["align"]["input_alignment"] = self.non_gapped_aln_fn
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
        if isinstance(self.non_gapped_aln.alphabet.letters, str):
            alphabet = self.non_gapped_aln.alphabet.letters
        elif isinstance(self.non_gapped_aln.alphabet.letters, list):
            alphabet = ''.join(self.non_gapped_aln.alphabet.letters)
        else:
            raise TypeError('Alphabet cannot be properly processed when letters are type: {}'.format(
                self.non_gapped_aln.alphabet.letters))
        final_alphabet = '-' + alphabet.replace('-', '')
        config["couplings"]["alphabet"] = final_alphabet
        config["couplings"]["ignore_gaps"] = False
        config["couplings"]["reuse_ecs"] = True
        config["couplings"]["min_sequence_distance"] = 0
        # Compare - skipping
        # Mutate - skipping
        # Fold - skipping
        write_config_file(os.path.join(self.out_dir, "{}_config.txt".format(self.non_gapped_aln.query_id)), config)
        return config

    def import_covariance_scores(self, out_path):
        """
        Import Covariance Scores

        This method imports the predicted covariation scores into numpy arrays with the shape defined by the length of
        the query sequence (the same format expected from Evolutionary Trace covariation scores, allowing for easier
        evaluation).

        Args:
            out_path (str): Path to the file from which the data should be read.
        """
        if not os.path.isfile(out_path):
            raise ValueError('Provided file does not exist: {}!'.format(out_path))
        data = pd.read_csv(out_path, sep=',', dtype={'i': np.int64, 'j': np.int64, 'cn': np.float64,
                                                     'probability': np.float64})
        self.scores = np.zeros((self.non_gapped_aln.seq_length, self.non_gapped_aln.seq_length))
        self.probability = np.zeros((self.non_gapped_aln.seq_length, self.non_gapped_aln.seq_length))
        for ind in data.index:
            i = data.loc[ind, 'i'] - 1
            j = data.loc[ind, 'j'] - 1
            self.scores[i, j] = self.scores[j, i] = data.loc[ind, 'cn']
            self.probability[i, j] = self.probability[j, i] = data.loc[ind, 'probability']

    def calculate_scores(self, cores=1, delete_files=True):
        """
        Calculate Scores

        This function calls the specified EVcouplings pipeline for the provided protein alignment. This is achieved by
        creating a configuration which passes the alignment through unedited and then executing the desired coupling
        pipeline for that alignment.

        Args:
            cores (int): The number of processors available to perform this analysis.
            delete_files (bool): Whether or not to delete the data files generated by the EVCouplings pipeline
            (intermediate files created by the wrapper and the serialization file will be kept).
        Returns:
            float: The time in seconds required to calculate the EVcouplings covariation scores for the provided protein
            alignment.
        """
        serialized_path = os.path.join(self.out_dir, 'EVCouplings.npz')
        if os.path.isfile(serialized_path):
            loaded_data = np.load(serialized_path)
            self.scores = loaded_data['scores']
            self.probability = loaded_data['probability']
            self.coverages = loaded_data['coverages']
            self.rankings = loaded_data['rankings']
            self.time = loaded_data['time']
        else:
            config = self.configure_run(cores=cores)
            out_path = os.path.join(self.out_dir, 'couplings', '_CouplingScores.csv')
            start = time()
            # Call EVCouplings pipeline
            if self.protocol == 'mean_field':
                print('SETTINGS!')
                print(config)
            execute(**config)
            end = time()
            self.time = end - start
            self.import_covariance_scores(out_path=out_path)
            self.rankings, self.coverages = compute_rank_and_coverage(seq_length=self.non_gapped_aln.seq_length,
                                                                      scores=self.scores, pos_size=2, rank_type='max')
            if delete_files:
                rmtree(os.path.join(self.out_dir, 'couplings'))
                rmtree(os.path.join(self.out_dir, 'align'))
                os.remove(os.path.join(self.out_dir, '_final.outcfg'))
            np.savez(serialized_path, scores=self.scores, probability=self.probability, coverages=self.coverages,
                     rankings=self.rankings, time=self.time)
        print(self.time)
        return self.time
