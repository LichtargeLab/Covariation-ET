"""
Created on June 16, 2019

@author: Daniel Konecki
"""
import os

import numpy as np
from scipy.stats import rankdata
from Bio.Alphabet import Alphabet, Gapped
import shutil

# Common gap characters
gap_characters = {'-', '.', '*'}

def remove_sequences_with_ambiguous_characters(fp, out_dir, additional_chars = []):
    alphabet = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','-']
    alphabet.extend(additional_chars)
    aln = open(fp, 'r')
    Lines = aln.readlines()
    aln.close()
    aln = {}
    removed = []
    
    for line in Lines:
        if '>' in line:
            seqid = line.rstrip('\n') 
            aln[seqid] = ''
        else:
            aln[seqid] = aln[seqid] + line.rstrip('\n') 
            
    for seqid in list(aln):
        if not all(c in alphabet for c in aln[seqid]):
            aln.pop(seqid)
            removed.append(seqid)
            
    if '/' in fp:
        fn = fp.split('/')[-1]
    on = '_filtered.'.join(fn.split('.'))
    op = f'{out_dir}/{on}'
    
    with open(op, 'w') as handle:
        for key, value in aln.items():
            handle.write(key)
            handle.write('\n')
            handle.write(value)
            handle.write('\n')
     print(f'{len(removed)} sequences removed for containing ambiguous characters')
     return op


def build_mapping(alphabet, skip_letters=None):
    """
    Build Mapping

    Constructs a dictionary mapping characters in the given alphabet to their position in the alphabet (which
    may correspond to their index in substitution matrices). It also maps gap characters and skip letters to positions
    outside of that range.

    Args:
        alphabet (Bio.Alphabet.Alphabet,list,str): An alphabet object with the letters that should be mapped, or a list
        or string containing all of the letters which should be mapped.
        skip_letters (list): Which characters to skip when scoring sequences in the alignment.
    Returns:
        int: The size of the alphabet (not including gaps or skip letters) represented by this map.
        set: The gap characters in this map (if there are gap characters in the provided alphabet then these will not be
        present in the returned set).
        dict: Dictionary mapping a character to a number corresponding to its position in the alphabet and/or in the
        scoring/substitution matrix.
        np.array: Array mapping a number to character such that the character can be decoded from a position in an
        array or table built based on the alphabet.
    """
    if isinstance(alphabet, Alphabet) or isinstance(alphabet, Gapped):
        letters = alphabet.letters
        character_size = alphabet.size
    elif type(alphabet) == list:
        letters = alphabet
        character_size = len(alphabet[0])
    elif type(alphabet) == str:
        letters = list(alphabet)
        character_size = 1
    else:
        raise ValueError("'alphabet' expects values of type Bio.Alphabet, list, or str.")
    if skip_letters:
        letters = [letter for letter in letters if letter not in skip_letters]
    alphabet_size = len(letters)
    alpha_map = {char: i for i, char in enumerate(letters)}
    curr_gaps = {g * character_size for g in gap_characters}
    if skip_letters:
        for sl in skip_letters:
            if len(sl) != character_size:
                raise ValueError(f'skip_letters contained a character {sl} which did not match the alphabet character '
                                 f'size: {character_size}')
        skip_map = {char: alphabet_size + 1 for char in skip_letters}
        alpha_map.update(skip_map)
        curr_gaps = curr_gaps - set(skip_letters)
    curr_gaps = curr_gaps - set(letters)
    gap_map = {char: alphabet_size for char in curr_gaps}
    alpha_map.update(gap_map)
    reverse_map = np.array(list(letters))
    return alphabet_size, curr_gaps, alpha_map, reverse_map


def convert_seq_to_numeric(seq, mapping):
    """
    Convert Seq To Numeric

    This function uses an alphabet mapping (see build_mapping) to convert a sequence to a 1D array of integers.

    Args:
        seq (Bio.Seq|Bio.SeqRecord|str): A protein or DNA sequence.
        mapping (dict): A dictionary mapping a character to its position in the alphabet (can be produced using
        build_mapping).
    Return:
        numpy.array: A 1D array containing the numerical representation of the passed in sequence.
    """
    numeric = [mapping[char] for char in seq]
    return np.array(numeric)


def compute_rank_and_coverage(seq_length, scores, pos_size, rank_type):
    """
    Compute Rank and Coverage

    This function generates rank and coverage values for a set of scores.

    Args:
        seq_length (int): The length of the sequences used to generate the scores for which rank and coverage are being
        computed.
        scores (np.array): A set of scores to rank and compute coverage for.
        pos_size (int): The dimensionality of the array (whether single, 1, positions or pair, 2, positions are
        being characterized).
        rank_type (str): Whether the optimal value of a set of scores is its 'max' or its 'min'.
    Returns:
        np.array: An array of ranks for the set of scores.
        np.array: An array of coverage scores (what percentile of values are at or below the given score).
    """
    if rank_type == 'max':
        weight = -1.0
    elif rank_type == 'min':
        weight = 1.0
    else:
        raise ValueError('No support for rank types other than max or min, {} provided'.format(rank_type))
    if len(scores.shape) != pos_size:
        raise ValueError('Position size does not match score shape!')
    if pos_size == 1:
        indices = range(seq_length)
        normalization = float(seq_length)
        to_rank = scores * weight
        ranks = np.zeros(seq_length)
        coverages = np.zeros(seq_length)
    elif pos_size == 2:
        indices = np.triu_indices(seq_length, k=1)
        normalization = float(len(indices[0]))
        to_rank = scores[indices] * weight
        ranks = np.zeros((seq_length, seq_length))
        coverages = np.zeros((seq_length, seq_length))
    else:
        raise ValueError('Ranking not supported for position sizes other than 1 or 2, {} provided'.format(pos_size))
    ranks[indices] = rankdata(to_rank, method='dense')
    coverages[indices] = rankdata(to_rank, method='max')
    coverages /= normalization
    return ranks, coverages


def write_slurm(query, aln_fn, project_dir, server_proj_dir, source, job=None, partition='short', mem='64328',
                cores='24', metric='mismatch_diversity', pos_type='pair'):
    """
    Write Slurm

    This Function is used to write a slurm file for submitting a CovET job to the taco server.

    args:
        query (str): The sequence identifier for the sequence being analyzed.
        aln_fn (str): The path to the alignment to analyze on the server.
        project_dir (str): The local project directory where all inputs and outputs will be stored
        server_proj_dir (str): The server project directory where all inputs and outputs will be stored
        source (str): The path to the CovET source files on the server
        job (str): The job name of the run
        partition (str): Server partition to run on
        mem (str): Server memory to be used for the run
        cores (str): The number of CPU cores which this analysis can use while running.
        metric (str): Scoring metric to use when performing trace algorithm, method availability depends on
                        the specified position_type: single: [identity, plain_entropy]; pair: [identity,
                        plain_entropy, mutual_information, normalized_mutual_information,
                        average_product_corrected_mutual_information,
                        filtered_average_product_corrected_mutual_information, match_count, match_entropy,
                        match_diversity, mismatch_count, mismatch_entropy, mismatch_diversity,
                        match_mismatch_count_ratio, match_mismatch_entropy_ratio, match_mismatch_diversity_ratio
                        match_mismatch_count_angle, match_mismatch_entropy_angle, match_mismatch_diversity_angle
                        match_diversity_mismatch_entropy_ratio, match_diversity_mismatch_entropy_angle].
        pos_type (str): Whether to score individual positions 'single' or pairs of positions 'pair'. This will
                        affect which scoring metrics can be selected.
    Returns: Slurm File in the project_directory/Input directory for submission to server
    """
    if not job:
        job = query
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(f'{project_dir}/Input', exist_ok=True)
    os.makedirs(f'{project_dir}/Output', exist_ok=True)
    slurm_path = f'{project_dir}/Input/{job}.slurm'
    server_out_dir = f'{server_proj_dir}/Output'
    server_aln_path = f'{server_proj_dir}/Input/{aln_fn}'
    lines = ['#!/usr/bin/bash' '\n',
             f'#SBATCH --partition={partition}',
             f'#SBATCH --output={server_out_dir}/{job}%j.out',
             f'#SBATCH -e {server_out_dir}/{job}%j.err',
             f'#SBATCH --mem={mem}',
             f'#SBATCH -c {cores}',
             '#SBATCH -n 1',
             f'#SBATCH --job-name={job}',
             '#SBATCH -N 1' '\n',
             "echo 'Switching to file directory'" '\n',
             f"cd {source}" '\n',
             "echo 'Activating Python Environment'" '\n',
             "source activate PyET" '\n',
             "echo 'Starting Match Count Predictions'" '\n',
             f"python EvolutionaryTrace.py --query {query} --alignment {server_aln_path} --output_dir {server_out_dir} "
             f"--position_type {pos_type} --scoring_metric {metric} --processors {cores}" '\n',
             "echo 'Predictions Completed'" '\n',
             "source deactivate" '\n',
             "echo 'Environment Deactivated'"]
    with open(slurm_path, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')


def write_iterative_slurm(query, aln_fn, project_dir, server_proj_dir, source, job=None, partition='mhgcp', mem='64328',
                          cores='24', metric='mismatch_diversity', pos_type='pair'):
    """
    Write Slurm

    This Function is used to write an iterative slurm file for submitting a CovET job to the taco server. Every
    subdirectory in the project-dir will be interpreted as a separate EvolutionaryTrace command.

    args:
        query (str): The sequence identifier for the sequence being analyzed.
        aln_fn (str): The path to the alignment to analyze on the server.
        project_dir (str): The local meta-project directory where all inputs and outputs will be stored.
                         should contain sub-directories each with Input/Output folders with the input alignments.
        server_proj_dir (str): The server meta-project directory where all inputs and outputs will be stored
        source (str): The path to the CovET source files on the server
        job (str): The job name of the run
        partition (str): Server partition to run on
        mem (str): Server memory to be used for the run
        cores (str): The number of CPU cores which this analysis can use while running.
        metric (str): Scoring metric to use when performing trace algorithm, method availability depends on
                        the specified position_type: single: [identity, plain_entropy]; pair: [identity,
                        plain_entropy, mutual_information, normalized_mutual_information,
                        average_product_corrected_mutual_information,
                        filtered_average_product_corrected_mutual_information, match_count, match_entropy,
                        match_diversity, mismatch_count, mismatch_entropy, mismatch_diversity,
                        match_mismatch_count_ratio, match_mismatch_entropy_ratio, match_mismatch_diversity_ratio
                        match_mismatch_count_angle, match_mismatch_entropy_angle, match_mismatch_diversity_angle
                        match_diversity_mismatch_entropy_ratio, match_diversity_mismatch_entropy_angle].
        pos_type (str): Whether to score individual positions 'single' or pairs of positions 'pair'. This will
                        affect which scoring metrics can be selected.
    Returns: Iterative Slurm File in the project_dir/Input directory for submission to server
    """
    if not job:
        job = query
    iterations = os.listdir(project_dir)
    slurm_path = f'{project_dir}/{job}.slurm'
    lines = ['#!/usr/bin/bash' '\n',
             f'#SBATCH --partition={partition}',
             f'#SBATCH --output={server_proj_dir}/{job}%j.out',
             f'#SBATCH -e {server_proj_dir}/{job}%j.err',
             f'#SBATCH --mem={mem}',
             f'#SBATCH -c {cores}',
             '#SBATCH -n 1',
             f'#SBATCH --job-name={job}',
             '#SBATCH -N 1' '\n',
             "echo 'Switching to file directory'" '\n',
             f"cd {source}" '\n',
             "echo 'Activating Python Environment'" '\n',
             "source activate PyET" '\n',
             "echo 'Starting Match Count Predictions'" '\n']
    for iteration in iterations:
        sub_dir = iteration.split('/')[-1]
        server_out_dir = f'{server_proj_dir}/{sub_dir}/Output'
        server_aln_path = f'{server_proj_dir}/{sub_dir}/Input/{aln_fn}'
        lines.append(f"python EvolutionaryTrace.py --query {query} --alignment {server_aln_path} "
                     f"--output_dir {server_out_dir} --position_type {pos_type} --scoring_metric {metric} "
                     f"--processors {cores}" '\n')
    lines.append("echo 'Predictions Completed' \nsource deactivate \necho 'Environment Deactivated'")
    with open(slurm_path, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
