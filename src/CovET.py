"""
Created on March 23, 2022

@author: Chen Wang
"""
import os
import argparse
import subprocess
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from multiprocessing import Pool
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator
from SupportingClasses.Predictor import Predictor
from SupportingClasses.utils import remove_sequences_with_ambiguous_characters
from SupportingClasses.PhylogeneticTree import PhylogeneticTree
from SupportingClasses.AlignmentDistanceCalculator import AlignmentDistanceCalculator



def parse_args():
    """
    Parse Arguments

    This method processes the command line arguments for CovET.
    """
    # Create input parser
    parser = argparse.ArgumentParser(description='Process CovET options.')
    # Mandatory arguments (no defaults)
    parser.add_argument('--query', metavar='Q', type=str, nargs='?', required=True,
                        help='The name of the protein being queried in this analysis (should be the sequence identifier'
                             ' in the specified alignment).')
    parser.add_argument('--alignment', metavar='A', type=str, nargs='?', required=True,
                        help='The file path to the alignment to analyze (fasta formatted alignment expected).')
    parser.add_argument('--output_dir', metavar='O', type=str, nargs='?', required=True,
                        help='File path to a directory where the results can be generated.')
    # Optional argument (has default)
    parser.add_argument('--processors', metavar='C', type=int, nargs='?', default=1,
                        help="The number of CPU cores available to this process while running (will determine the "
                             "number of processes forked by steps which can be completed using multiprocessing pools).")
    parser.add_argument('--filter_seqs', metavar='F', type=bool, nargs='?', default=False,
                        help="Remove sequences in the input alignment that contain ambiguous characters"
                            "Allowed Characters are ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V,'W','Y','-']")
    parser.add_argument('--Add_Chars', metavar='AC', type=str, nargs='*', default=[],
                        help="Add ambiguous characters to the alphabet, requires --filter_seqs = True."
                            "Supported ambiguous characters are ['B', 'X', 'Z']")
                        
    arguments = parser.parse_args()
    arguments = vars(arguments)
    return arguments

if __name__ == "__main__":
    # Read input from the command line
    args = parse_args()
    query=args['query']
    aln_file=args['alignment']
    out_dir=args['output_dir']
    cores=args['processors']
    filter_seqs=args['filter_seqs']
    add_chars=args['Add_Chars']
    if add_chars:
        assert filter_seqs
    
                        
        
    
    start_all = time()
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    #Filter Out Sequences with Ambiguous Characters
    if filter_seqs:
        aln_file = remove_sequences_with_ambiguous_characters(aln_file, out_dir, additional_chars = add_chars)
    
    # Generate file names
    phylo_tree_fn = f'{query}_ET_blosum62_dist_et_tree.nhx'
    phylo_tree_fn = os.path.join(out_dir, phylo_tree_fn)
    
    msa_fn = os.path.join(out_dir, "Non-Gapped_Alignment.fa")
    
    pair_output_fn = f'{query}_ET_blosum62_Dist_et_Tree_All_Ranks_mismatch_diversity_Scoring.ranks'
    pair_output_fn = os.path.join(out_dir, pair_output_fn)
    
    single_output_fn = f'{query}_ET_blosum62_Dist_et_Tree_All_Ranks_mismatch_diversity_Scoring_Converted_To_Single_Pos_TopScoring.ranks'
    single_output_fn = os.path.join(out_dir, single_output_fn)
    
    legacy_output_fn = f'{query}_ET_blosum62_Dist_et_Tree_All_Ranks_mismatch_diversity_Scoring_Converted_To_Single_Pos_TopScoring.legacy.ranks'
    legacy_output_fn = os.path.join(out_dir, legacy_output_fn)
    
    # Generate non-gapped alignments. Then write both alignments to output dir.
    print('Step 1/4: Removing gaps in query')
    predictor=Predictor(query=query, aln_file=aln_file, 
                        out_dir=out_dir)
    
    # Calculate distances between sequence pairs in MSA
    print('Step 2/4: Calculating distances')
    start_dist = time()
    calculator = AlignmentDistanceCalculator(protein=True,model="blosum62", skip_letters=None)
    _, distance_matrix, _, _ = calculator.get_et_distance(predictor.original_aln.alignment,processes=cores)
    end_dist = time()
    print('Calculating distances took: {} min'.format((end_dist - start_dist) / 60.0))

    
    # Build tree and write to file
    print('Step 3/4: Building phylogenetic tree')
    start_tree = time()
    phylo_tree = PhylogeneticTree(tree_building_method="et",
                              tree_building_args={})
    phylo_tree.construct_tree(dm=distance_matrix)
    phylo_tree.write_out_tree(filename=phylo_tree_fn)
    end_tree = time()
    print('Constructing tree took: {} min'.format((end_tree - start_tree) / 60.0))
    
    # Call R script to calculate CovET scores
    print('Step 4/4: Computing CovET scores')
    start_r = time()
    R_cmd = f'Rscript --vanilla CovET_Rcpp_cmd.R {phylo_tree_fn} {msa_fn} {query} ' \
            f'{pair_output_fn} {single_output_fn} {legacy_output_fn} {cores}'
    subprocess.run(R_cmd.split())
    end_r = time()
    print('Computing CovET took: {} min'.format((end_r - start_r) / 60.0))
    end_all = time()
    print('Total run time: {} min'.format((end_all - start_all) / 60.0))
