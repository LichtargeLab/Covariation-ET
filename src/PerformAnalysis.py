"""
Created on Aug 17, 2017

@author: daniel
"""
import os
import time
import argparse
import datetime
from multiprocessing import cpu_count

from SupportingClasses.ETMIPC import ETMIPC
from SupportingClasses.PDBReference import PDBReference
from SupportingClasses.ContactScorer import ContactScorer


def parse_arguments():
    """
    parse arguments

    This method provides a nice interface for parsing command line arguments
    and includes help functionality.

    Returns:
    --------
    dict:
        A dictionary containing the arguments parsed from the command line and
        their arguments.
    """
    # Create input parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    # Set up all variables to be parsed from the command line (no defaults)
    parser.add_argument('--alignment', metavar='A', type=str, nargs=1,
                        help='The file path to the alignment to analyze in this run.')
    parser.add_argument('--pdb', metavar='P', type=str, nargs='?',
                        help='The file path to the PDB structure associated with the provided alignment.')
    parser.add_argument('--query', metavar='Q', type=str, nargs=1,
                        help='The name of the protein being queried in this analysis.')
    parser.add_argument('--output', metavar='O', type=str, nargs='?',
                        default='./', help='File path to a directory where the results can be generated.')
    # Set up all optional variables to be parsed from the command line (defaults)
    parser.add_argument('--threshold', metavar='T', type=float, nargs='?', default=8.0,
                        help='The distance within the molecular structure at which two residues are considered '
                             'interacting.')
    parser.add_argument('--clusters', metavar='K', type=int, nargs='+', default=[2, 3, 5, 7, 10, 25],
                        help='The clustering constants to use when performing this analysis.')
    parser.add_argument('--combineKs', metavar='C', type=str, nargs='?', default='sum', choices=['sum', 'average'],
                        help='The method to use when combining across the specified clustering constants.')
    parser.add_argument('--combineClusters', metavar='c', type=str, nargs='?', default='sum',
                        choices=['sum', 'average', 'size_weighted', 'evidence_weighted', 'evidence_vs_size'],
                        help='How information should be integrated across clusters resulting from the same clustering '
                             'constant.')
    parser.add_argument('--ignoreAlignmentSize', metavar='i', type=bool, nargs='?', default=False,
                        help='Whether or not to allow alignments with fewer than 125 sequences as suggested by '
                             'PMID:16159918.')
    parser.add_argument('--lowMemoryMode', metavar='l', type=bool, nargs='?', default=False,
                        help='Whether to use low memory mode or not. If low memory mode is engaged intermediate values '
                             'in the ETMIPC class will be written to file instead of stored in memory. This will reduce'
                             ' the memory footprint but may increase the time to run. Only recommended for very large '
                             'analyses.')
    parser.add_argument('--processes', metavar='M', type=int, default=1, nargs='?',
                        help='The number of processes to spawn when multiprocessing this analysis.')
    parser.add_argument('--verbosity', metavar='V', type=int, default=1, nargs='?', choices=[1, 2, 3, 4, 5],
                        help='Which output to generate. 1 writes scores for all tested clustering constants, 2 tests '
                             'the clustering Z-score of the predictions and writes them to file as well as plotting ' 
                             'Z-Scores against resiude count, 3 tests the AUROC of contact prediction at different '
                             'levels of sequence separation and plots the resulting curves to file, 4 tests the '
                             'precision of  contact prediction at different levels of sequence separation and list '
                             'lengths (L, L/2 ... L/10). In all cases a file is written out with the final evaluation '
                             'of the scores, if no PDB is provided, this means only times will be recorded.')
    # Clean command line input
    arguments = parser.parse_args()
    arguments = vars(arguments)
    arguments['clusters'] = sorted(arguments['clusters'])
    processor_count = cpu_count()
    if arguments['processes'] > processor_count:
        arguments['processes'] = processor_count
    return arguments


def analyze_alignment(args):
    start = time.time()
    # Set up global variables
    today = str(datetime.date.today())
    aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
               'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
    # Set up output location
    if not os.path.isdir(args['output']):
        os.mkdir(args['output'])
    query_dir = os.path.join(args['output'], args['query'][0])
    if not os.path.isdir(query_dir):
        os.mkdir(query_dir)
    print 'Starting cET-MIp'
    # Create ETMIPC object to represent the analysis being performed.
    cetmip_obj = ETMIPC(alignment=args['alignment'][0])
    # Calculate the MI scores for all residues across all sequences
    # Calculate the the cET-MIp scores for various clustering constants.
    # Combine the clustering results across all clustering constants tested.
    # Compute normalized scores for ETMIPC
    # Write out cluster specific scores
    cetmip_obj.calculate_scores(out_dir=query_dir, today=today, query=args['query'][0], clusters=args['clusters'],
                                aa_dict=aa_dict, combine_clusters=args['combineClusters'], combine_ks=args['combineKs'],
                                processes=args['processes'], low_memory_mode=args['lowMemoryMode'],
                                ignore_alignment_size=args['ignoreAlignmentSize'])

    # Create PDBReference object to represent the structure for this analysis.
    if args['pdb']:
        query_structure = PDBReference(args['pdb'])
        # Import the structure information from the file.
        query_structure.import_pdb(args['query'][0], save_file=os.path.join(query_dir, 'pdbData.pkl'))
    else:
        query_structure = None
    # Evaluate against PDB if provided and produce figures.
    # Write out the statistics and final times for the different clustering constants tested.
    test_scorer_any = ContactScorer(seq_alignment=cetmip_obj.alignment, pdb_reference=query_structure,
                                    cutoff=args['threshold'])
    test_scorer_any.evaluate_predictor(query=args['query'][0], predictor=cetmip_obj, verbosity=args['verbosity'],
                                       out_dir=query_dir, dist='Any')
    test_scorer_beta = ContactScorer(seq_alignment=cetmip_obj.alignment, pdb_reference=query_structure,
                                     cutoff=args['threshold'])
    test_scorer_beta.evaluate_predictor(query=args['query'][0], predictor=cetmip_obj, verbosity=args['verbosity'],
                                        out_dir=query_dir, dist='CB')
    # If low memory mode was used clear out intermediate files saved in this
    if args['lowMemoryMode']:
        cetmip_obj.clear_intermediate_files()
    end = time.time()
    print('ET MIP took {} minutes to run!'.format((end - start) / 60.0))
    print 'Generated results in: {}'.format(query_dir)


if __name__ == '__main__':
    # Read input from the command line
    command_line_options = parse_arguments()
    # Perform analysis
    analyze_alignment(command_line_options)
