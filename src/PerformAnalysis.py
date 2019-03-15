"""
Created on Aug 17, 2017

@author: daniel
"""
import os
import time
import argparse
import datetime
from multiprocessing import cpu_count
from Bio.Phylo.TreeConstruction import DistanceCalculator

from SupportingClasses.ETMIPC import ETMIPC
from SupportingClasses.PDBReference import PDBReference
from SupportingClasses.ContactScorer import ContactScorer


def parse_arguments():
    """
    parse arguments

    This method provides a nice interface for parsing command line arguments
    and includes help functionality.

    Returns:
        dict. A dictionary containing the arguments parsed from the command line and their arguments.
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
    parser.add_argument('--treeDepth', metavar='K', type=int, nargs='+', default=[2, 3, 5, 7, 10, 25],
                        help='''The levels of the phylogenetic tree to consider when analyzing this alignment, which
                        determines the attributes sequence_assignments and tree_ordering. The following options are
                        available:
                            0 : Entering 0 means all branches from the top of the tree (1) to the leaves (size of the
                            provided alignment) will be analyzed.
                            x, y: If two integers are provided separated by a comma, this will be interpreted as a tuple
                            which will be taken as a range, the top of the tree (1), and all branches between the first
                            and second (non-inclusive) integer will be analyzed.
                            x, y, z, etc.: If one integer (that is not 0) or more than two integers are entered, this
                            will be interpreted as a list. All branches in the list will be analyzed, as well as the top
                            of the tree (1) even if not listed.''')
    parser.add_argument('--distanceModel', metavar='D', type=str, default='identity', nargs='?',
                        choices=['identity'] + DistanceCalculator.protein_models, help='''Which model to use when
                        calculating distances between sequences for phylogenetic tree construction. This choice will
                        influence any of the tree construction method chosen.''')
    parser.add_argument('--treeConstruction', metavar='t', type=str, default='agglomerative', nargs='?',
                        choices=['agglomerative', 'upgma', 'random'], help='''This specifies the method for tree
                        construction used to produce the phylogenetic tree for analysis. Selecting 'agglomerative'
                        produces a tree using the sklearn agglomerative clustering implementation, while 'upgma' uses
                        the Biopython upgma implementation. Selecting random does not use a tree structure, it selects
                        random sequences for each branch at each level specified.''')
    parser.add_argument('--treeConstructionArgs', metavar='a', type=str, nargs='+',
                        help="Additional settings for tree construction can be added here, each tree construction "
                             "method has different options which are described in the specific methods in the "
                             "SeqAlignment class. Provided options should always come in pairs with the name of the "
                             "option coming first and the value for that option coming second e.g. "
                             "'--treeConstructionArgs affinity euclidean' etc.")
    parser.add_argument('--combineBranches', metavar='C', type=str, default='sum', choices=['sum', 'average'],
                        nargs='?', help='The method to use when combining across the specified clustering constants.')
    parser.add_argument('--combineClusters', metavar='c', type=str, nargs='?', default='sum',
                        choices=['sum', 'average', 'size_weighted', 'evidence_weighted', 'evidence_vs_size'],
                        help='How information should be integrated across clusters resulting from the same clustering '
                             'constant.')
    parser.add_argument('--ignoreAlignmentSize', default=False, action='store_true',
                        help='Whether or not to allow alignments with fewer than 125 sequences as suggested by '
                             'PMID:16159918.')
    parser.add_argument('--lowMemoryMode', default=False, action='store_true',
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
                             'lengths (L, L/2 ... L/10), 5 produces heatmaps and surface plots of scores. In all cases '
                             'a file is written out with the final evaluation of the scores, if no PDB is provided, '
                             'this means only times will be recorded.')
    # Clean command line input
    arguments = parser.parse_args()
    arguments = vars(arguments)
    arguments['treeDepth'] = sorted(arguments['treeDepth'])
    if len(arguments['treeDepth']) == 1 and arguments['treeDepth'][0] == 0:
        arguments['treeDepth'] = None
    elif len(arguments['treeDepth']) == 2:
        arguments['treeDepth'] = tuple(arguments['treeDepth'])
    if arguments['treeConstructionArgs'] is not None:
        arguments['treeConstructionArgs'] = {arguments['treeConstructionArgs'][i]:
                                                 arguments['treeConstructionArgs'][i + 1]
                                             for i in range(0, len(arguments['treeConstructionArgs']), 2)}
    else:
        arguments['treeConstructionArgs'] = {}
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
    cetmip_obj.calculate_scores(out_dir=query_dir, curr_date=today, query=args['query'][0], aa_mapping=aa_dict,
                                tree_depth=args['treeDepth'], model=args['distanceModel'],
                                clustering=args['treeConstruction'], clustering_args=args['treeConstructionArgs'],
                                combine_clusters=args['combineClusters'], combine_branches=args['combineBranches'],
                                processes=args['processes'], low_mem=args['lowMemoryMode'],
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
    test_scorer_any = ContactScorer(query=args['query'][0], seq_alignment=cetmip_obj.alignment,
                                    pdb_reference=query_structure, cutoff=args['threshold'])
    test_scorer_any.evaluate_predictor(predictor=cetmip_obj, verbosity=args['verbosity'], out_dir=query_dir, dist='Any')
    test_scorer_beta = ContactScorer(query=args['query'][0], seq_alignment=cetmip_obj.alignment,
                                     pdb_reference=query_structure, cutoff=args['threshold'])
    test_scorer_beta.evaluate_predictor(predictor=cetmip_obj, verbosity=args['verbosity'], out_dir=query_dir, dist='CB')
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
