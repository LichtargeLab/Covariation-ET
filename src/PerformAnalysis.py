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
from SupportingClasses.SeqAlignment import SeqAlignment
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
    # Set up all optional variables to be parsed from the command line
    # (defaults)
    parser.add_argument('--threshold', metavar='T', type=float, nargs='?',
                        default=8.0,
                        help='The distance within the molecular structure at which two residues are considered '
                             'interacting.')
    parser.add_argument('--clusters', metavar='K', type=int, nargs='+',
                        default=[2, 3, 5, 7, 10, 25],
                        help='The clustering constants to use when performing this analysis.')
    parser.add_argument('--combineKs', metavar='C', type=str, nargs='?',
                        default='sum', choices=['sum', 'average'],
                        help='')
    parser.add_argument('--combineClusters', metavar='c', type=str, nargs='?',
                        default='sum',
                        choices=['sum', 'average', 'size_weighted',
                                 'evidence_weighted', 'evidence_vs_size'],
                        help='How information should be integrated across clusters resulting from the same clustering '
                             'constant.')
    parser.add_argument('--ignoreAlignmentSize', metavar='i', type=bool, nargs='?',
                        default=False,
                        help='Whether or not to allow alignments with fewer than 125 sequences as suggested by '
                             'PMID:16159918.')
    parser.add_argument('--lowMemoryMode', metavar='l', type=bool, nargs='?',
                        default=False, help='Whether to use low memory mode or not. If low memory mode is engaged '
                                            'intermediate values in the ETMIPC class will be written to file instead '
                                            'of stored in memory. This will reduce the memory footprint but may '
                                            'increase the time to run. Only recommended for very large analyses.')
    parser.add_argument('--processes', metavar='M', type=int, default=1, nargs='?',
                        help='The number of processes to spawn when multiprocessing this analysis.')
    parser.add_argument('--verbosity', metavar='V', type=int, default=1,
                        nargs='?', choices=[1, 2, 3, 4], help='How many figures to produce.\n1 = ROC Curves, ETMIP '
                                                              'Coverage file, and final AUC and Timing file\n2 = '
                                                              'files with all scores at each clustering\n3 = '
                                                              'sub-alignment files and plots\n4 = surface plots and '
                                                              'heatmaps of ETMIP raw and coverage scores.')
    # Clean command line input
    arguments = parser.parse_args()
    arguments = vars(arguments)
    arguments['clusters'] = sorted(arguments['clusters'])
    processor_count = cpu_count()
    if arguments['processes'] > processor_count:
        arguments['processes'] = processor_count
#     print args
#     embed()
#     exit()
    return arguments


def analyze_alignment(args):
    start = time.time()
    ###########################################################################
    # Set up global variables
    ###########################################################################
    today = str(datetime.date.today())
    aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
               'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    aa_dict = {aa_list[i]: i for i in range(len(aa_list))}
    ###########################################################################
    # Set up output location
    ###########################################################################
    start_dir = os.getcwd()
    print(start_dir)
    if args['output'].startswith('..'):
        args['output'] = os.path.abspath(os.path.join(start_dir, args['output']))
    create_folder = os.path.join(args['output'], str(today), args['query'][0])
    if not os.path.exists(create_folder):
        os.makedirs(create_folder)
        print "Creating output folder"
    # os.chdir(create_folder)
    # create_folder = os.getcwd()
    ###########################################################################
    # Import alignment
    ###########################################################################
    print 'Importing alignment'
    # Create SeqAlignment object to represent the alignment for this analysis.
    # if args['alignment'][0].startswith('..'):
    #     args['alignment'][0] = os.path.abspath(os.path.join(start_dir, args['alignment'][0]))
    #     query_alignment = SeqAlignment(query_id=args['query'][0], file_name=(os.path.join(start_dir,
    #                                                                                       args['alignment'][0])))
    # else:
    query_alignment = SeqAlignment(file_name=args['alignment'][0], query_id=args['query'][0])
    # Import alignment information from file.
    query_alignment.import_alignment(save_file=os.path.join(create_folder, 'alignment.pkl'))
    # Check if alignment meets analysis criteria:
    if (not args['ignoreAlignmentSize']) and (query_alignment.size < 125):
        raise ValueError('The multiple sequence alignment is smaller than recommended for performing this analysis ({'
                         '} < 125, see PMID:16159918), if you wish to proceed with the analysis anyway please call '
                         'the code again using the --ignoreAlignmentSize option.'.format(query_alignment.size))
    if query_alignment.size < max(args['clusters']):
        raise ValueError('The analysis could not be performed because the alignment has fewer sequences than the '
                         'requested number of clusters ({} < {}), please provide an alignment with more sequences or '
                         'change the clusters requested by using the --clusters option when using this '
                         'software.'.format(query_alignment.size, max(args['clusters'])))
    # Remove gaps from aligned query sequences
    query_alignment.remove_gaps(save_file=os.path.join(create_folder, 'ungapped_alignment.pkl'))
    # Create matrix converting sequences of amino acids to sequences of integers
    # representing sequences of amino acids.
    query_alignment.alignment_to_num(aa_dict)
    # Write the ungapped alignment to file.
    query_alignment.write_out_alignment(file_name=os.path.join(create_folder, 'UngappedAlignment.fa'))
    # Compute distance between all sequences in the alignment
    query_alignment.compute_distance_matrix(save_file=os.path.join(create_folder, 'X'))
    # Determine the full clustering tree for the alignment and the ordering of
    # its sequences.
    query_alignment.set_tree_ordering()
    print('Query Sequence:')
    print(query_alignment.query_sequence)
    test_cetmip = ETMIPC(alignment=args['alignment'][0])
    test_cetmip.output_dir = args['output']
    test_cetmip.import_alignment(query=args['query'][0], aa_dict=aa_dict,
                                 ignore_alignment_size=args['ignoreAlignmentSize'])
    from IPython import embed
    embed()
    exit()
    ###########################################################################
    # Import the PDB if provided and Create scoring object and initialize it
    ###########################################################################
    if args['pdb']:
        # Create PDBReference object to represent the structure for this
        # analysis.
        if args['pdb'].startswith('..'):
            query_structure = PDBReference(os.path.join(start_dir, args['pdb']))
        else:
            query_structure = PDBReference(args['pdb'])
        # Import the structure information from the file.
        query_structure.import_pdb(args['query'][0], save_file='pdbData.pkl')
        scorer = ContactScorer(query_alignment, query_structure, args['threshold'])
        scorer.fit()
        scorer.measure_distance(save_file='PDBdistances')
    else:
        query_structure = None
        scorer = None
    ###########################################################################
    # Perform multiprocessing of clustering method
    ###########################################################################
    print 'Starting ETMIP'
    # Create ETMIPC object to represent the analysis being performed.
    etmip_obj = ETMIPC(query_alignment, args['clusters'], query_structure,
                       create_folder, args['processes'], args['lowMemoryMode'])
    # Calculate the MI scores for all residues across all sequences
    etmip_obj.determine_whole_mip('evidence' in args['combineClusters'])
    # Calculate the the ETMIPC scores for various clustering constants.
    etmip_obj.calculate_clustered_mip_scores(aa_dict=aa_dict, combine_clusters=args['combineClusters'])
    # Combine the clustering results across all clustering constants tested.
    etmip_obj.combine_clustering_results(combination=args['combineKs'])
    # Compute normalized scores for ETMIPC and evaluate against PDB if
    # provided.
    etmip_obj.compute_coverage_and_auc(contact_scorer=scorer)#othreshold=args['threshold'])
    # Write out cluster specific scores and produce figures.
    # etmip_obj.produce_final_figures(today, cut_off=args['threshold'], verbosity=args['verbosity'])
    etmip_obj.produce_final_figures(today, scorer=scorer, verbosity=args['verbosity'])
    # Write out the AUCs and final times for the different clustering constants
    # tested.
    etmip_obj.write_final_results(today, args['threshold'])
    # If low memory mode was used clear out intermediate files saved in this
    # process.
    if args['lowMemoryMode']:
        etmip_obj.clear_intermediate_files()
    print "Generated results in: ", create_folder
    os.chdir(start_dir)
    end = time.time()
    print('ET MIP took {} minutes to run!'.format((end - start) / 60.0))


if __name__ == '__main__':
    # Read input from the command line
    command_line_options = parse_arguments()
    # Perform analysis
    analyze_alignment(command_line_options)
