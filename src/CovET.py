import os
import argparse
import numpy as np
import pandas as pd
from time import time
from SupportingClasses.Predictor import Predictor
from SupportingClasses.utils import convert_seq_to_numeric
from SupportingClasses.PhylogeneticTree import PhylogeneticTree
from SupportingClasses.AlignmentDistanceCalculator import AlignmentDistanceCalculator
from SupportingClasses.utils import remove_sequences_with_ambiguous_characters
from ctypes import CDLL, POINTER
from ctypes import c_double, c_int32


def Calculate_Fixed_Penalty(phylo_tree_assignments):
    """
    This function calculates the fixed penalty applied to all ij pairs in the alignment according to the CovET formula
    :param dict phylo_tree_assignments: Assignemnts generated from SupportingClasses.PhylogeneticTree.assign_group_rank.
            First level of the dictionary maps a rank to another dictionary.
            The second level of the dictionary maps a group value to another dictionary. This third level of the
            dictionary maps the key 'node' to the node which is the root of the group at the given rank, 'terminals' to
            a list of node names for the leaf/terminal nodes which are ancestors of the root node, and 'descendants' to
            a list of nodes which are descendants of 'node' from the closest assigned rank (.i.e. the nodes children, at
            the lowest rank this will be None).
    :return int fixed_penalty: fixed penalty according to CovET formula
    """
    fixed_penalty = 0
    for division in phylo_tree_assignments:
        for group in phylo_tree_assignments[division]:
            if not phylo_tree_assignments[division][group]['descendants']:
                fixed_penalty += 1 / division
    return fixed_penalty


def Generate_Processed_Tree(phylo_tree, tree_index_mapping):
    """
    This function takes a PhylogeneticTree Object and returns the processed version of the tree as a dataframe for ET
    Calculations.

    phylo_tree: A phylogenetic tree object defined by SupportingClasses/PhylogeneticTree.py
    tree_index_mapping(dict): mapping of the sequenceIDs of the tree to the sequence index

    return DataFrame [start]: first terminal node of the division
                      [end]: last terminal node of the division
                      [scale]: scale to multiply CovET penalty by (1/n term in CovET formula)
                      [tipn]: number of terminal nodes in the group
                      [groupsize]: how many ij comparisions there are within a group
    """

    processed_tree = {'start': [], 'end': [], 'scale': [], 'tipn': [], 'groupsize': []}
    for node_ind, node in enumerate(reversed(phylo_tree.tree.get_nonterminals())):
        term_ind = [tree_index_mapping[term.name] for term in node.get_terminals()]
        processed_tree['start'].append(int(min(term_ind)))
        processed_tree['end'].append(int(max(term_ind)))
        processed_tree['scale'].append((1) / (phylo_tree.size - (node_ind + 1)))

        nterm = node.count_terminals()
        processed_tree['tipn'].append(int(nterm))
        processed_tree['groupsize'].append(int(nterm * (nterm - 1) / 2))

    processed_tree = pd.DataFrame.from_dict(processed_tree, orient='index').transpose()
    return processed_tree


def Alphabetical_to_Numeric_Mapping():
    letters = ["-", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
               "X", "B", "Z"]
    alphabetical_to_numeric = {char: i + 11 for i, char in enumerate(letters)}
    numeric_to_alphabetical = {i + 11: char for i, char in enumerate(letters)}
    alphabetical_to_numeric[":"] = 0
    numeric_to_alphabetical[0] = ":"
    return alphabetical_to_numeric, numeric_to_alphabetical


def Generate_Numeric_MSA(predictor, alphabetical_to_numeric_map, tree_index_mapping):
    """

    :param predictor: A Predictor Object
    :return: msa_num(numpy.array): A numerical representation of the Multiple Sequence Alignment
    """
    numeric_reps = []
    for i in predictor.non_gapped_aln.seq_order:
        seq_record = predictor.non_gapped_aln.alignment[tree_index_mapping[i]]
        numeric_reps.append(convert_seq_to_numeric(seq_record, alphabetical_to_numeric_map))
    msa_num = np.stack(numeric_reps)
    return msa_num


def to_matrix(seq):
    """
    This function takes an amino acid sequence as a numpy array and generates a matrix for all ij pairs
    :return numpy matrix
    """
    return np.tile(seq, (len(seq), 1)), np.tile(seq, (len(seq), 1)).T


def Generate_Position_Matrix(numeric_msa):
    """
    :param numpy.ndarray numeric_msa: Numeric representation of the Multiple Sequence Alignment
    :return list pos_mat: List of all sequence positions, for each sequence position there is a list of two arrays,
    the first numpy nd.array matrix (bool) is whether the two amino acids are the same down the alignment,
    the second numpy array (int) is (i + 1000) + j. For example, if i is A (num 12) and j is C (num 13),
    ij will contain (12 + 1000) + 13 = 1213
    *note that ij here refers to different species, not different sequence positions as is standard elsewhere
    """
    pos_mat = []
    for i in range(len(numeric_msa[0])):
        pos_i_seq = numeric_msa[:, i]
        mat_1, mat_2 = to_matrix(pos_i_seq)
        mat_logi = mat_1 == mat_2
        lower_id = np.tril_indices_from(mat_logi)
        for x in range(len(lower_id[0])):
            mat_logi[lower_id[0][x], lower_id[1][x]] = True
        mat_vari = np.triu(mat_1 + 100 * mat_2, k=1)
        pos_mat.append([mat_logi, mat_vari])
    return pos_mat


def Non_Concerted_Matrix(position_matrix, pos_i, pos_j):
    """
    This function takes a position matrix and identifies all non-concerted variations between two positions in the
    alignment

    :param list position_matrix: Position matrix as generated by Generate_Position_Matrix: List of all sequence
    positions, for each sequence position there is a list of two arrays,
    the first numpy nd.array matrix (bool) is whether the two amino acids are the same down the alignment,
    the second numpy array (int) is (i + 1000) + j
    :param int pos_i: The first amino acid sequence position
    :param int pos_j: The second amino acid sequence position
    :return numpy.ndarray non_concerted_matrix: A matrix of non-concerted variations between pos_i and pos_j of the
    form i + j*10000, where i and j are the 4 digit pair code generated by Generate_Position_Matrix (ex AA --> AB =
    1212 --> 1313 = 1212 + (1313*10000) = 12121313.
    Concerted variation and conservation are represented as a 0 (ex AA --> BB = 0, AA --> AA = 0).
    """
    non_concerted_rev_index = ((position_matrix[pos_i][0].astype(int) + position_matrix[pos_j][0].astype(int)) != 1)
    non_concerted_matrix = (position_matrix[pos_i][1] + 10000 * position_matrix[pos_j][1])
    non_concerted_matrix[non_concerted_rev_index] = 0
    return non_concerted_matrix


def Variability_Count(pos_i, pos_j, numeric_msa):
    """
    This function identifies the unique pairs of amino acids at position ij
    :param int pos_i: sequence position i
    :param int pos_j: sequence position j
    :param numpy.ndarray numeric_msa: Numeric representation of the Multiple Sequence Alignment
    :return: int number of unique pairs of amino acids at position ij
    """
    return len(set(numeric_msa[:, pos_i] + 1000 * numeric_msa[:, pos_j]))


def Generate_Pair_Mapping(msa_num, ppi=False):
    """
    This function identifies the pairs of amino acids to be scored in the multiple sequence alignment and sets up a
    dataframe of the pair, as well holding spots for the score and other metrics for that position. :param
    numpy.ndarray msa_num: A numerical representation of the Multiple Sequence Alignment
    :return dict pair_mapping:
    First layer is dictionary of all position i's in the Multiple sequence alignment. The second layer of the
    dictionary contains position j's, the third contains holding values for the score and variability count of pair
    ij to be calculated and filled in later.
    """
    if not ppi:
        pair_mapping = {
            x: {y: {'score': 0, 'variability_count': 0} for y in list(np.arange(0, msa_num.shape[1])) if x < y}
            for x in list(np.arange(0, msa_num.shape[1]))}
    else:
        concat_index = np.where(msa_num[0] == 0)[0][0]
        for seq in np.arange(msa_num.shape[0]):
            assert msa_num[seq][
                       concat_index] == 0, "The concatenation point is not the same throughout the entire alignment"

        pair_mapping = {
            x: {y: {'score': 0, 'variability_count': 0} for y in list(np.arange(concat_index + 1, msa_num.shape[1]))}
            for x in list(np.arange(0, concat_index))}

    return pair_mapping


def CovET(processed_tree, msa_num, position_matrix, pair_mapping, fixed_penalty, query_sequence):
    """
    This function calls a c++ shared library to perform the CovET calculation and fills in the pair mapping
    :param msa_num: Numeric Representation of the MSA
    :param list position_matrix: Position matrix as generated by Generate_Position_Matrix: List of all sequence
    positions, for each sequence position there is a list of two arrays,
    the first numpy nd.array matrix (bool) is whether the two amino acids are the same down the alignment,
    the second numpy array (int) is (i + 1000) + j
    :param dict pair_mapping:
    First layer is dictionary of all position i's in the Multiple sequence alignment. The second layer of the
    dictionary contains position j's, the third contains holding values for the score and variability count of pair
    ij to be calculated and filled in later.
    :param int fixed_penalty: fixed penalty according to CovET formula
    :param Bio.Seq.Seq query_sequence: Query Sequence of Amino Acids
    :return: dict pair_mapping: Same pair mapping above, with filled in score, rank, coverage, etc..
    """
    # load the library
    cpp_function_path = os.path.join(os.getcwd(), 'CovET_func_py.so')
    mylib = CDLL(cpp_function_path)

    # C-type corresponding to matrix
    _doublepp = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')

    # C-type corresponding to numpy array
    ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C")

    # define prototypes, transform data types to be read by c++
    mylib.TraceGroupC_vec.argtypes = (ND_POINTER_1, ND_POINTER_1, ND_POINTER_1, c_int32, _doublepp)
    mylib.TraceGroupC_vec.restype = POINTER(c_double * processed_tree.shape[0])
    mylib.free_mem.argtype = POINTER(c_double * processed_tree.shape[0])

    groupsize = processed_tree['groupsize'].to_numpy().astype(np.int32)
    start = processed_tree['start'].to_numpy().astype(np.int32)
    end = processed_tree['end'].to_numpy().astype(np.int32)

    for pos_i in pair_mapping:
        for pos_j in pair_mapping[pos_i].keys():
            non_con_mat = Non_Concerted_Matrix(position_matrix, pos_i, pos_j).astype(np.int32)
            xpp = (non_con_mat.__array_interface__['data'][0]
                   + np.arange(non_con_mat.shape[0]) * non_con_mat.strides[0]).astype(np.uintp)

            c_return_scores = mylib.TraceGroupC_vec(start, end, groupsize, processed_tree.shape[0], xpp)

            pair_mapping[pos_i][pos_j]['score'] = sum(
                [processed_tree['scale'][i] * score for i, score in
                 enumerate(c_return_scores.contents)]) + fixed_penalty + 1

            mylib.free_mem(c_return_scores)

            pair_mapping[pos_i][pos_j]['variability_count'] = Variability_Count(pos_i, pos_j, msa_num)
            pair_mapping[pos_i][pos_j]['Query_i'] = query_sequence[pos_i]
            pair_mapping[pos_i][pos_j]['Query_j'] = query_sequence[pos_j]
    return pair_mapping


def Convert_Pair_Mapping_to_DataFrame(results, pair_output_path, return_frame=False):
    """
    This function takes the pair mapping generated by CovET and converts it to a dataframe
    :param results: dict containing the results of the CovET run. Must be of the form generated in CovET
    :param str pair_output_path: Path to where the results will be written
    :param bool return_frame: Whether or not to return a DataFrame object of the results.
    :return:  Dataframe of results
    """
    rows = []
    for pos_i in results:
        for pos_j in results[pos_i].keys():
            rows.append({"Position_i": pos_i,
                         "Query_i": results[pos_i][pos_j]["Query_i"],
                         "Position_j": pos_j,
                         "Query_j": results[pos_i][pos_j]["Query_j"],
                         "Variability_Count": results[pos_i][pos_j]["variability_count"],
                         "Rank": np.nan,
                         "Score": results[pos_i][pos_j]["score"]
                         })

    out_frame = pd.DataFrame(rows)
    out_frame["Rank"] = out_frame["Score"].rank(method="dense").astype(np.int64)
    out_frame["Coverage"] = out_frame["Score"].rank(method="max", pct=True)
    out_frame["Score"] = out_frame["Score"].round(decimals=3)
    out_frame["Coverage"] = out_frame["Coverage"].round(decimals=3)
    out_frame.to_csv(pair_output_path, header=True, index=False, sep='\t')
    if return_frame:
        return out_frame


def Convert_Paired_Results_To_Single_Residue_Rankings(out_frame, single_output_path, return_frame=False):
    """
    This function takes a dataframe of the CovET results and converts it to single residue format :param pd.DataFrame
    out_frame: dataframe of the paired CovET results :param str single_output_path: output path for the single
    residue rankings :return: pd.DataFrame top_single_residue_scores: (optional) the DataFrame object representing
    single residue rankings
    """
    j_frame = out_frame.loc[:, (out_frame.columns != "Position_i") &
                               (out_frame.columns != "Query_i")].rename(
        columns={"Position_j": "Position", "Query_j": "Query"})
    i_frame = out_frame.loc[:, (out_frame.columns != "Position_j") &
                               (out_frame.columns != "Query_j")].rename(
        columns={"Position_i": "Position", "Query_i": "Query"})

    all_single_residue_scores = pd.concat([i_frame, j_frame], ignore_index=True)
    groups = all_single_residue_scores.groupby("Position")

    top_scores_per_pos = []
    for group_name, group in groups:
        df = group[group["Rank"] == group["Rank"].min()]
        var_chars = ''.join([x for x in set(predictor.non_gapped_aln.alignment[:, group_name])])
        top_scores_per_pos.append({"Position": group_name,
                                   "Query": df["Query"].values[0][0],
                                   "Variability_Characters": var_chars,
                                   "Variability_Count": len(set(predictor.non_gapped_aln.alignment[:, group_name])),
                                   "Score": df["Score"].values[0]})

    top_single_residue_scores = pd.DataFrame(top_scores_per_pos)
    top_single_residue_scores["Rank"] = top_single_residue_scores["Score"].rank(method="dense").astype(np.int64)
    top_single_residue_scores["Coverage"] = top_single_residue_scores["Score"].rank(method="max", pct=True)
    top_single_residue_scores["Score"] = top_single_residue_scores["Score"].round(decimals=3)
    top_single_residue_scores["Coverage"] = top_single_residue_scores["Coverage"].round(decimals=3)
    top_single_residue_scores.to_csv(single_output_path, header=True, columns=['Position', 'Query', 'Rank',
                                                                               'Variability_Characters',
                                                                               'Variability_Count', 'Score',
                                                                               'Coverage'],
                                     index=False, sep='\t')
    if return_frame:
        return top_single_residue_scores


def convert_file_to_legacy_format(res_fn, reverse_score=False):
    """
    Convert File To Legacy Format
    This function accepts a single residue analysis file (tab separated .ranks file) and convets it to the legacy ETC
    ranks file format. This file format is needed to use previous tools like the PyETViewer.
    Args:
        res_fn (str): The path the the single residue ranks file to be converted.
        reverse_score (bool): If this flag is set to true, recompute scores so that each score is:
        score_new = 1 + |score_i - score_max|
        This is used to correct for metrics (like ET-MIp) where a higher score is better, but which does not interact
        correctly with PyETViewer, which imports rho or the score for residue selection and coloring and recomputes
        coverage on the fly.
    Return:
        str: The path to the converted output file.
    """
    res_dir = os.path.dirname(res_fn)
    res_base_name, res_base_extension = os.path.splitext(os.path.basename(res_fn))
    scoring_df = pd.read_csv(res_fn, sep='\t', header=0, index_col=None)
    assert {'Position', 'Query'}.issubset(scoring_df.columns), "Provided file does not include expected columns, make" \
                                                               " sure this is a single position analysis result!"
    if reverse_score:
        max_score = scoring_df['Score'].max()
        scoring_df['Reversed_Score'] = scoring_df['Score'].apply(lambda x: 1 + abs(x - max_score))
        scoring_df.rename(columns={'Score': 'Original Score'}, inplace=True)
        scoring_df.rename(columns={'Reversed_Score': 'Score'}, inplace=True)
    full_path = os.path.join(res_dir, f'{res_base_name}.legacy.ranks')
    with open(full_path, 'w') as handle:
        handle.write('% Note: in this file % is a comment sign.\n')
        handle.write(f'% This file converted from: {res_fn}\n')
        handle.write('%	 RESIDUE RANKS: \n')
        handle.write('% alignment#  residue#      type      rank              variability           rho     coverage\n')
        for ind in scoring_df.index:
            pos_str = str(scoring_df.loc[ind, 'Position'])
            aln_str = pos_str.rjust(10, ' ')
            res_str = aln_str
            type_str = scoring_df.loc[ind, 'Query'].rjust(10, ' ')
            rank_str = f'{scoring_df.loc[ind, "Rank"]}'.rjust(10, ' ')
            var_char_str = scoring_df.loc[ind, 'Variability_Characters'].rjust(32, ' ')
            rho_str = f'{scoring_df.loc[ind, "Score"]:.3f}'.rjust(10, ' ')
            cov_str = f'{scoring_df.loc[ind, "Coverage"]:.3f}'.rjust(10, ' ')
            handle.write(aln_str + res_str + type_str + rank_str + var_char_str + rho_str + cov_str +
                         '\n')
    return full_path


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
                             "Allowed Characters are ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R',"
                             "'S','T','V,'W','Y','-']")
    parser.add_argument('--Add_Chars', metavar='AC', type=str, nargs='*', default=[],
                        help="Add ambiguous characters to the alphabet, requires --filter_seqs = True."
                             "Supported ambiguous characters are ['B', 'X', 'Z']")
    parser.add_argument('--Write_Summary', metavar='WS', type=bool, nargs='*', default=False,
                        help="Return Summary Statistics from the run as a tsv file, including alignment size"
                             " sequence size, protien name, and run time")
    parser.add_argument('--PPI', metavar='PI', type=bool, nargs='*', default=False,
                        help="Whether or not this is a protein-protein interaction CovET run")

    arguments = parser.parse_args()
    arguments = vars(arguments)
    return arguments


if __name__ == "__main__":
    # Read input from the command line
    total_start = time()
    args = parse_args()
    query = args['query']
    aln_file = args['alignment']
    out_dir = args['output_dir']
    cores = args['processors']
    filter_seqs = args['filter_seqs']
    add_chars = args['Add_Chars']
    write_summary = args["Write_Summary"]
    ppi = args["PPI"]
    if add_chars:
        assert filter_seqs

    start_all = time()
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # Filter Out Sequences with Ambiguous Characters
    if filter_seqs:
        aln_file = remove_sequences_with_ambiguous_characters(aln_file, out_dir, additional_chars=add_chars)

    cores = 1
    cpp_function_path = "~/Desktop/Covariation-ET-faster/src/CovET_func.cpp"

    # home = os.path.expanduser('~')
    phylo_tree_fn = f'{query}_ET_blosum62_dist_et_tree.nhx'
    phylo_tree_fn = os.path.join(out_dir, phylo_tree_fn)

    msa_fn = os.path.join(out_dir, "Non-Gapped_Alignment.fa")

    pair_output_fn = f'{query}_ET_blosum62_Dist_et_Tree_All_Ranks_mismatch_diversity_Scoring.ranks'
    pair_output_fn = os.path.join(out_dir, pair_output_fn)

    single_output_fn = f'{query}_ET_blosum62_Dist_et_Tree_All_Ranks_mismatch_diversity_Scoring_Converted_To_Single_Pos_TopScoring.ranks '
    single_output_fn = os.path.join(out_dir, single_output_fn)

    legacy_output_fn = f'{query}_ET_blosum62_Dist_et_Tree_All_Ranks_mismatch_diversity_Scoring_Converted_To_Single_Pos_TopScoring.legacy.ranks '
    legacy_output_fn = os.path.join(out_dir, legacy_output_fn)

    # Generate non-gapped alignments. Then write both alignments to output dir.
    print('Step 1/4: Removing gaps in query')
    predictor = Predictor(query=query, aln_file=aln_file, out_dir=out_dir, ppi=ppi)

    # Calculate distances between sequence pairs in MSA
    print('Step 2/4: Calculating distances')
    start_dist = time()
    skip_letters = ":" if ppi else None
    calculator = AlignmentDistanceCalculator(protein=True, model="blosum62", skip_letters=skip_letters)
    _, distance_matrix, _, _ = calculator.get_et_distance(predictor.original_aln.alignment, processes=cores)
    query_sequence = predictor.non_gapped_aln.query_sequence
    end_dist = time()
    print('Calculating distances took: {} min'.format((end_dist - start_dist) / 60.0))

    # Build tree and write to file
    print('Step 3/4: Building phylogenetic tree')
    start_tree = time()
    phylo_tree = PhylogeneticTree(tree_building_method="et", tree_building_args={})
    phylo_tree.construct_tree(dm=distance_matrix)
    tree_index_mapping = {clade.name: i for i, clade in enumerate(phylo_tree.tree.get_terminals())}
    phylo_tree.write_out_tree(filename=phylo_tree_fn)
    assignments = phylo_tree.assign_group_rank(ranks=None)
    end_tree = time()
    print('Constructing tree took: {} min'.format((end_tree - start_tree) / 60.0))

    print('Step 4/4: Running CovET')
    start_calculation = time()
    fixed_penalty = Calculate_Fixed_Penalty(assignments)
    processed_tree = Generate_Processed_Tree(phylo_tree, tree_index_mapping)
    alpha_to_num, num_to_alpha = Alphabetical_to_Numeric_Mapping()
    msa_num = Generate_Numeric_MSA(predictor, alpha_to_num, tree_index_mapping)
    position_matrix = Generate_Position_Matrix(msa_num)
    pair_mapping = Generate_Pair_Mapping(msa_num, ppi=ppi)
    results_dictionary = CovET(processed_tree, msa_num, position_matrix, pair_mapping, fixed_penalty, query_sequence)
    paired_results = Convert_Pair_Mapping_to_DataFrame(results_dictionary, pair_output_fn, return_frame=True)
    # convert_file_to_legacy_format(single_output_fn)

    single_results = Convert_Paired_Results_To_Single_Residue_Rankings(paired_results,
                                                                       single_output_fn, return_frame=True)
    convert_file_to_legacy_format(single_output_fn)
    # Convert_Single_Residue_Scores_To_Legacy_Format(single_results, legacy_output_fn)
    end_calculation = time()
    print('Performing CovET Calculations took: {} min'.format((end_calculation - start_calculation) / 60.0))
    total_time = ((time() - total_start) / 60.0)

    if write_summary:
        comparison_dict = {"Protein": [query], "Alignment_Size": [msa_num.shape[0]],
                           "Sequence_Length": [msa_num.shape[1]], "Total_Time": [total_time]}
        df = pd.DataFrame.from_dict(comparison_dict)
        df.to_csv(f'{out_dir}/CovET_Run_Summary.tsv', sep='\t', index=False)
