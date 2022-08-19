import pandas as pd
from scipy import constants
import matplotlib.pyplot as plt
import math
import argparse
import re


def calculate_ddG(wt_affinity, mut_affinity, temp):
    """
    Calculates change in Gibbs Free Energy, using dG = RT ln (kd), given in units of Kcal (1 kcal = 4184 J)
    Args:
        wt_affinity: Kd of wild type complex in Moles
        mut_affinity: Kd of mutant complex in Moles
        temp: Temperature in Kelvin

    Returns:
        Change in Gibbs Free Energy (Kcal)
    """
    temp = re.sub('\D', '', temp)
    mut_affinity_float = float(re.sub('>', '', mut_affinity))
    wt_affinity_float = float(re.sub('>', '', wt_affinity))
    dG_mut = (constants.R / 4184) * float(temp) * math.log(mut_affinity_float)
    dG_wt = (constants.R / 4184) * float(temp) * math.log(wt_affinity_float)
    ddG = dG_mut - dG_wt
    return ddG


def format_skempi_dataset(protein1, chain1, protein2, chain2, skempi_dataset_path, output_dir):
    chain_map = {chain1: protein1, chain2: protein2}
    print(skempi_dataset_path)
    skempi = pd.read_csv(skempi_dataset_path, sep=';')
    df2 = skempi.loc[((skempi['Protein 1'] == protein1) & (skempi['Protein 2'] == protein2)) |
                     ((skempi['Protein 1'] == protein2) & (skempi['Protein 2'] == protein1))]
    # filter to max two mutations per protein pair
    df2 = df2.loc[df2['Mutation(s)_PDB'].str.count(',') < 2]
    print(df2)
    df2[["Mutation 1", "Mutation 2"]] = df2['Mutation(s)_PDB'].str.split(',', expand=True)
    df2 = df2.fillna(value='-----')

    df2['Mut_1_Start_res'] = df2['Mutation 1'].astype(str).str[0]
    df2['Mut_1_Chain'] = df2['Mutation 1'].astype(str).str[1]
    df2['Mut_1_Protein'] = df2['Mutation 1'].astype(str).str[1].map(chain_map)
    df2['Mut_1_Pos'] = df2['Mutation 1'].astype(str).str[2:-1]
    df2['Mut_1_End_res'] = df2['Mutation 1'].astype(str).str[-1]

    df2['Mut_2_Start_res'] = df2['Mutation 2'].astype(str).str[0]
    df2['Mut_2_Chain'] = df2['Mutation 2'].astype(str).str[1]
    df2['Mut_2_Protein'] = df2['Mutation 2'].astype(str).str[1]
    df2['Mut_2_Pos'] = df2['Mutation 2'].astype(str).str[2:-1]
    #get df2 mut_2_pos_ranks_matched
        # df2['Mut_2_pos_ranks_matched'] = df2["mut_2_pos"] + position_2_start_index if mut_2_protein == protein2 using df.apply
    df2['Mut_2_End_res'] = df2['Mutation 2'].astype(str).str[-1]

    df2['ddG'] = df2.apply(lambda x: calculate_ddG(x['Affinity_wt (M)'], x['Affinity_mut (M)'], x['Temperature']),
                           axis=1)
    out_frame = df2[['Protein 1', 'Protein 2', 'Affinity_mut (M)', 'Affinity_mut_parsed', 'Affinity_wt (M)',
                     'Temperature', 'Mutation 1', 'Mutation 2', 'Mut_1_Start_res', 'Mut_1_Chain', 'Mut_1_Protein',
                     'Mut_1_Pos', 'Mut_1_End_res', 'Mut_2_Start_res', 'Mut_2_Chain', 'Mut_2_Protein', 'Mut_2_Pos',
                     'Mut_2_End_res', 'ddG']]

    inter_double_mutants = out_frame.loc[(out_frame['Mutation 2'] != '-----') &
                                         (out_frame['Mut_2_Protein'] != out_frame['Mut_1_Protein'])]
    inter_double_mutants.to_csv(f'{output_dir}/{protein1}-{protein2}_paired_mutagenesis_data.tsv', sep='\t')

    intra_double_mutants = out_frame.loc[(out_frame['Mutation 2'] != '-----') &
                                         (out_frame['Mut_2_Protein'] == out_frame['Mut_1_Protein'])]
    intra_double_mutants.to_csv(f'{output_dir}/{protein1}_or_{protein2}_double_mutants_mutagenesis_data.tsv', sep='\t')

    single_mutants = out_frame.loc[out_frame['Mutation 2'] == '-----']
    single_mutants.to_csv(f'{output_dir}/{protein1}_or_{protein2}_single_mutants_mutagenesis_data.tsv', sep='\t')
    return inter_double_mutants, intra_double_mutants, single_mutants

def return_start_of_second_protein(Series):
    return [y for x, y in zip(Series, Series[1:]) if not x + 1 == y][0]


def map_skempi_ddg_to_ranks(ranks_path, mutations):
    if mutations['Mut_1_Protein'].equals(mutations['Mut_2_Protein']):
        analysis_type = 'Intra'
        res_index_col = 'Position_i'
    elif (mutations['Mutation 2'] == '-----').all():
        analysis_type = 'Single'
        res_index_col = 'Position'
    elif not mutations['Mut_1_Protein'].equals(mutations['Mut_2_Protein']):
        analysis_type = "Inter"
        ranks = pd.read_csv(ranks_path, sep='\t', index_col=False)
        n_res_protein1 = len(ranks['Position_i'].unique())
    else:
        return ValueError, "The given mutations are not all of the same type, or not Inter, Intra, or Single"


    ranks['ddG'] = '---'


    for ind, row in enumerate(mutations.iterrows()):
        seq_index_protein_1 = int(row[1]['Mut_1_Pos']) - 1
        seq_index_protein_2 = int(row[1]['Mut_2_Pos']) + n_res_protein1
        protein_1_res = row[1]['Mut_1_Start_res']
        protein_2_res = row[1]['Mut_2_Start_res']

        if analysis_type == "Inter":
            ranks_resi, ranks_resj = \
                ranks.loc[(ranks['Position_i'] == seq_index_protein_1) & (ranks['Position_j'] == seq_index_protein_2),
                          ['Query_i', 'Query_j']].values[0]

            index_check = ((protein_1_res == ranks_resi) + (protein_2_res == ranks_resj) == 2)
            if not index_check:
                print(f' WARNING: row {ind} of the skempi dataframe has a mismatch between CovET and '
                      f'Skempi sequences, these will not be included in the analysis')
                continue

            ranks.loc[(ranks['Position_i'] == seq_index_protein_1) & (ranks['Position_j'] == seq_index_protein_2),
                      'ddG'] = row[1]['ddG']

        elif analysis_type == "Intra":
            #need a standard CovET
            print("This program is not currently set up to analyze intra protein couplings")
        else:





        mutated_pairs = ranks.loc[ranks['ddG'] != '---']
    return mutated_pairs


def CovET_Score_v_other_scatterplot(dataframe, other_column, out_dir, other_axlab='', covETcolumn='Score', title=''):
    if not other_axlab:
        other_axlab = other_column
    pearsonsr = round((dataframe[covETcolumn].astype(float)).corr(dataframe[other_column].astype(float)), 2)
    plt.scatter(dataframe[covETcolumn], dataframe[other_column])
    plt.xlabel(f'CovET {covETcolumn}')
    plt.ylabel(other_axlab)
    plt.title(title)
    plt.text(0.75, 0.98, f'Pearsons r: {pearsonsr}', transform=plt.gca().transAxes)
    plt.savefig(f'{output_dir}/CovET_{covETcolumn}vs{other_column}.png')
    plt.clf()


def parse_args():
    """
    Parse Arguments
    This method processes the command line arguments for CovET.
    """
    # Create input parser
    parser = argparse.ArgumentParser(description='Process CovET options.')
    # Mandatory arguments (no defaults)
    parser.add_argument('--protein1', metavar='P1', type=str, nargs='?', required=True,
                        help='The name of the first protein being queried in this analysis (should be the sequence identifier'
                             ' in the skempiDB).')
    parser.add_argument('--protein2', metavar='P2', type=str, nargs='?', required=True,
                        help='The name of the second protein being queried in this analysis (should be the sequence identifier'
                             ' in the skempiDB).')
    parser.add_argument('--chain1', metavar='C1', type=str, nargs='?', required=True,
                        help='What chain the first protein is in the PDB')
    # Optional argument (has default)
    parser.add_argument('--chain2', metavar='C2', type=str, nargs='?', required=True,
                        help="What chain the second protein is in the PDB")
    parser.add_argument('--output_dir', metavar='F', type=str, nargs='?', required=True,
                        help="Where results should be written")
    parser.add_argument('--skempi', metavar='S', type=str, nargs='?', required=True,
                        help="Path to the skempi database")
    parser.add_argument('--ranks', metavar='R', type=str, nargs='?', required=True,
                        help="Path to CovET ranks file")
    parser.add_argument('--preset', metavar='p', type=str, nargs='*', default='ddG',
                        help="presets: currently available are ddg")

    arguments = parser.parse_args()
    arguments = vars(arguments)
    return arguments


if __name__ == "__main__":
    args = parse_args()
    protein1 = args['protein1']
    print(protein1)
    protein2 = args['protein2']
    chain1 = args['chain1']
    chain2 = args['chain2']
    skempi_path = args['skempi']
    output_dir = args['output_dir']
    ranks_path = args['ranks']
    preset = args['preset']
    if preset == 'ddG':
        comparison_col = 'ddG'
        title = 'CovET Score vs Change in Change in Gibbs Free Energy Upon Binding'

    ranks = pd.read_csv(ranks_path, sep='\t', index_col=None)
    if "Position_i" in ranks.columns:
        protein_2_start_position = len(ranks['Position_i'].unique()) +1
    else:
        protein_2_start_position = return_start_of_second_protein(ranks["Position"])

    interDF, intraDF, singleDF = format_skempi_dataset(protein1, chain1, protein2, chain2, skempi_path,
                                                       output_dir, protein_2_start_position)
    print(interDF)
    mutated_pairs = map_skempi_ddg_to_ranks(ranks_path, interDF)
    print(mutated_pairs)
    CovET_Score_v_other_scatterplot(mutated_pairs,
                                    comparison_col,
                                    output_dir,
                                    title=title)

# protein1 = 'Barnase'
# protein2 = 'Barstar'
# chain1 = 'A'
# chain2 = 'B'
# skempi_dataset_path = '/home/spencer/PPI-CovET/Paired_Mutagenesis_Dataset/skempi_v2.csv'
# output_dir = '/home/spencer/PPI-CovET/Paired_Mutagenesis_Dataset/Barnase-Barstar/'
# ranks_path = '/home/spencer/PPI-CovET/Paired_Mutagenesis_Dataset/Barnase-Barstar/Output' \
#              '/BARNASE_BARSTAR_ET_blosum62_Dist_et_Tree_All_Ranks_mismatch_diversity_Scoring.ranks '
# interDF, intraDF, singleDF = format_skempi_dataset(protein1, chain1, protein2, chain2, skempi_dataset_path, output_dir)
# mutated_pairs = map_skempi_ddg_to_ranks(ranks_path, interDF)
# CovET_Score_v_other_scatterplot(mutated_pairs,
#                                 'ddG',
#                                 output_dir,
#                                 title='CovET Score vs Change in Change in Gibbs Free Energy Upon Binding')
