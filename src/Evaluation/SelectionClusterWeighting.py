"""
Created on April 29, 2021

@author: Daniel Konecki
"""
import math
import numpy as np
import pandas as pd
from multiprocessing import Pool
from Evaluation.SequencePDBMap import SequencePDBMap


class SelectionClusterWeighting(object):
    """
    This class and supporting functions implement the Selection Cluster Weighting (SCW) scoring used to compute
    z-scores measuring how non-random the arrangement of top ranked residues on a protein structure is. Currently two
    of the previously published scoring methods are implemented: biased and unbiased SCW scoring. The unbiased method
    measures only whether residues are clustered on the protein structure. The biased measure determines if the
    clustering on the protein structure is significant when taking into account the sequence separation of the residues
    involved.

    Attributes:
        bias
        w_and_w2_ave_sub

    """

    def __init__(self, seq_pdb_map, pdb_dists, biased):
        """
        __init__

        Initialization for SelectionClusterWeighting class.

        Args:
            seq_pdb_map (SequencePDBMap): A mapping object containing the query, alignment, structure, and chain of
            interest for the SCW scoring analysis.
            pdb_dists (np.ndarray): The distances (in Å) between all pairs of residues in the specified chain of the
            structure of interest.
            biased (bool): Whether or not to perform the biased analysis or not. If True, the biased SCW score will be
            computed, taking residues sequence separation into account. If False, the unbiased SCW score will be
            computed, which considers only the position of residues on the protein structure into account.
        """
        # assert isinstance(seq_len, int) and (seq_len > 0), 'seq_len must be an integer > 0!'
        # self.query_seq_length = seq_len
        # assert aln_struct_map is not None, 'aln_struct_map must be provided!'
        # self.query_pdb_mapping = aln_struct_map
        # assert isinstance(pdb, PDBReference), 'pdb must be of type PDBReference!'
        # self.query_structure = pdb
        # assert chain in self.query_structure.chains, 'chain must be in pdb.chains!'
        # self.chain = chain
        assert isinstance(seq_pdb_map, SequencePDBMap), 'seq_pdb_map must be SequencePDBMap instance!'
        assert seq_pdb_map.is_aligned(), 'SequencePDBMap must be aligned before use in SelectionClusterWeighting!'
        self.query_pdb_mapper = seq_pdb_map
        assert pdb_dists is not None, 'pdb_dists must be provided!'
        self.distances = pdb_dists
        assert isinstance(biased, bool), 'biased must be of type bool!'
        self.bias = biased
        self.w_and_w2_ave_sub = None

    def compute_background_w_and_w2_ave(self, processes=1):
        """
        Compute Background W and W^2 Ave

        This function computes the pre-computable parts of the SCW Z-Score which contribute to the background (w and w^2
        average) terms. The results are stored in the w_and_w2_ave_sub attribute and can be reused if more than one
        score is being computed.

        Args:
            processes (int): The number of processes to use when computing the background terms for the SCW score.
        """
        if self.query_structure is None:
            print('SCW background cannot be measured because no PDB was provided.')
            return pd.DataFrame(), None, None
        if self.w_and_w2_ave_sub is None:
            self.w_and_w2_ave_sub = {'w_ave_pre': 0, 'Case1': 0, 'Case2': 0, 'Case3': 0}
            pool = Pool(processes=processes, initializer=init_compute_w_and_w2_ave_sub,
                        initargs=(self.distances, self.query_structure.pdb_residue_list[self.chain], bias))
            res = pool.map_async(compute_w_and_w2_ave_sub, range(self.distances.shape[0]))
            pool.close()
            pool.join()
            for cases in res.get():
                for key in cases:
                    self.w_and_w2_ave_sub[key] += cases[key]

    def clustering_z_score(self, res_list):
        """
        Clustering Z Score

        Calculate z-score (z_S) for residue selection res_list=[1,2,...]
            z_S = (w-<w>_S)/sigma_S
        The steps are:
            1. Calculate Selection Clustering Weight (SCW) 'w'
            2. Calculate mean SCW (<w>_S) in the ensemble of random selections of len(res_list) residues
            3. Calculate mean square SCW (<w^2>_S) and standard deviation (sigma_S) Reference 2

        Args:
            res_list (list): a list of int's of protein residue numbers, e.g. ET residues (residues of interest)
        Returns:
            float: The z-score calculated for the residues of interest for the PDB provided with this ContactScorer.
            float: The w (clustering) score calculated for the residues of interest for the PDB provided with this
            ContactScorer.
            float: The E[w] (clustering expectation) score calculated over all residues of the PDB provided with this
            ContactScorer.
            float: The E[w^2] score calculated over all pairs of pairs of residues for the PDB provided with this
            ContactScorer.
            float: The sigma score calculated over all pairs of pairs of residues for the PDB provided with his
            ContactScorer.
            int: The number of residues in the list of interest.

        This method was formalized by Ivana Mihalek (Reference 1) and adapted from code written by Rhonald Lua in Python
        (Reference 2) which was adapted from code written by Angela Wilkins (Reference 3).

        References:
            1. I. Mihalek, I. Res, H. Yao, O. Lichtarge, Combining Inference from Evolution and Geometric Probability in
            Protein Structure Evaluation, Journal of Molecular Biology, Volume 331, Issue 1, 2003, Pages 263-279,
            ISSN 0022-2836, https://doi.org/10.1016/S0022-2836(03)00663-6.
            (http://www.sciencedirect.com/science/article/pii/S0022283603006636)
            2. Lua RC, Lichtarge O. PyETV: a PyMOL evolutionary trace viewer to analyze functional site predictions in
            protein complexes. Bioinformatics. 2010;26(23):2981-2982. doi:10.1093/bioinformatics/btq566.
            3. Wilkins AD, Lua R, Erdin S, Ward RM, Lichtarge O. Sequence and structure continuity of evolutionary
            importance improves protein functional site discovery and annotation. Protein Sci. 2010;19(7):1296-311.
        """
        res_list = sorted(res_list)
        # Make sure all residues in res_list are mapped to the PDB structure in use
        if not all(res in self.query_pdb_mapping for res in res_list):
            print('At least one residue of interest is not present in the PDB provided')
            print(', '.join([str(x) for x in res_list]))
            print(', '.join([str(x) for x in self.query_pdb_mapping.keys()]))
            return None, None, None, None, None, None, '-', None, None, None, None, len(res_list)
        positions = list(range(distances.shape[0]))
        a = distances < cutoff
        a[positions, positions] = 0
        s_i = np.in1d(positions, [self.query_pdb_mapping[r] for r in res_list])
        s_ij = np.outer(s_i, s_i)
        s_ij[positions, positions] = 0
        if bias:
            converted_positions = np.array(self.query_structure.pdb_residue_list[self.chain])
            bias_ij = np.subtract.outer(converted_positions, converted_positions)
            bias_ij = np.abs(bias_ij)
        else:
            bias_ij = np.ones(s_ij.shape, dtype=np.float64)
        w = np.sum(np.tril(a * s_ij * bias_ij))
        # Calculate w, <w>_S, and <w^2>_S.
        # Use expressions (3),(4),(5),(6) in Reference.
        m = len(res_list)
        l = len(self.query_structure.seq[self.chain])
        pi1 = m * (m - 1.0) / (l * (l - 1.0))
        pi2 = pi1 * (m - 2.0) / (l - 2.0)
        pi3 = pi2 * (m - 3.0) / (l - 3.0)
        w_ave = self.w_and_w2_ave_sub['w_ave_pre'] * pi1
        w2_ave = ((pi1 * self.w_and_w2_ave_sub['Case1']) + (pi2 * self.w_and_w2_ave_sub['Case2']) +
                  (pi3 * self.w_and_w2_ave_sub['Case3']))
        sigma = math.sqrt(w2_ave - w_ave * w_ave)
        # Response to Bioinformatics reviewer 08/24/10
        if sigma == 0:
            return a, m, l, pi1, pi2, pi3, 'NA', w, w_ave, w2_ave, sigma, m
        z_score = (w - w_ave) / sigma
        return a, m, l, pi1, pi2, pi3, z_score, w, w_ave, w2_ave, sigma, m


def init_compute_w_and_w2_ave_sub(dists, structure_res_num, bias_bool):
    """
    Init Compute w and w^2 Average Sub-problems

    Method initializes shared values used by all processes working on parts of the w and w^2 average sub-problems.

    Args:
        dists (np.array): The array of distances between residues in the tested structure.
        structure_res_num (list/array): The residue numbers for each position in the structure as assigned in the PDB.
        bias_bool (int or bool): Option to calculate z_scores with bias (True) or no bias (False). If bias is used a j-i
        factor accounting for the sequence separation of residues, as well as their distance, is added to the
        calculation.
    """
    global distances
    distances = dists
    global w2_ave_residues
    w2_ave_residues = structure_res_num
    global bias
    bias = bias_bool
    global cutoff
    # The distance for structural clustering has historically been 4.0 Angstroms and can be found in all previous
    # versions of the code and paper descriptions, it had previously been set as a variable but this changes the meaning
    # of the result and makes it impossible to compare to previous work.
    cutoff = 4.0


def compute_w_and_w2_ave_sub(res_i):
    """
    Compute W and W^2 Average Sub-problems

    Solves the cases of the w and w^2 average sub-problem for a specific position in the protein. These can be combined
    across all positions to get the complete solution (after the pool of workers has finished).

    Args:
        res_i (int): The position in the protein for which to compute the parts of the w and w^2 average sub-problems.
    Returns:
        dict: The parts of E[w] (w_ave before multiplication with the pi1 coefficient, i.e. w_ave_pre) and E[w^2] (cases
         1, 2, and 3) which can be pre-calculated and reused for later computations.
    """
    cases = {'w_ave_pre': 0, 'Case1': 0, 'Case2': 0, 'Case3': 0}
    for res_j in range(res_i + 1, distances.shape[1]):
        if distances[res_i][res_j] >= cutoff:
            continue
        if bias:
            s_ij = np.abs(w2_ave_residues[res_j] - w2_ave_residues[res_i])
        else:
            s_ij = 1
        cases['w_ave_pre'] += s_ij
        for res_x in range(distances.shape[0]):
            for res_y in range(res_x + 1, distances.shape[1]):
                if distances[res_x][res_y] >= cutoff:
                    continue
                if bias:
                    s_xy = np.abs(w2_ave_residues[res_y] - w2_ave_residues[res_x])
                else:
                    s_xy = 1
                if (res_i == res_x and res_j == res_y) or (res_i == res_y and res_j == res_x):
                    curr_case = 'Case1'
                elif (res_i == res_x) or (res_j == res_y) or (res_i == res_y) or (res_j == res_x):
                    curr_case = 'Case2'
                else:
                    curr_case = 'Case3'
                cases[curr_case] += s_ij * s_xy
    return cases
