"""
Created on May 22, 2021

@author: Daniel Konecki
"""
import os
import numpy as np
import pandas as pd
from time import time
from pymol import cmd
from Evaluation.SequencePDBMap import SequencePDBMap


class Scorer(object):
    """
    This class is intended as the base for any scorer meant to perform evaluations on predictions of single position
    residue importance or paired position covariation predictions. It establishes a minimum set of attributes and the
    corresponding initializer as well as a method signature for the only other function all Predictor sub-classes must
    have, namely fit.

    Attributes:
        query_pdb_mapper (SequencePDBMap): A mapping from the index of the query sequence to the index of the
        pdb chain's sequence for those positions which match according to a pairwise global alignment and vice versa.
        data (panda.DataFrame): A data frame containing a mapping of the sequence, structure, and predictions as well as
        other fields which may be specific to different types of Scorer objects.
    """

    def __init__(self, query, seq_alignment, pdb_reference, chain=None):
        """
        __init__

        This function initializes a new Scorer, it accepts paths to an alignment and a PDB file to be used in the
        assessment of single residue importance or contact predictions for a given query. The PDB chain to use for this
        analysis may also be specified.

        Args:
            query (str): The name of the query sequence/structure.
            seq_alignment (str/path, SeqAlignment): Path to the alignment being evaluated in this contact scoring
            prediction task or an already initialized SeqAlignment object.
            pdb_reference (str/path, PDBReference): The path to a PDB structure file, or an already initialized
            PDBReference object.
            chain (str): Which chain in the PDB structure to use for comparison and evaluation. If left blank the best
            chain will be identified by aligning the query sequence from seq_alignment against the chains in
            pdb_reference and the closest match will be selected.
        """
        self.query_pdb_mapper = SequencePDBMap(query=query, query_alignment=seq_alignment,
                                               query_structure=pdb_reference, chain=chain)
        self.distances = None
        self.dist_type = None
        self.data = None

    def fit(self):
        """
        Fit

        This function maps sequence positions between the query sequence from the alignment and residues in PDB file.
        If there are multiple chains for the structure in the PDB, and no chain was specified upon initialization, the
        one which matches the query sequence best (highest global alignment score) is used and recorded in the
        seq_pdb_mapper variable. This method updates the seq_pdb_mapper attribute.
        """
        if not self.query_pdb_mapper.is_aligned():
            start = time()
            self.query_pdb_mapper.align()
            end = time()
            print('Mapping query sequence and pdb took {} min'.format((end - start) / 60.0))

    def _get_all_coords(self, residue):
        """
        Get All Coords

        This method retrieves the 3D coordinates for all atoms in a residue.

        Args:
            residue (Bio.PDB.Residue): The residue for which to return coordinates.
        Returns:
            list: A list of lists where each sub lists contains the x, y, and z coordinates of a given atom in a
            residue.
        """
        all_coords = []
        cmd.select('curr_res', f'best_chain and resi {residue.get_id()[1]}')
        model = cmd.get_model('curr_res')
        if len(model.atom) > 0:
            for atm in model.atom:
                all_coords.append(np.array([atm.coord[0], atm.coord[1], atm.coord[2]], dtype=np.float32))
        else:
            raise ValueError(f'No atoms found for Structure: {self.query_pdb_mapper.query} Chain: '
                             f'{self.query_pdb_mapper.best_chain} Residue: {residue.get_id()[1]}')
        return all_coords

    def _get_c_alpha_coords(self, residue):
        """
        Get C Alpha Coords

        This method retrieves the 3D coordinates for the C alpha atom of a residue. If the C alpha atom is not specified
        for the given residue _get_all_coords is called instead.

        Args:
            residue (Bio.PDB.Residue): The residue for which to return coordinates.
        Returns:
            list: A list of lists where the only sub list contains the x, y, and z coordinates of the C alpha atom.
        """
        c_alpha_coords = []
        cmd.select('curr_res_ca', f'best_chain and resi {residue.get_id()[1]} and name CA')
        model = cmd.get_model('curr_res_ca')
        if len(model.atom) > 0:
            for atm in model.atom:
                c_alpha_coords.append(np.array([atm.coord[0], atm.coord[1], atm.coord[2]], dtype=np.float32))
        else:
            c_alpha_coords = self._get_all_coords(residue)
        return c_alpha_coords

    def _get_c_beta_coords(self, residue):
        """
        Get C Beta Coords

        This method retrieves the 3D coordinates for the C beta atom of a residue. If the C beta atom is not specified
        for the given residue _get_c_alpha_coords is called instead.

        Args:
            residue (Bio.PDB.Residue): The residue for which to return coordinates.
        Returns:
            list: A list of lists where the only sub list contains the x, y, and z coordinates of the C beta atom.
        """
        c_beta_coords = []
        cmd.select('curr_res_cb', f'best_chain and resi {residue.get_id()[1]} and name CB')
        model = cmd.get_model('curr_res_cb')
        if len(model.atom) > 0:
            for atm in model.atom:
                c_beta_coords.append(np.array([atm.coord[0], atm.coord[1], atm.coord[2]], dtype=np.float32))
        else:
            c_beta_coords = self._get_c_alpha_coords(residue)
        return c_beta_coords

    def _get_coords(self, residue, method='Any'):
        """
        Get Coords

        This method serves as a switch statement, its only purpose is to reduce code duplication when calling one of
        the three specific get (all, alpha, or beta) coord functions.

        Args:
            residue (Bio.PDB.Residue): The residue for which to extract the specified atom or atoms' coordinates.
            method (str): Which method of coordinate extraction to use, expected values are 'Any', 'CA' for alpha
            carbon, or 'CB' for beta carbon.
        Returns:
            list: A list of lists where each list is a single set of x, y, and z coordinates for an atom in the
            specified residue.
        """

        if method == 'Any':
            return self._get_all_coords(residue)
        elif method == 'CA':
            return self._get_c_alpha_coords(residue)
        elif method == 'CB':
            return self._get_c_beta_coords(residue)
        else:
            raise ValueError("method variable must be either 'Any', 'CA', or 'CB' but {} was provided".format(method))

    def measure_distance(self, method='Any', save_file=None):
        """
        Measure Distance

        This method measures the distances between residues in the chain identified as the best match to the query by
        the fit function. Distance can be measured in different ways, currently three different options are supported.
        This function updates the class attributes distances and dist_type.

        Args:
            method (str): Method by which to compute distances, currently options are:
                'Any' - this measures the distance between two residues as the shortest distance between any two atoms
                in the two residues structure.
                'CA' - this measures the distance as the space between the alpha carbons of two residues. If the alpha
                carbon for a given residue is not available in the PDB structure, then the distance between any atom of
                that residue and the alpha carbons of all other residues is measured.
                'CB' - this measures the distance as the space between the beta carbons of the two residues. If the beta
                carbon of a given residue is not available in the PDB structure the alpha carbon is used, or all atoms
                in the residue if the alpha carbon is also not annotated.
            save_file (str or os.path): The path to a file where the computed distances can be stored, or may have been
            stored on a previous run (optional).
    """
        start = time()
        if self.query_pdb_mapper.pdb_ref is None:
            raise ValueError('Distance cannot be measured, because no PDB was provided.')
        elif (self.distances is not None) and (self.dist_type == method):
            return
        elif (save_file is not None) and os.path.exists(save_file):
            dists = np.load(save_file + '.npz')['dists']
        else:
            if not self.query_pdb_mapper.is_aligned():
                self.query_pdb_mapper.align()
            # Open and load PDB structure in pymol, needed for get_coord methods
            cmd.load(self.query_pdb_mapper.pdb_ref.file_name, self.query_pdb_mapper.query)
            cmd.select('best_chain', f'{self.query_pdb_mapper.query} and chain {self.query_pdb_mapper.best_chain}')
            chain_size = self.query_pdb_mapper.pdb_ref.size[self.query_pdb_mapper.best_chain]
            dists = np.zeros((chain_size, chain_size))
            coords = {}
            counter = 0
            pos = {}
            key = {}
            # Loop over all residues in the pdb
            for res_num1 in self.query_pdb_mapper.pdb_ref.pdb_residue_list[self.query_pdb_mapper.best_chain]:
                residue = self.query_pdb_mapper.pdb_ref.structure[0][self.query_pdb_mapper.best_chain][res_num1]
                # Loop over residues to calculate distance between all residues i and j
                if res_num1 not in coords:
                    pos[res_num1] = counter
                    key[counter] = res_num1
                    counter += 1
                    coords[res_num1] = np.vstack(self._get_coords(residue, method))
                for j in range(pos[res_num1]):
                    res_num2 = key[j]
                    if res_num2 not in pos:
                        continue
                    # Getting the 3d coordinates for every atom in each residue.
                    # iterating over all pairs to find all distances
                    res1 = (coords[res_num1] - coords[res_num2][:, np.newaxis])
                    norms = np.linalg.norm(res1, axis=2)
                    dists[pos[res_num1], pos[res_num2]] = dists[pos[res_num2], pos[res_num1]] = np.min(norms)
            # Delete loaded structure from pymol session
            cmd.delete(self.query_pdb_mapper.query)
            if save_file is not None:
                np.savez(save_file, dists=dists)
        end = time()
        print('Computing the distance matrix based on the PDB file took {} min'.format((end - start) / 60.0))
        self.distances = dists
        self.dist_type = method


def init_scw_z_score_selection(scw_scorer):
    """
    Init SCW Z-Score Selection

    This method initializes a set of processes in a multiprocessing pool so that they can compute the clustering Z-Score
    with minimal data duplication.

    Args:
        scw_scorer (SelectionClusterWeighting): An instance of the SelectionClusterWeighting scorer which has already
        had the precomputable parts of the score computed.
    """
    global selection_cluster_weighting_scorer
    selection_cluster_weighting_scorer = scw_scorer


def scw_z_score_selection(res_list):
    """
    SCW Z-Score Selection

    Use a pre-initialized SelectionCLusterWeighting instance to compute the SCW Z-Score for a given selection.

    Args:
        res_list (list): A list of of sequence positions to score for clustering on the protein structure.
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
    """
    res_list = sorted(res_list)
    return selection_cluster_weighting_scorer.clustering_z_score(res_list=res_list)
