"""
Created on Aug 21, 2017

@author: dmkonecki
"""
import os
import Queue
import numpy as np
from time import time
from multiprocessing import Manager, Pool
from sklearn.metrics import mutual_info_score
from ContactScorer import heatmap_plot, surface_plot


class ETMIPC(object):
    """
    classdocs
    """

    # def __init__(self, alignment, clusters, pdb, output_dir, processes, low_memory_mode=False):
    def __init__(self, alignment):
        """
        Constructor

        Initiates an instance of the ETMIPC class which stores the
        following data:

        alignment : SeqAlignment
            The SeqAlignment object containing relevant information for this
            ETMIPC analysis.
        clusters : list
            The k's for which to create different clusterings.
        sub_alignments : dict
            A dictionary mapping a clustering constant (k) to another dictionary
            which maps a cluster label (0 to k-1) to a SeqAlignment object
            containing only the sequences for that specific cluster.
        pdb : PDBReference
            The PDBReference object containing relevant information for this
            ETMIPC analysis.
        output_dir : str
            Directory name or path to directory where results from this analysis
            should be stored.
        processes : int
            The number of processes to spawn for more intense computations
            performed during this analysis.  If the number is higher than the
            number of jobs required to quickly perform this analysis the number
            of jobs is used as the number of processes.  If the number of
            processes is higher than the number of processors available to the
            program, the number of processors available is used instead.
        whole_mip_matrix : np.array
            Matrix scoring the coupling between all positions in the query
            sequence, as computed over all sequences in the input alignment.
        whole_evidence_matrix : np.array
            Matrix containing the number of sequences which are not gaps in
            either position used for scoring the whole_mip_matrix.
        result_times : dict
            Dictionary mapping k constant to the amount of time it took to
            perform analysis for that k constant.
        raw_scores : dict
            This dictionary maps clustering constant to an k x m x n matrix.
            This matrix has coupling scores for all positions in the query
            sequences for each of the clusters created by hierarchical
            clustering.
        result_matrices : dict
            This dictionary maps clustering constants to a matrix scoring the
            coupling between all residues in the query sequence over all of the
            clusters created at that constant.
        evidence_counts : dict
            This dictionary maps clustering constants to a matrix which has
            counts for the number of sequences which are not gaps in
            either position used for scoring at that position.
        summary_matrices : dict
            This dictionary maps clustering constants to a matrix which combines
            the scores from the whole_mip_matrix, all lower clustering constants,
            and this clustering constant.
        coverage : dict
            This dictionary maps clustering constants to a matrix of normalized
            coupling scores between 0 and 100, computed from the
            summary_matrices.
        aucs : dict
            This dictionary maps clustering constants to a tuple containing the
            auc, tpr, and fpr for comparing the coupled scores to the PDB
            reference (if provided) at a specified distance threshold.
        low_memory_mode: bool
            This boolean specifies whether or not to run in low memory mode. If
            True is specified a majority of the class variables are set to None
            and the data is saved to disk at run time and loaded when needed for
            downstream analysis. The intermediate files generated in this way
            can be removed using clear_intermediate_files. The default value is
            False, in which case all variables are kept in memory.
        """
        self.alignment = alignment
        # self.output_dir = output_dir
        self.output_dir = None
        # self.clusters = clusters
        self.clusters = None
        self.whole_mip_matrix = None
        # self.low_mem = low_memory_mode
        self.low_mem = None
        # self.sub_alignments = {c: {} for c in self.clusters}
        self.sub_alignments = None
        # self.pdb = pdb
        # self.processes = processes
        self.processes = None
        self.whole_evidence_matrix = None
        # self.time = {c: 0.0 for c in self.clusters}
        self.time = None
        # if low_memory_mode:
        self.raw_scores = None
        self.evidence_counts = None
        self.result_matrices = None
        # self.summary_matrices = None
        self.scores = None
        self.coverage = None
        # else:
        #     self.raw_scores = {c: np.zeros((c, self.alignment.seq_length, self.alignment.seq_length))
        #                        for c in self.clusters}
        #     self.evidence_counts = {c: np.zeros((c, self.alignment.seq_length, self.alignment.seq_length))
        #                             for c in self.clusters}
        #     self.result_matrices = {c: None for c in self.clusters}
        #     self.summary_matrices = {c: np.zeros((self.alignment.seq_length, self.alignment.seq_length))
        #     self.scores = {c: np.zeros((self.alignment.seq_length, self.alignment.seq_length))
        #                              for c in self.clusters}
        #     self.coverage = {c: np.zeros((self.alignment.seq_length, self.alignment.seq_length))
        #                      for c in self.clusters}
        self.aucs = None
        # for k in self.clusters:
        #     c_out_dir = os.path.join(self.output_dir, str(k))
        #     if not os.path.exists(c_out_dir):
        #         os.mkdir(c_out_dir)

    def import_alignment(self, query, aa_dict, ignore_alignment_size=False):
        # This should be moved to top once it works!
        from SeqAlignment import SeqAlignment
        print 'Importing alignment'
        # Create SeqAlignment object to represent the alignment for this analysis.
        query_alignment = SeqAlignment(file_name=self.alignment, query_id=query)
        # Import alignment information from file.
        query_alignment.import_alignment(save_file=os.path.join(self.output_dir, 'alignment.pkl'))
        # Check if alignment meets analysis criteria:
        if (not ignore_alignment_size) and (query_alignment.size < 125):
            raise ValueError('The multiple sequence alignment is smaller than recommended for performing this analysis '
                             '({} < 125, see PMID:16159918), if you wish to proceed with the analysis anyway please '
                             'call the code again using the --ignore_alignment_size option.'.format(query_alignment.size))
        # Remove gaps from aligned query sequences
        query_alignment.remove_gaps(save_file=os.path.join(self.output_dir, 'ungapped_alignment.pkl'))
        # Create matrix converting sequences of amino acids to sequences of integers
        # representing sequences of amino acids.
        query_alignment.alignment_to_num(aa_dict)
        # Write the ungapped alignment to file.
        query_alignment.write_out_alignment(file_name=os.path.join(self.output_dir, 'UngappedAlignment.fa'))
        # Compute distance between all sequences in the alignment
        query_alignment.compute_distance_matrix(save_file=os.path.join(self.output_dir, 'X'))
        # Determine the full clustering tree for the alignment and the ordering of
        # its sequences.
        query_alignment.set_tree_ordering()
        print('Query Sequence:')
        print(query_alignment.query_sequence)
        self.alignment = query_alignment

    def determine_whole_mip(self, evidence):
        """
        determine_whole_mip

        Paramters:
        evidence : bool
            Whether or not to normalize using the evidence using the evidence
            counts computed while performing the coupling scoring.

        This method performs the whole_analysis method on all sequences in the
        sequence alignment. This method updates the whole_mip_matrix and
        whole_evidence_matrix class variables.
        """
        mip_matrix, evidence_counts = whole_analysis(self.alignment, evidence,
                                                     save_file=os.path.join(self.output_dir, 'wholeMIP'))
        self.whole_mip_matrix = mip_matrix
        self.whole_evidence_matrix = evidence_counts

    def calculate_clustered_mip_scores(self, aa_dict, combine_clusters):
        """
        Calculate Clustered MIP Scores

        This method calculates the coupling scores for subsets of sequences
        from the alignment as determined by hierarchical clustering on the
        distance matrix between sequences of the alignment. This method updates
        the result_matrices, result_times, and raw_scores class variables.

        Parameters:
        -----------
        aa_dict : dict
            A dictionary mapping amino acids to numerical representations.
        combine_clusters : str
            Method by which to combine individual matrices from one round of
            clustering. The options supported now are: sum, average,
            size_weighted, evidence_weighted, and evidence_vs_size.
        """
        cetmip_manager = Manager()
        k_queue = cetmip_manager.Queue()
        sub_alignment_queue = cetmip_manager.Queue()
        res_queue = cetmip_manager.Queue()
        alignment_lock = cetmip_manager.Lock()
        for k in self.clusters:
            k_queue.put(k)
        if self.processes == 1:
            pool_init1(aa_dict, combine_clusters, self.alignment, self.output_dir, alignment_lock, k_queue,
                       sub_alignment_queue, res_queue, self.low_mem)
            cluster_sizes, sub_alignments, cluster_times = et_mip_worker1((1, 1))
            print sub_alignment_queue.qsize()
            self.time = cluster_times
            self.sub_alignments = sub_alignments
        else:
            pool = Pool(processes=self.processes, initializer=pool_init1,
                        initargs=(aa_dict, combine_clusters, self.alignment, self.output_dir, alignment_lock, k_queue,
                                  sub_alignment_queue, res_queue, self.low_mem))
            pool_res = pool.map_async(et_mip_worker1, [(x + 1, self.processes) for x in range(self.processes)])
            pool.close()
            pool.join()
            cluster_dicts = pool_res.get()
            cluster_sizes = {}
            for cD in cluster_dicts:
                for c in cD[0]:
                    if c not in cluster_sizes:
                        cluster_sizes[c] = {}
                    for s in cD[0][c]:
                        cluster_sizes[c][s] = cD[0][c][s]
                for c in cD[1]:
                    for s in cD[1][c]:
                        self.sub_alignments[c][s] = cD[1][c][s]
                for c in cD[2]:
                    self.time[c] += cD[2][c]
        # Retrieve results
        while (not self.low_mem) and (not res_queue.empty()):
            r = res_queue.get_nowait()
            self.raw_scores[r[0]][r[1]] = r[2]
            self.evidence_counts[r[0]][r[1]] = r[3]
#             self.time[r[0]] += r[4]
        # Combine results
        for c in self.clusters:
            if self.low_mem:
                curr_raw_scores, curr_evidence = load_raw_score_matrix(
                    self.alignment.seq_length, c, self.output_dir)
            else:
                curr_raw_scores = self.raw_scores[c]
                curr_evidence = self.evidence_counts[c]
            start = time()
            # Additive clusters
            if combine_clusters == 'sum':
                res_matrix = np.sum(curr_raw_scores, axis=0)
            # Normal average over clusters
            elif combine_clusters == 'average':
                res_matrix = np.mean(curr_raw_scores, axis=0)
            # Weighted average over clusters based on cluster sizes
            elif combine_clusters == 'size_weighted':
                weighting = np.array([cluster_sizes[c][s]
                                      for s in sorted(cluster_sizes[c].keys())])
                res_matrix = weighting[:, None, None] * curr_raw_scores
                res_matrix = np.sum(res_matrix, axis=0) / self.alignment.size
            # Weighted average over clusters based on evidence counts at each
            # pair vs. the number of sequences with evidence for that pairing.
            elif combine_clusters == 'evidence_weighted':
                res_matrix = (np.sum(curr_raw_scores * curr_evidence, axis=0) / np.sum(curr_evidence, axis=0))
            # Weighted average over clusters based on evidence counts at each
            # pair vs. the entire size of the alignment.
            elif combine_clusters == 'evidence_vs_size':
                res_matrix = (np.sum(curr_raw_scores * curr_evidence, axis=0) / float(self.alignment.size))
            else:
                print 'Combination method not yet implemented'
                raise NotImplementedError()
            res_matrix[np.isnan(res_matrix)] = 0.0
            if self.low_mem:
                save_single_matrix('Result', c, res_matrix, self.output_dir)
            else:
                self.result_matrices[c] = res_matrix
            end = time()
            self.time[c] += end - start

    def combine_clustering_results(self, combination):
        """
        Combine Clustering Result

        This method combines data from whole_mip_matrix and result_matrices to
        populate the summary_matrices.  The combination occurs by simple addition
        but if average is specified it is normalized by the number of elements
        added.

        Parameters:
        -----------
        combination: str
            Method by which to combine scores across clustering constants. By
            default only a sum is performed, the option average is also
            supported.
        """
        start = time()
        for i in range(len(self.clusters)):
            curr_clus = self.clusters[i]
            if self.low_mem:
                curr_summary = np.zeros((self.alignment.seq_length, self.alignment.seq_length))
            else:
                # curr_summary = self.summary_matrices[curr_clus]
                curr_summary = self.scores[curr_clus]
            curr_summary += self.whole_mip_matrix
            for j in [c for c in self.clusters if c <= curr_clus]:
                if self.low_mem:
                    curr_summary += load_single_matrix('Result', j, self.output_dir)
                else:
                    curr_summary += self.result_matrices[j]
            if combination == 'average':
                curr_summary /= (i + 2)
            if self.low_mem:
                save_single_matrix('Summary', curr_clus, curr_summary, self.output_dir)
        end = time()
        print('Combining data across clusters took {} min'.format(
            (end - start) / 60.0))

    # def compute_coverage_and_auc(self, contact_scorer):
    def compute_coverage(self):
        """
        Compute Coverage And AUC

        This method computes the coverage/normalized coupling scores between
        residues in the query sequence as well as the AUC summary values
        determined when comparing the coupling score to the distance between
        residues in the PDB file. This method updates the coverage, result_times,
        and aucs variables.

        Parameters:
        -----------
        threshold : float
            Distance in Angstroms between residues considered as a preliminary
            positive set of coupled residues based on spatial positions in
            the PDB file if provided.
        """
        start = time()
        if self.processes == 1:
            res2 = []
            for clus in self.clusters:
                # pool_init2(contact_scorer, self.output_dir)
                pool_init2(self.output_dir)
                # r = et_mip_worker2((clus, self.summary_matrices))
                r = et_mip_worker2((clus, self.scores))
                res2.append(r)
        else:
            # pool2 = Pool(processes=self.processes, initializer=pool_init2, initargs=(contact_scorer, self.output_dir))
            pool2 = Pool(processes=self.processes, initializer=pool_init2, initargs=(self.output_dir))
            # res2 = pool2.map_async(et_mip_worker2, [(clus, self.summary_matrices) for clus in self.clusters])
            res2 = pool2.map_async(et_mip_worker2, [(clus, self.scores) for clus in self.clusters])
            pool2.close()
            pool2.join()
            res2 = res2.get()
        for r in res2:
            if not self.low_mem:
                self.coverage[r[0]] = r[1]
            self.time[r[0]] += r[2]
            # self.aucs[r[0]] = r[3:]
        end = time()
        print('Computing coverage and AUC took {} min'.format((end - start) / 60.0))

    def write_out_scores(self, today):
        """
        Produce Final Figures

        This method writes out clustering scores and additional results, as well
        as plotting heatmaps and surface plots of the coupling data for the
        query sequence. This method updates the result_times class variable.

        Parameters:
        -----------
        today : str
            The current date which will be used for identifying the proper
            directory to store files in.
        """
        begin = time()
        q_name = self.alignment.query_id.split('_')[1]
        pool_manager = Manager()
        cluster_queue = pool_manager.Queue()
        for c in self.clusters:
            cluster_queue.put_nowait(c)
        if self.processes == 1:
            pool_init_new(cluster_queue, q_name, today, self.whole_mip_matrix, self.raw_scores, self.result_matrices,
                       # self.coverage, self.summary_matrices, self.alignment, self.output_dir)
                       self.coverage, self.scores, self.alignment, self.output_dir)
            et_mip_worker_new((1, 1))
        else:
            pool = Pool(processes=self.processes, initializer=pool_init_new,
                        initargs=(cluster_queue, q_name, today, self.whole_mip_matrix, self.raw_scores,
                                  # self.result_matrices, self.coverage, self.summary_matrices, self.alignment,
                                  self.result_matrices, self.coverage, self.scores, self.alignment,
                                  self.output_dir))
            res = pool.map_async(et_mip_worker_new, [(x + 1, self.processes) for x in range(self.processes)])
            pool.close()
            pool.join()
            for times in res.get():
                for c in times:
                    self.time[c] += times[c]
        finish = time()
        print('Producing final figures took {} min'.format(
            (finish - begin) / 60.0))

    def calculate_scores(self, out_dir, today, query, clusters, aa_dict, combine_clusters, combine_ks,
                         processes=1, low_memory_mode=False, ignore_alignment_size=False):
        #
        # This code has not been tested!
        # Move AUC computation out of this class
        # Move figure generation out of this class
        # Mover summary file writing out of this class
        #
        serialized_path = os.path.join(out_dir, 'cET-MIp.npz')
        if os.path.isfile(serialized_path):
            loaded_data = np.load(serialized_path)
            self.scores = loaded_data['scores']
            self.coverage = loaded_data['coverage']
            self.raw_scores = loaded_data['raw']
            self.result_matrices = loaded_data['res']
            self.time = loaded_data['time']
        else:
            self.output_dir = out_dir
            start = time()
            self.import_alignment(query=query, ignore_alignment_size=ignore_alignment_size)
            if self.alignment.size < max(clusters):
                raise ValueError('The analysis could not be performed because the alignment has fewer sequences than the '
                                 'requested number of clusters ({} < {}), please provide an alignment with more sequences '
                                 'or change the clusters requested by using the --clusters option when using this '
                                 'software.'.format(self.alignment.size, max(clusters)))
            self.clusters = clusters
            self.sub_alignments = {c: {} for c in self.clusters}
            self.time = {c: 0.0 for c in self.clusters}
            for k in self.clusters:
                c_out_dir = os.path.join(self.output_dir, str(k))
                if not os.path.exists(c_out_dir):
                    os.mkdir(c_out_dir)
            self.low_mem = low_memory_mode
            if not self.low_mem:
                self.raw_scores = {c: np.zeros((c, self.alignment.seq_length, self.alignment.seq_length))
                                       for c in self.clusters}
                self.evidence_counts = {c: np.zeros((c, self.alignment.seq_length, self.alignment.seq_length))
                                        for c in self.clusters}
                self.result_matrices = {c: None for c in self.clusters}
                # self.summary_matrices = {c: np.zeros((self.alignment.seq_length, self.alignment.seq_length))
                self.scores = {c: np.zeros((self.alignment.seq_length, self.alignment.seq_length))
                                         for c in self.clusters}
                self.coverage = {c: np.zeros((self.alignment.seq_length, self.alignment.seq_length))
                                 for c in self.clusters}
            self.determine_whole_mip(evidence=('evidence' in combine_clusters))
            self.processes = processes
            self.calculate_clustered_mip_scores(aa_dict=aa_dict, combine_clusters=combine_clusters)
            self.combine_clustering_results(combination=combine_ks)
            self.compute_coverage()
            self.write_out_scores(today=today)
            end = time()
            self.time['Total'] = end - start
            np.savez(serialized_path, scores=self.scores, coverage=self.coverage, raw=self.raw_scores,
                     res=self.result_matrices, time=self.time)
            if self.low_mem:
                self.clear_intermediate_files()
        return self.time

    def produce_final_figures(self, today, scorer, verbosity):
        """
        Produce Final Figures

        This method writes out clustering scores and additional results, as well
        as plotting heatmaps and surface plots of the coupling data for the
        query sequence. This method updates the result_times class variable.

        Parameters:
        -----------
        today : str
            The current date which will be used for identifying the proper
            directory to store files in.
        cut_off : float
            Distance in Angstroms between residues considered as a preliminary
            positive set of coupled residues based on spatial positions in
            the PDB file if provided.
        verbosity : int
            How many figures to produce.1 = ROC Curves, ETMIP Coverage file,
            and final AUC and Timing file. 2 = files with all scores at each
            clustering. 3 = sub-alignment files and plots. 4 = surface plots
            and heatmaps of ETMIP raw and coverage scores.'
        """
        begin = time()
        q_name = self.alignment.query_id.split('_')[1]
        pool_manager = Manager()
        cluster_queue = pool_manager.Queue()
        output_queue = pool_manager.Queue()
        for c in self.clusters:
            cluster_queue.put_nowait(c)
        if self.processes == 1:
            pool_init3(cluster_queue, output_queue, q_name, today, verbosity, self.whole_mip_matrix, self.raw_scores,
                       # self.result_matrices, self.coverage, self.summary_matrices, self.sub_alignments, self.alignment,
                       self.result_matrices, self.coverage, self.scores, self.sub_alignments, self.alignment,
                       self.aucs, scorer, self.output_dir)
            et_mip_worker3((1, 1))
        else:
            pool = Pool(processes=self.processes, initializer=pool_init3,
                        initargs=(cluster_queue, output_queue, q_name, today, verbosity, self.whole_mip_matrix,
                                  # self.raw_scores, self.result_matrices, self.coverage, self.summary_matrices,
                                  self.raw_scores, self.result_matrices, self.coverage, self.scores,
                                  self.sub_alignments, self.alignment, self.aucs, scorer, self.output_dir))
            res = pool.map_async(et_mip_worker3, [(x + 1, self.processes)
                                                  for x in range(self.processes)])
            pool.close()
            pool.join()
            for times in res.get():
                for c in times:
                    self.time[c] += times[c]
        finish = time()
        print('Producing final figures took {} min'.format(
            (finish - begin) / 60.0))

    def write_final_results(self, today, cutoff):
        """
        Write final results

        This method writes the final results to file for an analysis.  In this case
        that consists of the cluster numbers, the resulting AUCs, and the time
        spent in processing.

        Parameters:
        -----------
        today: str
            The current date in string format.
        cutoff: float
            The distance threshold for interaction between two residues in a
            protein structure.
        """
        start = time()
        q_name = self.alignment.query_id.split('_')[1]
        o = '{}_{}etmipAUC_results.txt'.format(q_name, today)
        outfile = open(o, 'w+')
        outfile.write(
            "Protein/id: {} Alignment Size: {} Length of protein: {} Cutoff: {}\n".format(
                q_name, self.alignment.size, self.alignment.seq_length, cutoff))
        outfile.write("#OfClusters\tAUC\tRunTime\n")
        for c in self.clusters:
            # if self.pdb:
            if self.aucs:
                outfile.write("\t{0}\t{1}\t{2}\n".format(c, round(self.aucs[c][2], 4), round(self.time[c], 4)))
            else:
                outfile.write("\t{0}\t{1}\t{2}\n".format(c, '-', round(self.time[c], 4)))
        end = time()
        print('Writing final results took {} min'.format((end - start) / 60.0))

    def clear_intermediate_files(self):
        """
        Clear Intermediate Files

        This method is intended to be used only if the ETMIPC low_mem variable is
        set to True. If this is the case and the complete analysis has been
        performed then this function will remove all intermediate file generated
        during execution.
        """
        for k in self.clusters:
            res_path = os.path.join(self.output_dir, str(k), 'K{}_Result.npz'.format(k))
            os.remove(res_path)
            summary_path = os.path.join(self.output_dir, str(k), 'K{}_Summary.npz'.format(k))
            os.remove(summary_path)
            coverage_path = os.path.join(self.output_dir, str(k), 'K{}_Coverage.npz'.format(k))
            os.remove(coverage_path)
            for sub in range(k):
                curr_path = os.path.join(self.output_dir, str(k), 'K{}_Sub{}.npz'.format(k, sub))
                os.remove(curr_path)
###############################################################################
#
###############################################################################


def save_raw_score_matrix(k, sub, mat, evidence, out_dir):
    """
    Save Raw Score Matrix

    This function can be used to save the rawScore and evidence_counts matrices
    which need to be saved to disk in order to reduce the memory footprint when
    the ETMIPC variable low_mem is set to true.

    Parameters:
    -----------
    k : int
        An integer specifying which clustering constant to load data for.
    sub : int
        An integer specifying the the cluster for which to save data (expected
        values are in range(0, k)).
    mat : np.array
        The array for the rawScore data to save for the specified cluster.
    evidence : np.array
        The array for the evidence_counts data to save for the specified cluster.
    out_dir : str
        The top level directory where data are being stored, where directories
        for each k can be found.
    """
    c_out_dir = os.path.join(out_dir, str(k))
    np.savez(os.path.join(c_out_dir, 'K{}_Sub{}.npz'.format(k, sub)), mat=mat,
             evidence=evidence)


def load_raw_score_matrix(seq_len, k, out_dir):
    """
    Load Raw Score Matrix

    This function can be used to load the rawScore and evidence_counts matrices
    which need to be saved to disk in order to reduce the memory footprint when
    the ETMIPC variable low_mem is set to true.

    Parameters:
    -----------
    k : int
        An integer specifying which clustering constant to load data for.
    sub : int
        An integer specifying the the cluster for which to save data (expected
        values are in range(0, k)).
    out_dir : str
        The top level directory where data are being stored, where directories
        for each k can be found.
    Returns:
    --------
    np.array
        The array for the rawScore data to save for the specified cluster.
    np.array
        The array for the evidence_counts data to save for the specified cluster.
    """
    mat = np.zeros((k, seq_len, seq_len))
    evidence = np.zeros((k, seq_len, seq_len))
    for sub in range(k):
        load_path = os.path.join(out_dir, str(k), 'K{}_Sub{}.npz'.format(k, sub))
        data = np.load(load_path)
        c_mat = data['mat']
        e_mat = data['evidence']
        mat[sub] = c_mat
        evidence[sub] = e_mat
    return mat, evidence


def save_single_matrix(name, k, mat, out_dir):
    """
    Save Single Matrix

    This function can be used to save any of the several matrices which need to
    be saved to disk in order to reduce the memory footprint when the ETMIPC
    variable low_mem is set to true.

    Parameters:
    -----------
    name : str
        A string specifying what kind of data is being stored, expected values
        include:
            Result
            Summary
            Coverage
    k : int
        An integer specifying which clustering constant to load data for.
    mat : np.array
        The array for the given type of data to save for the specified cluster.
    out_dir : str
        The top level directory where data are being stored, where directories
        for each k can be found.
    """
    c_out_dir = os.path.join(out_dir, str(k))
    np.savez(os.path.join(c_out_dir, 'K{}_{}.npz'.format(k, name)), mat=mat)


def load_single_matrix(name, k, out_dir):
    """
    Load Single Matrix

    This function can be used to load any of the several matrices which are
    saved to disk in order to reduce the memory footprint when the ETMIPC
    variable low_mem is set to true.

    Parameters:
    -----------
    name : str
        A string specifying what kind of data is being stored, expected values
        include:
            Result
            Summary
            Coverage
    k : int
        An integer specifying which clustering constant to load data for.
    out_dir : str
        The top level directory where data are being stored, where directories
        for each k can be found.
    Returns:
    --------
    np.array
        The array for the given type of data loaded for the specified cluster.
    """
    data = np.load(os.path.join(out_dir, str(k), 'K{}_{}.npz'.format(k, name)))
    return data['mat']


def whole_analysis(alignment, evidence, save_file=None):
    """
    Whole Analysis

    Generates the MIP matrix.

    Parameters:
    -----------
    alignment: SeqAlignment
        A class containing the query sequence alignment in different formats,
        as well as summary values.
    evidence : bool
        Whether or not to normalize using the evidence using the evidence
        counts computed while performing the coupling scoring.
    save_file: str
        File path to a previously stored MIP matrix (.npz should be excluded as
        it will be added automatically).
    Returns:
    --------
    matrix
        Matrix of MIP scores which has dimensions seq_length by seq_length.
    matrix
        Matrix containing the number of sequences which are not gaps in either
        position used for scoring the whole_mip_matrix.
    """
    start = time()
    if (save_file is not None) and os.path.exists(save_file + '.npz'):
        loaded_data = np.load(save_file + '.npz')
        mip_matrix = loaded_data['wholeMIP']
        evidence_matrix = loaded_data['evidence']
    else:
        overall_mmi = 0.0
        # generate an MI matrix for each cluster
        mi_matrix = np.zeros((alignment.seq_length, alignment.seq_length))
        evidence_matrix = np.zeros((alignment.seq_length, alignment.seq_length))
        # Vector of 1 column
        mmi = np.zeros(alignment.seq_length)
        apc_matrix = np.zeros((alignment.seq_length, alignment.seq_length))
        mip_matrix = np.zeros((alignment.seq_length, alignment.seq_length))
        # Generate MI matrix from alignment2Num matrix, the mmi matrix,
        # and overall_mmi
        for i in range(alignment.seq_length):
            for j in range(i + 1, alignment.seq_length):
                if evidence:
                    _I, _J, _pos, ev = alignment.identify_comparable_sequences(i, j)
                else:
                    ev = 0
                col_i = alignment.alignment_matrix[:, i]
                col_j = alignment.alignment_matrix[:, j]
                try:
                    curr_mis = mutual_info_score(col_i, col_j, contingency=None)
                except:
                    print col_i
                    print col_j
                    exit()
                # AW: divides by individual entropies to normalize.
                mi_matrix[i, j] = mi_matrix[j, i] = curr_mis
                evidence_matrix[i, j] = evidence_matrix[j, i] = ev
                overall_mmi += curr_mis
        mmi += np.sum(mi_matrix, axis=1)
        mmi -= mi_matrix[np.arange(alignment.seq_length), np.arange(alignment.seq_length)]
        mmi /= (alignment.seq_length - 1)
        overall_mmi = 2.0 * (overall_mmi / (alignment.seq_length - 1)) / alignment.seq_length
        # Calculating APC
        apc_matrix += np.outer(mmi, mmi)
        apc_matrix[np.arange(alignment.seq_length), np.arange(alignment.seq_length)] = 0
        apc_matrix /= overall_mmi
        # Defining MIP matrix
        mip_matrix += mi_matrix - apc_matrix
        mip_matrix[np.arange(alignment.seq_length), np.arange(alignment.seq_length)] = 0
        if save_file is not None:
            np.savez(save_file, wholeMIP=mip_matrix, evidence=evidence_matrix)
    end = time()
    print('Whole analysis took {} min'.format((end - start) / 60.0))
    return mip_matrix, evidence_matrix


def pool_init1(aa_reference, w_cc, original_alignment, save_dir, align_lock, k_queue, sub_alignment_queue, res_queue,
               low_mem):
    """
    poolInit

    A function which initializes processes spawned in a worker pool performing
    the etMIPWorker function.  This provides a set of variables to all working
    processes which are shared.

    Parameters:
    -----------
    aa_reference : dict
        Dictionary mapping amino acid abbreviations.
    w_cc : str
        Method by which to combine individual matrices from one round of
        clustering. The options supported now are: sum, average, size_weighted,
        and evidence_weighted.
    originialAlignment : SeqAlignment
        Alignment held by the instance of ETMIPC which called this method.
    save_dir : str
        The caching directory used to save results from agglomerative
        clustering.
    align_lock : multiprocessing.Manager.Lock()
        Lock used to regulate access to the alignment object for the purpose
        of setting the tree order.
    k_queue : multiprocessing.Manager.Queue()
        Queue used for tracking the k's for which clustering still needs to be
        performed.
    sub_alignment_queue : multiprocessing.Manager.Queue()
        Queue used to track the subalignments generated by clustering based on
        the k's in the k_queue.
    res_queue : multiprocessing.Manager.Queue()
        Queue used to track final results generated by this method.
    low_mem : bool
        Whether or not low memory mode should be used.
    """
    global aa_dict
    aa_dict = aa_reference
    global within_cluster_combi
    within_cluster_combi = w_cc
    global initial_alignment
    initial_alignment = original_alignment
    global cache_dir
    cache_dir = save_dir
    global k_lock
    k_lock = align_lock
    global queue1
    queue1 = k_queue
    global queue2
    queue2 = sub_alignment_queue
    global queue3
    queue3 = res_queue
    global pool1_mem_mode
    pool1_mem_mode = low_mem


def et_mip_worker1(in_tup):
    """
    ETMIP Worker

    Performs clustering and calculation of cluster dependent sequence distances.
    This function requires initialization of threads with poolInit, or setting
    of global variables as described in that function.

    Parameters:
    -----------
    in_tup: tuple
        Tuple containing the one int specifying which process this is,
        and a second int specifying the number of active processes.
    Returns:
    --------
    dict
        Mapping of k, to sub-cluster, to size of sub-cluster.
    dict
        Mapping of k, to sub-cluster, to the SeqAlignment object reprsenting
        the sequence IDs present in that sub-cluster.
    dict
        Mapping of k to the time spent working on data from that k by this
        process.
    """
    curr_process, total_processes = in_tup
    cluster_sizes = {}
    cluster_times = {}
    sub_alignments = {}
    while (not queue1.empty()) or (not queue2.empty()):
        try:
            print('Processes {}:{} acquiring sub alignment!'.format(
                curr_process, total_processes))
            clus, sub, new_alignment = queue2.get_nowait()
            print('Current alignment has {} sequences'.format(new_alignment.size))
            start = time()
            if 'evidence' in within_cluster_combi:
                clustered_mip_matrix, evidence_mat = whole_analysis(new_alignment, True)
            else:
                clustered_mip_matrix, evidence_mat = whole_analysis(new_alignment, False)
            end = time()
            time_elapsed = end - start
            if clus in cluster_times:
                cluster_times[clus] += time_elapsed
            else:
                cluster_times[clus] = time_elapsed
            print('ETMIP worker took {} min'.format(time_elapsed / 60.0))
            if pool1_mem_mode:
                save_raw_score_matrix(clus, sub, clustered_mip_matrix, evidence_mat, cache_dir)
            else:
                queue3.put((clus, sub, clustered_mip_matrix, evidence_mat))
            print('Processes {}:{} pushing cET-MIp scores!'.format(
                curr_process, total_processes))
            continue
        except Queue.Empty:
            print('Processes {}:{} failed to acquire-sub alignment!'.format(
                curr_process, total_processes))
            pass
        try:
            print('Processes {}:{} acquiring k to generate clusters!'.format(curr_process, total_processes))
            k_lock.acquire()
            print('Lock acquired by: {}'.format(curr_process))
            c = queue1.get_nowait()
            print('K: {} acquired setting tree'.format(c))
            start = time()
            cluster_sizes[c] = {}
            clus_dict, clus_det = initial_alignment.agg_clustering(n_cluster=c, cache_dir=cache_dir)
            tree_ordering = []
            sub_alignments[c] = {}
            for sub in clus_det:
                new_alignment = initial_alignment.generate_sub_alignment(clus_dict[sub])
                cluster_sizes[c][sub] = new_alignment.size
                # Create matrix converting sequences of amino acids to sequences of
                # integers representing sequences of amino acids
                new_alignment.alignment_to_num(aa_dict)
                queue2.put((c, sub, new_alignment))
                tree_ordering += new_alignment.tree_order
                sub_alignments[c][sub] = new_alignment
            initial_alignment.set_tree_ordering(t_order=tree_ordering)
            end = time()
            k_lock.release()
            if c in cluster_times:
                cluster_times[c] += (end - start)
            else:
                cluster_times[c] = (end - start)
            print('Processes {}:{} pushing new sub-alignment!'.format(
                curr_process, total_processes))
            continue
        except Queue.Empty:
            k_lock.release()
            print('Processes {}:{} failed to acquire k!'.format(
                curr_process, total_processes))
            pass
    print('Process: {} completed and returning!'.format(curr_process))
    return cluster_sizes, sub_alignments, cluster_times


# def pool_init2(q_scorer, out_dir):
def pool_init2(out_dir):
    """
    pool_init2

    A function which initializes processes spawned in a worker pool performing
    the et_mip_worker2 function.  This provides a set of variables to all working
    processes which are shared.

    Parameters:
    -----------
    c: float
        The threshold distance at which to consider two residues as interacting
        with one another.
    q_alignment: SeqAlignment
        Object containing the sequence alignment for this analysis.
    q_structure: PDBReference
        Object containing the PDB information for this analysis.
    """
    global w2_out_dir
    w2_out_dir = out_dir
    # global scorer
    # scorer = q_scorer


def et_mip_worker2(in_tup):
    """
    ETMIP Worker 2

    Performs clustering and calculation of cluster dependent sequence distances.
    This function requires initialization of threads with poolInit, or setting
    of global variables as described in that function.

    Parameters:
    -----------
    in_tup: tuple
        Tuple containing the number of clusters to form during agglomerative
        clustering and a matrix which is the result of summing the original
        MIP matrix and the matrix resulting from clustering at this clustering
        and lower clusterings.
    Returns:
    --------
    int
        Number of clusters.
    list
        Coverage values for this clustering.
    float
        The time in seconds which it took to perform clustering.
    list
        List of false positive rates.
    list
        List of true positive rates.
    float
        The ROCAUC value for this clustering.
    """
    clus, all_summed_matrix = in_tup
    if all_summed_matrix:
        summed_matrix = all_summed_matrix[clus]
    else:
        summed_matrix = load_single_matrix('Summary', clus, w2_out_dir)
    start = time()
    curr_coverage = np.zeros(summed_matrix.shape)
    test_mat = np.triu(summed_matrix)
    mask = np.triu(np.ones(summed_matrix.shape), k=1)
    normalization = ((summed_matrix.shape[0]**2 - summed_matrix.shape[0]) / 2.0)
    for i in range(summed_matrix.shape[0]):
        for j in range(i + 1, summed_matrix.shape[0]):
            bool_mat = (test_mat[i, j] >= test_mat) * 1.0
            corrected_mat = bool_mat * mask
            compute_coverage2 = (((np.sum(corrected_mat) - 1) * 100) / normalization)
            curr_coverage[i, j] = curr_coverage[j, i] = compute_coverage2
    # tpr, fpr, roc_auc = scorer.score_auc(curr_coverage)
    end = time()
    time_elapsed = end - start
    print('ETMIP worker 2 took {} min'.format(time_elapsed / 60.0))
    if all_summed_matrix is None:
        save_single_matrix('Coverage', clus, curr_coverage, w2_out_dir)
        curr_coverage = None
    # return clus, curr_coverage, time_elapsed, fpr, tpr, roc_auc
    return clus, curr_coverage, time_elapsed


def pool_init_new(cluster_queue, q_name, today, class_mip_matrix, class_raw_scores, class_result_matrices,
                  class_coverage, class_summary, class_alignment, output_dir):
    """
    pool_init_new

    A function which initializes processes spawned in a worker pool performing
    the et_mip_worker3 function.  This provides a set of variables to all working
    processes which are shared.

    Parameters:
    -----------
    cluster_queue : multiprocessing.Manager.Queue()
        Queue used for tracking the k's for which output still needs to be
        generated.
    q_name : str
        The name of the query string.
    today : str
        The current date which will be used for identifying the proper directory
        to store files in.
    cut_off : float
        Distance in Angstroms between residues considered as a preliminary
        positive set of coupled residues based on spatial positions in the PDB
        file if provided.
    class_mip_matrix : np.ndarray
        Matrix scoring the coupling between all positions in the query
        sequence, as computed over all sequences in the input alignment.
    class_raw_scores : dict
        The dictionary mapping clustering constant to coupling scores for all
        positions in the query sequences at the specified clustering constant
        created by hierarchical clustering.
    class_result_matrices : dict
        A dictionary mapping clustering constants to matrices which represent
        the integration of coupling scores across all clusters defined at that
        clustering constant.
    class_coverage : dict
        This dictionary maps clustering constants to a matrix of normalized
        coupling scores between 0 and 100, computed from the
        summary_matrices.
    class_summary : dict
        This dictionary maps clustering constants to a matrix which combines
        the scores from the whole_mip_matrix, all lower clustering constants,
        and this clustering constant.
    class_subalignments : dict
            A dictionary mapping a clustering constant (k) to another dictionary
            which maps a cluster label (0 to k-1) to a SeqAlignment object
            containing only the sequences for that specific cluster.
    class_alignment : SeqAlignment
        The SeqAlignment object containing relevant information for this
        ETMIPC analysis.
    output_dir : str
        The full path to where the output generated by this process should be
        stored. If None (default) the plot will be stored in the current working
        directory.
    """
    global queue1
    queue1 = cluster_queue
    global query_n
    query_n = q_name
    global date
    date = today
    global mip_matrix
    mip_matrix = class_mip_matrix
    global raw_scores
    raw_scores = class_raw_scores
    global res_mat
    res_mat = class_result_matrices
    global alignment
    alignment = class_alignment
    global coverage
    coverage = class_coverage
    global summary
    summary = class_summary
    global out_dir
    out_dir = output_dir


def et_mip_worker_new(input_tuple):
    """
    ETMIP Worker New

    This method uses queues to generate the jobs necessary to create the final
    output of the ETMIPC class ProduceFinalFigures method (figures and
    output files). One queue is used to hold the clustering constants to be
    processed (producer) while another queue is used to hold the functions
    to call and the input data to provide (producer). This method directs a
    process to preferentially pull jobs from the second queue, unless none are
    available, in which case it directs the process to generate additional jobs
    using queue 1. If both queues are empty the method terminates.

    Parameters:
    -----------
    inTup: tuple
        Tuple containing the one int specifying which process this is,
        and a second int specifying the number of active processes.
    Returns:
    --------
    dict
        Mapping of k to the time spent working on data from that k by this
        process.
    """
    from ContactScorer import write_out_contact_scoring
    curr_process, total_processes = input_tuple
    times = {}
    while not queue1.empty():
        try:
            c = queue1.get_nowait()
            print('Process {} of {} writing scores for cluster {}'.format(curr_process, total_processes, c))
            start = time()
            c_out_dir = os.path.join(out_dir, str(c))
            if (summary is None) and (coverage is None):
                c_summary = load_single_matrix('Summary', c, out_dir)
                c_coverage = load_single_matrix('Coverage', c, out_dir)
            else:
                c_summary = summary[c]
                c_coverage = coverage[c]
            if (res_mat is None) and (raw_scores is None):
                c_raw, _ = load_raw_score_matrix(alignment.seq_length, c, out_dir)
                c_result = load_single_matrix('Result', c, out_dir)
            else:
                c_raw = raw_scores[c]
                c_result = res_mat[c]
            res_fn = "{}_{}_{}.all_scores.txt".format(date, query_n, c)
            write_out_contact_scoring(date, alignment, c_result, c_coverage, mip_matrix, c_raw, c_summary, res_fn,
                                      c_out_dir)
            end = time()
            time_elapsed = end - start
            if c not in times:
                times[c] = time_elapsed
            else:
                times[c] += time_elapsed
        except Queue.Empty:
            pass
    return times


def pool_init3(cluster_queue, output_queue, q_name, today, verbosity, class_mip_matrix, class_raw_scores,
               class_result_matrices, class_coverage, class_summary, class_subalignments, class_alignment, class_aucs,
               class_scorer, output_dir):
    """
    pool_init3

    A function which initializes processes spawned in a worker pool performing
    the et_mip_worker3 function.  This provides a set of variables to all working
    processes which are shared.

    Parameters:
    -----------
    cluster_queue : multiprocessing.Manager.Queue()
        Queue used for tracking the k's for which output still needs to be
        generated.
    output_queue : multiprocessing.Manager.Queue()
        Queue used for tracking the types of output to be generated and the
        inputs for the dependent methods.
    q_name : str
        The name of the query string.
    today : str
        The current date which will be used for identifying the proper directory
        to store files in.
    cut_off : float
        Distance in Angstroms between residues considered as a preliminary
        positive set of coupled residues based on spatial positions in the PDB
        file if provided.
    verbosity : int
        How many figures to produce.1 = ROC Curves, ETMIP Coverage file,
        and final AUC and Timing file. 2 = files with all scores at each
        clustering. 3 = sub-alignment files and plots. 4 = surface plots
        and heatmaps of ETMIP raw and coverage scores.'
    class_mip_matrix : np.ndarray
        Matrix scoring the coupling between all positions in the query
        sequence, as computed over all sequences in the input alignment.
    class_raw_scores : dict
        The dictionary mapping clustering constant to coupling scores for all
        positions in the query sequences at the specified clustering constant
        created by hierarchical clustering.
    class_result_matrices : dict
        A dictionary mapping clustering constants to matrices which represent
        the integration of coupling scores across all clusters defined at that
        clustering constant.
    class_coverage : dict
        This dictionary maps clustering constants to a matrix of normalized
        coupling scores between 0 and 100, computed from the
        summary_matrices.
    class_summary : dict
        This dictionary maps clustering constants to a matrix which combines
        the scores from the whole_mip_matrix, all lower clustering constants,
        and this clustering constant.
    class_subalignments : dict
            A dictionary mapping a clustering constant (k) to another dictionary
            which maps a cluster label (0 to k-1) to a SeqAlignment object
            containing only the sequences for that specific cluster.
    class_alignment : SeqAlignment
        The SeqAlignment object containing relevant information for this
        ETMIPC analysis.
    class_aucs : dictionary
        AUC values stored in the ETMIPC class, used to identify the specific
        values for the specified clustering constant (clus).
    class_pdb : PDBReference
        Object representing the pdb structure used in the current
        analysis.
    output_dir : str
        The full path to where the output generated by this process should be
        stored. If None (default) the plot will be stored in the current working
        directory.
    """
    global queue1
    queue1 = cluster_queue
    global queue2
    queue2 = output_queue
    global query_n
    query_n = q_name
    global date
    date = today
    global ver
    ver = verbosity
    global mip_matrix
    mip_matrix = class_mip_matrix
    global raw_scores
    raw_scores = class_raw_scores
    global res_mat
    res_mat = class_result_matrices
    global sub_alignments
    sub_alignments = class_subalignments
    global alignment
    alignment = class_alignment
    global coverage
    coverage = class_coverage
    global summary
    summary = class_summary
    global aucs
    aucs = class_aucs
    global scorer
    scorer = class_scorer
    global out_dir
    out_dir = output_dir


def et_mip_worker3(input_tuple):
    """
    ETMIP Worker 3

    This method uses queues to generate the jobs necessary to create the final
    output of the ETMIPC class ProduceFinalFigures method (figures and 
    output files). One queue is used to hold the clustering constants to be
    processed (producer) while another queue is used to hold the functions
    to call and the input data to provide (producer). This method directs a
    process to preferentially pull jobs from the second queue, unless none are
    available, in which case it directs the process to generate additional jobs
    using queue 1. If both queues are empty the method terminates.

    Parameters:
    -----------
    inTup: tuple
        Tuple containing the one int specifying which process this is,
        and a second int specifying the number of active processes.
    Returns:
    --------
    dict
        Mapping of k to the time spent working on data from that k by this
        process.
    """
    from ContactScorer import write_out_contact_scoring
    curr_process, total_processes = input_tuple
    times = {}
    function_dict = {'heatmap': heatmap_plot, 'surface_plot': surface_plot,
                     'writeClusterResults': scorer.write_out_clustering_results,
                     # 'writeClusterScoring': write_out_contact_scoring,
                     'plot_auc': scorer.plot_auc, 'subAlignment': None}
    while (not queue1.empty()) or (not queue2.empty()):
        try:
            q_func, q_param = queue2.get_nowait()
            print('Calling: {} in processes {}:{}'.format(q_func, curr_process, total_processes))
            c = q_param[0]
            c_out_dir = os.path.join(out_dir, str(c))
            # if q_func in ['writeClusterResults', 'writeClusterScoring', 'heatmap', 'surface_plot']:
            if q_func in ['writeClusterResults', 'heatmap', 'surface_plot']:
                if (summary is None) and (coverage is None):
                    c_summary = load_single_matrix('Summary', c, out_dir)
                    c_coverage = load_single_matrix('Coverage', c, out_dir)
                else:
                    c_summary = summary[c]
                    c_coverage = coverage[c]
                if q_func == 'heatmap':
                    if q_param[1] == 'Raw Score':
                        function_dict[q_func]('Raw Score Heatmap K {}'.format(c), c_summary, c_out_dir)
                    else:
                        function_dict[q_func]('Coverage Heatmap K {}'.format(c), c_coverage, c_out_dir)
                elif q_func == 'surface_plot':
                    if q_param[1] == 'Raw Score':
                        function_dict[q_func]('Raw Score Surface K {}'.format(c), c_summary, c_out_dir)
                    else:
                        function_dict[q_func]('Coverage Surface K {}'.format(c), c_coverage, c_out_dir)
                # elif q_func == 'writeClusterResults':
                else:
                    start = time()
                    res_fn = '{}_{}_{}.etmipCVG.clustered.txt'.format(date, query_n, c)
                    function_dict[q_func](date, query_n, c_summary, c_coverage, res_fn, c_out_dir)
                    end = time()
                    time_elapsed = end - start
                    if c not in times:
                        times[c] = time_elapsed
                    else:
                        times[c] += time_elapsed
                # else:
                #     if (res_mat is None) and (raw_scores is None):
                #         c_raw_scores, _ = load_raw_score_matrix(alignment.seq_length, c, out_dir)
                #         c_result = load_single_matrix('Result', c, out_dir)
                #     else:
                #         c_raw_scores = raw_scores[c]
                #         c_result = res_mat[c]
                #     res_fn = "{}_{}_{}.all_scores.txt".format(date, query_n, c)
                #     function_dict[q_func](date, c_result, c_coverage, mip_matrix, c_raw_scores, c_summary, res_fn,
                #                           c_out_dir)
            elif q_func == 'plot_auc':
                auc_title = 'Ability to predict positive contacts in {}, Cluster = {}'.format(query_n, c)
                auc_fn = '{0}{1}A_C{2}_{3}roc.eps'.format(query_n, scorer.cutoff, c, date)
                function_dict[q_func](query_n, aucs[c], auc_title, auc_fn, c_out_dir)
            elif q_func == 'subAlignment':
                sub = q_param[1]
                sub_alignments[c][sub].set_tree_ordering(alignment.tree_order)
                sub_alignments[c][sub].write_out_alignment(os.path.join(c_out_dir,
                                                                        'AlignmentForK{}_{}.fa'.format(c, sub)))
                sub_alignments[c][sub].heatmap_plot('Alignment For K {} {}'.format(c, sub), c_out_dir)
            else:
                raise ValueError('Function: {} not implemented!'.format(q_func))
        except Queue.Empty:
            pass
        try:
            c = queue1.get_nowait()
            if ver >= 1:
                queue2.put_nowait(('writeClusterResults', (c,)))
                if scorer:
                    queue2.put_nowait(('plot_auc', (c,)))
            # if ver >= 2:
            #     queue2.put_nowait(('writeClusterScoring', (c,)))
            if ver >= 3:
                for sub in range(c):
                    queue2.put_nowait(('subAlignment', (c, sub)))
            if ver >= 4:
                queue2.put_nowait(('heatmap', (c, 'Raw Score')))
                queue2.put_nowait(('heatmap', (c, 'Coverage')))
                queue2.put_nowait(('surface_plot', (c, 'Raw Score')))
                queue2.put_nowait(('surface_plot', (c, 'Coverage')))
        except Queue.Empty:
            pass
    print('Function completed by {}:{}'.format(curr_process, total_processes))
    return times
