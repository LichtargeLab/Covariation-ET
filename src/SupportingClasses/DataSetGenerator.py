"""
Created on May 23, 2019

@author: Daniel Konecki
"""
import os
from re import compile
from numpy import floor
from Bio.Seq import Seq
from Bio.Blast import NCBIXML
from Bio.SeqIO import write, parse
from Bio.SeqRecord import SeqRecord
from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser
from Bio.pairwise2.align import localds
from Bio.SubsMat.MatrixInfo import blosum62
from Bio.Alphabet.IUPAC import ExtendedIUPACProtein
from Bio.PDB.Polypeptide import three_to_one, is_aa
from Bio.Align.Applications import MuscleCommandline
from Bio.Blast.Applications import NcbiblastpCommandline
from dotenv import find_dotenv, load_dotenv
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
except IOError:
    dotenv_path = find_dotenv(raise_error_if_not_found=True, usecwd=True)
load_dotenv(dotenv_path)


class DataSetGenerator(object):
    """
    This class is meant to automate data set generation and improve reproducibility of papers from our lab which rely on
    PDB structures and alignments generated by BLASTING the proteins represented in those structures.

    Attributes:
        input_path (str/path): The path to the directory where the data for this data set should be stored. It is
        expected to contain at least one directory name ProteinLists which should contain files where each line denotes
        a separate PDB id (four letter code only).
        file_name (str): The file name of the list in the ProteinLists folder (described above) which will be used to
        generate the data set for analysis. The file is expected to have a single (four letter) PDB coe on each line.
        protein_data (dict): A dictionary to hold the protein data generated over the course of data set construction.
        Initially it only contains the PDB ids parsed in from the provided file_name as keys, referencing empty
        dictionaries as values.
    """

    def __init__(self, protein_list, input_path):
        """
        Init

        This function overwrites the default init function for the DataSetGenerator class.

        Args:
            protein_list (str): The name of the file where the list of PDB ids (four letter codes) of which the data set
            consists can be found. Each id is expected to be on its own line.
            input_path (path): The path to the directory where the data for this data set should be stored. It is
            expected to contain at least one directory name ProteinLists which should contain files where each line
            denotes a separate PDB id (four letter code only).
        """
        if input_path is None:
            self.input_path = os.path.join(os.environ.get('PROJECT_PATH'), 'Input')
        else:
            self.input_path = input_path
        self.file_name = protein_list
        self.protein_data = self._import_protein_list()

    def _import_protein_list(self):
        """
        Import protein list

        This function opens the list file found at file_name and parses in the PDB ids specified there.

        Returns:
            dict: A dictionary where each key is a single PDB id parsed from the specified list file and each value is
            an empty dictionary which will be filled as the data set is constructed.
        """
        protein_list_fn = os.path.join(self.input_path, 'ProteinLists', self.file_name)
        protein_list = {}
        with open(protein_list_fn, mode='rb') as protein_list_handle:
            for line in protein_list_handle:
                protein_list[line.strip()] = {}
        return protein_list

    def build_dataset(self, num_threads=1, max_target_seqs=20000, e_value_threshold=0.05, min_fraction=0.7,
                      max_fraction=1.5, min_identity=75, abs_max_identity=95, abs_min_identity=30, interval=5,
                      ignore_filter_size=False, msf=True, fasta=True):
        """
        Build Dataset

        This method builds a complete data set based on the protein id list specified in the constructor.

        Args:
            num_threads (int): The number of threads to use when performing the BLAST search.
            max_target_seqs (int): The maximum number of hits to look for in the BLAST database.
            e_value_threshold (float): The maximum e-value for a passing hit.
            min_fraction (float): The minimum fraction of the query sequence length for a passing hit.
            max_fraction (float): The maximum fraction of the query sequence length for a passing hit.
            min_identity (int): The preferred minimum identity for a passing hit.
            abs_max_identity (int): The absolute maximum identity for a passing hit.
            abs_min_identity (int): The absolute minimum identity for a passing hit.
            interval (int): The interval on which to define bins between min_identity and abs_min_identity in case
            sufficient sequences are not found at min_identity.
            ignore_filter_size (bool): Whether or not to ignore the 125 sequence requirement before writing the filtered
            sequences to file.
            msf (bool): Whether or not to create an msf version of the MUSCLE alignment.
            fasta (bool): Whether or not to create an fasta version of the MUSCLE alignment.
        """
        for seq_id in self.protein_data:
            self._download_pdb(protein_id=seq_id)
            self._parse_query_sequence(protein_id=seq_id)
            self._blast_query_sequence(protein_id=seq_id, num_threads=num_threads, max_target_seqs=max_target_seqs)
            self._restrict_sequences(protein_id=seq_id, e_value_threshold=e_value_threshold, min_fraction=min_fraction,
                                     max_fraction=max_fraction, min_identity=min_identity,
                                     abs_max_identity=abs_max_identity, abs_min_identity=abs_min_identity,
                                     interval=interval, ignore_filter_size=ignore_filter_size)
            self._align_sequences(protein_id=seq_id, msf=msf, fasta=fasta)

    def _download_pdb(self, protein_id):
        """
        Download PDB

        This function downloads the PDB structure file for the given PDB id provided. The file is stored in a file named
        pdb{pdb id}.ent within a sub directory of the input_path provided to the class named
        PDB/{middle two characters of the PDB id}/

        Args:
            protein_id (str): Four letter code for a PDB id to be downloaded.
        Returns:
            str: The path to the PDB file downloaded.
        """
        pdb_path = os.path.join(self.input_path, 'PDB')
        if not os.path.isdir(pdb_path):
            os.makedirs(pdb_path)
        pdb_list = PDBList(server='ftp://ftp.wwpdb.org', pdb=pdb_path)
        pdb_file = pdb_list.retrieve_pdb_file(pdb_code=protein_id, file_format='pdb')
        self.protein_data[protein_id]['PDB_Path'] = pdb_file
        return pdb_file

    def _parse_query_sequence(self, protein_id):
        """
        Parse Query Sequence

        This function opens the downloaded PDB file for a given protein id (for which _download_pdb has already been
        called) and extracts the sequence of chain A for the structure. The parsed sequence is given in single letter
        amino acid codes. The sequence is saved to a file in a subdirectory of input_path with the name Sequences and a
        file name with the pattern {protein id}.fasta.

        Args:
            protein_id (str): Four letter code for a PDB id whose sequence should be parsed.
        Returns:
            str: The sequence parsed from the PDB file of the specified protein id.
            int: The length of the parsed sequence.
            str: The file path to the fasta file where the sequence has been written.
        """
        protein_fasta_path = os.path.join(self.input_path, 'Sequences')
        if not os.path.isdir(protein_fasta_path):
            os.makedirs(protein_fasta_path)
        protein_fasta_fn = os.path.join(protein_fasta_path, '{}.fasta'.format(protein_id))
        if os.path.isfile(protein_fasta_fn):
            with open(protein_fasta_fn, 'rb') as protein_fasta_handle:
                seq_iter = parse(handle=protein_fasta_handle, format='fasta')
                sequence = seq_iter.next()
                sequence.alphabet = ExtendedIUPACProtein
                # sequence = str(seq_record.seq)
        else:
            parser = PDBParser(PERMISSIVE=1)  # corrective
            structure = parser.get_structure(protein_id, self.protein_data[protein_id]['PDB_Path'])
            model = structure[0]
            chain = model['A']
            sequence = []
            for residue in chain:
                if is_aa(residue.get_resname(), standard=True):
                    res_name = three_to_one(residue.get_resname())
                    sequence.append(res_name)
            sequence = Seq(''.join(sequence), alphabet=ExtendedIUPACProtein)
            seq_records = [SeqRecord(sequence, id=protein_id)]
            with open(protein_fasta_fn, 'wb') as protein_fasta_handle:
                write(sequences=seq_records, handle=protein_fasta_handle, format='fasta')
        self.protein_data[protein_id]['Query_Sequence'] = sequence
        self.protein_data[protein_id]['Sequence_Length'] = len(sequence)
        self.protein_data[protein_id]['Fasta_File'] = protein_fasta_fn
        return sequence, len(sequence), protein_fasta_fn

    def _blast_query_sequence(self, protein_id, num_threads=1, max_target_seqs=20000):
        """
        BlAST Query Sequence

        This function uses a local instance of the BLAST tool and the uniref 90 database to search for homologs and
        orthologs of the specified protein. The blast results are stored in a subdirectory of the input_path named BLAST
        with a file name following the pattern {protein id}.xml. This method assumes that _parse_query_sequence has
        already been performed for the specified protein id.

        Args:
            protein_id (str): Four letter code for the PDB id whose sequence should be searched using BLAST.
            num_threads (int): The number of threads to use when performing the BLAST search.
            max_target_seqs (int): The maximum number of hits to look for in the BLAST database.
        Returns:
            str: The path to the xml file storing the BLAST output.
        """
        blast_path = os.path.join(self.input_path, 'BLAST')
        if not os.path.isdir(blast_path):
            os.makedirs(blast_path)
        blast_fn = os.path.join(blast_path, '{}.xml'.format(protein_id))
        if not os.path.isfile(blast_fn):
            blastp_cline = NcbiblastpCommandline(cmd=os.path.join(os.environ.get('BLAST_PATH'), 'blastp'), out=blast_fn,
                                                 query=self.protein_data[protein_id]['Fasta_File'], outfmt=5,
                                                 remote=False, ungapped=False, num_threads=num_threads,
                                                 max_target_seqs=max_target_seqs,
                                                 db=os.path.join(os.environ.get('BLAST_DB_PATH'),
                                                                 'customuniref90.fasta'))
            print(blastp_cline)
            stdout, stderr = blastp_cline()
            print(stdout)
            print(stderr)
        self.protein_data[protein_id]['BLAST_File'] = blast_fn
        return blast_fn

    @staticmethod
    def __determine_identity_bin(identity_count, length, interval, abs_max_identity, abs_min_identity,
                                 min_identity, identity_bins):
        """
        Determine Identity Bin

        This method determines which identity bin a sequence belongs in based on the settings used for filtering
        sequences in the _restrict_sequences function.

        Args:
            identity_count (int): The number of positions which match between the query and the BLAST hit.
            length (int): The number of positions in the aligned sequence (including gaps).
            interval (int): The interval on which to define bins between min_identity and abs_min_identity in case
            sufficient sequences are not found at min_identity.
            abs_max_identity (int): The absolute maximum identity for a passing hit.
            abs_min_identity (int): The absolute minimum identity for a passing hit.
            min_identity (int): The preferred minimum identity for a passing hit.
            identity_bins (set): All of the possible identity bin options.
        Returns:
            float: The identity bin in which the sequence belongs.
        """
        similarity = identity_count / float(length)
        similarity_int = floor(similarity)
        similarity_bin = similarity_int - (similarity_int % interval)
        final_bin = None
        if abs_max_identity >= similarity_bin and similarity_bin >= abs_min_identity:
            if similarity_bin >= min_identity:
                final_bin = min_identity
            elif similarity_bin not in identity_bins and similarity_bin >= abs_min_identity:
                final_bin = abs_min_identity
            else:
                final_bin = similarity_bin
        return final_bin

    def _restrict_sequences(self, protein_id, e_value_threshold=0.05, min_fraction=0.7, max_fraction=1.5,
                            min_identity=75, abs_max_identity=95, abs_min_identity=30, interval=5,
                            ignore_filter_size=False):
        """
        Restrict Sequences

        This method reads in the sequences found in a BLAST search for a given protein id. It then filters the sequences
        to make sure that there are no fragments or LOW QUALITY sequences (as defined by uniref90 which is the default
        BLAST database for this pipeline). The BLAST results are also filtered such that the e-value must be less than
        or equal to the specified cutoff (this should be done in the _blast_query_method already but it is checked here
        for completeness an in case a BLAST query is performed outside of this pipeline but used to generate a data
        set).It also filters sequences ensuring that they are all within the min_fraction and max_fraction, i.e. if you
        divide the sequence in question by the query sequence length is it within the specified range. Finally, the
        method filters based on sequence identity. It tests the identity of a sequence and if it is within the range
        covered by abs_max_identity and abs_min_identity it is placed in a bin. The first bin is in the range from
        abs_max_identity to min_identity and all other bins are intervals specified by interval between min_identity and
        abs_min_identity. These bins are combined (from highest identity, e.g. min_identity, to lowest identity, e.g.
        abs_min_identity) until at least 125 sequences have been accumulated, which are then written to file. If this
        cannot be achieved after considering all bins no restricted sequence set is written to file. If
        ignore_filter_size is set then the 125 sequence requirement is ignored.

        Args:
            protein_id (str): Four letter code for the PDB id whose BLAST search results should be filtered.
            e_value_threshold (float): The maximum e-value for a passing hit.
            min_fraction (float): The minimum fraction of the query sequence length for a passing hit.
            max_fraction (float): The maximum fraction of the query sequence length for a passing hit.
            min_identity (int): The preferred minimum identity for a passing hit.
            abs_max_identity (int): The absolute maximum identity for a passing hit.
            abs_min_identity (int): The absolute minimum identity for a passing hit.
            interval (int): The interval on which to define bins between min_identity and abs_min_identity in case
            sufficient sequences are not found at min_identity.
            ignore_filter_size (bool): Whether or not to ignore the 125 sequence requirement before writing the filtered
            sequences to file.
        Returns:
            float: The minimum identity for a passing hit used to find at least 125 passing sequences.
            int: The number of sequences passing the filters at the minimum identity (described in the first return).
            str: The file path to the list of sequences writen out after filtering, None if less than 125 sequences were
            left after filtering.

        Notes From Benu:
        Database used: A custom Uniprot 90 database using BLAST (Altschul et al. 1990).
        The sequences were aligned using MUSCLE with default parameters (Edgar 2004) and this alignment in FASTA format
        served as an input for cET-MIp
        Homologs were restricted to meet the following criteria:
            1. length of homolog within a 1.5 fractional length of query
            2. If 125+ sequences were not obtained as homologs, we relaxed the sequence identity values from 95 to 75 to
            a lower limit of >50%, >42%, and >30% (e.g. Sung 2016 paper). We also limited the
            3. Lastly, we restricted the hits to those with a maximum e-val of 0.05 and removed putative and incomplete
            sequences. Fractional length to define short sequences was kept at 0.7-0.8 wrt query length
        """
        pileup_path = os.path.join(self.input_path, 'Pileups')
        if not os.path.isdir(pileup_path):
            os.makedirs(pileup_path)
        pileup_fn = os.path.join(pileup_path, '{}.fasta'.format(protein_id))
        identity_bins = list(range(abs_min_identity, min_identity, interval))
        sequences = {x: [] for x in identity_bins}
        if min_identity not in sequences:
            identity_bins.append(min_identity)
            sequences[min_identity] = []
        fragment_pattern = compile(r'^.*(\(Fragment\)).*$')
        low_quality_pattern = compile(r'^.*(LOW QUALITY).*$')
        if os.path.isfile(pileup_fn):
            self.protein_data[protein_id]['Pileup_File'] = pileup_fn
            count = 0
            with open(pileup_fn, 'rb') as pileup_handle:
                fasta_iter = parse(handle=pileup_handle, format='fasta')
                final_identity_bin = min_identity
                for seq_record in fasta_iter:
                    count += 1
                    curr_alignments = localds(self.protein_data[protein_id]['Query_Sequence'], seq_record,
                                              match_dict=blosum62, open=11, extend=1)
                    seq_id = 0
                    seq_len = len(curr_alignments[0][0])
                    for i in range(seq_len):
                        if curr_alignments[0][0][i] == curr_alignments[0][1][i]:
                            seq_id += 1
                    similarity_bin = self.__determine_identity_bin(
                        identity_count=seq_id, query_length=seq_len, interval=interval,
                        abs_max_identity=abs_max_identity, abs_min_identity=abs_min_identity, min_identity=min_identity,
                        identity_bins=set(identity_bins))
                    if similarity_bin and similarity_bin < final_identity_bin:
                        final_identity_bin = similarity_bin
            count -= 1  # Since the query sequence would be added at the top of the pileup file (see code below)
        else:
            with open(self.protein_data[protein_id]['BLAST_File'], 'rb') as blast_handle:
                blast_record = NCBIXML.read(blast_handle)
                for alignment in blast_record.alignments:
                    fragment_check = fragment_pattern.search(alignment.hit_def)
                    low_quality_check = low_quality_pattern.search(alignment.hit_def)
                    if fragment_check or low_quality_check:
                        continue
                    for hsp in alignment.hsps:
                        if hsp.expect <= e_value_threshold:  # Should already be controlled by BLAST e-value
                            subject_length = len(hsp.sbjct)
                            subject_fraction = subject_length / float(self.protein_data[protein_id]['Sequence_Length'])
                            if (min_fraction <= subject_fraction) and (subject_fraction <= max_fraction):
                                similarity_bin = self.__determine_identity_bin(
                                    identity_count=hsp.identities, length=hsp.align_length, interval=interval,
                                    abs_max_identity=abs_max_identity, abs_min_identity=abs_min_identity,
                                    min_identity=min_identity, identity_bins=set(identity_bins))
                                if similarity_bin:
                                    subject_seq_record = SeqRecord(Seq(hsp.sbjct, alphabet=ExtendedIUPACProtein),
                                                                   id=alignment.hit_id, name=alignment.title,
                                                                   description=alignment.hit_def)
                                    sequences[similarity_bin].append(subject_seq_record)
            i = 0
            final_sequences = []
            while len(final_sequences) < 125 and i < len(identity_bins):
                final_sequences += sequences[identity_bins[i]]
                i += 1
            count = len(final_sequences)
            if ignore_filter_size or len(final_sequences) >= 125:
                # Add query sequence so that this file can be fed directly to the alignment method.
                final_sequences = [self.protein_data[protein_id]]['Query_Sequence'] + final_sequences
                with open(pileup_fn, 'wb') as pileup_handle:
                    write(sequences=final_sequences, handle=pileup_handle, format='fasta')
            else:
                i -= 1  # To ensure that the index is still within the identity_bins list length
                pileup_fn = None
                print('No pileup for pdb: {}, sufficient sequences could not be found'.format(protein_id))
            final_identity_bin = identity_bins[i]
        self.protein_data[protein_id]['Pileup_File'] = pileup_fn
        return final_identity_bin, count, pileup_fn

    def _align_sequences(self, protein_id, msf=True, fasta=True):
        """
        Align Sequences

        This method uses MUSCLE to align the query sequence and all of the hits from BLAST which passed the filtering
        process by default the alignment is performed twice, once to produce a fasta alignment file, and once to produce
        the msf alignment file, however either of these options can be turned off.

        Args:
            protein_id (str): Four letter code for the PDB id for which an alignment should be performed.
            msf (bool): Whether or not to create an msf version of the MUSCLE alignment.
            fasta (bool): Whether or not to create an fasta version of the MUSCLE alignment.
        Returns:
            str: The path to the msf alignment produced by this method (None if msf=False).
            str: The path to the fasta alignment produced by this method (None if fa=False).
        """
        muscle_path = os.environ.get('MUSCLE_PATH')
        alignment_path = os.path.join(self.input_path, 'Alignments')
        msf_fn = None
        if msf:
            msf_fn = os.path.join(alignment_path, '{}.msf'.format(protein_id))
            if not os.path.isfile(msf_fn):
                msf_cline = MuscleCommandline(muscle_path, input=self.protein_data[protein_id]['Pileup_File'],
                                              out=msf_fn, msf=True)
                print(msf_cline)
                stdout, stderr = msf_cline()
                print(stdout)
                print(stderr)
        fa_fn = None
        if fasta:
            fa_fn = os.path.join(alignment_path, '{}.fa'.format(protein_id))
            if not os.path.isfile(fa_fn):
                fa_cline = MuscleCommandline(muscle_path, input=self.protein_data[protein_id]['Pileup_File'], out=fa_fn,
                                         msf=False)
                print(fa_cline)
                stdout, stderr = fa_cline()
                print(stdout)
                print(stderr)
        self.protein_data[protein_id]['MSF_File'] = msf_fn
        self.protein_data[protein_id]['FA_File'] = fa_fn
        return msf_fn, fa_fn


def batch_iterator(iterator, batch_size):
    """
    Batch Iterator

    This can be used on any iterator, for example to batch up SeqRecord objects from Bio.SeqIO.parse(...), or to batch
    Alignment objects from Bio.AlignIO.parse(...), or simply lines from a file handle. It is a generator function, and
    it returns lists of the entries from the supplied iterator.  Each list will have
    batch_size entries, although the final list may be shorter.

    Args:
        iterator (iterable): An iterable object, in this case a parser from the Bio package is the intended input.
        batch_size (int): The maximum number of entries to return for each batch (the final batch from the iterator may
        be smaller).
    Returns:
        list: A list of length batch_size (unless it is the last batch in the iterator in which case it may be fewer) of
        entries from the provided iterator.

    Scribed from Biopython (https://biopython.org/wiki/Split_large_file)
    """
    entry = True  # Make sure we loop once
    while entry:
        batch = []
        while len(batch) < batch_size:
            try:
                entry = iterator.next()
            except StopIteration:
                entry = None
            if entry is None:
                # End of file
                break
            batch.append(entry)
        if batch:
            yield batch


def filter_uniref_fasta(in_path, out_path):
    """
    Filter Uniref Fasta

    This function can be used to filter the fasta files provided by Uniref to remove the low quality sequences and
    sequences which represent protein fragments.

    Args:
        in_path (str): The path to the fasta file to filter (should be provided by Uniref so that the expected patterns
        can be found).
        out_path (str): The path to which the filtered fasta should be written.
    """
    sequences = []
    fragment_pattern = compile(r'^.*(\(Fragment\)).*$')
    low_quality_pattern = compile(r'^.*(LOW QUALITY).*$')
    record_iter = parse(open(in_path), "fasta")
    for i, batch in enumerate(batch_iterator(record_iter, 10000)):
        print('Batch: {}'.format(i))
        for seq_record in batch:
            fragment_check = fragment_pattern.search(seq_record.description)
            low_quality_check = low_quality_pattern.search(seq_record.description)
            if fragment_check or low_quality_check:
                continue
            sequences.append(seq_record)
            if len(sequences) == 1000:
                with open(out_path, 'ab') as out_handle:
                    write(sequences=sequences, handle=out_handle, format='fasta')
                sequences = []
