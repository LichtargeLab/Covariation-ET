"""
Created on May 23, 2019

@author: Daniel Konecki
"""
import os
from Bio.Seq import Seq
from Bio.SeqIO import write
from Bio.Blast import NCBIXML
from Bio.SeqRecord import SeqRecord
from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser
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

    def __init__(self, protein_list, input_path):
        if input_path is None:
            self.input_path = os.path.join(os.environ.get('PROJECT_PATH'), 'Input')
        else:
            self.input_path = input_path
        self.file_name = protein_list
        self.protein_data = self._import_protein_list()

    def _import_protein_list(self):
        protein_list_fn = os.path.join(self.input_path, 'ProteinLists', self.file_name)
        protein_list = {}
        with open(protein_list_fn, mode='rb') as protein_list_handle:
            for line in protein_list_handle:
                protein_list[line.strip()] = {}
        return protein_list

    def build_dataset(self):
        for seq_id in self.protein_data:
            self._download_pdb(protein_id=seq_id)
            self._parse_query_sequences(protein_id=seq_id)
            self._blast_query_sequences(protein_id=seq_id)
            self._restrict_sequences(protein_id=seq_id)
            self._align_sequences(protein_id=seq_id)

    def _download_pdb(self, protein_id):
        pdb_path = os.path.join(self.input_path, 'PDB')
        if not os.path.isdir(pdb_path):
            os.makedirs(pdb_path)
        pdb_list = PDBList(server='ftp://ftp.wwpdb.org', pdb=pdb_path)
        pdb_file = pdb_list.retrieve_pdb_file(pdb_code=protein_id, file_format='pdb')
        self.protein_data[protein_id]['PDB_Path'] = pdb_file

    def _parse_query_sequences(self, protein_id):
        protein_fasta_path = os.path.join(self.input_path, 'Sequences')
        if not os.path.isdir(protein_fasta_path):
            os.makedirs(protein_fasta_path)
        parser = PDBParser(PERMISSIVE=1)  # corrective
        structure = parser.get_structure(protein_id, self.protein_data[protein_id]['PDB_Path'])
        model = structure[0]
        chain = model['A']
        sequence = []
        for residue in chain:
            if is_aa(residue.get_resname(), standard=True):
                res_name = three_to_one(residue.get_resname())
                sequence.append(res_name)
        sequence = ''.join(sequence)
        self.protein_data[protein_id]['Query_Sequence'] = Seq(sequence, alphabet=ExtendedIUPACProtein)
        self.protein_data[protein_id]['Sequence_Length'] = len(sequence)
        seq_records = [SeqRecord(self.protein_data[protein_id]['Query_Sequence'], id=protein_id)]
        protein_fasta_fn = os.path.join(protein_fasta_path, '{}.fasta'.format(protein_id))
        self.protein_data[protein_id]['Fasta_File'] = protein_fasta_fn
        with open(protein_fasta_fn, 'wb') as protein_fasta_handle:
            write(sequences=seq_records, handle=protein_fasta_handle, format='fasta')

    def _blast_query_sequences(self, protein_id, threads=1):
        """
        From Benu

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
        blast_path = os.path.join(self.input_path, 'BLAST')
        if not os.path.isdir(blast_path):
            os.makedirs(blast_path)
        blast_fn = os.path.join(blast_path, '{}.xml'.format(protein_id))
        self.protein_data[protein_id]['BLAST_File'] = blast_fn
        blastp_cline = NcbiblastpCommandline(query=self.protein_data[protein_id]['Fasta_File'], db='uniref90',
                                             out=blast_fn, out_fmt='xml', remote=False, ungapped=False,
                                             num_threads=threads)
        print(blastp_cline)
        stdout, stderr = blastp_cline()
        print(stdout)
        print(stderr)

    def _restrict_sequences(self, protein_id, e_value_threshold = 0.05, min_fraction=0.7, max_fraction=1.5,
                            min_identity=0.75, max_identity=0.95):
        """

        :return:
        """
        pileup_path = os.path.join(self.input_path, 'Pileups')
        if not os.path.isdir(pileup_path):
            os.makedirs(pileup_path)
        sequences = []
        with open(self.protein_data[protein_id]['BLAST_File'], 'rb') as blast_handle:
            blast_record = NCBIXML.read(blast_handle)
            for alignment in blast_record.alignments:
                for hsp in alignment.hsps:
                    if hsp.expect <= e_value_threshold:
                        subject_length = len(hsp.sbjct)
                        subject_fraction = subject_length / float(self.protein_data[protein_id]['Sequence_Length'])
                        if (min_fraction <= subject_fraction) and (subject_fraction <= max_fraction):
                            subject_similarity = hsp.identities / float(hsp.align_length)
                            if (min_identity <= subject_similarity) and (subject_similarity <= max_identity):
                                sequences.append(hsp.sbjct)
        size = len(sequences)
        if size < 125:
            if min_identity > 0.5:
                new_min_identity = 0.5
            elif min_identity > 0.42:
                new_min_identity = 0.42
            elif min_identity > 0.3:
                new_min_identity = 0.3
            else:
                self.protein_data[protein_id]['Pileup_File'] = None
                print('No pileup for pdb: {}, sufficient sequences could not be found'.format(pdb_id))
            self.restrict_sequences(protein_id=protein_id, min_identity=new_min_identity)
        else:
            pileup_fn = os.path.join(pileup_path, '{}.fasta'.format(protein_id))
            self.protein_data[protein_id]['Pileup_File'] = pileup_fn
            with open(pileup_fn, 'wb') as pileup_handle:
                write(sequences=sequences, handle=pileup_handle, format='fasta')

    def _align_sequences(self, protein_id):
        muscle_path = os.environ.get('MUSCLE_PATH')
        alignment_path = os.path.join(self.input_path, 'Alignments')
        msf_fn = os.path.join(alignment_path, '{}.msf'.format(protein_id))
        msf_cline = MuscleCommandline(muscle_path, input=self.protein_data[protein_id]['Pileup_File'], out=msf_fn,
                                      msf=True)

        print(msf_cline)
        stdout, stderr = msf_cline()
        print(stdout)
        print(stderr)
        fa_fn = os.path.join(alignment_path, '{}.fa'.format(protein_id))
        fa_cline = MuscleCommandline(muscle_path, input=self.protein_data[protein_id]['Pileup_File'], out=fa_fn,
                                     msf=False)
        print(fa_cline)
        stdout, stderr = fa_cline()
        print(stdout)
        print(stderr)
        self.protein_data[protein_id]['MSF_File'] = msf_fn
        self.protein_data[protein_id]['FA_File'] = fa_fn
