#!/usr/bin/env python3
import random
import pandas as pd
from Bio import SeqIO
from collections import defaultdict
import os
from pathlib import Path
import time
from tqdm import tqdm

SPECIES_CONFIG = {
    'species1': {
        'gff': 'PATH_TO_SPECIES1_GFF3',
        'fasta': 'PATH_TO_SPECIES1_GENOME.fa'
    },
    'species2': {
        'gff': 'PATH_TO_SPECIES2_GFF3',
        'fasta': 'PATH_TO_SPECIES2_GENOME.fa'
    },
    'species3': {
        'gff': 'PATH_TO_SPECIES3_GFF3',
        'fasta': 'PATH_TO_SPECIES3_GENOME.fa'
    },
    'species4': {
        'gff': 'PATH_TO_SPECIES4_GFF3',
        'fasta': 'PATH_TO_SPECIES4_GENOME.fa'
    },
    'species5': {
        'gff': 'PATH_TO_SPECIES5_GFF3',
        'fasta': 'PATH_TO_SPECIES5_GENOME.fa'
    }
}

OUTPUT_DIR = "<OUTPUT_DIR>"
SEQ_LENGTH = 1024
NUM_SAMPLES_PER_SPECIES = 12000
MIN_GAP = 1000
MAX_ATTEMPTS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_genome(fasta_file):
    chromosomes = list(SeqIO.parse(fasta_file, 'fasta'))
    return chromosomes

def is_valid_sequence(seq):
    return all(c in 'ATCG' for c in seq.upper())

def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement.get(b, b) for b in reversed(seq.upper()))

def build_cds_regions(gff_file, genome_dict):
    cds_regions = []
    with open(gff_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            chrom, _, typ, start, end, _, strand, _, attrs = parts[:9]
            if typ != "CDS":
                continue
            attr_dict = dict(kv.split("=") for kv in attrs.split(";") if "=" in kv)
            parent = attr_dict.get("Parent")
            if not parent:
                continue
            if chrom not in genome_dict:
                continue
            cds_regions.append((chrom, int(start)-1, int(end), strand))
    return cds_regions

def sample_sequence(chromosomes, cds_regions, seq_length, used_positions, species):
    genome_dict = {ch.id: ch for ch in chromosomes}
    attempts = 0
    while attempts < MAX_ATTEMPTS:
        region = random.choice(cds_regions)
        chrom_id, r_start, r_end, strand = region
        chrom = genome_dict[chrom_id]
        region_len = r_end - r_start
        if region_len < seq_length:
            attempts += 1
            continue
        start = random.randint(r_start, r_end - seq_length)
        end = start + seq_length
        if any(chrom_id == pos[0] and abs(start - pos[1]) < MIN_GAP for pos in used_positions.get(species, [])):
            attempts += 1
            continue
        raw_seq = str(chrom.seq[start:end]).upper()
        if strand == "-":
            seq = reverse_complement(raw_seq)
        else:
            seq = raw_seq
        if is_valid_sequence(seq):
            return seq, (chrom_id, start, end)
        attempts += 1
    return None, None

def main():
    dataset = []
    used_positions = {sp: [] for sp in SPECIES_CONFIG}
    start_time = time.time()

    for species, paths in SPECIES_CONFIG.items():
        print(f"Processing {species}")
        chromosomes = load_genome(paths['fasta'])
        genome_dict = {ch.id: ch for ch in chromosomes}
        cds_regions = build_cds_regions(paths['gff'], genome_dict)

        if not cds_regions:
            print(f"No CDS regions found for {species}")
            continue

        collected = 0
        pbar = tqdm(total=NUM_SAMPLES_PER_SPECIES, desc=species, unit="seq")
        while collected < NUM_SAMPLES_PER_SPECIES:
            seq, pos = sample_sequence(chromosomes, cds_regions, SEQ_LENGTH, used_positions, species)
            if seq is None:
                continue
            dataset.append({'sequence': seq, 'species': species})
            used_positions[species].append(pos)
            collected += 1
            pbar.update(1)
        pbar.close()

    output_file = Path(OUTPUT_DIR) / "species_CDS_sequences.csv"
    df = pd.DataFrame(dataset)
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    print(f"Total sequences: {len(dataset)}")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()