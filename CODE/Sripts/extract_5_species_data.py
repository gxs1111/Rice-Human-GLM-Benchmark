#!/usr/bin/env python3
import random
import pandas as pd
from Bio import SeqIO
import os
from pathlib import Path
import time
from tqdm import tqdm

SPECIES_GENOMES = {
    'species1': 'PATH_TO_SPECIES1_GENOME.fa',
    'species2': 'PATH_TO_SPECIES2_GENOME.fa',
    'species3': 'PATH_TO_SPECIES3_GENOME.fa',
    'species4': 'PATH_TO_SPECIES4_GENOME.fa',
    'species5': 'PATH_TO_SPECIES5_GENOME.fa'
}
c
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

def sample_sequence(chromosomes, seq_length, used_positions, species):
    total_len = sum(len(ch.seq) for chrom in chromosomes)
    weights = [len(ch.seq) / total_len for chrom in chromosomes]
    attempts = 0
    while attempts < MAX_ATTEMPTS:
        chrom = random.choices(chromosomes, weights=weights, k=1)[0]
        if len(chrom.seq) < seq_length:
            attempts += 1
            continue
        start = random.randint(0, len(chrom.seq) - seq_length)
        end = start + seq_length
        if any(chrom.id == pos[0] and abs(start - pos[1]) < MIN_GAP for pos in used_positions.get(species, [])):
            attempts += 1
            continue
        seq = str(chrom.seq[start:end]).upper()
        if is_valid_sequence(seq):
            return seq, (chrom.id, start, end)
        attempts += 1
    return None, None

def main():
    dataset = []
    used_positions = {sp: [] for sp in SPECIES_GENOMES}
    start_time = time.time()

    for species, fasta_path in SPECIES_GENOMES.items():
        print(f"Processing {species}")
        chromosomes = load_genome(fasta_path)
        collected = 0
        pbar = tqdm(total=NUM_SAMPLES_PER_SPECIES, desc=species, unit="seq")

        while collected < NUM_SAMPLES_PER_SPECIES:
            seq, pos = sample_sequence(chromosomes, SEQ_LENGTH, used_positions, species)
            if seq is None:
                continue
            dataset.append({'sequence': seq, 'species': species})
            used_positions[species].append(pos)
            collected += 1
            pbar.update(1)
        pbar.close()

    output_file = Path(OUTPUT_DIR) / "species_sequences.csv"
    df = pd.DataFrame(dataset)
    df.to_csv(output_file, index=False)

    print(f"Dataset saved to {output_file}")
    print(f"Total sequences: {len(dataset)}")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()