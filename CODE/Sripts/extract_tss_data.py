#!/usr/bin/env python3
import random
import pandas as pd
from Bio import SeqIO
import os
import re
from tqdm import tqdm

FASTA_FILE = "<GENOME_FASTA>"
GFF_FILE = "<GENE_GFF3>"
OUTPUT_DIR = "<OUTPUT_DIR>"
OUTPUT_POSITIVE = "positive.fa"
OUTPUT_NEGATIVE = "negative.fa"
OFFSET_FILE = "tss_offsets.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

seq_length = 1024
atcg_pattern = re.compile(r'^[ATCG]+$')

genome = SeqIO.to_dict(SeqIO.parse(FASTA_FILE, "fasta"))
valid_chroms = [chrom for chrom in genome if len(genome[chrom].seq) >= seq_length]

gff = pd.read_csv(GFF_FILE, sep="\t", comment="#",
                  names=["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"],
                  dtype={"seqid": str})

mrna = gff[gff["type"] == "mRNA"].copy()

tss_positions = []
tss_set = set()
for _, row in mrna.iterrows():
    chrom = row["seqid"]
    if chrom not in genome:
        continue
    strand = row["strand"]
    tss = row["start"] - 1 if strand == "+" else row["end"] - 1
    if (chrom, tss) in tss_set:
        continue
    tss_set.add((chrom, tss))
    tss_positions.append((chrom, tss, strand, row["attributes"]))

positive_sequences = []
offsets = []
for chrom, tss, strand, attrs in tqdm(tss_positions, desc="Positive"):
    if chrom not in valid_chroms:
        continue
    max_up = min(tss, seq_length - 1)
    max_down = len(genome[chrom].seq) - tss - 1
    window = min(max_up, max_down, seq_length // 2)
    start = tss - random.randint(0, window)
    end = start + seq_length
    if start < 0 or end > len(genome[chrom].seq):
        continue
    seq = genome[chrom].seq[start:end]
    if strand == "-":
        seq = seq.reverse_complement()
    seq_str = str(seq).upper()
    if not atcg_pattern.match(seq_str):
        continue
    offset = tss - start if strand == "+" else end - tss - 1
    positive_sequences.append(f">{chrom}_TSS_{tss}_{strand}\n{seq_str}\n")
    offsets.append(f"{chrom}\t{start}\t{end}\tTSS_{tss}\t{offset}\t{strand}\n")

negative_sequences = []
chrom_lengths = {c: len(genome[c].seq) for c in valid_chroms}
tss_set = {(c, t) for c, t, _, _ in tss_positions}
num_needed = len(positive_sequences)

with tqdm(total=num_needed, desc="Negative") as pbar:
    while len(negative_sequences) < num_needed:
        chrom = random.choice(valid_chroms)
        start = random.randint(0, chrom_lengths[chrom] - seq_length)
        end = start + seq_length
        if any((chrom, pos) in tss_set for pos in range(start, end)):
            continue
        seq = genome[chrom].seq[start:end]
        seq_str = str(seq).upper()
        if atcg_pattern.match(seq_str):
            negative_sequences.append(f">{chrom}_nonTSS_{start}\n{seq_str}\n")
            pbar.update(1)

with open(OUTPUT_POSITIVE, "w") as f:
    f.writelines(positive_sequences)

with open(OUTPUT_NEGATIVE, "w") as f:
    f.writelines(negative_sequences)

with open(OFFSET_FILE, "w") as f:
    f.write("chrom\tstart\tend\tid\toffset\tstrand\n")
    f.writelines(offsets)