#!/usr/bin/env python3
import gffutils
import random
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FASTA_FILE = "<GENOME_FASTA>"
GFF_FILE = "<GENE_GFF3>"
OUTPUT_DIR = "<OUTPUT_DIR>"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "genomic_regions.csv")
MIN_SEQ_LENGTH = 512
MAX_SEQ_LENGTH = 2048
NUM_PER_CLASS = 20000
VALID_BASES = set('ATCG')

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.info("Loading genome FASTA...")
genome = SeqIO.to_dict(SeqIO.parse(FASTA_FILE, "fasta"))

logging.info("Loading GFF3 into database...")
db = gffutils.create_db(GFF3_FILE, ":memory:", merge_strategy="create_unique", force=True)

logging.info("Extracting CDS regions...")
cds_regions = [(f.chrom, f.start, f.end) for f in db.features_of_type("CDS")]

logging.info("Computing introns...")
intron_regions = []
for gene in db.features_of_type("gene"):
    exons = sorted(db.children(gene, featuretype="exon", order_by="start"), key=lambda x: x.start)
    for i in range(len(exons)-1):
        start = exons[i].end + 1
        end = exons[i+1].start - 1
        if start <= end:
            intron_regions.append((gene.chrom, start, end))

logging.info("Computing intergenic regions...")
gene_regions = [(f.chrom, f.start, f.end) for f in db.features_of_type("gene")]
chrom_lengths = {chrom: len(genome[chrom].seq) for chrom in genome}
intergenic_regions = []

for chrom in chrom_lengths:
    genes_on_chrom = sorted([g for g in gene_regions if g[0] == chrom], key=lambda x: x[1])
    if not genes_on_chrom:
        intergenic_regions.append((chrom, 1, chrom_lengths[chrom]))
        continue
    starts = [g[1] for g in genes_on_chrom]
    ends = [g[2] for g in genes_on_chrom]
    if starts[0] > 1:
        intergenic_regions.append((chrom, 1, starts[0]-1))
    for i in range(len(starts)-1):
        gap_start = ends[i] + 1
        gap_end = starts[i+1] - 1
        if gap_start <= gap_end:
            intergenic_regions.append((chrom, gap_start, gap_end))
    if ends[-1] < chrom_lengths[chrom]:
        intergenic_regions.append((chrom, ends[-1]+1, chrom_lengths[chrom]))

logging.info("Extracting UTR regions for exclusion...")
utr_regions = [(f.chrom, f.start, f.end) for f in db.all_features() 
               if f.featuretype in ("five_prime_UTR", "three_prime_UTR", "5_prime_UTR", "3_prime_UTR")]

def is_valid_seq(seq):
    return all(c in VALID_BASES for c in seq.upper())

def overlaps_any(lst, chrom, start, end):
    return any(chrom == r[0] and start <= r[2] and end >= r[1] for r in lst)

def sample_from_regions(regions, label, n):
    if not regions:
        logging.warning(f"No valid regions for {label}")
        return []
    sequences = []
    used = set()
    pbar = tqdm(total=n, desc=label, unit="seq")
    attempts = 0
    while len(sequences) < n and attempts < n * 20:
        chrom, rstart, rend = random.choice(regions)
        length = random.randint(MIN_SEQ_LENGTH, min(MAX_SEQ_LENGTH, rend - rstart + 1))
        if length < MIN_SEQ_LENGTH:
            attempts += 1
            continue
        start = random.randint(rstart, rend - length + 1)
        end = start + length - 1
        key = (chrom, start, end)
        if key in used:
            attempts += 1
            continue
        if overlaps_any(utr_regions, chrom, start, end):
            attempts += 1
            continue
        try:
            seq = str(genome[chrom].seq[start-1:end]).upper()
            if is_valid_seq(seq):
                sequences.append((seq, label))
                used.add(key)
                pbar.update(1)
        except:
            pass
        attempts += 1
    pbar.close()
    logging.info(f"Collected {len(sequences)} {label} sequences")
    return sequences

logging.info("Sampling sequences...")
cds_seqs = sample_from_regions(cds_regions, "CDS", NUM_PER_CLASS)
intron_seqs = sample_from_regions(intron_regions, "intron", NUM_PER_CLASS)
intergenic_seqs = sample_from_regions(intergenic_regions, "intergenic", NUM_PER_CLASS)

all_data = cds_seqs + intron_seqs + intergenic_seqs
df = pd.DataFrame(all_data, columns=["sequence", "label"])
df.to_csv(OUTPUT_CSV, index=False)

logging.info(f"Dataset saved to {OUTPUT_CSV}")
logging.info(f"Total sequences: {len(df)} (CDS: {len(cds_seqs)}, intron: {len(intron_seqs)}, intergenic: {len(intergenic_seqs)})")
logging.info("Done")