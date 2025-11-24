import os
import random
import re
from Bio import SeqIO
from Bio.Seq import Seq

GFF_FILE = "annotation.gff"
FASTA_FILE = "genome.fa"
PROMOTER_LENGTH = 500
UPSTREAM_FOR_NEGATIVE = 5000
OUTPUT_POSITIVE = "positive.fa"
OUTPUT_NEGATIVE = "negative.fa"
RANDOM_SEED = 42

os.makedirs(os.path.dirname(OUTPUT_POSITIVE), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_NEGATIVE), exist_ok=True)
random.seed(RANDOM_SEED)

fasta_records = {rec.id: rec.seq for rec in SeqIO.parse(FASTA_FILE, "fasta")}
seq_lengths = {rec.id: len(rec.seq) for rec in SeqIO.parse(FASTA_FILE, "fasta")}

promoter_regions = {}
gene_list = []

with open(GFF_FILE) as gff:
    for line in gff:
        if line.startswith("#"):
            continue
        fields = line.strip().split("\t")
        if len(fields) < 9 or fields[2] != "gene":
            continue
        seqid = fields[0]
        start = int(fields[3])
        end = int(fields[4])
        strand = fields[6]
        attrs = dict(item.split("=") for item in fields[8].split(";") if "=" in item)
        gene_id = attrs.get("ID", f"gene_{len(gene_list)+1}")

        if seqid not in promoter_regions:
            promoter_regions[seqid] = []

        if strand == "+":
            p_start = max(1, start - UPSTREAM_FOR_NEGATIVE)
            p_end = start
        else:
            p_start = end
            p_end = min(seq_lengths.get(seqid, end), end + UPSTREAM_FOR_NEGATIVE)

        if p_start < p_end:
            promoter_regions[seqid].append((p_start, p_end))

        gene_list.append((gene_id, seqid, start, end, strand))

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged

non_promoter_segments = []
for seqid in seq_lengths:
    merged = merge_intervals(promoter_regions.get(seqid, []))
    gaps = []
    pos = 1
    for s, e in merged:
        if pos < s:
            gaps.append((pos, s - 1))
        pos = max(pos, e + 1)
    if pos <= seq_lengths[seqid]:
        gaps.append((pos, seq_lengths[seqid]))
    for s, e in gaps:
        length = e - s + 1
        if length >= PROMOTER_LENGTH:
            for i in range(length // PROMOTER_LENGTH):
                seg_start = s + i * PROMOTER_LENGTH
                seg_end = seg_start + PROMOTER_LENGTH - 1
                non_promoter_segments.append((seqid, seg_start, seg_end))

random.shuffle(non_promoter_segments)

with open(OUTPUT_POSITIVE, "w") as pos_out, open(OUTPUT_NEGATIVE, "w") as neg_out:
    extracted_neg = 0
    for gene_id, seqid, start, end, strand in gene_list:
        if seqid not in fasta_records:
            continue
        genome_seq = fasta_records[seqid]

        if strand == "+":
            up_start = max(0, start - PROMOTER_LENGTH)
            seq = genome_seq[up_start:start]
        else:
            down_end = min(len(genome_seq), end + PROMOTER_LENGTH)
            seq = genome_seq[end:down_end].reverse_complement()

        if len(seq) >= PROMOTER_LENGTH:
            seq = seq[-PROMOTER_LENGTH:]
        else:
            seq = seq + "N" * (PROMOTER_LENGTH - len(seq))

        pos_out.write(f">{gene_id}\n{seq}\n")

        while extracted_neg < len(gene_list) and non_promoter_segments:
            cand_seqid, s, e = non_promoter_segments.pop()
            cand_seq = fasta_records[cand_seqid][s-1:e]
            if re.match(r'^[ATCG]+$', str(cand_seq).upper()):
                neg_out.write(f>{cand_seqid}_{s}_{e}\n{cand_seq}\n")
                extracted_neg += 1
                break

print(f"Extracted {len(gene_list)} positive sequences to {OUTPUT_POSITIVE}")
print(f"Extracted {extracted_neg} negative sequences to {OUTPUT_NEGATIVE}")
if extracted_neg < len(gene_list):
    print("Warning: Not enough valid non-promoter segments available")