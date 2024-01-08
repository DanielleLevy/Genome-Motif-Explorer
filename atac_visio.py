max_coverage = float('-inf')  # Initial maximum coverage value
min_coverage = float('inf')   # Initial minimum coverage value

with open('atac_files/coverage_profile.bedgraph', 'r') as file:
    for line in file:
        if line.startswith('track'):
            continue
        chrom, start, end, cov = line.strip().split('\t')
        cov = int(cov)
        if cov > max_coverage:
            max_coverage = cov
        if cov < min_coverage:
            min_coverage = cov

print(f"Maximum Coverage: {max_coverage}")
print(f"Minimum Coverage: {min_coverage}")
