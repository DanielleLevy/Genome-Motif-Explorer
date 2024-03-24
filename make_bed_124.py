
# Define the minimum desired length
desired_length = 124



# Open the input BED file and the output BED file
with open('pos_bed_files/HEK_iM.bed', 'r') as infile, open('pos_bed_file_124/Hek_iM_124.bed', 'w') as outfile:
    for line in infile:
        fields = line.strip().split('\t')
        chrom, start, end = fields[0], int(fields[1]), int(fields[2])
        fields = line.strip().split('\t')
        chrom, start, end = fields[0], int(fields[1]), int(fields[2])
        current_length = end - start
        midpoint = (start + end) // 2  # Calculate the midpoint
        half_new_length = 62  # Half of the desired new length

        # Calculate new start and end positions
        new_start = midpoint - half_new_length
        new_end = midpoint + half_new_length

        # Adjust if new positions fall outside the original segment
        if new_start < start:
            new_end += start - new_start
            new_start = start
        elif new_end > end:
            new_start -= new_end - end
            new_end = end
        if (new_end-new_start) == desired_length:
            outfile.write(f"{chrom}\t{int(new_start)}\t{int(new_end)}\n")

