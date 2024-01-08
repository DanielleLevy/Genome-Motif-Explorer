import math

# Define the minimum desired length
desired_length = 124

# Dictionary to store chromosome lengths
chromosome_lengths = {}

# Read chromosome lengths from a text file
with open('genome/hg38.chrom.sizes', 'r') as length_file:
    for line in length_file:
        fields = line.strip().split('\t')
        if len(fields) == 2:
            chrom, length = fields[0], int(fields[1])
            chromosome_lengths[chrom] = length

# Open the input BED file and the output BED file
with open('negSetWdlpsIm.bed', 'r') as infile, open('neg_WDLPS_iM.bed', 'w') as outfile:
    for line in infile:
        fields = line.strip().split('\t')
        chrom, start, end = fields[0], int(fields[1]), int(fields[2])

        # Retrieve the chromosome length from the dictionary
        chrom_length = chromosome_lengths.get(chrom)

        if chrom_length is not None:
            # Calculate the current length of the feature
            current_length = end - start

            if current_length < desired_length:
                # Calculate how much to extend or trim from both start and end
                extension_length = (desired_length - current_length) / 2
                extension_length = math.ceil(extension_length)  # Round up to the nearest whole number

                # Calculate the new start and end coordinates
                new_start = max(0, start - extension_length)
                new_end = min(chrom_length, end + extension_length)

                # Ensure that the feature is at least the desired length
                if new_end - new_start >= desired_length:
                    # Write the modified feature to the output BED file
                    outfile.write(f"{chrom}\t{int(new_start)}\t{int(new_end)}\n")
            else:
                # If the feature is already longer than desired_length, keep it as is
                outfile.write(line)

# No need to explicitly close the output BED file, as it's handled by the "with" statement
