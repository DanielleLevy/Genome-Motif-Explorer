import random
import re

def calculate_reverse_complement(sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reversed_sequence = sequence[::-1]  # Reverse the sequence
    reverse_complement = [complement_dict[base] for base in reversed_sequence]
    return ''.join(reverse_complement)

def create_neg_from_positive(positive_file, negative_file):
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    with open(positive_file, 'r') as file:
        data = file.read()

    # Create a dictionary to store negative sequences along with their chromosome
    negative_dict = {}
    dict = {}
    matches = re.findall(pattern, data)
    for match in matches:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if len(extracted_sequence) == 124:
            complement = calculate_reverse_complement(extracted_sequence)
            g_count_sequence = extracted_sequence.count('G')
            g_count_complement = complement.count('G')
            if g_count_sequence >= g_count_complement:
                dict[extracted_sequence] = chromosome
            else:
                dict[complement] = chromosome

    # Generate permutations for each sequence in the dictionary
    for sequence, chromosome in dict.items():
        char_list = list(sequence)
        random.shuffle(char_list)
        random_permutation = ''.join(char_list)
        negative_dict[random_permutation] = chromosome

    # Write the negative sequences and their permutations into a text file
    with open(negative_file, 'w') as file:
        for sequence, chromosome in negative_dict.items():
            file.write(f">{chromosome}\n{sequence}\n")

    return negative_dict

# Use the function to create a negative file and dictionary
positive_file = 'pos_txt_files/WDLPS_iM.txt'  # Replace with your positive file path
negative_file = 'txt_permutaion/WDLPS_iM_perm_neg.txt'  # Replace with the desired negative file name
negatives_dict = create_neg_from_positive(positive_file, negative_file)
