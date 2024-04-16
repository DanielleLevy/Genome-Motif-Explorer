import re
import pandas as pd
import matplotlib.pyplot as plt
# Function to parse FASTA file and calculate nucleotide frequencies
def createlist(file):
    # Define a regular expression pattern to match sequences with 124 bases centered at position 124
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    seq_list = []
    with open(file, 'r') as file:
        data = file.read()

    # Use re.findall to extract all positive sequences
    matches = re.findall(pattern, data)

    for match in matches:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if (len(extracted_sequence) == 124):
            seq_list.append(extracted_sequence)

    return seq_list
# Function to calculate nucleotide frequencies
def calculate_frequencies(seq_list):
    nucleotide_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    for seq in seq_list:
        for nucleotide in nucleotide_counts.keys():
            nucleotide_counts[nucleotide] += seq.count(nucleotide)
    total_nucleotides = sum(nucleotide_counts.values())
    frequencies = {nt: count / total_nucleotides for nt, count in nucleotide_counts.items()}
    return frequencies

# Assuming you have a file for each condition for WDLPS and HEK
files = {
    'WDLPS_Positive': 'pos_txt_files/WDLPS_iM.txt',
    'WDLPS_RandomNegative': 'random_neg/WDLPS_iM_neg.txt',
    'WDLPS_NegativePermutation': 'txt_permutaion/WDLPS_iM_perm_neg.txt',
    'WDLPS_NegativeGenNullSeq': 'genNellSeq/negWDLPSiM.txt',
    'HEK_Positive': 'pos_txt_files/HEK_iM.txt',
    'HEK_RandomNegative': 'random_neg/HEK_iM_neg.txt',
    'HEK_NegativePermutation': 'txt_permutaion/HEK_iM_perm_neg.txt',
    'HEK_NegativeGenNullSeq': 'genNellSeq/negHekiM.txt'
}

# Calculate frequencies for each file
seq_lists = {group: createlist(file_path) for group, file_path in files.items()}

# Calculate frequencies for each group
frequencies = {group: calculate_frequencies(seq_list) for group, seq_list in seq_lists.items()}

# Convert the frequencies to a DataFrame for easier plotting
df_frequencies = pd.DataFrame(frequencies).T

# Prepare data for plotting
# Extracting values for A, C, G, T for each group and condition
A_values = df_frequencies['A'].values
C_values = df_frequencies['C'].values
G_values = df_frequencies['G'].values
T_values = df_frequencies['T'].values

# Define the categories and groups for labeling
groups = ['WDLPS', 'HEK']
subcategories = ['Positive', 'Random', 'Permutation', 'GenNullSeq']

# Setup the plot
fig, ax = plt.subplots(figsize=(10, 6))  # You can adjust the values accordingly

# Define the y positions to ensure correct gaps
y_positions = [0, 1, 2, 3, 5, 6, 7, 8]  # Notice the gap between the 3rd and 5th positions

# Define colors for each nucleotide type
colors = ['#a1c9f4', '#ffb3e6', '#c2c2f0', '#fdae61']  # Colors for A, C, G, T

# Plot the bars for each nucleotide
for idx, (a, c, g, t) in zip(y_positions, zip(A_values, C_values, G_values, T_values)):
    values = [a, c, g, t]
    left = 0  # Starting point for each group's set of bars
    for value, color in zip(values, colors):
        ax.barh(idx, value, left=left, color=color, edgecolor='none', height=1.0)  # Use height=1.0 for no gaps within groups
        # Centering the numeric values on each bar segment
        ax.text(left + value / 2, idx, f'{value:.3f}', ha='center', va='center', fontsize=9)
        left += value

# Set the y-tick labels
ax.set_yticks(y_positions)
ax.set_yticklabels((subcategories * 2), fontsize=10)  # Repeat labels appropriately

# Position the group labels between the 3rd and 2nd bars from the top within each group
ax.text(1.05, 1.5, 'HEK', ha='right', va='center', fontsize=12, fontweight='bold', color='black', transform=ax.transData)
ax.text(1.08, 6.5, 'WDLPS', ha='right', va='center', fontsize=12, fontweight='bold', color='black', transform=ax.transData)

# Set labels for axes
ax.set_xlabel('Nucleotide frequency', fontsize=12, fontweight='bold')
ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')

# Create and place the legend outside the plot area
legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
ax.legend(legend_elements, ['A', 'C', 'G', 'T'], title="Nucleotides", bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0)

# Remove spines and ticks
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=True)

plt.tight_layout()
plt.savefig("nucleotide_frequencies_plot.png")


