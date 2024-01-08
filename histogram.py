import re
import numpy as np
import matplotlib.pyplot as plt

def histogram():
    # Define a regular expression pattern to match the sequences (including both upper and lower case letters)
    pattern = r'>([^\n]+)\n([ACGTacgt\n]+)'


    # Create a dictionary to store both positive and negative sequences with classifications
    sequence_dict = {}

    with open('genNellSeq/negHekG4gen.txt', 'r') as file:
        data = file.read()

    # Use re.findall to extract all positive sequences
    matches = re.findall(pattern, data)

    # Convert the matched sequences to uppercase and store them in the dictionary with classification 1
    for header, sequence in matches:
        sequence = sequence.replace('\n', '')  # Remove newline characters within the sequence
        sequence_dict[sequence]=1
    # Extract the lengths of the sequences and store them in a list
    sequence_lengths = [len(seq) for seq in sequence_dict]
    # Calculate the median length
    median_length = np.median(sequence_lengths)
    # Create a histogram
    n, bins, patches = plt.hist(sequence_lengths, bins=1000, color='blue', edgecolor='black')
    plt.title('Histogram of DNA Sequence Lengths HEK-iM')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')

    # Set the x-axis limits to focus on a specific range (adjust these values as needed)
    plt.xlim(0, 1250)

    # Find the bin with the highest frequency
    max_bin_frequency = max(n)
    max_bin_index = np.where(n == max_bin_frequency)[0][0]
    max_bin_value = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2

    # Display the median, mode, and the value of the bin with the highest frequency on the plot
    plt.axvline(max_bin_value, color='purple', linestyle='dashed', linewidth=2, label='Max Bin Value')
    plt.legend()

    plt.show()
    print(max_bin_value)
    print(median_length)
    min_length = min(sequence_lengths)
    print(min_length)


histogram()
