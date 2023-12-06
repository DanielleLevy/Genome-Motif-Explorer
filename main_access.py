import random

import tensorflow
from keras.layers import Input, Conv1D, Dense, Activation, GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Define the model hyperparameters
n_filters = 256
kernel_size = 12
fc = 32
lr = 1e-3
batch_size = 128
epochs = 5
window = random.randint(5, 20)  # Adjust the range as needed
st = random.randint(1, 10)  # Adjust the range as needed
nt = random.randint(1, 10)  # Adjust the range as needed
seq_lengh = 124
# Reading the data from the file into a DataFrame
columns = ['Chromosome', 'Start', 'End', 'Score']
df_positive = pd.read_csv('HEK_G4_SCORES', sep=',', header=0, names=columns)
df_negative = pd.read_csv('HEK_G4_neg_SCORES', sep=',', header=0, names=columns)

# Define the SequenceData class
class SequenceData:
    def __init__(self, chromosome, sequence, classification, accessibility, start_coordinate, end_coordinate):
        self.chromosome = chromosome
        self.sequence = sequence
        self.classification = classification
        self.accessibility = accessibility
        self.start_coordinate = start_coordinate
        self.end_coordinate = end_coordinate


def model(shape, window, st, nt):
    # Creating Input Layer
    in1 = Input(shape=shape)
    in2 = Input(shape=(1,))  # Additional input for accessibility scores

    # Creating Convolutional Layer
    conv_layer = Conv1D(filters=n_filters, kernel_size=kernel_size, strides=1, kernel_initializer='RandomNormal',
                        activation='relu', kernel_regularizer=l2(5e-3), padding='same', use_bias=True,
                        bias_initializer='RandomNormal')(in1)

    # Creating Pooling Layer
    pool = GlobalMaxPooling1D()(conv_layer)

    # Concatenate the output of the convolutional layer with the accessibility input
    merged = tensorflow.keras.layers.concatenate([pool, in2])

    # Creating Hidden Layer
    hidden1 = Dense(fc)(merged)
    hidden1 = Activation('relu')(hidden1)

    # Creating Output Layer
    output = Dense(1)(hidden1)
    output = Activation('sigmoid')(output)

    # Final Model Definition
    mdl = Model(inputs=[in1, in2], outputs=output, name='{}_{}_{}nt_base_mdl_crossval'.format(st, nt, str(window)))

    opt = tensorflow.keras.optimizers.legacy.Adam(lr=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-8)

    mdl.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tensorflow.keras.metrics.AUC()])

    return mdl

def one_hot_encoding(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoded_sequence = [mapping.get(base, [0, 0, 0, 0]) for base in sequence]
    return np.array(encoded_sequence)
def calculate_reverse_complement(sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reversed_sequence = sequence[::-1]  # Reverse the sequence
    reverse_complement = [complement_dict[base] for base in reversed_sequence]
    return ''.join(reverse_complement)
# with chromosom1 test



def createDict(positive_file):
    # Define a regular expression pattern to match sequences with 124 bases centered at position 124
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'

    # Create a list to store SequenceData objects for both positive and negative sequences
    train_data = []
    test_data = []

    with open(positive_file, 'r') as file:
        data = file.read()

    # Use re.findall to extract all positive sequences
    matches = re.findall(pattern, data)

    for match in matches:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if len(extracted_sequence) == 124:
            complement = calculate_reverse_complement(extracted_sequence)
            g_count_sequence = extracted_sequence.count('G')
            g_count_complement = complement.count('G')
            chrom_parts = chromosome.split(':')
            start_pos = int(chrom_parts[1].split('-')[0]) + (midpoint - 62)
            end_pos = int(chrom_parts[1].split('-')[0]) + (midpoint + 62)
            matching_rows = df_positive[
                (df_positive['Chromosome'] == chrom_parts[0]) & (df_positive['Start'] == start_pos) & (df_positive['End'] == end_pos)]
            accessibility_score = matching_rows['Score'].values[0]
            if g_count_sequence >= g_count_complement:
                seq_data = SequenceData(chrom_parts[0], extracted_sequence, 1, accessibility_score, start_pos, end_pos)  # Assuming positive classification
                if re.search(r'\bchr1\b', chromosome):
                    test_data.append(seq_data)
                else:
                    train_data.append(seq_data)
            else:
                complement_seq_data = SequenceData(chrom_parts[0], complement, 1, accessibility_score, start_pos, end_pos)
                if re.search(r'\bchr1\b', chromosome):
                    test_data.append(complement_seq_data)
                else:
                    train_data.append(complement_seq_data)
    return test_data, train_data

def createDictWDLPS():
    # Define a regular expression pattern to match sequences with 124 bases centered at position 124
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'

    # Create a dictionary to store both positive and negative sequences with classifications
    train_dict = {}
    test_dict = {}

    with open(positive_file_test, 'r') as file:
        data_test = file.read()
    with open(positive_file_train, 'r') as file:
        data_train = file.read()
    # Use re.findall to extract all positive sequences
    matches_test = re.findall(pattern, data_test)
    matches_train = re.findall(pattern, data_train)
    for match in matches_test:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if (len(extracted_sequence) == 124):
            complement = calculate_reverse_complement(extracted_sequence)
            g_count_sequence = sequence.count('G')
            g_count_complement = complement.count('G')

            if g_count_sequence >= g_count_complement:
                test_dict[extracted_sequence] = 1
            else:
                test_dict[complement] = 1
    for match in matches_train:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if (len(extracted_sequence) == 124):
            complement = calculate_reverse_complement(extracted_sequence)
            g_count_sequence = sequence.count('G')
            g_count_complement = complement.count('G')

            if g_count_sequence >= g_count_complement:
                   train_dict[extracted_sequence] = 1
            else:
                  train_dict[complement] = 1
    return test_dict, train_dict
def add_negatives_to_dict(test_dict, train_dict, negative_file):
    with open(negative_file, 'r') as file:
        data = file.read()
    # Define a regular expression pattern to match sequences with 124 bases centered at position 124
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    matches = re.findall(pattern, data)
    for match in matches:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if len(extracted_sequence) == 124:
            # Add the negative sequence to the appropriate dictionary with label 0
            if re.search(r'\bchr1\b', chromosome):
                test_dict.append(SequenceData(chromosome, extracted_sequence, 0, None))  # Assuming SequenceData class is used
            else:
                train_dict.append(SequenceData(chromosome, extracted_sequence, 0, None))  # Assuming SequenceData class is used

    return test_dict, train_dict





import re

def add_negatives_to_dict(test_dict, train_dict, negative_file, df_negative):
    with open(negative_file, 'r') as file:
        data = file.read()
    # Define a regular expression pattern to match sequences with 124 bases centered at position 124
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    matches = re.findall(pattern, data)
    for match in matches:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if len(extracted_sequence) == 124:
            chrom_parts = chromosome.split(':')
            start_pos = int(chrom_parts[1].split('-')[0]) + (midpoint - 62)
            end_pos = int(chrom_parts[1].split('-')[0]) + (midpoint + 62)
            matching_rows = df_negative[
                (df_negative['Chromosome'] == chrom_parts[0]) & (df_negative['Start'] == start_pos) & (df_negative['End'] == end_pos)]
            if not matching_rows.empty:
                accessibility_score = matching_rows['Score'].values[0]
            else:
                accessibility_score = None
            # Add the negative sequence to the appropriate dictionary with label 0
            if re.search(r'\bchr1\b', chromosome):
                test_dict.append(SequenceData(chrom_parts[0], extracted_sequence, 0, accessibility_score, start_pos, end_pos))
            else:
                train_dict.append(SequenceData(chrom_parts[0], extracted_sequence, 0, accessibility_score, start_pos, end_pos))

    return test_dict, train_dict



def add_negatives_to_dict_WDLPS(dic, negative_file):
    with open(negative_file, 'r') as file:
        data = file.read()

    # Assuming the negative sequences are stored line by line in the file
    neg_sequences = data.strip().split('\n')
    for sequence in neg_sequences:
        # You may need to process the sequence here if needed (e.g., uppercase, length check)
        # For example:
        sequence = sequence.upper()
        dic[sequence] = 0

    return dic
def plot_metrics(history1, history2, history3, title1, title2, title3):
    # Plot training & validation loss values
    plt.figure(figsize=(12, 9))

    # Plot AUC values
    plt.subplot(3, 1, 3)
    plt.plot(history1.history['auc'], label=f'{title1} - Train AUC')
    plt.plot(history1.history['val_auc'], label=f'{title1} - Validation AUC')
    plt.plot(history2.history['auc_1'], label=f'{title2} - Train AUC', linestyle='dashed')
    plt.plot(history2.history['val_auc_1'], label=f'{title2} - Validation AUC', linestyle='dashed')
    plt.plot(history3.history['auc_2'], label=f'{title3} - Train AUC', linestyle='dotted')
    plt.plot(history3.history['val_auc_2'], label=f'{title3} - Validation AUC', linestyle='dotted')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.suptitle(f"{title1} vs {title2} vs {title3}")
    plt.tight_layout()
    plt.show()
positive_file='HEK_G4.txt'

#main with permutaions
def main_permutions():
    negative_file='HEK_G4_perm_neg.txt'
    positive_file_train='HEK_iM.txt'
    negative_file_train='HEK_iM_neg.txt'
    positive_file_test='WDLPS_iM.txt'
    negative_file_test='WDLPS_iM_neg.txt'
    test_dict, train_dict = createDict()
    test_dict,train_dict= add_negatives_to_dict(test_dict,train_dict,negative_file)
    items_train = list(train_dict.items())
    random.shuffle(items_train)
    shuffled_dict_train = dict(items_train)
    items_test = list(test_dict.items())
    random.shuffle(items_test)
    shuffled_dict_test = dict(items_test)
    # Extract sequences and classifications from the dictionary
    sequences_train = list(shuffled_dict_train.keys())
    classifications_train = [shuffled_dict_train[seq] for seq in shuffled_dict_train]
    sequences_test = list(shuffled_dict_test.keys())
    classifications_test = [shuffled_dict_test[seq] for seq in shuffled_dict_test]
    # Convert the sequences to numerical input data (X) with padding
    x_train = np.array([one_hot_encoding(seq) for seq in sequences_train])
    # Define your target labels (y) based on the classifications
    y_train = np.array(classifications_train)
    x_test = np.array([one_hot_encoding(seq) for seq in sequences_test])
    y_test = np.array(classifications_test)

    # Create the model
    my_model = model(x_train.shape[1:], window, st, nt)

    # Fit the model on your data
    history = my_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    # Evaluate the model
    test_scores = my_model.evaluate(x_test, y_test)
    print("Test loss:", test_scores[0])
    print("Test Accuracy:", test_scores[1])
    print("Test AUC:", test_scores[2])
    #plot_metric(history, title="Permutations Method")
    return history

#main with rangom genertive
def main_random():
    negative_file = 'HEK_G4_neg.txt'
    positive_file_train = 'HEK_iM.txt'
    negative_file_train = 'HEK_iM_neg.txt'
    positive_file_test = 'WDLPS_iM.txt'
    negative_file_test = 'WDLPS_iM_neg.txt'

    # Create the lists to store SequenceData objects
    test_data, train_data = createDict('HEK_G4.txt')

    # Add negative sequences to the appropriate lists
    test_data, train_data = add_negatives_to_dict(test_data, train_data, negative_file,df_negative)

    # Shuffle the data
    random.shuffle(train_data)
    random.shuffle(test_data)

    # Extract sequences and classifications from the list of SequenceData objects
    sequences_train = [seq.sequence for seq in train_data]
    classifications_train = [seq.classification for seq in train_data]
    sequences_test = [seq.sequence for seq in test_data]
    classifications_test = [seq.classification for seq in test_data]

    # Convert the sequences to numerical input data (X) with padding
    x_train = np.array([one_hot_encoding(seq) for seq in sequences_train])
    y_train = np.array(classifications_train)

    x_test = np.array([one_hot_encoding(seq) for seq in sequences_test])
    y_test = np.array(classifications_test)
    # Extract accessibility scores from SequenceData objects
    x_train_accessibility = np.array([seq.accessibility for seq in train_data]).reshape(-1, 1)
    x_test_accessibility = np.array([seq.accessibility for seq in test_data]).reshape(-1, 1)
    # Create the model
    my_model = model(x_train.shape[1:], window, st, nt)

    # Fit the model on your data (including accessibility and classifications)
    history = my_model.fit([x_train, x_train_accessibility], y_train, batch_size=batch_size,
                           epochs=epochs, validation_data=([x_test, x_test_accessibility], y_test))

    # Evaluate the model
    test_scores = my_model.evaluate([x_test, x_test_accessibility], y_test)

    # Evaluate the model
    print("Test loss:", test_scores[0])
    print("Test Accuracy:", test_scores[1])
    print("Test AUC:", test_scores[2])

    return history


#main with rangom genertive
def main_genNullSeq():
    negative_file = 'negHekG4gen.txt'
    positive_file_train = 'HEK_iM.txt'
    negative_file_train = 'HEK_iM_neg.txt'
    positive_file_test = 'WDLPS_iM.txt'
    negative_file_test = 'WDLPS_iM_neg.txt'

    # Create the lists to store SequenceData objects
    test_data, train_data = createDict('HEK_G4.txt')

    # Add negative sequences to the appropriate lists
    test_data, train_data = add_negatives_to_dict(test_data, train_data, negative_file,df_negative)

    # Shuffle the data
    random.shuffle(train_data)
    random.shuffle(test_data)

    # Extract sequences and classifications from the list of SequenceData objects
    sequences_train = [seq.sequence for seq in train_data]
    classifications_train = [seq.classification for seq in train_data]
    sequences_test = [seq.sequence for seq in test_data]
    classifications_test = [seq.classification for seq in test_data]

    # Convert the sequences to numerical input data (X) with padding
    x_train = np.array([one_hot_encoding(seq) for seq in sequences_train])
    y_train = np.array(classifications_train)

    x_test = np.array([one_hot_encoding(seq) for seq in sequences_test])
    y_test = np.array(classifications_test)
    # Extract accessibility scores from SequenceData objects
    x_train_accessibility = np.array([seq.accessibility for seq in train_data]).reshape(-1, 1)
    x_test_accessibility = np.array([seq.accessibility for seq in test_data]).reshape(-1, 1)
    # Create the model
    my_model = model(x_train.shape[1:], window, st, nt)

    # Fit the model on your data (including accessibility and classifications)
    history = my_model.fit([x_train, x_train_accessibility], y_train, batch_size=batch_size,
                           epochs=epochs, validation_data=([x_test, x_test_accessibility], y_test))

    # Evaluate the model
    test_scores = my_model.evaluate([x_test, x_test_accessibility], y_test)


    print("Test loss:", test_scores[0])
    print("Test Accuracy:", test_scores[1])
    print("Test AUC:", test_scores[2])

    return history

#history_permutations = main_permutions()
history_random = main_random()
#history_genNull = main_genNullSeq()

#plot_metrics(history_permutations, history_random,history_genNull, "Permutations Method", "Random Method","genNullSeq Method")