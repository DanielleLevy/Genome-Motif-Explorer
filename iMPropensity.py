import random

import tensorflow
from keras.layers import Input, Conv1D, Dense, Activation, GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

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
seq_length = 124
window_seq=60

class SequenceData:
    def __init__(self, chromosome, sequence, classification=None, accessibility=None,
                 start_coordinate=None, end_coordinate=None, micro=None):
        self.chromosome = chromosome
        self.sequence = sequence.upper()  # Ensuring the sequence is in uppercase
        self.classification = classification
        self.accessibility = accessibility
        self.start_coordinate = start_coordinate
        self.end_coordinate = end_coordinate
        self.microarray_signal = micro

    def calculate_reverse_complement(self):
        complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        reversed_sequence = self.sequence[::-1]  # Reverse the sequence
        return ''.join(complement_dict.get(base, base) for base in reversed_sequence)

    def extracted_sequence(self, length=124):
        midpoint = len(self.sequence) // 2
        cut_from_seq = length // 2
        return self.sequence[midpoint - cut_from_seq:midpoint + cut_from_seq]

def create_train_list(file_path):
    train_list = []
    data = pd.read_csv(file_path)

    for index, row in data.iterrows():
        seq_data = SequenceData(
            chromosome=None,  # Fill in as appropriate
            sequence=row['ProbeSeq'],
            classification=None,  # Fill in as appropriate
            accessibility=None,  # Fill in as appropriate
            start_coordinate=None,  # Fill in as appropriate
            end_coordinate=None,  # Fill in as appropriate
            micro=row['iMab100nM_6.5_5']
        )
        seq_data.sequence = seq_data.extracted_sequence(window_seq)
        train_list.append(seq_data)

    return train_list

def create_test_list(file_path):
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    test_list = []

    with open(file_path, 'r') as file:
        data = file.read()
    matches = re.findall(pattern, data)

    for match in matches:
        chromosome, sequence = match
        seq_data = SequenceData(chromosome, sequence)
        extracted_sequence = seq_data.extracted_sequence(124)

        if len(extracted_sequence) == 124:
            complement = seq_data.calculate_reverse_complement()
            g_count_sequence = extracted_sequence.count('C')
            g_count_complement = complement.count('C')

            # Select the sequence with the most G's
            chosen_sequence = extracted_sequence if g_count_sequence >= g_count_complement else complement
            seq_data.sequence = chosen_sequence  # Update the sequence in seq_data
            test_list.append(seq_data)

    return test_list



def save_signals_to_csv(sequences_test, signals, filename):
    df = pd.DataFrame({'Sequence': sequences_test, 'Signal': signals})
    df.to_csv(filename, index=False)


# Save signals to CSV file

def model(shape, window, st, nt):
    # Creating Input Layer
    in1 = Input(shape=shape)

    # Creating Convolutional Layer
    conv_layer = Conv1D(filters=n_filters, kernel_size=kernel_size, strides=1, kernel_initializer='RandomNormal',
                        activation='relu', kernel_regularizer=l2(5e-3), padding='same', use_bias=True,
                        bias_initializer='RandomNormal')(in1)

    # Creating Pooling Layer
    pool = GlobalMaxPooling1D()(conv_layer)

    # Creating Hidden Layer
    hidden1 = Dense(fc)(pool)
    hidden1 = Activation('relu')(hidden1)

    # Creating Output Layer for regression
    output = Dense(1, activation='linear')(hidden1)  # Changed activation to 'linear' for regression

    # Final Model Definition
    mdl = Model(inputs=in1, outputs=output, name='{}_{}_{}nt_base_mdl_crossval'.format(st, nt, str(window)))

    opt = tensorflow.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-8)

    mdl.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error', 'mean_absolute_error'])
    # Changed loss to 'mean_squared_error' for regression and added appropriate metrics

    return mdl

def one_hot_encoding(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1],  'N': [0.25, 0.25, 0.25, 0.25]}
    encoded_sequence = [mapping.get(base, [0, 0, 0, 0]) for base in sequence]
    return np.array(encoded_sequence)

def load_sequences_from_csv(csv_path):
    """Load sequences from the provided CSV file and create SequenceData objects."""
    data = pd.read_csv(csv_path)
    sequences = data['Mutated_Sequence'].tolist()
    return [SequenceData(chromosome=None, sequence=seq) for seq in sequences]  # Simplified object creation



def main():
    #file_test = 'pos_txt_files/HEK_iM.txt'
    file_train = 'microarray_files/final_table_microarray.csv'
    sequences_df = pd.read_csv(
        'interpation_file/mutations.csv')
    # Create train and test lists using the respective functions
    train_list = create_train_list(file_train)
    #test_list = create_test_list(file_test)
    test_list = load_sequences_from_csv('interpation_file/mutations.csv')

    # Shuffle the training data
    random.shuffle(train_list)

    # Prepare data for training
    x_train = np.array([one_hot_encoding(seq_data.sequence) for seq_data in train_list])
    y_train = np.array([seq_data.microarray_signal for seq_data in train_list])

    # Prepare test data
    #x_test = [seq_data.extracted_sequence(seq_length) for seq_data in test_list]
    x_test = test_list
    # Create the model
    my_model = model((window_seq, 4), window, st, nt)

    # Fit the model on your training data
    history = my_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    # Prepare data for batch prediction
    all_encoded_windows = []
    sequence_index_map = []  # Maps each window to its original sequence index

    # Before starting the loop, ensure x_test is a list of SequenceData objects
    x_test = [seq_data.sequence for seq_data in test_list]  # This will extract the sequence strings

    # Then, when preparing data for batch prediction
    for sequence_index, sequence_str in enumerate(x_test):
        sequence_length = len(sequence_str)  # Now, this should work as expected
        for i in range(0, sequence_length - window_seq + 1, window_seq):
            extracted_sequence = sequence_str[i:i + window_seq]
            encoded_sequence = one_hot_encoding(extracted_sequence)
            all_encoded_windows.append(encoded_sequence)
            sequence_index_map.append(sequence_index)

    # Perform batch predictions
    all_encoded_windows = np.array(all_encoded_windows)
    all_predictions = my_model.predict(all_encoded_windows).flatten()

    # Process predictions to find the max signal for each original sequence
    predictions = [0] * len(x_test)  # Initialize list for max predictions per sequence
    for idx, prediction in zip(sequence_index_map, all_predictions):
        predictions[idx] = max(predictions[idx], prediction)

    # Save predictions to CSV
    #sequences_test = [seq_data.sequence for seq_data in test_list]  # Extract full sequences for saving
    #print(len(sequences_test), len(predictions))
    sequences_df['Signal'] = predictions
    # Save the dataframe with the new predictions to the same CSV file
    sequences_df.to_csv('mutations.csv',
                        index=False)  # Replace with your desired file path

    print("Predictions have been added to the CSV file.")
    #save_signals_to_csv(sequences_test, predictions, 'microarray_files/signals_data_interapt.csv')

main()



