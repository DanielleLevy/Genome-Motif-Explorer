import re
import random

import tensorflow
import numpy as np
from keras.layers import Input, Conv1D, Dense, Activation, GlobalMaxPooling1D
from keras.models import Model
from keras import optimizers
from keras.regularizers import l2
from itertools import permutations

# Define the model hyperparameters
n_filters = 256
kernel_size = 12
fc = 32
lr = 1e-3
batch_size = 128
epochs = 1
window = random.randint(5, 20)  # Adjust the range as needed
st = random.randint(1, 10)  # Adjust the range as needed
nt = random.randint(1, 10)  # Adjust the range as needed


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

    # Creating Output Layer
    output = Dense(1)(hidden1)
    output = Activation('sigmoid')(output)

    # Final Model Definition
    mdl = Model(inputs=in1, outputs=output, name='{}_{}_{}nt_base_mdl_crossval'.format(st, nt, str(window)))

    opt = tensorflow.keras.optimizers.legacy.Adam(lr=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-8)

    mdl.compile(loss='mse', optimizer=opt, metrics=['accuracy', 'mae'])

    return mdl

def generate_permutations(seq_dict):
    pos_sequences = list(seq_dict.keys())
    for perm in permutations(pos_sequences, len(seq_dict)):
        # Convert the permutation tuple to a list
        perm_list = list(perm)
        # Join the permutation list to create a sequence
        neg_sequence = ''.join(perm_list)
        # Ensure the negative sequence is not in the positive set
        if neg_sequence not in seq_dict:
            yield neg_sequence, 0


def one_hot_encoding(sequence, max_length):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoded_sequence = [mapping.get(base, [0, 0, 0, 0]) for base in sequence]

    # Ensure the sequence has the fixed length by padding it
    while len(encoded_sequence) < max_length:
        encoded_sequence.append([0, 0, 0, 0])

    return np.array(encoded_sequence)

def main():
    # Define a regular expression pattern to match the sequences (including both upper and lower case letters)
    pattern = r'>chr\d+:\d+-\d+\n([ACGTacgt]+)'

    # Create a dictionary to store both positive and negative sequences with classifications
    sequence_dict = {}

    with open('WDLPS_G4.txt', 'r') as file:
        data = file.read()

    # Use re.findall to extract all positive sequences
    matches = re.findall(pattern, data)

    # Convert the matched sequences to uppercase and store them in the dictionary with classification 1
    for seq in matches:
        sequence = seq.upper()
        sequence_dict[sequence] = 1

    # Generate a set of negative sequences that are permutations of the positive set with classification 0
    num_negatives = len(sequence_dict)  # Same number of negatives as positives
    # Iterate through the generator and add negative sequences to the dictionary
    neg_generator = generate_permutations(sequence_dict)
    for neg_seq, classification in neg_generator:
        sequence_dict[neg_seq] = classification
        # Stop when we have generated the desired number of negative sequences
        if len(sequence_dict) >= num_negatives * 2:  # Multiply by 2 to have an equal number of positives and negatives
            break
    items = list(sequence_dict.items())
    random.shuffle(items)
    shuffled_dict = dict(items)
    # Extract sequences and classifications from the dictionary
    sequences = list(shuffled_dict.keys())
    classifications = [shuffled_dict[seq] for seq in sequences]
    # Convert the sequences to numerical input data (X) with padding
    sequence_lengths = np.array([len(seq) for seq in sequences])
    max_sequence_length = np.max(sequence_lengths)
    X = np.array([one_hot_encoding(seq,max_sequence_length) for seq in sequences], max_sequence_length)
    # Define your target labels (y) based on the classifications
    y = np.array(classifications)
    # Convert the sequences to numerical input data (X)
    # You'll need to convert your DNA sequences to numerical data, e.g., one-hot encoding.

    # Split the data into training and testing sets
    split_ratio = 0.8  # Adjust as needed
    split_index = int(len(X) * split_ratio)
    x_train, x_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Create the model
    my_model = model(X.shape[1:], window, st, nt)

    # Compile the model
    my_model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'mae'])

    # Fit the model on your data
    my_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    # Evaluate the model
    test_scores = my_model.evaluate(x_test, y_test)
    print("Test loss:", test_scores[0])
    print("Test Accuracy:", test_scores[1])
    print("Test MAE:", test_scores[2])

if __name__ == "__main__":
    main()
