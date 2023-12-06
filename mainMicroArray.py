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
seq_lengh = 124
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

    mdl.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tensorflow.keras.metrics.AUC()])

    return mdl
def one_hot_encoding(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1],  'N': [0, 0, 0, 0]}
    encoded_sequence = [mapping.get(base, [0, 0, 0, 0]) for base in sequence]
    return np.array(encoded_sequence)
def calculate_reverse_complement(sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reversed_sequence = sequence[::-1]  # Reverse the sequence
    reverse_complement = [complement_dict[base] for base in reversed_sequence]
    return ''.join(reverse_complement)

def createTestDict(positive_file_test):
    test_dict = {}
    # Read the CSV file into a DataFrame using pandas
    data = pd.read_csv(positive_file_test)

    # Iterate through the rows of the DataFrame
    for index, row in data.iterrows():
        sequence = row['ProbeSeq']
        signal = row['iMab100nM_6.5_5']
        test_dict[sequence] = signal


    return test_dict
def createDict(positive_file_train):
    # Define a regular expression pattern to match sequences with 124 bases centered at position 124
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'

    # Create a dictionary to store both positive and negative sequences with classifications
    train_dict = {}
    with open(positive_file_train, 'r') as file:
        data_train = file.read()
    # Use re.findall to extract all positive sequences
    matches_train = re.findall(pattern, data_train)

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
    return train_dict







def add_negatives_to_dict_WDLPS(dic, negative_file):
    with open(negative_file, 'r') as file:
        data = file.read()

    # Assuming the negative sequences are stored line by line in the file
    neg_sequences = data.strip().split('\n')
    for sequence in neg_sequences:
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if (len(extracted_sequence) == 124):
            sequence = sequence.upper()
            dic[extracted_sequence] = 0

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

def main_permutions():
    positive_file_train='HEK_iM.txt'
    negative_file_train = 'HEK_iM_perm_neg.txt'
    train_dict = createDict(positive_file_train)
    train_dict = add_negatives_to_dict_WDLPS(train_dict, negative_file_train)
    test_dict=createTestDict('final_table_microarray.csv')
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
    model_predictions = my_model.predict(x_test)
    model_predictions = model_predictions.reshape(-1)

    # חישוב קורלציות
    pearson_corr, _ = pearsonr(y_test, model_predictions)
    spearman_corr, _ = spearmanr(y_test, model_predictions)

    # הדפסת הקורלציות
    print("Pearson Correlation:", pearson_corr)
    print("Spearman Correlation:", spearman_corr)
    return history
def main_random():
    positive_file_train='HEK_iM.txt'
    negative_file_train = 'HEK_iM_neg.txt'
    train_dict = createDict(positive_file_train)
    train_dict = add_negatives_to_dict_WDLPS(train_dict, negative_file_train)
    test_dict=createTestDict('final_table_microarray.csv')
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

    model_predictions = my_model.predict(x_test)
    model_predictions = model_predictions.reshape(-1)

    # חישוב קורלציות
    pearson_corr, _ = pearsonr(y_test, model_predictions)
    spearman_corr, _ = spearmanr(y_test, model_predictions)

    # הדפסת הקורלציות
    print("Pearson Correlation:", pearson_corr)
    print("Spearman Correlation:", spearman_corr)
    return history

def main_genNullSeq():
    positive_file_train='HEK_iM.txt'
    negative_file_train = 'negHekiM.txt'
    train_dict = createDict(positive_file_train)
    train_dict = add_negatives_to_dict_WDLPS(train_dict, negative_file_train)
    test_dict=createTestDict('final_table_microarray.csv')
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

    model_predictions = my_model.predict(x_test)
    model_predictions = model_predictions.reshape(-1)

    # חישוב קורלציות
    pearson_corr, _ = pearsonr(y_test, model_predictions)
    spearman_corr, _ = spearmanr(y_test, model_predictions)

    # הדפסת הקורלציות
    print("Pearson Correlation:", pearson_corr)
    print("Spearman Correlation:", spearman_corr)
    return history

history_permutations = main_permutions()
history_random = main_random()
history_genNull = main_genNullSeq()
model_predictions_permutations = history_permutations.model.predict(x_test)

plot_metrics(history_permutations, history_random,history_genNull, "Permutations Method", "Random Method","genNullSeq Method")