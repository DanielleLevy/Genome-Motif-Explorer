import time
import random
import tensorflow
from keras.layers import Input, Conv1D, Dense, Activation, GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
import numpy as np
import pandas as pd
from Bio import SeqIO
import json
import re
from keras.layers import Lambda
from tensorflow.keras.metrics import AUC


# Define the model hyperparameters
n_filters = 256
kernel_size = 12
fc = 32
lr = 1e-3
batch_size = 128
epochs = 1
window = random.randint(5, 20)
st = random.randint(1, 10)
nt = random.randint(1, 10)
seq_length = 124
class Sequence:
    def __init__(self, sequence, coordinate ,label_nuc, label=None):
        self.sequence = sequence
        self.coordinate = coordinate
        self.label = label
        self.label_first_nuc =label_nuc
        self.prediction = None
    def set_prediction(self, prediction):
        self.prediction = prediction

    def get_encoded_sequence(self):
        mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
        return np.array([mapping.get(base, [0, 0, 0, 0]) for base in self.sequence], dtype=np.int8)




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


def calculate_reverse_complement(sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reversed_sequence = sequence[::-1]  # Reverse the sequence
    reverse_complement = [complement_dict[base] for base in reversed_sequence]
    return ''.join(reverse_complement)

def createTrainlist():
    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    train_sequences = []

    with open(positive_file, 'r') as file:
        data = file.read()

    matches = re.findall(pattern, data)

    for match in matches:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        # Extract 62 bases to the right and 62 bases to the left from the midpoint
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if (len(extracted_sequence) == 124):
            complement = calculate_reverse_complement(extracted_sequence)
            g_count_sequence = extracted_sequence.count('G')
            g_count_complement = complement.count('G')

            sequence_to_use = extracted_sequence if g_count_sequence >= g_count_complement else complement
            train_sequences.append(Sequence(sequence_to_use, chromosome, 0, 1))


    return train_sequences


def add_negatives_to_list(sequence_list, negative_file):
    with open(negative_file, 'r') as file:
        data = file.read()

    pattern = r'>(chr\d+:\d+-\d+)\n([ACGTacgt]+)'
    matches = re.findall(pattern, data)

    for match in matches:
        chromosome, sequence = match
        sequence = sequence.upper()
        midpoint = len(sequence) // 2
        extracted_sequence = sequence[midpoint - 62:midpoint + 62]
        if len(extracted_sequence) == 124:
            sequence_list.append(Sequence(extracted_sequence, chromosome, [], 0))

    return sequence_list



def read_fasta_file(fasta_file_path, chromosome):
    # Reads a FASTA file and returns the sequence for the specified chromosome
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        if record.id == chromosome:
            return str(record.seq).upper()
    return None

def read_bed_file(bed_file_path):
    # Reads a BED file and returns the regions with IM structures
    bed_df = pd.read_csv(bed_file_path, sep='\t', header=None, names=['chr', 'start', 'end'])
    return bed_df


def extract_sequences_and_labels(genome_sequence, bed_regions, chunk_start_index, sequence_length=124):
    labeled_genome = [0] * len(genome_sequence)
    chunk_end_index = chunk_start_index + len(genome_sequence)

    for _, row in bed_regions.iterrows():
        if row['chr'] == 'chr1':
            start_in_chunk = max(row['start'], chunk_start_index) - chunk_start_index
            end_in_chunk = min(row['end'], chunk_end_index) - chunk_start_index

            for pos in range(start_in_chunk, end_in_chunk):
                if pos < len(labeled_genome):
                    labeled_genome[pos] = 1

    sequence_objects = []
    for i in range(len(genome_sequence) - sequence_length + 1):
        seq = genome_sequence[i:i + sequence_length]
        label = max(labeled_genome[i:i + sequence_length])
        label_first_nucleotide = labeled_genome[i]
        sequence_objects.append(Sequence(seq, chunk_start_index + i, label_first_nucleotide, label))
    return sequence_objects




def save_sequences_to_csv(sequences, filename, mode='a'):
    header = True if mode == 'w' else False
    data = [{
        'Coordinate': seq.coordinate,
        'Sequence': seq.sequence,
        'Label': seq.label,
        'Labels': json.dumps(seq.labels),
        'Prediction': seq.prediction
    } for seq in sequences]
    df = pd.DataFrame(data)
    df.to_csv(filename, mode=mode, index=False, header=header)





# Main function
if __name__ == "__main__":
    # Part 1: Training the model with training data
    positive_file = 'pos_txt_files/HEK_iM.txt'
    negative_file_train = 'txt_permutaion/HEK_iM_perm_neg.txt'

    # Generate training sequences
    train_sequences = createTrainlist()
    train_sequences = add_negatives_to_list(train_sequences, negative_file_train)

    # Prepare training data
    x_train = np.array([seq.get_encoded_sequence() for seq in train_sequences])
    y_train = np.array([seq.label for seq in train_sequences])

    # Create and fit the model
    nn_model = model(x_train.shape[1:], window, st, nt)
    nn_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    # Part 2: Processing the genome sequence in chunks
    bed_regions = read_bed_file('pos_bed_file_124/Hek_iM_124.bed')
    genome_sequence = read_fasta_file('genome/hg38.fa', 'chr1')
    #make genome_sequence without N:
    genome_sequence = genome_sequence.replace('N','')
    segment_size = 200000
    for start_idx in range(10000, len(genome_sequence), segment_size):
        start = time.time()
        end_idx = min(start_idx + segment_size, len(genome_sequence))
        part_genome_sequence = genome_sequence[start_idx:end_idx]

        # Extract sequences and labels into Sequence objects
        print(time.time())
        sequences = extract_sequences_and_labels(part_genome_sequence, bed_regions, start_idx)
        print(time.time())

        # Prepare data for model application
        encoded_sequences = np.array([seq_obj.get_encoded_sequence() for seq_obj in sequences])
        labels = np.array([seq_obj.label for seq_obj in sequences])  # Extract labels

        # Predict and set predictions for each Sequence object
        predictions = nn_model.predict(encoded_sequences, batch_size=1000)
        for seq_obj, prediction in zip(sequences, predictions):
            seq_obj.set_prediction(prediction[0])
        # Update AUC metric
        # Save to CSV
        csv_mode = 'w' if start_idx == 10000 else 'a'
        save_sequences_to_csv(sequences, 'sequence_predictions_WDLPS_iM.csv', mode=csv_mode)
        end = time.time()

        # Print final AUC score
    print("Processing complete.")


