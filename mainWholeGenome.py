
import random
import tensorflow
from keras.layers import Input, Conv1D, Dense, Activation, GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from Bio import SeqIO
import json
import os

# Define the model hyperparameters
n_filters = 256
kernel_size = 12
fc = 32
lr = 1e-3
batch_size = 128
epochs = 5
window = random.randint(5, 20)
st = random.randint(1, 10)
nt = random.randint(1, 10)
seq_length = 124

class Sequence:
    def __init__(self, sequence, coordinate ,labels, label=None):
        self.sequence = sequence
        self.coordinate = coordinate
        self.label = label
        self.labels =labels
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
            # מציאת החיתוך בין האזור מהקובץ BED לבין הקטע שאנו מעבדים
            start_in_chunk = max(row['start'], chunk_start_index) - chunk_start_index
            end_in_chunk = min(row['end'], chunk_end_index) - chunk_start_index

            # סימון הפוזיציות בתוך הקטע
            for pos in range(start_in_chunk, end_in_chunk):
                if pos < len(labeled_genome):
                    labeled_genome[pos] = 1

    sequence_objects = []
    for i in range(len(genome_sequence) - sequence_length + 1):
        seq = genome_sequence[i:i + sequence_length]
        label = max(labeled_genome[i:i + sequence_length])
        sequence_objects.append(Sequence(seq, chunk_start_index + i, labeled_genome[i:i + sequence_length], label))
    return sequence_objects




def save_sequences_to_csv(sequences, filename, mode='a'):
    # mode='a' מאפשר להוסיף נתונים לקובץ קיים
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



def generate_predictions_and_plot(sequences, output_dir='plots', output_filename='prediction_plot.png'):
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract predictions from each Sequence object
    predictions = [seq_obj.prediction for seq_obj in sequences]

    # Convert to numpy array for processing
    predictions_array = np.array(predictions)

    # Determine the number of bins
    num_unique_predictions = len(np.unique(predictions_array))
    num_bins = min(num_unique_predictions, 100)  # Set a practical upper limit

    plt.hist(predictions_array, bins=num_bins)
    plt.xlabel('Sequence Index')
    plt.ylabel('Probability of IM Structure')
    plt.title('IM Structure Probability Distribution')

    # Save the plot instead of showing it
    plt.savefig(os.path.join(output_dir, output_filename), bbox_inches='tight')
    plt.close()  # Close the plot to free memory



# MAIN Execution
if __name__ == "__main__":
    # Read BED file and FASTA file
    bed_regions = read_bed_file('pos_bed_files/WDLPS_iM_high_confidence_peaks.bed')
    genome_sequence = read_fasta_file('genome/hg38.fa', 'chr1')
    for start_idx in range(0, len(genome_sequence), 20000):
        end_idx = min(start_idx + 20000, len(genome_sequence))
        part_genome_sequence = genome_sequence[start_idx:end_idx]
        # Extract sequences and labels into Sequence objects
        sequences = extract_sequences_and_labels(part_genome_sequence, bed_regions,start_idx)

        # Prepare and compile the model
        model_shape = (124, 4)  # Adjust based on your encoding
        nn_model = model(model_shape, window, st, nt)
        # Prepare data for training
        encoded_sequences = np.array([seq_obj.get_encoded_sequence() for seq_obj in sequences])
        labels = np.array([seq_obj.label for seq_obj in sequences])
        # Train the model
        nn_model.fit(encoded_sequences, labels, epochs=epochs, batch_size=batch_size)
        # Predict and set predictions for each Sequence object
        predictions = nn_model.predict(encoded_sequences)
        for seq_obj, prediction in zip(sequences, predictions):
            seq_obj.set_prediction(prediction[0])
        # Save to CSV
        csv_mode = 'w' if start_idx == 0 else 'a'
        save_sequences_to_csv(sequences, 'sequence_predictions_WDLPS_iM.csv', mode=csv_mode)
    print("im Done")