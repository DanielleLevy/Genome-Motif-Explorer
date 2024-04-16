import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Function to load data
def load_data(filepath):
    return pd.read_csv(filepath)


# Function to plot ROC curves
def plot_roc_curves(files_and_models):
    fig, axes = plt.subplots(nrows=1, ncols=len(files_and_models), figsize=(18, 6), sharey=True)
    fig.suptitle('ROC Curves by Negative Group and Model')

    for ax, (group, file_model_tuples) in zip(axes, files_and_models.items()):
        for file_name, model_label in file_model_tuples:
            df = load_data(file_name)
            true_labels = df['True_Labels']
            predictions = df['Predictions']
            fpr, tpr, _ = roc_curve(true_labels, predictions)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{model_label} (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_title(group)
        if ax == axes[0]:
            ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'AUROC.png')
    plt.show()


# Function to plot prediction distributions
def plot_prediction_distributions(df, prediction_columns):
    sns.set(style="whitegrid")
    for prediction_column in prediction_columns:
        plt.figure(figsize=(18, 6))
        for i, loop in enumerate(['Loop1_Length', 'Loop2_Length', 'Loop3_Length'], start=1):
            plt.subplot(1, 3, i)
            sns.boxplot(x=df[loop], y=df[prediction_column], palette='viridis')
            plt.title(f'{loop} - {prediction_column}')
            plt.xlabel('Loop Length')
            plt.ylabel('Prediction Score')
        plt.tight_layout()
        plt.savefig(f'plot_{prediction_column}.png')
        plt.show()


# Function to plot heatmaps for mutation effects
def plot_and_save_heatmap_for_model(prediction_column, mutations_data):
    baseline = mutations_data[mutations_data['Original_Nucleotide'] == mutations_data['Mutated_Nucleotide']]
    baseline_dict = dict(zip(baseline['Position'], baseline[prediction_column]))

    def calculate_difference(row):
        return row[prediction_column] - baseline_dict.get(row['Position'], row[prediction_column])

    mutations_data['Difference'] = mutations_data.apply(calculate_difference, axis=1)
    differences = mutations_data[mutations_data['Original_Nucleotide'] != mutations_data['Mutated_Nucleotide']]
    heatmap_data = differences.pivot_table(index='Mutated_Nucleotide', columns='Position', values='Difference',
                                           aggfunc='mean').fillna(0).reindex(['A', 'T', 'C', 'G'])
    plt.figure(figsize=(20, 5))
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, cbar_kws={'label': 'Prediction Score Difference'})
    plt.title(f'Heatmap of {prediction_column} Score Differences')
    plt.xlabel('Position')
    plt.ylabel('Mutated Nucleotide')
    plt.savefig(f"{prediction_column}_heatmap.png")
    plt.close()


# Main execution
if __name__ == "__main__":
    # Define your file names and models here
    files_and_models = {
        'random': [
            ('../AUROC/predictions_and_true_labels_seq_random.csv', 'Sequence only'),
            ('../AUROC/predictions_and_true_labels_acc_random.csv', 'Sequence + ATAC'),
            ('../AUROC/predictions_and_true_labels_micro_rand.csv', 'Sequence + microarray signals'),
            ('../AUROC/predictions_and_true_labels_acc_mic_ran.csv', 'Sequence + microarray signals + ATAC')
        ],
        'permutation': [
            ('../AUROC/predictions_and_true_labels_seq_perm.csv', 'Sequence only'),
            ('../AUROC/predictions_and_true_labels_micro_perm.csv', 'Sequence + microarray signals')
        ],
        'genNullSeq': [
            ('../AUROC/predictions_and_true_labels_seq_gen.csv', 'Sequence only'),
            ('../AUROC/predictions_and_true_labels_acc_gen.csv', 'Sequence + ATAC'),
            ('../AUROC/predictions_and_true_labels_micro_gen.csv', 'Sequence + microarray signals'),
            ('../AUROC/predictions_and_true_labels_acc_mic_gen.csv', 'Sequence + microarray signals + ATAC')
        ]
    }
    plot_roc_curves(files_and_models)
    df = load_data('../interpation_file/merged_data.csv')
    prediction_columns = [
    'Prediction_perm_seq_0.25',
    'Prediction_rand_seq_0.25',
    'Prediction_gen_seq_0.25',
    'Prediction_perm_seq_0.33',
    'Prediction_rand_seq_0.33',
    'Prediction_gen_seq_0.33',
    'Prediction_perm_mic_0.25',
    'Prediction_rand_mic_0.25',
    'Prediction_gen_mic_0.25',
    'Prediction_perm_mic_0.33',
    'Prediction_rand_mic_0.33',
    'Prediction_gen_mic_0.33',
    'Signal'
]
    plot_prediction_distributions(df, prediction_columns)
    mutations_data = load_data('../interpation_file/mutations.csv')
    prediction_models = [
        'Prediction_perm_seq',
        'Prediction_rand_seq',
        'Prediction_gen_seq',
        'Signal',
        'Prediction_perm_mic',
        'Prediction_rand_mic',
        'Prediction_gen_mic'
    ]
    for model in prediction_models:
        plot_and_save_heatmap_for_model(model, mutations_data)
