import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Define your file names and models here
files_and_models = {
    'random': [
        ('predictions_and_true_labels_seq_random.csv', 'Sequence only'),
        ('predictions_and_true_labels_acc_random.csv', 'Sequence + ATAC'),
        ('predictions_and_true_labels_micro_rand.csv', 'Sequence + microarray signals'),
        ('predictions_and_true_labels_acc_mic_ran.csv', 'Sequence + microarray signals + ATAC')
    ],
    'permutation': [
        ('predictions_and_true_labels_seq_perm.csv', 'Sequence only'),
        ('predictions_and_true_labels_micro_perm.csv', 'Sequence + microarray signals')
    ],
    'genNullSeq': [
        ('predictions_and_true_labels_seq_gen.csv', 'Sequence only'),
        ('predictions_and_true_labels_acc_gen.csv', 'Sequence + ATAC'),
        ('predictions_and_true_labels_micro_gen.csv', 'Sequence + microarray signals'),
        ('predictions_and_true_labels_acc_mic_gen.csv', 'Sequence + microarray signals + ATAC')
    ]
}

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=len(files_and_models), figsize=(18, 6), sharey=True)
fig.suptitle('ROC Curves by Negative Group and Model')
for ax, (group, file_model_tuples) in zip(axes, files_and_models.items()):
    for file_name, model_label in file_model_tuples:
        # Read CSV file
        df = pd.read_csv(file_name)
        true_labels = df['True_Labels']
        predictions = df['Predictions']

        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
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
plt.show()