import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, precision_score, recall_score, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import seaborn as sns
import matplotlib.pyplot as plt
import time
import psutil
import wandb
import pickle

wandb.login(key="<weights and biasis api key>")
wandb.init(project="<project name>", name="<run name>")

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    df['formatted_message'] = df['formatted_message'].fillna('').astype(str)

    return df

data = load_and_preprocess_data('<path to data file>')
print("data loaded successfully!")

X_train, X_temp, y_train, y_temp = train_test_split(data['formatted_message'], data['radiant_win'], test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train_data = Dataset.from_dict({'text': X_train, 'label': y_train.astype(int)})
val_data = Dataset.from_dict({'text': X_val, 'label': y_val.astype(int)})
test_data = Dataset.from_dict({'text': X_test, 'label': y_test.astype(int)})
dataset = DatasetDict({'train': train_data, 'validation': val_data, 'test': test_data})

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
print("tokenizer and model loaded")


def tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

token_lengths = [len(tokenizer.tokenize(text)) for text in dataset['train']['text']]
avg_token_length = sum(token_lengths) / len(token_lengths)
wandb.log({'avg_token_length': avg_token_length})

training_args = TrainingArguments(
    output_dir='<path to output directory>',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    weight_decay=0.01,
    logging_dir='<path to output directory>',
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',
    report_to="wandb"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()

    predictions = np.argmax(probabilities, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    roc_auc = roc_auc_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)

    wandb.log({
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall
    })

    report = classification_report(labels, predictions, output_dict=True)
    for class_label, metrics in report.items():
        if isinstance(metrics, dict):
            wandb.log({
                f'class_{class_label}_precision': metrics['precision'],
                f'class_{class_label}_recall': metrics['recall'],
                f'class_{class_label}_f1': metrics['f1-score']
            })

    conf_matrix = confusion_matrix(labels, predictions)
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                               y_true=labels,
                                                               preds=predictions,
                                                               class_names=["Radiant Loss", "Radiant Win"])})

    return {'accuracy': accuracy, 'f1': f1, 'roc_auc': roc_auc, 'precision': precision, 'recall': recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics
)

memory_info = psutil.virtual_memory()
wandb.log({'memory_usage': memory_info.percent})

start_time = time.time()

trainer.train()

epoch_duration = time.time() - start_time
wandb.log({'epoch_duration': epoch_duration})

print("model trained")

model.save_pretrained('<path to model save>')
tokenizer.save_pretrained('<path to tokenizer save>')

with open('<path to pickle>', 'wb') as history_file:
    pickle.dump(trainer.state.log_history, history_file)


with open('<path to pickle>', 'wb') as model_file:
    pickle.dump(model, model_file)
eval_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
print("\nEvaluation results:", eval_results)

wandb.log(eval_results)

predictions = trainer.predict(tokenized_datasets['test'])

softmax_predictions = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()

wandb.log({
    'predicted_prob_radiant_win': softmax_predictions[:, 1].mean(),
    'predicted_prob_radiant_loss': softmax_predictions[:, 0].mean()
})

outcomes_df = pd.DataFrame({
    'Match ID': data.loc[X_test.index, 'match_id'],
    'Combined Messages': X_test,
    'True Outcome (Radiant Win)': y_test,
    'Predicted Outcome': softmax_predictions.argmax(axis=1),
    'Predicted Probability (Radiant Win)': softmax_predictions[:, 1]
})

outcomes_df.to_csv('<path to outcomes_df>', index=False)

y_pred = predictions.predictions.argmax(axis=1)
y_pred_proba = softmax_predictions[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print(f"Test ROC-AUC: {roc_auc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")

wandb.log({
    'test_accuracy': accuracy,
    'test_f1': f1,
    'test_roc_auc': roc_auc,
    'test_precision': precision,
    'test_recall': recall
})

conf_matrix = confusion_matrix(y_test, predictions.predictions.argmax(axis=1))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('DistilBERT with Objectives 80 10 10 Confusion Matrix')
plt.xlabel('Predicted Outcome')
plt.ylabel('True Outcome')

plt.savefig('<path to figure>')
print("Confusion matrix saved")

print("\nClassification Report:")
print(classification_report(y_test, predictions.predictions.argmax(axis=1)))



y_pred = predictions.predictions.argmax(axis=1)
y_pred_proba = softmax_predictions[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig('<path to figure>')  # Save the ROC curve
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig('<path to figure>')
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Loss", "Win"], yticklabels=["Loss", "Win"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig('<path to figure>')
plt.show()

residuals = y_test - y_pred_proba
plt.figure()
plt.scatter(y_pred_proba, residuals, alpha=0.5)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Predicted Probability (Radiant Win)")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig('<path to figure>')
plt.show()

from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker='o', label="DistilBERT")
plt.plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated")
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curve")
plt.legend()
plt.savefig('<path to figure>')
plt.show()

train_losses, val_losses, epochs = [], [], []

for log in trainer.state.log_history:
    if 'loss' in log:
        train_losses.append(log['loss'])
        epochs.append(log['epoch'])
    if 'eval_loss' in log:
        val_losses.append(log['eval_loss'])

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs[:len(val_losses)], val_losses, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.savefig('<path to figure>')
plt.show()

wandb.finish()
