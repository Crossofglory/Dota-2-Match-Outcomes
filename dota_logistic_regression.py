import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, precision_recall_curve, roc_curve, auc)
from gensim.models import Word2Vec
import wandb
import gensim.downloader as api
import wandb.sklearn
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

wandb.login(key="<wandb api key>")
wandb.init(project="<project name>", name="<run name>")

graph_dir = ''
os.makedirs(graph_dir, exist_ok=True)

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['formatted_message'] = df['formatted_message'].fillna('').astype(str)
    return df

def fine_tune_word2vec(data, pretrained_model):
    sentences = [text.split() for text in data['formatted_message']]
    fine_tuned_model = Word2Vec(vector_size=300, window=5, min_count=1, sg=1, workers=4)
    fine_tuned_model.build_vocab(sentences)
    fine_tuned_model.build_vocab([list(pretrained_model.key_to_index.keys())], update=True)
    fine_tuned_model.wv.vectors_lockf = np.ones(len(fine_tuned_model.wv))
    fine_tuned_model.wv.vectors = pretrained_model.vectors
    fine_tuned_model.train(sentences, total_examples=len(sentences), epochs=30)
    print("Word2Vec model fine-tuned successfully!")
    return fine_tuned_model

def sentence_to_embedding(sentence, model):
    words = sentence.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def prepare_dataset(data, model):
    return np.array([sentence_to_embedding(text, model) for text in data['formatted_message']])

data = load_and_preprocess_data('')

pretrained_word2vec_model = api.load("word2vec-google-news-300")
fine_tuned_word2vec_model = fine_tune_word2vec(data, pretrained_word2vec_model)
X_embeddings = prepare_dataset(data, fine_tuned_word2vec_model)

indices = np.arange(len(data))
X_train, X_temp, y_train, y_temp, train_indices, temp_indices = train_test_split(X_embeddings, data['radiant_win'], indices, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test, val_indices, test_indices = train_test_split(X_temp, y_temp, temp_indices, test_size=0.5, random_state=42)

logreg_model = LogisticRegression(max_iter=250, C=0.01, solver='lbfgs', verbose=1)
logreg_model.fit(X_train, y_train)
print(f"Number of iterations to converge: {logreg_model.n_iter_[0]}")



model_save_path = ''
with open(model_save_path, "wb") as f:
    pickle.dump(logreg_model, f)


wandb.sklearn.plot_learning_curve(logreg_model, X_train, y_train)


y_val_pred = logreg_model.predict(X_val)
y_val_pred_proba = logreg_model.predict_proba(X_val)[:, 1]

y_test_pred = logreg_model.predict(X_test)
y_test_pred_proba = logreg_model.predict_proba(X_test)[:, 1]

y_val_pred_proba_2d = np.column_stack((1 - y_val_pred_proba, y_val_pred_proba))
y_test_pred_proba_2d = np.column_stack((1 - y_test_pred_proba, y_test_pred_proba))

wandb.sklearn.plot_roc(y_val, y_val_pred_proba_2d, labels=["Radiant Loss", "Radiant Win"])
wandb.sklearn.plot_precision_recall(y_val, y_val_pred_proba_2d)
wandb.sklearn.plot_confusion_matrix(y_val, y_val_pred, labels=["Radiant Loss", "Radiant Win"])

val_f1 = f1_score(y_val, y_val_pred)
val_roc_auc = roc_auc_score(y_val, y_val_pred_proba)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
wandb.log({
    'val_f1': val_f1,
    'val_roc_auc': val_roc_auc,
    'val_precision': val_precision,
    'val_recall': val_recall,
    'val_accuracy': logreg_model.score(X_val, y_val)
})

print(f"Validation F1 Score: {val_f1:.4f}")
print(f"Validation ROC AUC: {val_roc_auc:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")


test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test ROC AUC: {test_roc_auc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))


fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig(os.path.join(graph_dir, ''))

precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
plt.figure()
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig(os.path.join(graph_dir, ''))

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Radiant Loss", "Radiant Win"], yticklabels=["Radiant Loss", "Radiant Win"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(graph_dir, ''))

feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]
coefficients = logreg_model.coef_.flatten()
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefficients)
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Feature Importance in Logistic Regression")
plt.savefig(os.path.join(graph_dir, ''))

wandb.log({
    'test_f1': test_f1,
    'test_roc_auc': test_roc_auc,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_accuracy': test_accuracy
})

wandb.finish()
