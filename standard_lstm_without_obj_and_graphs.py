import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, roc_curve, auc

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import save_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import wandb
import pickle

wandb.login(key="")
wandb.init(project="", name="")


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['formatted_message'] = df['formatted_message'].fillna('').astype(str)
    return df


def tokenize_and_pad(texts, max_length):
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequences, tokenizer


def create_model(vocab_size, max_length):
    model = Sequential([
        Embedding(vocab_size, 100, input_length=max_length),
        LSTM(64, return_sequences=True, recurrent_dropout=0.2),
        GlobalMaxPooling1D(),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model


class MetricsCallback(Callback):
    def __init__(self, validation_data):
        super(MetricsCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        X_val, y_val = self.validation_data
        y_pred_proba = self.model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int)

        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)

        print(
            f"Epoch {epoch + 1}: F1 Score = {f1:.4f}, ROC AUC = {roc_auc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")
        wandb.log({'epoch': epoch + 1, 'f1': f1, 'roc_auc': roc_auc, 'precision': precision, 'recall': recall,
                   'train_loss': logs['loss'], 'val_loss': logs['val_loss'],
                   'train_accuracy': logs['accuracy'], 'val_accuracy': logs['val_accuracy']})


def main():
    data = load_and_preprocess_data('')

    max_length = 512
    X, tokenizer = tokenize_and_pad(data['formatted_message'], max_length)
    y = data['radiant_win'].values

    indices = np.arange(len(data))

    X_train, X_temp, y_train, y_temp, train_indices, temp_indices = train_test_split(X, y, indices, test_size=0.2,
                                                                                     random_state=42)
    X_val, X_test, y_val, y_test, val_indices, test_indices = train_test_split(X_temp, y_temp, temp_indices,
                                                                               test_size=0.5, random_state=42)

    vocab_size = len(tokenizer.word_index) + 1
    model = create_model(vocab_size, max_length)

    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    metrics_callback = MetricsCallback(validation_data=(X_val, y_val))

    history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val),
                        callbacks=[early_stopping, reduce_lr, metrics_callback])

    model_save_path = ''
    save_model(model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save the trained model and tokenizer
    with open('', 'wb') as handle:
        pickle.dump(tokenizer, handle)


    with open('', 'wb') as file:
        pickle.dump(history.history, file)

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)


    outcomes_df = pd.DataFrame({
        'Match ID': data.loc[test_indices, 'match_id'],
        'Combined Messages': data.loc[test_indices, 'formatted_message'],
        'True Outcome (Radiant Win)': y_test,
        'Predicted Outcome': y_pred.flatten(),
        'Predicted Probability (Radiant Win)': y_pred_proba.flatten()
    })

    outcomes_df.to_csv('', index=False)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    loss, accuracy = model.evaluate(X_test, y_test)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Radiant Loss", "Radiant Win"],
                yticklabels=['Dire Win', 'Radiant Win'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix standard lstm ')
    confusion_matrix_path = ''
    plt.savefig(confusion_matrix_path)
    plt.show()

    wandb.log({"confusion_matrix_image": wandb.Image(confusion_matrix_path)})

    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig('')
    plt.show()

    y_pred_proba = model.predict(X_test).flatten()
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig('')
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig('')
    plt.show()

    y_pred = (y_pred_proba > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Radiant Loss", "Radiant Win"],
                yticklabels=["Radiant Loss", "Radiant Win"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig('')
    plt.show()



if __name__ == "__main__":
    main()

wandb.finish()
