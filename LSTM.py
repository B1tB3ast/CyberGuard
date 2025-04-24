import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split

# ----------------- GPU Setup -----------------
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU detected and configured.")
    except RuntimeError as e:
        print("Error setting GPU memory growth:", e)
else:
    print("No GPU found. Using CPU.")

# ----------------- Load Data -----------------
df = pd.read_csv("merged_cleaned_dataset_balanced.csv")  # Replace with your actual CSV file path

# ----------------- Settings -----------------
embedding_dim = 200
num_classes = 1  # Binary Classification
max_len = 100

# ----------------- Data Preparation -----------------
texts = df['cleaned_text'].astype(str).tolist()
labels = df['label'].tolist()

X_train_texts, X_temp_texts, y_train, y_temp = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)
X_valid_texts, X_test_texts, y_valid, y_test = train_test_split(X_temp_texts, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_texts)

X_train_seq = tokenizer.texts_to_sequences(X_train_texts)
X_valid_seq = tokenizer.texts_to_sequences(X_valid_texts)
X_test_seq  = tokenizer.texts_to_sequences(X_test_texts)

X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_valid_padded = pad_sequences(X_valid_seq, maxlen=max_len, padding='post')
X_test_padded  = pad_sequences(X_test_seq,  maxlen=max_len, padding='post')

# ----------------- Callbacks -----------------
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

checkpoint = ModelCheckpoint(
    filepath='best_lstm_model.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

# ----------------- Build Model -----------------
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_len),
        LSTM(units=100, dropout=0.2, recurrent_dropout=0.2),
        Dense(units=1, activation='sigmoid')
    ])

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # ----------------- Train -----------------
    history = model.fit(
        X_train_padded, np.array(y_train),
        epochs=10,
        batch_size=32,
        validation_data=(X_valid_padded, np.array(y_valid)),
        callbacks=[early_stopping, checkpoint]
    )

# ----------------- Evaluate -----------------
def plotting_funct(history_df, title):
    plt.plot(history_df['loss'], label='Train loss')
    plt.plot(history_df['val_loss'], label='Validation loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss_curve.png')
    plt.show()

history_df = pd.DataFrame(history.history)
plotting_funct(history_df, 'Simple LSTM (Binary)')

# Load best saved model
model = load_model('best_lstm_model.h5')

y_pred_probs = model.predict(X_test_padded)
y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

print('Classification Report:\n', classification_report(y_test, y_pred, target_names=["Not Cyberbullying", "Cyberbullying"]))

def conf_matrix(y_true, y_pred, model_name, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confmat.png')
    plt.show()

conf_matrix(y_test, y_pred, 'Binary LSTM', ["Not Cyberbullying", "Cyberbullying"])
