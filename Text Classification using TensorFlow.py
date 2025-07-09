import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import numpy as np
import os
import re # Make sure 're' is imported for regex operations

# --- 1. Configuration and Hyperparameters ---
# You can adjust these for your specific dataset and performance needs
VOCAB_SIZE = 10000  # Max number of unique words in the vocabulary
MAX_SEQUENCE_LENGTH = 250 # Max length of a text sequence (padding/truncating)
EMBEDDING_DIM = 128 # Dimension of the word embeddings
LSTM_UNITS = 64     # Units in the LSTM layer
BATCH_SIZE = 32     # Number of samples per gradient update
EPOCHS = 10         # Number of training iterations

# --- 2. Data Loading and Preprocessing ---
# For demonstration, we'll use a simple dummy dataset.
# In a real application, you'd load your dataset from files (CSV, text, etc.)
# and split it into training and validation sets.

# Dummy data for demonstration
raw_texts = [
    "This movie was fantastic! I loved every minute of it.",
    "The acting was terrible and the plot made no sense.",
    "An absolutely brilliant film, highly recommended.",
    "I've seen better. Very boring and predictable.",
    "Great movie, highly entertaining.",
    "Worst film ever, complete waste of time.",
    "A masterpiece of storytelling.",
    "Disappointing and uninspired.",
    "Enjoyed it thoroughly, a real gem.",
    "Couldn't stand it, so bad.",
    "The best movie of the year, a must-watch!",
    "This was truly awful.",
    "Such a compelling story.",
    "Dull and unengaging.",
    "Fantastic plot and great characters.",
    "Absolutely terrible movie.",
]

# Dummy labels (0 for negative, 1 for positive)
raw_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# Convert to TensorFlow Datasets for efficiency
dataset = tf.data.Dataset.from_tensor_slices((raw_texts, raw_labels))

# Shuffle and split into training and validation sets
# For a real dataset, you'd typically have separate train/test files or a larger split.
BUFFER_SIZE = len(raw_texts) # For shuffling
train_size = int(0.8 * len(raw_texts)) # 80% for training

# Use take() and skip() to create train and validation splits
# Shuffle the entire dataset first for better randomness in splits
shuffled_dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False) # Ensure consistent split

train_ds = shuffled_dataset.take(train_size)
val_ds = shuffled_dataset.skip(train_size)


# Further configure for performance: batch and prefetch
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# --- 3. Text Vectorization Layer ---
# This layer handles standardization, tokenization, and integer mapping.
# It's crucial for efficiency as it runs within the TensorFlow graph.

# Custom standardization function (fixed syntax error here)
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    # Remove HTML tags (if applicable, common in review datasets)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    
    # Define common punctuation characters.
    # The previous error was due to an unescaped double quote inside the string literal.
    # This approach is robust and correctly handles the characters.
    punctuation_chars = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    
    # Replace any character found in punctuation_chars with an empty string.
    # re.escape() ensures that special regex characters in punctuation_chars are treated literally.
    cleaned_punctuation = tf.strings.regex_replace(stripped_html, f'[{re.escape(punctuation_chars)}]', '')
    
    return cleaned_punctuation

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=VOCAB_SIZE,
    output_mode='int',          # Output integer indices
    output_sequence_length=MAX_SEQUENCE_LENGTH # Pad/truncate sequences
)

# Adapt the TextVectorization layer to the training data.
# This builds the vocabulary based on the training texts.
train_text_only_ds = train_ds.map(lambda text, label: text)
vectorize_layer.adapt(train_text_only_ds)

# Helper function to apply the vectorization layer to datasets
# Note: TextVectorization will be used as the first layer in the model,
# so we don't *pre-vectorize* the full dataset with this map function
# unless we wanted to save pre-vectorized data.
# For direct model integration, we only need to map the original text/label
# format to the model's input.
# The `model` will handle the `vectorize_layer` itself.
# So, for the data passed to `model.fit`, it should be the raw text.

# The lines below are actually NOT needed if TextVectorization is the first layer of the model.
# The model itself will call vectorize_layer on the raw text inputs from train_ds and val_ds.
# Keeping it for clarity, but the model's input expects raw text.
# If you run these lines, you'd effectively be vectorizing twice, or
# expecting the model's first layer to *not* be TextVectorization.
# Let's remove them to make the `model` definition simpler and clearer as intended.

# The `model` will take the raw text and `vectorize_layer` will process it.
# So we just need to ensure our `train_ds` and `val_ds` are set up correctly
# with raw text for the model's input. They already are.

# --- 4. Model Definition (Bidirectional LSTM) ---
# Bidirectional LSTMs/GRUs are highly effective for text classification
# as they process sequences in both forward and backward directions,
# capturing more context.

model = tf.keras.Sequential([
    # The TextVectorization layer directly processes raw string inputs from the dataset.
    vectorize_layer,
    # Embedding layer: Converts integer indices into dense vectors
    layers.Embedding(
        input_dim=VOCAB_SIZE + 1, # vocabulary size + 1 for padding token (0)
        output_dim=EMBEDDING_DIM,
        mask_zero=True # Crucial for variable sequence lengths with padding
    ),
    # Bidirectional LSTM: Processes sequence in both directions
    layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=False)),
    layers.Dropout(0.3), # Dropout for regularization
    layers.Dense(1, activation='sigmoid') # Output layer for binary classification
])

# --- 5. Compile the Model ---
model.compile(
    optimizer='adam', # Adam is a good default optimizer
    loss='binary_crossentropy', # Appropriate for binary classification
    metrics=['accuracy']
)

model.summary()

# --- 6. Train the Model ---
print("\n--- Training the model ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# --- 7. Evaluate the Model ---
print("\n--- Evaluating the model ---")
loss, accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# --- 8. Make Predictions ---
print("\n--- Making predictions on new text ---")
def predict_sentiment(text_list):
    # The model expects a batch of inputs, so pass the list directly.
    # The TextVectorization layer within the model will handle processing these strings.
    predictions = model.predict(tf.constant(text_list))
    return ["Positive" if p > 0.5 else "Negative" for p in predictions]

sample_texts = [
    "This film was truly amazing, I loved it!",
    "What a waste of time, absolutely terrible.",
    "It was okay, not great, not bad.",
    "A perfect movie for a relaxing evening."
]

predicted_sentiments = predict_sentiment(sample_texts)

for text, sentiment in zip(sample_texts, predicted_sentiments):
    print(f"Text: '{text}' -> Sentiment: {sentiment}")

# --- Optional: Save and Load the Model ---
# This saves the entire model, including the TextVectorization layer.
model_save_path = './text_classification_model'
print(f"\nSaving model to: {model_save_path}")
model.save(model_save_path)

print(f"Loading model from: {model_save_path}")
# When loading a model with a custom standardization function in TextVectorization,
# you need to pass it in `custom_objects`.
loaded_model = tf.keras.models.load_model(
    model_save_path,
    custom_objects={'custom_standardization': custom_standardization}
)

# Test the loaded model
loaded_predictions = loaded_model.predict(tf.constant(sample_texts))
loaded_sentiments = ["Positive" if p > 0.5 else "Negative" for p in loaded_predictions]
print("\nPredictions from loaded model:")
for text, sentiment in zip(sample_texts, loaded_sentiments):
    print(f"Text: '{text}' -> Sentiment: {sentiment}")
