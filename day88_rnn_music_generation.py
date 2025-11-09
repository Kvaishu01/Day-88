import numpy as np
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical
import random
import matplotlib.pyplot as plt

st.set_page_config(page_title="🎵 RNN Music Generator", layout="centered")
st.title("🎵 Day 88 — RNN Music Generation (LSTM)")

st.markdown("""
This demo trains an **LSTM-based RNN** to predict the next note in a melody sequence.
Once trained, it can generate a new melody step-by-step! 🎶
""")

# -----------------------------
# Synthetic Dataset Generator
# -----------------------------
def generate_scale_sequences(scale, n_sequences=500, seq_length=20):
    """
    Generate simple repeating note sequences using a given musical scale.
    Returns X (input sequences) and y (next note labels).
    """
    X, y = [], []
    for _ in range(n_sequences):
        start = random.randint(0, len(scale) - 1)
        seq = [(start + i) % len(scale) for i in range(seq_length + 1)]
        X.append(seq[:-1])
        y.append(seq[-1])
    return np.array(X), np.array(y)

# Define a simple "C major" scale (MIDI note numbers)
scale_notes = list(range(60, 68))  # 60=C, 61=D, ..., 67=C (next octave)
vocab_size = len(scale_notes)

# Sidebar controls
st.sidebar.header("🎚️ Settings")
n_sequences = st.sidebar.slider("Number of sequences", 100, 1000, 400, 50)
seq_length = st.sidebar.slider("Sequence length", 10, 40, 20)
embedding_dim = st.sidebar.slider("Embedding dimension", 8, 64, 32)
lstm_units = st.sidebar.slider("LSTM units", 32, 256, 128)
epochs = st.sidebar.slider("Epochs", 10, 100, 30)
batch_size = st.sidebar.selectbox("Batch size", [16, 32, 64], index=1)

# -----------------------------
# Train Model
# -----------------------------
if st.button("🎼 Generate and Train Model"):
    st.write("🎵 Preparing dataset...")
    X, y = generate_scale_sequences(scale_notes, n_sequences, seq_length)

    # ✅ FIX: Ensure all target notes exist in scale_notes
    y = np.array([note % vocab_size for note in y])  # map into range 0–7
    X_idx = np.array([[n % vocab_size for n in seq] for seq in X])

    # One-hot encode targets
    y_cat = to_categorical(y, num_classes=vocab_size)

    # Build model
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length),
        LSTM(lstm_units),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    st.write("🎹 Training model...")
    history = model.fit(X_idx, y_cat, epochs=epochs, batch_size=batch_size, verbose=0)
    st.success("✅ Training complete!")

    # Plot training accuracy
    fig, ax = plt.subplots()
    ax.plot(history.history["accuracy"], label="Training Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

    st.session_state["model"] = model
    st.session_state["vocab_size"] = vocab_size
    st.session_state["seq_length"] = seq_length

# -----------------------------
# Melody Generation
# -----------------------------
if "model" in st.session_state:
    st.subheader("🎶 Generate New Melody")
    model = st.session_state["model"]
    vocab_size = st.session_state["vocab_size"]
    seq_length = st.session_state["seq_length"]

    # Random starting seed
    seed = [random.randint(0, vocab_size - 1) for _ in range(seq_length)]
    st.write(f"🎵 Starting seed: {seed}")

    n_generate = st.slider("Number of new notes to generate", 10, 100, 40)
    generated = []
    current_seq = seed.copy()

    for _ in range(n_generate):
        input_seq = np.array(current_seq[-seq_length:]).reshape(1, seq_length)
        pred = model.predict(input_seq, verbose=0)
        next_note = np.argmax(pred)
        generated.append(next_note)
        current_seq.append(next_note)

    full_melody = seed + generated
    st.write("🎼 Generated melody (MIDI-like indices):")
    st.code(full_melody)

    # Plot melody pattern
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(full_melody, marker='o')
    ax2.set_title("Generated Melody (Note Sequence)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("MIDI Note Index")
    st.pyplot(fig2)

    st.success("🎵 Melody generation complete!")
else:
    st.info("🎹 Train a model first to generate new melodies.")
