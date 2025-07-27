from flask import Flask, request, render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

import numpy as np

app = Flask(__name__)

# Load model and tokenizer
model = load_model('faq_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 56  # Same as training

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['user_input']
    seq = tokenizer.texts_to_sequences([input_text])[0]

    num_words_to_predict = 6
    predicted_words = []
    confidences = []

    for _ in range(num_words_to_predict):
        pad_seq = pad_sequences([seq], maxlen=max_len-1, padding='pre')
        pred = model.predict(pad_seq, verbose=0)
        predicted_index = np.argmax(pred)
        confidence = pred[0][predicted_index] * 100  # Confidence %

        # Convert index back to word
        predicted_word = "<unknown>"
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                predicted_word = word
                break

        predicted_words.append(predicted_word)
        confidences.append(f"{confidence:.2f}%")
        seq.append(predicted_index)

    # Build the complete predicted sentence
    full_sentence = input_text + " " + " ".join(predicted_words)
    
    # Build the confidence list
    prediction_details = "<ul>"
    for word, conf in zip(predicted_words, confidences):
        prediction_details += f"<li><b>{word}</b> - {conf}</li>"
    prediction_details += "</ul>"

    return f"""
    <div class="prediction-text">
        <p><strong>Complete Prediction:</strong> {full_sentence}</p>
    </div>
    <div class="confidence-scores">
        <p><strong>Confidence Scores:</strong></p>
        {prediction_details}
    </div>
    """

if __name__ == "__main__":
    app.run(debug=True)
