from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and preprocessing files
MODEL_PATH = r"C:\Users\KEERTHANA\RNN_LSTM_model\emotion_lstm_model.h5"
TOKENIZER_PATH = r"C:\Users\KEERTHANA\RNN_LSTM_model\tokenizer.pkl"
LABEL_ENCODER_PATH = r"C:\Users\KEERTHANA\RNN_LSTM_model\label_encoder.pkl"

model = load_model(MODEL_PATH)
tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))
label_encoder = pickle.load(open(LABEL_ENCODER_PATH, "rb"))

# Emoji mapping
emoji_map = {
    "joy": "😂🎉✨",
    "anger": "😡🔥💢",
    "fear": "😨👻⚡",
    "sadness": "😢💔🌧️",
    "love": "❤️😍💖",
    "surprise": "😲🎊🌟"
}

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)  # adjust maxlen to your training
    pred = model.predict(padded)
    emotion = label_encoder.inverse_transform([np.argmax(pred)])[0]
    emoji = emoji_map.get(emotion, "🤔")

    return render_template("index.html", prediction=emotion.upper(), emoji=emoji)

if __name__ == "__main__":
    app.run(debug=True)
