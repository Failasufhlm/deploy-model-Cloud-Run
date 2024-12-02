import pickle
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

nltk.download('stopwords')

stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    # Menghapus URL
    text = re.sub(r'http\S+|www\S+', '', text)

    # Menghapus angka
    text = re.sub(r'\d+', '', text)

    # Menghapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Mengubah teks ke lowercase
    text = text.lower()

    # Menghapus stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Stemming
    text = stemmer.stem(text)

    return text


# Load model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    model = pickle.load(open('model/sentiment_jurnal.pkl', 'rb'))
    tokenizer = pickle.load(open('model/tokenizer_jurnal.pkl', 'rb'))
    return model, tokenizer


def predict_sentiment(text, model, tokenizer):
    # Preprocess teks
    text = preprocess_text(text)

    # Tokenisasi
    seq = tokenizer.texts_to_sequences([text])

    # Padding
    seq = pad_sequences(seq, maxlen=200, dtype='int32', value=0)

    # Prediksi sentimen
    sentiment = model.predict(seq, batch_size=1, verbose=2)[0]

    # Menentukan hasil prediksi
    # Mengambil probabilitas positif dan negatif
    negative_prob = sentiment[0] * 100  # Persentase untuk negatif
    positive_prob = sentiment[1] * 100  # Persentase untuk positif

    return negative_prob, positive_prob


def run():

    # Input teks
    user_input = st.text_area("input teks")

    if st.button("Prediksi Sentimen"):
        if user_input:
            # Load model dan tokenizer dari file .pkl
            model = pickle.load(open('model/sentiment_jurnal.pkl', 'rb'))
            tokenizer = pickle.load(open('model/tokenizer_jurnal.pkl', 'rb'))

            # Prediksi sentimen
            negative_prob, positive_prob = predict_sentiment(user_input, model, tokenizer)

            # Tampilkan hasil persentase prediksi
            st.write(f"Sentimen Negatif: {negative_prob:.2f}%")
            st.write(f"Sentimen Positif: {positive_prob:.2f}%")
        else:
            st.warning("Silakan masukkan teks terlebih dahulu")


if __name__ == "__main__":
    run()
