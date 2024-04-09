import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from response import Response

class Trainer:
    train_data = [
        "Hola",
        "Que día es hoy?",
        "Te gusta el futbol",
        "De que equipo eres?",
        "Tienes algún pasatiempo?",
        "Que edad tienes?",
        "Tienes perros?",
        "Tu edad?",
        "Vamos por un café?",
        "cuál es tu equipo?"
    ]

    train_labels = [
        "Hola, en qué te puedo ayudar?",
        "Hoy es viernes",
        "Si, claro",
        "Soy del rey de copas",
        "Si, me gusta muchos los videojuegos",
        "Mi edad es de 41 años",
        "Si tengo 2 perros",
        "41 años",
        "No me gusta el café, lo siento",
        "Soy del Barcelona"
    ]

    train_sequences=[]

    label_encoder = LabelEncoder()
    tokenizer = keras.preprocessing.text.Tokenizer()
    model = keras.models.Sequential()

    def encoded_labels_fn(self):
        encoded_labels = []
        encoded_labels = self.label_encoder.fit_transform(self.train_labels) 
        
        return pd.Series(encoded_labels).to_json(orient='table')
    
    def tokenized_data(self):
        self.tokenizer.fit_on_texts(self.train_data)
        tokenized_data_index = self.tokenizer.word_index
        
        return pd.Series(tokenized_data_index).to_json(orient='table')
    
    def text_to_sequences(self):    
        self.tokenizer.fit_on_texts(self.train_data)
        sequenced_text = self.tokenizer.texts_to_sequences(self.train_data)
        keras.preprocessing.sequence.pad_sequences(self.train_sequences)

        return pd.Series(sequenced_text).to_json(orient='table')
    
    def train(self):
        encoded_labels = []
        encoded_labels = self.label_encoder.fit_transform(self.train_labels)    
        
        self.tokenizer.fit_on_texts(self.train_data)
        
        self.train_sequences = self.tokenizer.texts_to_sequences(self.train_data)
        self.train_sequences = keras.preprocessing.sequence.pad_sequences(self.train_sequences)

        self.model.add(keras.layers.Embedding(len(self.tokenizer.word_index) + 1, 200, input_length=self.train_sequences.shape[1]))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(64, activation='relu'))
        self.model.add(keras.layers.Dense(len(self.train_labels), activation='softmax'))
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.train_sequences, encoded_labels, epochs=10)
        self.model.summary()
    
    def generate_response(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        sequence = keras.preprocessing.sequence.pad_sequences(sequence, maxlen=self.train_sequences.shape[1])
        prediction = self.model.predict(sequence)
        predicted_label = np.argmax(prediction)
        response = self.label_encoder.inverse_transform([predicted_label])

        formated_response = Response(response[0], prediction, self.label_encoder.classes_, predicted_label, self.train_data)

        return pd.Series(formated_response).to_json(orient='table')
