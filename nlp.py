import json

from flask import Flask, request
from trainer import Trainer
from response import Response

app = Flask(__name__)

trainer = Trainer()
trainer.train()

@app.route('/chatBot', methods=['POST'])
def chat_bot():
    request_data = request.get_json()

    return trainer.generate_response(request_data['user_input'])

@app.route('/encodedLabels')
def encoded_labels():
    return trainer.encoded_labels_fn()

@app.route('/tokenizedData')
def tokenized_data():
    return trainer.tokenized_data()

@app.route('/textToSequences')
def text_to_sequences():
    return trainer.text_to_sequences()
