class Response:
    response = ''
    prediction = 0
    train_data = []
    encoded_labels = []
    decoded_labels = []

    def __init__(self, response, encoded_labels, decoded_labels, prediction, train_data):
        self.response = response
        self.train_data = train_data
        self.encoded_labels = encoded_labels
        self.decoded_labels = decoded_labels
        self.prediction = prediction
        
