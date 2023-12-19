class AlgorithmInterface:
    def train(self, train_data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError
