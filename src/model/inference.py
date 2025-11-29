import os
import pickle


class ModelInference:
    def __init__(self, model_name:str):
        try:
            self.model_directory = "artifacts/models"
            self.encoder_directory = "artifacts/encoders"
            with open(os.path.join(self.model_directory, model_name), 'rb') as f:
                self.model = pickle.load(f)
            self.count_encoder=self.__load_encoder("count_encoder.pkl")
            self.target_encoder=self.__load_encoder("target_encoder.pkl")
            
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the model: {e}")
    
    def __load_encoder(self, encoder_name:str):
        try:
            with open(os.path.join(self.encoder_directory, encoder_name), 'rb') as f:
                encoder = pickle.load(f)
            return encoder
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the encoder: {e}")
    
    def predict(self, input_data):
        try:
            prediction = self.model.predict(input_data)
            return prediction
        except Exception as e:
            raise RuntimeError(f"An error occurred during prediction: {e}")