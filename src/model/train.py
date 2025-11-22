from sklearn.model_selection import train_test_split
from src.model.svm_classifier import SVMClassifier
from pandas import DataFrame
import pickle

class ModelTrainer:
    def __init__(self, data:DataFrame, target_column:str, features_list:list ,test_size:float=0.2, random_state:int=42):
        try:
            self.data = data
            self.target_column = target_column
            self.test_size = test_size
            self.random_state = random_state
            self.classifier = SVMClassifier()
            self.model_path = "artifacts/models/svm_model.pkl"
        except Exception as e:
            raise RuntimeError(f"An error occurred while initializing ModelTrainer: {e}")
    
    def separate_features_and_target(self):
        try:
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            return X, y
        except Exception as e:
            raise RuntimeError(f"An error occurred while separating features and target: {e}")
    
    def split_data(self, X, y):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise RuntimeError(f"An error occurred while splitting the data: {e}")
    
    def train_model(self):
        try:
            print("Training the SVM model...")
            X, y = self.separate_features_and_target()
            print("Splitting data into training and testing sets...")
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            print("Fitting the model...")
            self.model = self.classifier.train(X_train, y_train)
            print("Evaluating the model...")
            accuracy = self.classifier.evaluate(X_test, y_test)
            return self.model, accuracy, X_test, y_test
        except Exception as e:
            raise RuntimeError(f"An error occurred during model training: {e}")
    
    def save_model(self, file_path:str):
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)
        except Exception as e:
            raise RuntimeError(f"An error occurred while saving the model: {e}")