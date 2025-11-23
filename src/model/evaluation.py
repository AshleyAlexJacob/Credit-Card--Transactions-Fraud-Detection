import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self,  X_test, y_test):
        try:
            self.X_test = X_test
            self.y_test = y_test
            self.model_directory = "artifacts/models"
        except Exception as e:
            raise RuntimeError(f"An error occurred while initializing ModelEvaluator: {e}")

    def load_model(self, model_name):
        try:
            with open(os.path.join(self.model_directory, model_name), 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the model: {e}")
        
    def evaluate(self):
        try:
            y_pred = self.model.predict(self.X_test)
            report = classification_report(self.y_test, y_pred)
            cm = confusion_matrix(self.y_test, y_pred)
            return report, cm
        except Exception as e:
            raise RuntimeError(f"An error occurred during evaluation: {e}")

    def plot_confusion_matrix(self, cm, class_names):
        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.show()
        except Exception as e:
            raise RuntimeError(f"An error occurred while plotting the confusion matrix: {e}")
