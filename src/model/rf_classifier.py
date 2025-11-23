from sklearn.ensemble import RandomForestClassifier

class RFClassifier:
    def __init__(self, n_estimators=20):
        try:
            self.model = RandomForestClassifier(n_estimators=n_estimators)
        except ImportError:
            raise ImportError("scikit-learn is required to use Random Forest Classifier. Please install it via 'pip install scikit-learn'.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while initializing RandomForestClassifier: {e}")

    def train(self, X_train, y_train):
        try:
            self.model.fit(X_train, y_train)
            return self.model
        except Exception as e:
            raise RuntimeError(f"An error occurred during training: {e}")
            
    def predict(self, X_test):
        try:
            return self.model.predict(X_test)
        except Exception as e:
            raise RuntimeError(f"An error occurred during prediction: {e}")

    def evaluate(self, X_test, y_test):
        try:
            accuracy = self.model.score(X_test, y_test)
            return accuracy
        except Exception as e:
            raise RuntimeError(f"An error occurred during evaluation: {e}")