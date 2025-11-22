from src.data.loader import DataLoader
from src.model.train import ModelTrainer
from src.model.evaluation import ModelEvaluator

features = ['amt','zip','city_pop','unix_time','year','month','day'
            ,'hour','dayofweek','merchant_encoded','category_encoded',
            'gender_encoded','job_encoded','state_encoded','age','distance_km']
target = 'is_fraud'

data_path = "data/processed/data_resampled_v1.csv"

def main():
    try:
        # Load data
        data_loader = DataLoader(file_path=data_path)
        data = data_loader.load_data()

        # Train model
        print("Starting model training...")
        trainer = ModelTrainer(data=data, target_column=target, features_list=features, test_size=0.3)
        _, accuracy, X_test,y_test = trainer.train_model()
        print(f"Model trained with accuracy: {accuracy:.2f}")
        trainer.save_model(trainer.model_path)

        # Evaluate model
        print("Starting model evaluation...")
        evaluator = ModelEvaluator(model_path=trainer.model_path, X_test=X_test, y_test=y_test)
        evaluator.load_model()
        print("Model loaded for evaluation.")
        report, cm = evaluator.evaluate()
        print("Classification Report:\n", report)
        class_names = ['Not Fraud', 'Fraud']
        evaluator.plot_confusion_matrix(cm, class_names)
    except Exception as e:
        print(f"An error occurred in the training pipeline: {e}")

if __name__ == "__main__":
    main()