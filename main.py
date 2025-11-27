from npl.sentiment_model_trainer import SentimentModelTrainer
from app import start_flask_app

if __name__ == "__main__":
    trainer = SentimentModelTrainer(
        dataset_path="./npl/dataset",
        model_dir="./npl/model"
    )

    success = trainer.generate_model()

    if success:
        print("🚀 Model successfully generated. Starting Flask app...")
        start_flask_app()
    else:
        print("⛔ Flask app NOT started because model generation failed.")
