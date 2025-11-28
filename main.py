from npl.sentiment_model_trainer import SentimentModelTrainer
from app import start_flask_app
from dotenv import load_dotenv
import os

load_dotenv()

EXECUTE_NPL = os.getenv("EXECUTE_NPL", "False").lower() == "true"

if __name__ == "__main__":

    npl_execution_success = False

    if EXECUTE_NPL:
        print("🤖 EXECUTE_NPL is set to True. Generating the sentiment analysis model...")
        trainer = SentimentModelTrainer(
            dataset_path="./npl/dataset",
            model_dir="./npl/model"
        )
        npl_execution_success = trainer.generate_model()
    else:
        print("🤖 EXECUTE_NPL is set to False. Skipping model generation...")
        print("🚀 Starting Flask app without generating the model...")
        start_flask_app()

    if npl_execution_success:
        print("🚀 Model successfully generated. Starting Flask app...")
        start_flask_app()
    else:
        print("❌ Model generation failed or was skipped. Flask app will not start.")