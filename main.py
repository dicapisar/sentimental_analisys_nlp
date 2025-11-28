"""
Entry point for the sentiment analysis application.

This script coordinates two main tasks:
    1. (Optional) Training and saving the sentiment analysis model, using
       `SentimentModelTrainer`.
    2. Starting the Flask web application for interactive sentiment analysis.

The behaviour is controlled by the environment variable `EXECUTE_NPL`:
    - If EXECUTE_NPL is set to "True" (case-insensitive), the script attempts
      to train and save the model before starting the web app.
    - If EXECUTE_NPL is "False" or unset, the script skips model training and
      runs the Flask app directly, assuming a pre-trained model already exists.
"""

from npl.sentiment_model_trainer import SentimentModelTrainer
from app import start_flask_app
from dotenv import load_dotenv
import os

# Load environment variables from a .env file if present.
load_dotenv()

# Flag that determines whether the NLP model training pipeline should be executed.
EXECUTE_NPL = os.getenv("EXECUTE_NPL", "False").lower() == "true"


if __name__ == "__main__":

    # Tracks whether the model training pipeline completed successfully.
    npl_execution_success = False

    if EXECUTE_NPL:
        print("🤖 EXECUTE_NPL is set to True. Generating the sentiment analysis model...")

        # Create the trainer with the dataset and model output paths.
        trainer = SentimentModelTrainer(
            dataset_path="./npl/dataset",
            model_dir="./npl/model"
        )

        # Run the full training pipeline (may take several minutes depending
        # on dataset size and hardware).
        npl_execution_success = trainer.generate_model()
    else:
        print("🤖 EXECUTE_NPL is set to False. Skipping model generation...")
        print("🚀 Starting Flask app without generating the model...")
        start_flask_app()

    if npl_execution_success:
        # Start the Flask app only if the model was successfully generated.
        print("🚀 Model successfully generated. Starting Flask app...")
        start_flask_app()
    else:
        # Inform the user that training either failed or did not run.
        print("❌ Model generation failed or was skipped. Flask app will not start.")