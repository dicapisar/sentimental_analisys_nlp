# Sentiment Analysis NLP Project
ISY503 – Intelligent Systems

## 1. Introduction

This repository contains a Natural Language Processing (NLP) project that performs binary sentiment analysis on Amazon-style product reviews. The work has been developed for the ISY503 Final Project assessment at Torrens University Australia.

The system trains a neural network model to classify reviews as either positive or negative and deploys the model through a simple Flask web application. Users can enter a review into a text box and obtain a sentiment prediction together with a confidence score.

## 2. Group Members

This project has been completed by the following group of three students:

- Diego Pinto  
- Mayerli Almario  
- Joao Victor De Souza  

## 3. Project Description

The main objective of the project is to demonstrate an end-to-end intelligent system that:

- Cleans and prepares a real-world text dataset  
- Trains a supervised learning model for sentiment analysis  
- Evaluates the model using separate validation and test sets  
- Exposes the model via a lightweight, user-friendly web interface  

The project follows the requirements for the NLP option in the ISY503 Assessment 3 brief, including data preprocessing, model training, and deployment as a website with a text input field and sentiment output.

## 4. How the System Works

### 4.1 Data Processing

The dataset consists of labelled Amazon product reviews divided into positive and negative classes. The following preprocessing steps are implemented in `SentimentModelTrainer`:

1. Parsing of `.review` files to extract the `<review_text>` content.  
2. Cleaning of text:
   - Conversion to lower case  
   - Removal of HTML tags  
   - Removal of URLs  
   - Removal of non-alphanumeric characters  
   - Normalisation of whitespace  
3. Outlier removal:
   - Reviews with fewer than a minimum number of words are discarded.  
4. Tokenisation and padding:
   - A Keras `Tokenizer` converts words to integer indices.  
   - Sequences are padded or truncated to a fixed maximum length (300 tokens).  
5. Dataset splitting:
   - Training, validation, and test sets are generated using `train_test_split`.  

These steps improve data quality and ensure that the model receives consistent numerical input.

### 4.2 Model Architecture

The sentiment classifier is implemented with TensorFlow/Keras in `sentiment_model_trainer.py`. The architecture is:

- Embedding layer for word vector representation (vocabulary size up to `MAX_WORDS`)  
- Bidirectional LSTM layer (64 units) with `return_sequences=True`  
- Dropout layer for regularisation  
- 1D Convolutional layer (64 filters, kernel size 3, ReLU activation)  
- Global Max Pooling layer  
- Dense layer (64 units, ReLU activation)  
- Dropout layer  
- Output Dense layer (1 unit, sigmoid activation)  

The model is compiled with:

- Loss: binary cross-entropy  
- Optimiser: Adam  
- Metric: accuracy  

An `EarlyStopping` callback monitors the validation loss to reduce overfitting by restoring the best weights observed during training.

### 4.3 Training and Model Export

The full training pipeline is executed via:

```bash
python main.py
```

`main.py`:

1. Instantiates `SentimentModelTrainer` with dataset and model directories.  
2. Executes `generate_model()`, which:
   - Loads and preprocesses all reviews  
   - Trains the neural network  
   - Evaluates the model on the test set  
   - Saves the trained model to `./npl/model/sentiment_model.keras`  
   - Saves the fitted tokenizer to `./npl/model/tokenizer.pkl`  
3. If training is successful, the Flask application is started automatically.

### 4.4 Web Application Behaviour

The web interface is defined in `app.py` using Flask:

- At application start, the saved Keras model and tokenizer are loaded once into memory.  
- The root route `/` accepts both `GET` and `POST` requests.  
- For a `POST` request:
  - The review text is read from the form.  
  - The text is tokenised and padded using the same configuration as during training.  
  - The model predicts a probability score between 0 and 1.  
  - If the score is ≥ 0.5, the review is classified as “Positive”; otherwise it is “Negative”.  
- The HTML template renders:
  - The predicted sentiment label  
  - The numeric probability score  
  - A coloured result panel  
  - A simple progress bar to visualise confidence  
  - A table explaining how to interpret different score ranges  

This design allows the facilitator to test sample inputs directly through a browser.

## 5. Project Structure

The repository is organised as follows:

```text
project_root/
│
├── app.py                          # Flask web application and HTML template
├── main.py                         # Orchestrates model training and app startup
├── requirements.txt                # Python dependency list
│
├── npl/
│   ├── dataset/                    # Amazon review data (.review files by category)
│   ├── model/                      # Trained model (.keras) and tokenizer (.pkl)
│   └── sentiment_model_trainer.py  # Training pipeline and model definition
│
└── README.md                       # Project documentation
```

## 6. Prerequisites

- Python 3.x (tested with Python 3.10–3.12)  
- pip or another package manager for Python  
- Sufficient memory to train the neural network using TensorFlow  

## 7. Installation

1. Clone or download this repository into a local directory.  
2. (Optional but recommended) Create and activate a virtual environment.  
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## 8. Execution Steps

### 8.1 Train the Model and Launch the Web App

From the project root directory, run:

```bash
python main.py
```

If training completes successfully, the script prints a success message and starts the Flask server.

### 8.2 Access the Application

Open a web browser and navigate to:

```text
http://0.0.0.0:5001
```

or, on a local machine:

```text
http://localhost:5001
```

You can now type or paste a product review into the text area and click the button to obtain the sentiment analysis result.

## 9. Usage Notes

- The interface is designed for short to medium-length product reviews written in English.  
- Scores closer to 0 indicate strong negative sentiment; scores closer to 1 indicate strong positive sentiment.  
- Reviews that are extremely short, unclear, or unrelated to products may produce less reliable predictions.

## 10. Ethical Considerations

The project recognises several ethical considerations in sentiment analysis:

- The original labels in the dataset may contain subjective or biased judgements.  
- Certain linguistic styles or demographic groups may be under-represented, which can affect model fairness.  
- Model predictions should not be used for high-stakes decision making without additional validation and monitoring.  

To address some of these issues, the implementation:

- Removes incomplete or very short reviews that are unlikely to reflect meaningful sentiment.  
- Applies the same preprocessing pipeline to all inputs for consistent treatment.  
- Presents outputs as probabilistic scores instead of absolute truth, encouraging critical interpretation.  

The system is intended solely for educational use within the scope of the ISY503 unit.

## 11. Technologies Used

- Python  
- TensorFlow and Keras  
- scikit-learn  
- NumPy  
- Flask  

## 12. Conclusion

This project demonstrates an end-to-end intelligent system that integrates data preprocessing, neural network training, model evaluation, and web deployment. By implementing a BiLSTM–CNN architecture and providing a functional user interface, the repository illustrates how modern AI techniques can be applied to sentiment analysis tasks in a transparent and reproducible manner.
