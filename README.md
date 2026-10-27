
# IMDB Sentiment Analysis Using Deep Learning

An interactive movie review sentiment analysis application built with Python, TensorFlow/Keras, and Streamlit.

## Project Overview

The application predicts whether a movie review expresses positive or negative sentiment.

## Models

- SimpleRNN
- LSTM
- GRU

## Features

- Interactive sentiment prediction
- Positive and negative sentiment classification
- Prediction confidence display
- Model performance visualization
- Pre-trained model inference without retraining

## Technologies

- Python
- TensorFlow / Keras
- Pandas and NumPy
- Streamlit
- Natural Language Processing

## Project Structure

- `app.py`: Streamlit application
- `models/`: Saved model and tokenizer files
- `data/`: Dataset files, if included
- `results/`: Model evaluation results

## Installation

1. Clone this repository.
2. Install the required dependencies:

   `python -m pip install -r requirements.txt`

3. Run the application:

   `python -m streamlit run app.py`

## Notes

The application uses saved model artifacts for predictions.
Training is not required to run the inference dashboard.

For model performance metrics, refer to the included evaluation results when available.
